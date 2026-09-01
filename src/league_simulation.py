"""
league_simulation.py

Simula los 30 equipos de la NBA con sus rosters REALES -- no solo el
equipo de team_config.yaml contra un WinPCT genérico de rival (ver
simulation.py). Construye una temporada regular con calendario
round-robin de rivales reales, y a partir del resultado, un bracket
completo de playoffs con el formato real de la NBA (1-6 directos, 7-10 al
play-in).

LIMITACIÓN DE DATOS: el calendario real de la temporada del config puede
no existir todavía (ver schedule_strength.py), así que el calendario aquí
es un round-robin equilibrado (cada equipo juega contra cada rival ~2-3
veces hasta sumar `games_per_season`) en vez del calendario oficial real.
Cuando la NBA publique el calendario completo, se puede sustituir --
`simulate_league_regular_season` solo necesita una lista de (día, equipo,
equipo).

MECÁNICA DE UN PARTIDO EQUIPO-CONTRA-EQUIPO
----------------------------------------------
A diferencia de simulation.py (que compara el Game Score del equipo
propio contra un WinPCT genérico de rival, y por eso necesita restar una
línea base de "equipo promedio" -- ver
simulation.compute_game_net_rating_estimate), aquí se comparan los Game
Score de AMBOS equipos reales directamente. La línea base se cancela:

    diferencial      = (team_game_score_A - team_game_score_B) * game_score_to_net_rating_scale
    prob_victoria_A  = 1 / (1 + exp(-diferencial / outcome_variance_scale))

IMPORTANTE: la diferencia de Game Score entre equipos se debe convertir
a puntos de diferencial con `game_score_to_net_rating_scale` (calibrada
en simulation.py, ver su docstring) antes de entrar en la logística --
usar la diferencia de Game Score en bruto equivale a asumir "1 punto de
Game Score = 1 punto de diferencial", una escala ~4.8x más fuerte de lo
calibrado contra datos reales. Esta escala debe mantenerse en sync con
simulation.py: si diverge, la MISMA franquicia da números de victorias
distintos en "Mi equipo" y en "Liga NBA" sin que sea una diferencia de
diseño real entre los dos motores. `outcome_variance_scale` significa lo
mismo en los dos motores: puntos de DIFERENCIAL, no de Game Score.

SIMPLIFICACIONES EN PLAYOFFS (documentadas, no ocultas)
------------------------------------------------------------
- Bracket FIJO (1v8, 2v7, 3v6, 4v5 y así sucesivamente), sin re-seeding
  entre rondas -- la NBA real sí resiembra; se simplifica aquí.
- Disponibilidad en playoffs: SÍ se sortea (Bernoulli por partido con la
  misma `risk_score` de la temporada regular) -- simular a plena salud
  favorecería indebidamente a equipos de estrellas frágiles justo en el
  tramo que más importa; ver el docstring de `_sample_team_game_score`.
  Lo que NO se replica es el tramo CONTIGUO de lesión de
  `sample_injury_absences` (en series de 4-7 partidos la diferencia
  entre racha y sorteo por partido es pequeña).
- Sin back-to-backs en playoffs (el calendario de playoffs no está tan
  comprimido como la temporada regular).
- Desgaste de fatiga de fin de temporada aplicado de forma constante
  (season_progress=1.0), no creciente partido a partido dentro de la
  postemporada.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import get_paths  # noqa: E402
from season_utils import dedupe_traded_seasons, season_start_year  # noqa: E402
from aging_curve import (  # noqa: E402
    DEFAULT_N_SEASONS_LOOKBACK,
    DEFAULT_RECENCY_HALF_LIFE_SEASONS,
    compute_reliability_weighted_minutes_per_game,
    project_player_season,
)
from advanced_impact import (  # noqa: E402
    adjust_with_context,
    build_advanced_context,
    load_advanced_stats,
)
from lineup_synergy import (  # noqa: E402
    build_synergy_matrix,
    compute_game_synergy_adjustment,
    compute_style_profile,
)
from simulation import (  # noqa: E402
    DEFAULT_MONTE_CARLO_CONFIG,
    apply_star_bonus,
    compute_player_contributions,
    normalize_rotation_minutes,
    sample_injury_absences,
    sample_team_quality_noise,
)
from context.injury_model import compute_risk_score  # noqa: E402
from context.fatigue_accumulation import compute_fatigue_score  # noqa: E402
from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID  # noqa: E402

# Conferencia de cada franquicia -- hecho de liga estable, no específico
# de ningún equipo/jugador simulado (igual que ABBREVIATION_TO_TEAM_ID en
# opponent_weighting.py).
TEAM_CONFERENCE: Dict[str, str] = {
    "ATL": "East", "BOS": "East", "BKN": "East", "CHA": "East", "CHI": "East",
    "CLE": "East", "DET": "East", "IND": "East", "MIA": "East", "MIL": "East",
    "NYK": "East", "ORL": "East", "PHI": "East", "TOR": "East", "WAS": "East",
    "DAL": "West", "DEN": "West", "GSW": "West", "HOU": "West", "LAC": "West",
    "LAL": "West", "MEM": "West", "MIN": "West", "NOP": "West", "OKC": "West",
    "PHX": "West", "POR": "West", "SAC": "West", "SAS": "West", "UTA": "West",
}

DEFAULT_LEAGUE_N_SEASONS = 1000  # más barato que el n_seasons de simulation.py -- ver docstring
DEFAULT_ROTATION_SIZE = 10  # tamaño de rotación real NBA -- ver docstring de project_team_roster

# Escenario "sin lesiones": mismo motor, mismos rosters proyectados --
# solo se pone a cero el risk_score de todo el mundo (ver
# _apply_scenario) antes de simular. "with_injuries" escribe los mismos
# nombres de archivo que el proyecto ya usaba antes de que existiera el
# concepto de escenario, así que no rompe nada existente (webapp/,
# tests) que llame sin pasar `scenario`.
SCENARIO_WITH_INJURIES = "with_injuries"
SCENARIO_NO_INJURIES = "no_injuries"


def _scenario_suffix(scenario: str) -> str:
    if scenario == SCENARIO_NO_INJURIES:
        return "_no_injuries"
    if scenario == SCENARIO_WITH_INJURIES:
        return ""
    raise ValueError(f"scenario desconocido: {scenario!r} (usa {SCENARIO_WITH_INJURIES!r} o {SCENARIO_NO_INJURIES!r})")


def _apply_scenario(
    team_projections: Dict[int, Dict[str, Any]], scenario: str
) -> Dict[int, Dict[str, Any]]:
    """
    Devuelve una copia de `team_projections` con el `risk_score` de TODO
    el mundo puesto a cero si `scenario == SCENARIO_NO_INJURIES` -- un
    solo punto de intervención que cascada correctamente a la simulación
    de temporada regular, a los playoffs (ambos leen `risk_scores`, el
    array por equipo) y a `league_player_projections.csv` (cada fila de
    `player_rows` trae su propio `risk_score`, de donde salen GP/MPG
    mostrados en cualquier tabla y el umbral de elegibilidad de
    All-NBA/All-Defensive en awards_projection.py). No vuelve a proyectar
    nada -- es una mutación barata sobre proyecciones ya calculadas.
    """
    if scenario == SCENARIO_WITH_INJURIES:
        return team_projections
    if scenario != SCENARIO_NO_INJURIES:
        raise ValueError(f"scenario desconocido: {scenario!r}")

    healthy: Dict[int, Dict[str, Any]] = {}
    for team_id, proj in team_projections.items():
        healthy_rows = [{**row, "risk_score": 0.0} for row in proj["player_rows"]]
        healthy[team_id] = {**proj, "risk_scores": np.zeros_like(proj["risk_scores"]), "player_rows": healthy_rows}
    return healthy


def _most_recent_season_row(player_seasons: pd.DataFrame):
    """Fila (deduplicada por trade) de la temporada más reciente de un jugador."""
    deduped = dedupe_traded_seasons(player_seasons)
    years = deduped["SEASON_ID"].apply(season_start_year)
    return deduped.loc[years.idxmax()]


def project_team_roster(
    roster_slice: pd.DataFrame,
    player_regular_stats: pd.DataFrame,
    player_playoff_stats: pd.DataFrame,
    config: Dict[str, Any],
    advanced_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Proyecta un equipo cualquiera (no solo el del config) para
    config["team"]["season"]: game_score_per36, minutos, risk_score,
    fatigue_score y matriz de sinergia por jugador de su roster real.

    `advanced_context` (de `advanced_impact.build_advanced_context`): si
    se pasa, el game_score_per36 es la métrica COMPUESTA. DEBE ser el
    mismo para los 30 equipos -- mezclar métricas entre rivales los haría
    incomparables, que es justo lo que este motor compara.

    A diferencia del roster propio (que tiene minutes_projection curado a
    mano en team_config.yaml, y que ya suma ~230-240 porque una persona lo
    ajustó), aquí los minutos parten de los minutos/partido REALES de la
    temporada más reciente de cada jugador (continuidad de rol).

    POR QUÉ NO SE USA LA SUMA DE MINUTOS "EN BRUTO" DIRECTAMENTE:

    1. La suma de minutos/partido reales de TODO el roster (15-22
       jugadores, incluyendo a cualquiera que jugó un solo partido por
       una lesión ajena) no suma 240 (5 posiciones x 48 min, lo único que
       existe de verdad en un partido) -- a lo largo de 82 partidos rotan
       más de 5-8 jugadores, así que la suma real es bastante más alta,
       y crece más cuanto más movimiento de plantilla tiene el equipo
       (lesiones, tanking, llamados de two-way/G-League).
    2. Escalar TODO el roster para que sume exactamente 240 penaliza
       injustamente a las estrellas de un roster con mucho movimiento de
       plantilla: los minutos reales de un jugador de rotación bajan
       artificialmente porque sus compañeros de banquillo (varios con
       pocos partidos) inflan el total del equipo que hay que repartir
       entre 240. Un jugador de banquillo con pocos minutos reales por
       *movimiento de plantilla* no debería diluir los minutos de la
       estrella real del equipo.

    SOLUCIÓN: normalizar solo dentro de una ROTACIÓN REALISTA -- los
    `rotation_size` jugadores (10 por defecto) con más minutos/partido en
    bruto, escalados para sumar 240 entre ellos. El resto del roster
    (suplentes de fondo de plantilla, two-way, llamados puntuales) se
    trata como 0 minutos -- no afectan de forma relevante a la fuerza de
    un equipo en una aproximación de este tipo, y diluir la rotación real
    con ellos es precisamente lo que causa el problema de arriba.

    El ranking que decide quién entra en esa rotación usa
    `aging_curve.compute_reliability_weighted_minutes_per_game()` (mismo
    criterio recencia+fiabilidad que `compute_recency_weighted_baseline`)
    en vez de solo la última temporada: un jugador de rotación real con
    una temporada corta por lesión reciente no debe caer del corte de
    `rotation_size` cuando sus temporadas previas muestran claramente un
    rol de rotación -- validado contra el backtest sweep de 480 casos.
    """
    player_ids = roster_slice["PLAYER_ID"].astype(int).tolist()
    player_names = roster_slice.set_index(roster_slice["PLAYER_ID"].astype(int))["PLAYER"].to_dict()
    # POSITION viene de league_rosters.csv (CommonTeamRoster) -- solo
    # existe para los 29 equipos REALES; project_own_team_for_league la
    # inyecta aparte desde roster_positions.csv (ver su docstring).
    positions_by_player = (
        roster_slice.set_index(roster_slice["PLAYER_ID"].astype(int))["POSITION"].to_dict()
        if "POSITION" in roster_slice.columns
        else {}
    )
    target_year = season_start_year(config["team"]["season"])
    rotation_size = config.get("league_simulation", {}).get("rotation_size", DEFAULT_ROTATION_SIZE)
    # IMPORTANTE: project_player_season() debe recibir n_seasons/half_life_seasons
    # explícitos aquí -- sin ellos cae a los defaults del módulo
    # (aging_curve.DEFAULT_*) e ignora config["aging_curve"] por completo
    # para los 30 equipos reales de Liga NBA, aunque build_aging_projection_dataset()
    # (roster propio) sí lo respete. Necesario para poder calibrar el
    # encogimiento hacia la media que afecta la compresión de victorias
    # entre equipos (ver scripts/experiments/aging_curve_shrinkage.py).
    aging_cfg = config.get("aging_curve", {})
    aging_n_seasons = aging_cfg.get("n_seasons_lookback", DEFAULT_N_SEASONS_LOOKBACK)
    aging_half_life = aging_cfg.get("recency_half_life_seasons", DEFAULT_RECENCY_HALF_LIFE_SEASONS)

    # --- Paso 1: minutos "en bruto" de cada jugador, sin proyectar todavía ---
    # Ponderados por recencia + fiabilidad (compute_reliability_weighted_minutes_per_game),
    # NO solo la última temporada -- ver su docstring (caso Dereck Lively
    # II: una sola temporada corta por lesión lo sacaba de la rotación
    # pese a ser titular real). `recent_row` (la fila más reciente sola)
    # se sigue usando aparte, más abajo, para current_age/current_year.
    raw_minutes: Dict[int, float] = {}
    recent_rows: Dict[int, Any] = {}
    for player_id in player_ids:
        player_regular = player_regular_stats[player_regular_stats["PLAYER_ID"] == player_id]
        if player_regular.empty:
            raw_minutes[player_id] = 0.0
            continue
        recent_rows[player_id] = _most_recent_season_row(player_regular)
        raw_minutes[player_id] = compute_reliability_weighted_minutes_per_game(
            player_regular, aging_n_seasons, aging_half_life
        )

    # Solo la rotación (top N por minutos en bruto) participa en la
    # normalización -- el resto queda en 0, no diluye a la rotación real.
    # Lógica compartida con backtesting.py (ver simulation.normalize_rotation_minutes).
    normalized_minutes = normalize_rotation_minutes(raw_minutes, rotation_size)

    # --- Paso 2: proyectar cada jugador con los minutos ya normalizados ---
    game_score_per36, minutes, risk_scores, fatigue_scores = [], [], [], []
    profiles: Dict[int, Dict[str, float]] = {}
    minutes_by_player: Dict[int, float] = {}
    player_rows: List[Dict[str, Any]] = []

    for player_id in player_ids:
        player_regular = player_regular_stats[player_regular_stats["PLAYER_ID"] == player_id]
        player_playoff = (
            player_playoff_stats[player_playoff_stats["PLAYER_ID"] == player_id]
            if player_playoff_stats is not None and not player_playoff_stats.empty
            else pd.DataFrame()
        )

        if player_regular.empty:
            game_score_per36.append(0.0)
            risk_scores.append(0.0)
            fatigue_scores.append(0.0)
            minutes.append(0.0)
            profiles[player_id] = {"usage": 0.0, "playmaking": 0.0, "spacing": 0.0, "interior": 0.0}
            minutes_by_player[player_id] = 0.0
            player_rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_names.get(player_id),
                    "position": positions_by_player.get(player_id),
                    "game_score_per36": 0.0,
                    "risk_score": 0.0,
                    "fatigue_score": 0.0,
                    "minutes_projection": 0.0,
                }
            )
            continue

        recent_row = recent_rows[player_id]
        current_age = float(recent_row["PLAYER_AGE"])
        current_year = season_start_year(recent_row["SEASON_ID"])
        target_age = current_age + max(target_year - current_year, 0)
        # Fuera de la rotación (top N por minutos en bruto) -> 0 minutos,
        # ver docstring: no diluyen la normalización de la rotación real.
        minutes_per_game = normalized_minutes[player_id]

        projection = project_player_season(
            player_regular,
            target_age=target_age,
            minutes_per_game=minutes_per_game,
            games_per_season=config["simulation"]["games_per_season"],
            n_seasons=aging_n_seasons,
            half_life_seasons=aging_half_life,
        )
        risk = compute_risk_score(player_regular)
        fatigue = compute_fatigue_score(player_regular, player_playoff if not player_playoff.empty else None)

        # Sobrescribir DENTRO de `projection`, no solo en la lista que usa
        # la simulación: `projection` se vuelca tal cual en player_rows ->
        # league_player_projections.csv, y ese CSV es el que lee
        # simulation.compute_league_average_game_score_per36 para su línea
        # base. Si la simulación usara la métrica ajustada y el CSV la
        # cruda, "Mi equipo" se compararía contra una referencia medida en
        # otra escala -- mantener ambas en sync es lo que evita ese
        # desajuste entre motores.
        projection["game_score_per36_box"] = projection["game_score_per36"]
        projection["game_score_per36"] = adjust_with_context(
            projection["game_score_per36"], player_id, config["team"]["season"], advanced_context
        )

        game_score_per36.append(projection["game_score_per36"])
        risk_scores.append(risk["risk_score"])
        fatigue_scores.append(fatigue["fatigue_score"])
        minutes.append(minutes_per_game)
        profiles[player_id] = compute_style_profile(projection)
        minutes_by_player[player_id] = minutes_per_game
        player_rows.append(
            {
                "player_id": player_id,
                "player_name": player_names.get(player_id),
                "position": positions_by_player.get(player_id),
                "current_age": current_age,
                "target_age": target_age,
                "minutes_projection": minutes_per_game,
                "risk_score": risk["risk_score"],
                "fatigue_score": fatigue["fatigue_score"],
                **projection,
            }
        )

    syn_cfg = config.get("lineup_synergy", {})
    synergy_matrix = build_synergy_matrix(
        player_ids,
        profiles,
        minutes_by_player,
        usage_threshold=syn_cfg.get("usage_threshold", 18.0),
        usage_clash_weight=syn_cfg.get("usage_clash_weight", 0.05),
        playmaking_spacing_weight=syn_cfg.get("playmaking_spacing_weight", 0.02),
    )

    return {
        "player_rows": player_rows,
        "player_ids": player_ids,
        "game_score_per36": np.array(game_score_per36),
        "minutes_projection": np.array(minutes),
        "risk_scores": np.array(risk_scores),
        "fatigue_scores": np.array(fatigue_scores),
        "synergy_matrix": synergy_matrix,
    }


def build_round_robin_schedule(
    team_ids: List[int], games_per_season: int, rng: np.random.Generator
) -> List[Tuple[int, int, int]]:
    """
    Calendario round-robin equilibrado: cada equipo juega contra cada
    rival de forma rotativa hasta sumar `games_per_season` partidos. Usa
    el método clásico del círculo (round-robin de torneos): con N equipos
    se generan N-1 "rondas" donde cada equipo juega exactamente una vez;
    se repiten rondas hasta completar `games_per_season`. Devuelve una
    lista de (día, equipo_a, equipo_b) -- ambos equipos de una misma fila
    comparten el mismo índice de día, así el desgaste de fatiga de
    temporada avanza en sincronía entre todos los equipos (aproximación
    razonable: en la realidad los equipos progresan la temporada casi al
    unísono).
    """
    n = len(team_ids)
    if n % 2 != 0:
        raise ValueError("build_round_robin_schedule requiere un número par de equipos.")

    arr = list(team_ids)
    rng.shuffle(arr)

    single_cycle_rounds: List[List[Tuple[int, int]]] = []
    working = arr[:]
    for _ in range(n - 1):
        pairs = list(zip(working[: n // 2], working[n // 2 :][::-1]))
        single_cycle_rounds.append(pairs)
        working = [working[0]] + [working[-1]] + working[1:-1]

    schedule: List[Tuple[int, int, int]] = []
    for day in range(games_per_season):
        round_idx = day % (n - 1)
        for team_a, team_b in single_cycle_rounds[round_idx]:
            schedule.append((day, team_a, team_b))
    return schedule


def _build_team_game_score_arrays(
    team_projections: Dict[int, Dict[str, Any]],
    n_seasons: int,
    games_per_season_by_team: Dict[int, int],
    mc_config: Dict[str, float],
    rng: np.random.Generator,
    is_back_to_back_by_team: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Disponibilidad + contribución por jugador + sinergia, por equipo --
    núcleo COMPARTIDO entre `simulate_league_regular_season` (calendario
    SINTÉTICO, back-to-back sorteado por partido/temporada simulada,
    mismo nº de partidos para todos), `simulate_league_regular_season_real_schedule`
    y `simulate_single_league_season_game_log_real_schedule` (calendario
    REAL, back-to-back ya es un HECHO fijo del calendario -- no se
    sortea -- y cada equipo puede tener su propio nº de partidos).
    Devuelve `(team_game_scores, availability_by_team)`, ambos
    `{team_id: array (n_seasons, games_per_season_de_ESE_equipo, ...)}`
    -- el segundo lo necesita `simulate_single_league_season_player_box_scores`
    para saber quién jugó cada partido.

    `is_back_to_back_by_team`: si se pasa, usa ese array FIJO por equipo
    (shape `(1, n_partidos)`, hace broadcast contra cualquier n_seasons)
    -- calendario real. Si no, sortea uno por partido/temporada simulada,
    idéntico al comportamiento de siempre -- calendario sintético.
    """
    team_game_scores: Dict[int, np.ndarray] = {}
    availability_by_team: Dict[int, np.ndarray] = {}
    for team_id, proj in team_projections.items():
        games_per_season = games_per_season_by_team[team_id]
        available = sample_injury_absences(
            proj["risk_scores"], n_seasons, games_per_season, rng, mc_config["injury_dispersion"]
        )
        if is_back_to_back_by_team is not None:
            is_back_to_back = is_back_to_back_by_team[team_id]
        else:
            is_back_to_back = rng.random((n_seasons, games_per_season)) < mc_config["b2b_probability"]
            is_back_to_back[:, 0] = False

        contributions = compute_player_contributions(
            proj["game_score_per36"], proj["minutes_projection"], proj["fatigue_scores"],
            available, is_back_to_back, rng, mc_config,
        )
        team_game_score = contributions.sum(axis=2)

        if proj.get("synergy_matrix") is not None:
            team_game_score = team_game_score + compute_game_synergy_adjustment(available, proj["synergy_matrix"])

        team_game_scores[team_id] = team_game_score
        availability_by_team[team_id] = available
    return team_game_scores, availability_by_team


def simulate_league_regular_season(
    team_projections: Dict[int, Dict[str, Any]],
    schedule: List[Tuple[int, int, int]],
    n_seasons: int,
    games_per_season: int,
    mc_config: Dict[str, float],
    random_seed: int,
) -> Dict[int, np.ndarray]:
    """
    Simula la temporada regular completa para todos los equipos en
    team_projections a la vez (vectorizado por equipo sobre n_seasons),
    sobre un calendario SINTÉTICO (`build_round_robin_schedule`) -- ver
    `simulate_league_regular_season_real_schedule` para el calendario
    real. Devuelve {team_id: wins_array(n_seasons,)}.
    """
    rng = np.random.default_rng(random_seed)

    # Incertidumbre de calidad de equipo -- ver simulation.sample_team_quality_noise:
    # un valor por (equipo, temporada simulada), constante en los 82
    # partidos de esa temporada. Desactivado por defecto (std=0.0). Ya en
    # puntos de diferencial (no en unidades de Game Score), así que se
    # suma DESPUÉS de convertir el diferencial de Game Score con
    # game_score_to_net_rating_scale, no antes.
    team_quality_noise = {
        team_id: sample_team_quality_noise(
            n_seasons, mc_config.get("team_quality_uncertainty_std", 0.0), rng
        ).ravel()
        for team_id in team_projections
    }
    games_per_season_by_team = {team_id: games_per_season for team_id in team_projections}
    team_game_scores, _availability_by_team = _build_team_game_score_arrays(
        team_projections, n_seasons, games_per_season_by_team, mc_config, rng
    )

    wins = {team_id: np.zeros(n_seasons, dtype=int) for team_id in team_projections}
    for day, team_a, team_b in schedule:
        score_a = team_game_scores[team_a][:, day]
        score_b = team_game_scores[team_b][:, day]
        # La diferencia de Game Score se convierte a puntos de diferencial
        # con la escala calibrada antes de entrar en la logística -- ver el
        # bug documentado en el docstring del módulo.
        point_differential = (score_a - score_b) * mc_config["game_score_to_net_rating_scale"]
        point_differential = point_differential + (team_quality_noise[team_a] - team_quality_noise[team_b])
        win_prob_a = 1 / (1 + np.exp(-point_differential / mc_config["outcome_variance_scale"]))
        team_a_wins = rng.random(n_seasons) < win_prob_a
        wins[team_a] += team_a_wins
        wins[team_b] += ~team_a_wins

    return wins


def real_schedule_to_games(
    schedule_df: pd.DataFrame, abbreviation_to_team_id: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Convierte el calendario REAL (gameDate, homeTeam_teamTricode,
    awayTeam_teamTricode -- ver data_pipeline.build_league_schedule_dataset)
    en una lista de partidos ordenados cronológicamente, cada uno con el
    índice de partido SECUENCIAL de cada equipo dentro de su propia
    temporada (0-indexado) y si es back-to-back REAL para cada lado
    (partido anterior de ESE equipo exactamente un día antes).

    A diferencia del calendario sintético (`build_round_robin_schedule`,
    donde "el mismo día" implica que TODOS los equipos juegan, así que un
    único índice compartido sirve para los dos lados de cada partido), en
    un calendario real el descanso varía por equipo -- cada equipo
    necesita su PROPIO índice de partido, nunca uno compartido (ver
    docstring del módulo).
    """
    df = schedule_df.copy()
    df["gameDate"] = pd.to_datetime(df["gameDate"])
    df = df.sort_values("gameDate").reset_index(drop=True)

    game_index_by_team: Dict[int, int] = {}
    last_date_by_team: Dict[int, pd.Timestamp] = {}
    games: List[Dict[str, Any]] = []

    for row in df.itertuples():
        home_id = abbreviation_to_team_id[row.homeTeam_teamTricode]
        away_id = abbreviation_to_team_id[row.awayTeam_teamTricode]

        home_idx = game_index_by_team.get(home_id, 0)
        away_idx = game_index_by_team.get(away_id, 0)
        one_day = pd.Timedelta(days=1)
        is_b2b_home = last_date_by_team.get(home_id) == row.gameDate - one_day
        is_b2b_away = last_date_by_team.get(away_id) == row.gameDate - one_day

        games.append({
            "date": row.gameDate,
            "home_team_id": home_id, "home_game_index": home_idx, "is_b2b_home": is_b2b_home,
            "away_team_id": away_id, "away_game_index": away_idx, "is_b2b_away": is_b2b_away,
        })

        game_index_by_team[home_id] = home_idx + 1
        game_index_by_team[away_id] = away_idx + 1
        last_date_by_team[home_id] = row.gameDate
        last_date_by_team[away_id] = row.gameDate

    return games


def _load_real_schedule_games(config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Carga `league_schedule_full.csv` (data_pipeline.build_league_schedule_dataset)
    y lo convierte con `real_schedule_to_games` -- `None` si el CSV no
    existe todavía (temporada sin calendario real publicado, o
    simplemente no se ha corrido `data_pipeline.py --league` con esta
    versión del proyecto). Punto de entrada ÚNICO para "¿hay calendario
    real disponible?" -- lo usan tanto `build_league_simulation_dataset`
    como `run_single_league_season_simulation`, así los dos caen al
    mismo criterio de degradar al calendario sintético si falta.
    """
    paths = get_paths(config)
    schedule_path = paths["processed"] / "league_schedule_full.csv"
    if not schedule_path.exists() or schedule_path.stat().st_size == 0:
        return None
    schedule_df = pd.read_csv(schedule_path)
    return real_schedule_to_games(schedule_df, ABBREVIATION_TO_TEAM_ID)


def simulate_league_regular_season_real_schedule(
    team_projections: Dict[int, Dict[str, Any]],
    games: List[Dict[str, Any]],
    n_seasons: int,
    mc_config: Dict[str, float],
    random_seed: int,
) -> Dict[int, np.ndarray]:
    """
    Igual que `simulate_league_regular_season` pero sobre el calendario
    REAL (`real_schedule_to_games`): cada equipo tiene su propio nº de
    partidos (hoy 80, no 82 -- ver docstring de
    `data_pipeline.build_league_schedule_dataset`, limitación temporal
    real mientras la NBA Cup no se resuelve del todo), el back-to-back es
    un HECHO del calendario (no se sortea) y se aplica
    `home_court_advantage` (ya calibrado en `simulation.py`, 2.41 puntos)
    al equipo local -- ambos datos son reales aquí, a diferencia del
    calendario sintético, donde "local" era arbitrario y por eso nunca se
    aplicaba.
    """
    rng = np.random.default_rng(random_seed)

    games_per_season_by_team: Dict[int, int] = {}
    for game in games:
        for side in ("home", "away"):
            team_id, idx = game[f"{side}_team_id"], game[f"{side}_game_index"]
            games_per_season_by_team[team_id] = max(games_per_season_by_team.get(team_id, 0), idx + 1)

    is_back_to_back_by_team = {
        team_id: np.zeros((1, n_games), dtype=bool) for team_id, n_games in games_per_season_by_team.items()
    }
    for game in games:
        is_back_to_back_by_team[game["home_team_id"]][0, game["home_game_index"]] = game["is_b2b_home"]
        is_back_to_back_by_team[game["away_team_id"]][0, game["away_game_index"]] = game["is_b2b_away"]

    team_quality_noise = {
        team_id: sample_team_quality_noise(
            n_seasons, mc_config.get("team_quality_uncertainty_std", 0.0), rng
        ).ravel()
        for team_id in team_projections
    }
    team_game_scores, _availability_by_team = _build_team_game_score_arrays(
        team_projections, n_seasons, games_per_season_by_team, mc_config, rng,
        is_back_to_back_by_team=is_back_to_back_by_team,
    )

    hca = mc_config.get("home_court_advantage", 0.0)
    wins = {team_id: np.zeros(n_seasons, dtype=int) for team_id in team_projections}
    for game in games:
        team_a, idx_a = game["home_team_id"], game["home_game_index"]
        team_b, idx_b = game["away_team_id"], game["away_game_index"]
        score_a = team_game_scores[team_a][:, idx_a]
        score_b = team_game_scores[team_b][:, idx_b]
        point_differential = (score_a - score_b) * mc_config["game_score_to_net_rating_scale"] + hca
        point_differential = point_differential + (team_quality_noise[team_a] - team_quality_noise[team_b])
        win_prob_a = 1 / (1 + np.exp(-point_differential / mc_config["outcome_variance_scale"]))
        team_a_wins = rng.random(n_seasons) < win_prob_a
        wins[team_a] += team_a_wins
        wins[team_b] += ~team_a_wins

    return wins


# Fracción de la media por-partido usada como sigma del ruido gaussiano
# de cada categoría de boxscore -- un jugador de más volumen tiene más
# varianza ABSOLUTA de forma proporcional (20 PPG de media varía más en
# términos absolutos que 5 PPG), no un sigma fijo para toda la liga.
DEFAULT_BOX_SCORE_NOISE_STD_FRACTION = 0.35

# <stat de boxscore> -> columna de TOTAL de temporada ya proyectada
# (player_rows, de project_team_roster/project_own_team_for_league) --
# dividida entre games_per_season da la media por-partido, la misma
# base que ya usa build_league_simulation_dataset para las columnas
# PPG/RPG/etc. mostradas en cualquier tabla de roster.
BOX_SCORE_STAT_COLUMNS: Dict[str, str] = {
    "PTS": "PTS_projected", "REB": "REB_projected", "AST": "AST_projected",
    "STL": "STL_projected", "BLK": "BLK_projected", "TOV": "TOV_projected",
    "3PM": "FG3M_projected",
}


def simulate_single_league_season_game_log(
    team_projections: Dict[int, Dict[str, Any]],
    schedule: List[Tuple[int, int, int]],
    team_abbrev_by_id: Dict[int, str],
    games_per_season: int,
    mc_config: Dict[str, float],
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Igual que `simulate_league_regular_season` -- MISMA matemática exacta
    (mismas funciones, mismo orden: disponibilidad, contribución por
    jugador, sinergia, escala calibrada, logística) -- pero para UNA
    temporada concreta (no vectorizada sobre miles de réplicas Monte
    Carlo) y grabando el resultado de CADA partido en vez de solo
    acumular victorias. Mismo principio que
    `simulation.simulate_single_season_player_log`: no es un modelo
    paralelo, es la misma mecánica con n_seasons=1.

    Devuelve `(game_log, availability_by_team)`: `game_log` una fila por
    partido (`game_id`, `day`, equipos, resultado, y
    `home_game_index`/`away_game_index` -- aquí siempre iguales a `day`,
    ver por qué en `simulate_single_league_season_game_log_real_schedule`);
    `availability_by_team` -- `{team_id: bool array (games_per_season,
    n_players)}`, la MISMA máscara de disponibilidad ya usada para
    decidir el resultado de cada partido, para que
    `simulate_single_league_season_player_box_scores` reutilice
    exactamente esas ausencias en vez de sortearlas de nuevo (un jugador
    de baja en el resultado del partido también aparece en 0 en su
    boxscore, de forma consistente).
    """
    rng = np.random.default_rng(random_seed)

    team_quality_noise = {
        team_id: sample_team_quality_noise(
            1, mc_config.get("team_quality_uncertainty_std", 0.0), rng
        ).ravel()[0]
        for team_id in team_projections
    }

    team_game_scores: Dict[int, np.ndarray] = {}
    availability_by_team: Dict[int, np.ndarray] = {}
    for team_id, proj in team_projections.items():
        available = sample_injury_absences(
            proj["risk_scores"], 1, games_per_season, rng, mc_config["injury_dispersion"]
        )
        is_back_to_back = rng.random((1, games_per_season)) < mc_config["b2b_probability"]
        is_back_to_back[:, 0] = False

        contributions = compute_player_contributions(
            proj["game_score_per36"], proj["minutes_projection"], proj["fatigue_scores"],
            available, is_back_to_back, rng, mc_config,
        )
        team_game_score = contributions.sum(axis=2)

        if proj.get("synergy_matrix") is not None:
            team_game_score = team_game_score + compute_game_synergy_adjustment(available, proj["synergy_matrix"])

        team_game_scores[team_id] = team_game_score[0]
        availability_by_team[team_id] = available[0]

    rows = []
    for game_id, (day, team_a, team_b) in enumerate(schedule):
        score_a = float(team_game_scores[team_a][day])
        score_b = float(team_game_scores[team_b][day])
        point_differential = (score_a - score_b) * mc_config["game_score_to_net_rating_scale"]
        point_differential += team_quality_noise[team_a] - team_quality_noise[team_b]
        win_prob_a = 1 / (1 + np.exp(-point_differential / mc_config["outcome_variance_scale"]))
        team_a_wins = bool(rng.random() < win_prob_a)
        winner_team_id = team_a if team_a_wins else team_b

        rows.append({
            "game_id": game_id,
            "day": day,
            "home_team_id": team_a,
            "home_abbreviation": team_abbrev_by_id[team_a],
            "home_game_index": day,
            "away_team_id": team_b,
            "away_abbreviation": team_abbrev_by_id[team_b],
            "away_game_index": day,
            "home_score": score_a,
            "away_score": score_b,
            "point_differential": float(point_differential),
            "winner_team_id": winner_team_id,
            "winner_abbreviation": team_abbrev_by_id[winner_team_id],
        })

    return pd.DataFrame(rows), availability_by_team


def simulate_single_league_season_game_log_real_schedule(
    team_projections: Dict[int, Dict[str, Any]],
    games: List[Dict[str, Any]],
    team_abbrev_by_id: Dict[int, str],
    mc_config: Dict[str, float],
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Igual que `simulate_single_league_season_game_log` pero sobre el
    calendario REAL (`real_schedule_to_games`) -- misma matemática que
    `simulate_league_regular_season_real_schedule` con n_seasons=1,
    grabando cada partido en vez de solo acumular victorias. La columna
    `day` pasa a ser la FECHA real (no un entero sintético 0-81), y se
    aplica `home_court_advantage` al equipo local -- mismo motivo que en
    la versión agregada.

    Devuelve el MISMO esquema que `simulate_single_league_season_game_log`
    (mismas columnas, incluidos `home_game_index`/`away_game_index`) --
    `simulate_single_league_season_player_box_scores` funciona sin
    cambios sobre cualquiera de los dos.
    """
    rng = np.random.default_rng(random_seed)

    games_per_season_by_team: Dict[int, int] = {}
    for game in games:
        for side in ("home", "away"):
            team_id, idx = game[f"{side}_team_id"], game[f"{side}_game_index"]
            games_per_season_by_team[team_id] = max(games_per_season_by_team.get(team_id, 0), idx + 1)

    is_back_to_back_by_team = {
        team_id: np.zeros((1, n_games), dtype=bool) for team_id, n_games in games_per_season_by_team.items()
    }
    for game in games:
        is_back_to_back_by_team[game["home_team_id"]][0, game["home_game_index"]] = game["is_b2b_home"]
        is_back_to_back_by_team[game["away_team_id"]][0, game["away_game_index"]] = game["is_b2b_away"]

    team_quality_noise = {
        team_id: sample_team_quality_noise(
            1, mc_config.get("team_quality_uncertainty_std", 0.0), rng
        ).ravel()[0]
        for team_id in team_projections
    }
    team_game_scores, availability_by_team_raw = _build_team_game_score_arrays(
        team_projections, 1, games_per_season_by_team, mc_config, rng,
        is_back_to_back_by_team=is_back_to_back_by_team,
    )
    # squeeze la dimensión n_seasons=1 -- mismo shape (games, n_players) que
    # devuelve simulate_single_league_season_game_log, para que
    # simulate_single_league_season_player_box_scores funcione igual con
    # cualquiera de los dos calendarios.
    availability_by_team = {team_id: arr[0] for team_id, arr in availability_by_team_raw.items()}

    hca = mc_config.get("home_court_advantage", 0.0)
    rows = []
    for game_id, game in enumerate(games):
        team_a, idx_a = game["home_team_id"], game["home_game_index"]
        team_b, idx_b = game["away_team_id"], game["away_game_index"]
        score_a = float(team_game_scores[team_a][0, idx_a])
        score_b = float(team_game_scores[team_b][0, idx_b])
        point_differential = (score_a - score_b) * mc_config["game_score_to_net_rating_scale"] + hca
        point_differential += team_quality_noise[team_a] - team_quality_noise[team_b]
        win_prob_a = 1 / (1 + np.exp(-point_differential / mc_config["outcome_variance_scale"]))
        team_a_wins = bool(rng.random() < win_prob_a)
        winner_team_id = team_a if team_a_wins else team_b

        rows.append({
            "game_id": game_id,
            "day": game["date"].strftime("%Y-%m-%d"),
            "home_team_id": team_a,
            "home_abbreviation": team_abbrev_by_id[team_a],
            "home_game_index": idx_a,
            "away_team_id": team_b,
            "away_abbreviation": team_abbrev_by_id[team_b],
            "away_game_index": idx_b,
            "home_score": score_a,
            "away_score": score_b,
            "point_differential": float(point_differential),
            "winner_team_id": winner_team_id,
            "winner_abbreviation": team_abbrev_by_id[winner_team_id],
        })

    return pd.DataFrame(rows), availability_by_team


def simulate_single_league_season_player_box_scores(
    team_projections: Dict[int, Dict[str, Any]],
    game_log: pd.DataFrame,
    availability_by_team: Dict[int, np.ndarray],
    games_per_season: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Boxscore ILUSTRATIVO por jugador y partido -- NO una simulación
    conjunta/correlacionada de categorías reales (más puntos no implica
    menos asistencias de forma realista, por ejemplo: cada categoría se
    sortea de forma independiente). Se deriva de la media por-partido de
    temporada YA proyectada (`<stat>_projected / games_per_season`,
    mismos totales que ya alimentan las columnas PPG/RPG/etc. en
    `build_league_simulation_dataset`) más ruido gaussiano (ver
    `DEFAULT_BOX_SCORE_NOISE_STD_FRACTION`), recortado en 0. Mismo
    espíritu honesto que `simulation.DEFAULT_INJURY_TYPE_CATEGORIES`:
    una aproximación ilustrativa, no un boxscore realista jugada a
    jugada.

    Un jugador NO disponible ese partido (`availability_by_team`, la
    MISMA máscara que ya decidió el resultado del partido en
    `simulate_single_league_season_game_log`) aparece en 0 en todas las
    categorías -- consistente con que no contribuyó al Game Score de su
    equipo ese partido.
    """
    rng = np.random.default_rng(random_seed)
    rows = []
    for game in game_log.itertuples():
        sides = ((game.home_team_id, game.home_game_index), (game.away_team_id, game.away_game_index))
        for team_id, game_index in sides:
            proj = team_projections[team_id]
            available_this_game = availability_by_team[team_id][game_index]
            for player_index, player_row in enumerate(proj["player_rows"]):
                box_row = {
                    "game_id": game.game_id,
                    "day": game.day,
                    "team_id": team_id,
                    "player_id": player_row.get("player_id"),
                    "player_name": player_row.get("player_name"),
                }
                is_available = bool(available_this_game[player_index])
                for stat, total_col in BOX_SCORE_STAT_COLUMNS.items():
                    if not is_available:
                        box_row[stat] = 0.0
                        continue
                    per_game_mean = (player_row.get(total_col) or 0.0) / games_per_season
                    noise = rng.normal(0, per_game_mean * DEFAULT_BOX_SCORE_NOISE_STD_FRACTION)
                    box_row[stat] = max(0.0, per_game_mean + noise)
                rows.append(box_row)
    return pd.DataFrame(rows)


def compute_head_to_head_record(game_log: pd.DataFrame, team_a_id: int, team_b_id: int) -> Dict[str, Any]:
    """
    Récord entre dos equipos en la temporada regular de `game_log` (de
    `simulate_single_league_season_game_log`). Función pura, sin
    aleatoriedad -- toda la incertidumbre ya está fijada en el game log
    que recibe.
    """
    mask = (
        ((game_log["home_team_id"] == team_a_id) & (game_log["away_team_id"] == team_b_id))
        | ((game_log["home_team_id"] == team_b_id) & (game_log["away_team_id"] == team_a_id))
    )
    matchups = game_log[mask].sort_values("day")
    return {
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "team_a_wins": int((matchups["winner_team_id"] == team_a_id).sum()),
        "team_b_wins": int((matchups["winner_team_id"] == team_b_id).sum()),
        "games": matchups.to_dict("records"),
    }


def run_single_league_season_simulation(
    config: Dict[str, Any], scenario: str = SCENARIO_WITH_INJURIES, random_seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Orquestador de UNA temporada concreta con calendario, resultado de
    cada partido y boxscore por jugador -- mismo patrón que
    `build_league_simulation_dataset` pero sin Monte Carlo agregado
    (sería carísimo simular miles de réplicas con detalle partido a
    partido). `random_seed=None` (por defecto): seed nuevo basado en el
    reloj, re-lanzable -- mismo criterio que
    `simulation.simulate_single_season_player_log` vía
    `dashboard.data_loader.run_single_season_player_log_simulation` (a
    diferencia de `simulate_single_bracket`, que reutiliza
    `config["simulation"]["random_seed"]` fijo: aquí interesa una
    temporada NUEVA cada vez que se pulsa el botón, no reproducir
    siempre la misma).

    Guarda `league_single_season_game_log{suffix}.csv` y
    `league_single_season_player_box_scores{suffix}.csv` (mismo
    `_scenario_suffix` que el resto del módulo).
    """
    paths = get_paths(config)
    suffix = _scenario_suffix(scenario)
    seed = int(random_seed if random_seed is not None else np.random.default_rng().integers(0, 2**31 - 1))

    team_ids, team_abbrev_by_id, _team_conference, team_projections = load_and_project_all_teams(config)
    team_projections = _apply_scenario(team_projections, scenario)

    games_per_season = config["simulation"]["games_per_season"]
    mc_config = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}

    real_games = _load_real_schedule_games(config)
    if real_games is not None:
        print(f"  Calendario REAL ({len(real_games)} partidos) -- corre `data_pipeline.py --league` para refrescarlo.")
        game_log, availability_by_team = simulate_single_league_season_game_log_real_schedule(
            team_projections, real_games, team_abbrev_by_id, mc_config, seed
        )
        box_scores = simulate_single_league_season_player_box_scores(
            team_projections, game_log, availability_by_team, games_per_season, seed + 1
        )
    else:
        print("  Aviso: no se encontró league_schedule_full.csv -- usando calendario SINTÉTICO "
              "(corre `data_pipeline.py --league` para el real).")
        schedule_rng = np.random.default_rng(seed)
        schedule = build_round_robin_schedule(team_ids, games_per_season, schedule_rng)
        game_log, availability_by_team = simulate_single_league_season_game_log(
            team_projections, schedule, team_abbrev_by_id, games_per_season, mc_config, seed
        )
        box_scores = simulate_single_league_season_player_box_scores(
            team_projections, game_log, availability_by_team, games_per_season, seed + 1
        )

    game_log_path = paths["processed"] / f"league_single_season_game_log{suffix}.csv"
    box_scores_path = paths["processed"] / f"league_single_season_player_box_scores{suffix}.csv"
    game_log.to_csv(game_log_path, index=False)
    box_scores.to_csv(box_scores_path, index=False)
    print(f"Guardado: {game_log_path} ({len(game_log)} partidos)")
    print(f"Guardado: {box_scores_path} ({len(box_scores)} líneas de boxscore)")

    return {"game_log": game_log, "player_box_scores": box_scores}


def _sample_team_game_score(proj: Dict[str, Any], rng: np.random.Generator, mc_config: Dict[str, float]) -> float:
    """
    Un partido de playoffs para UN equipo -- ver simplificaciones en el
    docstring del módulo. Aplica la misma prima de estrella que la
    temporada regular (ver simulation.apply_star_bonus) -- si no, un
    equipo top-heavy se vería penalizado justo en la parte de la
    temporada donde más importa.

    IMPORTANTE: la disponibilidad SÍ se sortea en playoffs, con la misma
    `risk_score` que usa la temporada regular (Bernoulli por partido: la
    semántica de risk_score es "fracción esperada de partidos perdidos",
    así que aplicarla por partido conserva esa media). Simular playoffs a
    plena salud produce un sesgo grave: un equipo construido sobre
    estrellas frágiles queda castigado los 82 partidos de temporada
    regular por lesiones y luego llega a playoffs milagrosamente sano,
    lo que puede hacerlo más favorito al título que un equipo con mejor
    récord real de temporada regular -- un equipo no puede ser más
    favorito al título solo porque el modelo le perdona las lesiones en
    el momento decisivo. NO se replica el tramo contiguo de
    `sample_injury_absences` -- en una serie de 4-7 partidos la
    diferencia entre "racha" y "sorteo por partido" es pequeña frente al
    beneficio de no asumir salud perfecta.
    """
    effective_game_score_per36 = apply_star_bonus(proj["game_score_per36"], proj["minutes_projection"], mc_config)
    base = effective_game_score_per36 * (proj["minutes_projection"] / 36.0)
    fatigue_decay = 1 - proj["fatigue_scores"] * mc_config["season_fatigue_decay"]
    noise = rng.normal(0, mc_config["game_variance_std"], size=len(base))
    contribution = base * fatigue_decay + noise

    available = rng.random(len(base)) >= proj["risk_scores"]
    contribution = np.where(available, contribution, 0.0)
    team_score = float(contribution.sum())

    if proj.get("synergy_matrix") is not None:
        team_score += float(
            compute_game_synergy_adjustment(available[None, None, :], proj["synergy_matrix"])[0, 0]
        )
    return team_score


# Formato real de sede de una serie NBA a 7 partidos (2-2-1-1-1): el
# equipo con MEJOR seed es local en los partidos 1, 2, 5 y 7. Hecho del
# deporte, no un parámetro del modelo.
# Temporadas muestreadas para estimar la sinergia esperada de cada equipo
# (ver league_team_synergy_baseline.csv). Es una media sobre 60 x 82
# partidos por equipo -- de sobra para una media.
LEAGUE_SYNERGY_SAMPLING_SEASONS = 60

SERIES_HOME_GAMES_FOR_HIGHER_SEED = (True, True, False, False, True, False, True)


def simulate_playoff_game(
    team_a_proj: Dict[str, Any],
    team_b_proj: Dict[str, Any],
    rng: np.random.Generator,
    mc_config: Dict[str, float],
    team_a_is_home: Optional[bool] = None,
) -> bool:
    """
    True si team_a gana este partido individual de playoffs.

    `team_a_is_home`: si se pasa, aplica `playoff_home_court_advantage`
    (medido en +3.98 puntos sobre 15 temporadas reales) al local. None =
    campo neutral, sin ventaja -- solo para tests o comparaciones donde
    la sede no importa.
    """
    score_a = _sample_team_game_score(team_a_proj, rng, mc_config)
    score_b = _sample_team_game_score(team_b_proj, rng, mc_config)
    # A puntos de DIFERENCIAL antes de nada -- ver el bug del docstring.
    # Antes se hacía al revés (convertir la ventaja de campo a unidades de
    # Game Score dividiendo por la escala), lo que mantenía la proporción
    # entre ambas pero dejaba `outcome_variance_scale` interpretado en
    # unidades de Game Score, incompatible con simulation.py.
    diff = (score_a - score_b) * mc_config["game_score_to_net_rating_scale"]

    if team_a_is_home is not None:
        hca = mc_config.get("playoff_home_court_advantage", 0.0)
        diff += hca if team_a_is_home else -hca

    win_prob_a = 1 / (1 + np.exp(-diff / mc_config["outcome_variance_scale"]))
    return bool(rng.random() < win_prob_a)


def simulate_series(
    team_a_proj: Dict[str, Any],
    team_b_proj: Dict[str, Any],
    rng: np.random.Generator,
    mc_config: Dict[str, float],
    best_of: int = 7,
    team_a_has_home_court: bool = True,
) -> int:
    """
    Simula una serie a mejor-de-`best_of`. Devuelve 0 si gana team_a, 1 si
    gana team_b.

    `team_a_has_home_court`: True si team_a es el de MEJOR seed (y por
    tanto local en los partidos 1, 2, 5 y 7 -- formato 2-2-1-1-1). Todos
    los llamantes de este módulo pasan los equipos en orden de seed, así
    que el valor por defecto es el correcto; se expone para poder
    simular una serie en campo neutral o invertida.
    """
    wins_needed = best_of // 2 + 1
    wins_a = wins_b = 0
    game_index = 0
    while wins_a < wins_needed and wins_b < wins_needed:
        higher_seed_home = SERIES_HOME_GAMES_FOR_HIGHER_SEED[
            game_index % len(SERIES_HOME_GAMES_FOR_HIGHER_SEED)
        ]
        team_a_is_home = higher_seed_home if team_a_has_home_court else not higher_seed_home
        if simulate_playoff_game(team_a_proj, team_b_proj, rng, mc_config, team_a_is_home=team_a_is_home):
            wins_a += 1
        else:
            wins_b += 1
        game_index += 1
    return 0 if wins_a > wins_b else 1


def _single_game_winner(
    team_a_id: int, team_b_id: int, team_projections: Dict[int, Dict[str, Any]],
    rng: np.random.Generator, mc_config: Dict[str, float],
) -> int:
    """Partido único de play-in. `team_a_id` es SIEMPRE el de mejor seed
    en todos los llamantes, y en el play-in real el mejor seed juega en
    casa -- por eso team_a_is_home=True."""
    a_wins = simulate_playoff_game(
        team_projections[team_a_id], team_projections[team_b_id], rng, mc_config, team_a_is_home=True
    )
    return team_a_id if a_wins else team_b_id


def _series_winner(
    team_a_id: int, team_b_id: int, team_projections: Dict[int, Dict[str, Any]],
    rng: np.random.Generator, mc_config: Dict[str, float], best_of: int = 7,
    seed_rank: Optional[Dict[int, int]] = None,
) -> int:
    """
    `seed_rank`: {team_id: posición en la clasificación} (menor = mejor).
    Determina quién tiene la ventaja de campo. Es OBLIGATORIO pasarlo a
    partir de semifinales: en primera ronda el emparejamiento ya viene
    ordenado (1v8, 4v5...), pero después NO -- si el seed 8 elimina al 1,
    el ganador de esa serie ya no es el mejor seed de su cruce. Sin
    seed_rank se asume que team_a es el mejor seed (correcto solo en
    primera ronda y en el play-in).
    """
    if seed_rank is None:
        team_a_has_home_court = True
    else:
        team_a_has_home_court = seed_rank.get(team_a_id, 99) <= seed_rank.get(team_b_id, 99)

    result = simulate_series(
        team_projections[team_a_id], team_projections[team_b_id], rng, mc_config, best_of,
        team_a_has_home_court=team_a_has_home_court,
    )
    return team_a_id if result == 0 else team_b_id


def resolve_play_in(
    seeds: List[int], team_projections: Dict[int, Dict[str, Any]],
    rng: np.random.Generator, mc_config: Dict[str, float],
) -> List[int]:
    """
    `seeds`: 10 team_id de UNA conferencia, ordenados 1..10 por victorias
    descendente. Formato real de play-in de la NBA: 7 vs 8 -> el ganador
    es el seed 7 de playoffs; el perdedor de 7v8 se enfrenta al ganador de
    9 vs 10 por el último cupo (seed 8). Devuelve los 8 team_id que
    avanzan a playoffs, en orden de seed 1-8.
    """
    if len(seeds) != 10:
        raise ValueError("resolve_play_in espera exactamente 10 seeds.")
    top6 = seeds[:6]
    seed7, seed8, seed9, seed10 = seeds[6], seeds[7], seeds[8], seeds[9]

    winner_7_8 = _single_game_winner(seed7, seed8, team_projections, rng, mc_config)
    loser_7_8 = seed8 if winner_7_8 == seed7 else seed7
    winner_9_10 = _single_game_winner(seed9, seed10, team_projections, rng, mc_config)
    winner_final_spot = _single_game_winner(loser_7_8, winner_9_10, team_projections, rng, mc_config)

    return top6 + [winner_7_8, winner_final_spot]


def simulate_conference_bracket(
    seeds_8: List[int], team_projections: Dict[int, Dict[str, Any]],
    rng: np.random.Generator, mc_config: Dict[str, float],
) -> Dict[str, Any]:
    """
    `seeds_8`: 8 team_id ya resueltos (post play-in), en orden 1-8.
    Bracket FIJO 1v8/4v5/3v6/2v7 (sin re-seeding entre rondas -- ver
    simplificaciones documentadas en el docstring del módulo).
    """
    seed_rank = {tid: i for i, tid in enumerate(seeds_8)}
    round1_pairs = [
        (seeds_8[0], seeds_8[7]), (seeds_8[3], seeds_8[4]),
        (seeds_8[2], seeds_8[5]), (seeds_8[1], seeds_8[6]),
    ]
    round1_winners = [
        _series_winner(a, b, team_projections, rng, mc_config, seed_rank=seed_rank) for a, b in round1_pairs
    ]

    semi_pairs = [(round1_winners[0], round1_winners[1]), (round1_winners[2], round1_winners[3])]
    semi_winners = [
        _series_winner(a, b, team_projections, rng, mc_config, seed_rank=seed_rank) for a, b in semi_pairs
    ]

    conference_champion = _series_winner(
        semi_winners[0], semi_winners[1], team_projections, rng, mc_config, seed_rank=seed_rank
    )

    return {
        "round1_winners": round1_winners,
        "conf_semis_winners": semi_winners,
        "conference_champion": conference_champion,
    }


def simulate_playoffs_once(
    wins_by_team: Dict[int, int],
    team_conference: Dict[int, str],
    team_projections: Dict[int, Dict[str, Any]],
    rng: np.random.Generator,
    mc_config: Dict[str, float],
) -> Dict[str, Any]:
    """Una realización completa de playoffs (play-in + 3 rondas + Finales) para UNA temporada simulada."""
    east_seeds = sorted(
        (t for t in wins_by_team if team_conference[t] == "East"), key=lambda t: -wins_by_team[t]
    )
    west_seeds = sorted(
        (t for t in wins_by_team if team_conference[t] == "West"), key=lambda t: -wins_by_team[t]
    )

    # Solo los 10 mejores de cada conferencia participan en playoffs/play-in
    # -- los seeds 11+ quedan eliminados de la temporada regular.
    east_8 = resolve_play_in(east_seeds[:10], team_projections, rng, mc_config)
    west_8 = resolve_play_in(west_seeds[:10], team_projections, rng, mc_config)

    east_result = simulate_conference_bracket(east_8, team_projections, rng, mc_config)
    west_result = simulate_conference_bracket(west_8, team_projections, rng, mc_config)

    # En las Finales la ventaja de campo NO va por seed de conferencia
    # (ambos son el seed 1 de la suya) sino por el mejor récord de
    # temporada regular de toda la liga -- regla real de la NBA.
    finals_seed_rank = {
        tid: -wins_by_team.get(tid, 0)
        for tid in (east_result["conference_champion"], west_result["conference_champion"])
    }
    nba_champion = _series_winner(
        east_result["conference_champion"], west_result["conference_champion"], team_projections, rng, mc_config,
        seed_rank=finals_seed_rank,
    )

    return {
        "east_8": east_8, "west_8": west_8,
        "east_result": east_result, "west_result": west_result,
        "nba_champion": nba_champion,
    }


def project_own_team_for_league(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Proyecta el equipo del config -- el roster HIPOTÉTICO de
    team_config.yaml (con `minutes_projection` curado a mano), NO el
    roster real actual de esa franquicia -- con el mismo shape de salida
    que project_team_roster(), para poder sustituir su entrada dentro de
    los 30 equipos de load_and_project_all_teams().

    IMPORTANTE: sin esta función, "Liga y Playoffs" trataría TODOS los 30
    equipos por igual, incluido el propio -- descargaría el roster REAL
    actual de esa franquicia (league_rosters.csv, vía CommonTeamRoster) y
    recalcularía minutos automáticamente (rotación top-10 por minutos
    reales, normalizada a 240), ignorando el roster hipotético de
    team_config.yaml por completo. Eso haría que la misma franquicia
    apareciera con números de victorias distintos en "Mi equipo" y en
    "Liga NBA", y que ni siquiera el roster de jugadores coincidiera (el
    roster real de una franquicia puede no incluir a los fichajes
    hipotéticos configurados por el usuario). Esta función reutiliza los
    MISMOS CSV ya calculados que usa la pestaña "Mi equipo"
    (aging_curve_projection.csv, injury_risk.csv, fatigue_risk.csv) --
    así el equipo del usuario aparece con exactamente los mismos números
    en ambas pestañas. Los otros 29 equipos siguen usando sus rosters
    reales sin cambios.
    """
    paths = get_paths(config)
    required = {
        "aging_curve_projection.csv": "context.aging_curve.build_aging_projection_dataset",
        "injury_risk.csv": "context.injury_model.build_injury_risk_dataset",
        "fatigue_risk.csv": "context.fatigue_accumulation.build_fatigue_dataset",
    }
    for filename, builder in required.items():
        path = paths["processed"] / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}. Corre `{builder}` primero.")

    aging = pd.read_csv(paths["processed"] / "aging_curve_projection.csv").set_index("player_id")
    injury = pd.read_csv(paths["processed"] / "injury_risk.csv").set_index("player_id")
    fatigue = pd.read_csv(paths["processed"] / "fatigue_risk.csv").set_index("player_id")
    positions_path = paths["processed"] / "roster_positions.csv"
    # Opcional: sin roster_positions.csv (falta correr
    # data_pipeline.build_roster_positions_dataset), el equipo propio
    # simplemente no participa en los quintetos All-NBA/All-Defensive ni
    # en el chequeo de cuota del All-Star (que exigen posición/país) pero
    # el resto del pipeline sigue funcionando -- degradar es mejor que
    # fallar. Mismo CSV trae las dos columnas (una sola llamada a
    # CommonPlayerInfo por jugador, ver data_pipeline.py).
    positions_df = (
        pd.read_csv(positions_path) if positions_path.exists() and positions_path.stat().st_size > 0 else None
    )
    positions_by_player = positions_df.set_index("player_id")["position"].to_dict() if positions_df is not None else {}
    countries_by_player = (
        positions_df.set_index("player_id")["country"].to_dict()
        if positions_df is not None and "country" in positions_df.columns
        else {}
    )

    roster_cfg = [p for p in config["roster"] if p.get("player_id")]
    player_ids = [p["player_id"] for p in roster_cfg]
    missing = [pid for pid in player_ids if pid not in aging.index]
    if missing:
        raise ValueError(
            f"player_id(s) {missing} están en el roster pero no en aging_curve_projection.csv. "
            "Corre el pipeline de proyección para todos los jugadores del roster."
        )

    minutes_by_player = {p["player_id"]: p.get("minutes_projection", 0) for p in roster_cfg}
    names_by_player = {p["player_id"]: p["name"] for p in roster_cfg}

    game_score_per36, minutes, risk_scores, fatigue_scores = [], [], [], []
    profiles: Dict[int, Dict[str, float]] = {}
    player_rows: List[Dict[str, Any]] = []

    for player_id in player_ids:
        row = aging.loc[player_id]
        minutes_per_game = minutes_by_player[player_id]
        risk = float(injury.loc[player_id, "risk_score"]) if player_id in injury.index else 0.0
        fat = float(fatigue.loc[player_id, "fatigue_score"]) if player_id in fatigue.index else 0.0

        game_score_per36.append(row["game_score_per36"])
        risk_scores.append(risk)
        fatigue_scores.append(fat)
        minutes.append(minutes_per_game)
        profiles[player_id] = compute_style_profile(row)
        player_rows.append(
            {
                "player_id": player_id,
                "player_name": names_by_player[player_id],
                "position": positions_by_player.get(player_id),
                "country": countries_by_player.get(player_id),
                "minutes_projection": minutes_per_game,
                "risk_score": risk,
                "fatigue_score": fat,
                **row.to_dict(),
            }
        )

    syn_cfg = config.get("lineup_synergy", {})
    synergy_matrix = build_synergy_matrix(
        player_ids,
        profiles,
        minutes_by_player,
        usage_threshold=syn_cfg.get("usage_threshold", 18.0),
        usage_clash_weight=syn_cfg.get("usage_clash_weight", 0.05),
        playmaking_spacing_weight=syn_cfg.get("playmaking_spacing_weight", 0.02),
    )

    return {
        "player_rows": player_rows,
        "player_ids": player_ids,
        "game_score_per36": np.array(game_score_per36, dtype=float),
        "minutes_projection": np.array(minutes, dtype=float),
        "risk_scores": np.array(risk_scores, dtype=float),
        "fatigue_scores": np.array(fatigue_scores, dtype=float),
        "synergy_matrix": synergy_matrix,
    }


def load_and_project_all_teams(
    config: Dict[str, Any],
) -> Tuple[List[int], Dict[int, str], Dict[int, str], Dict[int, Dict[str, Any]]]:
    """
    Lee league_rosters.csv y league_player*_stats.csv (generados por
    data_pipeline.py --league) y proyecta los 30 equipos con
    project_team_roster(). Compartido entre build_league_simulation_dataset()
    y simulate_single_bracket() para no duplicar esta ingesta+proyección.
    Devuelve (team_ids, team_abbrev_by_id, team_conference, team_projections).
    """
    paths = get_paths(config)
    required = {
        "league_rosters.csv": "data_pipeline.build_league_rosters_dataset",
        "league_player_career_stats.csv": "data_pipeline.build_league_player_stats_dataset",
    }
    for filename, builder in required.items():
        path = paths["processed"] / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}. Corre `{builder}` primero.")

    rosters = pd.read_csv(paths["processed"] / "league_rosters.csv")
    player_regular_stats = pd.read_csv(paths["processed"] / "league_player_career_stats.csv")
    playoff_path = paths["processed"] / "league_player_playoff_career_stats.csv"
    player_playoff_stats = (
        pd.read_csv(playoff_path) if playoff_path.exists() and playoff_path.stat().st_size > 0 else pd.DataFrame()
    )

    team_ids = sorted(rosters["team_id"].unique().tolist())
    team_abbrev_by_id = rosters.drop_duplicates("team_id").set_index("team_id")["team_abbreviation"].to_dict()
    team_conference = {tid: TEAM_CONFERENCE[abbrev] for tid, abbrev in team_abbrev_by_id.items()}

    # Nacionalidad -- ver data_pipeline.build_league_player_countries_dataset.
    # NO viene en league_rosters.csv (CommonTeamRoster no la trae); es un
    # lookup GLOBAL por player_id, no por equipo, así que se aplica después
    # de proyectar cada equipo en vez de threadearla por project_team_roster
    # (que solo ve el roster de UN equipo a la vez). Opcional: sin el CSV,
    # cada jugador queda con country=None -- el chequeo de cuota del
    # All-Star se degrada a "no verificable", no falla.
    countries_path = paths["processed"] / "league_player_countries.csv"
    countries_by_player = (
        pd.read_csv(countries_path).set_index("player_id")["country"].to_dict()
        if countries_path.exists() and countries_path.stat().st_size > 0
        else {}
    )

    # MISMO contexto para los 30 equipos (ver project_team_roster): este
    # motor los compara entre sí, así que todos tienen que estar medidos
    # con la misma métrica.
    advanced_context = build_advanced_context(load_advanced_stats(paths["processed"]), config)

    team_projections: Dict[int, Dict[str, Any]] = {}
    for team_id in team_ids:
        roster_slice = rosters[rosters["team_id"] == team_id]
        player_regular_slice = player_regular_stats[player_regular_stats["team_id"] == team_id]
        player_playoff_slice = (
            player_playoff_stats[player_playoff_stats["team_id"] == team_id]
            if not player_playoff_stats.empty
            else pd.DataFrame()
        )
        team_projections[team_id] = project_team_roster(
            roster_slice, player_regular_slice, player_playoff_slice, config,
            advanced_context=advanced_context,
        )
        for row in team_projections[team_id]["player_rows"]:
            row["country"] = countries_by_player.get(row["player_id"])

    # El equipo del usuario usa su roster HIPOTÉTICO (team_config.yaml),
    # no el roster real actual de esa franquicia -- ver docstring de
    # project_own_team_for_league. Los otros 29 equipos quedan sin tocar.
    own_team_id = config["team"].get("team_id")
    if own_team_id in team_projections:
        team_projections[own_team_id] = project_own_team_for_league(config)

    return team_ids, team_abbrev_by_id, team_conference, team_projections


def _play_in_with_detail(
    seeds: List[int], team_projections: Dict[int, Dict[str, Any]],
    rng: np.random.Generator, mc_config: Dict[str, float],
) -> Tuple[List[int], Dict[str, Any]]:
    """Igual que resolve_play_in(), pero conserva el detalle de cada partido
    de play-in (para poder mostrarlo en un bracket, no solo el resultado final)."""
    if len(seeds) != 10:
        raise ValueError("_play_in_with_detail espera exactamente 10 seeds.")
    top6 = seeds[:6]
    seed7, seed8, seed9, seed10 = seeds[6], seeds[7], seeds[8], seeds[9]

    winner_7_8 = _single_game_winner(seed7, seed8, team_projections, rng, mc_config)
    loser_7_8 = seed8 if winner_7_8 == seed7 else seed7
    winner_9_10 = _single_game_winner(seed9, seed10, team_projections, rng, mc_config)
    winner_final_spot = _single_game_winner(loser_7_8, winner_9_10, team_projections, rng, mc_config)

    seeds_8 = top6 + [winner_7_8, winner_final_spot]
    detail = {
        "game_7_vs_8": {"team_a": seed7, "team_b": seed8, "winner": winner_7_8},
        "game_9_vs_10": {"team_a": seed9, "team_b": seed10, "winner": winner_9_10},
        "game_elimination": {"team_a": loser_7_8, "team_b": winner_9_10, "winner": winner_final_spot},
    }
    return seeds_8, detail


def _conference_bracket_with_matchups(
    seeds_8: List[int], team_projections: Dict[int, Dict[str, Any]],
    rng: np.random.Generator, mc_config: Dict[str, float],
) -> Dict[str, Any]:
    """Igual que simulate_conference_bracket(), pero además devuelve los
    EMPAREJAMIENTOS de cada ronda (no solo los ganadores), para dibujar el bracket."""
    seed_rank = {tid: i for i, tid in enumerate(seeds_8)}
    round1_pairs = [
        (seeds_8[0], seeds_8[7]), (seeds_8[3], seeds_8[4]),
        (seeds_8[2], seeds_8[5]), (seeds_8[1], seeds_8[6]),
    ]
    round1 = [
        {"team_a": a, "team_b": b,
         "winner": _series_winner(a, b, team_projections, rng, mc_config, seed_rank=seed_rank)}
        for a, b in round1_pairs
    ]
    semi_pairs = [(round1[0]["winner"], round1[1]["winner"]), (round1[2]["winner"], round1[3]["winner"])]
    conf_semis = [
        {"team_a": a, "team_b": b,
         "winner": _series_winner(a, b, team_projections, rng, mc_config, seed_rank=seed_rank)}
        for a, b in semi_pairs
    ]
    conference_champion = _series_winner(
        conf_semis[0]["winner"], conf_semis[1]["winner"], team_projections, rng, mc_config, seed_rank=seed_rank
    )
    return {"round1": round1, "conf_semis": conf_semis, "conference_champion": conference_champion}


def simulate_single_bracket(
    config: Dict[str, Any], random_seed: Optional[int] = None, scenario: str = SCENARIO_WITH_INJURIES
) -> Dict[str, Any]:
    """
    Simula UN bracket de playoffs concreto (no una distribución agregada
    de miles de temporadas) para poder mostrarlo como un bracket real en
    el dashboard -- play-in, ronda 1, semis de conferencia, finales de
    conferencia y Finales de la NBA, con el emparejamiento y ganador de
    cada serie. El seeding usa `league_regular_season_summary.csv` (la
    media de victorias simuladas ya calculada) como el récord de
    temporada regular de cada equipo -- no vuelve a simular 82 partidos
    para esto, sería redundante y lento para una sola vista interactiva.
    Todos los IDs de equipo en el resultado se devuelven como
    `team_abbreviation` (no team_id), listos para mostrar.

    `scenario` (ver `_apply_scenario`) decide si se lee
    `league_regular_season_summary.csv` o su variante
    `..._no_injuries.csv`, y si el roster de cada equipo se juega el
    bracket con o sin riesgo de lesión -- para que el bracket sea
    coherente con el escenario activo en el resto de la app.
    """
    paths = get_paths(config)
    regular_season_path = paths["processed"] / f"league_regular_season_summary{_scenario_suffix(scenario)}.csv"
    if not regular_season_path.exists():
        raise FileNotFoundError(
            f"No se encontró {regular_season_path}. Corre "
            "`league_simulation.build_league_simulation_dataset` primero."
        )
    regular_season = pd.read_csv(regular_season_path)
    wins_by_team = regular_season.set_index("team_id")["wins_mean"].to_dict()

    team_ids, team_abbrev_by_id, team_conference, team_projections = load_and_project_all_teams(config)
    team_projections = _apply_scenario(team_projections, scenario)

    seed = random_seed if random_seed is not None else config["simulation"]["random_seed"]
    rng = np.random.default_rng(seed)
    mc_cfg = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}

    east_seeds = sorted((t for t in team_ids if team_conference[t] == "East"), key=lambda t: -wins_by_team[t])[:10]
    west_seeds = sorted((t for t in team_ids if team_conference[t] == "West"), key=lambda t: -wins_by_team[t])[:10]

    east_8, east_play_in = _play_in_with_detail(east_seeds, team_projections, rng, mc_cfg)
    west_8, west_play_in = _play_in_with_detail(west_seeds, team_projections, rng, mc_cfg)

    east_bracket = _conference_bracket_with_matchups(east_8, team_projections, rng, mc_cfg)
    west_bracket = _conference_bracket_with_matchups(west_8, team_projections, rng, mc_cfg)

    # Ventaja de campo en las Finales: mejor récord de temporada regular
    # de toda la liga (no seed de conferencia) -- misma regla real que en
    # simulate_playoffs_once.
    finals_seed_rank = {
        tid: -wins_by_team.get(tid, 0)
        for tid in (east_bracket["conference_champion"], west_bracket["conference_champion"])
    }
    nba_champion = _series_winner(
        east_bracket["conference_champion"], west_bracket["conference_champion"], team_projections, rng, mc_cfg,
        seed_rank=finals_seed_rank,
    )

    def _to_abbrev(value):
        if isinstance(value, dict):
            return {k: _to_abbrev(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_abbrev(v) for v in value]
        return team_abbrev_by_id.get(value, value)

    return {
        "east": {
            "seeds_10": _to_abbrev(east_seeds),
            "seeds_8": _to_abbrev(east_8),
            "play_in": _to_abbrev(east_play_in),
            **_to_abbrev(east_bracket),
        },
        "west": {
            "seeds_10": _to_abbrev(west_seeds),
            "seeds_8": _to_abbrev(west_8),
            "play_in": _to_abbrev(west_play_in),
            **_to_abbrev(west_bracket),
        },
        "nba_champion": team_abbrev_by_id.get(nba_champion, nba_champion),
    }


def build_league_simulation_dataset(
    config: Dict[str, Any], scenario: str = SCENARIO_WITH_INJURIES
) -> Dict[str, pd.DataFrame]:
    """
    Punto de entrada: lee league_rosters.csv y league_player*_stats.csv
    (generados por data_pipeline.py), proyecta los 30 equipos, simula la
    temporada regular y luego los playoffs (con un nº de temporadas de
    playoffs configurable y más barato, ver league_n_seasons), y guarda:
    - data/processed/league_player_projections.csv (una fila por jugador
      de los 30 equipos: proyección, risk_score, fatigue_score, stats por
      partido -- para poder navegar cualquier equipo en el dashboard)
    - data/processed/league_regular_season_summary.csv (una fila por equipo)
    - data/processed/league_playoff_summary.csv (una fila por equipo: %
      de veces que hace playoffs / ronda alcanzada / campeonato)

    `scenario` (`SCENARIO_WITH_INJURIES` por defecto, o
    `SCENARIO_NO_INJURIES`) decide si se sortean ausencias por lesión --
    ver `_apply_scenario`. Los 3 CSV de salida llevan el sufijo
    correspondiente (`_scenario_suffix`); con el escenario por defecto
    los nombres son EXACTAMENTE los de siempre, sin sufijo.
    """
    paths = get_paths(config)
    suffix = _scenario_suffix(scenario)
    team_ids, team_abbrev_by_id, team_conference, team_projections = load_and_project_all_teams(config)
    team_projections = _apply_scenario(team_projections, scenario)

    games_per_season_for_players = config["simulation"]["games_per_season"]
    player_projection_rows = []
    for team_id in team_ids:
        for row in team_projections[team_id]["player_rows"]:
            player_projection_rows.append(
                {
                    "team_id": team_id,
                    "team_abbreviation": team_abbrev_by_id[team_id],
                    "conference": team_conference[team_id],
                    **row,
                }
            )
    player_projections_df = pd.DataFrame(player_projection_rows)
    for stat, total_col in {
        "PPG": "PTS_projected", "RPG": "REB_projected", "APG": "AST_projected",
        "SPG": "STL_projected", "BPG": "BLK_projected", "TOPG": "TOV_projected",
        "3PM": "FG3M_projected",
    }.items():
        if total_col in player_projections_df.columns:
            player_projections_df[stat] = player_projections_df[total_col] / games_per_season_for_players
    if {"FGM_projected", "FGA_projected"}.issubset(player_projections_df.columns):
        player_projections_df["FG%"] = (
            player_projections_df["FGM_projected"] / player_projections_df["FGA_projected"] * 100
        ).round(1)
    if {"FG3M_projected", "FG3A_projected"}.issubset(player_projections_df.columns):
        player_projections_df["3P%"] = (
            player_projections_df["FG3M_projected"] / player_projections_df["FG3A_projected"] * 100
        ).round(1)

    player_projections_path = paths["processed"] / f"league_player_projections{suffix}.csv"
    player_projections_df.to_csv(player_projections_path, index=False)
    print(f"Guardado: {player_projections_path} ({len(player_projections_df)} jugadores)")

    # Ajuste de sinergia ESPERADO de cada equipo, en puntos de net rating.
    # Se guarda para que simulation.py (motor de "Mi equipo") pueda centrar
    # su línea base -- ver el bug de suma cero en el docstring de
    # backtesting.expected_team_game_score_equivalent: la sinergia es
    # SIEMPRE POSITIVA y se suma al net rating de todos los equipos, así
    # que la línea base tiene que llevar la media de la liga incorporada.
    # Solo este módulo puede calcularla: es el único que tiene la matriz de
    # sinergia de los 30 equipos a la vez.
    synergy_rng = np.random.default_rng(config["simulation"]["random_seed"])
    synergy_rows = []
    for team_id, projection in team_projections.items():
        if projection.get("synergy_matrix") is None:
            continue
        available = sample_injury_absences(
            projection["risk_scores"],
            LEAGUE_SYNERGY_SAMPLING_SEASONS,
            config["simulation"]["games_per_season"],
            synergy_rng,
            {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}["injury_dispersion"],
        )
        synergy_rows.append({
            "team_abbreviation": team_abbrev_by_id[team_id],
            "expected_synergy_net_rating": float(
                compute_game_synergy_adjustment(available, projection["synergy_matrix"]).mean()
            ),
        })
    synergy_path = paths["processed"] / "league_team_synergy_baseline.csv"
    pd.DataFrame(synergy_rows).to_csv(synergy_path, index=False)
    print(f"Guardado: {synergy_path} ({len(synergy_rows)} equipos)")

    league_cfg = config.get("league_simulation", {})
    mc_cfg = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}
    games_per_season = config["simulation"]["games_per_season"]
    n_seasons = league_cfg.get("n_seasons", DEFAULT_LEAGUE_N_SEASONS)
    random_seed = config["simulation"]["random_seed"]

    real_games = _load_real_schedule_games(config)
    if real_games is not None:
        print(f"Calendario REAL ({len(real_games)} partidos) -- corre `data_pipeline.py --league` para refrescarlo.")
        wins_by_team_arrays = simulate_league_regular_season_real_schedule(
            team_projections, real_games, n_seasons, mc_cfg, random_seed
        )
    else:
        print("Aviso: no se encontró league_schedule_full.csv -- usando calendario SINTÉTICO "
              "(corre `data_pipeline.py --league` para el real).")
        schedule_rng = np.random.default_rng(random_seed)
        schedule = build_round_robin_schedule(team_ids, games_per_season, schedule_rng)
        wins_by_team_arrays = simulate_league_regular_season(
            team_projections, schedule, n_seasons, games_per_season, mc_cfg, random_seed
        )

    regular_season_rows = [
        {
            "team_id": team_id,
            "team_abbreviation": team_abbrev_by_id[team_id],
            "conference": team_conference[team_id],
            "wins_mean": float(wins_by_team_arrays[team_id].mean()),
            "wins_p10": float(np.quantile(wins_by_team_arrays[team_id], 0.1)),
            "wins_p90": float(np.quantile(wins_by_team_arrays[team_id], 0.9)),
        }
        for team_id in team_ids
    ]
    regular_season_df = pd.DataFrame(regular_season_rows).sort_values("wins_mean", ascending=False)
    regular_out = paths["processed"] / f"league_regular_season_summary{suffix}.csv"
    regular_season_df.to_csv(regular_out, index=False)
    print(f"Guardado: {regular_out} ({len(regular_season_df)} equipos)")

    playoff_rng = np.random.default_rng(random_seed + 1)
    n_playoff_seasons = league_cfg.get("n_playoff_seasons", min(n_seasons, DEFAULT_LEAGUE_N_SEASONS))

    made_playoffs = {tid: 0 for tid in team_ids}
    reached_conf_semis = {tid: 0 for tid in team_ids}
    reached_conf_finals = {tid: 0 for tid in team_ids}
    reached_finals = {tid: 0 for tid in team_ids}
    won_championship = {tid: 0 for tid in team_ids}

    for i in range(n_playoff_seasons):
        wins_this_season = {tid: int(wins_by_team_arrays[tid][i]) for tid in team_ids}
        result = simulate_playoffs_once(wins_this_season, team_conference, team_projections, playoff_rng, mc_cfg)

        # IMPORTANTE: no confundir estas tres métricas de ronda -- cada
        # una cuenta un hito distinto. `conf_semis_winners` son los que
        # GANARON las semifinales (llegaron a las FINALES de conferencia),
        # no los que simplemente llegaron a semis -- esos son
        # `round1_winners`. `conference_champion` cuenta como "llegó a
        # finales de conferencia" (lo ganó) Y como "llegó a las Finales de
        # la NBA" (el mismo equipo, dos hitos distintos) -- si se
        # confunden, conf_finals_pct y finals_pct salen idénticos.
        for tid in result["east_8"] + result["west_8"]:
            made_playoffs[tid] += 1
        for tid in result["east_result"]["round1_winners"] + result["west_result"]["round1_winners"]:
            reached_conf_semis[tid] += 1
        for tid in result["east_result"]["conf_semis_winners"] + result["west_result"]["conf_semis_winners"]:
            reached_conf_finals[tid] += 1
        reached_finals[result["east_result"]["conference_champion"]] += 1
        reached_finals[result["west_result"]["conference_champion"]] += 1
        won_championship[result["nba_champion"]] += 1

    playoff_rows = [
        {
            "team_id": tid,
            "team_abbreviation": team_abbrev_by_id[tid],
            "conference": team_conference[tid],
            "playoff_pct": made_playoffs[tid] / n_playoff_seasons * 100,
            "conf_semis_pct": reached_conf_semis[tid] / n_playoff_seasons * 100,
            "conf_finals_pct": reached_conf_finals[tid] / n_playoff_seasons * 100,
            "finals_pct": reached_finals[tid] / n_playoff_seasons * 100,
            "championship_pct": won_championship[tid] / n_playoff_seasons * 100,
        }
        for tid in team_ids
    ]
    playoff_df = pd.DataFrame(playoff_rows).sort_values("championship_pct", ascending=False)
    playoff_out = paths["processed"] / f"league_playoff_summary{suffix}.csv"
    playoff_df.to_csv(playoff_out, index=False)
    print(f"Guardado: {playoff_out} ({len(playoff_df)} equipos, {n_playoff_seasons} temporadas de playoffs simuladas)")

    return {
        "regular_season": regular_season_df,
        "playoffs": playoff_df,
        "player_projections": player_projections_df,
    }


if __name__ == "__main__":
    from config_loader import load_config

    build_league_simulation_dataset(load_config())
