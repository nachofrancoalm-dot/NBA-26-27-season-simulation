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

BUG REAL ARREGLADO AQUÍ (mismo patrón que ya pasó con la normalización
de minutos): este módulo NO aplicaba `game_score_to_net_rating_scale` --
metía la diferencia de Game Score en bruto en la logística, o sea seguía
asumiendo "1 punto de Game Score = 1 punto de diferencial", justo la
suposición que el backtest sweep midió y refutó (0.21). El fix se aplicó
en su día solo a simulation.py y nunca se propagó aquí. Consecuencia: la
liga trataba las diferencias entre equipos ~4.8x más fuerte de lo
calibrado, y ese era el motivo real de que la MISMA franquicia diera
números distintos en "Mi equipo" y en "Liga NBA" -- lo que se había
documentado como "diferencia de diseño entre motores" era, en su mayor
parte, este bug. `outcome_variance_scale` significa ahora lo mismo en los
dos motores: puntos de DIFERENCIAL, no de Game Score.

SIMPLIFICACIONES EN PLAYOFFS (documentadas, no ocultas)
------------------------------------------------------------
- Bracket FIJO (1v8, 2v7, 3v6, 4v5 y así sucesivamente), sin re-seeding
  entre rondas -- la NBA real sí resiembra; se simplifica aquí.
- Disponibilidad en playoffs: SÍ se sortea (Bernoulli por partido con la
  misma `risk_score` de la temporada regular). Antes se asumía roster a
  plena salud "porque el beneficio sería marginal" -- resultó ser un bug
  grave que hacía a los equipos de estrellas frágiles más favoritos al
  título que al mejor equipo de temporada regular; ver el docstring de
  `_sample_team_game_score`. Lo que NO se replica es el tramo CONTIGUO de
  lesión de `sample_injury_absences` (en series de 4-7 partidos la
  diferencia entre racha y sorteo por partido es pequeña).
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
from aging_curve import project_player_season  # noqa: E402
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
    TOTAL_TEAM_MINUTES_PER_GAME,
    apply_star_bonus,
    compute_player_contributions,
    normalize_rotation_minutes,
    sample_injury_absences,
)
from context.injury_model import compute_risk_score  # noqa: E402
from context.fatigue_accumulation import compute_fatigue_score  # noqa: E402

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

    DOS problemas reales encontrados, en dos iteraciones de este fix:

    1. La suma de esos minutos "en bruto" por TODO el roster (15-22
       jugadores, incluyendo a cualquiera que jugó un solo partido por
       una lesión ajena) no tiene por qué sumar 240 (5 posiciones x 48
       min, lo único que existe de verdad en un partido) -- de hecho casi
       nunca lo hace, porque a lo largo de 82 partidos rotan más de 5-8
       jugadores. Un roster con mucho movimiento de plantilla (lesiones,
       tanking, muchos jugadores de two-way/G-League llamados) suma
       mucho más que uno estable -- Utah llegó a sumar 449 "en bruto"
       (22 jugadores, varios con muy pocos partidos jugados) mientras
       Los Angeles Lakers sumaban 318 con un roster más corto.
    2. La primera versión de este fix escalaba TODO el roster para que
       sumara exactamente 240 -- pero eso penaliza injustamente a las
       ESTRELLAS de un roster con mucho movimiento de plantilla: Luka
       Dončić pasó de sus ~35.8 min/partido reales a 26.98 tras esa
       normalización, solo porque sus compañeros de banquillo (varios
       con pocos partidos, de rotación de temporada) inflaban el total
       del equipo. Un jugador de banquillo con pocos minutos reales por
       *movimiento de plantilla* no debería diluir los minutos de la
       estrella real del equipo.

    SOLUCIÓN: normalizar solo dentro de una ROTACIÓN REALISTA -- los
    `rotation_size` jugadores (10 por defecto) con más minutos/partido en
    bruto, escalados para sumar 240 entre ellos. El resto del roster
    (suplentes de fondo de plantilla, two-way, llamados puntuales) se
    trata como 0 minutos -- no afectan de forma relevante a la fuerza de
    un equipo en una aproximación de este tipo, y diluir la rotación real
    con ellos es precisamente lo que causaba el problema.
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

    # --- Paso 1: minutos "en bruto" de cada jugador, sin proyectar todavía ---
    raw_minutes: Dict[int, float] = {}
    recent_rows: Dict[int, Any] = {}
    for player_id in player_ids:
        player_regular = player_regular_stats[player_regular_stats["PLAYER_ID"] == player_id]
        if player_regular.empty:
            raw_minutes[player_id] = 0.0
            continue
        recent_row = _most_recent_season_row(player_regular)
        recent_rows[player_id] = recent_row
        raw_minutes[player_id] = (
            float(recent_row["MIN"]) / float(recent_row["GP"]) if float(recent_row["GP"]) > 0 else 0.0
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
        )
        risk = compute_risk_score(player_regular)
        fatigue = compute_fatigue_score(player_regular, player_playoff if not player_playoff.empty else None)

        # Sobrescribir DENTRO de `projection`, no solo en la lista que usa
        # la simulación: `projection` se vuelca tal cual en player_rows ->
        # league_player_projections.csv, y ese CSV es el que lee
        # simulation.compute_league_average_game_score_per36 para su línea
        # base. Si la simulación usara la métrica ajustada y el CSV la
        # cruda, "Mi equipo" se compararía contra una referencia medida en
        # otra escala -- exactamente la clase de desajuste entre motores
        # que este proyecto ya arrastró dos veces.
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
    team_projections a la vez (vectorizado por equipo sobre n_seasons).
    Devuelve {team_id: wins_array(n_seasons,)}.
    """
    rng = np.random.default_rng(random_seed)

    team_game_scores: Dict[int, np.ndarray] = {}
    for team_id, proj in team_projections.items():
        available = sample_injury_absences(
            proj["risk_scores"], n_seasons, games_per_season, rng, mc_config["injury_dispersion"]
        )
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

    wins = {team_id: np.zeros(n_seasons, dtype=int) for team_id in team_projections}
    for day, team_a, team_b in schedule:
        score_a = team_game_scores[team_a][:, day]
        score_b = team_game_scores[team_b][:, day]
        # La diferencia de Game Score se convierte a puntos de diferencial
        # con la escala calibrada antes de entrar en la logística -- ver el
        # bug documentado en el docstring del módulo.
        point_differential = (score_a - score_b) * mc_config["game_score_to_net_rating_scale"]
        win_prob_a = 1 / (1 + np.exp(-point_differential / mc_config["outcome_variance_scale"]))
        team_a_wins = rng.random(n_seasons) < win_prob_a
        wins[team_a] += team_a_wins
        wins[team_b] += ~team_a_wins

    return wins


def _sample_team_game_score(proj: Dict[str, Any], rng: np.random.Generator, mc_config: Dict[str, float]) -> float:
    """
    Un partido de playoffs para UN equipo -- ver simplificaciones en el
    docstring del módulo. Aplica la misma prima de estrella que la
    temporada regular (ver simulation.apply_star_bonus) -- si no, un
    equipo top-heavy se vería penalizado justo en la parte de la
    temporada donde más importa.

    BUG REAL ARREGLADO -- DISPONIBILIDAD EN PLAYOFFS: esta función asumía
    el roster a PLENA SALUD en playoffs (era una "simplificación
    documentada": extender el modelo de lesiones a partidos de playoff
    "añadiría complejidad por un beneficio marginal limitado"). No era
    marginal: producía un resultado abiertamente contradictorio. Un
    equipo construido sobre estrellas frágiles era castigado los 82
    partidos de temporada regular y luego llegaba a playoffs
    milagrosamente sano. Caso real encontrado por el usuario -- PHI vs
    SAS con los datos de 2026-27:

        equipo   GS a plena salud   GS esperado con lesiones   victorias
        PHI            111.0                  76.7               45.5
        SAS            106.0                  87.0               56.4

    Es decir: SAS ganaba 11 victorias MÁS que PHI en temporada regular
    (porque PHI pierde el 31% de su producción por lesiones, con Embiid
    en riesgo 0.65) y aun así PHI tenía **23.7%** de título contra
    **10.8%** de SAS, por jugar los playoffs a plena salud. Un equipo con
    peor temporada regular no puede ser más favorito al título solo
    porque el modelo le perdona las lesiones en el momento decisivo.

    Arreglado sorteando disponibilidad por partido con la misma
    `risk_score` que usa la temporada regular (Bernoulli por partido: la
    semántica de risk_score es "fracción esperada de partidos perdidos",
    así que aplicarla por partido conserva esa media). NO se replica el
    tramo contiguo de `sample_injury_absences` -- en una serie de 4-7
    partidos la diferencia entre "racha" y "sorteo por partido" es
    pequeña, y el sesgo grave (salud perfecta gratis) ya queda corregido.
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

    BUG REAL encontrado por el usuario: antes de esto, "Liga y Playoffs"
    trataba TODOS los 30 equipos por igual, incluido el propio -- volvía
    a descargar el roster REAL actual de esa franquicia
    (league_rosters.csv, vía CommonTeamRoster) y recalculaba minutos
    automáticamente (rotación top-10 por minutos reales, normalizada a
    240), ignorando por completo el roster hipotético de
    team_config.yaml. Resultado: la misma franquicia (p. ej. PHI)
    aparecía con victorias medias distintas en "Mi equipo" (50.4) que en
    "Liga NBA" (42.994) -- ni siquiera el roster de jugadores coincidía
    (el roster real de PHI puede no incluir a los mismos fichajes
    hipotéticos que configuró el usuario). Esta función reutiliza los
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


def simulate_single_bracket(config: Dict[str, Any], random_seed: Optional[int] = None) -> Dict[str, Any]:
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
    """
    paths = get_paths(config)
    regular_season_path = paths["processed"] / "league_regular_season_summary.csv"
    if not regular_season_path.exists():
        raise FileNotFoundError(
            f"No se encontró {regular_season_path}. Corre "
            "`league_simulation.build_league_simulation_dataset` primero."
        )
    regular_season = pd.read_csv(regular_season_path)
    wins_by_team = regular_season.set_index("team_id")["wins_mean"].to_dict()

    team_ids, team_abbrev_by_id, team_conference, team_projections = load_and_project_all_teams(config)

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


def build_league_simulation_dataset(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
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
    """
    paths = get_paths(config)
    team_ids, team_abbrev_by_id, team_conference, team_projections = load_and_project_all_teams(config)

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

    player_projections_path = paths["processed"] / "league_player_projections.csv"
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
    regular_out = paths["processed"] / "league_regular_season_summary.csv"
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

        # BUG REAL ARREGLADO: las tres métricas de ronda estaban
        # desplazadas una ronda, y dos eran idénticas por construcción.
        # `conf_semis_winners` son los que GANARON las semifinales (o sea,
        # los que llegaron a las FINALES de conferencia), no los que
        # llegaron a semis -- esos son `round1_winners`. Y
        # `conference_champion` se contaba a la vez como "llegó a finales
        # de conferencia" y como "llegó a las Finales de la NBA", así que
        # conf_finals_pct y finals_pct salían iguales en los 30 equipos.
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
    playoff_out = paths["processed"] / "league_playoff_summary.csv"
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
