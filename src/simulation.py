"""
simulation.py

Motor Monte Carlo: combina las 6 señales de contexto (injury_model,
fatigue_accumulation, schedule_strength, performance_curve,
opponent_weighting, conference_adjustment) con la proyección individual
(aging_curve.py) para simular n_seasons del equipo configurado.

Calendario: no existe el calendario real 2026-27 todavía, así que cada
partido muestrea el WinPCT de un rival de la distribución de la liga y
un flag de back-to-back (b2b_probability, 0.18). Sustituible por
team_schedule.csv en cuanto la NBA lo publique -- la interfaz no cambia.

Por temporada: las lesiones sortean una racha continua de ausencia
(binomial negativa); cada partido puntúa a los disponibles por Game
Score/36 escalado a minutos, con penalización de fatiga, desgaste de
temporada y ruido; el Game Score de equipo vs. rival alimenta una
probabilidad de victoria logística. Escalas en config["monte_carlo"],
aproximadas, no ajustadas por regresión.

Línea base de liga: la línea base "equipo promedio" contra la que se mide
el Game Score de equipo se recalibra desde los 30 equipos reales
proyectados (compute_league_average_game_score_per36) en vez de usar el
valor genérico de Hollinger (10.0), que infla las victorias proyectadas
al comparar contra un rival ficticio demasiado flojo. backtesting.py usa
en su lugar la línea base por-era de
aging_curve.compute_league_game_score_baseline(), calculada de game logs
históricos, porque no existe league_player_projections.csv por temporada
pasada.

Escala Game Score -> diferencial: la pendiente empírica
(game_score_to_net_rating_scale, 0.1617) es muy inferior a 1.0;
recalibrada junto con advanced_impact.py y validada con
leave-one-season-out. Ventaja de campo medida en los game logs del
backtest sweep: +2.41 pts en casa en temporada regular, +3.98 en
playoffs.

Limitación conocida: el simulador reparte títulos a seeds 4+ con más
frecuencia que la realidad porque comprime la señal de talento entre
equipos relativo al ruido de temporada (señal/ruido 0.85 vs. 2.49 real).
La compresión de talento es correcta para minimizar error de proyección;
tratarla como el talento verdadero al simular series es la causa, y
arreglarlo requiere separar incertidumbre de estimación de ruido de
temporada (cambio arquitectónico, no un ajuste de parámetro).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import get_paths  # noqa: E402
from lineup_synergy import build_synergy_matrix, compute_game_synergy_adjustment, compute_style_profile  # noqa: E402

TOTAL_TEAM_MINUTES_PER_GAME = 240.0  # 5 posiciones x 48 minutos -- constante del deporte, no del equipo

DEFAULT_MONTE_CARLO_CONFIG: Dict[str, float] = {
    "injury_dispersion": 2.0,  # binomial negativa: menor = más varianza en partidos perdidos
    "b2b_probability": 0.18,
    "b2b_fatigue_penalty": 0.15,  # % de reducción máxima en back-to-back para fatigue_score=1
    "season_fatigue_decay": 0.10,  # % de reducción máxima al final de temporada para fatigue_score=1
    "game_variance_std": 3.0,  # desviación estándar del ruido de Game Score por partido
    # Fallback genérico (Hollinger, promedio de TODA la liga); se recalibra
    # automáticamente desde league_player_projections.csv cuando existe
    # (ver compute_league_average_game_score_per36 y el docstring del módulo).
    "league_average_game_score_per36": 10.0,
    # Pendiente empírica Game Score -> diferencial de puntos, ver docstring
    # del módulo. Recalibrar junto con advanced_impact.py si cambia.
    "game_score_to_net_rating_scale": 0.1617,
    "opponent_strength_scale": 20.0,  # cuánto resta/suma un rival top/flojo (WinPCT 1.0 vs 0.0) al diferencial
    "outcome_variance_scale": 12.0,  # dispersión típica de resultado de un partido NBA individual
    # Puntos de diferencial (no Game Score), medidos en el backtest sweep -- ver docstring del módulo.
    "home_court_advantage": 2.41,
    "playoff_home_court_advantage": 3.98,
    # Desactivada por defecto -- ver docstring de apply_star_bonus.
    "star_bonus_top_n": 0,  # nº de jugadores (por valor de temporada) que reciben la prima -- ver apply_star_bonus
    "star_bonus_multiplier": 1.0,  # multiplicador de Game Score/36 efectivo para esos jugadores
    # Desactivada por defecto (0.0) -- ver docstring de sample_team_quality_noise.
    # Puntos de diferencial (mismas unidades que home_court_advantage).
    "team_quality_uncertainty_std": 0.0,
}


def apply_star_bonus(
    game_score_per36: np.ndarray, minutes_projection: np.ndarray, mc_config: Dict[str, float]
) -> np.ndarray:
    """
    Prima de "estrella": multiplica el Game Score/36 efectivo de los
    `star_bonus_top_n` jugadores con más "valor de temporada"
    (game_score_per36 * minutos/36) por `star_bonus_multiplier`.

    Desactivada por defecto (`star_bonus_top_n=0`): se probó y se
    descartó porque no aísla bien el efecto buscado (todo equipo tiene su
    propio mejor jugador) y empeora el backtesting contra los
    comparables históricos de superequipos, que en la realidad rindieron
    por debajo de la suma de su talento. Queda como palanca de
    experimentación opcional en `config["monte_carlo"]`.
    """
    top_n = int(mc_config.get("star_bonus_top_n", 0))
    multiplier = mc_config.get("star_bonus_multiplier", 1.0)
    if top_n <= 0 or multiplier == 1.0 or len(game_score_per36) == 0:
        return game_score_per36

    season_value = game_score_per36 * minutes_projection / 36.0
    star_indices = np.argsort(-season_value)[:top_n]
    boosted = game_score_per36.copy()
    boosted[star_indices] = boosted[star_indices] * multiplier
    return boosted


DEFAULT_ROTATION_SIZE = 10  # tamaño de rotación real NBA -- ver normalize_rotation_minutes


def normalize_rotation_minutes(
    raw_minutes_by_player: Dict[Any, float], rotation_size: int = DEFAULT_ROTATION_SIZE
) -> Dict[Any, float]:
    """
    Escala los minutos/partido reales de un roster para que la rotación
    (los `rotation_size` jugadores con más minutos) sume exactamente
    TOTAL_TEAM_MINUTES_PER_GAME (240 = 5 posiciones x 48 min). Los
    jugadores fuera de esa rotación quedan en 0.

    Necesario porque un roster de temporada completa suma 280-345
    minutos/partido reales (rotan 14-20 jugadores en 82 partidos), y
    sumar el Game Score de todos ellos sin normalizar infla la fuerza del
    equipo 18-43%. Restringir la normalización a la rotación (en vez de
    escalar todo el roster) evita diluir a las estrellas. Función
    compartida entre `backtesting.py` y `league_simulation.py` para que
    la lógica no diverja entre los dos motores.
    """
    if not raw_minutes_by_player:
        return {}
    rotation_ids = sorted(raw_minutes_by_player, key=lambda pid: -raw_minutes_by_player[pid])[:rotation_size]
    total_rotation_minutes = sum(raw_minutes_by_player[pid] for pid in rotation_ids)
    if total_rotation_minutes <= 0:
        return {pid: 0.0 for pid in raw_minutes_by_player}

    scale = TOTAL_TEAM_MINUTES_PER_GAME / total_rotation_minutes
    rotation = set(rotation_ids)
    return {
        pid: (raw_minutes_by_player[pid] * scale if pid in rotation else 0.0)
        for pid in raw_minutes_by_player
    }


def load_league_mean_synergy(processed_dir: Path) -> float:
    """
    Media del ajuste de sinergia esperado de los 30 equipos, en puntos de
    net rating (de `league_team_synergy_baseline.csv`, que escribe
    league_simulation.py -- es el único módulo con las 30 matrices de
    sinergia a la vez). 0.0 si el archivo no existe, que deja el
    comportamiento anterior intacto.
    """
    path = processed_dir / "league_team_synergy_baseline.csv"
    if not path.exists() or path.stat().st_size == 0:
        return 0.0
    df = pd.read_csv(path)
    if df.empty or "expected_synergy_net_rating" not in df.columns:
        return 0.0
    return float(df["expected_synergy_net_rating"].mean())


def compute_league_average_game_score_per36(
    player_projections: pd.DataFrame,
    league_mean_synergy_net_rating: float = 0.0,
    game_score_to_net_rating_scale: float = DEFAULT_MONTE_CARLO_CONFIG["game_score_to_net_rating_scale"],
) -> float:
    """
    Recalibra `league_average_game_score_per36` desde los 30 equipos reales
    ya proyectados (league_player_projections.csv) en vez de usar el valor
    genérico de Hollinger (10.0, promedio sobre toda la liga incluida la
    basura de banquillo), que infla las victorias proyectadas del equipo
    propio. `player_projections` requiere team_abbreviation,
    game_score_per36, minutes_projection. Devuelve la tasa por-36
    equivalente al Game Score total medio de esos 30 equipos.

    Cada contribución se descuenta por `(1 - risk_score)` (fracción de
    partidos que se espera que el jugador esté disponible, media exacta
    de `sample_injury_absences`) para comparar contra un equipo propio
    igualmente penalizado por lesiones; se omite si no hay columna
    `risk_score`.
    """
    contribution = player_projections["game_score_per36"] * player_projections["minutes_projection"] / 36.0
    if "risk_score" in player_projections.columns:
        contribution = contribution * (1 - player_projections["risk_score"].clip(0, 1))

    team_totals = player_projections.assign(_contribution=contribution).groupby("team_abbreviation")["_contribution"].sum()
    average_team_total = float(team_totals.mean())

    # El ajuste de sinergia que run_monte_carlo suma al net rating es
    # sistemáticamente positivo; si la línea base no lo incorpora, el
    # equipo propio lo cobra "gratis". Se divide por la escala porque la
    # simulación lo suma después de convertir a diferencial.
    if league_mean_synergy_net_rating:
        average_team_total += league_mean_synergy_net_rating / game_score_to_net_rating_scale

    return average_team_total / (TOTAL_TEAM_MINUTES_PER_GAME / 36.0)


def compute_expected_games_played(risk_scores: np.ndarray, games_per_season: int) -> np.ndarray:
    """
    Partidos jugados esperados en la temporada simulada, por jugador:
    games_per_season * (1 - risk_score). Es la media exacta (forma
    cerrada) de la binomial negativa que usa `sample_injury_absences`, no
    una aproximación por Monte Carlo.
    """
    return games_per_season * (1 - np.clip(risk_scores, 0, 1))


def compute_expected_effective_minutes_per_game(
    minutes_projection: np.ndarray, risk_scores: np.ndarray
) -> np.ndarray:
    """
    Minutos por partido efectivos de la temporada simulada, por jugador:
    minutes_projection * (1 - risk_score) -- promedio sobre toda la
    temporada, contando como 0 los partidos que se espera perder por
    lesión. Distinto de `minutes_projection`, que es el input fijo de los
    partidos en que sí juega (los minutos no fluctúan partido a partido
    en este modelo).
    """
    return minutes_projection * (1 - np.clip(risk_scores, 0, 1))


def sample_injury_absences(
    risk_scores: np.ndarray,
    n_seasons: int,
    games_per_season: int,
    rng: np.random.Generator,
    dispersion: float,
) -> np.ndarray:
    """
    Devuelve un array booleano (n_seasons, games_per_season, n_players):
    True = jugador disponible ese partido esa temporada simulada. Cada
    jugador pierde, en cada temporada, un tramo CONTIGUO de partidos
    (no partidos sueltos al azar) de longitud sorteada de una binomial
    negativa con media risk_score * games_per_season.
    """
    n_players = len(risk_scores)
    available = np.ones((n_seasons, games_per_season, n_players), dtype=bool)
    day_idx = np.arange(games_per_season)

    for p in range(n_players):
        mean_missed = risk_scores[p] * games_per_season
        if mean_missed <= 0:
            continue
        p_param = dispersion / (dispersion + mean_missed)
        games_missed = rng.negative_binomial(dispersion, p_param, size=n_seasons)
        games_missed = np.clip(games_missed, 0, games_per_season)

        max_start = games_per_season - games_missed
        start = (rng.random(n_seasons) * (max_start + 1)).astype(int)

        absent = (day_idx[None, :] >= start[:, None]) & (day_idx[None, :] < (start + games_missed)[:, None])
        available[:, :, p] = ~absent

    return available


# Categorías ilustrativas de tipo de lesión, derivadas solo de cuántos
# partidos seguidos falta un jugador -- no son un diagnóstico real (nba_api
# no expone la lesión concreta). Puntos de corte estimados, no clínicos.
DEFAULT_INJURY_TYPE_CATEGORIES: List[Dict[str, Any]] = [
    {"max_games": 3, "label": "Molestia menor (día a día)"},
    {"max_games": 10, "label": "Lesión leve-moderada"},
    {"max_games": 20, "label": "Lesión moderada"},
    {"max_games": None, "label": "Lesión significativa / baja prolongada"},  # None = sin tope
]


def categorize_injury_absence(
    games_missed: int, categories: List[Dict[str, Any]] = DEFAULT_INJURY_TYPE_CATEGORIES
) -> str:
    """Etiqueta ilustrativa (ver DEFAULT_INJURY_TYPE_CATEGORIES) para una
    racha de `games_missed` partidos seguidos."""
    for category in categories:
        if category["max_games"] is None or games_missed <= category["max_games"]:
            return category["label"]
    return categories[-1]["label"]


def _extract_absence_streaks(player_available: np.ndarray) -> List[Dict[str, int]]:
    """
    `player_available`: bool 1D (games_per_season,) de un jugador en una
    temporada. Devuelve la lista de rachas contiguas de ausencia
    (`start_game` 1-indexado, `length`).
    """
    streaks = []
    absent = ~player_available
    games_per_season = len(absent)
    i = 0
    while i < games_per_season:
        if absent[i]:
            start = i
            while i < games_per_season and absent[i]:
                i += 1
            streaks.append({"start_game": start + 1, "length": i - start})
        else:
            i += 1
    return streaks


def simulate_single_season_player_log(
    player_ids: list,
    player_names: Dict[int, str],
    risk_scores: np.ndarray,
    games_per_season: int,
    mc_config: Dict[str, float],
    random_seed: int,
) -> pd.DataFrame:
    """
    Simula una temporada concreta (no la distribución agregada de
    run_monte_carlo) y devuelve, por jugador, partidos jugados/perdidos y
    el detalle de cada racha de ausencia con su categoría ilustrativa.
    Reutiliza `sample_injury_absences` con n_seasons=1 para no duplicar
    el mecanismo de ausencias.
    """
    rng = np.random.default_rng(random_seed)
    available = sample_injury_absences(risk_scores, 1, games_per_season, rng, mc_config["injury_dispersion"])[0]

    rows = []
    for player_index, player_id in enumerate(player_ids):
        player_available = available[:, player_index]
        games_played = int(player_available.sum())
        streaks = _extract_absence_streaks(player_available)
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_names.get(player_id, str(player_id)),
                "games_played": games_played,
                "games_missed": games_per_season - games_played,
                "injury_events": [
                    {**streak, "category": categorize_injury_absence(streak["length"])} for streak in streaks
                ],
            }
        )
    return pd.DataFrame(rows)


def sample_schedule_context(
    league_win_pcts: np.ndarray,
    n_seasons: int,
    games_per_season: int,
    rng: np.random.Generator,
    b2b_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calendario sintético representativo -- ver docstring del módulo.
    Devuelve (opponent_win_pct, is_back_to_back, is_home), los tres
    (n_seasons, games_per_season).

    `is_home` se fija en exactamente mitad y mitad (41/41, como en la NBA
    real) y se baraja por temporada, en vez de sortearse con Bernoulli,
    para no introducir varianza artificial en el conteo de local/visita.
    """
    opponent_win_pct = rng.choice(league_win_pcts, size=(n_seasons, games_per_season))
    is_back_to_back = rng.random((n_seasons, games_per_season)) < b2b_probability
    is_back_to_back[:, 0] = False  # el primer partido de temporada nunca es back-to-back

    is_home = np.zeros((n_seasons, games_per_season), dtype=bool)
    is_home[:, : games_per_season // 2] = True
    is_home = rng.permuted(is_home, axis=1)
    return opponent_win_pct, is_back_to_back, is_home


def compute_player_contributions(
    game_score_per36: np.ndarray,
    minutes_projection: np.ndarray,
    fatigue_scores: np.ndarray,
    available: np.ndarray,
    is_back_to_back: np.ndarray,
    rng: np.random.Generator,
    mc_config: Dict[str, float],
) -> np.ndarray:
    """
    Contribución de Game Score de cada jugador a cada partido de cada
    temporada simulada: (n_seasons, games_per_season, n_players).
    """
    n_seasons, games_per_season, n_players = available.shape
    effective_game_score_per36 = apply_star_bonus(game_score_per36, minutes_projection, mc_config)
    base_contribution = effective_game_score_per36[None, None, :] * (minutes_projection[None, None, :] / 36.0)

    season_progress = np.arange(games_per_season) / games_per_season
    fatigue_decay = 1 - (
        fatigue_scores[None, None, :] * mc_config["season_fatigue_decay"] * season_progress[None, :, None]
    )
    b2b_penalty = 1 - (
        fatigue_scores[None, None, :] * mc_config["b2b_fatigue_penalty"] * is_back_to_back[:, :, None]
    )
    noise = rng.normal(0, mc_config["game_variance_std"], size=(n_seasons, games_per_season, n_players))

    contribution = base_contribution * fatigue_decay * b2b_penalty + noise
    return np.where(available, contribution, 0.0)


def compute_game_net_rating_estimate(
    player_contributions: np.ndarray,
    opponent_win_pct: np.ndarray,
    mc_config: Dict[str, float],
    is_home: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Game Score de equipo (suma de jugadores) menos la línea base de un
    equipo "promedio" (league_average_game_score_per36), menos el ajuste
    por fuerza de rival, más la ventaja de campo. (n_seasons,
    games_per_season).

    `is_home`: bool array del mismo shape; suma `home_court_advantage` en
    casa y lo resta fuera. Si es None no se aplica.
    """
    team_game_score = player_contributions.sum(axis=2)
    league_average_team_game_score = (
        mc_config["league_average_game_score_per36"] * TOTAL_TEAM_MINUTES_PER_GAME / 36.0
    )
    opponent_adjustment = (opponent_win_pct - 0.5) * mc_config["opponent_strength_scale"]
    net_rating = (
        team_game_score - league_average_team_game_score
    ) * mc_config["game_score_to_net_rating_scale"] - opponent_adjustment

    if is_home is not None:
        hca = mc_config.get("home_court_advantage", 0.0)
        net_rating = net_rating + np.where(is_home, hca, -hca)
    return net_rating


def compute_win_probabilities(net_rating_estimate: np.ndarray, outcome_variance_scale: float) -> np.ndarray:
    """Probabilidad de victoria vía función logística sobre el diferencial estimado."""
    return 1 / (1 + np.exp(-net_rating_estimate / outcome_variance_scale))


def sample_team_quality_noise(n_seasons: int, std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Un valor por temporada simulada (no por partido): incertidumbre de
    "¿el equipo es en realidad algo mejor o peor que la proyección de
    talento?" (química, salud, entrenador -- nada que un box score
    capture). Shape (n_seasons, 1), mismo valor en los 82 partidos de esa
    temporada, a diferencia del ruido partido-a-partido de
    `compute_player_contributions`.

    Con std=0 (default) es identidad. Con std>0 ensancha la banda P10-P90
    alrededor de la proyección de cada equipo, pero no mueve `wins_mean`
    (ruido de media cero que se cancela al promediar). No sirve para
    separar más las victorias medias entre equipos distintos -- eso está
    limitado por la señal de talento, no por esta incertidumbre.
    """
    if std <= 0:
        return np.zeros((n_seasons, 1))
    return rng.normal(0.0, std, size=(n_seasons, 1))


def run_monte_carlo(
    player_ids: list,
    game_score_per36: np.ndarray,
    minutes_projection: np.ndarray,
    risk_scores: np.ndarray,
    fatigue_scores: np.ndarray,
    league_win_pcts: np.ndarray,
    n_seasons: int,
    games_per_season: int,
    mc_config: Dict[str, float],
    random_seed: int,
    synergy_matrix: Optional[np.ndarray] = None,
    fixed_schedule: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Orquesta una simulación Monte Carlo completa y devuelve un DataFrame
    con una fila por temporada simulada: wins, losses,
    net_rating_estimate_mean, total_games_missed. `synergy_matrix`
    (lineup_synergy.py) es opcional -- sin ella se suman contribuciones
    individuales sin modelar encaje de alineación.

    `fixed_schedule`: opcional, (opponent_win_pct_1d, is_back_to_back_1d)
    o con is_home_1d como tercer elemento, shape (games_in_season,). Lo
    usa backtesting.py para reproducir el calendario real de una
    temporada ya jugada en vez de muestrear uno sintético; si se pasa,
    `games_per_season` se ignora y se deriva de su longitud.
    """
    rng = np.random.default_rng(random_seed)

    if fixed_schedule is not None:
        fixed_opponent_win_pct, fixed_is_back_to_back, *rest = fixed_schedule
        games_per_season = len(fixed_opponent_win_pct)
        opponent_win_pct = np.tile(fixed_opponent_win_pct, (n_seasons, 1))
        is_back_to_back = np.tile(fixed_is_back_to_back, (n_seasons, 1))
        is_home = np.tile(rest[0], (n_seasons, 1)) if rest else None
    else:
        opponent_win_pct, is_back_to_back, is_home = sample_schedule_context(
            league_win_pcts, n_seasons, games_per_season, rng, mc_config["b2b_probability"]
        )

    available = sample_injury_absences(
        risk_scores, n_seasons, games_per_season, rng, mc_config["injury_dispersion"]
    )
    contributions = compute_player_contributions(
        game_score_per36, minutes_projection, fatigue_scores, available, is_back_to_back, rng, mc_config
    )
    net_rating_estimate = compute_game_net_rating_estimate(
        contributions, opponent_win_pct, mc_config, is_home=is_home
    )

    if synergy_matrix is not None:
        synergy_adjustment = compute_game_synergy_adjustment(available, synergy_matrix)
        net_rating_estimate = net_rating_estimate + synergy_adjustment

    team_quality_noise = sample_team_quality_noise(
        n_seasons, mc_config.get("team_quality_uncertainty_std", 0.0), rng
    )
    net_rating_estimate = net_rating_estimate + team_quality_noise

    win_probability = compute_win_probabilities(net_rating_estimate, mc_config["outcome_variance_scale"])

    outcomes = rng.random((n_seasons, games_per_season)) < win_probability

    wins = outcomes.sum(axis=1)
    games_missed_per_season = (~available).sum(axis=(1, 2))

    return pd.DataFrame(
        {
            "season_index": np.arange(n_seasons),
            "wins": wins,
            "losses": games_per_season - wins,
            "net_rating_estimate_mean": net_rating_estimate.mean(axis=1),
            "total_games_missed": games_missed_per_season,
        }
    )


def compute_simulation_results(
    config: Dict[str, Any], risk_scores_override: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Toda la lógica de `build_simulation_dataset` salvo el guardado en
    disco -- extraída para que el frontend web pueda pedir una variante
    "en vivo" (p.ej. `risk_scores_override=zeros` para un modo "sin
    lesiones") sin escribir sobre `simulation_results.csv`.
    `risk_scores_override`, si se pasa, sustituye los `risk_score` de
    `injury_risk.csv` (mismo orden que `player_ids`); todo lo demás se
    calcula igual que la corrida real.
    """
    paths = get_paths(config)
    required = {
        "aging_curve_projection.csv": "context.aging_curve.build_aging_projection_dataset",
        "injury_risk.csv": "context.injury_model.build_injury_risk_dataset",
        "fatigue_risk.csv": "context.fatigue_accumulation.build_fatigue_dataset",
        "prior_season_standings.csv": "data_pipeline.build_prior_season_standings_dataset",
    }
    for filename, builder in required.items():
        path = paths["processed"] / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}. Corre `{builder}` primero.")

    aging = pd.read_csv(paths["processed"] / "aging_curve_projection.csv").set_index("player_id")
    injury = pd.read_csv(paths["processed"] / "injury_risk.csv").set_index("player_id")
    fatigue = pd.read_csv(paths["processed"] / "fatigue_risk.csv").set_index("player_id")
    standings = pd.read_csv(paths["processed"] / "prior_season_standings.csv")

    player_ids = [p["player_id"] for p in config["roster"] if p.get("player_id")]
    missing = [pid for pid in player_ids if pid not in aging.index]
    if missing:
        raise ValueError(
            f"player_id(s) {missing} están en el roster pero no en aging_curve_projection.csv. "
            "Corre el pipeline de proyección para todos los jugadores del roster."
        )

    game_score_per36 = aging.loc[player_ids, "game_score_per36"].to_numpy()
    minutes_projection_by_player = {
        p["player_id"]: p.get("minutes_projection", 0) for p in config["roster"] if p.get("player_id")
    }
    minutes_projection = np.array([minutes_projection_by_player[pid] for pid in player_ids])
    risk_scores = (
        risk_scores_override if risk_scores_override is not None else injury.loc[player_ids, "risk_score"].to_numpy()
    )
    fatigue_scores = fatigue.loc[player_ids, "fatigue_score"].to_numpy()
    league_win_pcts = standings["WinPCT"].to_numpy()

    profiles = {pid: compute_style_profile(aging.loc[pid]) for pid in player_ids}
    syn_cfg = config.get("lineup_synergy", {})
    synergy_matrix = build_synergy_matrix(
        player_ids,
        profiles,
        minutes_projection_by_player,
        usage_threshold=syn_cfg.get("usage_threshold", 18.0),
        usage_clash_weight=syn_cfg.get("usage_clash_weight", 0.05),
        playmaking_spacing_weight=syn_cfg.get("playmaking_spacing_weight", 0.02),
    )

    mc_cfg = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}

    # Recalibra la línea base de "equipo promedio" desde los 30 equipos
    # reales si están disponibles, salvo que el usuario haya fijado su
    # propio valor a mano en config["monte_carlo"].
    league_projections_path = paths["processed"] / "league_player_projections.csv"
    if "league_average_game_score_per36" not in config.get("monte_carlo", {}) and league_projections_path.exists():
        league_projections = pd.read_csv(league_projections_path)
        mc_cfg["league_average_game_score_per36"] = compute_league_average_game_score_per36(
            league_projections,
            league_mean_synergy_net_rating=load_league_mean_synergy(paths["processed"]),
            game_score_to_net_rating_scale=mc_cfg["game_score_to_net_rating_scale"],
        )

    n_seasons = config["simulation"]["n_seasons"]
    games_per_season = config["simulation"]["games_per_season"]
    random_seed = config["simulation"]["random_seed"]

    return run_monte_carlo(
        player_ids,
        game_score_per36,
        minutes_projection,
        risk_scores,
        fatigue_scores,
        league_win_pcts,
        n_seasons,
        games_per_season,
        mc_cfg,
        random_seed,
        synergy_matrix=synergy_matrix,
    )


def build_simulation_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Corre `compute_simulation_results` con los `risk_score` reales de
    `injury_risk.csv` y guarda `data/processed/simulation_results.csv`
    (una fila por temporada simulada) -- el resultado "oficial" de la
    configuración actual, el que lee el resto de la app.
    """
    results = compute_simulation_results(config)

    paths = get_paths(config)
    out_path = paths["processed"] / "simulation_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(results)} temporadas simuladas)")
    print(
        f"Wins: media={results['wins'].mean():.1f}, "
        f"p10={results['wins'].quantile(0.1):.0f}, "
        f"mediana={results['wins'].median():.0f}, "
        f"p90={results['wins'].quantile(0.9):.0f}"
    )
    return results


if __name__ == "__main__":
    from config_loader import load_config

    build_simulation_dataset(load_config())
