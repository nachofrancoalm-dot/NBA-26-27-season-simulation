"""
backtesting.py

Valida el motor de simulación corriéndolo retrospectivamente sobre los 4
`historical_comparables` reales (y, en `build_backtest_sweep_dataset`,
sobre un sweep de 30 equipos x N temporadas), con su roster y calendario
reales -- no el roster/calendario hipotético de team_config.yaml.

Regla de no look-ahead (la más importante de este módulo): para cada
caso, la proyección de cada jugador solo puede usar sus temporadas
anteriores a la del caso -- nunca la que se está prediciendo ni las
posteriores. `filter_seasons_before()` es la única puerta de entrada a
los datos de carrera de un jugador en este módulo. Excepción deliberada:
la edad y los minutos/partido reales de esa temporada sí se usan (son
insumos externos, como `minutes_projection` en la simulación hacia
delante) -- lo que el modelo predice es rendimiento, riesgo y desgaste,
no la asignación de minutos.

Como estas temporadas ya se jugaron, se construye el calendario real
(rival de cada partido, back-to-backs reales) en vez de muestrear uno
sintético -- ver `build_real_schedule_context()`.

Métrica de validación: por cada caso se corren `simulation.n_seasons`
temporadas Monte Carlo con el roster y calendario reales, y se calcula
en qué percentil de la distribución simulada cae el resultado real. Un
percentil extremo (0 o 100) señala una desconexión entre el modelo y lo
que de verdad pasó.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import get_paths  # noqa: E402
from season_utils import dedupe_traded_seasons, season_start_year  # noqa: E402
from aging_curve import (  # noqa: E402
    DEFAULT_GENERAL_AGE_CURVE,
    DEFAULT_N_SEASONS_LOOKBACK,
    DEFAULT_RECENCY_HALF_LIFE_SEASONS,
    DEFAULT_SHOOTING_AGE_CURVE,
    compute_league_game_score_baseline,
    project_player_season,
)
from lineup_synergy import (  # noqa: E402
    build_synergy_matrix,
    compute_game_synergy_adjustment,
    compute_style_profile,
)
from simulation import (  # noqa: E402
    DEFAULT_MONTE_CARLO_CONFIG,
    DEFAULT_ROTATION_SIZE,
    TOTAL_TEAM_MINUTES_PER_GAME,
    compute_player_contributions,
    normalize_rotation_minutes,
    run_monte_carlo,
    sample_injury_absences,
)
from context.injury_model import compute_risk_score  # noqa: E402
from context.fatigue_accumulation import compute_fatigue_score  # noqa: E402
from context.opponent_weighting import resolve_opponent_team_id  # noqa: E402
from advanced_impact import adjust_with_context, build_advanced_context, load_advanced_stats  # noqa: E402


def filter_seasons_before(player_seasons: pd.DataFrame, target_season_start_year: int) -> pd.DataFrame:
    """
    Único punto de entrada para filtrar las temporadas de carrera de un
    jugador a las ANTERIORES a target_season_start_year -- ver regla de
    no look-ahead en el docstring del módulo. Ninguna otra función de
    este archivo debe leer career stats sin pasar antes por aquí.
    """
    years = player_seasons["SEASON_ID"].apply(season_start_year)
    return player_seasons[years < target_season_start_year].reset_index(drop=True)


def get_actual_season_row(player_regular_seasons: pd.DataFrame, target_season: str):
    """
    Fila (deduplicada por trade) de las stats REALES de un jugador en
    target_season -- usada solo para leer minutos/partido reales (insumo
    externo, no una predicción), nunca para alimentar la proyección.
    """
    deduped = dedupe_traded_seasons(player_regular_seasons)
    match = deduped[deduped["SEASON_ID"] == target_season]
    return match.iloc[0] if not match.empty else None


def build_real_schedule_context(team_game_log: pd.DataFrame, standings: pd.DataFrame) -> tuple:
    """
    Calendario REAL de un equipo en una temporada ya jugada: opponent_win_pct
    (resuelto desde MATCHUP + standings de esa misma temporada, mismo
    mecanismo que opponent_weighting.py) y is_back_to_back (desde huecos
    reales entre GAME_DATE). Devuelve (opponent_win_pct, is_back_to_back),
    ambos 1D, ordenados cronológicamente.
    """
    df = team_game_log.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    win_pct_by_team = standings.set_index("TeamID")["WinPCT"]
    opponent_team_ids = df["MATCHUP"].apply(resolve_opponent_team_id)
    opponent_win_pct = opponent_team_ids.map(win_pct_by_team).fillna(win_pct_by_team.mean()).to_numpy()

    gap_days = df["GAME_DATE"].diff().dt.days
    is_back_to_back = (gap_days <= 1).fillna(False).to_numpy()

    return opponent_win_pct, is_back_to_back


def project_historical_player(
    player_regular_seasons: pd.DataFrame,
    player_playoff_seasons: pd.DataFrame,
    target_season: str,
    actual_age: float,
    actual_minutes_per_game: float,
    games_in_season: int,
    player_id: Optional[int] = None,
    advanced_context: Optional[Dict[str, Any]] = None,
    n_seasons_lookback: Optional[int] = None,
    recency_half_life_seasons: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Proyección, risk_score y fatigue_score de un jugador para el caso
    histórico `target_season`, usando solo temporadas anteriores (ver
    filter_seasons_before).

    `advanced_context`: si se pasa junto con `player_id`, el
    `game_score_per36` devuelto es la métrica compuesta en vez del Game
    Score puro; opcional para que el backtest corra sin
    `league_advanced_player_stats.csv`.

    `n_seasons_lookback`/`recency_half_life_seasons` deben pasarse
    explícitos -- sin ellos, `project_player_season` ignora
    `config["aging_curve"]` (mismo requisito que en
    `league_simulation.project_team_roster`). `project_backtest_team()`
    los rellena desde `config`.
    """
    target_year = season_start_year(target_season)
    prior_regular = filter_seasons_before(player_regular_seasons, target_year)
    prior_playoff = (
        filter_seasons_before(player_playoff_seasons, target_year)
        if player_playoff_seasons is not None and not player_playoff_seasons.empty
        else None
    )

    if prior_regular.empty:
        # Sin historial previo (rookie en esa temporada): sin datos para
        # proyectar rendimiento/riesgo -- se asume el piso (0), no se
        # inventa un valor "de liga" sin evidencia de a qué nivel jugará.
        return {
            "game_score_per36": 0.0,
            "risk_score": 0.0,
            "fatigue_score": 0.0,
            "style_profile": {"usage": 0.0, "playmaking": 0.0, "spacing": 0.0, "interior": 0.0},
        }

    projection = project_player_season(
        prior_regular,
        target_age=actual_age,
        minutes_per_game=actual_minutes_per_game,
        games_per_season=games_in_season,
        general_curve=DEFAULT_GENERAL_AGE_CURVE,
        shooting_curve=DEFAULT_SHOOTING_AGE_CURVE,
        n_seasons=n_seasons_lookback if n_seasons_lookback is not None else DEFAULT_N_SEASONS_LOOKBACK,
        half_life_seasons=(
            recency_half_life_seasons if recency_half_life_seasons is not None else DEFAULT_RECENCY_HALF_LIFE_SEASONS
        ),
    )
    risk = compute_risk_score(prior_regular)
    fatigue = compute_fatigue_score(prior_regular, prior_playoff)

    game_score_per36 = projection["game_score_per36"]
    if player_id is not None:
        game_score_per36 = adjust_with_context(
            game_score_per36, player_id, target_season, advanced_context
        )

    return {
        "game_score_per36": game_score_per36,
        "risk_score": risk["risk_score"],
        "fatigue_score": fatigue["fatigue_score"],
        "style_profile": compute_style_profile(projection),
    }


def project_backtest_team(
    case: Dict[str, Any],
    rosters: pd.DataFrame,
    player_regular_stats: pd.DataFrame,
    player_playoff_stats: pd.DataFrame,
    config: Dict[str, Any],
    games_in_season: int,
    advanced_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Proyecta el roster de UN caso histórico (sin simular todavía):
    game_score_per36, minutos normalizados, risk/fatigue y matriz de
    sinergia. Separado de run_backtest_case() para poder hacer una
    primera pasada barata sobre TODOS los casos de una temporada y
    calcular la media de la liga -- ver
    compute_projected_league_baselines() y la restricción de suma cero.
    """
    case_roster = rosters[(rosters["comparable_name"] == case["name"]) & (rosters["season"] == case["season"])]
    player_ids = case_roster["PLAYER_ID"].astype(int).tolist()

    # Minutos REALES de cada jugador esa temporada, normalizados a los 240
    # de un partido (ver simulation.normalize_rotation_minutes) -- sin esto
    # el roster completo suma 280-345 min/partido e infla la fuerza del
    # equipo un 18-43%.
    raw_minutes: Dict[int, float] = {}
    for player_id in player_ids:
        player_regular = player_regular_stats[player_regular_stats["PLAYER_ID"] == player_id]
        season_row = get_actual_season_row(player_regular, case["season"])
        raw_minutes[player_id] = (
            float(season_row["MIN"]) / float(season_row["GP"])
            if season_row is not None and float(season_row["GP"]) > 0
            else 0.0
        )
    rotation_size = config.get("league_simulation", {}).get("rotation_size", DEFAULT_ROTATION_SIZE)
    normalized_minutes = normalize_rotation_minutes(raw_minutes, rotation_size)

    # Ver el docstring de project_historical_player: sin esto, el
    # backtesting ignoraba config["aging_curve"] por completo.
    aging_cfg = config.get("aging_curve", {})
    aging_n_seasons = aging_cfg.get("n_seasons_lookback", DEFAULT_N_SEASONS_LOOKBACK)
    aging_half_life = aging_cfg.get("recency_half_life_seasons", DEFAULT_RECENCY_HALF_LIFE_SEASONS)

    game_score_per36, risk_scores, fatigue_scores, minutes_projection = [], [], [], []
    profiles: Dict[int, Dict[str, float]] = {}
    minutes_by_player: Dict[int, float] = {}

    for player_id in player_ids:
        roster_row = case_roster[case_roster["PLAYER_ID"] == player_id].iloc[0]
        actual_age = float(roster_row["AGE"])

        player_regular = player_regular_stats[player_regular_stats["PLAYER_ID"] == player_id]
        player_playoff = (
            player_playoff_stats[player_playoff_stats["PLAYER_ID"] == player_id]
            if not player_playoff_stats.empty
            else pd.DataFrame()
        )

        minutes_per_game = normalized_minutes[player_id]
        result = project_historical_player(
            player_regular,
            player_playoff,
            case["season"],
            actual_age,
            minutes_per_game,
            games_in_season,
            player_id=player_id,
            advanced_context=advanced_context,
            n_seasons_lookback=aging_n_seasons,
            recency_half_life_seasons=aging_half_life,
        )
        game_score_per36.append(result["game_score_per36"])
        risk_scores.append(result["risk_score"])
        fatigue_scores.append(result["fatigue_score"])
        minutes_projection.append(minutes_per_game)
        profiles[player_id] = result["style_profile"]
        minutes_by_player[player_id] = minutes_per_game

    syn_cfg = config.get("lineup_synergy", {})
    synergy_matrix = build_synergy_matrix(
        player_ids,
        profiles,
        minutes_by_player,
        usage_threshold=syn_cfg.get("usage_threshold", 18.0),
        usage_clash_weight=syn_cfg.get("usage_clash_weight", 0.05),
        playmaking_spacing_weight=syn_cfg.get("playmaking_spacing_weight", 0.02),
    )

    game_score_per36 = np.array(game_score_per36)
    minutes_projection = np.array(minutes_projection)
    return {
        "player_ids": player_ids,
        "game_score_per36": game_score_per36,
        "minutes_projection": minutes_projection,
        "risk_scores": np.array(risk_scores),
        "fatigue_scores": np.array(fatigue_scores),
        "synergy_matrix": synergy_matrix,
        # Game Score total del equipo por partido, a plena salud -- la
        # magnitud que se compara contra la línea base de liga.
        "team_game_score": float((game_score_per36 * minutes_projection / 36.0).sum()),
    }


# Temporadas simuladas para estimar la contribución esperada de un equipo
# en expected_team_game_score_equivalent(). 60 basta: es una MEDIA sobre
# 60 x 82 = ~5.000 partidos por equipo, y luego se promedia otra vez sobre
# los 30 equipos de la temporada. Subirlo solo encarece la primera pasada.
BASELINE_SAMPLING_SEASONS = 60


def expected_team_game_score_equivalent(
    projection: Dict[str, Any], config: Dict[str, Any], games_in_season: int
) -> float:
    """
    Game Score de equipo esperado por partido de un equipo ya proyectado,
    en las mismas unidades que la línea base y contando todo lo que la
    simulación le va a hacer: ausencias por lesión, desgaste de temporada,
    penalización de back-to-back y ajuste de sinergia.

    No basta `projection["team_game_score"]` (equipo a plena salud, sin
    sinergia): las ausencias por lesión bajan el Game Score efectivo del
    equipo promedio (~-4.6 puntos de diferencial no descontados) y
    `compute_game_synergy_adjustment` es siempre positivo (+4.4 a +11.9),
    desplazando a la liga entera hacia arriba. Ambos términos pueden
    cancelarse por casualidad con una escala de conversión concreta,
    ocultando el problema -- por eso esta función se centra
    explícitamente en vez de depender de esa cancelación fortuita.

    El ajuste de sinergia se divide por `game_score_to_net_rating_scale`
    para expresarlo en unidades de Game Score (la simulación lo suma
    después de aplicar la escala). Se estima por muestreo, no
    analíticamente, porque `compute_game_synergy_adjustment` va capada en
    [min, max] y no tiene forma cerrada.
    """
    mc_config = dict(DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {}))
    rng = np.random.default_rng(config.get("simulation", {}).get("random_seed", 42))

    available = sample_injury_absences(
        projection["risk_scores"],
        BASELINE_SAMPLING_SEASONS,
        games_in_season,
        rng,
        mc_config["injury_dispersion"],
    )
    is_back_to_back = rng.random((BASELINE_SAMPLING_SEASONS, games_in_season)) < mc_config["b2b_probability"]
    contributions = compute_player_contributions(
        projection["game_score_per36"],
        projection["minutes_projection"],
        projection["fatigue_scores"],
        available,
        is_back_to_back,
        rng,
        mc_config,
    )
    expected_game_score = float(contributions.sum(axis=2).mean())

    synergy = projection.get("synergy_matrix")
    if synergy is None:
        return expected_game_score
    expected_synergy = float(compute_game_synergy_adjustment(available, synergy).mean())
    return expected_game_score + expected_synergy / mc_config["game_score_to_net_rating_scale"]


def compute_projected_league_baselines(
    cases: list,
    rosters: pd.DataFrame,
    player_regular_stats: pd.DataFrame,
    player_playoff_stats: pd.DataFrame,
    config: Dict[str, Any],
    games_in_season: int = 82,
    advanced_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Línea base de "equipo promedio" por temporada: la media del Game
    Score de equipo proyectado de todos los casos de esa temporada,
    expresada en las mismas unidades que `league_average_game_score_per36`.

    Se usa esta definición (y no la media de los jugadores de la liga)
    por la restricción de suma cero: en una liga real la media de
    victorias es games/2, así que el equipo promedio debe tener
    net_rating = 0, y eso solo se cumple con la media de lo que el modelo
    proyecta. `aging_curve.compute_league_game_score_baseline()` deja un
    sesgo residual de ~+5 de Game Score.

    Solo tiene sentido cuando `cases` cubre la liga entera de cada
    temporada (el sweep de 30 equipos sí; los 4 comparables narrativos
    usan el valor genérico del config). `advanced_context` debe ser el
    mismo que se pase luego a `_run_backtest_cases`, o la restricción de
    suma cero se rompe en silencio.
    """
    by_season: Dict[str, list] = {}
    for case in cases:
        try:
            projection = project_backtest_team(
                case,
                rosters,
                player_regular_stats,
                player_playoff_stats,
                config,
                games_in_season,
                advanced_context=advanced_context,
            )
        except Exception:  # noqa: BLE001 -- un caso roto no debe abortar el cálculo de la línea base
            continue
        by_season.setdefault(case["season"], []).append(
            expected_team_game_score_equivalent(projection, config, games_in_season)
        )

    return {
        season: float(np.mean(totals)) / (TOTAL_TEAM_MINUTES_PER_GAME / 36.0)
        for season, totals in by_season.items()
        if totals
    }


def run_backtest_case(
    case: Dict[str, Any],
    rosters: pd.DataFrame,
    player_regular_stats: pd.DataFrame,
    player_playoff_stats: pd.DataFrame,
    standings: pd.DataFrame,
    game_log: pd.DataFrame,
    config: Dict[str, Any],
    league_baseline_per36: Optional[float] = None,
    advanced_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Corre el backtest completo (proyección + simulación) para un caso
    histórico.

    `league_baseline_per36`: Game Score/36 medio de la liga en la
    temporada del caso. Si se pasa, sustituye a
    `league_average_game_score_per36` del config -- imprescindible para
    no premiar a equipos modernos solo por jugar en una era de más
    anotación (ver docstring de simulation.py). Si es None se usa el
    valor del config, correcto solo si todos los casos son de la misma
    temporada.
    """
    case_roster = rosters[(rosters["comparable_name"] == case["name"]) & (rosters["season"] == case["season"])]
    player_ids = case_roster["PLAYER_ID"].astype(int).tolist()

    case_standings = standings[standings["season"] == case["season"]]
    case_game_log = game_log[
        (game_log["comparable_name"] == case["name"]) & (game_log["game_phase"] == "regular")
    ]

    opponent_win_pct, is_back_to_back = build_real_schedule_context(case_game_log, case_standings)
    games_in_season = len(case_game_log)
    actual_wins = int((case_game_log["WL"] == "W").sum())

    projection = project_backtest_team(
        case,
        rosters,
        player_regular_stats,
        player_playoff_stats,
        config,
        games_in_season,
        advanced_context=advanced_context,
    )
    player_ids = projection["player_ids"]
    game_score_per36 = projection["game_score_per36"]
    risk_scores = projection["risk_scores"]
    fatigue_scores = projection["fatigue_scores"]
    minutes_projection = projection["minutes_projection"]
    synergy_matrix = projection["synergy_matrix"]

    mc_cfg = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}
    if league_baseline_per36 is not None:
        mc_cfg["league_average_game_score_per36"] = league_baseline_per36
    n_seasons = config["simulation"]["n_seasons"]

    results = run_monte_carlo(
        player_ids,
        np.array(game_score_per36),
        np.array(minutes_projection),
        np.array(risk_scores),
        np.array(fatigue_scores),
        league_win_pcts=case_standings["WinPCT"].to_numpy(),
        n_seasons=n_seasons,
        games_per_season=games_in_season,
        mc_config=mc_cfg,
        random_seed=config["simulation"]["random_seed"],
        synergy_matrix=synergy_matrix,
        fixed_schedule=(opponent_win_pct, is_back_to_back),
    )

    simulated_wins = results["wins"]
    actual_percentile = float((simulated_wins <= actual_wins).mean() * 100)

    return {
        "comparable_name": case["name"],
        "season": case["season"],
        "games_in_season": games_in_season,
        "actual_wins": actual_wins,
        "simulated_wins_mean": float(simulated_wins.mean()),
        "simulated_wins_p10": float(simulated_wins.quantile(0.1)),
        "simulated_wins_p50": float(simulated_wins.quantile(0.5)),
        "simulated_wins_p90": float(simulated_wins.quantile(0.9)),
        "actual_percentile": actual_percentile,
    }


def _run_backtest_cases(
    cases: list,
    rosters: pd.DataFrame,
    player_regular_stats: pd.DataFrame,
    player_playoff_stats: pd.DataFrame,
    standings: pd.DataFrame,
    game_log: pd.DataFrame,
    config: Dict[str, Any],
    show_progress: bool = False,
    league_baseline_by_season: Optional[Dict[str, float]] = None,
    advanced_context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Núcleo compartido entre build_backtest_dataset (4 casos narrativos) y
    build_backtest_sweep_dataset (cientos de casos): corre
    run_backtest_case() para cada caso y arma el DataFrame resultado. Un
    caso individual que falle (datos incompletos) se salta con un aviso
    en vez de abortar el sweep completo.

    `league_baseline_by_season`: {season: game_score_per36 medio de la
    liga esa temporada}. Una temporada que no esté en el dict usa el
    valor genérico del config.
    """
    rows = []
    iterator = tqdm(cases, desc="Corriendo backtest") if show_progress else cases
    for case in iterator:
        baseline = (league_baseline_by_season or {}).get(case["season"])
        try:
            result = run_backtest_case(
                case, rosters, player_regular_stats, player_playoff_stats, standings, game_log, config,
                league_baseline_per36=baseline,
                advanced_context=advanced_context,
            )
        except Exception as e:  # noqa: BLE001 -- un caso con datos incompletos no debe abortar el resto del sweep
            print(f"  [omitido] {case['name']}: {e}")
            continue
        result["league_baseline_per36"] = baseline
        rows.append(result)
    return pd.DataFrame(rows)


def load_league_baselines(processed_dir: Path) -> Dict[str, float]:
    """
    Lee `league_game_score_baseline.csv` y devuelve {season:
    baseline_per36}, prefiriendo `projected_team_baseline_per36` (media
    de equipos proyectados, cumple la restricción de suma cero) sobre
    `league_game_score_per36` (media de jugadores, sesgo residual de ~+5).
    {} si el archivo no existe -- el llamante cae al valor genérico del config.
    """
    path = processed_dir / "league_game_score_baseline.csv"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    column = (
        "projected_team_baseline_per36"
        if "projected_team_baseline_per36" in df.columns
        else "league_game_score_per36"
    )
    valid = df.dropna(subset=[column])
    return dict(zip(valid["season"], valid[column]))


def build_backtest_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Lee los datasets de comparables históricos generados por
    data_pipeline.py, corre el backtest para cada uno de
    `historical_comparables` (4 casos narrativos, parte del pipeline
    normal) y guarda data/processed/backtest_summary.csv (una fila por
    caso). Para el backtesting sistemático a gran escala (30 equipos x
    15 temporadas), ver build_backtest_sweep_dataset.
    """
    paths = get_paths(config)
    required = {
        "historical_comparables_rosters.csv": "data_pipeline.build_historical_comparables_rosters_dataset",
        "historical_comparables_player_career_stats.csv": "data_pipeline.build_historical_comparables_player_stats_dataset",
        "historical_comparables_standings.csv": "data_pipeline.build_historical_comparables_standings_dataset",
        "historical_comparables_advanced_game_logs.csv": "data_pipeline.build_historical_comparables_advanced_dataset",
    }
    for filename, builder in required.items():
        path = paths["processed"] / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}. Corre `{builder}` primero.")

    rosters = pd.read_csv(paths["processed"] / "historical_comparables_rosters.csv")
    player_regular_stats = pd.read_csv(paths["processed"] / "historical_comparables_player_career_stats.csv")
    playoff_path = paths["processed"] / "historical_comparables_player_playoff_career_stats.csv"
    player_playoff_stats = (
        pd.read_csv(playoff_path) if playoff_path.exists() and playoff_path.stat().st_size > 0 else pd.DataFrame()
    )
    standings = pd.read_csv(paths["processed"] / "historical_comparables_standings.csv")
    game_log = pd.read_csv(paths["processed"] / "historical_comparables_advanced_game_logs.csv")

    # Reutiliza las líneas base por temporada que dejó el sweep, si
    # existen -- los 4 comparables por su cuenta no pueden calcularlas
    # (no son una muestra de liga).
    league_baseline_by_season = load_league_baselines(paths["processed"])

    result_df = _run_backtest_cases(
        config["historical_comparables"], rosters, player_regular_stats, player_playoff_stats, standings, game_log, config,
        league_baseline_by_season=league_baseline_by_season,
    )
    out_path = paths["processed"] / "backtest_summary.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} casos)")
    return result_df


def compute_calibration_summary(sweep_results: pd.DataFrame) -> Dict[str, float]:
    """
    Resumen agregado de calibración sobre un DataFrame de resultados de
    backtest (ver run_backtest_case). Si el motor predice bien:
    `pct_within_p10_p90` debería rondar 80%; `mean_percentile`/
    `median_percentile` deberían rondar 50 (sin sesgo sistemático);
    `mean_error_wins` (actual - predicho) positivo indica que el modelo
    subestima victorias, negativo que las sobreestima;
    `correlation_actual_vs_predicted` cercano a 1 indica que el modelo
    distingue equipos buenos de malos.
    """
    if sweep_results.empty:
        return {
            "n_cases": 0,
            "pct_within_p10_p90": float("nan"),
            "mean_percentile": float("nan"),
            "median_percentile": float("nan"),
            "mean_absolute_error_wins": float("nan"),
            "mean_error_wins": float("nan"),
            "correlation_actual_vs_predicted": float("nan"),
        }

    within_p10_p90 = (sweep_results["actual_wins"] >= sweep_results["simulated_wins_p10"]) & (
        sweep_results["actual_wins"] <= sweep_results["simulated_wins_p90"]
    )
    errors = sweep_results["actual_wins"] - sweep_results["simulated_wins_mean"]

    return {
        "n_cases": int(len(sweep_results)),
        "pct_within_p10_p90": float(within_p10_p90.mean() * 100),
        "mean_percentile": float(sweep_results["actual_percentile"].mean()),
        "median_percentile": float(sweep_results["actual_percentile"].median()),
        "mean_absolute_error_wins": float(errors.abs().mean()),
        "mean_error_wins": float(errors.mean()),
        "correlation_actual_vs_predicted": float(
            sweep_results["actual_wins"].corr(sweep_results["simulated_wins_mean"])
        ),
    }


def build_backtest_sweep_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Backtesting sistemático a gran escala: corre el backtest para todos
    los casos de `resolve_backtest_sweep_cases(config)` (30 equipos NBA x
    cada temporada en config["backtest_sweep"]["seasons"]) y guarda
    backtest_sweep_summary.csv (una fila por caso) y
    backtest_sweep_calibration.csv (resumen agregado, ver
    compute_calibration_summary). Requiere haber corrido antes `python
    src/data_pipeline.py --backtest-sweep` -- a diferencia de
    build_backtest_dataset, no se ejecuta como parte del pipeline normal.
    """
    from config_loader import resolve_backtest_sweep_cases

    paths = get_paths(config)
    cases = resolve_backtest_sweep_cases(config)
    if not cases:
        raise ValueError(
            "config['backtest_sweep'] no está definido en team_config.yaml -- "
            "añade un bloque 'backtest_sweep: {seasons: [...]}' primero."
        )

    required = {
        "backtest_sweep_rosters.csv": "data_pipeline.build_backtest_sweep_rosters_dataset",
        "backtest_sweep_player_career_stats.csv": "data_pipeline.build_backtest_sweep_player_stats_dataset",
        "backtest_sweep_standings.csv": "data_pipeline.build_backtest_sweep_standings_dataset",
        "backtest_sweep_advanced_game_logs.csv": "data_pipeline.build_backtest_sweep_advanced_game_logs_dataset",
    }
    for filename, builder in required.items():
        path = paths["processed"] / filename
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. Corre `python src/data_pipeline.py --backtest-sweep` "
                f"(o `{builder}` directamente) primero."
            )

    rosters = pd.read_csv(paths["processed"] / "backtest_sweep_rosters.csv")
    player_regular_stats = pd.read_csv(paths["processed"] / "backtest_sweep_player_career_stats.csv")
    playoff_path = paths["processed"] / "backtest_sweep_player_playoff_career_stats.csv"
    player_playoff_stats = (
        pd.read_csv(playoff_path) if playoff_path.exists() and playoff_path.stat().st_size > 0 else pd.DataFrame()
    )
    standings = pd.read_csv(paths["processed"] / "backtest_sweep_standings.csv")
    game_log = pd.read_csv(paths["processed"] / "backtest_sweep_advanced_game_logs.csv")

    # Línea base de liga por temporada: el sweep cubre los 30 equipos, así
    # que su muestra sí es representativa de la liga (a diferencia de los
    # 4 comparables narrativos, ~60 jugadores de 4 superequipos). Primera
    # pasada barata (sin Monte Carlo) que toma la media como línea base --
    # ver compute_projected_league_baselines().
    # advanced_context: None si falta el CSV o está desactivado, entonces
    # el sweep corre con Game Score puro.
    advanced_context = build_advanced_context(load_advanced_stats(paths["processed"]), config)
    print(
        "Métrica de impacto: "
        + ("compuesta (Game Score + NET_RATING)" if advanced_context else "Game Score puro")
    )

    print("Calculando línea base de liga por temporada (primera pasada)...")
    league_baseline_by_season = compute_projected_league_baselines(
        cases, rosters, player_regular_stats, player_playoff_stats, config,
        advanced_context=advanced_context,
    )
    # Referencia complementaria: la media de los JUGADORES de la liga,
    # que es la que muestra la inflación de era de forma más directa.
    baseline_df = compute_league_game_score_baseline(player_regular_stats)
    baseline_df["projected_team_baseline_per36"] = baseline_df["season"].map(league_baseline_by_season)
    baseline_path = paths["processed"] / "league_game_score_baseline.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f"Guardado: {baseline_path} ({len(baseline_df)} temporadas)")

    result_df = _run_backtest_cases(
        cases, rosters, player_regular_stats, player_playoff_stats, standings, game_log, config,
        show_progress=True, league_baseline_by_season=league_baseline_by_season,
        advanced_context=advanced_context,
    )
    out_path = paths["processed"] / "backtest_sweep_summary.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} de {len(cases)} casos)")

    calibration = compute_calibration_summary(result_df)
    calibration_path = paths["processed"] / "backtest_sweep_calibration.csv"
    pd.DataFrame([calibration]).to_csv(calibration_path, index=False)
    print(f"Guardado: {calibration_path}")
    print(calibration)

    return result_df


if __name__ == "__main__":
    from config_loader import load_config

    build_backtest_dataset(load_config())
