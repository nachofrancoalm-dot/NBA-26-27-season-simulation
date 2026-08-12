"""
backtesting.py

Valida el motor de simulación (simulation.py + lineup_synergy.py +
aging_curve.py + injury_model.py + fatigue_accumulation.py) corriéndolo
retrospectivamente sobre los 4 `historical_comparables` reales, con su
roster y calendario reales -- no el roster/calendario hipotético de
team_config.yaml que usa `simulation.build_simulation_dataset`.

REGLA DE NO LOOK-AHEAD (la más importante de este módulo)
-------------------------------------------------------------
Para cada caso histórico, la proyección de cada jugador SOLO puede usar
sus temporadas ANTERIORES a la temporada del caso -- nunca la temporada
que se está prediciendo ni las posteriores. `filter_seasons_before()` es
la única puerta de entrada a los datos de carrera de un jugador en este
módulo.

Excepciones deliberadas (NO son look-ahead, son insumos externos igual
que en la simulación hacia delante):
- La EDAD del jugador esa temporada (de historical_comparables_rosters.csv,
  no de sus stats de carrera) y sus MINUTOS/PARTIDO reales esa temporada
  sí se usan. Un front office real conoce la edad de sus jugadores y
  decide minutos de antemano -- igual que `minutes_projection` en
  team_config.yaml para el roster hipotético. Lo que el modelo predice es
  RENDIMIENTO, RIESGO y DESGASTE, no la asignación de minutos.

VENTAJA METODOLÓGICA SOBRE LA SIMULACIÓN HACIA DELANTE
-----------------------------------------------------------
Como estas temporadas ya se jugaron, se construye el calendario REAL
(rival de cada partido, back-to-backs reales) en vez de muestrear uno
sintético -- ver `build_real_schedule_context()`.

MÉTRICA DE VALIDACIÓN
------------------------
Por cada caso: se corren `simulation.n_seasons` temporadas Monte Carlo
con el roster y calendario reales, se obtiene la distribución simulada de
victorias, y se calcula en qué percentil de esa distribución cae el
resultado REAL. Un percentil razonable (10-90) indica que el motor no
está lejos de la realidad; un percentil extremo (0 o 100) señala una
desconexión entre el modelo y lo que de verdad pasó.
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
    Proyección, risk_score y fatigue_score de UN jugador para el caso
    histórico `target_season`, usando solo temporadas anteriores (ver
    filter_seasons_before).

    `advanced_context` (de `advanced_impact.build_advanced_context`): si
    se pasa junto con `player_id`, el `game_score_per36` devuelto es la
    métrica COMPUESTA (Game Score + ajuste por NET_RATING, ver
    src/advanced_impact.py) en vez del Game Score puro. Opcional para que
    el backtest siga corriendo sin `league_advanced_player_stats.csv`.

    `n_seasons_lookback`/`recency_half_life_seasons`: BUG REAL -- esta
    función nunca pasaba estos parámetros a `project_player_season`, así
    que ignoraba `config["aging_curve"]` por completo y usaba siempre los
    defaults del módulo (mismo bug que en `league_simulation.py`, ver su
    docstring). `None` (default) preserva el comportamiento de siempre --
    `project_backtest_team()` es quien los rellena desde `config` cuando
    los llama.
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
    Game Score de equipo ESPERADO por partido de un equipo ya proyectado,
    en las mismas unidades que la línea base y contando todo lo que la
    simulación le va a hacer: ausencias por lesión, desgaste de temporada,
    penalización de back-to-back y ajuste de sinergia.

    POR QUÉ NO BASTA `projection["team_game_score"]` (BUG REAL, ver
    CLAUDE.md): ese valor es el del equipo a PLENA SALUD y sin sinergia.
    Usarlo como línea base rompía la restricción de suma cero por dos
    términos que NO se anulan entre sí y que ninguno estaba centrado:

      - Las ausencias por lesión bajan el Game Score efectivo del equipo
        promedio de ~88.7 a ~66.6 (medido en 2024-25), o sea unos -4.6
        puntos de diferencial que la línea base no descontaba.
      - `compute_game_synergy_adjustment` devuelve un valor SIEMPRE
        POSITIVO (medido: +4.4 a +11.9, media +9.7 en 2024-25). Se suma
        al net rating de todos los equipos por igual, así que desplaza a
        la liga entera hacia arriba.

    Los dos venían cancelándose por casualidad con la escala antigua
    (0.29): -6.4 + 9.7 = +3.3, que explica casi exactamente el error medio
    residual de -3.5 victorias que quedó documentado como "sin explicar".
    Al re-calibrar la escala a 0.21 el término negativo encogió y el sesgo
    saltó a +5.0. Dos errores grandes de signo opuesto se veían como un
    error pequeño.

    El ajuste de sinergia se divide por `game_score_to_net_rating_scale`
    para expresarlo en unidades de Game Score, que son las de la línea
    base -- la simulación lo suma DESPUÉS de aplicar la escala
    (`run_monte_carlo`), así que hay que deshacer esa conversión.

    Se estima por muestreo en vez de analíticamente porque
    `compute_game_synergy_adjustment` va capada en [min, max]: la
    esperanza de una forma cuadrática capada no tiene forma cerrada, y
    reutilizar las mismas funciones que la simulación garantiza que la
    línea base mide exactamente lo mismo que se le va a comparar.
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
    Línea base de "equipo promedio" POR TEMPORADA, definida como la MEDIA
    del Game Score de equipo PROYECTADO de todos los casos de esa
    temporada, expresada en las mismas unidades que
    `league_average_game_score_per36` (Game Score/36).

    POR QUÉ ESTA DEFINICIÓN Y NO LA MEDIA DE LOS JUGADORES DE LA LIGA:
    por la restricción de SUMA CERO. En una liga real la media de
    victorias es exactamente games/2, lo que en el modelo logístico
    significa que el equipo promedio debe tener net_rating = 0. Eso solo
    se cumple si la línea base es la media de lo que el modelo PROYECTA
    para los equipos, no una media calculada por otra vía. Usar
    `aging_curve.compute_league_game_score_baseline()` (media de los
    jugadores de la liga) deja un sesgo residual de ~+5 puntos de Game
    Score, porque el pipeline de proyección (media ponderada por recencia
    de las 3 últimas temporadas + ajuste de edad, sobre los 10 de la
    rotación) produce sistemáticamente más que la media de todo jugador
    con >=500 minutos.

    Solo tiene sentido cuando `cases` cubre la liga entera de cada
    temporada (el sweep de 30 equipos sí; los 4 comparables narrativos
    NO -- para esos se usa el valor genérico del config).

    `advanced_context` DEBE ser el mismo que se pase luego a
    `_run_backtest_cases`. Si la línea base se calculara con Game Score
    puro y los casos con la métrica compuesta, la restricción de suma cero
    se rompería en silencio: los equipos se compararían contra una
    referencia medida en otra escala.
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
    Corre el backtest completo (proyección + simulación) para UN caso histórico.

    `league_baseline_per36`: Game Score/36 medio de la liga EN LA
    TEMPORADA DEL CASO (ver aging_curve.compute_league_game_score_baseline).
    Si se pasa, sustituye a `league_average_game_score_per36` del config
    para este caso -- imprescindible para no premiar a los equipos
    modernos solo por jugar en una era de más anotación (ver "BUG REAL:
    INFLACIÓN DE ERA" en el docstring de simulation.py). Si es None se
    usa el valor del config, que es lo correcto solo si todos los casos
    son de la misma temporada.
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
    caso individual que falle (datos incompletos de una temporada/equipo
    concreto -- ej. 0 partidos registrados) se SALTA con un aviso en vez
    de abortar el sweep completo -- con 450 casos reales de 15 años de
    historia NBA, algún hueco de datos es más probable que con los 4
    casos elegidos a mano.

    `league_baseline_by_season`: {season: game_score_per36 medio de la
    liga esa temporada} -- ver run_backtest_case. Una temporada que no
    esté en el dict usa el valor genérico del config.
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
    Lee `league_game_score_baseline.csv` (lo genera
    build_backtest_sweep_dataset) y devuelve {season: baseline_per36},
    prefiriendo la columna `projected_team_baseline_per36` (media de los
    EQUIPOS proyectados, la que cumple la restricción de suma cero) sobre
    `league_game_score_per36` (media de los jugadores de la liga, que deja
    un sesgo residual de ~+5 de Game Score). {} si el archivo no existe --
    entonces el llamante cae al valor genérico del config.
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

    # Reutiliza las líneas base por temporada que dejó el sweep, si existen.
    # Los 4 comparables por su cuenta NO pueden calcularlas (son ~60
    # jugadores de 4 superequipos, no una muestra de liga), pero sí pueden
    # aprovechar las del sweep -- que cubren 2010-11..2024-25, o sea las 4.
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
    backtest (una fila por caso, columnas actual_wins/simulated_wins_mean
    /simulated_wins_p10/simulated_wins_p90/actual_percentile -- ver
    run_backtest_case). Si el motor de simulación predice bien:
    - `pct_within_p10_p90` debería rondar el 80% (por construcción: P10
      a P90 es el 80% central de la distribución simulada).
    - `mean_percentile`/`median_percentile` deberían rondar 50 (sin
      sesgo sistemático hacia sobreestimar o subestimar victorias).
    - `mean_error_wins` (actual - predicho) positivo indica que el
      modelo SUBESTIMA victorias en promedio; negativo, que las
      SOBREESTIMA (ver el patrón de "fricción de superequipo" en el
      README -- los 4 `historical_comparables` sesgan hacia negativo).
    - `correlation_actual_vs_predicted` cercano a 1 indica que el modelo
      SÍ distingue equipos buenos de malos (aunque esté mal calibrado en
      el nivel absoluto de victorias); cercano a 0, que no.
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
    Backtesting sistemático a gran escala: corre el backtest para TODOS
    los casos de `resolve_backtest_sweep_cases(config)` (30 equipos NBA x
    cada temporada en config["backtest_sweep"]["seasons"], 450 casos por
    defecto) y guarda data/processed/backtest_sweep_summary.csv (una fila
    por caso) y data/processed/backtest_sweep_calibration.csv (resumen
    agregado, ver compute_calibration_summary). Requiere haber corrido
    antes `python src/data_pipeline.py --backtest-sweep` (ingesta cara,
    horas la primera vez) -- a diferencia de build_backtest_dataset, NO
    se ejecuta como parte del pipeline normal.
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

    # Línea base de liga POR TEMPORADA (ver aging_curve.compute_league_game_score_baseline
    # y "BUG REAL: INFLACIÓN DE ERA" en simulation.py). El sweep cubre los
    # 30 equipos, así que su muestra de jugadores SÍ es representativa de
    # la liga de cada temporada -- por eso se calcula aquí y no en el
    # backtest de 4 casos narrativos, cuya muestra son solo ~60 jugadores
    # de 4 superequipos (nada parecido a una media de liga).
    # Primera pasada (barata, sin Monte Carlo): proyecta los 30 equipos de
    # cada temporada y toma la MEDIA como línea base -- así el equipo
    # promedio tiene net_rating 0 por construcción (restricción de suma
    # cero). Ver compute_projected_league_baselines().
    # Métrica de impacto compuesta (Game Score + NET_RATING, ver
    # src/advanced_impact.py). None si falta el CSV o está desactivada en
    # el config -- entonces todo el sweep corre con Game Score puro, que
    # es como corría antes de integrarla.
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
