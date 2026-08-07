"""
aging_curve.py

Modelo de proyección individual (fuera de src/context/ -- este módulo no
es parte de la capa de contexto de temporada, es un item separado en el
roadmap del proyecto). Proyecta la producción por-36-minutos de cada
jugador del roster para la temporada del config, combinando su propio
historial reciente (línea base) con un ajuste por curva de edad
(informado por literatura, no inventado).

LITERATURA USADA PARA LA FORMA DE LA CURVA
---------------------------------------------
No hay suficientes datos propios en este proyecto (solo el roster de 10
jugadores) para AJUSTAR una curva de edad poblacional con rigor -- eso
requeriría carreras completas de cientos de jugadores históricos, fuera
del alcance actual de `data_pipeline.py`. En su lugar, la UBICACIÓN de
los puntos de quiebre de la curva viene de investigación pública:

- El rendimiento medio de un jugador NBA sube más rápido entre 19-20 años
  y de nuevo entre 23-25; baja más rápido entre 29-31 y de nuevo entre
  36-38; el pico general está ~26-27 años.
- Los tiros de 2 puntos y tiros libres alcanzan su pico ~25 años seguido
  de declive marcado; el tiro de 3 puntos alcanza su pico más tarde, ~30
  años -- consistente con que el tiro es más una habilidad que se
  mantiene con la edad, mientras que las estadísticas ligadas a
  atletismo (finalización cerca del aro, rebotes, robos, bloqueos) se
  degradan antes.
  (Ver "Large data and Bayesian modeling -- aging curves of NBA players",
  PubMed 30684225; y "Effects of age on physical and technical
  performance in NBA players".)

Las MAGNITUDES exactas (% de cambio anual en cada tramo) sí son una
estimación de este proyecto, calibrada para que la forma sea consistente
con los puntos de quiebre citados arriba -- no vienen literalmente de un
paper. Están expuestas como parámetros configurables en
`config["aging_curve"]`, no hardcodeadas, precisamente porque son una
estimación y no un hecho medido.

DOS CURVAS, NO UNA
--------------------
Se usa una curva "general" (pico ~26-27) para las estadísticas ligadas a
atletismo -- puntos totales, rebotes, asistencias, robos, bloqueos,
pérdidas, tiros de 2/tiro libre -- y una curva "shooting" (pico ~30) SOLO
para volumen y estadísticas de triples, reflejando el hallazgo de que el
tiro exterior envejece mejor que el resto del juego.

DISEÑO
------
1. `compute_per36_stats()` -- normaliza cada temporada a producción por
   36 minutos (estándar en análisis de básquet, permite comparar
   temporadas con distinta carga de minutos).
2. `compute_recency_weighted_baseline()` -- línea base = media ponderada
   por recencia (decaimiento exponencial, mismo patrón que
   injury_model.py/fatigue_accumulation.py) de las últimas N temporadas
   por-36 del jugador. Esto es "su nivel de talento reciente", antes de
   ajustar por edad.
3. `compute_age_adjustment_factor()` -- factor multiplicativo de la
   línea base a la temporada proyectada, según la curva de edad
   correspondiente (general o shooting) y la transición de edad real del
   jugador (puede ser más de un año si faltan temporadas en los datos).
4. `project_player_season()` -- combina baseline * factor de edad, y
   escala a totales de temporada usando `minutes_projection` (ya
   definido por jugador en team_config.yaml) y
   `simulation.games_per_season` -- reutiliza config existente en vez de
   inventar campos nuevos.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import get_paths  # noqa: E402
from season_utils import dedupe_traded_seasons, season_start_year  # noqa: E402

# Estadísticas ligadas a atletismo/skill general -- curva "general", pico ~26-27.
# Incluye OREB/DREB/FGM/FGA/FTM/FTA/PF (además de PTS/AST/REB/STL/BLK/TOV)
# porque el Game Score de Hollinger (ver compute_game_score_per36) las
# necesita por separado -- roster_career_stats.csv ya las trae, no hace
# falta pedir datos nuevos.
GENERAL_STATS = [
    "PTS", "AST", "REB", "STL", "BLK", "TOV",
    "OREB", "DREB", "FGM", "FGA", "FTM", "FTA", "PF",
]
# Volumen de triples -- curva "shooting", pico ~30 (el tiro envejece mejor).
SHOOTING_STATS = ["FG3M", "FG3A"]

# Pesos del Game Score de Hollinger (fórmula pública estándar, no inventada
# para este proyecto): PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB +
# 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV. Usado por
# src/simulation.py para convertir la proyección por-36 de cada jugador en
# una única "nota de impacto" que sumar al diferencial de puntos del equipo.
def compute_game_score_per36(per36: Dict[str, float]) -> float:
    """Game Score de Hollinger aplicado a valores por-36 minutos."""
    return (
        per36["PTS_per36"]
        + 0.4 * per36["FGM_per36"]
        - 0.7 * per36["FGA_per36"]
        - 0.4 * (per36["FTA_per36"] - per36["FTM_per36"])
        + 0.7 * per36["OREB_per36"]
        + 0.3 * per36["DREB_per36"]
        + per36["STL_per36"]
        + 0.7 * per36["AST_per36"]
        + 0.7 * per36["BLK_per36"]
        - 0.4 * per36["PF_per36"]
        - per36["TOV_per36"]
    )

# Puntos de quiebre citados en la literatura (ver docstring del módulo);
# las tasas de cambio anual son una estimación de este proyecto calibrada
# para esa forma, expuestas como config, no un hecho medido.
DEFAULT_GENERAL_AGE_CURVE: List[Dict[str, float]] = [
    {"up_to_age": 20, "annual_rate": 0.06},
    {"up_to_age": 22, "annual_rate": 0.03},
    {"up_to_age": 25, "annual_rate": 0.05},
    {"up_to_age": 27, "annual_rate": 0.0},
    {"up_to_age": 28, "annual_rate": -0.015},
    {"up_to_age": 31, "annual_rate": -0.04},
    {"up_to_age": 35, "annual_rate": -0.025},
    {"up_to_age": 38, "annual_rate": -0.06},
    {"up_to_age": 999, "annual_rate": -0.08},
]
DEFAULT_SHOOTING_AGE_CURVE: List[Dict[str, float]] = [
    {"up_to_age": 22, "annual_rate": 0.05},
    {"up_to_age": 26, "annual_rate": 0.03},
    {"up_to_age": 29, "annual_rate": 0.01},
    {"up_to_age": 31, "annual_rate": 0.0},
    {"up_to_age": 34, "annual_rate": -0.02},
    {"up_to_age": 37, "annual_rate": -0.04},
    {"up_to_age": 999, "annual_rate": -0.06},
]

DEFAULT_N_SEASONS_LOOKBACK = 3
DEFAULT_RECENCY_HALF_LIFE_SEASONS = 1.5

# Minutos mínimos para que un jugador cuente en la línea base de liga
# (ver compute_league_game_score_baseline). 500 minutos ~ deja ~350
# jugadores por temporada, que es aproximadamente la población de
# rotación real de la NBA (30 equipos x ~11 jugadores); incluir a todo
# el que pisó la cancha metería ruido de muestras de 20 minutos.
DEFAULT_LEAGUE_BASELINE_MIN_MINUTES = 500


def compute_league_game_score_baseline(
    career_stats: pd.DataFrame, min_minutes: int = DEFAULT_LEAGUE_BASELINE_MIN_MINUTES
) -> pd.DataFrame:
    """
    Game Score/36 medio de la liga POR TEMPORADA, ponderado por minutos.
    Devuelve un DataFrame con columnas (season, league_game_score_per36,
    n_players).

    POR QUÉ EXISTE (ver "BUG REAL: INFLACIÓN DE ERA" en el docstring de
    simulation.py): el nivel de Game Score de la NBA NO es estable en el
    tiempo -- subió de ~10.7 por-36 en 2010-11 a ~13.4 en 2024-25 (más
    ritmo de juego y revolución del triple). Comparar a un equipo de
    2024-25 contra una línea base fija calibrada para "un jugador
    promedio de Hollinger" (10.0) le regala ~22 puntos de Game Score de
    ventaja que no son mérito suyo, sino de la época en la que juega.
    Cada equipo debe compararse contra la media de SU PROPIA temporada.

    `career_stats` debe tener el esquema de roster_career_stats.csv
    (PLAYER_ID, SEASON_ID, MIN + las columnas de caja). Para que la media
    sea representativa de la liga hace falta una muestra amplia de
    jugadores (el sweep de 30 equipos x N temporadas la da; el roster de
    un solo equipo NO -- ver la advertencia en
    backtesting.build_league_baseline_dataset).
    """
    rows = []
    for season, group in career_stats[career_stats["MIN"] >= min_minutes].groupby("SEASON_ID"):
        per36 = compute_per36_stats(group)
        if per36.empty:
            continue
        game_scores = per36.apply(
            lambda r: compute_game_score_per36(
                {f"{stat}_per36": r[f"{stat}_per36"] for stat in GENERAL_STATS + SHOOTING_STATS}
            ),
            axis=1,
        )
        weights = per36["MIN"]
        rows.append(
            {
                "season": season,
                "league_game_score_per36": float((game_scores * weights).sum() / weights.sum()),
                "n_players": int(len(per36)),
            }
        )
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def compute_per36_stats(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas <stat>_per36 para cada estadística en GENERAL_STATS +
    SHOOTING_STATS, dividiendo por MIN y escalando a 36 minutos."""
    df = dedupe_traded_seasons(player_seasons).copy()
    minutes = df["MIN"].replace(0, pd.NA)
    for stat in GENERAL_STATS + SHOOTING_STATS:
        df[f"{stat}_per36"] = (df[stat] / minutes * 36).fillna(0.0)
    return df


def _most_recent_n_seasons(df_per36: pd.DataFrame, n_seasons: int) -> pd.DataFrame:
    df = df_per36.assign(_start_year=df_per36["SEASON_ID"].apply(season_start_year))
    df = df.sort_values("_start_year", ascending=False)
    return df.head(n_seasons).reset_index(drop=True)


def compute_recency_weighted_baseline(
    player_seasons: pd.DataFrame,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    half_life_seasons: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
) -> Dict[str, float]:
    """
    Línea base por-36 de un jugador: media ponderada por recencia
    (decaimiento exponencial, la temporada más reciente pesa más) de las
    últimas n_seasons. Devuelve un dict {stat_per36: valor} para cada
    estadística en GENERAL_STATS + SHOOTING_STATS.
    """
    df = compute_per36_stats(player_seasons)
    recent = _most_recent_n_seasons(df, n_seasons)
    if recent.empty:
        return {f"{stat}_per36": 0.0 for stat in GENERAL_STATS + SHOOTING_STATS}

    seasons_ago = recent.index.to_numpy()  # 0 = más reciente
    weights = 0.5 ** (seasons_ago / half_life_seasons)
    total_weight = weights.sum()

    baseline = {}
    for stat in GENERAL_STATS + SHOOTING_STATS:
        col = f"{stat}_per36"
        baseline[col] = float((weights * recent[col].to_numpy()).sum() / total_weight)
    return baseline


def _annual_rate_for_age(age: float, curve: List[Dict[str, float]]) -> float:
    for breakpoint in curve:
        if age <= breakpoint["up_to_age"]:
            return breakpoint["annual_rate"]
    return curve[-1]["annual_rate"]


def compute_age_adjustment_factor(from_age: float, to_age: float, curve: List[Dict[str, float]]) -> float:
    """
    Factor multiplicativo acumulado de from_age a to_age, aplicando la
    tasa anual correspondiente a cada año de la transición (permite
    saltos de más de un año si faltan temporadas en los datos del
    jugador). Si to_age <= from_age, devuelve 1.0 (sin ajuste).
    """
    factor = 1.0
    age = from_age
    while age < to_age:
        factor *= 1 + _annual_rate_for_age(age, curve)
        age += 1
    return factor


def project_player_season(
    player_seasons: pd.DataFrame,
    target_age: float,
    minutes_per_game: float,
    games_per_season: int,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    half_life_seasons: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
    general_curve: Optional[List[Dict[str, float]]] = None,
    shooting_curve: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Proyecta la temporada de UN jugador: línea base por-36 ajustada por
    edad, escalada a totales usando minutes_per_game * games_per_season.
    """
    general_curve = general_curve or DEFAULT_GENERAL_AGE_CURVE
    shooting_curve = shooting_curve or DEFAULT_SHOOTING_AGE_CURVE

    baseline = compute_recency_weighted_baseline(player_seasons, n_seasons, half_life_seasons)
    most_recent_row = _most_recent_n_seasons(compute_per36_stats(player_seasons), 1).iloc[0]
    current_age = float(most_recent_row["PLAYER_AGE"])
    # GP de la temporada más reciente registrada -- informativo (cuántos
    # partidos jugó realmente esa temporada), NO una proyección de
    # partidos que jugará en la temporada simulada (eso requeriría un
    # modelo de supervivencia que este proyecto no implementa).
    games_played_last_season = int(most_recent_row["GP"])

    projected_total_minutes = minutes_per_game * games_per_season

    result: Dict[str, float] = {
        "current_age": current_age,
        "target_age": target_age,
        "games_played_last_season": games_played_last_season,
        "minutes_per_game_last_season": float(most_recent_row["MIN"]) / games_played_last_season
        if games_played_last_season > 0 else 0.0,
        "projected_total_minutes": projected_total_minutes,
    }
    for stat in GENERAL_STATS:
        col = f"{stat}_per36"
        factor = compute_age_adjustment_factor(current_age, target_age, general_curve)
        projected_per36 = baseline[col] * factor
        result[f"{stat}_per36_projected"] = projected_per36
        result[f"{stat}_projected"] = projected_per36 / 36 * projected_total_minutes
    for stat in SHOOTING_STATS:
        col = f"{stat}_per36"
        factor = compute_age_adjustment_factor(current_age, target_age, shooting_curve)
        projected_per36 = baseline[col] * factor
        result[f"{stat}_per36_projected"] = projected_per36
        result[f"{stat}_projected"] = projected_per36 / 36 * projected_total_minutes

    per36_projected = {k: v for k, v in result.items() if k.endswith("_per36_projected")}
    per36_projected = {k.replace("_projected", ""): v for k, v in per36_projected.items()}
    result["game_score_per36"] = compute_game_score_per36(per36_projected)
    # Copia del Game Score de CAJA puro. build_aging_projection_dataset()
    # sobrescribe `game_score_per36` con la métrica compuesta cuando hay
    # estadísticas avanzadas (src/advanced_impact.py); esta columna
    # conserva el valor sin ajustar para poder comparar las dos.
    result["game_score_per36_box"] = result["game_score_per36"]

    return result


def zero_player_projection(target_age: Optional[float], minutes_per_game: float, games_per_season: int) -> Dict[str, float]:
    """
    Proyección "piso" (todo cero) para un jugador del roster SIN ninguna
    temporada previa registrada -- un rookie de verdad que aún no jugó un
    partido de liga regular. Mismo principio que
    backtesting.project_historical_player() para el caso análogo en el
    backtest: "sin datos para proyectar rendimiento/riesgo -- se asume el
    piso (0), no se inventa un valor 'de liga' sin evidencia". Devuelve
    las MISMAS claves que project_player_season() (con todos los
    <stat>_per36_projected/<stat>_projected en 0.0) para que el resto del
    pipeline (dashboard, simulation.py, league_simulation.py) no tenga
    que tratar a estos jugadores como un caso especial -- su fila
    simplemente no aporta nada al equipo, en vez de faltar por completo
    y hacer fallar aguas abajo por un player_id inesperado.
    """
    projected_total_minutes = minutes_per_game * games_per_season
    result: Dict[str, float] = {
        "current_age": target_age,  # sin temporada previa no hay edad real que leer; se usa la objetivo como mejor estimación
        "target_age": target_age,
        "games_played_last_season": 0,
        "minutes_per_game_last_season": 0.0,
        "projected_total_minutes": projected_total_minutes,
    }
    for stat in GENERAL_STATS + SHOOTING_STATS:
        result[f"{stat}_per36_projected"] = 0.0
        result[f"{stat}_projected"] = 0.0
    result["game_score_per36"] = 0.0
    result["game_score_per36_box"] = 0.0
    return result


def build_aging_projection_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Lee roster_career_stats.csv (generado por data_pipeline.py) y proyecta
    la temporada de cada jugador del roster para config["team"]["season"],
    usando su propio minutes_projection y simulation.games_per_season del
    config. Guarda data/processed/aging_curve_projection.csv.

    Si existe `league_advanced_player_stats.csv`, el `game_score_per36`
    resultante es la métrica COMPUESTA (Game Score + ajuste por
    NET_RATING, ver src/advanced_impact.py). La columna
    `game_score_per36_box` conserva el Game Score puro para poder
    comparar -- y porque el dashboard muestra las dos.
    """
    from advanced_impact import adjust_with_context, build_advanced_context, load_advanced_stats

    paths = get_paths(config)
    stats_path = paths["processed"] / "roster_career_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"No se encontró {stats_path}. Corre `python src/data_pipeline.py` "
            "primero para generar roster_career_stats.csv."
        )
    df = pd.read_csv(stats_path)

    aging_cfg = config.get("aging_curve", {})
    n_seasons = aging_cfg.get("n_seasons_lookback", DEFAULT_N_SEASONS_LOOKBACK)
    half_life = aging_cfg.get("recency_half_life_seasons", DEFAULT_RECENCY_HALF_LIFE_SEASONS)
    general_curve = aging_cfg.get("general_curve", DEFAULT_GENERAL_AGE_CURVE)
    shooting_curve = aging_cfg.get("shooting_curve", DEFAULT_SHOOTING_AGE_CURVE)

    games_per_season = config["simulation"]["games_per_season"]
    target_season_start_year = season_start_year(config["team"]["season"])

    roster_by_id = {p["player_id"]: p for p in config["roster"] if p.get("player_id")}
    advanced_context = build_advanced_context(load_advanced_stats(paths["processed"]), config)

    rows = []
    covered_player_ids = set()
    for player_id, group in df.groupby("PLAYER_ID"):
        player_cfg = roster_by_id.get(player_id)
        if player_cfg is None:
            continue
        covered_player_ids.add(player_id)

        current_age = float(
            _most_recent_n_seasons(compute_per36_stats(group), 1)["PLAYER_AGE"].iloc[0]
        )
        current_season_start_year = season_start_year(
            _most_recent_n_seasons(compute_per36_stats(group), 1)["SEASON_ID"].iloc[0]
        )
        target_age = current_age + (target_season_start_year - current_season_start_year)

        projection = project_player_season(
            group,
            target_age=target_age,
            minutes_per_game=player_cfg.get("minutes_projection", 0),
            games_per_season=games_per_season,
            n_seasons=n_seasons,
            half_life_seasons=half_life,
            general_curve=general_curve,
            shooting_curve=shooting_curve,
        )
        projection["game_score_per36"] = adjust_with_context(
            projection["game_score_per36"], player_id, config["team"]["season"], advanced_context
        )
        rows.append({"player_id": player_id, "player_name": player_cfg["name"], **projection})

    # Jugadores del roster sin NINGUNA temporada en roster_career_stats.csv
    # (rookies de verdad, recién incorporados sin partidos de liga regular
    # jugados todavía) -- ver zero_player_projection(). Sin este piso,
    # simulation.build_simulation_dataset() no podría simularlos en
    # absoluto (fallaría pidiendo correr "el pipeline de proyección" que
    # no tiene nada que proyectar para ellos).
    missing_player_ids = set(roster_by_id) - covered_player_ids
    for player_id in missing_player_ids:
        player_cfg = roster_by_id[player_id]
        print(f"  Aviso: {player_cfg['name']} (player_id={player_id}) no tiene ninguna "
              "temporada registrada -- se proyecta con piso 0 (rookie sin historial).")
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_cfg["name"],
                **zero_player_projection(
                    target_age=None,
                    minutes_per_game=player_cfg.get("minutes_projection", 0),
                    games_per_season=games_per_season,
                ),
            }
        )

    result_df = pd.DataFrame(rows)
    out_path = paths["processed"] / "aging_curve_projection.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} jugadores)")
    return result_df


if __name__ == "__main__":
    from config_loader import load_config

    build_aging_projection_dataset(load_config())
