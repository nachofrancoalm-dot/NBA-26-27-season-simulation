"""
Métrica de impacto COMPUESTA: Game Score de caja + estadísticas avanzadas
(PIE, NET_RATING, PCT_PLUSMINUS).

El Game Score (`aging_curve.compute_game_score_per36`) es casi puramente
ofensivo, así que infravalora a defensores de bajo perfil en caja (los
Knicks campeones 2025-26 proyectaban 12º del Este usando solo Game
Score). Este módulo suma NET_RATING y PCT_PLUSMINUS como ajuste ADITIVO
en unidades de Game Score/36, centrado en la media de liga de esa
temporada:

    impacto/36 = game_score/36
                 + k_pie * (PIE - PIE_medio)
                 + k_net * (NET_RATING - NET_medio)
                 + k_pct * (PCT_PLUSMINUS - PCT_medio)

Centrar por temporada preserva la suma cero por construcción (el ajuste
medio sobre los 30 equipos es ~0, no mueve la línea base de equipo
promedio) y da ajuste de era gratis. Pesos derivados por regresión sobre
el backtest sweep (480 casos, leave-one-season-out): NET_RATING sube el R
fuera de muestra de 0.702 a 0.754; PCT_PLUSMINUS lo sube más (R² 0.512 →
0.528). PIE no aporta nada una vez NET_RATING está en el modelo (su
coeficiente sale con signo invertido), así que `pie_weight=0.0` por
defecto pero se conserva la palanca. Los pesos no se encogen: el R mejora
monótonamente hasta el peso completo, sin señal de sobreajuste.

CAVEAT: NET_RATING y PCT_PLUSMINUS reflejan el equipo real del jugador,
no su valor aislado. Predicen bien rosters REALES (lo que valida el
backtest) pero, en un roster HIPOTÉTICO -- el caso de uso central del
simulador --, heredan sin corregir el contexto de equipo de origen de
cada jugador. Usar `advanced_impact: {enabled: false}` para comparar
contra Game Score puro.

Cada métrica se promedia solo sobre sus propias filas no nulas
(PCT_PLUSMINUS es NaN antes de 2013-14 o sin cobertura de tracking);
nunca se inventa un 0.0 para una métrica sin dato.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from season_utils import season_start_year

# Pesos en "puntos de Game Score/36 por unidad de desviación de la
# métrica respecto a la media de su temporada" (ver docstring del
# módulo). pct_plusminus_weight es negativo a propósito: un defensor
# bueno tiene PCT_PLUSMINUS más negativo que la media.
DEFAULT_ADVANCED_IMPACT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "pie_weight": 0.0,
    "net_rating_weight": 0.42,
    "pct_plusminus_weight": -57.03,
    "n_seasons_lookback": 3,
    "recency_half_life_seasons": 1.5,
    # Por debajo de este umbral de minutos, NET_RATING/PIE son ruidosos
    # (un suplente con 40 minutos puede salir +30); se cae al Game Score
    # puro para ese jugador.
    "min_minutes_for_advanced": 500,
}

ADVANCED_METRICS = ("PIE", "NET_RATING", "PCT_PLUSMINUS")

# MIN * GP: `leaguedashplayerstats` con measure_type="Advanced" devuelve
# MIN por PARTIDO, no total (ver data_pipeline.fetch_league_advanced_player_stats).
TOTAL_MINUTES_COLUMN = "total_minutes"


def resolve_advanced_impact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mezcla `config["advanced_impact"]` sobre los valores por defecto."""
    resolved = dict(DEFAULT_ADVANCED_IMPACT_CONFIG)
    resolved.update(config.get("advanced_impact") or {})
    return resolved


def compute_league_advanced_baselines(
    advanced_stats: pd.DataFrame, min_minutes: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Media de liga de cada métrica avanzada POR TEMPORADA, ponderada por
    minutos (el punto de referencia contra el que se centra cada
    jugador). Devuelve {season: {"PIE": x, "NET_RATING": y}}.

    Cada métrica se promedia sobre sus propias filas no nulas, no sobre
    el grupo entero, para no sesgar la media hacia 0 con filas que no
    tienen ese dato (PCT_PLUSMINUS antes de 2013-14).
    """
    if advanced_stats.empty:
        return {}

    eligible = advanced_stats[advanced_stats[TOTAL_MINUTES_COLUMN] >= min_minutes]
    baselines: Dict[str, Dict[str, float]] = {}
    for season, group in eligible.groupby("season"):
        season_baseline: Dict[str, float] = {}
        for metric in ADVANCED_METRICS:
            if metric not in group.columns:
                continue
            metric_rows = group[group[metric].notna()]
            weights = metric_rows[TOTAL_MINUTES_COLUMN]
            if weights.sum() <= 0:
                continue
            season_baseline[metric] = float((metric_rows[metric] * weights).sum() / weights.sum())
        if season_baseline:
            baselines[str(season)] = season_baseline
    return baselines


def compute_recency_weighted_advanced(
    player_advanced: pd.DataFrame,
    target_season: str,
    n_seasons: int = 3,
    half_life: float = 1.5,
    min_minutes: int = 500,
) -> Optional[Dict[str, float]]:
    """
    Media ponderada por recencia de las métricas avanzadas de UN jugador
    sobre sus N temporadas más recientes ANTERIORES a `target_season`.

    NO LOOK-AHEAD: filtra estrictamente a temporadas previas. Devuelve
    None si el jugador no tiene ninguna temporada elegible (rookie, o
    solo por debajo de `min_minutes`) -- el llamante cae al Game Score
    puro en ese caso.

    Un jugador traspasado a mitad de temporada aparece con una fila por
    equipo; se colapsan a una sola ponderando por minutos
    (`dedupe_traded_seasons` de season_utils no sirve: ese CSV no trae
    fila 'TOT'). Cada métrica se promedia solo sobre las temporadas donde
    hay dato para ella, no sobre las `n_seasons` completas.
    """
    if player_advanced.empty:
        return None

    target_year = season_start_year(target_season)
    eligible = player_advanced[
        (player_advanced["season"].apply(season_start_year) < target_year)
        & (player_advanced[TOTAL_MINUTES_COLUMN] >= min_minutes)
    ].copy()
    if eligible.empty:
        return None

    # Colapsar temporadas partidas por traspaso (varias filas, un año).
    # Cada métrica ponderada solo sobre SUS PROPIAS filas no nulas -- ver
    # el mismo razonamiento en compute_league_advanced_baselines.
    eligible["_year"] = eligible["season"].apply(season_start_year)
    collapsed = []
    for year, group in eligible.groupby("_year"):
        row: Dict[str, float] = {"_year": year}
        for metric in ADVANCED_METRICS:
            if metric not in group.columns:
                continue
            metric_rows = group[group[metric].notna()]
            metric_weights = metric_rows[TOTAL_MINUTES_COLUMN]
            if metric_weights.sum() > 0:
                row[metric] = float((metric_rows[metric] * metric_weights).sum() / metric_weights.sum())
        collapsed.append(row)

    recent = sorted(collapsed, key=lambda r: r["_year"], reverse=True)[:n_seasons]
    seasons_ago = {r["_year"]: target_year - r["_year"] for r in recent}
    result: Dict[str, float] = {}
    for metric in ADVANCED_METRICS:
        rows_with_metric = [r for r in recent if metric in r]
        if not rows_with_metric:
            continue  # ninguna temporada reciente tiene este dato -- no se inventa un valor
        weights = np.array([0.5 ** (seasons_ago[r["_year"]] / half_life) for r in rows_with_metric])
        total = weights.sum()
        if total <= 0:
            continue
        result[metric] = float(sum(r[metric] * w for r, w in zip(rows_with_metric, weights)) / total)
    return result if result else None


def blend_impact_per36(
    game_score_per36: float,
    advanced: Optional[Dict[str, float]],
    season_baseline: Optional[Dict[str, float]],
    impact_config: Dict[str, Any],
) -> float:
    """
    Aplica el ajuste aditivo del docstring del módulo a UN jugador.

    Devuelve `game_score_per36` sin tocar si el ajuste está desactivado,
    si el jugador no tiene métricas avanzadas utilizables, o si falta la
    línea base de su temporada -- degradar al Game Score puro es siempre
    seguro.
    """
    if not impact_config.get("enabled", True) or not advanced or not season_baseline:
        return float(game_score_per36)

    adjustment = 0.0
    for metric, weight_key in (
        ("PIE", "pie_weight"),
        ("NET_RATING", "net_rating_weight"),
        ("PCT_PLUSMINUS", "pct_plusminus_weight"),
    ):
        if metric in advanced and metric in season_baseline:
            adjustment += impact_config[weight_key] * (advanced[metric] - season_baseline[metric])
    return float(game_score_per36 + adjustment)


def adjusted_game_score_per36(
    game_score_per36: float,
    player_advanced: Optional[pd.DataFrame],
    target_season: str,
    baselines: Dict[str, Dict[str, float]],
    impact_config: Dict[str, Any],
) -> float:
    """
    Punto de entrada ÚNICO para los tres motores del proyecto
    (`backtesting.project_historical_player`,
    `aging_curve.build_aging_projection_dataset`,
    `league_simulation.project_team_roster`): encadena
    `compute_recency_weighted_advanced` + `blend_impact_per36`.

    Debe ser compartida a propósito -- este proyecto ya arrastró el mismo
    bug de normalización de minutos en dos módulos por tenerlo duplicado
    (ver `simulation.normalize_rotation_minutes`). `target_season` es la
    temporada que se PROYECTA; las métricas se toman solo de temporadas
    anteriores.
    """
    if not impact_config.get("enabled", True) or player_advanced is None or player_advanced.empty:
        return float(game_score_per36)

    advanced = compute_recency_weighted_advanced(
        player_advanced,
        target_season,
        n_seasons=impact_config["n_seasons_lookback"],
        half_life=impact_config["recency_half_life_seasons"],
        min_minutes=impact_config["min_minutes_for_advanced"],
    )
    # La línea base de referencia es la de la temporada ANTERIOR a la
    # proyectada: es de donde vienen las métricas del jugador, así que es
    # contra esa población contra la que hay que centrarlas.
    prior_season = _previous_season(target_season)
    return blend_impact_per36(
        game_score_per36, advanced, baselines.get(prior_season), impact_config
    )


def _previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'. Duplicado a propósito de data_pipeline: este
    módulo no debe importar la capa de ingesta (dependencia al revés)."""
    start_year = season_start_year(season) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def build_advanced_context(
    advanced_stats: pd.DataFrame, config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Empaqueta todo lo que hace falta para ajustar jugadores: filas
    avanzadas ya AGRUPADAS por PLAYER_ID, líneas base por temporada y la
    config resuelta. Devuelve None si el ajuste está desactivado o no hay
    datos -- los llamantes tratan None como "usa Game Score puro". El
    groupby se hace una sola vez porque filtrar el DataFrame completo por
    jugador (el backtest sweep proyecta ~7.000 jugador-caso) convertía la
    pasada en minutos.
    """
    impact_config = resolve_advanced_impact_config(config)
    if not impact_config.get("enabled", True) or advanced_stats is None or advanced_stats.empty:
        return None
    return {
        "by_player": {int(pid): group for pid, group in advanced_stats.groupby("PLAYER_ID")},
        "baselines": compute_league_advanced_baselines(
            advanced_stats, min_minutes=impact_config["min_minutes_for_advanced"]
        ),
        "impact_config": impact_config,
    }


def adjust_with_context(
    game_score_per36: float,
    player_id: int,
    target_season: str,
    context: Optional[Dict[str, Any]],
) -> float:
    """
    Aplica el ajuste a un jugador usando el contexto de
    `build_advanced_context`. Es la firma que usan los tres motores; con
    `context=None` devuelve el Game Score sin tocar.
    """
    if context is None:
        return float(game_score_per36)
    return adjusted_game_score_per36(
        game_score_per36,
        context["by_player"].get(int(player_id)),
        target_season,
        context["baselines"],
        context["impact_config"],
    )


def load_advanced_stats(processed_dir) -> pd.DataFrame:
    """
    Lee `league_advanced_player_stats.csv` (lo genera
    `data_pipeline.build_league_advanced_player_stats_dataset`). Devuelve
    un DataFrame VACÍO si no existe -- dataset opcional a propósito, para
    que clonar el repo no exija 19 llamadas más a la API. Si
    `league_pt_defend_stats.csv` también existe, le fusiona
    `PCT_PLUSMINUS` (ver `merge_pt_defend_stats`); su ausencia también es
    segura.
    """
    path = processed_dir / "league_advanced_player_stats.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    df = prepare_advanced_stats(df)

    pt_defend_path = processed_dir / "league_pt_defend_stats.csv"
    if pt_defend_path.exists() and pt_defend_path.stat().st_size > 0:
        df = merge_pt_defend_stats(df, pd.read_csv(pt_defend_path))
    return df


def prepare_advanced_stats(advanced_stats: pd.DataFrame) -> pd.DataFrame:
    """Añade `total_minutes` = MIN * GP, separada de `load_advanced_stats`
    para que los tests puedan pasar un DataFrame en memoria sin tocar disco."""
    if advanced_stats.empty:
        return advanced_stats
    df = advanced_stats.copy()
    df[TOTAL_MINUTES_COLUMN] = df["MIN"] * df["GP"]
    return df


def merge_pt_defend_stats(advanced_stats: pd.DataFrame, pt_defend_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna `PCT_PLUSMINUS` (defensa por tracking, ver
    `data_pipeline.fetch_league_pt_defend_stats`) a `advanced_stats`,
    cruzando por (PLAYER_ID, season). Colapsa filas duplicadas de un
    jugador traspasado a mitad de temporada ponderando por D_FGA (tiros
    defendidos de cerca -- este endpoint no trae minutos). Deja
    `PCT_PLUSMINUS` en NaN para jugador/temporada sin datos de tracking.
    """
    if advanced_stats.empty or pt_defend_stats is None or pt_defend_stats.empty:
        return advanced_stats

    pt = pt_defend_stats[pt_defend_stats["D_FGA"] > 0].copy()
    if pt.empty:
        return advanced_stats
    pt["season"] = pt["season"].astype(str)
    pt["_weighted"] = pt["PCT_PLUSMINUS"] * pt["D_FGA"]
    collapsed = (
        pt.groupby(["PLAYER_ID", "season"])
        .apply(lambda g: g["_weighted"].sum() / g["D_FGA"].sum(), include_groups=False)
        .rename("PCT_PLUSMINUS")
        .reset_index()
    )
    return advanced_stats.merge(collapsed, on=["PLAYER_ID", "season"], how="left")
