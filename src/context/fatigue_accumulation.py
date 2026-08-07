"""
fatigue_accumulation.py

Segundo submódulo de la capa de contexto de temporada (ver roadmap en
README.md): calcula un "fatigue_score" (0-1) por jugador que mide desgaste
acumulado por carga de minutos a lo largo de la carrera -- distinto de
`injury_model.py`, que mide riesgo a partir de partidos perdidos.

A diferencia de injury_model.py, este módulo NO tiene una variable de edad
explícita. El desgaste por carrera larga ya se refleja de forma natural en
`cumulative_load_score` (más temporadas jugadas = más minutos acumulados),
así que añadir edad como componente aparte duplicaría esa señal.

INPUTS
------
- `roster_career_stats.csv` (temporada regular, generado por
  `data_pipeline.py`).
- `roster_playoff_career_stats.csv` (playoffs, mismo pipeline). Un
  jugador sin apariciones en playoffs una temporada dada simplemente no
  tiene fila para esa SEASON_ID -- se trata como 0 minutos/partidos de
  playoffs esa temporada, no como dato faltante.

DISEÑO DEL fatigue_score
--------------------------
Tres componentes 0-1, pesos configurables en `config["fatigue_model"]`
(nunca hardcodeados). A diferencia de injury_model.py, aquí no hay
literatura publicada que justifique una jerarquía clara entre
componentes, así que los pesos por defecto son similares entre sí:

1. `cumulative_load_score` -- minutos totales de carrera (regular +
   playoffs), normalizados con un tope configurable de "carrera
   longeva" (curva lineal-y-saturada: simple y transparente, no se
   inventa una curva no lineal sin evidencia que la respalde).
2. `recent_intensity_score` -- minutos/partido (regular + playoffs) en
   las últimas N temporadas frente a un umbral configurable de "uso
   pesado", ponderado por recencia con decaimiento exponencial (mismo
   estilo que injury_model.py, implementado de forma independiente para
   mantener cada submódulo autocontenido).
3. `sustained_streak_score` -- nº de temporadas recientes consecutivas
   sin una caída de carga (sin "temporada de descarga/descanso"), con
   retornos decrecientes vía curva de saturación exponencial.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_paths  # noqa: E402
from season_utils import dedupe_traded_seasons, season_start_year  # noqa: E402

DEFAULT_WEIGHTS: Dict[str, float] = {
    "cumulative_load": 0.35,
    "recent_intensity": 0.35,
    "sustained_streak": 0.30,
}

DEFAULT_HIGH_MILEAGE_MINUTES = 35000.0
DEFAULT_HEAVY_MINUTES_PER_GAME = 34.0
DEFAULT_FULL_SEASON_GP = 65  # temporada "sin descanso": jugó casi todo el calendario
DEFAULT_N_SEASONS_LOOKBACK = 3
DEFAULT_RECENCY_HALF_LIFE_SEASONS = 1.0
DEFAULT_STREAK_HALF_LIFE_SEASONS = 2.0


def merge_regular_and_playoff_seasons(
    regular_seasons: pd.DataFrame, playoff_seasons: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Combina temporada regular + playoffs de UN jugador en una fila por
    SEASON_ID con GP_total y MIN_total sumados. Si el jugador no tiene
    playoffs esa temporada (o ninguna), esas filas simplemente aportan 0.
    """
    reg = dedupe_traded_seasons(regular_seasons)
    reg_agg = reg.groupby("SEASON_ID", as_index=False)[["GP", "MIN"]].sum()
    reg_agg = reg_agg.rename(columns={"GP": "GP_regular", "MIN": "MIN_regular"})

    if playoff_seasons is not None and not playoff_seasons.empty:
        po = dedupe_traded_seasons(playoff_seasons)
        po_agg = po.groupby("SEASON_ID", as_index=False)[["GP", "MIN"]].sum()
        po_agg = po_agg.rename(columns={"GP": "GP_playoff", "MIN": "MIN_playoff"})
        merged = reg_agg.merge(po_agg, on="SEASON_ID", how="left")
    else:
        merged = reg_agg
        merged["GP_playoff"] = 0
        merged["MIN_playoff"] = 0.0

    merged[["GP_playoff", "MIN_playoff"]] = merged[["GP_playoff", "MIN_playoff"]].fillna(0)
    merged["GP_total"] = merged["GP_regular"] + merged["GP_playoff"]
    merged["MIN_total"] = merged["MIN_regular"] + merged["MIN_playoff"]
    return merged


def _most_recent_n_seasons(merged_seasons: pd.DataFrame, n_seasons: int) -> pd.DataFrame:
    df = merged_seasons.assign(_start_year=merged_seasons["SEASON_ID"].apply(season_start_year))
    df = df.sort_values("_start_year", ascending=False)
    return df.head(n_seasons).reset_index(drop=True)


def compute_cumulative_load_score(
    merged_seasons: pd.DataFrame, high_mileage_minutes: float = DEFAULT_HIGH_MILEAGE_MINUTES
) -> float:
    """% de minutos totales de carrera (regular+playoffs) frente al tope de
    'carrera longeva', capado en 1.0."""
    if high_mileage_minutes <= 0:
        return 0.0
    total_minutes = float(merged_seasons["MIN_total"].sum())
    return float(min(total_minutes / high_mileage_minutes, 1.0))


def compute_recent_intensity_score(
    merged_seasons: pd.DataFrame,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    heavy_minutes_per_game: float = DEFAULT_HEAVY_MINUTES_PER_GAME,
    half_life_seasons: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
) -> float:
    """
    Minutos/partido (regular+playoffs) en las últimas N temporadas frente
    al umbral de 'uso pesado', ponderado por recencia (decaimiento
    exponencial: la temporada más reciente pesa más).
    """
    recent = _most_recent_n_seasons(merged_seasons, n_seasons)
    if recent.empty or heavy_minutes_per_game <= 0:
        return 0.0

    min_per_game = recent["MIN_total"] / recent["GP_total"].replace(0, pd.NA)
    min_per_game = min_per_game.fillna(0.0)
    intensity = (min_per_game / heavy_minutes_per_game).clip(upper=1.0)

    seasons_ago = recent.index.to_numpy()  # 0 = más reciente
    weights = 0.5 ** (seasons_ago / half_life_seasons)
    weighted_sum = (weights * intensity.to_numpy()).sum()
    return float(weighted_sum / weights.sum())


def compute_sustained_streak_score(
    merged_seasons: pd.DataFrame,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    full_season_gp: int = DEFAULT_FULL_SEASON_GP,
    streak_half_life_seasons: float = DEFAULT_STREAK_HALF_LIFE_SEASONS,
) -> float:
    """
    Nº de temporadas recientes consecutivas (empezando por la más
    reciente) sin una caída de carga -- GP_total >= full_season_gp -- con
    retornos decrecientes: score = 1 - 0.5 ** (streak / streak_half_life).
    Una sola temporada de descanso/descarga rompe la racha.
    """
    recent = _most_recent_n_seasons(merged_seasons, n_seasons)
    if recent.empty:
        return 0.0

    streak = 0
    for _, row in recent.iterrows():
        if row["GP_total"] >= full_season_gp:
            streak += 1
        else:
            break

    if streak == 0:
        return 0.0
    return float(1 - 0.5 ** (streak / streak_half_life_seasons))


def compute_fatigue_score(
    regular_seasons: pd.DataFrame,
    playoff_seasons: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    high_mileage_minutes: float = DEFAULT_HIGH_MILEAGE_MINUTES,
    heavy_minutes_per_game: float = DEFAULT_HEAVY_MINUTES_PER_GAME,
    full_season_gp: int = DEFAULT_FULL_SEASON_GP,
    recency_half_life_seasons: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
    streak_half_life_seasons: float = DEFAULT_STREAK_HALF_LIFE_SEASONS,
) -> Dict[str, float]:
    """Combina los tres componentes en un fatigue_score (0-1) para UN jugador."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    merged = merge_regular_and_playoff_seasons(regular_seasons, playoff_seasons)

    cumulative_load_score = compute_cumulative_load_score(merged, high_mileage_minutes)
    recent_intensity_score = compute_recent_intensity_score(
        merged, n_seasons, heavy_minutes_per_game, recency_half_life_seasons
    )
    sustained_streak_score = compute_sustained_streak_score(
        merged, n_seasons, full_season_gp, streak_half_life_seasons
    )

    fatigue = (
        w["cumulative_load"] * cumulative_load_score
        + w["recent_intensity"] * recent_intensity_score
        + w["sustained_streak"] * sustained_streak_score
    )

    return {
        "cumulative_load_score": cumulative_load_score,
        "recent_intensity_score": recent_intensity_score,
        "sustained_streak_score": sustained_streak_score,
        "fatigue_score": float(min(max(fatigue, 0.0), 1.0)),
    }


def build_fatigue_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Punto de entrada: lee roster_career_stats.csv y
    roster_playoff_career_stats.csv (generados por data_pipeline.py),
    calcula el fatigue_score de cada jugador del roster y guarda el
    resultado en data/processed/fatigue_risk.csv.
    """
    paths = get_paths(config)
    reg_path = paths["processed"] / "roster_career_stats.csv"
    if not reg_path.exists():
        raise FileNotFoundError(
            f"No se encontró {reg_path}. Corre `python src/data_pipeline.py` "
            "primero para generar roster_career_stats.csv."
        )
    regular_df = pd.read_csv(reg_path)

    playoff_path = paths["processed"] / "roster_playoff_career_stats.csv"
    playoff_df = pd.read_csv(playoff_path) if playoff_path.exists() and playoff_path.stat().st_size > 0 else pd.DataFrame()

    fatigue_cfg = config.get("fatigue_model", {})
    weights = {**DEFAULT_WEIGHTS, **fatigue_cfg.get("weights", {})}
    n_seasons = fatigue_cfg.get("n_seasons_lookback", DEFAULT_N_SEASONS_LOOKBACK)
    high_mileage_minutes = fatigue_cfg.get("high_mileage_minutes", DEFAULT_HIGH_MILEAGE_MINUTES)
    heavy_minutes_per_game = fatigue_cfg.get("heavy_minutes_per_game", DEFAULT_HEAVY_MINUTES_PER_GAME)
    full_season_gp = fatigue_cfg.get("full_season_gp", DEFAULT_FULL_SEASON_GP)
    recency_half_life = fatigue_cfg.get("recency_half_life_seasons", DEFAULT_RECENCY_HALF_LIFE_SEASONS)
    streak_half_life = fatigue_cfg.get("streak_half_life_seasons", DEFAULT_STREAK_HALF_LIFE_SEASONS)

    rows = []
    covered_player_ids = set()
    for (player_id, player_name), group in regular_df.groupby(["PLAYER_ID", "player_name"]):
        covered_player_ids.add(player_id)
        player_playoffs = (
            playoff_df[playoff_df["PLAYER_ID"] == player_id] if not playoff_df.empty else None
        )
        result = compute_fatigue_score(
            group,
            player_playoffs,
            weights=weights,
            n_seasons=n_seasons,
            high_mileage_minutes=high_mileage_minutes,
            heavy_minutes_per_game=heavy_minutes_per_game,
            full_season_gp=full_season_gp,
            recency_half_life_seasons=recency_half_life,
            streak_half_life_seasons=streak_half_life,
        )
        rows.append({"player_id": player_id, "player_name": player_name, **result})

    # Jugadores del roster sin NINGUNA temporada en roster_career_stats.csv
    # (rookies de verdad) -- sin historial no hay carga acumulada que
    # medir, se asume el piso (0.0). Mismo principio que injury_model.py
    # / aging_curve.zero_player_projection() para el mismo caso.
    roster_player_ids = {p["player_id"] for p in config["roster"] if p.get("player_id")}
    for player_id in roster_player_ids - covered_player_ids:
        player_name = next(p["name"] for p in config["roster"] if p.get("player_id") == player_id)
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "cumulative_load_score": 0.0,
                "recent_intensity_score": 0.0,
                "sustained_streak_score": 0.0,
                "fatigue_score": 0.0,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("fatigue_score", ascending=False).reset_index(drop=True)
    out_path = paths["processed"] / "fatigue_risk.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} jugadores)")
    return result_df


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config

    build_fatigue_dataset(load_config())
