"""
injury_model.py

Primer submódulo de la capa de contexto de temporada (ver roadmap en
README.md): calcula un "risk_score" (0-1) de lesión por jugador a partir
de su historial de disponibilidad en `roster_career_stats.csv`.

nba_api no expone historial de lesiones (fecha/tipo/gravedad), así que
este módulo usa como proxy la disponibilidad histórica -- partidos
jugados (GP) vs. partidos de calendario esa temporada.

El risk_score combina tres componentes 0-1 con pesos configurables
(`config["injury_model"]`, default historical_load=0.45, recency=0.35,
age=0.20 -- el historial concreto de un jugador pesa más que su edad en
abstracto):

1. `historical_load_score` -- % medio de partidos perdidos en las
   últimas N temporadas, sin ponderar por recencia.
2. `recency_score` -- lo mismo pero con decaimiento exponencial por
   recencia: el historial reciente es el predictor individual más
   fuerte de lesión futura (Ruddy et al., PMC6176657).
3. `age_score` -- curva de riesgo por edad que sube y luego SE APLANA
   (no decrece, no sigue subiendo). La epidemiología (Mack et al.,
   PMC11569584) muestra pico de incidencia en 6-15 años de experiencia
   (~27-34 años) y sin tasas más altas en veteranos muy longevos,
   probablemente por sesgo de supervivencia.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_paths  # noqa: E402
from season_utils import dedupe_traded_seasons, season_start_year  # noqa: E402

# Duración de temporada (partidos de calendario). 82 es el estándar;
# estas son las excepciones históricas conocidas (hecho de calendario de
# la liga, no algo específico de equipo/jugador).
_SEASON_LENGTH_EXCEPTIONS: Dict[str, int] = {
    "2011-12": 66,  # lockout
    "2019-20": 72,  # burbuja COVID-19: varió 64-75 según equipo, 72 es una media conservadora
    "2020-21": 72,  # COVID-19
}
_DEFAULT_SEASON_LENGTH = 82

DEFAULT_WEIGHTS: Dict[str, float] = {
    "historical_load": 0.45,
    "recency": 0.35,
    "age": 0.20,
}

# `peak_end_age` documenta la ventana de mayor incidencia (~32-36) pero la
# fórmula no lo usa: el plateau ya se alcanza en `peak_start_age` y se
# mantiene plano después.
DEFAULT_AGE_CURVE: Dict[str, float] = {
    "low_risk_age": 24,
    "peak_start_age": 32,
    "peak_end_age": 36,
    "base_risk": 0.05,
    "plateau_risk": 0.85,
}

DEFAULT_N_SEASONS_LOOKBACK = 3
DEFAULT_RECENCY_HALF_LIFE_SEASONS = 1.0


def season_length(season_id: str) -> int:
    """Partidos de calendario para una temporada dada (82 salvo excepciones)."""
    return _SEASON_LENGTH_EXCEPTIONS.get(str(season_id), _DEFAULT_SEASON_LENGTH)


def compute_games_missed_pct(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """
    Añade `season_length` y `games_missed_pct` a un DataFrame de temporadas
    de UN jugador (una fila por temporada, columnas SEASON_ID y GP como en
    roster_career_stats.csv).
    """
    df = dedupe_traded_seasons(player_seasons).copy()
    df["season_length"] = df["SEASON_ID"].apply(season_length)
    df["games_missed_pct"] = (1 - df["GP"] / df["season_length"]).clip(lower=0.0, upper=1.0)
    return df


def _most_recent_n_seasons(df_with_missed_pct: pd.DataFrame, n_seasons: int) -> pd.DataFrame:
    """Las N temporadas más recientes, ordenadas de más a menos reciente."""
    df = df_with_missed_pct.assign(
        _start_year=df_with_missed_pct["SEASON_ID"].apply(season_start_year)
    )
    df = df.sort_values("_start_year", ascending=False)
    return df.head(n_seasons).reset_index(drop=True)


def compute_historical_load(
    player_seasons: pd.DataFrame, n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK
) -> float:
    """% medio (sin ponderar) de partidos perdidos en las últimas N temporadas."""
    df = compute_games_missed_pct(player_seasons)
    recent = _most_recent_n_seasons(df, n_seasons)
    if recent.empty:
        return 0.0
    return float(recent["games_missed_pct"].mean())


def compute_recency_score(
    player_seasons: pd.DataFrame,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    half_life_seasons: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
) -> float:
    """
    % de partidos perdidos en las últimas N temporadas, ponderado con
    decaimiento exponencial por recencia: peso = 0.5 ** (temporadas_atrás /
    half_life_seasons). Con half_life_seasons=1.0 (por defecto), una
    ausencia en la temporada más reciente pesa el doble que una de la
    temporada anterior, y el cuádruple que una de hace dos temporadas.
    """
    df = compute_games_missed_pct(player_seasons)
    recent = _most_recent_n_seasons(df, n_seasons)
    if recent.empty:
        return 0.0

    seasons_ago = recent.index.to_numpy()  # 0 = más reciente
    weights = 0.5 ** (seasons_ago / half_life_seasons)
    weighted_sum = (weights * recent["games_missed_pct"].to_numpy()).sum()
    return float(weighted_sum / weights.sum())


def compute_age_score(age: float, age_curve_params: Optional[Dict[str, float]] = None) -> float:
    """Plana en `base_risk` hasta `low_risk_age`, rampa lineal hasta
    `plateau_risk` en `peak_start_age`, y se mantiene plana después."""
    params = {**DEFAULT_AGE_CURVE, **(age_curve_params or {})}
    low = params["low_risk_age"]
    peak_start = params["peak_start_age"]
    base = params["base_risk"]
    plateau = params["plateau_risk"]

    if age <= low:
        return float(base)
    if age >= peak_start:
        return float(plateau)

    fraction = (age - low) / (peak_start - low)
    return float(base + fraction * (plateau - base))


def _current_age(player_seasons: pd.DataFrame) -> float:
    """Edad del jugador en su temporada más reciente disponible."""
    df = player_seasons.assign(
        _start_year=player_seasons["SEASON_ID"].apply(season_start_year)
    )
    most_recent = df.sort_values("_start_year", ascending=False).iloc[0]
    return float(most_recent["PLAYER_AGE"])


def compute_risk_score(
    player_seasons: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    age_curve_params: Optional[Dict[str, float]] = None,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    recency_half_life_seasons: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
) -> Dict[str, float]:
    """
    Combina los tres componentes en un risk_score (0-1) para UN jugador.
    `player_seasons` es un DataFrame con una fila por temporada de ese
    jugador (subconjunto de roster_career_stats.csv).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    historical_load_score = compute_historical_load(player_seasons, n_seasons)
    recency_score = compute_recency_score(player_seasons, n_seasons, recency_half_life_seasons)
    current_age = _current_age(player_seasons)
    age_score = compute_age_score(current_age, age_curve_params)

    risk = (
        w["historical_load"] * historical_load_score
        + w["recency"] * recency_score
        + w["age"] * age_score
    )

    return {
        "current_age": current_age,
        "historical_load_score": historical_load_score,
        "recency_score": recency_score,
        "age_score": age_score,
        "risk_score": float(min(max(risk, 0.0), 1.0)),
    }


def build_injury_risk_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Punto de entrada: lee roster_career_stats.csv (generado por
    data_pipeline.py), calcula el risk_score de cada jugador del roster y
    guarda el resultado en data/processed/injury_risk.csv.
    """
    paths = get_paths(config)
    stats_path = paths["processed"] / "roster_career_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"No se encontró {stats_path}. Corre `python src/data_pipeline.py` "
            "primero para generar roster_career_stats.csv."
        )

    df = pd.read_csv(stats_path)

    injury_cfg = config.get("injury_model", {})
    weights = {**DEFAULT_WEIGHTS, **injury_cfg.get("weights", {})}
    age_curve_params = {**DEFAULT_AGE_CURVE, **injury_cfg.get("age_curve", {})}
    n_seasons = injury_cfg.get("n_seasons_lookback", DEFAULT_N_SEASONS_LOOKBACK)
    half_life = injury_cfg.get("recency_half_life_seasons", DEFAULT_RECENCY_HALF_LIFE_SEASONS)

    rows = []
    covered_player_ids = set()
    for (player_id, player_name), group in df.groupby(["PLAYER_ID", "player_name"]):
        covered_player_ids.add(player_id)
        result = compute_risk_score(
            group,
            weights=weights,
            age_curve_params=age_curve_params,
            n_seasons=n_seasons,
            recency_half_life_seasons=half_life,
        )
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "seasons_used": min(len(group), n_seasons),
                **result,
            }
        )

    # Rookies sin filas en roster_career_stats.csv: sin historial no hay
    # evidencia de riesgo, se asume el piso (0.0).
    roster_player_ids = {p["player_id"] for p in config["roster"] if p.get("player_id")}
    for player_id in roster_player_ids - covered_player_ids:
        player_name = next(p["name"] for p in config["roster"] if p.get("player_id") == player_id)
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "seasons_used": 0,
                "current_age": None,
                "historical_load_score": 0.0,
                "recency_score": 0.0,
                "age_score": 0.0,
                "risk_score": 0.0,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    out_path = paths["processed"] / "injury_risk.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} jugadores)")
    return result_df


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config

    build_injury_risk_dataset(load_config())
