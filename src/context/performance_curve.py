"""
performance_curve.py

Cuarto submódulo de la capa de contexto de temporada (ver roadmap en
README.md): calcula Net Rating estimado en ventanas móviles sobre los
`historical_comparables`, para detectar arranques lentos de integración
y picos de forma en playoffs -- la narrativa central de "efecto
superequipo" que este proyecto busca validar por backtesting. Opera por
CASO (comparable histórico), no por jugador ni por partido del equipo
simulado.

`nba_api` no expone Net Rating oficial a nivel de partido individual sin
llamadas adicionales, así que este módulo lo ESTIMA:

    net_rating_estimate = PLUS_MINUS / posesiones_estimadas * 100
    POSS ≈ FGA - OREB + TOV + 0.44 * FTA

Es una aproximación ampliamente usada (p. ej. Basketball-Reference) pero
no el Off/Def Rating oficial de NBA.com; suficiente para comparar la
forma relativa de un mismo equipo a lo largo de su propia temporada sin
llamadas extra por el boxscore del rival.

`trend_slope` es la pendiente de una regresión lineal simple del rolling
Net Rating durante la temporada regular (positiva = el equipo mejora
según avanza la temporada).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_paths  # noqa: E402

DEFAULT_ROLLING_WINDOW_GAMES = 10
DEFAULT_EARLY_SEASON_GAMES = 15


def estimate_possessions(games: pd.DataFrame) -> pd.Series:
    """POSS ≈ FGA - OREB + TOV + 0.44 * FTA (fórmula estándar de analítica de básquet)."""
    return games["FGA"] - games["OREB"] + games["TOV"] + 0.44 * games["FTA"]


def compute_net_rating_estimate(games: pd.DataFrame) -> pd.DataFrame:
    """Añade `possessions_estimate` y `net_rating_estimate` a un DataFrame
    de partidos de UN caso histórico (columnas PLUS_MINUS, FGA, OREB, TOV, FTA)."""
    df = games.copy()
    df["possessions_estimate"] = estimate_possessions(df)
    df["net_rating_estimate"] = (
        df["PLUS_MINUS"] / df["possessions_estimate"].replace(0, np.nan) * 100
    ).fillna(0.0)
    return df


def compute_rolling_net_rating(
    games: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW_GAMES
) -> pd.DataFrame:
    """
    Ordena por GAME_DATE (temporada regular + playoffs como una única
    secuencia cronológica) y añade `rolling_net_rating`: media móvil de
    `net_rating_estimate` sobre las últimas `window` partidos.
    """
    df = compute_net_rating_estimate(games)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["game_number"] = range(1, len(df) + 1)
    df["rolling_net_rating"] = (
        df["net_rating_estimate"].rolling(window=window, min_periods=1).mean()
    )
    return df


def _linear_trend_slope(y: pd.Series) -> float:
    """Pendiente de una regresión lineal simple de y contra el índice (0, 1, 2, ...)."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    slope, _intercept = np.polyfit(x, y.to_numpy(), 1)
    return float(slope)


def summarize_season_narrative(
    games_with_rolling: pd.DataFrame, early_season_games: int = DEFAULT_EARLY_SEASON_GAMES
) -> Dict[str, float]:
    """
    Resumen de UN caso histórico: Net Rating de arranque vs. resto de
    temporada regular, Net Rating de playoffs, playoff_boost y
    trend_slope. `games_with_rolling` ya debe traer game_phase,
    net_rating_estimate y rolling_net_rating (ver compute_rolling_net_rating).
    """
    regular = games_with_rolling[games_with_rolling["game_phase"] == "regular"]
    playoffs = games_with_rolling[games_with_rolling["game_phase"] == "playoffs"]

    early = regular.head(early_season_games)
    rest = regular.iloc[early_season_games:]

    early_season_net_rating = float(early["net_rating_estimate"].mean()) if not early.empty else 0.0
    rest_of_season_net_rating = float(rest["net_rating_estimate"].mean()) if not rest.empty else 0.0
    full_regular_season_net_rating = (
        float(regular["net_rating_estimate"].mean()) if not regular.empty else 0.0
    )
    playoff_net_rating = float(playoffs["net_rating_estimate"].mean()) if not playoffs.empty else None
    playoff_boost = (
        playoff_net_rating - full_regular_season_net_rating if playoff_net_rating is not None else None
    )
    trend_slope = _linear_trend_slope(regular["rolling_net_rating"])

    return {
        "early_season_net_rating": early_season_net_rating,
        "rest_of_season_net_rating": rest_of_season_net_rating,
        "full_regular_season_net_rating": full_regular_season_net_rating,
        "playoff_net_rating": playoff_net_rating,
        "playoff_boost": playoff_boost,
        "trend_slope": trend_slope,
    }


def build_performance_curve_dataset(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Lee historical_comparables_advanced_game_logs.csv (generado por
    data_pipeline.py), calcula la serie de rolling Net Rating y el
    resumen narrativo por caso histórico. Guarda
    data/processed/performance_curve_by_game.csv (serie completa) y
    data/processed/performance_curve_summary.csv (un resumen por caso).
    """
    paths = get_paths(config)
    advanced_path = paths["processed"] / "historical_comparables_advanced_game_logs.csv"
    if not advanced_path.exists():
        raise FileNotFoundError(
            f"No se encontró {advanced_path}. Corre `python src/data_pipeline.py` "
            "primero para generar historical_comparables_advanced_game_logs.csv."
        )

    df = pd.read_csv(advanced_path)

    perf_cfg = config.get("performance_curve", {})
    window = perf_cfg.get("rolling_window_games", DEFAULT_ROLLING_WINDOW_GAMES)
    early_season_games = perf_cfg.get("early_season_games", DEFAULT_EARLY_SEASON_GAMES)

    by_game_frames = []
    summary_rows = []
    for comparable_name, group in df.groupby("comparable_name"):
        rolling = compute_rolling_net_rating(group, window)
        rolling["comparable_name"] = comparable_name
        by_game_frames.append(rolling)

        summary = summarize_season_narrative(rolling, early_season_games)
        summary_rows.append({"comparable_name": comparable_name, **summary})

    by_game_df = pd.concat(by_game_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    by_game_path = paths["processed"] / "performance_curve_by_game.csv"
    summary_path = paths["processed"] / "performance_curve_summary.csv"
    by_game_df.to_csv(by_game_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Guardado: {by_game_path} ({len(by_game_df)} filas)")
    print(f"Guardado: {summary_path} ({len(summary_df)} casos)")

    return {"by_game": by_game_df, "summary": summary_df}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config

    build_performance_curve_dataset(load_config())
