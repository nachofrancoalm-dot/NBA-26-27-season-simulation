"""
conference_adjustment.py

Sexto submódulo del roadmap de contexto de temporada (ver README.md):
normaliza la fuerza relativa Este/Oeste por temporada, para poder
comparar el récord y el Net Rating de los `historical_comparables`
entre sí aunque cada uno juegue en una conferencia y temporada distinta.

Fundamento: a nivel de liga, WinPCT medio y DiffPointsPG medio son
exactamente 0.5 y 0 (suma cero), pero no dentro de cada conferencia por
separado, ya que los partidos inter-conferencia no son de suma cero
dentro del grupo. La desviación de la media de conferencia respecto a
esa línea base mide directamente cuánto más dura era esa conferencia esa
temporada, sin datos adicionales.

`compute_conference_strength_index()` calcula ese índice por
(temporada, conferencia); `compute_conference_adjusted_value()` resta el
índice al valor bruto del equipo (crédito por conferencia más dura);
`build_conference_adjustment_dataset()` combina
`historical_comparables_standings.csv` con `performance_curve_summary.csv`
(a diferencia de opponent_weighting.py, este módulo sí depende del
resumen ya agregado de performance_curve.py, porque comparar résumenes
entre casos es su propósito).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_paths  # noqa: E402

DEFAULT_NET_RATING_METRIC = "DiffPointsPG"
NET_RATING_BASELINE = 0.0
WIN_PCT_BASELINE = 0.5


def compute_conference_strength_index(
    standings: pd.DataFrame, metric: str = DEFAULT_NET_RATING_METRIC, baseline: float = NET_RATING_BASELINE
) -> pd.DataFrame:
    """
    Por (season, Conference): media de `metric` menos `baseline`. Esa
    desviación es el índice de fuerza de conferencia esa temporada (ver
    fundamento estadístico en el docstring del módulo).
    """
    index = (
        standings.groupby(["season", "Conference"])[metric]
        .mean()
        .reset_index(name="conference_mean")
    )
    index["conference_index"] = index["conference_mean"] - baseline
    return index


def compute_conference_adjusted_value(raw_value: float, conference_index: float) -> float:
    """Resta el índice de conferencia al valor bruto -- da crédito por
    jugar en una conferencia más dura, penaliza levemente jugar en una más floja."""
    return raw_value - conference_index


def get_team_conference_row(standings: pd.DataFrame, team_id: int, season: str) -> pd.Series:
    """Fila de standings de UN equipo en UNA temporada (para leer su Conference, WinPCT, etc.)."""
    match = standings[(standings["TeamID"] == team_id) & (standings["season"] == season)]
    if match.empty:
        raise ValueError(f"No hay standings para team_id={team_id}, season={season}.")
    return match.iloc[0]


def build_conference_adjustment_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Lee historical_comparables_standings.csv y performance_curve_summary.csv
    (generados por data_pipeline.py y performance_curve.py
    respectivamente) y guarda
    data/processed/conference_adjustment_summary.csv con WinPCT y Net
    Rating crudos vs. ajustados por conferencia, por caso histórico.
    """
    paths = get_paths(config)
    standings_path = paths["processed"] / "historical_comparables_standings.csv"
    perf_summary_path = paths["processed"] / "performance_curve_summary.csv"

    for path, builder in [
        (standings_path, "data_pipeline.build_historical_comparables_standings_dataset"),
        (perf_summary_path, "context.performance_curve.build_performance_curve_dataset"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}. Corre `{builder}` primero.")

    standings = pd.read_csv(standings_path)
    perf_summary = pd.read_csv(perf_summary_path).set_index("comparable_name")

    net_rating_cfg = config.get("conference_adjustment", {})
    metric = net_rating_cfg.get("net_rating_metric", DEFAULT_NET_RATING_METRIC)

    net_rating_index = compute_conference_strength_index(standings, metric, NET_RATING_BASELINE)
    win_pct_index = compute_conference_strength_index(standings, "WinPCT", WIN_PCT_BASELINE)

    rows = []
    for case in config["historical_comparables"]:
        team_row = get_team_conference_row(standings, case["team_id"], case["season"])
        conference = team_row["Conference"]

        nr_idx_row = net_rating_index[
            (net_rating_index["season"] == case["season"]) & (net_rating_index["Conference"] == conference)
        ]
        wp_idx_row = win_pct_index[
            (win_pct_index["season"] == case["season"]) & (win_pct_index["Conference"] == conference)
        ]
        conference_net_rating_index = float(nr_idx_row["conference_index"].iloc[0])
        conference_win_pct_index = float(wp_idx_row["conference_index"].iloc[0])

        raw_win_pct = float(team_row["WinPCT"])
        adjusted_win_pct = compute_conference_adjusted_value(raw_win_pct, conference_win_pct_index)

        raw_net_rating = float(perf_summary.loc[case["name"], "full_regular_season_net_rating"])
        adjusted_net_rating = compute_conference_adjusted_value(raw_net_rating, conference_net_rating_index)

        rows.append(
            {
                "comparable_name": case["name"],
                "season": case["season"],
                "conference": conference,
                "conference_net_rating_index": conference_net_rating_index,
                "conference_win_pct_index": conference_win_pct_index,
                "raw_win_pct": raw_win_pct,
                "conference_adjusted_win_pct": adjusted_win_pct,
                "raw_net_rating": raw_net_rating,
                "conference_adjusted_net_rating": adjusted_net_rating,
            }
        )

    result_df = pd.DataFrame(rows)
    out_path = paths["processed"] / "conference_adjustment_summary.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} casos)")
    return result_df


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config

    build_conference_adjustment_dataset(load_config())
