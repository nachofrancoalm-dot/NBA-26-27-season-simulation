"""
routers/team.py

Endpoints para las 4 sub-pestañas de "Mi equipo" (Roster, Simulación,
Sinergia, Backtesting). Cada uno reutiliza dashboard/data_loader.py sin
reimplementar transformaciones -- solo empaqueta la salida en JSON (ver
webapp/serializers.py) y decide qué código HTTP devolver cuando falta
un CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from dashboard.data_loader import (  # noqa: E402 -- importar primero inserta src/ en sys.path
    BACKTEST_GLOSSARY,
    CALIBRATION_GLOSSARY,
    ROSTER_STAT_GLOSSARY,
    SIMULATION_GLOSSARY,
    SYNERGY_GLOSSARY,
    compute_win_distribution_summary,
    load_backtest_summary,
    load_backtest_sweep_calibration,
    load_backtest_sweep_summary,
    load_lineup_synergy_pairs,
    load_roster_overview,
    load_simulation_results,
    run_single_season_player_log_simulation,
    select_roster_view,
)
from config_loader import load_config  # noqa: E402
from simulation import compute_simulation_results  # noqa: E402

from webapp.serializers import df_to_records, series_to_dict

router = APIRouter()


@router.get("/roster")
def get_roster(mode: str = Query("per_game", pattern="^(per_game|totals)$")):
    config = load_config()
    overview = load_roster_overview(config)
    if overview is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontró aging_curve_projection.csv. Corre `python src/data_pipeline.py` "
                "y luego build_aging_projection_dataset (ver README)."
            ),
        )
    roster_view = select_roster_view(
        overview, mode=mode, games_per_season=config["simulation"]["games_per_season"]
    )
    # player_id se adjunta aparte (select_roster_view no lo expone) para
    # el popup de doble clic -- el orden de filas no cambia, zip por posición seguro.
    roster_view = roster_view.assign(player_id=overview["player_id"].to_numpy())
    glossary = {col: ROSTER_STAT_GLOSSARY[col] for col in roster_view.columns if col in ROSTER_STAT_GLOSSARY}
    return {"players": df_to_records(roster_view), "glossary": glossary}


@router.get("/simulation")
def get_simulation():
    config = load_config()
    results = load_simulation_results(config)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontró simulation_results.csv. Corre build_simulation_dataset (ver README).",
        )
    summary = compute_win_distribution_summary(results)
    wins_histogram = series_to_dict(results["wins"].value_counts().sort_index())
    net_rating_sorted = results["net_rating_estimate_mean"].sort_values().reset_index(drop=True).tolist()
    return {
        "summary": summary,
        "n_seasons": len(results),
        "wins_histogram": wins_histogram,
        "net_rating_sorted": net_rating_sorted,
        "glossary": SIMULATION_GLOSSARY,
    }


@router.post("/simulation/no-injuries")
def post_simulation_no_injuries():
    """Variante "en vivo" de la simulación con salud perfecta, para
    comparar contra /simulation. No escribe simulation_results.csv."""
    config = load_config()
    player_ids = [p["player_id"] for p in config["roster"] if p.get("player_id")]
    if not player_ids:
        raise HTTPException(status_code=404, detail="El roster del config no tiene player_id definidos.")
    try:
        results = compute_simulation_results(config, risk_scores_override=np.zeros(len(player_ids)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    summary = compute_win_distribution_summary(results)
    wins_histogram = series_to_dict(results["wins"].value_counts().sort_index())
    net_rating_sorted = results["net_rating_estimate_mean"].sort_values().reset_index(drop=True).tolist()
    return {
        "summary": summary,
        "n_seasons": len(results),
        "wins_histogram": wins_histogram,
        "net_rating_sorted": net_rating_sorted,
        "glossary": SIMULATION_GLOSSARY,
    }


@router.post("/simulation/season-log")
def post_simulation_season_log():
    config = load_config()
    log = run_single_season_player_log_simulation(config)
    if log is None:
        raise HTTPException(
            status_code=404,
            detail="Faltan aging_curve_projection.csv o injury_risk.csv para simular una temporada.",
        )
    games_per_season = config["simulation"]["games_per_season"]
    view = log[["player_name", "games_played", "games_missed", "injury_events"]].copy()
    view = view.sort_values("games_missed", ascending=False).reset_index(drop=True)
    return {"games_per_season": games_per_season, "players": view.to_dict(orient="records")}


@router.get("/synergy")
def get_synergy():
    config = load_config()
    pairs = load_lineup_synergy_pairs(config)
    if pairs is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontró lineup_synergy_pairs.csv. Corre build_lineup_synergy_dataset (ver README).",
        )
    return {"pairs": df_to_records(pairs), "glossary": SYNERGY_GLOSSARY}


@router.get("/backtest")
def get_backtest():
    config = load_config()
    summary = load_backtest_summary(config)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontró backtest_summary.csv. Corre build_backtest_dataset (ver README).",
        )
    extreme = summary[(summary["actual_percentile"] < 5) | (summary["actual_percentile"] > 95)]
    return {
        "cases": df_to_records(summary),
        "glossary": BACKTEST_GLOSSARY,
        "n_extreme_percentile_cases": int(len(extreme)),
        "n_cases": int(len(summary)),
    }


@router.get("/backtest/sweep")
def get_backtest_sweep():
    config = load_config()
    calibration = load_backtest_sweep_calibration(config)
    summary = load_backtest_sweep_summary(config)
    if calibration is None or summary is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontró backtest_sweep_summary.csv. Requiere "
                "`python src/data_pipeline.py --backtest-sweep` y luego build_backtest_sweep_dataset."
            ),
        )
    row = calibration.iloc[0].to_dict()

    percentile_bins = pd.cut(
        summary["actual_percentile"],
        bins=range(0, 101, 10),
        include_lowest=True,
        labels=[f"{low}-{low + 10}" for low in range(0, 100, 10)],
    )
    percentile_histogram = series_to_dict(percentile_bins.value_counts().sort_index())

    scatter = summary[["actual_wins", "simulated_wins_mean"]].rename(
        columns={"actual_wins": "actual", "simulated_wins_mean": "simulated"}
    )

    return {
        "calibration": {k: (None if pd.isna(v) else v) for k, v in row.items()},
        "percentile_histogram": percentile_histogram,
        "scatter": df_to_records(scatter),
        "cases": df_to_records(summary.sort_values("actual_percentile")),
        "glossary": CALIBRATION_GLOSSARY,
    }
