"""
routers/status.py

GET /api/status -- identidad del equipo y disponibilidad de cada dataset,
para poblar la barra lateral del frontend. Mismo criterio que
DATASET_STATUS en dashboard/app.py (líneas 158-167), reutilizando los
mismos loaders de dashboard/data_loader.py -- ninguna lógica nueva.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from dashboard.data_loader import (  # noqa: E402 -- importar primero inserta src/ en sys.path
    load_backtest_summary,
    load_backtest_sweep_summary,
    load_champion_title_paths,
    load_league_regular_season_summary,
    load_lineup_synergy_pairs,
    load_roster_overview,
    load_simulation_results,
)
from config_loader import load_config  # noqa: E402 -- mismo estilo de import que dashboard/data_loader.py

router = APIRouter()


@router.get("/status")
def get_status():
    config = load_config()
    team = config["team"]

    roster_overview = load_roster_overview(config)
    simulation_results = load_simulation_results(config)
    synergy_pairs = load_lineup_synergy_pairs(config)
    backtest_summary = load_backtest_summary(config)
    backtest_sweep_summary = load_backtest_sweep_summary(config)
    champion_title_paths = load_champion_title_paths(config)
    regular_season_summary = load_league_regular_season_summary(config)
    regular_season_summary_no_injuries = load_league_regular_season_summary(config, scenario="no_injuries")

    return {
        "team": {
            "name": team["name"],
            "abbreviation": team.get("abbreviation"),
            "season": team["season"],
            "team_id": team["team_id"],
        },
        "datasets": {
            "roster": roster_overview is not None,
            "simulation": simulation_results is not None,
            "synergy": synergy_pairs is not None,
            "backtest": backtest_summary is not None,
            "backtest_sweep": backtest_sweep_summary is not None,
            "league": regular_season_summary is not None,
            "league_no_injuries": regular_season_summary_no_injuries is not None,
            "champions": champion_title_paths is not None,
            "explainer": bool(os.environ.get("GROQ_API_KEY")),
        },
    }
