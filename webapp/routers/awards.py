"""
routers/awards.py

GET /api/awards -- serializa el dict de dashboard/data_loader.compute_awards_summary
(que a su vez llama src/awards_projection.py) a JSON. Ninguna heurística
nueva: solo empaquetado.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from dashboard.data_loader import (  # noqa: E402 -- importar primero inserta src/ en sys.path
    AWARDS_GLOSSARY,
    SEASON_AWARDS_GLOSSARY,
    compute_awards_summary,
)
from config_loader import load_config  # noqa: E402
from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID  # noqa: E402

from webapp.serializers import df_to_records

router = APIRouter(prefix="/awards")

AWARD_DF_KEYS = ["mvp", "dpoy", "roy", "mip", "sixth_man", "coy", "all_star", "all_nba", "all_defensive"]


@router.get("")
def get_awards(scenario: str = Query("with_injuries", pattern="^(with_injuries|no_injuries)$")):
    config = load_config()
    summary = compute_awards_summary(config, scenario=scenario)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontró aging_curve_projection.csv ni league_player_projections.csv. "
                "Corre el pipeline de tu equipo (o --league para los 30 equipos) primero."
            ),
        )

    all_star_final = summary.get("all_star_final")
    commissioner_picks = []
    if all_star_final is not None and "commissioner_pick" in all_star_final.columns:
        commissioner_picks = df_to_records(all_star_final[all_star_final["commissioner_pick"]])

    return {
        "scope": summary["scope"],
        **{key: df_to_records(summary.get(key)) for key in AWARD_DF_KEYS},
        "all_star_nationality_quota": summary.get("all_star_nationality_quota"),
        "commissioner_picks": commissioner_picks,
        "glossary": AWARDS_GLOSSARY,
        "season_awards_glossary": SEASON_AWARDS_GLOSSARY,
        "team_ids": ABBREVIATION_TO_TEAM_ID,
    }
