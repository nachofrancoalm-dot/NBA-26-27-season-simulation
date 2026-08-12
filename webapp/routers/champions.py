"""
routers/champions.py

GET /api/champions -- pestaña "Liga NBA -> Campeones reales". Reutiliza
dashboard/data_loader.py sin reimplementar el cálculo de comparación
real-vs-simulado (mismo que hoy hace dashboard/app.py inline en la
sub-pestaña de campeones).
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from dashboard.data_loader import (  # noqa: E402 -- importar primero inserta src/ en sys.path
    CHAMPION_GLOSSARY,
    compute_champion_seed_distribution,
    compute_conference_standings,
    load_champion_roster_profiles,
    load_champion_seed_trajectories,
    load_champion_title_paths,
    load_league_playoff_summary,
    load_league_regular_season_summary,
)
from config_loader import load_config  # noqa: E402

from webapp.serializers import df_to_records

router = APIRouter(prefix="/champions")


def _seed_comparison(title_paths: pd.DataFrame, config) -> list[dict]:
    seed_dist = compute_champion_seed_distribution(title_paths)
    real_pct = {int(r.seed): r.pct for r in seed_dist.itertuples()}

    simulated_pct: dict[int, float] = {}
    regular = load_league_regular_season_summary(config)
    playoff = load_league_playoff_summary(config)
    if regular is not None and playoff is not None:
        standings = compute_conference_standings(regular, playoff)
        rows = [
            {"seed": int(row["seed"]), "champ": row["championship_pct"]}
            for conf in ("East", "West")
            for _, row in standings[conf].iterrows()
        ]
        sim = pd.DataFrame(rows).groupby("seed")["champ"].sum()
        if sim.sum() > 0:
            simulated_pct = (sim / sim.sum() * 100).to_dict()

    seeds = sorted(set(real_pct) | set(simulated_pct))
    return [
        {
            "seed": seed,
            "real_pct": round(real_pct.get(seed, 0.0), 1),
            "simulated_pct": round(simulated_pct.get(seed, 0.0), 1),
        }
        for seed in seeds
    ]


@router.get("")
def get_champions():
    config = load_config()
    title_paths = load_champion_title_paths(config)
    if title_paths is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontró champion_title_paths.csv. Requiere "
                "`python src/data_pipeline.py --backtest-sweep` y luego build_champion_analysis_dataset."
            ),
        )

    roster_profiles = load_champion_roster_profiles(config)
    seed_trajectories = load_champion_seed_trajectories(config)

    n_seasons = len(title_paths)
    n_distinct = int(title_paths["team_abbreviation"].nunique())
    counts = title_paths["team_abbreviation"].value_counts()
    most_titles = {"team_abbreviation": counts.idxmax(), "titles": int(counts.max())}

    return {
        "n_seasons": n_seasons,
        "n_distinct_champions": n_distinct,
        "most_titles": most_titles,
        "title_paths": df_to_records(title_paths),
        "roster_profiles": df_to_records(roster_profiles.round(1) if roster_profiles is not None else None),
        "seed_trajectories": df_to_records(seed_trajectories.reset_index() if seed_trajectories is not None else None),
        "seed_comparison": _seed_comparison(title_paths, config),
        "glossary": CHAMPION_GLOSSARY,
    }
