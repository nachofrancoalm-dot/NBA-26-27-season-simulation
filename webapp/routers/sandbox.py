"""
routers/sandbox.py

Endpoints del roster hipotético editable del splash (ver
webapp/static/js/views/splash.js): reemplazan el punto de partida fijo
"los 76ers del config" por cualquier combinación de jugadores reales de
los 30 equipos, simulada EN VIVO -- ver src/sandbox_simulation.py para
el motor (reutiliza simulation.run_monte_carlo tal cual, cero
matemática nueva).

GET /players       -> catálogo completo para el buscador del picker.
GET /default       -> el roster inicial (los 13 del config -- el punto de
                       partida real del proyecto, no un roster vacío).
POST /simulate      -> corre la simulación de TU equipo para el roster
                       que el usuario haya ensamblado.
POST /roster-stats  -> estadísticas individuales de cada jugador para
                       ese roster.
POST /league        -> la LIGA COMPLETA de 30 equipos con ese roster
                       sustituido en tu equipo (src/league_sandbox.py) --
                       standings, playoffs y premios "en vivo".
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dashboard.data_loader import (  # noqa: E402 -- importar primero inserta src/ en sys.path
    AWARDS_GLOSSARY,
    LEAGUE_GLOSSARY,
    ROSTER_STAT_GLOSSARY,
    SEASON_AWARDS_GLOSSARY,
    SIMULATION_GLOSSARY,
    compute_conference_standings,
    compute_win_distribution_summary,
)
from config_loader import load_config  # noqa: E402
from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID  # noqa: E402
from league_sandbox import compute_hypothetical_awards, simulate_hypothetical_league  # noqa: E402
from sandbox_simulation import (  # noqa: E402
    MAX_ROSTER_SIZE,
    MIN_ROSTER_SIZE,
    SandboxRosterError,
    compute_roster_player_stats,
    load_player_pool,
    simulate_custom_roster,
)

from webapp.routers.awards import AWARD_DF_KEYS
from webapp.serializers import df_to_records, series_to_dict

router = APIRouter(prefix="/sandbox")

PLAYER_POOL_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "team_abbreviation",
    "conference",
    "position",
    "current_age",
    "game_score_per36",
    "minutes_per_game_last_season",
    "games_played_last_season",
    "risk_score",
    "PPG",
    "RPG",
    "APG",
]


@router.get("/players")
def get_player_pool():
    config = load_config()
    try:
        pool = load_player_pool(config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    available = [c for c in PLAYER_POOL_COLUMNS if c in pool.columns]
    return {"players": df_to_records(pool[available]), "max_roster_size": MAX_ROSTER_SIZE, "min_roster_size": MIN_ROSTER_SIZE}


@router.get("/default")
def get_default_roster():
    """El roster inicial del sandbox: el mismo roster de 13 jugadores de
    `config/team_config.yaml` que ya se ve en Mi equipo -- el sandbox
    SUSTITUYE ese punto de partida, no arranca vacío."""
    config = load_config()
    player_ids = [p["player_id"] for p in config["roster"] if p.get("player_id")]
    return {"player_ids": player_ids, "team_name": config["team"]["name"]}


class SimulateRequest(BaseModel):
    player_ids: List[int] = Field(min_length=1)
    mc_overrides: Optional[dict] = None


@router.post("/simulate")
def post_simulate(body: SimulateRequest):
    config = load_config()
    try:
        results = simulate_custom_roster(config, body.player_ids, mc_overrides=body.mc_overrides)
    except SandboxRosterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


class RosterStatsRequest(BaseModel):
    player_ids: List[int] = Field(min_length=1)
    mode: str = "per_game"


@router.post("/roster-stats")
def post_roster_stats(body: RosterStatsRequest):
    """Estadísticas individuales de cada jugador PARA el roster hipotético
    que se esté editando -- las de `league_player_projections.csv` (PPG,
    minutos...) son las REALES de cada jugador en su equipo actual, no las
    que tendría en un roster inventado. Endpoint aparte de /simulate (no
    se fusiona en su respuesta) para que la tabla de jugadores se pueda
    pedir sin tener que correr también los 2.000 escenarios de Monte
    Carlo -- son dos preguntas distintas."""
    config = load_config()
    try:
        view = compute_roster_player_stats(config, body.player_ids, mode=body.mode)
    except SandboxRosterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"players": df_to_records(view), "glossary": ROSTER_STAT_GLOSSARY}


class LeagueRequest(BaseModel):
    player_ids: List[int] = Field(min_length=1)


@router.post("/league")
def post_league(body: LeagueRequest):
    """Liga completa de 30 equipos CON tu roster hipotético sustituido en
    el hueco de tu equipo (src/league_sandbox.py) -- decisión de diseño
    confirmada: los otros 29 equipos se simulan sin tocar, como si
    hubieras "tomado prestados" a los jugadores que uses de ahí. Tarda
    unos segundos (temporada regular + playoffs + premios en vivo, sin el
    cuello de botella real de recalcular los 577 jugadores desde cero --
    ver el docstring de league_sandbox.py), así que el frontend debe
    mostrar un estado de carga, pero no hace falta una cola en
    background. Devuelve standings/playoffs/awards en la MISMA forma que
    /api/league/standings, /api/league/playoffs y /api/awards -- para que
    el frontend reutilice el mismo código de render con un simple `if`."""
    config = load_config()
    try:
        league_result = simulate_hypothetical_league(config, body.player_ids)
    except SandboxRosterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    regular = league_result["regular_season_df"]
    playoff = league_result["playoff_df"]
    standings = compute_conference_standings(regular, playoff)

    my_team_abbrev = config["team"]["abbreviation"]
    my_team_row = playoff[playoff["team_abbreviation"] == my_team_abbrev]

    awards_summary = compute_hypothetical_awards(config, league_result)
    all_star_final = awards_summary.get("all_star_final")
    commissioner_picks = []
    if all_star_final is not None and "commissioner_pick" in all_star_final.columns:
        commissioner_picks = df_to_records(all_star_final[all_star_final["commissioner_pick"]])

    return {
        "standings": {
            "east": df_to_records(standings["East"]),
            "west": df_to_records(standings["West"]),
            "team_ids": ABBREVIATION_TO_TEAM_ID,
        },
        "playoffs": {
            "teams": df_to_records(playoff),
            "my_team": df_to_records(my_team_row)[0] if not my_team_row.empty else None,
            "glossary": LEAGUE_GLOSSARY,
            "team_ids": ABBREVIATION_TO_TEAM_ID,
        },
        "awards": {
            "scope": awards_summary["scope"],
            **{key: df_to_records(awards_summary.get(key)) for key in AWARD_DF_KEYS},
            "all_star_nationality_quota": awards_summary.get("all_star_nationality_quota"),
            "commissioner_picks": commissioner_picks,
            "glossary": AWARDS_GLOSSARY,
            "season_awards_glossary": SEASON_AWARDS_GLOSSARY,
            "team_ids": ABBREVIATION_TO_TEAM_ID,
        },
    }
