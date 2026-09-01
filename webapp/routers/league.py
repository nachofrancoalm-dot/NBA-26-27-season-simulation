"""
routers/league.py

Endpoints de la pestaña "Liga NBA -> Liga y Playoffs". team_id se adjunta
con ABBREVIATION_TO_TEAM_ID para que el frontend arme URLs de logo.

Todos los endpoints de lectura aceptan `scenario` ("with_injuries" por
defecto, o "no_injuries"). Solo POST /simulate DISPARA la simulación de
un escenario; los demás leen el CSV ya calculado.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from dashboard.data_loader import (  # noqa: E402 -- importar primero inserta src/ en sys.path
    LEAGUE_GLOSSARY,
    LEAGUE_PLAYER_META_COLUMNS,
    ROSTER_STAT_GLOSSARY,
    compute_conference_standings,
    load_league_player_projections,
    load_league_playoff_summary,
    load_league_regular_season_summary,
    load_league_single_season_game_log,
    load_league_single_season_player_box_scores,
    run_single_bracket_simulation,
    run_single_league_season_simulation,
    select_roster_view,
)
from config_loader import load_config  # noqa: E402
from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID  # noqa: E402
from league_simulation import build_league_simulation_dataset, compute_head_to_head_record  # noqa: E402

from webapp.serializers import df_to_records

router = APIRouter(prefix="/league")

SCENARIO_QUERY = Query("with_injuries", pattern="^(with_injuries|no_injuries)$")

LEAGUE_MISSING_DETAIL = (
    "No se encontraron los CSV de liga completa para este escenario. Corre "
    "`python src/data_pipeline.py --league` y luego, para el escenario 'con lesiones', "
    "`build_league_simulation_dataset`; para 'sin lesiones', el botón 'Simular sin lesiones' "
    "de la interfaz (POST /api/league/simulate?scenario=no_injuries)."
)


def _require_league_data(config, scenario: str):
    regular = load_league_regular_season_summary(config, scenario=scenario)
    playoff = load_league_playoff_summary(config, scenario=scenario)
    if regular is None or playoff is None:
        raise HTTPException(status_code=404, detail=LEAGUE_MISSING_DETAIL)
    return regular, playoff


@router.post("/simulate")
def post_simulate(scenario: str = SCENARIO_QUERY):
    """Dispara una corrida COMPLETA (temporada regular + playoffs +
    jugadores) para `scenario` y persiste los 3 CSV. Tarda del orden de
    un minuto -- el frontend debe mostrar un estado de carga largo."""
    config = load_config()
    try:
        result = build_league_simulation_dataset(config, scenario=scenario)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"scenario": scenario, "teams": len(result["regular_season"])}


@router.get("/standings")
def get_standings(scenario: str = SCENARIO_QUERY):
    config = load_config()
    regular, playoff = _require_league_data(config, scenario)
    standings = compute_conference_standings(regular, playoff)
    return {
        "east": df_to_records(standings["East"]),
        "west": df_to_records(standings["West"]),
        "team_ids": ABBREVIATION_TO_TEAM_ID,
    }


@router.get("/playoffs")
def get_playoffs(scenario: str = SCENARIO_QUERY):
    config = load_config()
    _, playoff = _require_league_data(config, scenario)
    my_team_abbrev = config["team"].get("abbreviation")
    my_team_row = playoff[playoff["team_abbreviation"] == my_team_abbrev]
    return {
        "teams": df_to_records(playoff),
        "my_team": df_to_records(my_team_row)[0] if not my_team_row.empty else None,
        "glossary": LEAGUE_GLOSSARY,
        "team_ids": ABBREVIATION_TO_TEAM_ID,
    }


@router.get("/teams")
def get_teams(scenario: str = SCENARIO_QUERY):
    config = load_config()
    regular, _ = _require_league_data(config, scenario)
    return {"teams": sorted(regular["team_abbreviation"].unique().tolist())}


@router.get("/team/{abbreviation}")
def get_team(
    abbreviation: str,
    mode: str = Query("per_game", pattern="^(per_game|totals)$"),
    scenario: str = SCENARIO_QUERY,
):
    config = load_config()
    regular, playoff = _require_league_data(config, scenario)
    players = load_league_player_projections(config, scenario=scenario)

    team_regular = regular[regular["team_abbreviation"] == abbreviation]
    team_playoff = playoff[playoff["team_abbreviation"] == abbreviation]
    if team_regular.empty or team_playoff.empty:
        raise HTTPException(status_code=404, detail=f"Equipo '{abbreviation}' no encontrado.")

    # Seed y conferencia recalculados aquí (no dependen de /standings).
    conference = team_regular["conference"].iloc[0] if "conference" in team_regular.columns else None
    seed = None
    if conference:
        standings = compute_conference_standings(regular, playoff)
        conf_rows = standings.get(conference)
        if conf_rows is not None:
            match = conf_rows[conf_rows["team_abbreviation"] == abbreviation]
            if not match.empty:
                seed = int(match["seed"].iloc[0])

    response = {
        "team_id": ABBREVIATION_TO_TEAM_ID.get(abbreviation),
        "conference": conference,
        "seed": seed,
        "regular": df_to_records(team_regular)[0],
        "playoff": df_to_records(team_playoff)[0],
        "players": [],
        "glossary": ROSTER_STAT_GLOSSARY,
    }

    if players is not None:
        team_players = players[players["team_abbreviation"] == abbreviation]
        view = select_roster_view(
            team_players,
            mode=mode,
            meta_columns=LEAGUE_PLAYER_META_COLUMNS,
            games_per_season=config["simulation"]["games_per_season"],
        )
        # player_id se adjunta ANTES del sort (select_roster_view preserva
        # el orden de entrada) para que viaje con la fila correcta.
        view = view.assign(player_id=team_players["player_id"].to_numpy())
        response["players"] = df_to_records(view.sort_values("game_score_per36", ascending=False))

    return response


@router.post("/bracket")
def post_bracket(scenario: str = SCENARIO_QUERY):
    config = load_config()
    _require_league_data(config, scenario)
    result = run_single_bracket_simulation(config, scenario=scenario)
    result["team_ids"] = ABBREVIATION_TO_TEAM_ID
    return result


SEASON_LOG_MISSING_DETAIL = (
    "Todavía no se ha simulado un calendario concreto para este escenario. Pulsa "
    "'Simular calendario de la temporada' (POST /api/league/simulate-season-log)."
)


def _require_season_log(config, scenario: str) -> pd.DataFrame:
    game_log = load_league_single_season_game_log(config, scenario=scenario)
    if game_log is None:
        raise HTTPException(status_code=404, detail=SEASON_LOG_MISSING_DETAIL)
    return game_log


@router.post("/simulate-season-log")
def post_simulate_season_log(scenario: str = SCENARIO_QUERY):
    """Simula UNA temporada regular concreta (calendario + resultado por
    partido + boxscore) y persiste los CSV. Tirada distinta cada vez
    (seed por reloj); no toca standings/playoffs/premios."""
    config = load_config()
    _require_league_data(config, scenario)  # necesita los 30 equipos ya proyectados/descargados
    result = run_single_league_season_simulation(config, scenario=scenario)
    return {"scenario": scenario, "games": len(result["game_log"])}


@router.get("/schedule")
def get_schedule(team: Optional[str] = None, scenario: str = SCENARIO_QUERY):
    """Calendario de la temporada concreta ya simulada, opcionalmente
    filtrado a los partidos (local o visitante) de un equipo."""
    config = load_config()
    game_log = _require_season_log(config, scenario)
    if team:
        game_log = game_log[(game_log["home_abbreviation"] == team) | (game_log["away_abbreviation"] == team)]
    return {"games": df_to_records(game_log.sort_values("day"))}


@router.get("/boxscore/{game_id}")
def get_boxscore(game_id: int, scenario: str = SCENARIO_QUERY):
    config = load_config()
    game_log = _require_season_log(config, scenario)
    box_scores = load_league_single_season_player_box_scores(config, scenario=scenario)

    game_row = game_log[game_log["game_id"] == game_id]
    if game_row.empty:
        raise HTTPException(status_code=404, detail=f"No existe el partido game_id={game_id} en este calendario.")
    game = df_to_records(game_row)[0]

    home_players, away_players = [], []
    if box_scores is not None:
        game_box = box_scores[box_scores["game_id"] == game_id]
        home_players = df_to_records(
            game_box[game_box["team_id"] == game["home_team_id"]].sort_values("PTS", ascending=False)
        )
        away_players = df_to_records(
            game_box[game_box["team_id"] == game["away_team_id"]].sort_values("PTS", ascending=False)
        )

    return {"game": game, "home_players": home_players, "away_players": away_players}


@router.get("/head-to-head")
def get_head_to_head(team_a: str, team_b: str, scenario: str = SCENARIO_QUERY):
    config = load_config()
    game_log = _require_season_log(config, scenario)

    team_a_id = ABBREVIATION_TO_TEAM_ID.get(team_a)
    team_b_id = ABBREVIATION_TO_TEAM_ID.get(team_b)
    if team_a_id is None or team_b_id is None:
        raise HTTPException(status_code=404, detail=f"Abreviación de equipo desconocida: '{team_a}' o '{team_b}'.")

    record = compute_head_to_head_record(game_log, team_a_id, team_b_id)
    return {
        "team_a": team_a, "team_b": team_b,
        "team_a_wins": record["team_a_wins"], "team_b_wins": record["team_b_wins"],
        "games": df_to_records(pd.DataFrame(record["games"])),
    }
