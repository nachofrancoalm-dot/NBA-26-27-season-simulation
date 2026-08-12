"""
routers/league.py

Endpoints de la pestaña "Liga NBA -> Liga y Playoffs". Reutiliza
dashboard/data_loader.py sin reimplementar transformaciones. team_id se
adjunta con ABBREVIATION_TO_TEAM_ID (tabla estática ya existente en
src/context/opponent_weighting.py) para que el frontend arme URLs de
logo con el mismo mecanismo que ui.js ya usa desde la Fase 1.

Todos los endpoints de lectura aceptan `scenario` ("with_injuries" por
defecto, o "no_injuries") -- ver league_simulation._apply_scenario. El
único punto donde se DISPARA la simulación de un escenario es
POST /simulate; los demás solo leen el CSV correspondiente, ya
calculado.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from dashboard.data_loader import (  # noqa: E402 -- importar primero inserta src/ en sys.path
    LEAGUE_GLOSSARY,
    LEAGUE_PLAYER_META_COLUMNS,
    ROSTER_STAT_GLOSSARY,
    compute_conference_standings,
    load_league_player_projections,
    load_league_playoff_summary,
    load_league_regular_season_summary,
    run_single_bracket_simulation,
    select_roster_view,
)
from config_loader import load_config  # noqa: E402
from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID  # noqa: E402
from league_simulation import build_league_simulation_dataset  # noqa: E402

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
    """
    Dispara una corrida COMPLETA (temporada regular + playoffs + tabla de
    jugadores) para `scenario` y persiste los 3 CSV correspondientes
    (con sufijo `_no_injuries` si aplica). Es lo que disparan los dos
    botones "Simular con/sin lesiones" -- tarda del orden de un minuto o
    más (no es instantáneo, el frontend debe mostrar un estado de carga
    largo).
    """
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

    # Seed y conferencia -- misma clasificación que /league/standings, se
    # recalcula aquí para no depender de que el frontend la haya pedido
    # antes de abrir el popup de equipo.
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
        # player_id se adjunta ANTES del sort_values de abajo (no después)
        # para que viaje con la fila correcta -- select_roster_view
        # preserva el orden de entrada, pero aquí se reordena a
        # continuación por game_score_per36.
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
