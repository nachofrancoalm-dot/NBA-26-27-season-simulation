"""
Tests de POST /api/sandbox/league -- separado de test_webapp_api.py porque
necesita una abreviatura de equipo REAL (TEAM_CONFERENCE la resuelve),
a diferencia del resto de la suite de la API que usa "TST" como equipo
de prueba genérico.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from league_simulation import TEAM_CONFERENCE  # noqa: E402

from webapp.main import app  # noqa: E402
from webapp.routers import sandbox as sandbox_router  # noqa: E402

OTHER_TEAMS = [abbrev for abbrev in TEAM_CONFERENCE if abbrev != "PHI"]


def _player_row(player_id, name, team_abbrev, game_score_per36, mpg, team_id):
    return {
        "player_id": player_id,
        "player_name": name,
        "team_id": team_id,
        "team_abbreviation": team_abbrev,
        "conference": TEAM_CONFERENCE[team_abbrev],
        "position": "G",
        "current_age": 25,
        "country": "USA",
        "game_score_per36": game_score_per36,
        "minutes_projection": mpg,
        "minutes_per_game_last_season": mpg,
        "risk_score": 0.2,
        "fatigue_score": 0.2,
        "games_played_last_season": 70,
        "projected_total_minutes": mpg * 82,
        "PTS_per36_projected": game_score_per36,
        "AST_per36_projected": 4.0,
        "REB_per36_projected": 5.0,
        "STL_per36_projected": 1.0,
        "BLK_per36_projected": 0.5,
        "DREB_per36_projected": 4.0,
        "TOV_per36_projected": 2.0,
        "PF_per36_projected": 2.0,
        "FG3M_per36_projected": 1.5,
        "PTS_projected": game_score_per36 * mpg / 36.0 * 82,
        "PPG": game_score_per36 * mpg / 36.0,
        "FG%": 45.0,
        "3P%": 35.0,
    }


@pytest.fixture
def league_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)

    rows = []
    next_id = 1
    hypothetical_player_ids = []
    for team_index, abbrev in enumerate(OTHER_TEAMS):
        team_id = 1000 + team_index
        for slot in range(2):
            player_id = next_id
            next_id += 1
            rows.append(_player_row(player_id, f"Player {player_id}", abbrev, 12.0 + slot, 28.0 - slot * 4, team_id))
            if len(hypothetical_player_ids) < 5:
                hypothetical_player_ids.append(player_id)

    pd.DataFrame(rows).to_csv(processed / "league_player_projections.csv", index=False)
    pd.DataFrame(columns=["player_id", "SEASON_ID", "GP", "MIN"]).to_csv(
        processed / "league_player_career_stats.csv", index=False
    )

    config = {
        "team": {"team_id": 9999, "abbreviation": "PHI", "name": "Hypothetical PHI", "season": "2026-27"},
        "simulation": {"games_per_season": 82, "n_seasons": 20, "random_seed": 1},
        "league_simulation": {"n_seasons": 20, "n_playoff_seasons": 20},
        "monte_carlo": {},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }
    return config, hypothetical_player_ids


@pytest.fixture
def client(league_config, monkeypatch):
    config, _ = league_config
    monkeypatch.setattr(sandbox_router, "load_config", lambda: config)
    return TestClient(app)


def test_sandbox_league_returns_standings_playoffs_and_awards(client, league_config):
    _, player_ids = league_config
    response = client.post("/api/sandbox/league", json={"player_ids": player_ids})
    assert response.status_code == 200
    body = response.json()

    assert "east" in body["standings"] and "west" in body["standings"]
    assert len(body["standings"]["east"]) + len(body["standings"]["west"]) == 30
    assert body["playoffs"]["my_team"]["team_abbreviation"] == "PHI"
    assert body["awards"]["scope"] == "league"
    assert "mvp" in body["awards"]


def test_sandbox_league_400_for_invalid_roster(client):
    response = client.post("/api/sandbox/league", json={"player_ids": [1, 2]})
    assert response.status_code == 400
