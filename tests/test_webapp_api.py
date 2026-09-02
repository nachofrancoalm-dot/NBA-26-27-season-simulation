"""Tests de la API del frontend web (webapp/) con config + CSV sintéticos; load_config() se monkeypatchea por router."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from webapp.main import app
from webapp.routers import awards as awards_router
from webapp.routers import champions as champions_router
from webapp.routers import explainer as explainer_router
from webapp.routers import league as league_router
from webapp.routers import players as players_router
from webapp.routers import sandbox as sandbox_router
from webapp.routers import status as status_router
from webapp.routers import team as team_router


@pytest.fixture
def config(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    return {
        "team": {"name": "Test Team", "abbreviation": "TST", "season": "2026-27", "team_id": 1610612755},
        "roster": [
            {"player_id": 1, "name": "Player One", "role_expected": "scorer", "minutes_projection": 34, "unit": "starter"},
        ],
        "simulation": {"games_per_season": 82, "n_seasons": 50, "random_seed": 1},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed_dir)},
    }


@pytest.fixture
def client(config, monkeypatch):
    monkeypatch.setattr(status_router, "load_config", lambda: config)
    monkeypatch.setattr(team_router, "load_config", lambda: config)
    monkeypatch.setattr(league_router, "load_config", lambda: config)
    monkeypatch.setattr(awards_router, "load_config", lambda: config)
    monkeypatch.setattr(champions_router, "load_config", lambda: config)
    monkeypatch.setattr(explainer_router, "load_config", lambda: config)
    monkeypatch.setattr(players_router, "load_config", lambda: config)
    monkeypatch.setattr(sandbox_router, "load_config", lambda: config)
    return TestClient(app)


def test_status_reports_missing_datasets_when_no_csv_exists(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["team"]["abbreviation"] == "TST"
    assert body["datasets"] == {
        "roster": False,
        "simulation": False,
        "synergy": False,
        "backtest": False,
        "backtest_sweep": False,
        "league": False,
        "league_no_injuries": False,
        "champions": False,
        "explainer": False,
    }


def test_roster_endpoint_404_when_aging_csv_missing(client):
    response = client.get("/api/roster")
    assert response.status_code == 404
    assert "aging_curve_projection.csv" in response.json()["detail"]


def test_roster_endpoint_returns_json_serializable_players(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Player One",
                "current_age": 27,
                "target_age": 28,
                "game_score_per36": 18.0,
                "PTS_projected": 1600,
                "AST_projected": 400,
                "REB_projected": 500,
                "FG3M_projected": 150,
                "FGM_projected": 600,
                "FGA_projected": 1200,
                "FG3A_projected": 400,
            }
        ]
    ).to_csv(processed_dir / "aging_curve_projection.csv", index=False)

    response = client.get("/api/roster?mode=per_game")
    assert response.status_code == 200
    body = response.json()
    assert "NaN" not in response.text
    assert body["players"][0]["player_name"] == "Player One"


def tmp_path_from_config(config):
    from pathlib import Path

    return Path(config["paths"]["processed_data_dir"])


def test_simulation_endpoint_returns_histogram_and_glossary(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        {
            "wins": [40, 41, 41, 45, 50],
            "losses": [42, 41, 41, 37, 32],
            "net_rating_estimate_mean": [-1.0, 0.5, 0.5, 2.0, 5.0],
            "total_games_missed": [10, 8, 8, 5, 2],
        }
    ).to_csv(processed_dir / "simulation_results.csv", index=False)

    response = client.get("/api/simulation")
    assert response.status_code == 200
    body = response.json()
    assert body["n_seasons"] == 5
    assert body["wins_histogram"]["41"] == 2
    assert len(body["net_rating_sorted"]) == 5
    assert "wins" in body["glossary"]


def test_simulation_no_injuries_404_when_missing_inputs(client):
    response = client.post("/api/simulation/no-injuries")
    assert response.status_code == 404


def test_simulation_no_injuries_ignores_cached_risk_score(client, config):
    """/no-injuries debe simular con risk_score=0 real, ignorando el 0.5 cacheado en injury_risk.csv."""
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [{
            "player_id": 1, "player_name": "Player One", "current_age": 27, "target_age": 28,
            "game_score_per36": 20.0,
            "FGA_per36_projected": 18.0, "FTA_per36_projected": 4.0, "TOV_per36_projected": 2.5,
            "AST_per36_projected": 5.0, "FG3A_per36_projected": 3.0,
            "BLK_per36_projected": 0.5, "DREB_per36_projected": 4.0,
        }]
    ).to_csv(processed_dir / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "risk_score": 0.5}]).to_csv(processed_dir / "injury_risk.csv", index=False)
    pd.DataFrame([{"player_id": 1, "fatigue_score": 0.3}]).to_csv(processed_dir / "fatigue_risk.csv", index=False)
    pd.DataFrame({"WinPCT": [0.3, 0.4, 0.5, 0.6, 0.7]}).to_csv(processed_dir / "prior_season_standings.csv", index=False)

    response = client.post("/api/simulation/no-injuries")
    assert response.status_code == 200
    body = response.json()
    assert "NaN" not in response.text
    assert body["n_seasons"] == 50

    # Determinista: mismo config y seed -> mismo resultado en dos llamadas seguidas.
    again = client.post("/api/simulation/no-injuries").json()
    assert again["summary"]["mean"] == body["summary"]["mean"]

    assert body["summary"]["mean"] > 0


def test_backtest_sweep_endpoint_computes_percentile_histogram(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [{
            "n_cases": 2,
            "pct_within_p10_p90": 80.0,
            "mean_percentile": 50.0,
            "median_percentile": 50.0,
            "mean_absolute_error_wins": 5.0,
            "mean_error_wins": 0.0,
            "correlation_actual_vs_predicted": 0.9,
        }]
    ).to_csv(processed_dir / "backtest_sweep_calibration.csv", index=False)
    pd.DataFrame(
        [
            {"comparable_name": "A 2020-21", "actual_wins": 40, "simulated_wins_mean": 41.0, "actual_percentile": 5},
            {"comparable_name": "B 2020-21", "actual_wins": 50, "simulated_wins_mean": 48.0, "actual_percentile": 95},
        ]
    ).to_csv(processed_dir / "backtest_sweep_summary.csv", index=False)

    response = client.get("/api/backtest/sweep")
    assert response.status_code == 200
    body = response.json()
    assert body["calibration"]["n_cases"] == 2
    assert sum(body["percentile_histogram"].values()) == 2
    assert len(body["scatter"]) == 2


# --- Fase 2: Liga NBA, Premios, Campeones, Explicador ---


LEAGUE_REGULAR_ROWS = [
    {"team_id": 1, "team_abbreviation": "AAA", "conference": "East", "wins_mean": 55.0, "wins_p10": 48.0, "wins_p90": 60.0},
    {"team_id": 2, "team_abbreviation": "BBB", "conference": "West", "wins_mean": 30.0, "wins_p10": 24.0, "wins_p90": 36.0},
]
LEAGUE_PLAYOFF_ROWS = [
    {"team_abbreviation": "AAA", "playoff_pct": 95.0, "conf_semis_pct": 70.0, "conf_finals_pct": 40.0, "finals_pct": 20.0, "championship_pct": 10.0},
    {"team_abbreviation": "BBB", "playoff_pct": 15.0, "conf_semis_pct": 5.0, "conf_finals_pct": 1.0, "finals_pct": 0.0, "championship_pct": 0.0},
]


def _write_league_csvs(processed_dir):
    pd.DataFrame(LEAGUE_REGULAR_ROWS).to_csv(processed_dir / "league_regular_season_summary.csv", index=False)
    pd.DataFrame(LEAGUE_PLAYOFF_ROWS).to_csv(processed_dir / "league_playoff_summary.csv", index=False)


def test_league_standings_404_when_missing(client):
    response = client.get("/api/league/standings")
    assert response.status_code == 404
    assert "league" in response.json()["detail"].lower()


def test_league_simulate_404_without_league_rosters(client):
    # Sin league_rosters.csv / league_player_career_stats.csv, debe degradar a 404, no a un 500.
    response = client.post("/api/league/simulate?scenario=no_injuries")
    assert response.status_code == 404


def test_league_standings_no_injuries_404_when_scenario_not_simulated_yet(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_league_csvs(processed_dir)  # solo el escenario "with_injuries" (sin sufijo)

    response = client.get("/api/league/standings?scenario=no_injuries")
    assert response.status_code == 404

    # El escenario por defecto sigue funcionando sin pasar el parámetro.
    default_response = client.get("/api/league/standings")
    assert default_response.status_code == 200


def test_league_standings_reads_no_injuries_scenario_once_simulated(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_league_csvs(processed_dir)
    pd.DataFrame(
        [{"team_id": 1, "team_abbreviation": "AAA", "conference": "East", "wins_mean": 60.0, "wins_p10": 55.0, "wins_p90": 65.0}]
    ).to_csv(processed_dir / "league_regular_season_summary_no_injuries.csv", index=False)
    pd.DataFrame(
        [{"team_abbreviation": "AAA", "playoff_pct": 99.0, "conf_semis_pct": 90.0, "conf_finals_pct": 60.0, "finals_pct": 30.0, "championship_pct": 15.0}]
    ).to_csv(processed_dir / "league_playoff_summary_no_injuries.csv", index=False)

    response = client.get("/api/league/standings?scenario=no_injuries")
    assert response.status_code == 200
    body = response.json()
    assert body["east"][0]["wins_mean"] == 60.0

    # El escenario "with_injuries" (los CSV sin sufijo) no se ve afectado.
    normal = client.get("/api/league/standings").json()
    assert normal["east"][0]["wins_mean"] == 55.0


def test_league_standings_and_playoffs_return_data(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_league_csvs(processed_dir)

    standings = client.get("/api/league/standings")
    assert standings.status_code == 200
    body = standings.json()
    assert body["east"][0]["team_abbreviation"] == "AAA"
    assert body["west"][0]["team_abbreviation"] == "BBB"
    # team_ids es la tabla estática real de las 30 franquicias (ABBREVIATION_TO_TEAM_ID),
    # no se deriva de los datos sintéticos -- solo confirmamos que viaja en la respuesta.
    assert body["team_ids"]["BOS"] == 1610612738

    playoffs = client.get("/api/league/playoffs")
    assert playoffs.status_code == 200
    assert len(playoffs.json()["teams"]) == 2

    teams = client.get("/api/league/teams")
    assert teams.status_code == 200
    assert teams.json()["teams"] == ["AAA", "BBB"]

    team_detail = client.get("/api/league/team/AAA")
    assert team_detail.status_code == 200
    body = team_detail.json()
    assert body["regular"]["wins_mean"] == 55.0
    # AAA es el único equipo del Este sintético -> seed 1 de esa conferencia.
    assert body["conference"] == "East"
    assert body["seed"] == 1


def test_league_team_404_for_unknown_abbreviation(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_league_csvs(processed_dir)
    response = client.get("/api/league/team/ZZZ")
    assert response.status_code == 404


def test_league_leaders_returns_all_players_with_filter_columns(client, config):
    processed_dir = tmp_path_from_config(config)
    # PPG/RPG/ORPG/... vienen YA calculadas en el CSV real (ver
    # league_simulation.build_league_simulation_dataset) -- select_roster_view
    # no las deriva de los totales, así que el fixture las incluye directamente.
    pd.DataFrame(
        [
            {
                "player_id": 1, "player_name": "Big Man", "team_abbreviation": "AAA", "conference": "East",
                "position": "C", "country": "USA", "current_age": 25, "game_score_per36": 20.0,
                "minutes_projection": 32, "minutes_per_game_last_season": 32, "games_played_last_season": 70,
                "risk_score": 0.2, "fatigue_score": 0.2,
                "PTS_projected": 1500, "REB_projected": 900, "OREB_projected": 300, "DREB_projected": 600,
                "AST_projected": 200, "STL_projected": 60, "BLK_projected": 150, "TOV_projected": 100,
                "FG3M_projected": 10,
                "PPG": 18.3, "RPG": 11.0, "ORPG": 300 / 82, "DRPG": 600 / 82,
                "APG": 2.4, "SPG": 0.7, "BPG": 1.8, "TOPG": 1.2, "3PM": 0.1,
            },
            {
                "player_id": 2, "player_name": "Sharpshooter", "team_abbreviation": "BBB", "conference": "West",
                "position": "G", "country": "France", "current_age": 27, "game_score_per36": 15.0,
                "minutes_projection": 30, "minutes_per_game_last_season": 30, "games_played_last_season": 75,
                "risk_score": 0.1, "fatigue_score": 0.1,
                "PTS_projected": 1800, "REB_projected": 250, "OREB_projected": 30, "DREB_projected": 220,
                "AST_projected": 400, "STL_projected": 80, "BLK_projected": 20, "TOV_projected": 150,
                "FG3M_projected": 200,
                "PPG": 22.0, "RPG": 3.0, "ORPG": 30 / 82, "DRPG": 220 / 82,
                "APG": 4.9, "SPG": 1.0, "BPG": 0.2, "TOPG": 1.8, "3PM": 2.4,
            },
        ]
    ).to_csv(processed_dir / "league_player_projections.csv", index=False)

    response = client.get("/api/league/leaders")
    assert response.status_code == 200
    body = response.json()
    assert len(body["players"]) == 2
    row = next(p for p in body["players"] if p["player_name"] == "Big Man")
    assert row["team_abbreviation"] == "AAA"
    assert row["position"] == "C"
    assert row["country"] == "USA"
    # select_roster_view redondea a 1 decimal en modo "per_game".
    assert row["ORPG"] == pytest.approx(300 / 82, abs=0.05)

    totals = client.get("/api/league/leaders?mode=totals").json()
    row_totals = next(p for p in totals["players"] if p["player_name"] == "Big Man")
    # Los TOTALES se escalan por (1 - risk_score) -- ver
    # _apply_simulated_games_and_minutes -- risk_score=0.2 aquí.
    assert row_totals["OREB"] == 300 * 0.8
    assert row_totals["DREB"] == 600 * 0.8


def test_league_leaders_404_when_no_league_data(client, config):
    response = client.get("/api/league/leaders")
    assert response.status_code == 404


def test_awards_404_when_no_projection_available(client):
    response = client.get("/api/awards")
    assert response.status_code == 404


def test_champions_404_when_missing(client):
    response = client.get("/api/champions")
    assert response.status_code == 404


def test_champions_returns_seed_comparison(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [
            {"season": "2020-21", "team_abbreviation": "AAA", "seed": 1, "regular_season_wins": 60,
             "playoff_games": 20, "playoff_wins": 16, "opponents_faced": "BBB", "seeds_beaten": "4"},
        ]
    ).to_csv(processed_dir / "champion_title_paths.csv", index=False)

    response = client.get("/api/champions")
    assert response.status_code == 200
    body = response.json()
    assert body["n_seasons"] == 1
    assert body["most_titles"]["team_abbreviation"] == "AAA"
    assert body["seed_comparison"][0]["seed"] == 1
    assert body["seed_comparison"][0]["real_pct"] == 100.0


def test_explainer_context_returns_snapshot_text(client):
    response = client.get("/api/explainer/context")
    assert response.status_code == 200
    assert "snapshot" in response.json()


def test_explainer_ask_503_without_api_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.post("/api/explainer/ask", json={"question": "hola"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_explainer_ask_forwards_pasted_news_text(client, monkeypatch):
    # news_text es opcional; el router debe reenviarlo a explain_question tal cual.
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    captured = {}

    def _fake_explain_question(question, config, api_key=None, news_text=None):
        captured["question"] = question
        captured["news_text"] = news_text
        return "respuesta"

    monkeypatch.setattr(explainer_router, "explain_question", _fake_explain_question)

    response = client.post(
        "/api/explainer/ask",
        json={"question": "¿Que le paso a Embiid?", "news_text": "Embiid se lesiono la rodilla."},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "respuesta"}
    assert captured["news_text"] == "Embiid se lesiono la rodilla."


def test_explainer_search_news_503_without_api_key(client, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    response = client.get("/api/explainer/search-news", params={"query": "lesiones 76ers"})
    assert response.status_code == 503
    assert "TAVILY_API_KEY" in response.json()["detail"]


def test_explainer_search_news_forwards_query_and_returns_text(client, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    captured = {}

    def _fake_search(query, api_key=None):
        captured["query"] = query
        captured["api_key"] = api_key
        return "Embiid se lesiono la rodilla."

    monkeypatch.setattr(explainer_router, "search_recent_news", _fake_search)

    response = client.get("/api/explainer/search-news", params={"query": "lesiones 76ers"})

    assert response.status_code == 200
    assert response.json() == {"news_text": "Embiid se lesiono la rodilla."}
    assert captured["query"] == "lesiones 76ers"
    assert captured["api_key"] == "test-key"


def test_explainer_search_news_502_on_search_failure(client, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    def _fake_search(query, api_key=None):
        raise RuntimeError("timeout")

    monkeypatch.setattr(explainer_router, "search_recent_news", _fake_search)

    response = client.get("/api/explainer/search-news", params={"query": "lesiones 76ers"})

    assert response.status_code == 502
    assert "timeout" in response.json()["detail"]


# --- Popup de detalle de jugador (doble clic en el nombre) ---


def test_player_detail_404_without_any_season_data(client):
    response = client.get("/api/player/42")
    assert response.status_code == 404


def test_player_detail_returns_seasons_without_bio_when_cache_missing(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [
            {
                "PLAYER_ID": 42, "SEASON_ID": "2023-24", "TEAM_ABBREVIATION": "TST",
                "GP": 70, "MIN": 2000, "PTS": 1200, "REB": 400, "AST": 300,
                "STL": 60, "BLK": 30, "FG_PCT": 0.5, "FG3_PCT": 0.35, "FT_PCT": 0.8,
                "player_name": "Test Player",
            },
            {
                "PLAYER_ID": 42, "SEASON_ID": "2024-25", "TEAM_ABBREVIATION": "TST",
                "GP": 72, "MIN": 2100, "PTS": 1300, "REB": 420, "AST": 320,
                "STL": 65, "BLK": 32, "FG_PCT": 0.51, "FG3_PCT": 0.36, "FT_PCT": 0.81,
                "player_name": "Test Player",
            },
        ]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    response = client.get("/api/player/42")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Test Player"
    assert body["bio"] is None
    assert body["qualities"] == []
    assert [s["SEASON_ID"] for s in body["seasons"]] == ["2023-24", "2024-25"]


def test_player_detail_reads_cached_bio_without_any_network_call(client, config, monkeypatch):
    processed_dir = tmp_path_from_config(config)
    raw_dir = Path(config["paths"]["raw_data_dir"])
    (raw_dir / "player_common_info").mkdir(parents=True)

    pd.DataFrame(
        [{
            "PLAYER_ID": 42, "SEASON_ID": "2023-24", "TEAM_ABBREVIATION": "TST",
            "GP": 70, "MIN": 2000, "PTS": 1200, "REB": 400, "AST": 300,
            "STL": 60, "BLK": 30, "FG_PCT": 0.5, "FG3_PCT": 0.35, "FT_PCT": 0.8,
            "player_name": "Test Player",
        }]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    pd.DataFrame(
        [{
            "POSITION": "Guard", "HEIGHT": "6-4", "WEIGHT": 210, "BIRTHDATE": "1998-05-10T00:00:00",
            "COUNTRY": "USA", "SCHOOL": "Duke", "DRAFT_YEAR": "2019", "DRAFT_ROUND": "1", "DRAFT_NUMBER": "12",
        }]
    ).to_csv(raw_dir / "player_common_info" / "42.csv", index=False)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("El endpoint no debe llamar a fetch_player_common_info (llamada de red)")

    monkeypatch.setattr(players_router, "fetch_player_common_info", _fail_if_called, raising=False)

    response = client.get("/api/player/42")
    assert response.status_code == 200
    bio = response.json()["bio"]
    assert bio["position"] == "Guard"
    assert bio["height"] == "6-4"
    assert bio["weight"] == 210
    assert bio["country"] == "USA"
    assert bio["draft"] == "2019 · ronda 1 · pick 12"
    assert bio["age"] is not None


def test_player_detail_derives_qualities_from_projection(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [{
            "PLAYER_ID": 42, "SEASON_ID": "2023-24", "TEAM_ABBREVIATION": "TST",
            "GP": 70, "MIN": 2000, "PTS": 1200, "REB": 400, "AST": 300,
            "STL": 60, "BLK": 30, "FG_PCT": 0.5, "FG3_PCT": 0.35, "FT_PCT": 0.8,
            "player_name": "Test Player",
        }]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    pd.DataFrame(
        [{
            "player_id": 42, "player_name": "Test Player",
            "FGA_per36_projected": 20.0, "FTA_per36_projected": 5.0, "TOV_per36_projected": 3.0,
            "AST_per36_projected": 9.0, "FG3A_per36_projected": 2.0,
            "BLK_per36_projected": 0.5, "DREB_per36_projected": 4.0,
        }]
    ).to_csv(processed_dir / "aging_curve_projection.csv", index=False)

    response = client.get("/api/player/42")
    assert response.status_code == 200
    qualities = response.json()["qualities"]
    assert "Alto uso ofensivo" in qualities
    assert "Buen creador de juego" in qualities
    assert "Amenaza desde el triple" not in qualities


def test_player_detail_appends_projected_season_row(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [{
            "PLAYER_ID": 42, "SEASON_ID": "2023-24", "TEAM_ABBREVIATION": "TST",
            "GP": 70, "MIN": 2000, "PTS": 1200, "REB": 400, "AST": 300,
            "STL": 60, "BLK": 30, "FG_PCT": 0.5, "FG3_PCT": 0.35, "FT_PCT": 0.8,
            "player_name": "Test Player",
        }]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    pd.DataFrame(
        [{
            "player_id": 42, "player_name": "Test Player",
            "FGA_per36_projected": 20.0, "FTA_per36_projected": 5.0, "TOV_per36_projected": 3.0,
            "AST_per36_projected": 9.0, "FG3A_per36_projected": 2.0,
            "BLK_per36_projected": 0.5, "DREB_per36_projected": 4.0,
            "projected_total_minutes": 2200,
            "PTS_projected": 1500, "REB_projected": 450, "AST_projected": 350,
            "STL_projected": 70, "BLK_projected": 40,
            "FGM_projected": 550, "FGA_projected": 1100,
            "FG3M_projected": 100, "FG3A_projected": 250,
            "FTM_projected": 300, "FTA_projected": 350,
        }]
    ).to_csv(processed_dir / "aging_curve_projection.csv", index=False)

    response = client.get("/api/player/42")
    assert response.status_code == 200
    seasons = response.json()["seasons"]
    assert len(seasons) == 2
    assert seasons[0]["is_projection"] is False
    projected = seasons[-1]
    assert projected["is_projection"] is True
    assert projected["SEASON_ID"] == "2026-27 (proyección)"
    assert projected["GP"] == 82
    assert projected["PTS"] == 1500
    assert projected["TEAM_ABBREVIATION"] == "TST"


def test_player_detail_projection_discounts_injury_risk(client, config):
    """Regresión: la fila de proyección mostraba siempre 82 GP; debe usar la misma fórmula de disponibilidad que la columna GP."""
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [{
            "PLAYER_ID": 42, "SEASON_ID": "2023-24", "TEAM_ABBREVIATION": "TST",
            "GP": 70, "MIN": 2000, "PTS": 1200, "REB": 400, "AST": 300,
            "STL": 60, "BLK": 30, "FG_PCT": 0.5, "FG3_PCT": 0.35, "FT_PCT": 0.8,
            "player_name": "Test Player",
        }]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    pd.DataFrame(
        [{
            "player_id": 42, "player_name": "Test Player",
            "FGA_per36_projected": 20.0, "FTA_per36_projected": 5.0, "TOV_per36_projected": 3.0,
            "AST_per36_projected": 9.0, "FG3A_per36_projected": 2.0,
            "BLK_per36_projected": 0.5, "DREB_per36_projected": 4.0,
            "projected_total_minutes": 2000,
            "PTS_projected": 2000, "REB_projected": 500, "AST_projected": 400,
            "STL_projected": 100, "BLK_projected": 50,
            "FGM_projected": 700, "FGA_projected": 1400,
            "FG3M_projected": 150, "FG3A_projected": 400,
            "FTM_projected": 400, "FTA_projected": 450,
        }]
    ).to_csv(processed_dir / "aging_curve_projection.csv", index=False)

    pd.DataFrame([{"player_id": 42, "player_name": "Test Player", "risk_score": 0.6}]).to_csv(
        processed_dir / "injury_risk.csv", index=False
    )

    response = client.get("/api/player/42")
    assert response.status_code == 200
    projected = response.json()["seasons"][-1]
    # 82 * (1 - 0.6) = 32.8 -> 33
    assert projected["GP"] == 33
    assert projected["GP"] < 82
    # Totales también descontados por el mismo factor de disponibilidad.
    assert projected["PTS"] == round(2000 * (33 / 82))


def test_player_shot_chart_empty_without_csv(client):
    response = client.get("/api/player/42/shot-chart")
    assert response.status_code == 200
    body = response.json()
    assert body["shots"] == []
    assert body["season"] is None


def test_player_shot_chart_returns_shots_for_that_player_only(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [
            {"player_id": 42, "season": "2024-25", "loc_x": -10, "loc_y": 20, "shot_made": True, "shot_type": "2PT Field Goal"},
            {"player_id": 42, "season": "2024-25", "loc_x": 200, "loc_y": 230, "shot_made": False, "shot_type": "3PT Field Goal"},
            {"player_id": 99, "season": "2024-25", "loc_x": 0, "loc_y": 0, "shot_made": True, "shot_type": "2PT Field Goal"},
        ]
    ).to_csv(processed_dir / "roster_shot_charts.csv", index=False)

    response = client.get("/api/player/42/shot-chart")
    assert response.status_code == 200
    body = response.json()
    assert body["season"] == "2024-25"
    assert len(body["shots"]) == 2


def test_player_shot_chart_falls_back_to_league_csv_for_non_roster_players(client, config):
    processed_dir = tmp_path_from_config(config)
    # roster_shot_charts.csv existe pero no trae a este jugador (no es del
    # roster propio) -- debe caer a league_shot_charts.csv en vez de 404.
    pd.DataFrame(
        [{"player_id": 1, "season": "2024-25", "loc_x": 0, "loc_y": 0, "shot_made": True, "shot_type": "2PT Field Goal"}]
    ).to_csv(processed_dir / "roster_shot_charts.csv", index=False)
    pd.DataFrame(
        [{"player_id": 777, "season": "2025-26", "loc_x": 50, "loc_y": 60, "shot_made": True, "shot_type": "2PT Field Goal"}]
    ).to_csv(processed_dir / "league_shot_charts.csv", index=False)

    response = client.get("/api/player/777/shot-chart")
    assert response.status_code == 200
    body = response.json()
    assert body["season"] == "2025-26"
    assert len(body["shots"]) == 1

    # Un jugador que no está en NINGUNO de los dos CSV sigue degradando a
    # lista vacía, no a un 404.
    missing = client.get("/api/player/999/shot-chart").json()
    assert missing["shots"] == []


def test_player_shot_chart_real_picks_the_most_recent_of_several_cached_seasons(client, config):
    # Regresión: con varias temporadas cacheadas por jugador (para poder
    # ponderar por recencia en la proyección), kind=real no debe mezclar
    # temporadas ni quedarse con la que aparezca primero en el CSV.
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [
            {"player_id": 42, "season": "2025-26", "loc_x": 10, "loc_y": 10, "shot_made": True, "shot_type": "2PT Field Goal", "shot_zone_basic": "Mid-Range"},
            {"player_id": 42, "season": "2023-24", "loc_x": -50, "loc_y": -50, "shot_made": False, "shot_type": "3PT Field Goal", "shot_zone_basic": "Above the Break 3"},
            {"player_id": 42, "season": "2024-25", "loc_x": 5, "loc_y": 5, "shot_made": True, "shot_type": "2PT Field Goal", "shot_zone_basic": "Mid-Range"},
        ]
    ).to_csv(processed_dir / "roster_shot_charts.csv", index=False)

    body = client.get("/api/player/42/shot-chart").json()
    assert body["season"] == "2025-26"
    assert len(body["shots"]) == 1
    assert body["shots"][0]["loc_x"] == 10


def test_player_shot_chart_projected_matches_projected_totals(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [
            {"player_id": 42, "season": "2024-25", "loc_x": 0, "loc_y": 0, "shot_made": True, "shot_type": "2PT Field Goal", "shot_zone_basic": "Restricted Area"},
            {"player_id": 42, "season": "2024-25", "loc_x": 0, "loc_y": 0, "shot_made": False, "shot_type": "2PT Field Goal", "shot_zone_basic": "Restricted Area"},
            {"player_id": 42, "season": "2025-26", "loc_x": 200, "loc_y": 200, "shot_made": True, "shot_type": "3PT Field Goal", "shot_zone_basic": "Above the Break 3"},
            {"player_id": 42, "season": "2025-26", "loc_x": 200, "loc_y": 200, "shot_made": False, "shot_type": "3PT Field Goal", "shot_zone_basic": "Above the Break 3"},
        ]
    ).to_csv(processed_dir / "roster_shot_charts.csv", index=False)
    pd.DataFrame(
        [{
            "player_id": 42, "player_name": "Test Player",
            "FGA_per36_projected": 20.0, "FTA_per36_projected": 5.0, "TOV_per36_projected": 3.0,
            "AST_per36_projected": 9.0, "FG3A_per36_projected": 2.0,
            "BLK_per36_projected": 0.5, "DREB_per36_projected": 4.0,
            "projected_total_minutes": 2200,
            "PTS_projected": 1500, "REB_projected": 450, "AST_projected": 350,
            "STL_projected": 70, "BLK_projected": 40,
            "FGM_projected": 6, "FGA_projected": 10,
            "FG3M_projected": 2, "FG3A_projected": 4,
            "FTM_projected": 300, "FTA_projected": 350,
        }]
    ).to_csv(processed_dir / "aging_curve_projection.csv", index=False)

    body = client.get("/api/player/42/shot-chart?kind=projected").json()
    assert body["kind"] == "projected"
    assert len(body["shots"]) == 10
    assert sum(s["shot_type"] == "3PT Field Goal" for s in body["shots"]) == 4
    assert sum(s["shot_made"] for s in body["shots"]) == 6

    # Semilla estable por jugador -- dos peticiones seguidas dan el mismo mapa.
    again = client.get("/api/player/42/shot-chart?kind=projected").json()
    assert again["shots"] == body["shots"]


def test_player_shot_chart_projected_empty_without_projection(client, config):
    processed_dir = tmp_path_from_config(config)
    pd.DataFrame(
        [{"player_id": 42, "season": "2025-26", "loc_x": 0, "loc_y": 0, "shot_made": True, "shot_type": "2PT Field Goal", "shot_zone_basic": "Mid-Range"}]
    ).to_csv(processed_dir / "roster_shot_charts.csv", index=False)
    # sin aging_curve_projection.csv ni league_player_projections.csv

    body = client.get("/api/player/42/shot-chart?kind=projected").json()
    assert body["shots"] == []
    assert body["kind"] == "projected"


def _write_single_season_game_log(processed_dir, rows, suffix=""):
    pd.DataFrame(rows).to_csv(processed_dir / f"league_single_season_game_log{suffix}.csv", index=False)


def test_league_schedule_404_without_season_log(client):
    response = client.get("/api/league/schedule")
    assert response.status_code == 404
    assert "calendario" in response.json()["detail"].lower()


def test_league_simulate_season_log_calls_orchestrator_and_returns_game_count(client, config, monkeypatch):
    processed_dir = tmp_path_from_config(config)
    _write_league_csvs(processed_dir)  # _require_league_data necesita regular+playoff, mismo criterio que /bracket

    fake_game_log = pd.DataFrame([{"game_id": 0, "day": 0, "home_team_id": 1, "away_team_id": 2, "winner_team_id": 1}])
    captured = {}

    def _fake_run(cfg, scenario="with_injuries"):
        captured["scenario"] = scenario
        return {"game_log": fake_game_log, "player_box_scores": pd.DataFrame()}

    monkeypatch.setattr(league_router, "run_single_league_season_simulation", _fake_run)

    response = client.post("/api/league/simulate-season-log")
    assert response.status_code == 200
    assert response.json() == {"scenario": "with_injuries", "games": 1}
    assert captured["scenario"] == "with_injuries"


def test_league_schedule_filters_by_team(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_single_season_game_log(processed_dir, [
        {"game_id": 0, "day": 0, "home_team_id": 1, "home_abbreviation": "AAA", "away_team_id": 2,
         "away_abbreviation": "BBB", "home_score": 100.0, "away_score": 95.0, "point_differential": 5.0,
         "winner_team_id": 1, "winner_abbreviation": "AAA"},
        {"game_id": 1, "day": 1, "home_team_id": 3, "home_abbreviation": "CCC", "away_team_id": 4,
         "away_abbreviation": "DDD", "home_score": 90.0, "away_score": 92.0, "point_differential": -2.0,
         "winner_team_id": 4, "winner_abbreviation": "DDD"},
    ])

    all_games = client.get("/api/league/schedule")
    assert all_games.status_code == 200
    assert len(all_games.json()["games"]) == 2

    filtered = client.get("/api/league/schedule?team=AAA")
    assert filtered.status_code == 200
    games = filtered.json()["games"]
    assert len(games) == 1
    assert games[0]["home_abbreviation"] == "AAA"


def test_league_boxscore_returns_home_and_away_players(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_single_season_game_log(processed_dir, [
        {"game_id": 0, "day": 0, "home_team_id": 1, "home_abbreviation": "AAA", "away_team_id": 2,
         "away_abbreviation": "BBB", "home_score": 100.0, "away_score": 95.0, "point_differential": 5.0,
         "winner_team_id": 1, "winner_abbreviation": "AAA"},
    ])
    pd.DataFrame([
        {"game_id": 0, "day": 0, "team_id": 1, "player_id": 100, "player_name": "Local Player",
         "PTS": 20.0, "REB": 5.0, "AST": 3.0, "STL": 1.0, "BLK": 0.0, "TOV": 2.0, "3PM": 2.0},
        {"game_id": 0, "day": 0, "team_id": 2, "player_id": 200, "player_name": "Away Player",
         "PTS": 15.0, "REB": 8.0, "AST": 1.0, "STL": 0.0, "BLK": 1.0, "TOV": 1.0, "3PM": 1.0},
    ]).to_csv(processed_dir / "league_single_season_player_box_scores.csv", index=False)

    response = client.get("/api/league/boxscore/0")
    assert response.status_code == 200
    body = response.json()
    assert body["game"]["home_abbreviation"] == "AAA"
    assert len(body["home_players"]) == 1 and body["home_players"][0]["player_name"] == "Local Player"
    assert len(body["away_players"]) == 1 and body["away_players"][0]["player_name"] == "Away Player"


def test_league_boxscore_404_for_unknown_game_id(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_single_season_game_log(processed_dir, [
        {"game_id": 0, "day": 0, "home_team_id": 1, "home_abbreviation": "AAA", "away_team_id": 2,
         "away_abbreviation": "BBB", "home_score": 100.0, "away_score": 95.0, "point_differential": 5.0,
         "winner_team_id": 1, "winner_abbreviation": "AAA"},
    ])
    response = client.get("/api/league/boxscore/999")
    assert response.status_code == 404


def test_league_head_to_head_counts_wins_between_two_real_teams(client, config):
    from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID

    processed_dir = tmp_path_from_config(config)
    bos_id, mia_id, cle_id = ABBREVIATION_TO_TEAM_ID["BOS"], ABBREVIATION_TO_TEAM_ID["MIA"], ABBREVIATION_TO_TEAM_ID["CLE"]
    _write_single_season_game_log(processed_dir, [
        {"game_id": 0, "day": 0, "home_team_id": bos_id, "home_abbreviation": "BOS", "away_team_id": mia_id,
         "away_abbreviation": "MIA", "home_score": 100.0, "away_score": 90.0, "point_differential": 10.0,
         "winner_team_id": bos_id, "winner_abbreviation": "BOS"},
        {"game_id": 1, "day": 10, "home_team_id": mia_id, "home_abbreviation": "MIA", "away_team_id": bos_id,
         "away_abbreviation": "BOS", "home_score": 95.0, "away_score": 92.0, "point_differential": 3.0,
         "winner_team_id": mia_id, "winner_abbreviation": "MIA"},
        # Partido contra un tercer equipo -- no debe contar en el H2H de BOS vs MIA.
        {"game_id": 2, "day": 20, "home_team_id": bos_id, "home_abbreviation": "BOS", "away_team_id": cle_id,
         "away_abbreviation": "CLE", "home_score": 100.0, "away_score": 80.0, "point_differential": 20.0,
         "winner_team_id": bos_id, "winner_abbreviation": "BOS"},
    ])

    response = client.get("/api/league/head-to-head?team_a=BOS&team_b=MIA")
    assert response.status_code == 200
    body = response.json()
    assert body["team_a_wins"] == 1
    assert body["team_b_wins"] == 1
    assert len(body["games"]) == 2


def test_league_head_to_head_404_for_unknown_abbreviation(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_single_season_game_log(processed_dir, [
        {"game_id": 0, "day": 0, "home_team_id": 1, "home_abbreviation": "AAA", "away_team_id": 2,
         "away_abbreviation": "BBB", "home_score": 100.0, "away_score": 95.0, "point_differential": 5.0,
         "winner_team_id": 1, "winner_abbreviation": "AAA"},
    ])
    response = client.get("/api/league/head-to-head?team_a=ZZZ&team_b=BBB")
    assert response.status_code == 404


def _write_sandbox_pool(processed_dir, rows):
    pd.DataFrame(rows).to_csv(processed_dir / "league_player_projections.csv", index=False)


def _sandbox_pool_row(player_id, name, team, game_score_per36, mpg, risk=0.2, fatigue=0.2):
    return {
        "player_id": player_id,
        "player_name": name,
        "team_abbreviation": team,
        "conference": "East",
        "position": "G",
        "current_age": 25,
        "game_score_per36": game_score_per36,
        "minutes_projection": mpg,
        "minutes_per_game_last_season": mpg,
        "games_played_last_season": 70,
        "risk_score": risk,
        "fatigue_score": fatigue,
        "PPG": 15.0,
        "RPG": 5.0,
        "APG": 4.0,
    }


def test_sandbox_players_404_without_league_pipeline(client):
    response = client.get("/api/sandbox/players")
    assert response.status_code == 404


def test_sandbox_players_returns_pool(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_sandbox_pool(processed_dir, [
        _sandbox_pool_row(10, "Player Ten", "BOS", 18.0, 30.0),
        _sandbox_pool_row(11, "Player Eleven", "MIA", 12.0, 20.0),
    ])
    response = client.get("/api/sandbox/players")
    assert response.status_code == 200
    body = response.json()
    assert len(body["players"]) == 2
    assert body["players"][0]["player_name"] == "Player Ten"


def test_sandbox_default_returns_config_roster(client, config):
    response = client.get("/api/sandbox/default")
    assert response.status_code == 200
    assert response.json()["player_ids"] == [1]


def test_sandbox_simulate_rejects_short_roster(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_sandbox_pool(processed_dir, [_sandbox_pool_row(10, "Player Ten", "BOS", 18.0, 30.0)])
    pd.DataFrame({"WinPCT": [0.5] * 30}).to_csv(processed_dir / "prior_season_standings.csv", index=False)

    response = client.post("/api/sandbox/simulate", json={"player_ids": [10]})
    assert response.status_code == 400


def test_sandbox_simulate_returns_summary_for_custom_roster(client, config):
    processed_dir = tmp_path_from_config(config)
    rows = [_sandbox_pool_row(10 + i, f"Player {i}", "BOS", 15.0 + i, 25.0 - i) for i in range(6)]
    _write_sandbox_pool(processed_dir, rows)
    pd.DataFrame({"WinPCT": [0.5] * 30}).to_csv(processed_dir / "prior_season_standings.csv", index=False)

    response = client.post("/api/sandbox/simulate", json={"player_ids": [10, 11, 12, 13, 14, 15]})
    assert response.status_code == 200
    body = response.json()
    assert "mean" in body["summary"]
    assert body["n_seasons"] > 0


def test_sandbox_roster_stats_returns_players_for_hypothetical_roster(client, config):
    processed_dir = tmp_path_from_config(config)
    rows = []
    for i in range(6):
        row = _sandbox_pool_row(10 + i, f"Player {i}", "BOS", 15.0 + i, 25.0 - i)
        row["PTS_per36_projected"] = 18.0
        rows.append(row)
    _write_sandbox_pool(processed_dir, rows)
    pd.DataFrame({"WinPCT": [0.5] * 30}).to_csv(processed_dir / "prior_season_standings.csv", index=False)

    response = client.post("/api/sandbox/roster-stats", json={"player_ids": [10, 11, 12, 13, 14, 15], "mode": "per_game"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["players"]) == 6
    assert "PPG" in body["players"][0]


def test_sandbox_roster_stats_400_for_short_roster(client, config):
    processed_dir = tmp_path_from_config(config)
    _write_sandbox_pool(processed_dir, [_sandbox_pool_row(10, "Player Ten", "BOS", 18.0, 30.0)])

    response = client.post("/api/sandbox/roster-stats", json={"player_ids": [10]})
    assert response.status_code == 400
