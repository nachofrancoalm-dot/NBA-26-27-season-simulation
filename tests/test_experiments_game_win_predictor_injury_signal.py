"""
Tests de select_key_players(), compute_missing_key_players(),
compute_recently_missing_key_players() y build_matchups_with_injury_signal()
en game_win_predictor_injury_signal.py. No cubre fetch_key_player_game_logs()
(API real) ni run_loso_three_way() (entrena de verdad, lento).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

import game_win_predictor_injury_signal as gwpis  # noqa: E402


def test_select_key_players_picks_top_n_by_minutes_per_game():
    rosters = pd.DataFrame([
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 10},
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 20},
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 30},
    ])
    career = pd.DataFrame([
        {"PLAYER_ID": 10, "SEASON_ID": "2020-21", "GP": 70, "MIN": 2100, "player_name": "Estrella"},  # 30 MPG
        {"PLAYER_ID": 20, "SEASON_ID": "2020-21", "GP": 70, "MIN": 700, "player_name": "Rol menor"},  # 10 MPG
        {"PLAYER_ID": 30, "SEASON_ID": "2019-20", "GP": 82, "MIN": 3000, "player_name": "Temporada equivocada"},  # otra temporada, no debe usarse
    ])

    key_players = gwpis.select_key_players(rosters, career, top_n=1)

    assert len(key_players) == 1
    assert key_players.iloc[0]["PLAYER_ID"] == 10
    assert key_players.iloc[0]["mpg"] == pytest.approx(30.0)


def test_select_key_players_excludes_players_with_zero_games():
    rosters = pd.DataFrame([{"TeamID": 1, "season": "2020-21", "PLAYER_ID": 10}])
    career = pd.DataFrame([{"PLAYER_ID": 10, "SEASON_ID": "2020-21", "GP": 0, "MIN": 0, "player_name": "Lesionado toda la temporada"}])

    key_players = gwpis.select_key_players(rosters, career, top_n=5)
    assert key_players.empty


def test_compute_missing_key_players_counts_absences_on_exact_date():
    team_game_features = pd.DataFrame([
        {"TEAM_ID": 1, "season": "2020-21", "GAME_DATE": pd.Timestamp("2020-12-25")},
    ])
    key_players = pd.DataFrame([
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 100, "player_name": "A", "mpg": 30},
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 200, "player_name": "B", "mpg": 28},
    ])
    # jugador 100 tiene fila esa fecha (jugó); jugador 200 no (faltó)
    key_player_game_logs = pd.DataFrame([
        {"Player_ID": 100, "season": "2020-21", "GAME_DATE": "2020-12-25"},
    ])

    result = gwpis.compute_missing_key_players(team_game_features, key_players, key_player_game_logs)
    assert result.iloc[0] == 1


def test_compute_missing_key_players_returns_nan_when_team_season_not_covered():
    # equipo-temporada sin jugadores clave seleccionados -- NaN, no 0 ("no se comprobó" != "nadie faltó")
    team_game_features = pd.DataFrame([
        {"TEAM_ID": 999, "season": "2020-21", "GAME_DATE": pd.Timestamp("2020-12-25")},
    ])
    key_players = pd.DataFrame([
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 100, "player_name": "A", "mpg": 30},
    ])
    key_player_game_logs = pd.DataFrame([
        {"Player_ID": 100, "season": "2020-21", "GAME_DATE": "2020-12-25"},
    ])

    result = gwpis.compute_missing_key_players(team_game_features, key_players, key_player_game_logs)
    assert result.isna().iloc[0]


def test_compute_recently_missing_key_players_looks_only_at_prior_team_games():
    dates = [pd.Timestamp(f"2020-12-{d:02d}") for d in [1, 3, 5, 7]]
    team_game_features = pd.DataFrame([
        {"TEAM_ID": 1, "season": "2020-21", "GAME_DATE": d} for d in dates
    ])
    key_players = pd.DataFrame([
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 100, "player_name": "Siempre presente", "mpg": 30},
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 200, "player_name": "Falto una vez", "mpg": 28},
    ])
    # jugador 100 jugó los 4 partidos; jugador 200 se perdió el 2o (dates[1])
    key_player_game_logs = pd.DataFrame([
        {"Player_ID": 100, "season": "2020-21", "GAME_DATE": d.strftime("%Y-%m-%d")} for d in dates
    ] + [
        {"Player_ID": 200, "season": "2020-21", "GAME_DATE": d.strftime("%Y-%m-%d")} for d in [dates[0], dates[2], dates[3]]
    ])

    result = gwpis.compute_recently_missing_key_players(team_game_features, key_players, key_player_game_logs, window=3)

    assert result.iloc[0:3].isna().all()  # sin 3 partidos previos -> NaN
    assert result.iloc[3] == 1  # ventana [0,1,2]: jugador 200 faltó en dates[1]


def test_compute_recently_missing_key_players_never_counts_the_current_game_itself():
    # faltar HOY (el partido que se predice) no debe contar -- eso es oráculo, no tendencia pregame
    dates = [pd.Timestamp(f"2020-12-{d:02d}") for d in [1, 3, 5, 7]]
    team_game_features = pd.DataFrame([
        {"TEAM_ID": 1, "season": "2020-21", "GAME_DATE": d} for d in dates
    ])
    key_players = pd.DataFrame([
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 100, "player_name": "Falta hoy", "mpg": 30},
    ])
    # jugador 100 jugó los 3 primeros partidos, faltó en el 4o (dates[3])
    key_player_game_logs = pd.DataFrame([
        {"Player_ID": 100, "season": "2020-21", "GAME_DATE": d.strftime("%Y-%m-%d")} for d in dates[:3]
    ])

    result = gwpis.compute_recently_missing_key_players(team_game_features, key_players, key_player_game_logs, window=3)
    assert result.iloc[3] == 0


def test_build_matchups_with_injury_signal_computes_home_away_diff():
    team_game_features = pd.DataFrame([
        {
            "GAME_ID": "G1", "season": "2020-21", "WL": "W", "is_home": True,
            "TEAM_ID": 1, "GAME_DATE": pd.Timestamp("2020-12-25"),
            "net_rating_rolling": 5.0, "rest_days": 2, "is_back_to_back": 0,
            "three_pt_rate_rolling": 0.3, "pace_rolling": 100.0,
        },
        {
            "GAME_ID": "G1", "season": "2020-21", "WL": "L", "is_home": False,
            "TEAM_ID": 2, "GAME_DATE": pd.Timestamp("2020-12-25"),
            "net_rating_rolling": 3.0, "rest_days": 2, "is_back_to_back": 0,
            "three_pt_rate_rolling": 0.3, "pace_rolling": 100.0,
        },
    ])
    key_players = pd.DataFrame([
        {"TeamID": 1, "season": "2020-21", "PLAYER_ID": 100, "player_name": "A", "mpg": 30},  # jugo
        {"TeamID": 2, "season": "2020-21", "PLAYER_ID": 200, "player_name": "B", "mpg": 28},  # falto
    ])
    key_player_game_logs = pd.DataFrame([
        {"Player_ID": 100, "season": "2020-21", "GAME_DATE": "2020-12-25"},
    ])

    matchups = gwpis.build_matchups_with_injury_signal(team_game_features, key_players, key_player_game_logs)

    assert len(matchups) == 1
    assert matchups.iloc[0]["missing_key_players_diff"] == -1  # local 0 ausentes, visitante 1
