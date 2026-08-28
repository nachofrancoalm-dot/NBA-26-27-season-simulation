"""
Test de la parte pura de scripts/experiments/game_win_predictor.py --
build_team_game_features() y build_matchup_dataset(). No cubre
run_loso() (entrena HistGradientBoostingClassifier/LogisticRegression
real, lento y es el experimento en sí, no lógica que deba cubrir un
test rápido) -- ver el docstring del módulo: RESULTADO NEGATIVO (el GBT
con seis features no mejora sobre una logística de una sola variable),
documentado con el mismo rigor que injury_survival_model.py.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

import game_win_predictor as gwp  # noqa: E402


def _team_game_row(team_id, game_id, date, matchup, wl, plus_minus, fga=100, extra=None):
    row = {
        "TEAM_ID": team_id, "season": "2020-21", "game_phase": "regular",
        "GAME_ID": game_id, "GAME_DATE": date, "MATCHUP": matchup, "WL": wl,
        "PLUS_MINUS": plus_minus, "FGA": fga, "FG3A": fga * 0.3, "OREB": 0, "TOV": 0, "FTA": 0,
    }
    if extra:
        row.update(extra)
    return row


def test_build_team_game_features_uses_only_prior_games_no_look_ahead():
    # net_rating por partido (OREB=TOV=FTA=0 -> POSS=FGA): 20.0, 10.0, -50.0.
    # Si hubiera look-ahead, el rolling del partido 3 incluiria su propio
    # -50.0 (media != 15.0).
    rows = [
        _team_game_row(1, "G1", "2020-10-01", "AAA vs. BBB", "W", 20, fga=100),
        _team_game_row(1, "G2", "2020-10-03", "AAA @ CCC", "W", 10, fga=100),
        _team_game_row(1, "G3", "2020-10-05", "AAA vs. BBB", "L", -50, fga=100),
    ]
    features = gwp.build_team_game_features(pd.DataFrame(rows))

    # El primer partido de temporada se descarta (sin historial previo).
    assert len(features) == 2
    game3 = features[features["GAME_ID"] == "G3"].iloc[0]
    assert game3["net_rating_rolling"] == pytest.approx(15.0)  # media de G1 (20.0) y G2 (10.0)


def test_build_team_game_features_default_uses_expanding_average_not_fixed_window():
    # net_rating por partido (FGA=100 -> POSS=100): 20, 10, -50, 40.
    # Con ventana fija=2, el pregame del partido 4 solo promediaria G2 y
    # G3 (10 y -50 -> -20.0). Con la media EXPANDIDA (default, sin pasar
    # rolling_window), el pregame del partido 4 promedia TODO el
    # historial previo: G1, G2, G3 (20, 10, -50 -> -6.67).
    rows = [
        _team_game_row(1, "G1", "2020-10-01", "AAA vs. BBB", "W", 20, fga=100),
        _team_game_row(1, "G2", "2020-10-03", "AAA @ CCC", "W", 10, fga=100),
        _team_game_row(1, "G3", "2020-10-05", "AAA vs. BBB", "L", -50, fga=100),
        _team_game_row(1, "G4", "2020-10-07", "AAA @ CCC", "W", 40, fga=100),
    ]
    df = pd.DataFrame(rows)

    expanding = gwp.build_team_game_features(df)  # default: rolling_window=None
    fixed_window = gwp.build_team_game_features(df, rolling_window=2)

    game4_expanding = expanding[expanding["GAME_ID"] == "G4"].iloc[0]
    game4_fixed = fixed_window[fixed_window["GAME_ID"] == "G4"].iloc[0]

    assert game4_expanding["net_rating_rolling"] == pytest.approx((20 + 10 - 50) / 3)
    assert game4_fixed["net_rating_rolling"] == pytest.approx((10 - 50) / 2)


def test_build_team_game_features_drops_first_game_of_season():
    rows = [_team_game_row(1, "G1", "2020-10-01", "AAA vs. BBB", "W", 20)]
    features = gwp.build_team_game_features(pd.DataFrame(rows))
    assert features.empty


def test_build_team_game_features_clips_rest_days_and_flags_back_to_back():
    rows = [
        _team_game_row(1, "G1", "2020-10-01", "AAA vs. BBB", "W", 10),  # se descarta (primer partido)
        _team_game_row(1, "G2", "2020-10-02", "AAA @ CCC", "L", -5),    # 1 dia de descanso -> B2B
        _team_game_row(1, "G3", "2020-10-13", "AAA vs. BBB", "W", 8),   # 11 dias -> recortado a 5
    ]
    features = gwp.build_team_game_features(pd.DataFrame(rows))

    game2 = features[features["GAME_ID"] == "G2"].iloc[0]
    assert game2["rest_days"] == pytest.approx(1.0)
    assert game2["is_back_to_back"] == 1

    game3 = features[features["GAME_ID"] == "G3"].iloc[0]
    assert game3["rest_days"] == pytest.approx(gwp.MAX_REST_DAYS)
    assert game3["is_back_to_back"] == 0


def test_build_team_game_features_ignores_playoff_games():
    rows = [
        _team_game_row(1, "G1", "2020-10-01", "AAA vs. BBB", "W", 10, extra={"game_phase": "playoffs"}),
    ]
    features = gwp.build_team_game_features(pd.DataFrame(rows))
    assert features.empty


def test_build_matchup_dataset_pairs_home_and_away_with_diff_features():
    team_game_features = pd.DataFrame([
        {
            "GAME_ID": "G1", "season": "2020-21", "WL": "W", "is_home": True,
            "net_rating_rolling": 10.0, "rest_days": 2, "is_back_to_back": 0,
            "three_pt_rate_rolling": 0.4, "pace_rolling": 100.0,
        },
        {
            "GAME_ID": "G1", "season": "2020-21", "WL": "L", "is_home": False,
            "net_rating_rolling": 4.0, "rest_days": 1, "is_back_to_back": 1,
            "three_pt_rate_rolling": 0.3, "pace_rolling": 98.0,
        },
    ])

    matchups = gwp.build_matchup_dataset(team_game_features)

    assert len(matchups) == 1
    row = matchups.iloc[0]
    assert row["home_win"] == 1
    assert row["net_rating_diff"] == pytest.approx(6.0)
    assert row["rest_days_diff"] == pytest.approx(1.0)
    assert row["is_b2b_home"] == 0
    assert row["is_b2b_away"] == 1
    assert row["three_pt_rate_diff"] == pytest.approx(0.1)
    assert row["pace_diff"] == pytest.approx(2.0)


def test_build_matchup_dataset_home_loss_labels_zero():
    team_game_features = pd.DataFrame([
        {
            "GAME_ID": "G1", "season": "2020-21", "WL": "L", "is_home": True,
            "net_rating_rolling": 1.0, "rest_days": 2, "is_back_to_back": 0,
            "three_pt_rate_rolling": 0.3, "pace_rolling": 100.0,
        },
        {
            "GAME_ID": "G1", "season": "2020-21", "WL": "W", "is_home": False,
            "net_rating_rolling": 5.0, "rest_days": 2, "is_back_to_back": 0,
            "three_pt_rate_rolling": 0.3, "pace_rolling": 100.0,
        },
    ])

    matchups = gwp.build_matchup_dataset(team_game_features)
    assert matchups.iloc[0]["home_win"] == 0
