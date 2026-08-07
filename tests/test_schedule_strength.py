"""
Tests de schedule_strength.py. Usan DataFrames sintéticos con el mismo
esquema que team_schedule.csv (de ScheduleLeagueV2) y
prior_season_standings.csv (de LeagueStandingsV3) -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from context.schedule_strength import (  # noqa: E402
    build_team_game_calendar,
    compute_back_to_back_scores,
    compute_difficulty_scores,
    compute_opponent_strength_scores,
    compute_travel_scores,
    haversine_km,
)

TEAM_ID = 1610612755  # Philadelphia 76ers (mismo que team_config.yaml)
OPP_A = 1610612738  # Boston Celtics
OPP_B = 1610612744  # Golden State Warriors


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameDate": r["date"],
                "arenaCity": r["city"],
                "homeTeam_teamId": TEAM_ID if r.get("home", True) else r["opponent"],
                "awayTeam_teamId": r["opponent"] if r.get("home", True) else TEAM_ID,
            }
            for r in rows
        ]
    )


def _standings(win_pcts: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"TeamID": team_id, "WinPCT": pct} for team_id, pct in win_pcts.items()]
    )


def test_haversine_known_distance_philadelphia_boston():
    # Distancia real Filadelfia-Boston ~440 km.
    dist = haversine_km((39.9526, -75.1652), (42.3601, -71.0589))
    assert 400 < dist < 480


def test_back_to_back_detected_only_for_zero_rest():
    games = _schedule(
        [
            {"date": "2026-10-22", "city": "Philadelphia", "opponent": OPP_A},
            {"date": "2026-10-23", "city": "Boston", "opponent": OPP_A, "home": False},  # b2b
            {"date": "2026-10-26", "city": "Philadelphia", "opponent": OPP_B},  # 3 días descanso
        ]
    )
    calendar = build_team_game_calendar(games, TEAM_ID)
    scores = compute_back_to_back_scores(calendar)

    assert list(scores) == [0.0, 1.0, 0.0]


def test_travel_score_scales_with_distance_and_caps_at_one():
    games = _schedule(
        [
            {"date": "2026-10-22", "city": "Philadelphia", "opponent": OPP_A},
            {"date": "2026-10-24", "city": "Boston", "opponent": OPP_A, "home": False},  # corto
            {"date": "2026-10-27", "city": "San Francisco", "opponent": OPP_B, "home": False},  # largo
        ]
    )
    calendar = build_team_game_calendar(games, TEAM_ID)
    scores = compute_travel_scores(calendar, high_travel_km=3000.0)

    assert scores.iloc[0] == 0.0  # primer partido, sin viaje previo
    assert 0.0 < scores.iloc[1] < 0.3  # Philly -> Boston es corto
    assert scores.iloc[2] == pytest.approx(1.0)  # Boston -> SF es larguísimo, capa en 1.0


def test_missing_city_coordinates_do_not_crash():
    games = _schedule(
        [
            {"date": "2026-01-10", "city": "Ciudad de México", "opponent": OPP_A},
            {"date": "2026-01-12", "city": "Philadelphia", "opponent": OPP_B},
        ]
    )
    calendar = build_team_game_calendar(games, TEAM_ID)
    scores = compute_travel_scores(calendar, high_travel_km=3000.0)

    assert scores.iloc[1] == 0.0  # sin coords para Ciudad de México -> tramo tratado como 0
    assert len(scores) == 2


def test_opponent_strength_uses_prior_season_win_pct():
    games = _schedule(
        [
            {"date": "2026-10-22", "city": "Philadelphia", "opponent": OPP_A},
            {"date": "2026-10-24", "city": "San Francisco", "opponent": OPP_B, "home": False},
        ]
    )
    calendar = build_team_game_calendar(games, TEAM_ID)
    standings = _standings({OPP_A: 0.30, OPP_B: 0.75})

    scores = compute_opponent_strength_scores(calendar, standings)
    assert scores.iloc[0] == pytest.approx(0.30)
    assert scores.iloc[1] == pytest.approx(0.75)


def test_opponent_strength_falls_back_to_league_mean_when_unknown():
    games = _schedule([{"date": "2026-10-22", "city": "Philadelphia", "opponent": 999999}])
    calendar = build_team_game_calendar(games, TEAM_ID)
    standings = _standings({OPP_A: 0.40, OPP_B: 0.60})

    scores = compute_opponent_strength_scores(calendar, standings)
    assert scores.iloc[0] == pytest.approx(0.50)  # media de la liga conocida


def test_difficulty_score_weights_are_configurable():
    games = _schedule(
        [
            {"date": "2026-10-22", "city": "Philadelphia", "opponent": OPP_A},
            {"date": "2026-10-23", "city": "San Francisco", "opponent": OPP_B, "home": False},
        ]
    )
    standings = _standings({OPP_A: 0.20, OPP_B: 0.80})

    default_result = compute_difficulty_scores(games, TEAM_ID, standings)
    travel_only_result = compute_difficulty_scores(
        games,
        TEAM_ID,
        standings,
        weights={"opponent_strength": 0.0, "back_to_back": 0.0, "travel": 1.0},
    )

    assert travel_only_result["difficulty_score"].iloc[1] == pytest.approx(
        travel_only_result["travel_score"].iloc[1]
    )
    assert not travel_only_result["difficulty_score"].equals(default_result["difficulty_score"])
