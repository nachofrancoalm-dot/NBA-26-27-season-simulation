"""
Tests de conference_adjustment.py. Usan DataFrames sintéticos con el
mismo esquema que historical_comparables_standings.csv (TeamID,
Conference, WinPCT, DiffPointsPG, season) -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from context.conference_adjustment import (  # noqa: E402
    NET_RATING_BASELINE,
    WIN_PCT_BASELINE,
    compute_conference_adjusted_value,
    compute_conference_strength_index,
    get_team_conference_row,
)


def _standings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TeamID": r["team_id"],
                "Conference": r["conf"],
                "WinPCT": r["win_pct"],
                "DiffPointsPG": r["diff"],
                "season": r["season"],
            }
            for r in rows
        ]
    )


def test_conference_strength_index_is_zero_for_a_perfectly_balanced_season():
    # East y West exactamente en 0.5 / 0 -- ninguna conferencia domina.
    standings = _standings(
        [
            {"team_id": 1, "conf": "East", "win_pct": 0.5, "diff": 0.0, "season": "2020-21"},
            {"team_id": 2, "conf": "East", "win_pct": 0.5, "diff": 0.0, "season": "2020-21"},
            {"team_id": 3, "conf": "West", "win_pct": 0.5, "diff": 0.0, "season": "2020-21"},
            {"team_id": 4, "conf": "West", "win_pct": 0.5, "diff": 0.0, "season": "2020-21"},
        ]
    )
    index = compute_conference_strength_index(standings, "DiffPointsPG", NET_RATING_BASELINE)
    assert index["conference_index"].abs().max() == pytest.approx(0.0)


def test_conference_strength_index_detects_dominant_conference():
    # West domina claramente al East en diferencial de puntos.
    standings = _standings(
        [
            {"team_id": 1, "conf": "East", "win_pct": 0.4, "diff": -3.0, "season": "2016-17"},
            {"team_id": 2, "conf": "East", "win_pct": 0.4, "diff": -3.0, "season": "2016-17"},
            {"team_id": 3, "conf": "West", "win_pct": 0.6, "diff": 3.0, "season": "2016-17"},
            {"team_id": 4, "conf": "West", "win_pct": 0.6, "diff": 3.0, "season": "2016-17"},
        ]
    )
    index = compute_conference_strength_index(standings, "DiffPointsPG", NET_RATING_BASELINE)
    west_index = index[index["Conference"] == "West"]["conference_index"].iloc[0]
    east_index = index[index["Conference"] == "East"]["conference_index"].iloc[0]

    assert west_index == pytest.approx(3.0)
    assert east_index == pytest.approx(-3.0)


def test_conference_strength_index_win_pct_uses_correct_baseline():
    standings = _standings(
        [
            {"team_id": 1, "conf": "East", "win_pct": 0.45, "diff": 0.0, "season": "2010-11"},
            {"team_id": 2, "conf": "West", "win_pct": 0.55, "diff": 0.0, "season": "2010-11"},
        ]
    )
    index = compute_conference_strength_index(standings, "WinPCT", WIN_PCT_BASELINE)
    east_index = index[index["Conference"] == "East"]["conference_index"].iloc[0]
    west_index = index[index["Conference"] == "West"]["conference_index"].iloc[0]

    assert east_index == pytest.approx(-0.05)
    assert west_index == pytest.approx(0.05)


def test_adjusted_value_credits_playing_in_a_tougher_conference():
    # Mismo Net Rating bruto, pero un equipo jugó en una conferencia más
    # dura (índice positivo) -- su valor ajustado debe ser MAYOR.
    raw_net_rating = 5.0
    adjusted_tough_conference = compute_conference_adjusted_value(raw_net_rating, conference_index=2.0)
    adjusted_weak_conference = compute_conference_adjusted_value(raw_net_rating, conference_index=-2.0)

    assert adjusted_tough_conference < raw_net_rating < adjusted_weak_conference
    assert adjusted_tough_conference == pytest.approx(3.0)
    assert adjusted_weak_conference == pytest.approx(7.0)


def test_get_team_conference_row_finds_correct_season_and_team():
    standings = _standings(
        [
            {"team_id": 1610612748, "conf": "East", "win_pct": 0.71, "diff": 5.0, "season": "2010-11"},
            {"team_id": 1610612748, "conf": "East", "win_pct": 0.30, "diff": -6.0, "season": "2005-06"},
        ]
    )
    row = get_team_conference_row(standings, 1610612748, "2010-11")
    assert row["WinPCT"] == pytest.approx(0.71)


def test_get_team_conference_row_raises_when_missing():
    standings = _standings(
        [{"team_id": 1, "conf": "East", "win_pct": 0.5, "diff": 0.0, "season": "2020-21"}]
    )
    with pytest.raises(ValueError):
        get_team_conference_row(standings, 999, "2020-21")
