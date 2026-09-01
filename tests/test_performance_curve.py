"""
Tests de performance_curve.py. Usan DataFrames sintéticos con el mismo
esquema que historical_comparables_advanced_game_logs.csv (GAME_DATE,
PLUS_MINUS, FGA, OREB, TOV, FTA, game_phase) -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from context.performance_curve import (  # noqa: E402
    compute_net_rating_estimate,
    compute_rolling_net_rating,
    estimate_possessions,
    summarize_season_narrative,
)


def _games(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_DATE": r["date"],
                "PLUS_MINUS": r["plus_minus"],
                "FGA": r.get("fga", 85),
                "OREB": r.get("oreb", 10),
                "TOV": r.get("tov", 14),
                "FTA": r.get("fta", 20),
                "game_phase": r.get("phase", "regular"),
            }
            for r in rows
        ]
    )


def test_estimate_possessions_matches_standard_formula():
    games = _games([{"date": "2010-11-01", "plus_minus": 5, "fga": 85, "oreb": 10, "tov": 14, "fta": 20}])
    poss = estimate_possessions(games)
    # 85 - 10 + 14 + 0.44*20 = 97.8
    assert poss.iloc[0] == pytest.approx(97.8)


def test_net_rating_estimate_sign_matches_plus_minus():
    games = _games(
        [
            {"date": "2010-11-01", "plus_minus": 20},  # victoria cómoda
            {"date": "2010-11-02", "plus_minus": -20},  # derrota cómoda
        ]
    )
    result = compute_net_rating_estimate(games)
    assert result["net_rating_estimate"].iloc[0] > 0
    assert result["net_rating_estimate"].iloc[1] < 0


def test_rolling_net_rating_smooths_early_variance():
    # partidos alternando +30/-30 -- el rolling debe cancelarse cerca de 0
    rows = [
        {"date": f"2010-11-{i+1:02d}", "plus_minus": 30 if i % 2 == 0 else -30}
        for i in range(10)
    ]
    games = _games(rows)
    result = compute_rolling_net_rating(games, window=10)

    assert abs(result["net_rating_estimate"].iloc[0]) > 20
    assert abs(result["rolling_net_rating"].iloc[-1]) < 5


def test_slow_start_detected_via_early_vs_rest_and_trend_slope():
    # primeros 10 partidos flojos, últimos 10 fuertes -- simula integración de superequipo
    rows = [{"date": f"2010-{11 if i < 20 else 12}-{(i % 28) + 1:02d}", "plus_minus": (2 if i < 10 else 18)} for i in range(20)]
    games = _games(rows)
    rolling = compute_rolling_net_rating(games, window=5)

    summary = summarize_season_narrative(rolling, early_season_games=10)

    assert summary["early_season_net_rating"] < summary["rest_of_season_net_rating"]
    assert summary["trend_slope"] > 0


def test_playoff_boost_detected_when_playoffs_outperform_regular_season():
    regular = _games(
        [{"date": f"2011-{i+1:02d}-01", "plus_minus": 3, "phase": "regular"} for i in range(10)]
    )
    playoffs = _games(
        [{"date": f"2011-05-{i+1:02d}", "plus_minus": 15, "phase": "playoffs"} for i in range(5)]
    )
    games = pd.concat([regular, playoffs], ignore_index=True)
    rolling = compute_rolling_net_rating(games, window=5)

    summary = summarize_season_narrative(rolling, early_season_games=10)

    assert summary["playoff_net_rating"] is not None
    assert summary["playoff_boost"] > 0


def test_no_playoff_games_gives_none_playoff_fields():
    regular = _games([{"date": f"2011-{i+1:02d}-01", "plus_minus": 3, "phase": "regular"} for i in range(5)])
    rolling = compute_rolling_net_rating(regular, window=5)

    summary = summarize_season_narrative(rolling)
    assert summary["playoff_net_rating"] is None
    assert summary["playoff_boost"] is None


def test_rolling_window_is_configurable():
    rows = [
        {"date": f"2010-11-{i+1:02d}", "plus_minus": 30 if i % 2 == 0 else -30}
        for i in range(10)
    ]
    games = _games(rows)

    narrow = compute_rolling_net_rating(games, window=2)
    wide = compute_rolling_net_rating(games, window=10)

    assert not narrow["rolling_net_rating"].equals(wide["rolling_net_rating"])
