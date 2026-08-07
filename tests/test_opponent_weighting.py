"""
Tests de opponent_weighting.py. Usan DataFrames sintéticos con el mismo
esquema que historical_comparables_advanced_game_logs.csv (MATCHUP,
PLUS_MINUS, FGA, OREB, TOV, FTA, WL) y historical_comparables_standings.csv
(TeamID, WinPCT) -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from context.opponent_weighting import (  # noqa: E402
    ABBREVIATION_TO_TEAM_ID,
    classify_contender_vs_rebuilding,
    compute_opponent_weight,
    compute_opponent_win_pct,
    compute_weighted_net_rating,
    extract_opponent_abbreviation,
    resolve_opponent_team_id,
    summarize_opponent_weighting,
)


def _games(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "MATCHUP": r["matchup"],
                "WL": r.get("wl", "W"),
                "PLUS_MINUS": r.get("plus_minus", 5),
                "FGA": r.get("fga", 85),
                "OREB": r.get("oreb", 10),
                "TOV": r.get("tov", 14),
                "FTA": r.get("fta", 20),
            }
            for r in rows
        ]
    )


def _standings(win_pcts: dict) -> pd.DataFrame:
    return pd.DataFrame([{"TeamID": team_id, "WinPCT": pct} for team_id, pct in win_pcts.items()])


def test_extract_opponent_abbreviation_home_and_away():
    assert extract_opponent_abbreviation("MIA @ TOR") == "TOR"
    assert extract_opponent_abbreviation("MIA vs. BOS") == "BOS"


def test_resolve_opponent_team_id_applies_historical_aliases():
    # Nets jugaban como "New Jersey Nets" (NJN) antes de mudarse a Brooklyn.
    assert resolve_opponent_team_id("MIA @ NJN") == ABBREVIATION_TO_TEAM_ID["BKN"]
    # Pelicans jugaban como "New Orleans Hornets" (NOH) antes de renombrarse.
    assert resolve_opponent_team_id("MIA @ NOH") == ABBREVIATION_TO_TEAM_ID["NOP"]


def test_resolve_opponent_team_id_unknown_abbreviation_returns_none():
    assert resolve_opponent_team_id("MIA @ ZZZ") is None


def test_opponent_win_pct_falls_back_to_mean_for_unresolved_matchup():
    games = _games([{"matchup": "MIA @ ZZZ"}])
    standings = _standings({ABBREVIATION_TO_TEAM_ID["BOS"]: 0.40, ABBREVIATION_TO_TEAM_ID["TOR"]: 0.60})

    win_pct = compute_opponent_win_pct(games, standings)
    assert win_pct.iloc[0] == pytest.approx(0.50)


def test_opponent_weight_scales_with_win_pct_and_steepness():
    win_pct = pd.Series([0.3, 0.6, 0.9])
    linear = compute_opponent_weight(win_pct, steepness=1.0)
    steep = compute_opponent_weight(win_pct, steepness=3.0)

    assert list(linear) == pytest.approx([0.3, 0.6, 0.9])
    # Con mayor steepness, los rivales débiles pesan proporcionalmente menos
    # frente a los fuertes.
    assert steep.iloc[0] / steep.iloc[2] < linear.iloc[0] / linear.iloc[2]


def test_weighted_net_rating_favors_games_against_strong_opponents():
    # Mismo net_rating_estimate en ambos partidos, pero un rival es mucho
    # más fuerte que el otro -- la media ponderada debe acercarse a lo que
    # pasó contra el rival fuerte, no a la media simple.
    net_rating = pd.Series([10.0, 10.0])
    weight = pd.Series([0.1, 0.9])
    weighted = compute_weighted_net_rating(net_rating, weight)
    assert weighted == pytest.approx(10.0)  # mismo valor en ambos, no cambia nada aquí

    net_rating_diff = pd.Series([2.0, 20.0])
    weighted_diff = compute_weighted_net_rating(net_rating_diff, weight)
    unweighted_diff = net_rating_diff.mean()
    assert weighted_diff > unweighted_diff  # el partido contra el rival fuerte pesa más


def test_weighted_net_rating_falls_back_to_simple_mean_when_all_weights_zero():
    net_rating = pd.Series([5.0, 15.0])
    weight = pd.Series([0.0, 0.0])
    assert compute_weighted_net_rating(net_rating, weight) == pytest.approx(10.0)


def test_classify_contender_vs_rebuilding_uses_configurable_thresholds():
    win_pct = pd.Series([0.30, 0.50, 0.70])
    tiers = classify_contender_vs_rebuilding(win_pct, contender_win_pct=0.55, rebuilding_win_pct=0.45)
    assert list(tiers) == ["reconstruccion", "medio", "contender"]


def test_summarize_distinguishes_contender_wins_from_rebuilding_wins():
    contender_id = ABBREVIATION_TO_TEAM_ID["BOS"]
    rebuilding_id = ABBREVIATION_TO_TEAM_ID["TOR"]
    games = _games(
        [
            {"matchup": "MIA @ BOS", "plus_minus": 3},  # gana ajustado a un contender
            {"matchup": "MIA @ TOR", "plus_minus": 25},  # aplasta a un equipo en reconstrucción
        ]
    )
    standings = _standings({contender_id: 0.65, rebuilding_id: 0.25})

    summary = summarize_opponent_weighting(games, standings)

    assert summary["contender_games"] == 1
    assert summary["reconstruccion_games"] == 1
    assert summary["contender_net_rating"] < summary["reconstruccion_net_rating"]
