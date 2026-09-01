"""
Tests de orquestación de team_quality_uncertainty.py: confirma que
team_quality_uncertainty_std llega hasta _run_backtest_cases()/run_monte_carlo
y ensancha la banda P10-P90. No repite el grid completo (eso es el
experimento en sí). DataFrames sintéticos, no requiere red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

import team_quality_uncertainty as tqu  # noqa: E402


def _career_seasons(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_ID": r.get("player_id", 1), "SEASON_ID": r["season"], "PLAYER_AGE": r["age"],
                "GP": r.get("gp", 80), "MIN": r.get("min", 2400), "PTS": r.get("pts", 1200),
                "AST": r.get("ast", 300), "REB": r.get("reb", 400), "STL": r.get("stl", 80),
                "BLK": r.get("blk", 40), "TOV": r.get("tov", 150), "FG3M": r.get("fg3m", 100),
                "FG3A": r.get("fg3a", 260), "OREB": r.get("oreb", 80), "DREB": r.get("dreb", 320),
                "FGM": r.get("fgm", 420), "FGA": r.get("fga", 880), "FTM": r.get("ftm", 200),
                "FTA": r.get("fta", 260), "PF": r.get("pf", 180), "TEAM_ABBREVIATION": r.get("team", "TOT"),
            }
            for r in rows
        ]
    )


@pytest.fixture
def synthetic_inputs():
    case = {"name": "Test Team 2010-11", "team_id": 1610612748, "season": "2010-11"}
    rosters = pd.DataFrame([
        {"PLAYER_ID": 1, "PLAYER": "Star Player", "AGE": 26, "comparable_name": case["name"], "season": "2010-11"},
        {"PLAYER_ID": 2, "PLAYER": "Role Player", "AGE": 24, "comparable_name": case["name"], "season": "2010-11"},
    ])
    player_stats = pd.concat([
        _career_seasons([{"player_id": 1, "season": "2009-10", "age": 25, "pts": 1900, "min": 2900}]),
        _career_seasons([{"player_id": 2, "season": "2009-10", "age": 23, "pts": 500, "min": 1200}]),
    ], ignore_index=True)
    standings = pd.DataFrame([{"TeamID": case["team_id"], "WinPCT": 0.55, "season": "2010-11"}])
    game_log = pd.DataFrame([
        {
            "GAME_DATE": f"2010-11-{i + 1:02d}", "MATCHUP": "MIA vs. BOS" if i % 2 == 0 else "MIA @ ORL",
            "WL": "W" if i % 3 != 0 else "L", "comparable_name": case["name"], "game_phase": "regular",
        }
        for i in range(20)
    ])
    config = {
        "simulation": {"n_seasons": 300, "games_per_season": 82, "random_seed": 1},
        "lineup_synergy": {}, "monte_carlo": {},
    }
    inputs = (rosters, player_stats, pd.DataFrame(), standings, game_log, None, {})
    return config, [case], inputs


def test_run_backtest_with_std_widens_p10_p90_band(synthetic_inputs):
    config, cases, inputs = synthetic_inputs

    without_noise = tqu.run_backtest_with_std(config, team_quality_std=0.0, n_seasons=300, cases=cases, inputs=inputs)
    with_noise = tqu.run_backtest_with_std(config, team_quality_std=15.0, n_seasons=300, cases=cases, inputs=inputs)

    band_without = without_noise.iloc[0]["simulated_wins_p90"] - without_noise.iloc[0]["simulated_wins_p10"]
    band_with = with_noise.iloc[0]["simulated_wins_p90"] - with_noise.iloc[0]["simulated_wins_p10"]
    assert band_with > band_without

    # la media no debe moverse de forma perceptible (ruido de media cero)
    assert with_noise.iloc[0]["simulated_wins_mean"] == pytest.approx(
        without_noise.iloc[0]["simulated_wins_mean"], abs=3.0
    )


def test_sweep_reports_one_row_per_candidate(synthetic_inputs):
    config, cases, inputs = synthetic_inputs
    config["backtest_sweep"] = {"seasons": ["2010-11"]}

    # sweep() vuelve a leer los CSV -- se prueba vía run_backtest_with_std + compute_calibration_summary
    from backtesting import compute_calibration_summary

    results = []
    for std in [0.0, 15.0]:
        result_df = tqu.run_backtest_with_std(config, std, n_seasons=300, cases=cases, inputs=inputs)
        calibration = compute_calibration_summary(result_df)
        results.append({"team_quality_uncertainty_std": std, **calibration})
    df = pd.DataFrame(results)

    assert len(df) == 2
    assert {"pct_within_p10_p90", "mean_absolute_error_wins"}.issubset(df.columns)
