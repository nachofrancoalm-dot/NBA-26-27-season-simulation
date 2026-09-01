"""
Tests de build_team_hustle_features() en hustle_stats_signal.py (join +
ponderación por minutos). No cubre la regresión en sí sobre datos reales.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

import hustle_stats_signal as hss  # noqa: E402


@pytest.fixture
def hustle_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    season = "2015-16"

    rosters = pd.DataFrame([
        {"PLAYER_ID": 1, "PLAYER": "Heavy Minutes", "comparable_name": "BOS 2015-16", "season": season},
        {"PLAYER_ID": 2, "PLAYER": "Bench Player", "comparable_name": "BOS 2015-16", "season": season},
    ])
    hustle = pd.DataFrame([
        {"PLAYER_ID": 1, "season": season, "MIN": 32.0, "CONTESTED_SHOTS": 4.0, "DEFLECTIONS": 2.0,
         "CHARGES_DRAWN": 0.2, "SCREEN_ASSISTS": 1.0, "LOOSE_BALLS_RECOVERED": 0.5, "BOX_OUTS": 1.0},
        {"PLAYER_ID": 2, "season": season, "MIN": 8.0, "CONTESTED_SHOTS": 40.0, "DEFLECTIONS": 20.0,
         "CHARGES_DRAWN": 5.0, "SCREEN_ASSISTS": 10.0, "LOOSE_BALLS_RECOVERED": 5.0, "BOX_OUTS": 10.0},
    ])
    rosters.to_csv(processed / "backtest_sweep_rosters.csv", index=False)
    hustle.to_csv(processed / "league_hustle_player_stats.csv", index=False)

    return {"paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)}}


def test_build_team_hustle_features_weights_by_minutes_not_a_plain_average(hustle_config):
    features = hss.build_team_hustle_features(hustle_config)

    assert len(features) == 1
    row = features.iloc[0]
    # sin ponderar daría (4+40)/2=22; ponderado por minutos debe acercarse al titular (4.0)
    assert row["CONTESTED_SHOTS"] == pytest.approx((4.0 * 32 + 40.0 * 8) / 40)
    assert row["CONTESTED_SHOTS"] < 15.0


def test_build_team_hustle_features_skips_players_without_hustle_data(hustle_config):
    # un jugador sin fila en league_hustle_player_stats no debe romper el join
    extra_roster = pd.DataFrame([
        {"PLAYER_ID": 3, "PLAYER": "No Hustle Data", "comparable_name": "BOS 2015-16", "season": "2015-16"},
    ])
    processed = Path(hustle_config["paths"]["processed_data_dir"])
    existing = pd.read_csv(processed / "backtest_sweep_rosters.csv")
    pd.concat([existing, extra_roster], ignore_index=True).to_csv(
        processed / "backtest_sweep_rosters.csv", index=False
    )

    features = hss.build_team_hustle_features(hustle_config)
    assert len(features) == 1
