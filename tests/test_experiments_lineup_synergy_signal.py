"""
Tests de _parse_group_id(), build_player_style_profiles() y
build_pair_dataset() en lineup_synergy_signal.py. No cubre la regresión en
sí (RESULTADO NEGATIVO, ver CLAUDE.md).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

import lineup_synergy_signal as lss  # noqa: E402


def test_parse_group_id_extracts_both_player_ids():
    assert lss._parse_group_id("-1628404-1628969-") == (1628404, 1628969)


@pytest.fixture
def synergy_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)

    career_stats = pd.DataFrame(
        [
            {
                "PLAYER_ID": 1, "SEASON_ID": "2016-17", "MIN": 2000,
                "FGA": 1200, "FTA": 400, "TOV": 200, "AST": 300, "FG3A": 100,
                "BLK": 50, "DREB": 400,
            },
            {
                "PLAYER_ID": 2, "SEASON_ID": "2016-17", "MIN": 2000,
                "FGA": 800, "FTA": 200, "TOV": 150, "AST": 500, "FG3A": 600,
                "BLK": 20, "DREB": 200,
            },
        ]
    )
    career_stats.to_csv(processed / "backtest_sweep_player_career_stats.csv", index=False)

    lineups = pd.DataFrame(
        [
            {"GROUP_ID": "-1-2-", "GROUP_NAME": "Player One - Player Two", "MIN": 500, "NET_RATING": 4.2, "season": "2016-17"},
            {"GROUP_ID": "-1-2-", "GROUP_NAME": "Player One - Player Two", "MIN": 50, "NET_RATING": 99.0, "season": "2017-18"},  # bajo umbral, excluida
        ]
    )
    lineups.to_csv(processed / "league_2man_lineups.csv", index=False)

    tracking = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "season": "2016-17", "MIN": 2000, "POST_TOUCH_FGA": 100, "PULL_UP_FGA": 200, "CATCH_SHOOT_FGA": 50, "DRIVES": 300},
            {"PLAYER_ID": 2, "season": "2016-17", "MIN": 2000, "POST_TOUCH_FGA": 20, "PULL_UP_FGA": 300, "CATCH_SHOOT_FGA": 400, "DRIVES": 100},
        ]
    )
    tracking.to_csv(processed / "league_tracking_stats.csv", index=False)

    return {"paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)}}


def test_build_player_style_profiles_computes_per36_rates(synergy_config):
    profiles = lss.build_player_style_profiles(synergy_config)

    profile_1 = profiles[(1, "2016-17")]
    assert profile_1["usage"] == pytest.approx(21.6 + 0.44 * 7.2 + 3.6)


def test_build_pair_dataset_filters_by_minimum_shared_minutes(synergy_config):
    df = lss.build_pair_dataset(synergy_config, min_shared_minutes=300.0)

    assert len(df) == 1  # solo la fila con 500 min juntos sobrevive al umbral
    assert df.iloc[0]["season"] == "2016-17"
    assert df.iloc[0]["net_rating"] == pytest.approx(4.2)


def test_build_pair_dataset_skips_pairs_without_both_profiles(synergy_config):
    processed = Path(synergy_config["paths"]["processed_data_dir"])
    lineups = pd.read_csv(processed / "league_2man_lineups.csv")
    # jugador 999 no tiene perfil de estilo en ninguna temporada
    lineups = pd.concat(
        [lineups, pd.DataFrame([{"GROUP_ID": "-1-999-", "GROUP_NAME": "x", "MIN": 500, "NET_RATING": 1.0, "season": "2016-17"}])],
        ignore_index=True,
    )
    lineups.to_csv(processed / "league_2man_lineups.csv", index=False)

    df = lss.build_pair_dataset(synergy_config, min_shared_minutes=300.0)

    assert len(df) == 1  # la pareja (1, 999) no debe aparecer


def test_build_tracking_style_features_computes_per36_rates(synergy_config):
    processed = Path(synergy_config["paths"]["processed_data_dir"])
    tracking = pd.DataFrame(
        [{"PLAYER_ID": 1, "season": "2016-17", "MIN": 2000, "POST_TOUCH_FGA": 200, "PULL_UP_FGA": 400, "CATCH_SHOOT_FGA": 100, "DRIVES": 800}]
    )
    tracking.to_csv(processed / "league_tracking_stats.csv", index=False)

    features = lss.build_tracking_style_features(synergy_config)

    row = features[(1, "2016-17")]
    assert row["post_volume"] == pytest.approx(200 / 2000 * 36.0)
    assert row["drive_volume"] == pytest.approx(800 / 2000 * 36.0)


def test_build_tracking_style_features_treats_missing_category_as_zero_not_nan(synergy_config):
    # regression: `row.get(col) or 0.0` no basta para NaN -- `NaN or 0.0` da NaN (NaN es truthy en
    # Python), colando NaN en vez de 0.0 y rompiendo statsmodels más adelante
    processed = Path(synergy_config["paths"]["processed_data_dir"])
    tracking = pd.DataFrame(
        [{"PLAYER_ID": 1, "season": "2016-17", "MIN": 2000, "POST_TOUCH_FGA": None, "PULL_UP_FGA": 400, "CATCH_SHOOT_FGA": 100, "DRIVES": 800}]
    )
    tracking.to_csv(processed / "league_tracking_stats.csv", index=False)

    features = lss.build_tracking_style_features(synergy_config)

    assert features[(1, "2016-17")]["post_volume"] == 0.0
