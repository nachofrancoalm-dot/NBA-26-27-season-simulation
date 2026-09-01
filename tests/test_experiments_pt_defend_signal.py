"""
Tests de build_team_pt_defend_features() en pt_defend_signal.py.
regression: la función debe usar solo la temporada previa de cada jugador,
nunca la del caso que se está prediciendo (bug de look-ahead corregido antes).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

import pt_defend_signal as pds  # noqa: E402


@pytest.fixture
def pt_defend_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    case_season = "2016-17"
    prior_season = "2015-16"

    rosters = pd.DataFrame([
        {"PLAYER_ID": 1, "PLAYER": "Star", "comparable_name": f"BOS {case_season}", "season": case_season},
    ])
    player_stats = pd.DataFrame([
        {"PLAYER_ID": 1, "SEASON_ID": case_season, "GP": 80, "MIN": 2800},
    ])
    # PCT_PLUSMINUS de la temporada del caso es un valor absurdo (+50) a propósito, para detectar look-ahead
    pt_defend = pd.DataFrame([
        {"PLAYER_ID": 1, "season": prior_season, "D_FGA": 10.0, "PCT_PLUSMINUS": -5.0},
        {"PLAYER_ID": 1, "season": case_season, "D_FGA": 10.0, "PCT_PLUSMINUS": 50.0},
    ])

    rosters.to_csv(processed / "backtest_sweep_rosters.csv", index=False)
    player_stats.to_csv(processed / "backtest_sweep_player_career_stats.csv", index=False)
    pt_defend.to_csv(processed / "league_pt_defend_stats.csv", index=False)

    return {"paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)}}


def test_build_team_pt_defend_features_uses_only_the_prior_season(pt_defend_config):
    features = pds.build_team_pt_defend_features(pt_defend_config)

    assert len(features) == 1
    row = features.iloc[0]
    assert row["team_pct_plusminus_prior"] == pytest.approx(-5.0)  # nunca 50.0 (temporada del caso)


def test_build_team_pt_defend_features_skips_players_without_a_prior_season(pt_defend_config):
    # un rookie sin temporada previa de tracking no debe generar una fila con datos inventados
    processed = Path(pt_defend_config["paths"]["processed_data_dir"])
    pt_defend = pd.read_csv(processed / "league_pt_defend_stats.csv")
    pt_defend[pt_defend["season"] != "2015-16"].to_csv(processed / "league_pt_defend_stats.csv", index=False)

    features = pds.build_team_pt_defend_features(pt_defend_config)
    assert len(features) == 0
