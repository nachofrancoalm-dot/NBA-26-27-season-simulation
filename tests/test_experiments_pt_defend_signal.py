"""
Test de la parte pura de scripts/experiments/pt_defend_signal.py --
build_team_pt_defend_features(). No cubre la regresión en sí (ver el
docstring del módulo, resultado: señal real pero modesta, ΔR²≈0.019
validado LOSO tras corregir un bug de look-ahead real en la primera
versión de este experimento).

El test más importante de este archivo es justamente ESE bug: confirma
que la función usa SOLO la temporada PREVIA de cada jugador, nunca la
temporada del caso que se está "prediciendo" -- si esto se rompe otra
vez, cualquier resultado de este experimento vuelve a estar inflado por
circularidad sin que se note.
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
    # PCT_PLUSMINUS de la temporada del CASO es un valor absurdo (+50) a
    # propósito -- si la función lo usara por error (look-ahead), el test
    # de abajo lo detectaría inmediatamente.
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
    # Debe ser -5.0 (temporada previa), NUNCA 50.0 (temporada del caso --
    # eso sería el bug de look-ahead ya encontrado y corregido una vez).
    assert row["team_pct_plusminus_prior"] == pytest.approx(-5.0)


def test_build_team_pt_defend_features_skips_players_without_a_prior_season(pt_defend_config):
    # Un rookie (o un jugador cuya única temporada de tracking es la del
    # caso) no debe generar una fila con datos inventados.
    processed = Path(pt_defend_config["paths"]["processed_data_dir"])
    pt_defend = pd.read_csv(processed / "league_pt_defend_stats.csv")
    pt_defend[pt_defend["season"] != "2015-16"].to_csv(processed / "league_pt_defend_stats.csv", index=False)

    features = pds.build_team_pt_defend_features(pt_defend_config)
    assert len(features) == 0
