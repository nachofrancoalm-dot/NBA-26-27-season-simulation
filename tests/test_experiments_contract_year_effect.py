"""
Tests de compute_game_score(), build_contract_year_panel() y
paired_delta_test() en contract_year_effect.py. No cubre la regresión de
efectos fijos en sí (RESULTADO NEGATIVO sobre datos de Kaggle, ver CLAUDE.md).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

import contract_year_effect as cye  # noqa: E402


def test_compute_game_score_matches_hollinger_formula():
    row = pd.Series(
        {"PTS": 27.0, "FG": 9.8, "FGA": 21.5, "FT": 6.0, "FTA": 7.4, "ORB": 1.1, "DRB": 4.3, "STL": 1.5, "AST": 5.0, "BLK": 0.3, "TOV": 3.2, "PF": 2.6}
    )
    expected = 27.0 + 0.4 * 9.8 - 0.7 * 21.5 - 0.4 * (7.4 - 6.0) + 0.7 * 1.1 + 0.3 * 4.3 + 1.5 + 0.7 * 5.0 + 0.7 * 0.3 - 0.4 * 2.6 - 3.2
    assert cye.compute_game_score(row) == pytest.approx(expected)


@pytest.fixture
def contract_config(tmp_path):
    raw = tmp_path / "raw" / "contract_data"
    raw.mkdir(parents=True)

    # Jugador A: temporada final mucho mejor (efecto). Jugador B: temporada final peor (contra-caso,
    # para que el test no pase por casualidad si el signo estuviera invertido).
    salaries = pd.DataFrame(
        [
            {"Player": "Player A", "Year": 2014, "Age": 24, "PTS": 15.0, "FG": 6.0, "FGA": 13.0, "FT": 2.0, "FTA": 2.5, "ORB": 1.0, "DRB": 3.0, "STL": 1.0, "AST": 3.0, "BLK": 0.3, "TOV": 2.0, "PF": 2.0},
            {"Player": "Player A", "Year": 2015, "Age": 25, "PTS": 16.0, "FG": 6.2, "FGA": 13.2, "FT": 2.2, "FTA": 2.7, "ORB": 1.0, "DRB": 3.1, "STL": 1.0, "AST": 3.1, "BLK": 0.3, "TOV": 2.0, "PF": 2.0},
            {"Player": "Player A", "Year": 2016, "Age": 26, "PTS": 24.0, "FG": 9.0, "FGA": 18.0, "FT": 4.0, "FTA": 4.5, "ORB": 1.5, "DRB": 4.0, "STL": 1.5, "AST": 5.0, "BLK": 0.5, "TOV": 2.0, "PF": 2.0},
            {"Player": "Player B", "Year": 2014, "Age": 30, "PTS": 20.0, "FG": 7.5, "FGA": 15.0, "FT": 3.0, "FTA": 3.5, "ORB": 1.0, "DRB": 4.0, "STL": 1.0, "AST": 4.0, "BLK": 0.3, "TOV": 2.0, "PF": 2.0},
            {"Player": "Player B", "Year": 2015, "Age": 31, "PTS": 12.0, "FG": 4.5, "FGA": 10.0, "FT": 2.0, "FTA": 2.5, "ORB": 0.8, "DRB": 3.0, "STL": 0.8, "AST": 2.5, "BLK": 0.2, "TOV": 1.5, "PF": 2.0},
            # Sin datos de contrato (jugador no está en nba_contracts_history) -- no debe romper nada.
            {"Player": "Player C", "Year": 2014, "Age": 28, "PTS": 10.0, "FG": 4.0, "FGA": 9.0, "FT": 1.0, "FTA": 1.5, "ORB": 0.5, "DRB": 2.0, "STL": 0.5, "AST": 2.0, "BLK": 0.1, "TOV": 1.0, "PF": 1.5},
        ]
    )
    salaries.to_csv(raw / cye.SALARY_STATS_FILENAME, index=False)

    contracts = pd.DataFrame(
        [
            {"NAME": "Player A", "CONTRACT_START": 2013, "CONTRACT_END": 2016},
            {"NAME": "Player B", "CONTRACT_START": 2013, "CONTRACT_END": 2015},
            {"NAME": "Player D", "CONTRACT_START": 2014, "CONTRACT_END": 2015},  # 1 temporada -- descartado
        ]
    )
    contracts.to_csv(raw / cye.CONTRACTS_FILENAME, index=False)

    return {"paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(tmp_path / "processed")}}


def test_load_contracts_drops_contracts_shorter_than_minimum_span(contract_config):
    contracts = cye.load_contracts(contract_config)

    assert set(contracts["player"]) == {"Player A", "Player B"}


def test_build_contract_year_panel_flags_final_season_correctly(contract_config):
    panel = cye.build_contract_year_panel(contract_config)

    player_a = panel[panel["player"] == "Player A"].sort_values("season_end_year")
    assert list(player_a["season_end_year"]) == [2014, 2015, 2016]
    assert list(player_a["is_final_year"]) == [False, False, True]


def test_build_contract_year_panel_excludes_players_without_contract_data(contract_config):
    panel = cye.build_contract_year_panel(contract_config)

    assert "Player C" not in set(panel["player"])


def test_paired_delta_test_sign_matches_direction_of_synthetic_effect(contract_config):
    panel = cye.build_contract_year_panel(contract_config)

    deltas, _ = cye.paired_delta_test(panel)
    delta_a = deltas[deltas["player"] == "Player A"]["delta"].iloc[0]
    delta_b = deltas[deltas["player"] == "Player B"]["delta"].iloc[0]

    assert delta_a > 0  # año final mucho mejor
    assert delta_b < 0  # año final peor
