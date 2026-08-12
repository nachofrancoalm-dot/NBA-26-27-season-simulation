"""
Test de la parte pura y barata de testear de
scripts/experiments/aging_curve_shrinkage.py -- build_grid_features() y
evaluate_grid(). No cubre run_loso() (repite evaluate_grid 16 veces sobre
datos reales, es la parte "experimento", no lógica que pueda romperse en
silencio) -- ver el docstring del módulo: experimento exploratorio, no
módulo de producción de src/.

Usa DataFrames sintéticos con el esquema de backtest_sweep_rosters.csv /
_player_career_stats.csv / _standings.csv -- mismo patrón que
tests/test_experiments_bayesian_calibration.py, no requiere red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID  # noqa: E402

import aging_curve_shrinkage as acs  # noqa: E402


def _career_seasons(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_ID": r["player_id"], "SEASON_ID": r["season"], "PLAYER_AGE": r["age"],
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
def sweep_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    season = "2010-11"
    bos_id, mia_id = ABBREVIATION_TO_TEAM_ID["BOS"], ABBREVIATION_TO_TEAM_ID["MIA"]

    rosters = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER": "Star A", "AGE": 26, "comparable_name": f"BOS {season}", "season": season},
            {"PLAYER_ID": 2, "PLAYER": "Star B", "AGE": 27, "comparable_name": f"MIA {season}", "season": season},
        ]
    )
    player_stats = pd.concat(
        [
            _career_seasons([
                {"player_id": 1, "season": "2008-09", "age": 24, "pts": 1400, "min": 2800},
                {"player_id": 1, "season": "2009-10", "age": 25, "pts": 1800, "min": 2800},
                {"player_id": 1, "season": season, "age": 26, "pts": 2000, "min": 3000, "gp": 80},
            ]),
            _career_seasons([
                {"player_id": 2, "season": "2009-10", "age": 26, "pts": 1200, "min": 2400},
                {"player_id": 2, "season": season, "age": 27, "pts": 1300, "min": 2500, "gp": 78},
            ]),
        ],
        ignore_index=True,
    )
    standings = pd.DataFrame(
        [
            {"TeamID": bos_id, "season": season, "DiffPointsPG": 6.5},
            {"TeamID": mia_id, "season": season, "DiffPointsPG": 3.1},
        ]
    )

    rosters.to_csv(processed / "backtest_sweep_rosters.csv", index=False)
    player_stats.to_csv(processed / "backtest_sweep_player_career_stats.csv", index=False)
    standings.to_csv(processed / "backtest_sweep_standings.csv", index=False)

    return {
        "team": {"team_id": 999, "season": "2026-27"},
        "backtest_sweep": {"seasons": [season]},
        "league_simulation": {},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }


def test_build_grid_features_returns_one_row_per_case_and_grid_point(sweep_config):
    grid = [(1, 0.5), (3, 1.5)]
    features = acs.build_grid_features(sweep_config, grid)

    # 2 equipos con datos x 2 combinaciones del grid = 4 filas.
    assert len(features) == 4
    assert set(features["n_seasons_lookback"]) == {1, 3}
    assert set(zip(features["comparable_name"], features["n_seasons_lookback"])) == {
        ("BOS 2010-11", 1), ("BOS 2010-11", 3), ("MIA 2010-11", 1), ("MIA 2010-11", 3),
    }
    # BOS tiene 3 temporadas previas (una con PTS mucho más bajo) -- con
    # n_seasons_lookback=1 (solo la más reciente) su Game Score de equipo
    # debe ser distinto que con n_seasons_lookback=3 (promediado con la
    # temporada floja).
    bos = features[features["comparable_name"] == "BOS 2010-11"].set_index("n_seasons_lookback")
    assert bos.loc[1, "team_game_score"] != pytest.approx(bos.loc[3, "team_game_score"])


def test_evaluate_grid_ranks_by_closeness_to_target_spread(sweep_config):
    grid = [(1, 0.5), (3, 1.5)]
    features = acs.build_grid_features(sweep_config, grid)
    ranked = acs.evaluate_grid(features)

    assert len(ranked) == 2
    # Ordenado por |gap_to_target| ascendente -- la combinación más
    # cercana al objetivo de dispersión va primero.
    assert ranked["gap_to_target"].abs().is_monotonic_increasing
    assert {"talent_std_pts", "target_talent_std_pts", "correlation"}.issubset(ranked.columns)
