"""
Test del único trozo PURO y barato de testear de
scripts/experiments/bayesian_calibration.py -- build_calibration_features().
No cubre fit_hierarchical_model() (requiere pymc + sampling real, es
lento y es la parte "experimento", no la parte "lógica que puede
romperse en silencio") -- ver el propio docstring del módulo: es un
experimento exploratorio, no un módulo de producción de src/, así que el
test se limita a la extracción de features (el join real_standings +
la conversión de unidades), que sí es lógica susceptible de un bug
silencioso (join mal hecho, columna mal nombrada...).

Usa DataFrames sintéticos con el esquema de backtest_sweep_rosters.csv /
_player_career_stats.csv / _standings.csv -- mismo patrón que
tests/test_backtesting.py, no requiere red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID  # noqa: E402

import bayesian_calibration as bc  # noqa: E402


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

    # Solo se rellena roster/stats de 2 de los 30 equipos reales -- los
    # otros 28 casos de esa temporada deben SALTARSE sin abortar (mismo
    # comportamiento resiliente que el resto del backtest sweep).
    rosters = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER": "Star A", "AGE": 26, "comparable_name": f"BOS {season}", "season": season},
            {"PLAYER_ID": 2, "PLAYER": "Star B", "AGE": 27, "comparable_name": f"MIA {season}", "season": season},
        ]
    )
    player_stats = pd.concat(
        [
            _career_seasons([
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
            {"TeamID": bos_id, "season": season, "DiffPointsPG": 6.5, "WINS": 56, "LOSSES": 26},
            {"TeamID": mia_id, "season": season, "DiffPointsPG": 3.1, "WINS": 50, "LOSSES": 32},
        ]
    )

    rosters.to_csv(processed / "backtest_sweep_rosters.csv", index=False)
    player_stats.to_csv(processed / "backtest_sweep_player_career_stats.csv", index=False)
    standings.to_csv(processed / "backtest_sweep_standings.csv", index=False)

    return {
        "team": {"team_id": 999, "season": "2026-27"},
        "backtest_sweep": {"seasons": [season]},
        "lineup_synergy": {},
        "monte_carlo": {},
        "league_simulation": {},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }


def test_build_calibration_features_joins_real_standings_and_skips_teams_without_data(sweep_config):
    features = bc.build_calibration_features(sweep_config)

    # De los 30 equipos reales de esa temporada, solo BOS y MIA tenían
    # roster/stats sintéticos -- el resto se salta en silencio (aviso,
    # no excepción) en vez de abortar todo el experimento.
    assert len(features) == 2
    assert set(features["comparable_name"]) == {"BOS 2010-11", "MIA 2010-11"}

    bos_row = features[features["comparable_name"] == "BOS 2010-11"].iloc[0]
    assert bos_row["y_actual_diff_points_pg"] == pytest.approx(6.5)
    # x = Game Score esperado del equipo MENOS la línea base de liga esa
    # temporada -- por construcción (compute_projected_league_baselines)
    # la MEDIA de x entre los dos equipos del test no tiene por qué ser
    # exactamente 0 (solo 2 de 30 equipos reales), pero debe ser un
    # número finito, no NaN/inf, y BOS (mejor roster sintético, más
    # minutos/puntos) debe proyectar un x mayor que MIA.
    mia_row = features[features["comparable_name"] == "MIA 2010-11"].iloc[0]
    assert bos_row["x_game_score_vs_baseline"] > mia_row["x_game_score_vs_baseline"]
    assert pd.notna(bos_row["x_game_score_vs_baseline"])


def test_build_calibration_features_raises_a_clear_error_without_backtest_sweep_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    config = {
        "team": {"team_id": 999},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }
    with pytest.raises(ValueError, match="backtest_sweep"):
        bc.build_calibration_features(config)


# ---------------------------------------------------------------------------
# run_loso_validation() -- requiere pymc/arviz, deliberadamente FUERA de
# requirements.txt (ver scripts/experiments/requirements-experiments.txt).
# Se salta con importorskip en vez de fallar en cualquier entorno que solo
# tenga instaladas las dependencias del simulador -- mismo espíritu que
# @pytest.mark.network para tests que necesitan algo que no todo el mundo
# tiene. Usa draws/tune mínimos (100/100, 2 chains) -- este test verifica
# la ORQUESTACIÓN del leave-one-season-out (una predicción out-of-fold por
# fila, sin fugas de una temporada a su propio pliegue de entrenamiento),
# no la calidad estadística del ajuste -- eso ya se verificó manualmente
# contra los 480 casos reales.
# ---------------------------------------------------------------------------

pytest.importorskip("pymc")
pytest.importorskip("arviz")


def _synthetic_calibration_features(n_seasons=3, n_teams=6, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seasons):
        season = f"20{10 + s}-{11 + s}"
        x = rng.normal(0, 15, size=n_teams)
        y = 0.2 * x + rng.normal(0, 2, size=n_teams)  # relación real conocida, para poder comprobar signo
        for i in range(n_teams):
            rows.append({
                "comparable_name": f"TEAM{i} {season}", "team_id": 1000 + i, "season": season,
                "x_game_score_vs_baseline": x[i], "y_actual_diff_points_pg": y[i],
                "WINS": 41, "LOSSES": 41,
            })
    return pd.DataFrame(rows)


def test_run_loso_validation_produces_one_out_of_fold_prediction_per_row():
    features = _synthetic_calibration_features()

    results, diagnostics = bc.run_loso_validation(
        features, current_scale=0.21, draws=100, tune=100, chains=2, seed=1,
    )

    # Una predicción por cada fila de entrada -- ninguna temporada se
    # queda sin predecir ni se duplica entre pliegues.
    assert len(results) == len(features)
    assert set(results["held_out_season"]) == set(features["season"])
    assert len(diagnostics) == features["season"].nunique()

    # El fold de cada temporada aprendió un beta_mu SIN los datos de esa
    # misma temporada -- con la relación sintética conocida (y = 0.2x +
    # ruido) el signo tiene que salir positivo en los tres pliegues,
    # aunque el valor exacto varíe por el ruido de muestreo.
    assert (results.groupby("held_out_season")["beta_mu_fold"].first() > 0).all()

    assert results["pred_bayes_loso"].notna().all()
    assert results["pred_fixed_current"].notna().all()
