"""
Tests de build_survival_features() y heuristic_expected_games() en
injury_survival_model.py. No cubre run_loso() (ajuste de Cox real, lento).
lifelines es una dependencia opcional, de ahí el importorskip.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "experiments"))

pytest.importorskip("lifelines")

import injury_survival_model as ism  # noqa: E402


def _career_seasons(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_ID": r.get("player_id", 1), "SEASON_ID": r["season"], "PLAYER_AGE": r["age"],
                "GP": r["gp"], "MIN": r.get("min", r["gp"] * 30),
            }
            for r in rows
        ]
    )


def test_build_survival_features_uses_only_prior_seasons_no_look_ahead():
    # si hubiera look-ahead, historical_load_prior/recency_prior reflejarían el GP=10 de la temporada objetivo
    seasons = _career_seasons([
        {"season": "2017-18", "age": 25, "gp": 82},
        {"season": "2018-19", "age": 26, "gp": 10},  # temporada objetivo, no debe leerse
    ])
    features = ism.build_survival_features(seasons, ["2018-19"])

    assert len(features) == 1
    row = features.iloc[0]
    assert row["historical_load_prior"] == pytest.approx(0.0)
    assert row["recency_prior"] == pytest.approx(0.0)


def test_build_survival_features_duration_and_event_reflect_real_gp():
    seasons = _career_seasons([
        {"season": "2017-18", "age": 25, "gp": 82},
        {"season": "2018-19", "age": 26, "gp": 60},  # temporada objetivo -- se perdió el 82-60=22
    ])
    features = ism.build_survival_features(seasons, ["2018-19"])
    row = features.iloc[0]
    assert row["duration"] == 60
    assert row["event"] == 1  # GP < season_length -- "evento" observado


def test_build_survival_features_marks_full_attendance_as_censored():
    seasons = _career_seasons([
        {"season": "2017-18", "age": 25, "gp": 82},
        {"season": "2018-19", "age": 26, "gp": 82},  # jugó los 82 -- censurado, no "cero riesgo"
    ])
    features = ism.build_survival_features(seasons, ["2018-19"])
    row = features.iloc[0]
    assert row["duration"] == 82
    assert row["event"] == 0


def test_build_survival_features_skips_players_without_any_prior_season():
    seasons = _career_seasons([{"season": "2020-21", "age": 20, "gp": 82}])
    features = ism.build_survival_features(seasons, ["2020-21"])
    assert features.empty


def test_build_survival_features_mpg_prior_uses_the_most_recent_prior_season():
    seasons = _career_seasons([
        {"season": "2018-19", "age": 24, "gp": 82, "min": 82 * 20.0},  # más antigua, 20 mpg
        {"season": "2019-20", "age": 25, "gp": 82, "min": 82 * 35.0},  # más reciente, 35 mpg
        {"season": "2020-21", "age": 26, "gp": 70},  # objetivo
    ])
    features = ism.build_survival_features(seasons, ["2020-21"])
    assert features.iloc[0]["mpg_prior"] == pytest.approx(35.0)


def test_heuristic_expected_games_matches_the_production_formula():
    # debe reproducir exactamente injury_model.compute_risk_score, o la comparación contra Cox/OLS
    # mediría un heurístico distinto del que hay en producción
    from context.injury_model import DEFAULT_WEIGHTS

    features = pd.DataFrame([{
        "historical_load_prior": 0.4, "recency_prior": 0.3, "age_score": 0.5, "season_length": 82,
    }])
    expected_risk = (
        DEFAULT_WEIGHTS["historical_load"] * 0.4
        + DEFAULT_WEIGHTS["recency"] * 0.3
        + DEFAULT_WEIGHTS["age"] * 0.5
    )
    expected_gp = 82 * (1 - expected_risk)
    result = ism.heuristic_expected_games(features, DEFAULT_WEIGHTS)
    assert result[0] == pytest.approx(expected_gp)
