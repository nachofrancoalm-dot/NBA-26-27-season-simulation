"""
Tests de injury_model.py. Usan DataFrames sintéticos con el mismo esquema
que roster_career_stats.csv (PLAYER_ID, SEASON_ID, PLAYER_AGE, GP,
player_name) -- no requieren red ni el CSV real generado por
data_pipeline.py.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from context.injury_model import (  # noqa: E402
    build_injury_risk_dataset,
    compute_age_score,
    compute_historical_load,
    compute_recency_score,
    compute_risk_score,
    season_length,
)


def _make_seasons(seasons: list[dict]) -> pd.DataFrame:
    """Construye un DataFrame de temporadas de un jugador a partir de dicts
    {season, age, gp}."""
    return pd.DataFrame(
        [
            {"SEASON_ID": s["season"], "PLAYER_AGE": s["age"], "GP": s["gp"]}
            for s in seasons
        ]
    )


def test_healthy_young_player_has_low_risk_score():
    seasons = _make_seasons(
        [
            {"season": "2021-22", "age": 25, "gp": 82},
            {"season": "2022-23", "age": 26, "gp": 82},
            {"season": "2023-24", "age": 27, "gp": 82},
        ]
    )
    result = compute_risk_score(seasons)

    assert result["historical_load_score"] == 0.0
    assert result["recency_score"] == 0.0
    # Sin historial de lesiones, el risk_score viene solo del componente edad.
    assert result["risk_score"] < 0.15
    assert result["risk_score"] == pytest.approx(0.20 * result["age_score"])


def test_recent_absence_weighs_more_than_old_absence():
    # mismo total de partidos perdidos, distinta posición en la ventana de 3 temporadas
    recent_injury = _make_seasons(
        [
            {"season": "2021-22", "age": 25, "gp": 82},
            {"season": "2022-23", "age": 26, "gp": 82},
            {"season": "2023-24", "age": 27, "gp": 40},  # lesión reciente
        ]
    )
    old_injury = _make_seasons(
        [
            {"season": "2021-22", "age": 25, "gp": 40},  # lesión antigua
            {"season": "2022-23", "age": 26, "gp": 82},
            {"season": "2023-24", "age": 27, "gp": 82},
        ]
    )

    # historical_load es un promedio sin ponderar -> debe ser idéntico.
    assert compute_historical_load(recent_injury) == pytest.approx(
        compute_historical_load(old_injury)
    )

    # recency_score sí debe distinguirlos: la ausencia reciente pesa más.
    recency_recent = compute_recency_score(recent_injury)
    recency_old = compute_recency_score(old_injury)
    assert recency_recent > recency_old

    result_recent = compute_risk_score(recent_injury)
    result_old = compute_risk_score(old_injury)
    assert result_recent["risk_score"] > result_old["risk_score"]


def test_age_score_saturates_and_does_not_penalize_extreme_age_further():
    young = compute_age_score(20)
    prime = compute_age_score(28)
    veteran = compute_age_score(36)
    extreme = compute_age_score(41)  # caso LeBron

    assert young < prime < veteran
    # más allá de peak_start_age (32) el score se aplana
    assert veteran == pytest.approx(extreme)
    assert extreme == pytest.approx(compute_age_score(60))  # no sigue subiendo


def test_history_dominates_over_age_for_clean_veteran():
    # veterano de 41 con historial limpio vs. jugador de 27 con lesiones recientes
    clean_veteran = _make_seasons(
        [
            {"season": "2021-22", "age": 39, "gp": 78},
            {"season": "2022-23", "age": 40, "gp": 71},
            {"season": "2023-24", "age": 41, "gp": 70},
        ]
    )
    injury_prone_young = _make_seasons(
        [
            {"season": "2021-22", "age": 25, "gp": 82},
            {"season": "2022-23", "age": 26, "gp": 55},
            {"season": "2023-24", "age": 27, "gp": 45},
        ]
    )

    veteran_result = compute_risk_score(clean_veteran)
    young_result = compute_risk_score(injury_prone_young)

    assert veteran_result["risk_score"] < young_result["risk_score"]


def test_weights_are_configurable_not_hardcoded():
    seasons = _make_seasons(
        [
            {"season": "2021-22", "age": 38, "gp": 82},
            {"season": "2022-23", "age": 39, "gp": 82},
            {"season": "2023-24", "age": 40, "gp": 82},
        ]
    )

    default_result = compute_risk_score(seasons)
    age_only_result = compute_risk_score(
        seasons,
        weights={"historical_load": 0.0, "recency": 0.0, "age": 1.0},
    )

    assert age_only_result["risk_score"] == pytest.approx(age_only_result["age_score"])
    assert age_only_result["risk_score"] != default_result["risk_score"]


def test_lockout_season_games_missed_not_overpenalized():
    # 2011-12 tuvo 66 partidos de calendario (lockout), no 82.
    assert season_length("2011-12") == 66

    lockout_season = _make_seasons([{"season": "2011-12", "age": 25, "gp": 66}])
    assert compute_historical_load(lockout_season) == pytest.approx(0.0)


def test_default_season_length_is_82():
    assert season_length("2023-24") == 82


def test_traded_mid_season_not_double_counted():
    # regression: nba_api añade una fila 'TOT' junto a la de cada equipo tras un traspaso;
    # sin dedupe se infla la ventana de temporadas
    traded_season = pd.DataFrame(
        [
            {"SEASON_ID": "2021-22", "PLAYER_AGE": 27, "GP": 50, "TEAM_ABBREVIATION": "TOT"},
            {"SEASON_ID": "2021-22", "PLAYER_AGE": 27, "GP": 30, "TEAM_ABBREVIATION": "LAL"},
            {"SEASON_ID": "2021-22", "PLAYER_AGE": 27, "GP": 20, "TEAM_ABBREVIATION": "MIA"},
            {"SEASON_ID": "2022-23", "PLAYER_AGE": 28, "GP": 82, "TEAM_ABBREVIATION": "MIA"},
            {"SEASON_ID": "2023-24", "PLAYER_AGE": 29, "GP": 82, "TEAM_ABBREVIATION": "MIA"},
        ]
    )
    result = compute_risk_score(traded_season)

    expected_load = pytest.approx(((1 - 50 / 82) + 0 + 0) / 3)
    assert result["historical_load_score"] == expected_load


def test_build_injury_risk_dataset_gives_zero_floor_to_rookie_with_no_history(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [{"PLAYER_ID": 1, "player_name": "Veteran Player", "SEASON_ID": "2023-24", "PLAYER_AGE": 27, "GP": 80}]
    ).to_csv(processed / "roster_career_stats.csv", index=False)

    config = {
        "roster": [
            {"player_id": 1, "name": "Veteran Player"},
            {"player_id": 2, "name": "True Rookie"},
        ],
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }

    result = build_injury_risk_dataset(config)

    assert len(result) == 2
    rookie_row = result[result["player_id"] == 2].iloc[0]
    assert rookie_row["player_name"] == "True Rookie"
    assert rookie_row["risk_score"] == 0.0
    assert rookie_row["seasons_used"] == 0
