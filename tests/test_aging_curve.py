"""
Tests de aging_curve.py. Usan DataFrames sintéticos con el mismo esquema
que roster_career_stats.csv (SEASON_ID, PLAYER_AGE, MIN, PTS, AST, REB,
STL, BLK, TOV, FG3M, FG3A) -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aging_curve import (  # noqa: E402
    DEFAULT_GENERAL_AGE_CURVE,
    DEFAULT_SHOOTING_AGE_CURVE,
    build_aging_projection_dataset,
    compute_age_adjustment_factor,
    compute_league_game_score_baseline,
    compute_game_score_per36,
    compute_per36_stats,
    compute_recency_weighted_baseline,
    project_player_season,
    zero_player_projection,
)


def _seasons(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SEASON_ID": r["season"],
                "PLAYER_AGE": r["age"],
                "GP": r.get("gp", 72),
                "MIN": r.get("min", 2000),
                "PTS": r.get("pts", 720),
                "AST": r.get("ast", 200),
                "REB": r.get("reb", 300),
                "STL": r.get("stl", 60),
                "BLK": r.get("blk", 30),
                "TOV": r.get("tov", 100),
                "FG3M": r.get("fg3m", 80),
                "FG3A": r.get("fg3a", 220),
                "OREB": r.get("oreb", 60),
                "DREB": r.get("dreb", 240),
                "FGM": r.get("fgm", 280),
                "FGA": r.get("fga", 600),
                "FTM": r.get("ftm", 160),
                "FTA": r.get("fta", 200),
                "PF": r.get("pf", 150),
            }
            for r in rows
        ]
    )


def test_per36_normalizes_by_minutes():
    seasons = _seasons([{"season": "2022-23", "age": 25, "min": 720, "pts": 360}])
    df = compute_per36_stats(seasons)
    # 360 pts en 720 min -> 18 pts por 36 min.
    assert df["PTS_per36"].iloc[0] == pytest.approx(18.0)


def test_recency_weighted_baseline_favors_recent_seasons():
    seasons = _seasons(
        [
            {"season": "2021-22", "age": 25, "min": 2000, "pts": 400},  # bajo volumen
            {"season": "2022-23", "age": 26, "min": 2000, "pts": 1400},  # alto volumen, reciente
        ]
    )
    baseline = compute_recency_weighted_baseline(seasons, n_seasons=2, half_life_seasons=1.0)
    # Media simple sería (7.2+25.2)/2=16.2; ponderado por recencia debe
    # acercarse más al valor de la temporada reciente (25.2).
    assert baseline["PTS_per36"] > 16.2


def test_age_adjustment_factor_grows_for_young_rising_player():
    factor = compute_age_adjustment_factor(20, 21, DEFAULT_GENERAL_AGE_CURVE)
    assert factor > 1.0


def test_age_adjustment_factor_declines_for_aging_veteran():
    factor = compute_age_adjustment_factor(37, 38, DEFAULT_GENERAL_AGE_CURVE)
    assert factor < 1.0


def test_age_adjustment_factor_flat_at_peak():
    factor = compute_age_adjustment_factor(26, 27, DEFAULT_GENERAL_AGE_CURVE)
    assert factor == pytest.approx(1.0)


def test_age_adjustment_factor_no_change_when_target_not_older():
    assert compute_age_adjustment_factor(30, 30, DEFAULT_GENERAL_AGE_CURVE) == pytest.approx(1.0)
    assert compute_age_adjustment_factor(30, 28, DEFAULT_GENERAL_AGE_CURVE) == pytest.approx(1.0)


def test_age_adjustment_factor_compounds_over_multiple_years():
    one_year = compute_age_adjustment_factor(29, 30, DEFAULT_GENERAL_AGE_CURVE)
    two_years = compute_age_adjustment_factor(29, 31, DEFAULT_GENERAL_AGE_CURVE)
    # Dos años de declive consecutivo deben componer a un factor más bajo
    # que un solo año.
    assert two_years < one_year < 1.0


def test_shooting_curve_peaks_later_than_general_curve():
    # A los 29 años, la curva general ya está en declive claro mientras la
    # de tiro exterior todavía está cerca de su pico (declive tardío, ~30).
    general_factor = compute_age_adjustment_factor(29, 30, DEFAULT_GENERAL_AGE_CURVE)
    shooting_factor = compute_age_adjustment_factor(29, 30, DEFAULT_SHOOTING_AGE_CURVE)
    assert shooting_factor > general_factor


def test_project_player_season_scales_by_minutes_and_games():
    seasons = _seasons(
        [
            {"season": "2022-23", "age": 26, "min": 2952, "pts": 1600},  # 19.5 pts/36
            {"season": "2023-24", "age": 27, "min": 2952, "pts": 1600},
        ]
    )
    projection = project_player_season(
        seasons, target_age=28, minutes_per_game=36.0, games_per_season=82
    )

    assert projection["projected_total_minutes"] == pytest.approx(36.0 * 82)
    # PTS_projected debe ser coherente con PTS_per36_projected * minutos totales / 36.
    expected_pts = projection["PTS_per36_projected"] / 36 * projection["projected_total_minutes"]
    assert projection["PTS_projected"] == pytest.approx(expected_pts)


def test_project_player_season_includes_game_score():
    seasons = _seasons([{"season": "2023-24", "age": 27, "min": 2952, "pts": 1600}])
    projection = project_player_season(seasons, target_age=28, minutes_per_game=36.0, games_per_season=82)
    assert "game_score_per36" in projection


def test_game_score_rewards_efficient_scoring_over_inefficient_volume():
    efficient = {
        "PTS_per36": 25.0, "FGM_per36": 10.0, "FGA_per36": 16.0, "FTA_per36": 5.0, "FTM_per36": 5.0,
        "OREB_per36": 1.0, "DREB_per36": 5.0, "STL_per36": 1.5, "AST_per36": 5.0, "BLK_per36": 0.5,
        "PF_per36": 2.0, "TOV_per36": 2.0,
    }
    inefficient = {**efficient, "FGA_per36": 26.0, "FGM_per36": 10.0}  # mismos puntos, muchos más tiros fallados

    assert compute_game_score_per36(efficient) > compute_game_score_per36(inefficient)


def test_zero_player_projection_has_same_keys_as_project_player_season():
    real = project_player_season(
        _seasons([{"season": "2023-24", "age": 27, "min": 2000, "pts": 1200}]),
        target_age=28, minutes_per_game=20.0, games_per_season=82,
    )
    zero = zero_player_projection(target_age=20, minutes_per_game=10.0, games_per_season=82)

    assert set(zero.keys()) == set(real.keys())
    assert zero["game_score_per36"] == 0.0
    assert zero["PTS_projected"] == 0.0
    assert zero["projected_total_minutes"] == 10.0 * 82


def test_build_aging_projection_dataset_gives_zero_floor_to_rookie_with_no_history(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    # Solo el veterano tiene temporadas registradas -- el rookie no aparece
    # en absoluto en roster_career_stats.csv (0 partidos de liga regular).
    _seasons(
        [{"player_id": 1, "season": "2023-24", "age": 27, "min": 2000, "pts": 1200}]
    ).assign(PLAYER_ID=1).to_csv(processed / "roster_career_stats.csv", index=False)

    config = {
        "team": {"season": "2026-27"},
        "roster": [
            {"player_id": 1, "name": "Veteran Player", "minutes_projection": 30},
            {"player_id": 2, "name": "True Rookie", "minutes_projection": 10},
        ],
        "simulation": {"games_per_season": 82},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }

    result = build_aging_projection_dataset(config)

    assert len(result) == 2
    rookie_row = result[result["player_id"] == 2].iloc[0]
    assert rookie_row["player_name"] == "True Rookie"
    assert rookie_row["game_score_per36"] == 0.0
    assert rookie_row["projected_total_minutes"] == 10.0 * 82


def test_compute_league_game_score_baseline_returns_one_row_per_season():
    career = pd.concat(
        [
            _seasons([{"player_id": 1, "season": "2015-16", "age": 25, "min": 2000, "pts": 1000}]),
            _seasons([{"player_id": 2, "season": "2015-16", "age": 27, "min": 2400, "pts": 1400}]),
            _seasons([{"player_id": 1, "season": "2016-17", "age": 26, "min": 2200, "pts": 1300}]),
        ],
        ignore_index=True,
    )

    baseline = compute_league_game_score_baseline(career, min_minutes=500)

    assert baseline["season"].tolist() == ["2015-16", "2016-17"]
    assert baseline.loc[baseline["season"] == "2015-16", "n_players"].iloc[0] == 2
    assert (baseline["league_game_score_per36"] > 0).all()


def test_compute_league_game_score_baseline_excludes_low_minute_players():
    career = pd.concat(
        [
            _seasons([{"player_id": 1, "season": "2015-16", "age": 25, "min": 2000, "pts": 1000}]),
            # 100 minutos: por debajo del umbral, no debe contar
            _seasons([{"player_id": 2, "season": "2015-16", "age": 22, "min": 100, "pts": 200}]),
        ],
        ignore_index=True,
    )

    baseline = compute_league_game_score_baseline(career, min_minutes=500)

    assert baseline["n_players"].iloc[0] == 1


def test_compute_league_game_score_baseline_weights_by_minutes():
    # Dos jugadores con producción por-36 muy distinta: el de más minutos
    # debe dominar la media ponderada.
    career = pd.concat(
        [
            _seasons([{"player_id": 1, "season": "2015-16", "age": 25, "min": 3000, "pts": 1500}]),
            _seasons([{"player_id": 2, "season": "2015-16", "age": 25, "min": 600, "pts": 100}]),
        ],
        ignore_index=True,
    )

    weighted = compute_league_game_score_baseline(career, min_minutes=500)["league_game_score_per36"].iloc[0]
    only_heavy = compute_league_game_score_baseline(
        career[career["MIN"] >= 3000], min_minutes=500
    )["league_game_score_per36"].iloc[0]
    only_light = compute_league_game_score_baseline(
        career[career["MIN"] < 3000], min_minutes=500
    )["league_game_score_per36"].iloc[0]

    # La media ponderada queda entre ambos pero mucho más cerca del que juega 3000 min.
    assert only_light < weighted < only_heavy
    assert abs(weighted - only_heavy) < abs(weighted - only_light)
