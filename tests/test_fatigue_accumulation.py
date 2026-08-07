"""
Tests de fatigue_accumulation.py. Usan DataFrames sintéticos con el mismo
esquema que roster_career_stats.csv / roster_playoff_career_stats.csv
(SEASON_ID, GP, MIN) -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from context.fatigue_accumulation import (  # noqa: E402
    build_fatigue_dataset,
    compute_cumulative_load_score,
    compute_fatigue_score,
    compute_sustained_streak_score,
    merge_regular_and_playoff_seasons,
)


def _seasons(rows: list[dict], with_team: bool = False) -> pd.DataFrame:
    data = []
    for r in rows:
        row = {"SEASON_ID": r["season"], "GP": r["gp"], "MIN": r["min"]}
        if with_team:
            row["TEAM_ABBREVIATION"] = r.get("team", "TOT")
        data.append(row)
    return pd.DataFrame(data)


def test_young_short_career_has_low_fatigue():
    regular = _seasons(
        [
            {"season": "2022-23", "gp": 60, "min": 900},
            {"season": "2023-24", "gp": 65, "min": 1100},
        ]
    )
    result = compute_fatigue_score(regular, playoff_seasons=None)

    assert result["cumulative_load_score"] < 0.1
    assert result["fatigue_score"] < 0.3


def test_long_high_mileage_career_has_high_cumulative_load():
    # ~20 temporadas de ~2500 min regulares = 50000 min > tope de 35000.
    regular = _seasons(
        [{"season": f"{2003 + i}-{str(4 + i).zfill(2)}", "gp": 78, "min": 2500} for i in range(20)]
    )
    merged = merge_regular_and_playoff_seasons(regular, None)
    score = compute_cumulative_load_score(merged)

    assert score == pytest.approx(1.0)  # capado, no explota más allá de 1.0


def test_recent_intensity_distinguishes_heavy_starter_from_bench_role():
    heavy_starter = _seasons(
        [
            {"season": "2021-22", "gp": 82, "min": 2952},  # 36 min/partido
            {"season": "2022-23", "gp": 82, "min": 2952},
            {"season": "2023-24", "gp": 82, "min": 2952},
        ]
    )
    bench_player = _seasons(
        [
            {"season": "2021-22", "gp": 70, "min": 1050},  # 15 min/partido
            {"season": "2022-23", "gp": 70, "min": 1050},
            {"season": "2023-24", "gp": 70, "min": 1050},
        ]
    )

    result_heavy = compute_fatigue_score(heavy_starter, None)
    result_bench = compute_fatigue_score(bench_player, None)

    assert result_heavy["recent_intensity_score"] > result_bench["recent_intensity_score"]
    assert result_heavy["fatigue_score"] > result_bench["fatigue_score"]


def test_sustained_streak_broken_by_a_light_season():
    no_rest_streak = _seasons(
        [
            {"season": "2021-22", "gp": 78, "min": 2400},
            {"season": "2022-23", "gp": 79, "min": 2400},
            {"season": "2023-24", "gp": 80, "min": 2400},
        ]
    )
    broken_streak = _seasons(
        [
            {"season": "2021-22", "gp": 78, "min": 2400},
            {"season": "2022-23", "gp": 30, "min": 900},  # temporada de descarga/lesión
            {"season": "2023-24", "gp": 80, "min": 2400},
        ]
    )

    merged_no_rest = merge_regular_and_playoff_seasons(no_rest_streak, None)
    merged_broken = merge_regular_and_playoff_seasons(broken_streak, None)

    score_no_rest = compute_sustained_streak_score(merged_no_rest)
    score_broken = compute_sustained_streak_score(merged_broken)

    # La temporada más reciente en ambos casos es completa (streak=1 en
    # broken_streak, porque la racha se corta al llegar a la temporada
    # floja de 2022-23); no_rest_streak acumula 3 temporadas seguidas.
    assert score_no_rest > score_broken


def test_playoff_minutes_add_to_cumulative_load():
    regular = _seasons([{"season": "2023-24", "gp": 82, "min": 2800}])
    playoff = _seasons([{"season": "2023-24", "gp": 20, "min": 800}])

    merged_with_playoffs = merge_regular_and_playoff_seasons(regular, playoff)
    merged_without_playoffs = merge_regular_and_playoff_seasons(regular, None)

    assert merged_with_playoffs["MIN_total"].iloc[0] == pytest.approx(3600)
    assert merged_without_playoffs["MIN_total"].iloc[0] == pytest.approx(2800)


def test_weights_are_configurable_not_hardcoded():
    regular = _seasons(
        [
            {"season": "2021-22", "gp": 82, "min": 2952},
            {"season": "2022-23", "gp": 82, "min": 2952},
            {"season": "2023-24", "gp": 82, "min": 2952},
        ]
    )

    default_result = compute_fatigue_score(regular, None)
    load_only_result = compute_fatigue_score(
        regular,
        None,
        weights={"cumulative_load": 1.0, "recent_intensity": 0.0, "sustained_streak": 0.0},
    )

    assert load_only_result["fatigue_score"] == pytest.approx(
        load_only_result["cumulative_load_score"]
    )
    assert load_only_result["fatigue_score"] != default_result["fatigue_score"]


def test_traded_mid_season_not_double_counted_in_cumulative_load():
    regular = _seasons(
        [
            {"season": "2021-22", "gp": 50, "min": 1500, "team": "TOT"},
            {"season": "2021-22", "gp": 30, "min": 900, "team": "LAL"},
            {"season": "2021-22", "gp": 20, "min": 600, "team": "MIA"},
        ],
        with_team=True,
    )
    merged = merge_regular_and_playoff_seasons(regular, None)

    assert merged["MIN_total"].sum() == pytest.approx(1500)


def test_build_fatigue_dataset_gives_zero_floor_to_rookie_with_no_history(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [{"PLAYER_ID": 1, "player_name": "Veteran Player", "SEASON_ID": "2023-24", "GP": 80, "MIN": 2800}]
    ).to_csv(processed / "roster_career_stats.csv", index=False)

    config = {
        "roster": [
            {"player_id": 1, "name": "Veteran Player"},
            {"player_id": 2, "name": "True Rookie"},
        ],
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }

    result = build_fatigue_dataset(config)

    assert len(result) == 2
    rookie_row = result[result["player_id"] == 2].iloc[0]
    assert rookie_row["player_name"] == "True Rookie"
    assert rookie_row["fatigue_score"] == 0.0
