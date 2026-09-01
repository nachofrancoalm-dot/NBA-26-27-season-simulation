"""Tests de lineup_synergy.py con perfiles de estilo y matrices sintéticas."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lineup_synergy import (  # noqa: E402
    build_synergy_matrix,
    compute_game_synergy_adjustment,
    compute_playmaking_spacing_synergy,
    compute_usage_clash,
)


def test_usage_clash_zero_when_both_below_threshold():
    assert compute_usage_clash(10.0, 12.0, threshold=18.0) == 0.0


def test_usage_clash_zero_when_only_one_above_threshold():
    assert compute_usage_clash(25.0, 12.0, threshold=18.0) == 0.0


def test_usage_clash_positive_and_grows_with_both_excesses():
    small_clash = compute_usage_clash(19.0, 19.0, threshold=18.0)
    big_clash = compute_usage_clash(26.0, 24.0, threshold=18.0)
    assert small_clash > 0
    assert big_clash > small_clash


def test_playmaking_spacing_synergy_rewards_facilitator_shooter_pairing():
    facilitator_shooter = compute_playmaking_spacing_synergy(
        playmaking_i=8.0, spacing_i=1.0, playmaking_j=1.0, spacing_j=9.0
    )
    two_non_playmaking_non_shooters = compute_playmaking_spacing_synergy(
        playmaking_i=1.0, spacing_i=1.0, playmaking_j=1.0, spacing_j=1.0
    )
    assert facilitator_shooter > two_non_playmaking_non_shooters


def test_build_synergy_matrix_is_symmetric_with_zero_diagonal():
    player_ids = [1, 2, 3]
    profiles = {
        1: {"usage": 25.0, "playmaking": 8.0, "spacing": 2.0, "interior": 1.0},
        2: {"usage": 22.0, "playmaking": 2.0, "spacing": 8.0, "interior": 2.0},
        3: {"usage": 10.0, "playmaking": 3.0, "spacing": 1.0, "interior": 9.0},
    }
    minutes = {1: 34.0, 2: 30.0, 3: 20.0}

    matrix = build_synergy_matrix(player_ids, profiles, minutes)

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix, matrix.T)


def test_build_synergy_matrix_penalizes_two_high_usage_players_sharing_minutes():
    player_ids = [1, 2]
    high_usage_pair = {
        1: {"usage": 28.0, "playmaking": 3.0, "spacing": 3.0, "interior": 1.0},
        2: {"usage": 26.0, "playmaking": 3.0, "spacing": 3.0, "interior": 1.0},
    }
    one_high_one_low_usage = {
        1: {"usage": 28.0, "playmaking": 3.0, "spacing": 3.0, "interior": 1.0},
        2: {"usage": 10.0, "playmaking": 3.0, "spacing": 3.0, "interior": 1.0},
    }
    minutes = {1: 34.0, 2: 34.0}

    clash_matrix = build_synergy_matrix(player_ids, high_usage_pair, minutes, playmaking_spacing_weight=0.0)
    no_clash_matrix = build_synergy_matrix(
        player_ids, one_high_one_low_usage, minutes, playmaking_spacing_weight=0.0
    )

    assert clash_matrix[0, 1] < no_clash_matrix[0, 1]


def test_pair_weight_scales_with_lower_of_the_two_minutes():
    player_ids = [1, 2]
    profiles = {
        1: {"usage": 10.0, "playmaking": 8.0, "spacing": 1.0, "interior": 1.0},
        2: {"usage": 10.0, "playmaking": 1.0, "spacing": 8.0, "interior": 1.0},
    }
    full_minutes = build_synergy_matrix(player_ids, profiles, {1: 36.0, 2: 36.0}, usage_clash_weight=0.0)
    bench_minutes = build_synergy_matrix(player_ids, profiles, {1: 36.0, 2: 6.0}, usage_clash_weight=0.0)

    assert abs(bench_minutes[0, 1]) < abs(full_minutes[0, 1])


def test_game_synergy_adjustment_zero_when_synergy_player_unavailable():
    synergy_matrix = np.array([[0.0, 5.0], [5.0, 0.0]])
    available_both = np.ones((1, 1, 2), dtype=bool)
    available_one_out = np.array([[[True, False]]])

    adj_both = compute_game_synergy_adjustment(available_both, synergy_matrix)
    adj_one_out = compute_game_synergy_adjustment(available_one_out, synergy_matrix)

    assert adj_both[0, 0] == pytest.approx(10.0)  # quadratic form: 2 * 5.0 off-diagonal
    assert adj_one_out[0, 0] == pytest.approx(0.0)


def test_game_synergy_adjustment_is_clipped_to_configured_range():
    synergy_matrix = np.array([[0.0, 100.0], [100.0, 0.0]])
    available = np.ones((1, 1, 2), dtype=bool)

    adjustment = compute_game_synergy_adjustment(available, synergy_matrix, min_adjustment=-5.0, max_adjustment=5.0)
    assert adjustment[0, 0] == pytest.approx(5.0)
