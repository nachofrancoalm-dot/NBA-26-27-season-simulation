"""Tests de simulation.py con arrays de numpy sintéticos, sin red ni CSV reales."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simulation import (
    load_league_mean_synergy,  # noqa: E402
    DEFAULT_INJURY_TYPE_CATEGORIES,
    DEFAULT_MONTE_CARLO_CONFIG,
    TOTAL_TEAM_MINUTES_PER_GAME,
    apply_star_bonus,
    build_simulation_dataset,
    categorize_injury_absence,
    normalize_rotation_minutes,
    compute_game_net_rating_estimate,
    compute_league_average_game_score_per36,
    compute_player_contributions,
    compute_win_probabilities,
    run_monte_carlo,
    sample_injury_absences,
    sample_schedule_context,
    sample_team_quality_noise,
    simulate_single_season_player_log,
    compute_expected_games_played,
    compute_expected_effective_minutes_per_game,
)


def test_sample_injury_absences_is_reproducible_with_same_seed():
    risk_scores = np.array([0.1, 0.3])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a1 = sample_injury_absences(risk_scores, 100, 82, rng1, dispersion=2.0)
    a2 = sample_injury_absences(risk_scores, 100, 82, rng2, dispersion=2.0)
    assert np.array_equal(a1, a2)


def test_zero_risk_player_is_always_available():
    risk_scores = np.array([0.0, 0.5])
    rng = np.random.default_rng(1)
    available = sample_injury_absences(risk_scores, 200, 82, rng, dispersion=2.0)
    assert available[:, :, 0].all()


def test_higher_risk_score_means_more_missed_games_on_average():
    risk_scores = np.array([0.05, 0.40])
    rng = np.random.default_rng(7)
    available = sample_injury_absences(risk_scores, 2000, 82, rng, dispersion=2.0)
    missed_low = (~available[:, :, 0]).sum(axis=1).mean()
    missed_high = (~available[:, :, 1]).sum(axis=1).mean()
    assert missed_high > missed_low


def test_injury_absences_are_contiguous_within_a_season():
    risk_scores = np.array([0.5])
    rng = np.random.default_rng(3)
    available = sample_injury_absences(risk_scores, 50, 82, rng, dispersion=2.0)
    for season in range(50):
        absent = ~available[season, :, 0]
        if not absent.any():
            continue
        indices = np.where(absent)[0]
        # Un solo tramo contiguo: el rango completo (max-min+1) coincide con el nº de partidos.
        assert indices.max() - indices.min() + 1 == len(indices)


def test_schedule_context_first_game_never_back_to_back():
    league_win_pcts = np.array([0.3, 0.5, 0.7])
    rng = np.random.default_rng(5)
    _, is_b2b, _ = sample_schedule_context(league_win_pcts, 100, 82, rng, b2b_probability=0.5)
    assert not is_b2b[:, 0].any()


def test_schedule_context_opponent_win_pct_drawn_from_league_distribution():
    league_win_pcts = np.array([0.2, 0.8])
    rng = np.random.default_rng(9)
    opp_win_pct, _, _ = sample_schedule_context(league_win_pcts, 50, 82, rng, b2b_probability=0.2)
    assert set(np.unique(opp_win_pct)).issubset({0.2, 0.8})


def test_schedule_context_splits_home_and_away_exactly_in_half():
    # Hecho exacto del calendario NBA: 41 en casa y 41 fuera, no "aproximadamente".
    rng = np.random.default_rng(3)
    _, _, is_home = sample_schedule_context(np.array([0.5]), 50, 82, rng, b2b_probability=0.2)

    assert (is_home.sum(axis=1) == 41).all()


def test_home_court_advantage_helps_at_home_and_hurts_away():
    contributions = np.zeros((1, 2, 1))
    opponent_win_pct = np.full((1, 2), 0.5)
    is_home = np.array([[True, False]])
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "home_court_advantage": 3.0}

    net = compute_game_net_rating_estimate(contributions, opponent_win_pct, cfg, is_home=is_home)

    assert net[0, 0] - net[0, 1] == pytest.approx(6.0)  # +3 en casa, -3 fuera


def test_home_court_advantage_is_skipped_when_is_home_is_none():
    contributions = np.zeros((1, 2, 1))
    opponent_win_pct = np.full((1, 2), 0.5)
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "home_court_advantage": 3.0}

    net = compute_game_net_rating_estimate(contributions, opponent_win_pct, cfg, is_home=None)

    assert net[0, 0] == pytest.approx(net[0, 1])


def test_unavailable_player_contributes_zero():
    game_score = np.array([20.0])
    minutes = np.array([36.0])
    fatigue = np.array([0.5])
    available = np.zeros((1, 5, 1), dtype=bool)
    is_b2b = np.zeros((1, 5), dtype=bool)
    rng = np.random.default_rng(1)
    contributions = compute_player_contributions(
        game_score, minutes, fatigue, available, is_b2b, rng, DEFAULT_MONTE_CARLO_CONFIG
    )
    assert (contributions == 0.0).all()


def test_back_to_back_penalizes_high_fatigue_player_more_than_low_fatigue():
    game_score = np.array([20.0, 20.0])
    minutes = np.array([36.0, 36.0])
    fatigue = np.array([0.0, 1.0])  # jugador 0 sin fatiga, jugador 1 con fatiga máxima
    available = np.ones((1, 2, 2), dtype=bool)
    is_b2b = np.array([[False, True]])
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0}  # sin ruido para comparación exacta

    rng = np.random.default_rng(1)
    contributions = compute_player_contributions(game_score, minutes, fatigue, available, is_b2b, rng, cfg)

    assert contributions[0, 1, 1] < contributions[0, 0, 1]  # b2b penaliza al jugador fatigado
    assert contributions[0, 1, 0] == pytest.approx(contributions[0, 0, 0])  # sin fatiga, sin efecto


def test_tougher_opponent_reduces_net_rating_estimate():
    contributions = np.full((1, 2, 1), 10.0)
    opponent_win_pct = np.array([[0.2, 0.8]])
    net_rating = compute_game_net_rating_estimate(contributions, opponent_win_pct, DEFAULT_MONTE_CARLO_CONFIG)
    assert net_rating[0, 0] > net_rating[0, 1]


def test_win_probability_is_half_at_zero_net_rating():
    prob = compute_win_probabilities(np.array([0.0]), outcome_variance_scale=12.0)
    assert prob[0] == pytest.approx(0.5)


def test_win_probability_increases_with_net_rating():
    probs = compute_win_probabilities(np.array([-10.0, 0.0, 10.0]), outcome_variance_scale=12.0)
    assert probs[0] < probs[1] < probs[2]


def test_run_monte_carlo_is_reproducible_with_same_seed():
    kwargs = dict(
        player_ids=[1, 2],
        game_score_per36=np.array([15.0, 10.0]),
        minutes_projection=np.array([32.0, 20.0]),
        risk_scores=np.array([0.1, 0.2]),
        fatigue_scores=np.array([0.3, 0.2]),
        league_win_pcts=np.array([0.3, 0.5, 0.7]),
        n_seasons=50,
        games_per_season=82,
        mc_config=DEFAULT_MONTE_CARLO_CONFIG,
        random_seed=42,
    )
    result1 = run_monte_carlo(**kwargs)
    result2 = run_monte_carlo(**kwargs)
    pd_equal = (result1["wins"].to_numpy() == result2["wins"].to_numpy()).all()
    assert pd_equal


def test_sample_team_quality_noise_is_zero_when_disabled():
    rng = np.random.default_rng(1)
    noise = sample_team_quality_noise(n_seasons=10, std=0.0, rng=rng)
    assert noise.shape == (10, 1)
    assert (noise == 0.0).all()


def test_sample_team_quality_noise_varies_across_seasons():
    rng = np.random.default_rng(1)
    noise = sample_team_quality_noise(n_seasons=500, std=3.0, rng=rng)
    assert noise.shape == (500, 1)
    assert noise.std() == pytest.approx(3.0, rel=0.15)
    assert not np.allclose(noise, noise[0])  # varía de una temporada simulada a otra


def test_run_monte_carlo_wins_mean_unaffected_by_team_quality_uncertainty():
    """Ruido de calidad de equipo es de media cero: no debe mover wins.mean(), solo ensanchar la dispersión."""
    kwargs = dict(
        player_ids=[1, 2], game_score_per36=np.array([15.0, 10.0]), minutes_projection=np.array([32.0, 20.0]),
        risk_scores=np.array([0.1, 0.2]), fatigue_scores=np.array([0.3, 0.2]),
        league_win_pcts=np.array([0.3, 0.5, 0.7]), n_seasons=20000, games_per_season=82, random_seed=42,
    )
    without_noise = run_monte_carlo(mc_config=DEFAULT_MONTE_CARLO_CONFIG, **kwargs)
    with_noise = run_monte_carlo(
        mc_config={**DEFAULT_MONTE_CARLO_CONFIG, "team_quality_uncertainty_std": 4.0}, **kwargs
    )
    assert with_noise["wins"].mean() == pytest.approx(without_noise["wins"].mean(), abs=1.0)


def test_run_monte_carlo_widens_win_distribution_with_team_quality_uncertainty():
    kwargs = dict(
        player_ids=[1, 2], game_score_per36=np.array([15.0, 10.0]), minutes_projection=np.array([32.0, 20.0]),
        risk_scores=np.array([0.1, 0.2]), fatigue_scores=np.array([0.3, 0.2]),
        league_win_pcts=np.array([0.3, 0.5, 0.7]), n_seasons=20000, games_per_season=82, random_seed=42,
    )
    without_noise = run_monte_carlo(mc_config=DEFAULT_MONTE_CARLO_CONFIG, **kwargs)
    with_noise = run_monte_carlo(
        mc_config={**DEFAULT_MONTE_CARLO_CONFIG, "team_quality_uncertainty_std": 4.0}, **kwargs
    )
    assert with_noise["wins"].std() > without_noise["wins"].std()


def test_higher_injury_risk_reduces_average_wins():
    base_kwargs = dict(
        player_ids=[1],
        game_score_per36=np.array([20.0]),
        minutes_projection=np.array([36.0]),
        fatigue_scores=np.array([0.0]),
        league_win_pcts=np.array([0.5]),
        n_seasons=3000,
        games_per_season=82,
        mc_config=DEFAULT_MONTE_CARLO_CONFIG,
        random_seed=42,
    )
    low_risk = run_monte_carlo(risk_scores=np.array([0.02]), **base_kwargs)
    high_risk = run_monte_carlo(risk_scores=np.array([0.5]), **base_kwargs)

    assert high_risk["wins"].mean() < low_risk["wins"].mean()
    assert high_risk["total_games_missed"].mean() > low_risk["total_games_missed"].mean()


def test_synergy_matrix_shifts_simulation_results():
    kwargs = dict(
        player_ids=[1, 2],
        game_score_per36=np.array([15.0, 10.0]),
        minutes_projection=np.array([36.0, 36.0]),
        risk_scores=np.array([0.02, 0.02]),
        fatigue_scores=np.array([0.0, 0.0]),
        league_win_pcts=np.array([0.5]),
        n_seasons=500,
        games_per_season=82,
        mc_config=DEFAULT_MONTE_CARLO_CONFIG,
        random_seed=42,
    )
    no_synergy = run_monte_carlo(**kwargs)
    positive_synergy_matrix = np.array([[0.0, 10.0], [10.0, 0.0]])
    with_synergy = run_monte_carlo(**kwargs, synergy_matrix=positive_synergy_matrix)

    assert with_synergy["net_rating_estimate_mean"].mean() > no_synergy["net_rating_estimate_mean"].mean()


def test_fixed_schedule_overrides_sampling_and_sets_games_per_season():
    fixed_opponent_win_pct = np.array([0.3, 0.9, 0.3])  # calendario fijo de 3 partidos
    fixed_is_back_to_back = np.array([False, True, False])

    result = run_monte_carlo(
        player_ids=[1],
        game_score_per36=np.array([20.0]),
        minutes_projection=np.array([36.0]),
        risk_scores=np.array([0.0]),
        fatigue_scores=np.array([0.0]),
        league_win_pcts=np.array([0.5]),  # ignorado cuando hay fixed_schedule
        n_seasons=200,
        games_per_season=82,  # ignorado cuando hay fixed_schedule
        mc_config=DEFAULT_MONTE_CARLO_CONFIG,
        random_seed=42,
        fixed_schedule=(fixed_opponent_win_pct, fixed_is_back_to_back),
    )
    assert (result["wins"] + result["losses"] == 3).all()


def test_apply_star_bonus_boosts_only_the_top_n_players_by_season_value():
    # 1 estrella, 2 jugadores de banquillo con valor similar y bajo.
    game_score = np.array([25.0, 10.0, 9.0])
    minutes = np.array([36.0, 20.0, 20.0])
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "star_bonus_top_n": 1, "star_bonus_multiplier": 1.2}

    boosted = apply_star_bonus(game_score, minutes, cfg)

    assert boosted[0] == pytest.approx(25.0 * 1.2)
    assert boosted[1] == pytest.approx(10.0)  # sin prima -- no está en el top 1
    assert boosted[2] == pytest.approx(9.0)


def test_apply_star_bonus_disabled_when_top_n_is_zero():
    game_score = np.array([25.0, 10.0])
    minutes = np.array([36.0, 20.0])
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "star_bonus_top_n": 0, "star_bonus_multiplier": 1.5}

    boosted = apply_star_bonus(game_score, minutes, cfg)

    assert boosted is game_score  # devuelve el array original sin copiar, atajo intencional


def test_apply_star_bonus_disabled_when_multiplier_is_one():
    game_score = np.array([25.0, 10.0])
    minutes = np.array([36.0, 20.0])
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "star_bonus_top_n": 2, "star_bonus_multiplier": 1.0}

    boosted = apply_star_bonus(game_score, minutes, cfg)

    assert (boosted == game_score).all()


def test_star_bonus_is_disabled_by_default():
    # Se probó y se descartó: empeoraba el backtesting contra comparables históricos.
    assert DEFAULT_MONTE_CARLO_CONFIG["star_bonus_top_n"] == 0
    assert DEFAULT_MONTE_CARLO_CONFIG["star_bonus_multiplier"] == 1.0


def test_star_bonus_lets_a_top_heavy_team_beat_a_balanced_team_with_equal_total_production_when_enabled():
    # Misma producción total repartida distinto: A con una estrella clara, B pareja.
    game_score_a = np.array([30.0, 5.0, 5.0])
    minutes_a = np.array([36.0, 36.0, 36.0])
    game_score_b = np.array([13.333, 13.333, 13.334])
    minutes_b = np.array([36.0, 36.0, 36.0])

    available = np.ones((1, 1, 3), dtype=bool)
    is_b2b = np.zeros((1, 1), dtype=bool)
    fatigue = np.zeros(3)
    # Prima desactivada por defecto; se activa explícitamente para este test.
    cfg_with_bonus = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0, "star_bonus_top_n": 2, "star_bonus_multiplier": 1.15}

    contributions_a = compute_player_contributions(
        game_score_a, minutes_a, fatigue, available, is_b2b, np.random.default_rng(1), cfg_with_bonus
    )
    contributions_b = compute_player_contributions(
        game_score_b, minutes_b, fatigue, available, is_b2b, np.random.default_rng(1), cfg_with_bonus
    )

    assert contributions_a.sum() > contributions_b.sum()


def test_compute_league_average_game_score_per36_matches_hollinger_when_no_risk_column():
    # Sin columna risk_score, la tasa calibrada coincide con el valor genérico de Hollinger (10.0).
    df = pd.DataFrame(
        [
            {"team_abbreviation": "AAA", "game_score_per36": 10.0, "minutes_projection": 240.0},
            {"team_abbreviation": "BBB", "game_score_per36": 20.0, "minutes_projection": 120.0},
        ]
    )
    rate = compute_league_average_game_score_per36(df)
    assert rate == pytest.approx(10.0)


def test_compute_league_average_game_score_per36_discounts_by_availability():
    # risk_score=0.5 -> disponible la mitad de los partidos en promedio, contribución descontada a la mitad.
    df = pd.DataFrame(
        [{"team_abbreviation": "AAA", "game_score_per36": 10.0, "minutes_projection": 240.0, "risk_score": 0.5}]
    )
    rate = compute_league_average_game_score_per36(df)
    assert rate == pytest.approx(5.0)


def test_compute_league_average_game_score_per36_averages_across_teams():
    df = pd.DataFrame(
        [
            {"team_abbreviation": "AAA", "game_score_per36": 10.0, "minutes_projection": 240.0, "risk_score": 0.0},
            {"team_abbreviation": "BBB", "game_score_per36": 20.0, "minutes_projection": 240.0, "risk_score": 0.0},
        ]
    )
    # Media de los totales de equipo (66.67 y 133.33) = 100 -> tasa equivalente = 15.0.
    rate = compute_league_average_game_score_per36(df)
    assert rate == pytest.approx(15.0)


def _write_base_simulation_files(processed):
    pd.DataFrame(
        [
            {
                "player_id": 1, "game_score_per36": 10.0,
                "FGA_per36_projected": 10.0, "FTA_per36_projected": 2.0, "TOV_per36_projected": 2.0,
                "AST_per36_projected": 5.0, "FG3A_per36_projected": 3.0, "BLK_per36_projected": 0.5,
                "DREB_per36_projected": 4.0,
            }
        ]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "risk_score": 0.0}]).to_csv(processed / "injury_risk.csv", index=False)
    pd.DataFrame([{"player_id": 1, "fatigue_score": 0.0}]).to_csv(processed / "fatigue_risk.csv", index=False)
    pd.DataFrame([{"WinPCT": 0.5}]).to_csv(processed / "prior_season_standings.csv", index=False)


def _simulation_config(tmp_path, processed, extra_monte_carlo=None):
    config = {
        "team": {"season": "2026-27"},
        "roster": [{"player_id": 1, "name": "Solo Player", "minutes_projection": 36}],
        "simulation": {"n_seasons": 50, "games_per_season": 20, "random_seed": 1},
        "lineup_synergy": {},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }
    if extra_monte_carlo:
        config["monte_carlo"] = extra_monte_carlo
    return config


def test_build_simulation_dataset_calibrates_baseline_from_league_projections(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    _write_base_simulation_files(processed)
    config = _simulation_config(tmp_path, processed)

    import simulation as simulation_module

    calls = []
    original = simulation_module.compute_league_average_game_score_per36
    monkeypatch.setattr(
        simulation_module,
        "compute_league_average_game_score_per36",
        lambda df, **kwargs: (calls.append(True), original(df, **kwargs))[1],
    )

    build_simulation_dataset(config)  # sin league_player_projections.csv: no se recalibra
    assert calls == []

    pd.DataFrame(
        [{"team_abbreviation": "AAA", "game_score_per36": 2.0, "minutes_projection": 240.0}]
    ).to_csv(processed / "league_player_projections.csv", index=False)
    build_simulation_dataset(config)  # con league_player_projections.csv: sí se recalibra
    assert calls == [True]


def test_build_simulation_dataset_respects_explicit_monte_carlo_override(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    _write_base_simulation_files(processed)
    pd.DataFrame(
        [{"team_abbreviation": "AAA", "game_score_per36": 2.0, "minutes_projection": 240.0}]
    ).to_csv(processed / "league_player_projections.csv", index=False)
    config = _simulation_config(tmp_path, processed, extra_monte_carlo={"league_average_game_score_per36": 10.0})

    import simulation as simulation_module

    calls = []
    monkeypatch.setattr(
        simulation_module, "compute_league_average_game_score_per36", lambda df, **kwargs: calls.append(True)
    )

    build_simulation_dataset(config)

    assert calls == []  # valor fijado a mano en config no se recalibra aunque el CSV exista


def test_normalize_rotation_minutes_scales_rotation_to_240():
    raw = {i: 25.0 for i in range(15)}  # 375 min en bruto entre 15 jugadores

    out = normalize_rotation_minutes(raw, rotation_size=10)

    assert sum(out.values()) == pytest.approx(TOTAL_TEAM_MINUTES_PER_GAME)
    assert sum(1 for v in out.values() if v > 0) == 10  # solo la rotacion juega


def test_normalize_rotation_minutes_zeroes_players_outside_the_rotation():
    raw = {"star": 36.0, "starter": 32.0, "bench": 20.0, "deep_bench": 3.0}

    out = normalize_rotation_minutes(raw, rotation_size=3)

    assert out["deep_bench"] == 0.0
    assert out["star"] > out["starter"] > out["bench"] > 0


def test_normalize_rotation_minutes_preserves_relative_share_within_rotation():
    raw = {"a": 36.0, "b": 18.0}

    out = normalize_rotation_minutes(raw, rotation_size=2)

    assert out["a"] / out["b"] == pytest.approx(2.0)  # 'a' juega el doble que 'b', antes y después
    assert out["a"] + out["b"] == pytest.approx(TOTAL_TEAM_MINUTES_PER_GAME)


def test_normalize_rotation_minutes_handles_empty_and_all_zero_rosters():
    assert normalize_rotation_minutes({}) == {}
    assert normalize_rotation_minutes({1: 0.0, 2: 0.0}) == {1: 0.0, 2: 0.0}


# ---------------------------------------------------------------------------
# Centrado de la sinergia en la línea base (restricción de suma cero)
# ---------------------------------------------------------------------------


def test_load_league_mean_synergy_returns_zero_without_the_file(tmp_path):
    """Sin el CSV se conserva el comportamiento anterior, no se inventa nada."""
    assert load_league_mean_synergy(tmp_path) == 0.0


def test_load_league_mean_synergy_averages_the_thirty_teams(tmp_path):
    pd.DataFrame([
        {"team_abbreviation": "AAA", "expected_synergy_net_rating": 8.0},
        {"team_abbreviation": "BBB", "expected_synergy_net_rating": 12.0},
    ]).to_csv(tmp_path / "league_team_synergy_baseline.csv", index=False)

    assert load_league_mean_synergy(tmp_path) == pytest.approx(10.0)


def test_baseline_absorbs_the_league_synergy_offset():
    """Regresión: el bonus de sinergia (siempre positivo) sin incorporar a la línea base rompía la suma cero."""
    projections = pd.DataFrame(
        [{"team_abbreviation": "AAA", "game_score_per36": 15.0, "minutes_projection": 240.0}]
    )

    without = compute_league_average_game_score_per36(projections)
    with_synergy = compute_league_average_game_score_per36(
        projections, league_mean_synergy_net_rating=9.6, game_score_to_net_rating_scale=0.21
    )

    # Sinergia sube la línea base en 9.6/0.21 puntos de Game Score de equipo, convertido a unidades por-36.
    assert with_synergy - without == pytest.approx((9.6 / 0.21) / (240.0 / 36.0))


def test_baseline_is_unchanged_when_there_is_no_synergy_data():
    projections = pd.DataFrame(
        [{"team_abbreviation": "AAA", "game_score_per36": 15.0, "minutes_projection": 240.0}]
    )
    assert compute_league_average_game_score_per36(
        projections, league_mean_synergy_net_rating=0.0
    ) == pytest.approx(compute_league_average_game_score_per36(projections))


# ---------------------------------------------------------------------------
# Log de partidos por jugador de UNA temporada simulada
# ---------------------------------------------------------------------------


def test_categorize_injury_absence_uses_the_right_bucket():
    assert categorize_injury_absence(1) == DEFAULT_INJURY_TYPE_CATEGORIES[0]["label"]
    assert categorize_injury_absence(3) == DEFAULT_INJURY_TYPE_CATEGORIES[0]["label"]
    assert categorize_injury_absence(4) == DEFAULT_INJURY_TYPE_CATEGORIES[1]["label"]
    assert categorize_injury_absence(10) == DEFAULT_INJURY_TYPE_CATEGORIES[1]["label"]
    assert categorize_injury_absence(11) == DEFAULT_INJURY_TYPE_CATEGORIES[2]["label"]
    assert categorize_injury_absence(20) == DEFAULT_INJURY_TYPE_CATEGORIES[2]["label"]
    assert categorize_injury_absence(21) == DEFAULT_INJURY_TYPE_CATEGORIES[3]["label"]
    assert categorize_injury_absence(82) == DEFAULT_INJURY_TYPE_CATEGORIES[3]["label"]


def test_simulate_single_season_player_log_games_played_plus_missed_equals_season():
    player_ids = [1, 2, 3]
    player_names = {1: "A", 2: "B", 3: "C"}
    risk_scores = np.array([0.3, 0.0, 0.6])

    log = simulate_single_season_player_log(
        player_ids, player_names, risk_scores, 82, DEFAULT_MONTE_CARLO_CONFIG, random_seed=1
    )

    assert len(log) == 3
    assert (log["games_played"] + log["games_missed"] == 82).all()


def test_simulate_single_season_player_log_zero_risk_player_plays_every_game():
    log = simulate_single_season_player_log(
        [1], {1: "Ironman"}, np.array([0.0]), 82, DEFAULT_MONTE_CARLO_CONFIG, random_seed=7
    )
    row = log.iloc[0]
    assert row["games_played"] == 82
    assert row["games_missed"] == 0
    assert row["injury_events"] == []


def test_simulate_single_season_player_log_events_sum_to_games_missed():
    """Cada racha de ausencia debe sumar exactamente los partidos perdidos."""
    log = simulate_single_season_player_log(
        [1], {1: "Riesgo alto"}, np.array([0.7]), 82, DEFAULT_MONTE_CARLO_CONFIG, random_seed=3
    )
    row = log.iloc[0]
    total_from_events = sum(event["length"] for event in row["injury_events"])
    assert total_from_events == row["games_missed"]


def test_simulate_single_season_player_log_is_deterministic_for_a_given_seed():
    args = ([1, 2], {1: "A", 2: "B"}, np.array([0.4, 0.2]), 82, DEFAULT_MONTE_CARLO_CONFIG)
    first = simulate_single_season_player_log(*args, random_seed=99)
    second = simulate_single_season_player_log(*args, random_seed=99)
    pd.testing.assert_frame_equal(first, second)


def test_simulate_single_season_player_log_each_event_gets_an_illustrative_category():
    log = simulate_single_season_player_log(
        [1], {1: "Riesgo alto"}, np.array([0.7]), 82, DEFAULT_MONTE_CARLO_CONFIG, random_seed=3
    )
    for event in log.iloc[0]["injury_events"]:
        assert event["category"] in {c["label"] for c in DEFAULT_INJURY_TYPE_CATEGORIES}


# ---------------------------------------------------------------------------
# GP / MPG simulados (sustituyen a los históricos en el dashboard)
# ---------------------------------------------------------------------------


def test_compute_expected_games_played_matches_the_binomial_negative_mean():
    """Debe coincidir con la media empírica de sample_injury_absences para el mismo risk_score."""
    risk_scores = np.array([0.25])
    rng = np.random.default_rng(0)
    available = sample_injury_absences(risk_scores, 20000, 82, rng, dispersion=2.0)
    empirical_mean = available[:, :, 0].sum(axis=1).mean()

    analytic = compute_expected_games_played(risk_scores, 82)[0]

    assert analytic == pytest.approx(empirical_mean, abs=0.3)


def test_compute_expected_games_played_zero_risk_plays_full_season():
    assert compute_expected_games_played(np.array([0.0]), 82)[0] == pytest.approx(82.0)


def test_compute_expected_games_played_clips_risk_score_to_valid_range():
    # risk_score fuera de [0, 1] no debe dar partidos negativos o >82.
    result = compute_expected_games_played(np.array([-0.2, 1.5]), 82)
    assert result[0] == pytest.approx(82.0)
    assert result[1] == pytest.approx(0.0)


def test_compute_expected_effective_minutes_per_game_scales_by_availability():
    minutes_projection = np.array([36.0])
    risk_scores = np.array([0.5])
    result = compute_expected_effective_minutes_per_game(minutes_projection, risk_scores)
    assert result[0] == pytest.approx(18.0)


def test_compute_expected_effective_minutes_never_exceeds_the_assumed_minutes():
    minutes_projection = np.array([34.0, 34.0])
    risk_scores = np.array([0.0, 0.6])
    result = compute_expected_effective_minutes_per_game(minutes_projection, risk_scores)
    assert result[0] == pytest.approx(34.0)  # sin riesgo, coincide con lo asumido
    assert result[1] < 34.0  # con riesgo, siempre por debajo
