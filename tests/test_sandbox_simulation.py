import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sandbox_simulation  # noqa: E402
from sandbox_simulation import (  # noqa: E402
    MAX_ROSTER_SIZE,
    SandboxRosterError,
    compute_roster_player_stats,
    load_player_pool,
    simulate_custom_roster,
)


def _make_pool_row(player_id, name, team, game_score_per36, mpg, risk=0.2, fatigue=0.2, pts_per36=20.0):
    return {
        "player_id": player_id,
        "player_name": name,
        "team_abbreviation": team,
        "position": "G",
        "game_score_per36": game_score_per36,
        "minutes_per_game_last_season": mpg,
        # minutes_projection (el minutaje REAL en SU equipo, ya normalizado
        # por league_simulation.py): solo hace falta para que
        # compute_league_average_game_score_per36 pueda leerlo al recalibrar
        # la línea base de liga -- distinto de minutes_per_game_last_season,
        # que es lo que simulate_custom_roster usa como input "en bruto" a
        # normalize_rotation_minutes para el roster hipotético.
        "minutes_projection": mpg,
        "risk_score": risk,
        "fatigue_score": fatigue,
        # Tasa por-36 (independiente del equipo) + PPG "real" en SU equipo
        # (mpg real, no la rotación de 240 del roster hipotético) -- para
        # el test que comprueba que compute_roster_player_stats recalcula
        # en vez de copiar este PPG tal cual.
        "PTS_per36_projected": pts_per36,
        "PPG": pts_per36 * mpg / 36.0,
    }


@pytest.fixture
def sandbox_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)

    pool = pd.DataFrame(
        [
            _make_pool_row(1, "Star A", "ATL", game_score_per36=20.0, mpg=34.0),
            _make_pool_row(2, "Star B", "BOS", game_score_per36=18.0, mpg=32.0),
            _make_pool_row(3, "Role Player C", "CHI", game_score_per36=12.0, mpg=24.0),
            _make_pool_row(4, "Role Player D", "DAL", game_score_per36=11.0, mpg=20.0),
            _make_pool_row(5, "Bench E", "DEN", game_score_per36=9.0, mpg=14.0),
            _make_pool_row(6, "Deep Bench F", "DET", game_score_per36=6.0, mpg=6.0),
        ]
    )
    pool.to_csv(processed / "league_player_projections.csv", index=False)

    pd.DataFrame({"WinPCT": np.linspace(0.2, 0.8, 30)}).to_csv(processed / "prior_season_standings.csv", index=False)

    return {
        "team": {"team_id": 999, "season": "2026-27"},
        "simulation": {"games_per_season": 82},
        "monte_carlo": {},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }


def test_load_player_pool_raises_helpful_error_when_missing(tmp_path):
    config = {
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(tmp_path / "processed")},
    }
    with pytest.raises(FileNotFoundError, match="--league"):
        load_player_pool(config)


def test_simulate_custom_roster_returns_one_row_per_season(sandbox_config):
    result = simulate_custom_roster(sandbox_config, [1, 2, 3, 4, 5], n_seasons=50, random_seed=1)

    assert len(result) == 50
    assert {"wins", "losses", "net_rating_estimate_mean", "total_games_missed"} <= set(result.columns)
    assert (result["wins"] + result["losses"] == 82).all()


def test_simulate_custom_roster_rejects_too_few_players(sandbox_config):
    with pytest.raises(SandboxRosterError, match="al menos"):
        simulate_custom_roster(sandbox_config, [1, 2], n_seasons=10)


def test_simulate_custom_roster_rejects_too_many_players(sandbox_config):
    with pytest.raises(SandboxRosterError, match="Máximo"):
        simulate_custom_roster(sandbox_config, list(range(1, MAX_ROSTER_SIZE + 2)), n_seasons=10)


def test_simulate_custom_roster_rejects_duplicate_player(sandbox_config):
    with pytest.raises(SandboxRosterError, match="repetido"):
        simulate_custom_roster(sandbox_config, [1, 1, 2, 3, 4], n_seasons=10)


def test_simulate_custom_roster_rejects_unknown_player_id(sandbox_config):
    with pytest.raises(SandboxRosterError, match="999999"):
        simulate_custom_roster(sandbox_config, [1, 2, 3, 4, 999999], n_seasons=10)


def test_simulate_custom_roster_normalizes_minutes_to_240(sandbox_config, monkeypatch):
    """La rotación elegida debe sumar TOTAL_TEAM_MINUTES_PER_GAME (240),
    igual que league_simulation.project_team_roster para los 30 equipos
    reales -- se comprueba interceptando la llamada a run_monte_carlo en
    vez de inferirlo del resultado simulado (ruidoso)."""
    captured = {}

    def fake_run_monte_carlo(player_ids, game_score_per36, minutes_projection, *args, **kwargs):
        captured["minutes_projection"] = minutes_projection
        captured["player_ids"] = player_ids
        return pd.DataFrame({"wins": [41], "losses": [41], "net_rating_estimate_mean": [0.0], "total_games_missed": [0]})

    monkeypatch.setattr(sandbox_simulation, "run_monte_carlo", fake_run_monte_carlo)

    simulate_custom_roster(sandbox_config, [1, 2, 3, 4, 5], n_seasons=10)

    assert captured["minutes_projection"].sum() == pytest.approx(240.0)
    assert list(captured["player_ids"]) == [1, 2, 3, 4, 5]


def test_simulate_custom_roster_respects_mc_overrides(sandbox_config, monkeypatch):
    captured = {}

    def fake_run_monte_carlo(player_ids, game_score_per36, minutes_projection, risk_scores, fatigue_scores,
                              league_win_pcts, n_seasons, games_per_season, mc_config, random_seed, **kwargs):
        captured["mc_config"] = mc_config
        return pd.DataFrame({"wins": [41], "losses": [41], "net_rating_estimate_mean": [0.0], "total_games_missed": [0]})

    monkeypatch.setattr(sandbox_simulation, "run_monte_carlo", fake_run_monte_carlo)

    simulate_custom_roster(
        sandbox_config, [1, 2, 3, 4, 5], n_seasons=10, mc_overrides={"game_variance_std": 7.5}
    )

    assert captured["mc_config"]["game_variance_std"] == pytest.approx(7.5)


def test_compute_roster_player_stats_returns_one_row_per_player_sorted_by_game_score(sandbox_config):
    view = compute_roster_player_stats(sandbox_config, [1, 2, 3, 4, 5])

    assert len(view) == 5
    # El fixture ya da game_score_per36 decreciente para pid 1..5 (20, 18,
    # 12, 11, 9) -- confirma que el orden de salida es por game_score, no
    # simplemente el orden de entrada (que aquí coincide a propósito).
    assert list(view["player_id"]) == [1, 2, 3, 4, 5]
    assert list(view["game_score_per36"]) == sorted(view["game_score_per36"], reverse=True)


def test_compute_roster_player_stats_per_game_mode_has_ppg_not_totals(sandbox_config):
    view = compute_roster_player_stats(sandbox_config, [1, 2, 3, 4, 5], mode="per_game")

    assert "PPG" in view.columns
    assert "PTS" not in view.columns


def test_compute_roster_player_stats_totals_mode_has_pts_not_ppg(sandbox_config):
    view = compute_roster_player_stats(sandbox_config, [1, 2, 3, 4, 5], mode="totals")

    assert "PTS" in view.columns
    assert "PPG" not in view.columns


def test_compute_roster_player_stats_rejects_invalid_mode(sandbox_config):
    with pytest.raises(SandboxRosterError, match="mode"):
        compute_roster_player_stats(sandbox_config, [1, 2, 3, 4, 5], mode="bogus")


def test_compute_roster_player_stats_recalculates_ppg_for_new_minutes_not_real_team_ppg(sandbox_config):
    """Player 1 (Star A) tenía 34 mpg REALES en su equipo real (PPG real
    precargado en el pool de prueba) -- en un roster de solo 5 jugadores
    normalizado a 240, sus minutos nuevos son distintos (240 * 34/rotación),
    así que su PPG recalculado NO debe coincidir con el PPG "real" del pool,
    que es exactamente lo que este endpoint tenía que arreglar."""
    pool = load_player_pool(sandbox_config).set_index("player_id")
    real_team_ppg = pool.loc[1, "PPG"]

    view = compute_roster_player_stats(sandbox_config, [1, 2, 3, 4, 5], mode="per_game")
    recalculated_ppg = view.loc[view["player_id"] == 1, "PPG"].iloc[0]

    assert recalculated_ppg != pytest.approx(real_team_ppg)


def test_compute_roster_player_stats_gp_matches_expected_games_played_formula(sandbox_config):
    view = compute_roster_player_stats(sandbox_config, [1, 2, 3, 4, 5])
    row = view[view["player_id"] == 1].iloc[0]

    # Star A tiene risk=0.2 en el fixture -> GP esperado = 82 * (1 - 0.2)
    assert row["GP"] == pytest.approx(82 * 0.8, abs=1.0)


def test_simulate_custom_roster_baseline_excludes_league_synergy(sandbox_config, monkeypatch):
    """Regresión: el mismo roster sin editar daba 26 victorias en
    /sandbox/simulate y 42 en /sandbox/league. Causa: run_monte_carlo
    recibe synergy_matrix=None aquí (limitación documentada -- el sandbox
    no modela sinergia), pero la línea base de "equipo promedio" SÍ
    llevaba incorporada la sinergia media de los 30 equipos reales
    (~+10.65 de net rating) -- comparando tu equipo (sin sinergia) contra
    un rival que sí la tenía, cada partido. Mismo patrón que la línea base
    rompiendo la suma cero, ya arreglado una vez en simulation.py -- aquí
    se comprueba que no vuelva."""
    captured = {}

    def fake_compute_league_average_game_score_per36(player_projections, league_mean_synergy_net_rating=0.0, **kwargs):
        captured["league_mean_synergy_net_rating"] = league_mean_synergy_net_rating
        return 10.0

    monkeypatch.setattr(
        sandbox_simulation, "compute_league_average_game_score_per36", fake_compute_league_average_game_score_per36
    )

    simulate_custom_roster(sandbox_config, [1, 2, 3, 4, 5], n_seasons=10)

    assert captured["league_mean_synergy_net_rating"] == 0.0
