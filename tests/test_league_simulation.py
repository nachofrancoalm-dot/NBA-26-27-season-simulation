"""
Tests de league_simulation.py. Usan proyecciones de equipo sintéticas
(dicts con game_score_per36/minutes_projection/risk_scores/fatigue_scores)
y rosters/career stats sintéticos -- no requieren red.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from league_simulation import (  # noqa: E402
    DEFAULT_MONTE_CARLO_CONFIG,
    _conference_bracket_with_matchups,
    _play_in_with_detail,
    _sample_team_game_score,
    build_round_robin_schedule,
    load_and_project_all_teams,
    project_own_team_for_league,
    project_team_roster,
    resolve_play_in,
    simulate_conference_bracket,
    simulate_league_regular_season,
    simulate_playoff_game,
    simulate_playoffs_once,
    simulate_series,
)


def _team_proj(game_score: float, n_players: int = 5, fatigue: float = 0.0, risk: float = 0.0) -> dict:
    return {
        "player_ids": list(range(n_players)),
        "game_score_per36": np.full(n_players, game_score),
        "minutes_projection": np.full(n_players, 36.0),
        "risk_scores": np.full(n_players, risk),
        "fatigue_scores": np.full(n_players, fatigue),
        "synergy_matrix": None,
    }


def test_round_robin_schedule_every_team_plays_exactly_games_per_season():
    team_ids = list(range(10))
    rng = np.random.default_rng(1)
    schedule = build_round_robin_schedule(team_ids, games_per_season=18, rng=rng)

    games_played = {tid: 0 for tid in team_ids}
    for _, team_a, team_b in schedule:
        games_played[team_a] += 1
        games_played[team_b] += 1

    assert all(count == 18 for count in games_played.values())


def test_round_robin_schedule_never_pairs_a_team_with_itself():
    team_ids = list(range(8))
    rng = np.random.default_rng(2)
    schedule = build_round_robin_schedule(team_ids, games_per_season=14, rng=rng)
    assert all(team_a != team_b for _, team_a, team_b in schedule)


def test_round_robin_schedule_rejects_odd_number_of_teams():
    with pytest.raises(ValueError):
        build_round_robin_schedule([1, 2, 3], games_per_season=10, rng=np.random.default_rng(0))


def test_simulate_league_regular_season_favors_stronger_team():
    team_ids = [1, 2]
    projections = {1: _team_proj(game_score=25.0), 2: _team_proj(game_score=5.0)}
    rng = np.random.default_rng(3)
    schedule = build_round_robin_schedule(team_ids, games_per_season=20, rng=rng)

    wins = simulate_league_regular_season(
        projections, schedule, n_seasons=500, games_per_season=20,
        mc_config=DEFAULT_MONTE_CARLO_CONFIG, random_seed=42,
    )
    assert wins[1].mean() > wins[2].mean()


def test_simulate_playoff_game_favors_stronger_team_on_average():
    strong = _team_proj(game_score=30.0)
    weak = _team_proj(game_score=2.0)
    rng = np.random.default_rng(5)
    results = [simulate_playoff_game(strong, weak, rng, DEFAULT_MONTE_CARLO_CONFIG) for _ in range(300)]
    assert sum(results) / len(results) > 0.8


def test_simulate_series_almost_always_won_by_much_stronger_team():
    strong = _team_proj(game_score=35.0)
    weak = _team_proj(game_score=0.0)
    rng = np.random.default_rng(6)
    results = [simulate_series(strong, weak, rng, DEFAULT_MONTE_CARLO_CONFIG, best_of=7) for _ in range(50)]
    assert results.count(0) > 45  # strong (index 0) gana casi siempre


def test_resolve_play_in_top_6_advance_automatically():
    team_projections = {i: _team_proj(game_score=15.0) for i in range(1, 11)}
    seeds = list(range(1, 11))  # ya ordenados 1..10
    rng = np.random.default_rng(7)
    result = resolve_play_in(seeds, team_projections, rng, DEFAULT_MONTE_CARLO_CONFIG)

    assert len(result) == 8
    assert result[:6] == seeds[:6]


def test_resolve_play_in_last_two_spots_come_from_seeds_7_through_10():
    team_projections = {i: _team_proj(game_score=15.0) for i in range(1, 11)}
    seeds = list(range(1, 11))
    rng = np.random.default_rng(7)
    result = resolve_play_in(seeds, team_projections, rng, DEFAULT_MONTE_CARLO_CONFIG)

    assert set(result[6:]).issubset({7, 8, 9, 10})


def test_simulate_conference_bracket_returns_a_champion_from_the_8_seeds():
    seeds_8 = list(range(1, 9))
    team_projections = {i: _team_proj(game_score=10.0 + i) for i in seeds_8}
    rng = np.random.default_rng(9)
    result = simulate_conference_bracket(seeds_8, team_projections, rng, DEFAULT_MONTE_CARLO_CONFIG)

    assert result["conference_champion"] in seeds_8
    assert len(result["round1_winners"]) == 4
    assert len(result["conf_semis_winners"]) == 2


def test_play_in_with_detail_matches_resolve_play_in_seeds():
    team_projections = {i: _team_proj(game_score=15.0) for i in range(1, 11)}
    seeds = list(range(1, 11))

    seeds_8, detail = _play_in_with_detail(seeds, team_projections, np.random.default_rng(7), DEFAULT_MONTE_CARLO_CONFIG)

    assert len(seeds_8) == 8
    assert seeds_8[:6] == seeds[:6]
    assert detail["game_7_vs_8"]["team_a"] == 7
    assert detail["game_7_vs_8"]["team_b"] == 8
    assert detail["game_7_vs_8"]["winner"] in (7, 8)
    assert detail["game_9_vs_10"]["winner"] in (9, 10)
    assert detail["game_elimination"]["winner"] == seeds_8[7]


def test_conference_bracket_with_matchups_exposes_pairings():
    seeds_8 = list(range(1, 9))
    team_projections = {i: _team_proj(game_score=10.0 + i) for i in seeds_8}
    rng = np.random.default_rng(9)

    result = _conference_bracket_with_matchups(seeds_8, team_projections, rng, DEFAULT_MONTE_CARLO_CONFIG)

    assert len(result["round1"]) == 4
    assert result["round1"][0]["team_a"] == 1
    assert result["round1"][0]["team_b"] == 8
    assert result["round1"][0]["winner"] in (1, 8)
    assert len(result["conf_semis"]) == 2
    assert result["conference_champion"] in seeds_8


def test_simulate_playoffs_once_produces_a_valid_champion():
    east_ids = list(range(1, 11))
    west_ids = list(range(11, 21))
    team_conference = {tid: "East" for tid in east_ids}
    team_conference.update({tid: "West" for tid in west_ids})
    team_projections = {tid: _team_proj(game_score=10.0 + (tid % 10)) for tid in east_ids + west_ids}
    wins_by_team = {tid: 50 - tid for tid in east_ids + west_ids}  # orden arbitrario pero determinista

    rng = np.random.default_rng(11)
    result = simulate_playoffs_once(wins_by_team, team_conference, team_projections, rng, DEFAULT_MONTE_CARLO_CONFIG)

    assert result["nba_champion"] in east_ids + west_ids
    assert len(result["east_8"]) == 8
    assert len(result["west_8"]) == 8


def test_playoff_game_score_penalizes_injury_risk():
    """
    Regresión de un bug real: los playoffs asumían roster a PLENA SALUD,
    así que un equipo de estrellas frágiles era castigado 82 partidos en
    temporada regular y luego llegaba a playoffs milagrosamente sano --
    llegando a tener MÁS probabilidad de título que el mejor equipo de
    temporada regular. Ver el docstring de _sample_team_game_score.
    """
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0}  # sin ruido
    healthy = _team_proj(game_score=15.0, risk=0.0)
    fragile = _team_proj(game_score=15.0, risk=0.5)  # mismo talento, la mitad del tiempo lesionado

    n = 400
    healthy_scores = [_sample_team_game_score(healthy, np.random.default_rng(i), cfg) for i in range(n)]
    fragile_scores = [_sample_team_game_score(fragile, np.random.default_rng(i), cfg) for i in range(n)]

    assert np.mean(fragile_scores) < np.mean(healthy_scores)
    # Con risk=0.5 se espera aproximadamente la mitad de la produccion.
    assert np.mean(fragile_scores) == pytest.approx(np.mean(healthy_scores) * 0.5, rel=0.15)


def test_playoff_game_score_is_unaffected_by_risk_when_risk_is_zero():
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0}
    proj = _team_proj(game_score=15.0, risk=0.0)

    score = _sample_team_game_score(proj, np.random.default_rng(1), cfg)

    # 5 jugadores x 15 GS/36 x 36 min / 36 = 75, sin ruido ni fatiga.
    assert score == pytest.approx(75.0)


def test_playoff_home_court_helps_the_home_team():
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0}
    a = _team_proj(game_score=12.0)
    b = _team_proj(game_score=12.0)  # equipos idénticos: solo decide la sede

    home_wins = sum(
        simulate_playoff_game(a, b, np.random.default_rng(i), cfg, team_a_is_home=True) for i in range(600)
    )
    away_wins = sum(
        simulate_playoff_game(a, b, np.random.default_rng(i), cfg, team_a_is_home=False) for i in range(600)
    )

    assert home_wins > away_wins
    assert home_wins / 600 > 0.55  # ventaja real medida: ~60% de victorias locales


def test_playoff_game_is_neutral_when_home_flag_is_none():
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0}
    a = _team_proj(game_score=12.0)
    b = _team_proj(game_score=12.0)

    wins = sum(simulate_playoff_game(a, b, np.random.default_rng(i), cfg) for i in range(600))

    assert 0.42 < wins / 600 < 0.58  # sin sede, equipos iguales -> moneda al aire


def test_series_home_court_favors_the_higher_seed():
    """El formato 2-2-1-1-1 da 4 de 7 partidos en casa al mejor seed."""
    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0}
    a = _team_proj(game_score=12.0)
    b = _team_proj(game_score=12.0)

    a_with_home = sum(
        simulate_series(a, b, np.random.default_rng(i), cfg, team_a_has_home_court=True) == 0
        for i in range(400)
    )
    a_without_home = sum(
        simulate_series(a, b, np.random.default_rng(i), cfg, team_a_has_home_court=False) == 0
        for i in range(400)
    )

    assert a_with_home > a_without_home


def test_series_home_game_pattern_is_the_real_2_2_1_1_1():
    from league_simulation import SERIES_HOME_GAMES_FOR_HIGHER_SEED as pattern

    # El mejor seed es local en los partidos 1, 2, 5 y 7 (4 de 7).
    assert pattern == (True, True, False, False, True, False, True)
    assert sum(pattern) == 4


def test_series_winner_gives_home_court_to_the_better_seed_not_list_order():
    """
    Regresión: a partir de semifinales el orden de la lista NO garantiza
    que el primero sea el mejor seed (si el 8 elimina al 1). La ventaja
    de campo debe seguir al seed real, no a la posición en la tupla.
    """
    from league_simulation import _series_winner

    cfg = {**DEFAULT_MONTE_CARLO_CONFIG, "game_variance_std": 0.0}
    projections = {1: _team_proj(game_score=12.0), 2: _team_proj(game_score=12.0)}
    # El equipo 2 es MEJOR seed (rank 0) aunque se pase en segunda posición.
    seed_rank = {1: 5, 2: 0}

    wins_for_2 = sum(
        _series_winner(1, 2, projections, np.random.default_rng(i), cfg, seed_rank=seed_rank) == 2
        for i in range(400)
    )

    assert wins_for_2 / 400 > 0.5  # el mejor seed gana mas de la mitad pese a ir segundo


def test_simulate_playoffs_once_with_15_teams_per_conference_like_real_nba():
    # Caso real: 15 equipos por conferencia (30 en total), no 10 -- los
    # seeds 11-15 deben quedar eliminados antes del play-in. Regresión de
    # un bug real: resolve_play_in exige exactamente 10 seeds y
    # simulate_playoffs_once le pasaba las 15 sin recortar.
    east_ids = list(range(1, 16))
    west_ids = list(range(16, 31))
    team_conference = {tid: "East" for tid in east_ids}
    team_conference.update({tid: "West" for tid in west_ids})
    team_projections = {tid: _team_proj(game_score=10.0 + (tid % 15)) for tid in east_ids + west_ids}
    wins_by_team = {tid: 60 - tid for tid in east_ids + west_ids}

    rng = np.random.default_rng(13)
    result = simulate_playoffs_once(wins_by_team, team_conference, team_projections, rng, DEFAULT_MONTE_CARLO_CONFIG)

    assert result["nba_champion"] in east_ids + west_ids
    assert len(result["east_8"]) == 8
    assert len(result["west_8"]) == 8
    # Los peores 5 de cada conferencia (11-15 por wins) no deben aparecer.
    assert not set(result["east_8"]) & {11, 12, 13, 14, 15}
    assert not set(result["west_8"]) & {26, 27, 28, 29, 30}


def test_project_team_roster_normalizes_total_minutes_to_240():
    # Regresión de un hallazgo real: sin normalizar, un roster con
    # rotación históricamente profunda (varios jugadores con muchos
    # minutos/partido reales el año pasado) podía sumar muy por encima de
    # los 240 minutos que existen de verdad en un partido (5 posiciones x
    # 48 min), inflando artificialmente su Game Score de equipo frente a
    # rosters con rotaciones más cortas -- Utah llegó a sumar 449 minutos
    # "en bruto" y lideraba las probabilidades de título injustamente.
    roster = pd.DataFrame(
        [{"PLAYER_ID": 1, "PLAYER": "Heavy Minutes A"}, {"PLAYER_ID": 2, "PLAYER": "Heavy Minutes B"}]
    )
    regular = pd.DataFrame(
        [
            {
                "PLAYER_ID": pid, "SEASON_ID": "2025-26", "PLAYER_AGE": 27, "GP": 82, "MIN": 3200,
                "PTS": 1600, "AST": 400, "REB": 500, "STL": 80, "BLK": 40, "TOV": 150,
                "FG3M": 100, "FG3A": 260, "OREB": 80, "DREB": 420, "FGM": 600, "FGA": 1200,
                "FTM": 300, "FTA": 360, "PF": 180,
            }
            for pid in (1, 2)
        ]
    )
    config = {"team": {"season": "2026-27"}, "simulation": {"games_per_season": 82}, "lineup_synergy": {}}

    # Minutos "en bruto" por jugador: 3200/82 ≈ 39 -> suma bruta ≈ 78 (por
    # debajo de 240 en este caso sintético, pero el mismo mecanismo aplica
    # en la dirección contraria cuando la suma bruta excede 240).
    result = project_team_roster(roster, regular, pd.DataFrame(), config)
    assert sum(result["minutes_projection"]) == pytest.approx(240.0)


def _minutes_only_roster(rows):
    """rows: lista de (player_id, name, raw_minutes_per_game, gp)."""
    roster = pd.DataFrame([{"PLAYER_ID": pid, "PLAYER": name} for pid, name, _, _ in rows])
    regular = pd.DataFrame(
        [
            {
                "PLAYER_ID": pid, "SEASON_ID": "2025-26", "PLAYER_AGE": 27, "GP": gp, "MIN": raw_min * gp,
                "PTS": raw_min * gp * 0.5, "AST": 0, "REB": 0, "STL": 0, "BLK": 0, "TOV": 0,
                "FG3M": 0, "FG3A": 0, "OREB": 0, "DREB": 0, "FGM": 0, "FGA": 0,
                "FTM": 0, "FTA": 0, "PF": 0,
            }
            for pid, _, raw_min, gp in rows
        ]
    )
    return roster, regular


def test_project_team_roster_does_not_dilute_star_minutes_with_bench_churn():
    # Regresión de un hallazgo real (revisión manual del usuario): la
    # primera versión de la normalización a 240 escalaba TODO el roster
    # por igual, y una estrella real (Luka Dončić, ~35.8 min/partido
    # reales en los Lakers) terminaba con 26.98 -- diluida por muchos
    # suplentes de fondo de plantilla que solo jugaron unos pocos
    # partidos por movimiento de plantilla (lesiones, llamados de
    # two-way), no por mérito. La rotación real de los Lakers (top 10
    # por minutos) ya sumaba ~257, cerca de 240 -- el roster COMPLETO
    # (19 jugadores, muchos con 1-15 partidos) sumaba 318.
    star_and_bench = [
        (1, "Star Player", 35.8, 64),  # rotación real, como Luka
        (2, "Starter A", 34.5, 70),
        (3, "Starter B", 30.8, 60),
        (4, "Starter C", 29.4, 55),
        (5, "Rotation A", 25.1, 65),
        (6, "Rotation B", 23.7, 50),
        (7, "Rotation C", 22.9, 60),
        (8, "Rotation D", 21.9, 45),
        (9, "Rotation E", 17.4, 55),
        (10, "Rotation F", 16.0, 50),
        # Fuera de la rotación real (top 10) -- jugadores de fondo de
        # plantilla con pocos partidos, como en el roster real de UTA.
        (11, "Deep Bench A", 15.8, 9),
        (12, "Deep Bench B", 14.7, 4),
        (13, "Deep Bench C", 10.2, 14),
        (14, "Deep Bench D", 8.9, 7),
        (15, "Deep Bench E", 6.0, 1),
    ]
    roster, regular = _minutes_only_roster(star_and_bench)
    config = {
        "team": {"season": "2026-27"}, "simulation": {"games_per_season": 82},
        "lineup_synergy": {}, "league_simulation": {},
    }

    result = project_team_roster(roster, regular, pd.DataFrame(), config)

    star_minutes = result["player_rows"][0]["minutes_projection"]
    # Con la rotación real (top 10) sumando ~257.5, el factor de escala es
    # ~0.93 -- la estrella debe quedar cerca de sus 35.8 reales, no caer a ~27.
    assert star_minutes > 30.0
    # Los jugadores fuera de la rotación (índices 10-14, ids 11-15) deben
    # quedar en 0 -- no diluyen la normalización de la rotación real.
    bench_minutes = [row["minutes_projection"] for row in result["player_rows"][10:]]
    assert all(m == 0.0 for m in bench_minutes)
    # La rotación (los 10 primeros) debe sumar exactamente 240.
    rotation_minutes = [row["minutes_projection"] for row in result["player_rows"][:10]]
    assert sum(rotation_minutes) == pytest.approx(240.0)


def test_project_team_roster_handles_player_with_no_history():
    roster = pd.DataFrame([{"PLAYER_ID": 1, "PLAYER": "Rookie Player"}])
    empty_regular = pd.DataFrame(columns=["PLAYER_ID", "SEASON_ID", "PLAYER_AGE", "GP", "MIN"]).astype(
        {"PLAYER_ID": int}
    )
    config = {
        "team": {"season": "2026-27"},
        "simulation": {"games_per_season": 82},
        "lineup_synergy": {},
    }
    result = project_team_roster(roster, empty_regular, pd.DataFrame(), config)
    assert result["game_score_per36"][0] == 0.0
    assert result["risk_scores"][0] == 0.0
    assert result["player_rows"][0]["player_name"] == "Rookie Player"
    assert result["player_rows"][0]["game_score_per36"] == 0.0


def test_project_team_roster_includes_position_from_roster_slice():
    """Necesario para los quintetos All-NBA/All-Defensive de
    awards_projection.py, que exigen posición real -- ver POSITION_GROUPS."""
    roster = pd.DataFrame([{"PLAYER_ID": 1, "PLAYER": "Veteran Player", "POSITION": "F"}])
    regular = pd.DataFrame(
        [
            {
                "PLAYER_ID": 1, "SEASON_ID": "2023-24", "PLAYER_AGE": 27, "GP": 80, "MIN": 2800,
                "PTS": 1600, "AST": 400, "REB": 500, "STL": 80, "BLK": 40, "TOV": 150,
                "FG3M": 100, "FG3A": 260, "OREB": 80, "DREB": 420, "FGM": 600, "FGA": 1200,
                "FTM": 300, "FTA": 360, "PF": 180,
            }
        ]
    )
    config = {"team": {"season": "2026-27"}, "simulation": {"games_per_season": 82}, "lineup_synergy": {}}

    result = project_team_roster(roster, regular, pd.DataFrame(), config)

    assert result["player_rows"][0]["position"] == "F"


def test_project_team_roster_position_is_none_without_a_position_column():
    """Roster sin POSITION (esquema antiguo/sintético) no debe fallar --
    solo excluye a esos jugadores de los quintetos, no rompe el resto."""
    roster = pd.DataFrame([{"PLAYER_ID": 1, "PLAYER": "No Position Data"}])
    regular = pd.DataFrame(
        [
            {
                "PLAYER_ID": 1, "SEASON_ID": "2023-24", "PLAYER_AGE": 27, "GP": 80, "MIN": 2800,
                "PTS": 1600, "AST": 400, "REB": 500, "STL": 80, "BLK": 40, "TOV": 150,
                "FG3M": 100, "FG3A": 260, "OREB": 80, "DREB": 420, "FGM": 600, "FGA": 1200,
                "FTM": 300, "FTA": 360, "PF": 180,
            }
        ]
    )
    config = {"team": {"season": "2026-27"}, "simulation": {"games_per_season": 82}, "lineup_synergy": {}}

    result = project_team_roster(roster, regular, pd.DataFrame(), config)

    assert result["player_rows"][0]["position"] is None


def test_project_team_roster_player_rows_include_projected_stats():
    roster = pd.DataFrame([{"PLAYER_ID": 1, "PLAYER": "Veteran Player"}])
    regular = pd.DataFrame(
        [
            {
                "PLAYER_ID": 1, "SEASON_ID": "2023-24", "PLAYER_AGE": 27, "GP": 80, "MIN": 2800,
                "PTS": 1600, "AST": 400, "REB": 500, "STL": 80, "BLK": 40, "TOV": 150,
                "FG3M": 100, "FG3A": 260, "OREB": 80, "DREB": 420, "FGM": 600, "FGA": 1200,
                "FTM": 300, "FTA": 360, "PF": 180,
            }
        ]
    )
    config = {"team": {"season": "2026-27"}, "simulation": {"games_per_season": 82}, "lineup_synergy": {}}

    result = project_team_roster(roster, regular, pd.DataFrame(), config)
    row = result["player_rows"][0]

    assert row["player_name"] == "Veteran Player"
    assert "PTS_projected" in row
    assert row["game_score_per36"] == pytest.approx(result["game_score_per36"][0])


@pytest.fixture
def own_team_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "player_id": 1, "player_name": "Hypothetical Star", "current_age": 27, "target_age": 28,
                "game_score_per36": 22.0, "projected_total_minutes": 2870.0,  # 35 mpg x 82
                "PTS_projected": 1804, "REB_projected": 410, "AST_projected": 500,
                "STL_per36_projected": 1.2, "BLK_per36_projected": 0.5, "DREB_per36_projected": 5.0,
                "FGA_per36_projected": 15.0, "FTA_per36_projected": 4.0, "TOV_per36_projected": 2.5,
                "AST_per36_projected": 6.0, "FG3A_per36_projected": 5.0,
            },
        ]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "risk_score": 0.3}]).to_csv(processed / "injury_risk.csv", index=False)
    pd.DataFrame([{"player_id": 1, "fatigue_score": 0.2}]).to_csv(processed / "fatigue_risk.csv", index=False)

    return {
        "team": {"team_id": 999, "season": "2026-27"},
        "roster": [{"player_id": 1, "name": "Hypothetical Star", "minutes_projection": 30}],
        "simulation": {"games_per_season": 82},
        "lineup_synergy": {},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }


def test_project_own_team_for_league_uses_config_minutes_not_real_minutes(own_team_config):
    result = project_own_team_for_league(own_team_config)

    # 30 (config) en vez de 2870/82=35 (real, de aging_curve_projection.csv).
    assert result["minutes_projection"][0] == pytest.approx(30.0)
    assert result["player_rows"][0]["minutes_projection"] == pytest.approx(30.0)
    assert result["player_rows"][0]["player_name"] == "Hypothetical Star"
    assert result["risk_scores"][0] == pytest.approx(0.3)
    assert result["fatigue_scores"][0] == pytest.approx(0.2)


def test_project_own_team_for_league_merges_position_when_available(own_team_config):
    processed = Path(own_team_config["paths"]["processed_data_dir"])
    pd.DataFrame([{"player_id": 1, "position": "Guard"}]).to_csv(processed / "roster_positions.csv", index=False)

    result = project_own_team_for_league(own_team_config)

    assert result["player_rows"][0]["position"] == "Guard"


def test_project_own_team_for_league_degrades_gracefully_without_positions_csv(own_team_config):
    """roster_positions.csv es opcional -- sin él, el equipo propio
    simplemente no participa en los quintetos All-NBA/All-Defensive
    (ver awards_projection._position_group), pero el resto no debe fallar."""
    result = project_own_team_for_league(own_team_config)

    assert result["player_rows"][0]["position"] is None


def test_project_own_team_for_league_raises_when_player_missing_from_aging_csv(own_team_config):
    own_team_config["roster"].append({"player_id": 2, "name": "Not Projected Yet", "minutes_projection": 10})

    with pytest.raises(ValueError, match="2"):
        project_own_team_for_league(own_team_config)


def test_load_and_project_all_teams_overrides_own_team_with_hypothetical_roster(own_team_config):
    processed = Path(own_team_config["paths"]["processed_data_dir"])

    # Roster REAL de 2 franquicias (999 = la propia -- con OTRO jugador,
    # simulando que la franquicia real no tiene al fichaje hipotético --
    # 111 = una rival). Se reusan abreviaciones reales (PHI/BOS) porque
    # TEAM_CONFERENCE solo reconoce las 30 franquicias reales; el team_id
    # numérico en sí es arbitrario, no tiene que coincidir con el team_id
    # real de esas siglas.
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER": "Hypothetical Star", "team_id": 999, "team_abbreviation": "PHI"},
            {"PLAYER_ID": 3, "PLAYER": "Real Roster Player", "team_id": 999, "team_abbreviation": "PHI"},
            {"PLAYER_ID": 4, "PLAYER": "Rival Player", "team_id": 111, "team_abbreviation": "BOS"},
        ]
    ).to_csv(processed / "league_rosters.csv", index=False)

    def _career_row(player_id, team_id, team_abbrev):
        return {
            "PLAYER_ID": player_id, "SEASON_ID": "2024-25", "PLAYER_AGE": 26, "GP": 70, "MIN": 2000,
            "PTS": 1000, "AST": 200, "REB": 300, "STL": 50, "BLK": 25, "TOV": 100,
            "FG3M": 60, "FG3A": 160, "OREB": 50, "DREB": 250, "FGM": 380, "FGA": 800,
            "FTM": 180, "FTA": 220, "PF": 150, "team_id": team_id, "team_abbreviation": team_abbrev,
        }

    pd.DataFrame(
        [
            _career_row(1, 999, "PHI"),
            _career_row(3, 999, "PHI"),
            _career_row(4, 111, "BOS"),
        ]
    ).to_csv(processed / "league_player_career_stats.csv", index=False)

    team_ids, team_abbrev_by_id, team_conference, team_projections = load_and_project_all_teams(own_team_config)

    own_projection = team_projections[999]
    # El roster real de la franquicia 999 tenía a "Real Roster Player" (id 3),
    # pero la proyección usada es la hipotética -- solo el player_id 1 del
    # roster de team_config.yaml, con SUS minutos configurados (30, no un
    # valor derivado del roster/minutos reales).
    assert own_projection["player_ids"] == [1]
    assert own_projection["minutes_projection"][0] == pytest.approx(30.0)
    # El equipo rival (111) no se toca -- sigue proyectado desde su roster real.
    assert team_projections[111]["player_ids"] == [4]


def test_load_and_project_all_teams_merges_country_for_real_teams(own_team_config):
    """
    country NO viene en league_rosters.csv (CommonTeamRoster no la trae)
    -- es un lookup global por player_id desde league_player_countries.csv
    (data_pipeline.build_league_player_countries_dataset), necesario para
    el chequeo de cuota de nacionalidad del All-Star.
    """
    processed = Path(own_team_config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"PLAYER_ID": 4, "PLAYER": "Rival Player", "team_id": 111, "team_abbreviation": "BOS"}]
    ).to_csv(processed / "league_rosters.csv", index=False)
    pd.DataFrame(
        [{
            "PLAYER_ID": 4, "SEASON_ID": "2024-25", "PLAYER_AGE": 26, "GP": 70, "MIN": 2000,
            "PTS": 1000, "AST": 200, "REB": 300, "STL": 50, "BLK": 25, "TOV": 100,
            "FG3M": 60, "FG3A": 160, "OREB": 50, "DREB": 250, "FGM": 380, "FGA": 800,
            "FTM": 180, "FTA": 220, "PF": 150, "team_id": 111, "team_abbreviation": "BOS",
        }]
    ).to_csv(processed / "league_player_career_stats.csv", index=False)
    pd.DataFrame([{"player_id": 4, "country": "Slovenia"}]).to_csv(
        processed / "league_player_countries.csv", index=False
    )

    _, _, _, team_projections = load_and_project_all_teams(own_team_config)

    assert team_projections[111]["player_rows"][0]["country"] == "Slovenia"


def test_load_and_project_all_teams_degrades_gracefully_without_countries_csv(own_team_config):
    processed = Path(own_team_config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"PLAYER_ID": 4, "PLAYER": "Rival Player", "team_id": 111, "team_abbreviation": "BOS"}]
    ).to_csv(processed / "league_rosters.csv", index=False)
    pd.DataFrame(
        [{
            "PLAYER_ID": 4, "SEASON_ID": "2024-25", "PLAYER_AGE": 26, "GP": 70, "MIN": 2000,
            "PTS": 1000, "AST": 200, "REB": 300, "STL": 50, "BLK": 25, "TOV": 100,
            "FG3M": 60, "FG3A": 160, "OREB": 50, "DREB": 250, "FGM": 380, "FGA": 800,
            "FTM": 180, "FTA": 220, "PF": 150, "team_id": 111, "team_abbreviation": "BOS",
        }]
    ).to_csv(processed / "league_player_career_stats.csv", index=False)

    _, _, _, team_projections = load_and_project_all_teams(own_team_config)

    assert team_projections[111]["player_rows"][0]["country"] is None


def test_project_own_team_for_league_merges_country_when_available(own_team_config):
    processed = Path(own_team_config["paths"]["processed_data_dir"])
    pd.DataFrame([{"player_id": 1, "position": "Guard", "country": "Canada"}]).to_csv(
        processed / "roster_positions.csv", index=False
    )

    result = project_own_team_for_league(own_team_config)

    assert result["player_rows"][0]["country"] == "Canada"


# ---------------------------------------------------------------------------
# Métrica de impacto compuesta (src/advanced_impact.py) en el motor de liga
# ---------------------------------------------------------------------------


def _advanced_for(player_id, net_rating, seasons=("2023-24", "2024-25", "2025-26")):
    from advanced_impact import prepare_advanced_stats

    return prepare_advanced_stats(
        pd.DataFrame(
            [
                {"season": s, "PLAYER_ID": pid, "MIN": 30.0, "GP": 80,
                 "PIE": 0.10, "NET_RATING": net}
                for s in seasons
                for pid, net in ((player_id, net_rating), (999, 0.0))
            ]
        )
    )


def test_project_team_roster_applies_the_advanced_adjustment():
    from advanced_impact import build_advanced_context

    roster, regular = _minutes_only_roster([(1, "Buen defensor", 30.0, 80)])
    config = {"team": {"season": "2026-27"}, "simulation": {"games_per_season": 82}, "lineup_synergy": {}}
    context = build_advanced_context(_advanced_for(1, net_rating=10.0), config)

    plain = project_team_roster(roster, regular, pd.DataFrame(), config)
    adjusted = project_team_roster(roster, regular, pd.DataFrame(), config, advanced_context=context)

    # NET_RATING muy por encima de la media de liga -> más impacto.
    assert adjusted["game_score_per36"][0] > plain["game_score_per36"][0]


def test_exported_player_row_carries_the_same_metric_the_simulation_uses():
    """
    BUG REAL: el ajuste se aplicaba a la lista que consume la simulación
    pero NO a `projection`, que es lo que se vuelca en
    league_player_projections.csv -- y ese CSV es el que lee
    simulation.compute_league_average_game_score_per36 para su línea base.
    Simulación y línea base quedaban medidas en escalas distintas, que es
    exactamente el desajuste entre motores que este proyecto ya arrastró
    dos veces (normalización de minutos, escala a diferencial).
    """
    from advanced_impact import build_advanced_context

    roster, regular = _minutes_only_roster([(1, "Buen defensor", 30.0, 80)])
    config = {"team": {"season": "2026-27"}, "simulation": {"games_per_season": 82}, "lineup_synergy": {}}
    context = build_advanced_context(_advanced_for(1, net_rating=10.0), config)

    result = project_team_roster(roster, regular, pd.DataFrame(), config, advanced_context=context)
    exported = result["player_rows"][0]

    assert exported["game_score_per36"] == pytest.approx(result["game_score_per36"][0])
    # Y el Game Score de caja puro sigue disponible para comparar.
    assert exported["game_score_per36_box"] < exported["game_score_per36"]
