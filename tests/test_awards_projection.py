"""
Tests de src/awards_projection.py. Usan DataFrames sintéticos con el
esquema de league_player_projections.csv / aging_curve_projection.csv
(proyecciones) y roster_career_stats.csv / league_player_career_stats.csv
(carrera real, multi-temporada) -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from awards_projection import (  # noqa: E402
    ALL_DEFENSIVE_TEAM_NAMES,
    ALL_NBA_TEAM_NAMES,
    DEFAULT_MIN_GAMES_SEASON_AWARDS,
    compute_all_defensive_teams,
    compute_all_nba_teams,
    COMMISSIONER_ADDITION_SELECTION_TYPE,
    add_commissioner_picks_for_nationality_quota,
    check_all_star_nationality_quota,
    compute_all_star_selections,
    compute_bench_player_ids,
    compute_coy_candidates,
    compute_dpoy_candidates,
    compute_latest_real_season_stats,
    compute_mip_candidates,
    compute_mvp_candidates,
    compute_rookie_player_ids,
    compute_roy_candidates,
    compute_sixth_man_candidates,
)

GAMES_PER_SEASON = 82


def _player_row(
    player_id, name, team="AAA", game_score_per36=15.0, mpg=30.0,
    stl_per36=1.5, blk_per36=1.0, dreb_per36=5.0, pf_per36=2.0,
    risk_score=0.0, position=None, conference=None, country=None,
):
    return {
        "player_id": player_id,
        "player_name": name,
        "team_abbreviation": team,
        "conference": conference,
        "position": position,
        "country": country,
        "risk_score": risk_score,
        "game_score_per36": game_score_per36,
        "projected_total_minutes": mpg * GAMES_PER_SEASON,
        "STL_per36_projected": stl_per36,
        "BLK_per36_projected": blk_per36,
        "DREB_per36_projected": dreb_per36,
        "PF_per36_projected": pf_per36,
    }


@pytest.fixture
def player_df():
    return pd.DataFrame(
        [
            _player_row(1, "Star Player", team="AAA", game_score_per36=22.0, mpg=36.0),
            _player_row(2, "Solid Starter", team="AAA", game_score_per36=14.0, mpg=32.0),
            _player_row(3, "Marginal Bench", team="BBB", game_score_per36=8.0, mpg=8.0),
            _player_row(4, "Defensive Ace", team="BBB", game_score_per36=12.0, mpg=28.0, stl_per36=3.0, blk_per36=2.5),
        ]
    )


def test_compute_mvp_candidates_filters_low_minutes_and_ranks_by_value(player_df):
    result = compute_mvp_candidates(player_df, GAMES_PER_SEASON, top_n=5)

    # Marginal Bench (8 mpg) queda filtrado por debajo del umbral.
    assert "Marginal Bench" not in result["player_name"].tolist()
    assert result.iloc[0]["player_name"] == "Star Player"


def test_compute_mvp_candidates_weights_by_team_win_pct():
    df = pd.DataFrame(
        [
            _player_row(1, "Good Stats Bad Team", game_score_per36=20.0, mpg=35.0),
            _player_row(2, "Slightly Less Stats Great Team", game_score_per36=19.0, mpg=35.0),
        ]
    )
    team_win_pct = {1: 0.30, 2: 0.75}

    result = compute_mvp_candidates(df, GAMES_PER_SEASON, team_win_pct=team_win_pct, top_n=5)

    # El de peores stats pero mejor equipo debería superar al de mejores
    # stats en un equipo perdedor, tras el ajuste por win_pct.
    assert result.iloc[0]["player_name"] == "Slightly Less Stats Great Team"


def test_compute_dpoy_candidates_rewards_steals_and_blocks(player_df):
    result = compute_dpoy_candidates(player_df, GAMES_PER_SEASON, top_n=5)

    assert result.iloc[0]["player_name"] == "Defensive Ace"


def test_compute_bench_player_ids_uses_gs_gp_ratio_of_most_recent_season():
    career = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "SEASON_ID": "2024-25", "GP": 70, "GS": 5},   # banquillo
            {"PLAYER_ID": 2, "SEASON_ID": "2024-25", "GP": 70, "GS": 68},  # titular
            # Jugador 1 fue titular una temporada más vieja -- debe primar la más reciente.
            {"PLAYER_ID": 1, "SEASON_ID": "2022-23", "GP": 65, "GS": 60},
        ]
    )

    bench_ids = compute_bench_player_ids(career)

    assert bench_ids == {1}


def test_compute_sixth_man_candidates_only_considers_bench_players(player_df):
    result = compute_sixth_man_candidates(player_df, bench_player_ids={2, 4}, games_per_season=GAMES_PER_SEASON, top_n=5)

    assert set(result["player_id"]) <= {2, 4}
    assert result.iloc[0]["player_id"] == 2  # mayor season_value entre los dos


def test_compute_rookie_player_ids_flags_single_season_players():
    career = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "SEASON_ID": "2025-26", "GP": 70, "GS": 10},
            {"PLAYER_ID": 2, "SEASON_ID": "2025-26", "GP": 70, "GS": 10},
            {"PLAYER_ID": 2, "SEASON_ID": "2024-25", "GP": 60, "GS": 5},
        ]
    )

    rookie_ids = compute_rookie_player_ids(career)

    assert rookie_ids == {1}


def test_compute_roy_candidates_filters_to_rookies(player_df):
    result = compute_roy_candidates(player_df, rookie_player_ids={3}, games_per_season=GAMES_PER_SEASON, min_mpg=0.0, top_n=5)

    assert result["player_id"].tolist() == [3]


def _career_row(player_id, season, gp, minutes, pts, name="Player"):
    return {
        "PLAYER_ID": player_id, "SEASON_ID": season, "GP": gp, "MIN": minutes,
        "PTS": pts, "AST": 100, "REB": 200, "STL": 40, "BLK": 20, "TOV": 60,
        "FG3M": 40, "FG3A": 110, "OREB": 40, "DREB": 160, "FGM": 200, "FGA": 420,
        "FTM": 100, "FTA": 130, "PF": 120, "player_name": name,
    }


def test_compute_mip_candidates_ranks_by_game_score_improvement():
    career = pd.DataFrame(
        [
            _career_row(1, "2023-24", 70, 1800, 700, name="Big Leap"),
            _career_row(1, "2024-25", 75, 2200, 1400, name="Big Leap"),
            _career_row(2, "2023-24", 70, 1800, 900, name="Stayed Same"),
            _career_row(2, "2024-25", 75, 2200, 950, name="Stayed Same"),
        ]
    )

    result = compute_mip_candidates(career, min_minutes_per_game=10.0, top_n=5)

    assert result.iloc[0]["player_name"] == "Big Leap"
    assert result.iloc[0]["improvement"] > 0


def test_compute_mip_candidates_excludes_players_with_only_one_season():
    career = pd.DataFrame([_career_row(1, "2025-26", 70, 2000, 1200, name="Rookie")])

    result = compute_mip_candidates(career, top_n=5)

    assert result.empty


def test_compute_mip_candidates_excludes_low_minutes_seasons():
    # Salto de volumen bajísimo -- no debería contar como "mejora" real.
    career = pd.DataFrame(
        [
            _career_row(1, "2023-24", 20, 100, 40, name="Garbage Time"),
            _career_row(1, "2024-25", 20, 120, 90, name="Garbage Time"),
        ]
    )

    result = compute_mip_candidates(career, min_minutes_per_game=15.0, top_n=5)

    assert result.empty


def test_compute_latest_real_season_stats_uses_the_most_recent_season_per_game():
    # A diferencia de compute_mip_candidates (que usa la PENÚLTIMA
    # temporada real como "previous"), esto debe usar la ÚLTIMA
    # (2024-25), la que precede a la proyección.
    career = pd.DataFrame(
        [
            _career_row(1, "2023-24", 70, 1800, 700, name="Big Leap"),
            _career_row(1, "2024-25", 75, 2200, 1500, name="Big Leap"),
        ]
    )

    result = compute_latest_real_season_stats(career)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["prev_season"] == "2024-25"
    assert row["prev_PPG"] == pytest.approx(round(1500 / 75, 1))
    assert row["prev_APG"] == pytest.approx(round(100 / 75, 1))


def test_compute_latest_real_season_stats_handles_missing_shooting_columns():
    """FG_PCT/FG3_PCT no siempre están (p.ej. career_stats_df sintético de
    otros tests) -- no debe romper, solo devolver NaN para esas dos."""
    career = pd.DataFrame([_career_row(1, "2024-25", 70, 2000, 1200, name="Player")])

    result = compute_latest_real_season_stats(career)

    assert pd.isna(result.iloc[0]["prev_FG%"])
    assert pd.isna(result.iloc[0]["prev_3P%"])


def test_compute_latest_real_season_stats_empty_input():
    assert compute_latest_real_season_stats(pd.DataFrame()).empty


def test_compute_coy_candidates_ranks_by_win_improvement():
    team_wins = pd.DataFrame(
        [
            {"team_abbreviation": "AAA", "conference": "East", "wins_mean": 55.0},
            {"team_abbreviation": "BBB", "conference": "West", "wins_mean": 40.0},
        ]
    )
    prior_wins = {"AAA": 30.0, "BBB": 38.0}

    result = compute_coy_candidates(team_wins, prior_wins, top_n=5)

    assert result.iloc[0]["team_abbreviation"] == "AAA"
    assert result.iloc[0]["win_improvement"] == pytest.approx(25.0)


def test_compute_coy_candidates_drops_teams_without_prior_data():
    team_wins = pd.DataFrame(
        [
            {"team_abbreviation": "AAA", "wins_mean": 55.0},
            {"team_abbreviation": "CCC", "wins_mean": 45.0},  # sin dato de temporada anterior
        ]
    )
    prior_wins = {"AAA": 30.0}

    result = compute_coy_candidates(team_wins, prior_wins, top_n=5)

    assert result["team_abbreviation"].tolist() == ["AAA"]


# ---------------------------------------------------------------------------
# All-Star / All-NBA / All-Defensive
# ---------------------------------------------------------------------------


def test_all_star_selections_has_no_minimum_games_or_minutes_threshold():
    """El All-Star se vota a mitad de temporada -- ni el filtro de mpg de
    MVP/DPOY ni el umbral de 65 partidos de los quintetos de fin de
    temporada deben aplicarse aquí."""
    df = pd.DataFrame([
        _player_row(1, "Low Minutes High Value", game_score_per36=30.0, mpg=10.0, risk_score=0.9),
    ])
    result = compute_all_star_selections(df, GAMES_PER_SEASON)
    assert "Low Minutes High Value" in result["player_name"].tolist()


def test_all_star_selections_splits_by_conference_when_available():
    df = pd.DataFrame([
        _player_row(1, "East Star", game_score_per36=25.0, conference="East"),
        _player_row(2, "West Star", game_score_per36=20.0, conference="West"),
    ])
    result = compute_all_star_selections(df, GAMES_PER_SEASON, n_per_conference=1)
    assert set(result["player_name"]) == {"East Star", "West Star"}
    assert len(result) == 2  # uno por conferencia, no un top-1 global


def test_all_star_selections_falls_back_to_flat_ranking_without_conference():
    """Scope 'own': un solo equipo, sin conferencia -- top_n plano."""
    df = pd.DataFrame([
        _player_row(1, "A", game_score_per36=20.0),
        _player_row(2, "B", game_score_per36=15.0),
        _player_row(3, "C", game_score_per36=10.0),
    ])
    result = compute_all_star_selections(df, GAMES_PER_SEASON, n_per_conference=2)
    assert result["player_name"].tolist() == ["A", "B"]


def _all_nba_pool(n_per_position=3, risk_score=0.0):
    """Roster sintético con suficientes G/F/C para llenar los 3 quintetos
    All-NBA (2G+2F+1C cada uno -- necesita >=6 G, >=6 F, >=3 C)."""
    rows = []
    pid = 1
    for position, count in [("Guard", max(n_per_position, 6)), ("Forward", max(n_per_position, 6)), ("Center", max(n_per_position, 3))]:
        for i in range(count):
            rows.append(_player_row(
                pid, f"{position[0]}{i}", game_score_per36=30.0 - i, position=position, risk_score=risk_score,
            ))
            pid += 1
    return pd.DataFrame(rows)


def test_all_nba_teams_fills_the_classic_2_2_1_format():
    result = compute_all_nba_teams(_all_nba_pool(), GAMES_PER_SEASON)
    first_team = result[result["team"] == ALL_NBA_TEAM_NAMES[0]]
    assert sorted(first_team["position_slot"].tolist()) == ["C", "F", "F", "G", "G"]


def test_all_nba_teams_ranks_best_players_into_the_first_team():
    result = compute_all_nba_teams(_all_nba_pool(), GAMES_PER_SEASON)
    first_team_guards = result[(result["team"] == ALL_NBA_TEAM_NAMES[0]) & (result["position_slot"] == "G")]
    second_team_guards = result[(result["team"] == ALL_NBA_TEAM_NAMES[1]) & (result["position_slot"] == "G")]
    # G0/G1 tienen más season_value que G2/G3 (game_score_per36 decreciente por índice).
    assert set(first_team_guards["player_name"]) == {"G0", "G1"}
    assert set(second_team_guards["player_name"]) == {"G2", "G3"}


def test_all_nba_teams_never_repeats_a_player_across_teams():
    result = compute_all_nba_teams(_all_nba_pool(), GAMES_PER_SEASON)
    assert result["player_name"].is_unique


def test_all_nba_teams_excludes_players_below_the_games_threshold():
    df = _all_nba_pool(risk_score=0.0)
    # Un solo jugador con riesgo alto -- 82 * (1 - 0.5) = 41 < 65, inelegible.
    df.loc[df["player_name"] == "G0", "risk_score"] = 0.5
    result = compute_all_nba_teams(df, GAMES_PER_SEASON)
    assert "G0" not in result["player_name"].tolist()


def test_all_nba_teams_respects_a_custom_games_threshold():
    df = _all_nba_pool()
    df["risk_score"] = 0.3  # 82*(1-0.3) = 57.4 -- inelegible con 65, elegible con 50.
    assert compute_all_nba_teams(df, GAMES_PER_SEASON, min_games_played=65).empty
    assert not compute_all_nba_teams(df, GAMES_PER_SEASON, min_games_played=50).empty


def test_all_nba_teams_excludes_players_without_known_position():
    df = _all_nba_pool()
    df.loc[df["player_name"] == "G0", "position"] = None
    result = compute_all_nba_teams(df, GAMES_PER_SEASON)
    assert "G0" not in result["player_name"].tolist()


def test_all_nba_teams_leaves_a_slot_empty_when_no_eligible_candidate():
    """Roster sin ningún pívot real -- el cupo de C debe quedar vacío en
    vez de rellenarse con un jugador de otra posición."""
    df = _all_nba_pool()
    df = df[df["position"] != "Center"]
    result = compute_all_nba_teams(df, GAMES_PER_SEASON)
    assert "C" not in result["position_slot"].tolist()


def test_all_nba_teams_include_offensive_comparison_stats_and_team_record():
    df = pd.DataFrame([_player_row_with_stats(pid, f"P{pid}", position="Guard") for pid in range(1, 7)])
    result = compute_all_nba_teams(df, GAMES_PER_SEASON, team_record={1: "50-32"})

    for col in ["PPG", "RPG", "APG", "SPG", "BPG", "FG%", "3P%", "team_record"]:
        assert col in result.columns
    assert result[result["player_id"] == 1].iloc[0]["team_record"] == "50-32"


def test_all_defensive_teams_ranks_by_defensive_value_not_offensive_value():
    df = pd.DataFrame([
        # G0: mejor Game Score ofensivo, peor defensa -- no debería entrar.
        _player_row(1, "G0", position="Guard", game_score_per36=30.0, stl_per36=0.1, blk_per36=0.1),
        _player_row(2, "G1", position="Guard", game_score_per36=5.0, stl_per36=4.0, blk_per36=3.0),
        _player_row(3, "G2", position="Guard", game_score_per36=5.0, stl_per36=3.5, blk_per36=2.5),
        _player_row(4, "F0", position="Forward", stl_per36=0.1, blk_per36=0.1),
        _player_row(5, "F1", position="Forward", stl_per36=0.1, blk_per36=0.1),
        _player_row(6, "C0", position="Center", stl_per36=0.1, blk_per36=0.1),
    ])
    result = compute_all_defensive_teams(df, GAMES_PER_SEASON)
    first_team_guards = result[(result["team"] == ALL_DEFENSIVE_TEAM_NAMES[0]) & (result["position_slot"] == "G")]
    # G1 y G2 defienden mucho mejor que G0 -- deben ganarle el cupo (solo 2 de 3 caben).
    assert set(first_team_guards["player_name"]) == {"G1", "G2"}


def test_all_defensive_teams_has_two_teams_not_three():
    result = compute_all_defensive_teams(_all_nba_pool(), GAMES_PER_SEASON)
    assert set(result["team"]) <= set(ALL_DEFENSIVE_TEAM_NAMES)
    assert ALL_NBA_TEAM_NAMES[2] not in result["team"].tolist()  # no hay "tercer equipo" en defensivo


def test_all_defensive_teams_include_offensive_comparison_stats_and_team_record():
    df = pd.DataFrame([_player_row_with_stats(pid, f"P{pid}", position="Guard") for pid in range(1, 7)])
    result = compute_all_defensive_teams(df, GAMES_PER_SEASON, team_record={1: "50-32"})

    for col in ["PPG", "RPG", "APG", "SPG", "BPG", "FG%", "3P%", "team_record"]:
        assert col in result.columns
    assert result[result["player_id"] == 1].iloc[0]["team_record"] == "50-32"


def test_default_games_threshold_matches_the_real_nba_policy():
    assert DEFAULT_MIN_GAMES_SEASON_AWARDS == 65


# ---------------------------------------------------------------------------
# Titulares/Reservas + cuota de nacionalidad del All-Star
# ---------------------------------------------------------------------------


def test_all_star_selections_labels_top_5_as_starters_per_conference():
    df = pd.DataFrame([
        _player_row(i, f"P{i}", game_score_per36=30.0 - i, conference="East")
        for i in range(8)
    ])
    result = compute_all_star_selections(df, GAMES_PER_SEASON, n_per_conference=8, n_starters_per_conference=5)

    starters = result[result["selection_type"] == "Titular"]["player_name"].tolist()
    reserves = result[result["selection_type"] == "Reserva"]["player_name"].tolist()
    assert starters == ["P0", "P1", "P2", "P3", "P4"]
    assert reserves == ["P5", "P6", "P7"]


def test_all_star_selections_labels_starters_within_each_conference_independently():
    df = pd.DataFrame([
        _player_row(1, "East Best", game_score_per36=10.0, conference="East"),
        _player_row(2, "West Best", game_score_per36=50.0, conference="West"),
    ])
    result = compute_all_star_selections(df, GAMES_PER_SEASON, n_per_conference=1, n_starters_per_conference=1)

    # East Best es "peor" en valor absoluto que West Best, pero al ser el
    # ÚNICO de su conferencia, también debe salir "Titular" -- el split
    # es POR conferencia, no global.
    row = result[result["player_name"] == "East Best"].iloc[0]
    assert row["selection_type"] == "Titular"


def test_check_all_star_nationality_quota_counts_correctly():
    selections = pd.DataFrame([
        {"player_name": f"US{i}", "country": "USA"} for i in range(16)
    ] + [
        {"player_name": f"Intl{i}", "country": "Spain"} for i in range(8)
    ])
    result = check_all_star_nationality_quota(selections)

    assert result["checked"] is True
    assert result["us_count"] == 16
    assert result["international_count"] == 8
    assert result["unknown_count"] == 0
    assert result["meets_us_minimum"] is True
    assert result["meets_international_minimum"] is True
    assert result["meets_both"] is True


def test_check_all_star_nationality_quota_flags_shortfall():
    selections = pd.DataFrame(
        [{"player_name": f"US{i}", "country": "USA"} for i in range(20)]
        + [{"player_name": "Intl0", "country": "France"}]
    )
    result = check_all_star_nationality_quota(selections)

    assert result["international_count"] == 1
    assert result["meets_international_minimum"] is False
    assert result["meets_us_minimum"] is True
    assert result["meets_both"] is False


def test_check_all_star_nationality_quota_counts_unknown_separately():
    selections = pd.DataFrame([
        {"player_name": "A", "country": "USA"},
        {"player_name": "B", "country": None},
    ])
    result = check_all_star_nationality_quota(selections)

    assert result["us_count"] == 1
    assert result["international_count"] == 0
    assert result["unknown_count"] == 1


def test_check_all_star_nationality_quota_returns_unchecked_without_country_column():
    selections = pd.DataFrame([{"player_name": "A", "season_value": 10.0}])
    result = check_all_star_nationality_quota(selections)

    assert result["checked"] is False
    assert result["meets_both"] is None


def test_check_all_star_nationality_quota_respects_custom_thresholds():
    selections = pd.DataFrame([{"player_name": "A", "country": "USA"}])
    result = check_all_star_nationality_quota(selections, min_us=1, min_international=0)

    assert result["meets_both"] is True


# ---------------------------------------------------------------------------
# add_commissioner_picks_for_nationality_quota
# ---------------------------------------------------------------------------


def _pool_with_countries(n_us=20, n_international=2, start_id=100, extra_country="Spain"):
    rows = [_player_row(start_id + i, f"US{i}", game_score_per36=20.0 - i, country="USA") for i in range(n_us)]
    rows += [
        _player_row(start_id + 1000 + i, f"Intl{i}", game_score_per36=15.0 - i, country=extra_country)
        for i in range(n_international)
    ]
    return pd.DataFrame(rows)


def test_commissioner_additions_noop_when_quota_already_met():
    selections = pd.DataFrame([{"player_id": 1, "player_name": "A", "country": "USA"}])
    quota = check_all_star_nationality_quota(selections, min_us=1, min_international=0)

    result = add_commissioner_picks_for_nationality_quota(_pool_with_countries(), selections, quota)

    assert "commissioner_pick" in result.columns
    assert not result["commissioner_pick"].any()
    assert len(result) == len(selections)


def test_commissioner_additions_fill_the_international_shortfall():
    # 24 seleccionados, todos de EE.UU. -- 0 internacionales, por debajo del mínimo 8.
    selections = pd.DataFrame([_player_row(i, f"Sel{i}", country="USA") for i in range(24)])
    quota = check_all_star_nationality_quota(selections)  # mínimos por defecto: 16 US / 8 intl
    pool = _pool_with_countries(n_us=0, n_international=10)

    result = add_commissioner_picks_for_nationality_quota(pool, selections, quota)

    added = result[result["commissioner_pick"]]
    assert len(added) == 8  # exactamente lo que faltaba para llegar a 8
    assert (added["country"] == "Spain").all()
    assert (added["selection_type"] == COMMISSIONER_ADDITION_SELECTION_TYPE).all()


def test_commissioner_additions_pick_the_highest_value_eligible_player_first():
    selections = pd.DataFrame([_player_row(i, f"Sel{i}", country="USA") for i in range(24)])
    quota = check_all_star_nationality_quota(selections)
    pool = _pool_with_countries(n_us=0, n_international=10)  # Intl0 tiene el mayor game_score_per36

    result = add_commissioner_picks_for_nationality_quota(pool, selections, quota)

    added_names = result[result["commissioner_pick"]]["player_name"].tolist()
    assert "Intl0" in added_names  # el de mayor valor de temporada debe entrar


def test_commissioner_additions_never_repick_an_already_selected_player():
    selections = pd.DataFrame([_player_row(i, f"Sel{i}", country="USA") for i in range(24)])
    quota = check_all_star_nationality_quota(selections)
    # El pool incluye a los 24 ya seleccionados -- no deben poder "añadirse" de nuevo.
    pool = pd.concat([selections, _pool_with_countries(n_us=0, n_international=10)], ignore_index=True)

    result = add_commissioner_picks_for_nationality_quota(pool, selections, quota)

    assert result["player_id"].is_unique


def test_commissioner_additions_leaves_selection_unchanged_without_eligible_candidates():
    selections = pd.DataFrame([_player_row(i, f"Sel{i}", country="USA") for i in range(24)])
    quota = check_all_star_nationality_quota(selections)
    pool = _pool_with_countries(n_us=5, n_international=0)  # sin ningún internacional disponible

    result = add_commissioner_picks_for_nationality_quota(pool, selections, quota)

    assert not result["commissioner_pick"].any()
    assert len(result) == len(selections)


def test_commissioner_additions_require_country_data_in_quota():
    selections = pd.DataFrame([{"player_id": 1, "player_name": "A"}])  # sin country
    quota = check_all_star_nationality_quota(selections)  # checked=False

    result = add_commissioner_picks_for_nationality_quota(_pool_with_countries(), selections, quota)

    assert "commissioner_pick" not in result.columns or not result["commissioner_pick"].any()


# ---------------------------------------------------------------------------
# Stats de comparación (PPG/RPG/APG/SPG/BPG/FG%/3P%) y récord de equipo
# ---------------------------------------------------------------------------


def _player_row_with_stats(player_id, name, team="AAA", ppg=20.0, rpg=5.0, apg=5.0, spg=1.0, bpg=0.5, fg_pct=45.0, three_pct=35.0, pf_projected=None, **kwargs):
    row = _player_row(player_id, name, team=team, **kwargs)
    row.update({"PPG": ppg, "RPG": rpg, "APG": apg, "SPG": spg, "BPG": bpg, "FG%": fg_pct, "3P%": three_pct})
    if pf_projected is not None:
        row["PF_projected"] = pf_projected
    return row


def test_mvp_candidates_include_offensive_comparison_stats_and_team_record():
    df = pd.DataFrame([_player_row_with_stats(1, "Star", game_score_per36=22.0, mpg=36.0)])
    result = compute_mvp_candidates(df, GAMES_PER_SEASON, team_record={1: "50-32"}, top_n=5)

    for col in ["PPG", "RPG", "APG", "SPG", "BPG", "FG%", "3P%"]:
        assert col in result.columns
    assert result.iloc[0]["team_record"] == "50-32"


def test_dpoy_candidates_include_full_comparison_stats():
    """El ranking (dpoy_score) sigue siendo puramente defensivo, pero las
    columnas devueltas incluyen también PPG/APG/tiro, para que la vista
    previa al pasar el ratón en webapp/ muestre el mismo set de stats
    que el resto de premios."""
    df = pd.DataFrame([_player_row_with_stats(1, "Defender", pf_projected=200.0)])
    result = compute_dpoy_candidates(df, GAMES_PER_SEASON, team_record={1: "50-32"}, top_n=5)

    for col in ["RPG", "SPG", "BPG", "PFPG", "team_record", "PPG", "APG", "FG%", "3P%"]:
        assert col in result.columns
    assert result.iloc[0]["PFPG"] == pytest.approx(round(200.0 / GAMES_PER_SEASON, 1))


def test_sixth_man_candidates_include_offensive_comparison_stats():
    df = pd.DataFrame([_player_row_with_stats(2, "Bench Scorer", mpg=25.0)])
    result = compute_sixth_man_candidates(df, bench_player_ids={2}, games_per_season=GAMES_PER_SEASON, team_record={2: "40-42"})

    assert "PPG" in result.columns
    assert result.iloc[0]["team_record"] == "40-42"


def test_roy_candidates_include_offensive_comparison_stats():
    df = pd.DataFrame([_player_row_with_stats(3, "Rookie", mpg=28.0)])
    result = compute_roy_candidates(df, rookie_player_ids={3}, games_per_season=GAMES_PER_SEASON, min_mpg=0.0, team_record={3: "20-62"})

    assert "PPG" in result.columns
    assert result.iloc[0]["team_record"] == "20-62"


def test_comparison_stats_are_optional_and_do_not_break_without_them():
    """Los candidatos que no traigan PPG/RPG/etc en player_df (esquema
    antiguo o CSV incompleto) no deben fallar -- solo se omiten esas
    columnas, degradando en vez de romper."""
    df = pd.DataFrame([_player_row(1, "Star", game_score_per36=22.0, mpg=36.0)])
    result = compute_mvp_candidates(df, GAMES_PER_SEASON, top_n=5)

    assert "PPG" not in result.columns
    assert result.iloc[0]["player_name"] == "Star"
