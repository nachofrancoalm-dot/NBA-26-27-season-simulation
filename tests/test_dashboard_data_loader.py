"""
Tests de dashboard/data_loader.py. Usan un config + directorio temporal
con CSV sintéticos que imitan data/processed/ -- no requieren red ni el
pipeline real corrido.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import (  # noqa: E402
    LEAGUE_PLAYER_META_COLUMNS,
    PER_GAME_STATS,
    ROSTER_STAT_GLOSSARY,
    TOTAL_STATS,
    compute_awards_summary,
    compute_champion_seed_distribution,
    compute_conference_standings,
    compute_win_distribution_summary,
    load_backtest_summary,
    load_champion_title_paths,
    load_backtest_sweep_calibration,
    load_backtest_sweep_summary,
    load_league_playoff_summary,
    load_league_player_projections,
    load_league_regular_season_summary,
    load_league_single_season_game_log,
    load_league_single_season_player_box_scores,
    load_lineup_synergy_pairs,
    load_roster_overview,
    load_simulation_results,
    run_single_season_player_log_simulation,
    select_roster_view,
)


@pytest.fixture
def config(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    return {
        "roster": [
            {"player_id": 1, "name": "Player One", "role_expected": "scorer", "minutes_projection": 34, "unit": "starter"},
            {"player_id": 2, "name": "Player Two", "role_expected": "facilitator", "minutes_projection": 30, "unit": "starter"},
        ],
        "simulation": {"games_per_season": 82},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed_dir)},
    }


def test_load_roster_overview_returns_none_when_aging_csv_missing(config):
    assert load_roster_overview(config) is None


def test_load_roster_overview_merges_role_and_risk_and_fatigue(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {"player_id": 1, "player_name": "Player One", "current_age": 27, "target_age": 28,
             "game_score_per36": 18.0, "PTS_projected": 1600, "AST_projected": 400,
             "REB_projected": 500, "FG3M_projected": 150},
            {"player_id": 2, "player_name": "Player Two", "current_age": 24, "target_age": 25,
             "game_score_per36": 12.0, "PTS_projected": 900, "AST_projected": 600,
             "REB_projected": 300, "FG3M_projected": 90},
        ]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "risk_score": 0.3}, {"player_id": 2, "risk_score": 0.1}]).to_csv(
        processed / "injury_risk.csv", index=False
    )
    pd.DataFrame([{"player_id": 1, "fatigue_score": 0.4}, {"player_id": 2, "fatigue_score": 0.2}]).to_csv(
        processed / "fatigue_risk.csv", index=False
    )

    overview = load_roster_overview(config)

    assert overview is not None
    assert "role_expected" in overview.columns
    assert "risk_score" in overview.columns
    assert "fatigue_score" in overview.columns
    # Ordenado de mayor a menor game_score_per36.
    assert overview.iloc[0]["player_name"] == "Player One"
    assert overview.iloc[0]["risk_score"] == pytest.approx(0.3)
    # Stats por partido derivadas de los totales / games_per_season.
    assert overview.iloc[0]["PPG"] == pytest.approx(1600 / 82)
    assert overview.iloc[0]["RPG"] == pytest.approx(500 / 82)
    assert overview.iloc[0]["APG"] == pytest.approx(400 / 82)
    assert overview.iloc[0]["3PM"] == pytest.approx(150 / 82)


def test_load_roster_overview_merges_position(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Player One", "game_score_per36": 18.0}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "position": "Guard"}]).to_csv(processed / "roster_positions.csv", index=False)

    overview = load_roster_overview(config)

    assert overview.iloc[0]["position"] == "Guard"


def test_load_roster_overview_merges_country(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Player One", "game_score_per36": 18.0}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "position": "Guard", "country": "Serbia"}]).to_csv(
        processed / "roster_positions.csv", index=False
    )

    overview = load_roster_overview(config)

    assert overview.iloc[0]["country"] == "Serbia"


def test_load_roster_overview_works_without_positions_csv(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Player One", "game_score_per36": 18.0}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)

    overview = load_roster_overview(config)

    assert overview is not None
    assert "position" not in overview.columns


def test_load_roster_overview_computes_field_goal_percentage(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Player One", "current_age": 27, "target_age": 28,
          "game_score_per36": 18.0, "PTS_projected": 1600, "AST_projected": 400,
          "REB_projected": 500, "FG3M_projected": 150,
          "FGM_projected": 500, "FGA_projected": 1000}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)

    overview = load_roster_overview(config)
    assert overview.iloc[0]["FG%"] == pytest.approx(50.0)


def test_per_game_stats_and_glossary_share_the_same_keys():
    # La leyenda debe cubrir al menos todas las stats por partido definidas.
    assert set(PER_GAME_STATS.keys()).issubset(set(ROSTER_STAT_GLOSSARY.keys()))


def test_load_roster_overview_works_without_optional_injury_or_fatigue_csvs(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Player One", "current_age": 27, "target_age": 28,
          "game_score_per36": 18.0, "PTS_projected": 1600, "AST_projected": 400,
          "REB_projected": 500, "FG3M_projected": 150}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)

    overview = load_roster_overview(config)

    assert overview is not None
    assert "risk_score" not in overview.columns  # no se pudo mergear, no se inventa la columna


def test_load_simulation_results_returns_none_when_missing(config):
    assert load_simulation_results(config) is None


def test_load_simulation_results_reads_existing_csv(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame({"season_index": [0, 1], "wins": [45, 50], "losses": [37, 32],
                  "net_rating_estimate_mean": [1.2, 3.4], "total_games_missed": [10, 5]}).to_csv(
        processed / "simulation_results.csv", index=False
    )
    result = load_simulation_results(config)
    assert result is not None
    assert len(result) == 2


def test_compute_win_distribution_summary():
    df = pd.DataFrame({"wins": [40, 45, 50, 55, 60]})
    summary = compute_win_distribution_summary(df)
    assert summary["mean"] == pytest.approx(50.0)
    assert summary["min"] == 40.0
    assert summary["max"] == 60.0


def test_load_backtest_summary_and_synergy_pairs_return_none_when_missing(config):
    assert load_backtest_summary(config) is None
    assert load_lineup_synergy_pairs(config) is None


def test_load_backtest_sweep_files_return_none_when_missing(config):
    assert load_backtest_sweep_summary(config) is None
    assert load_backtest_sweep_calibration(config) is None


def test_load_backtest_sweep_files_read_existing_csvs(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame([{"comparable_name": "ATL 2010-11", "actual_wins": 44, "actual_percentile": 55.0}]).to_csv(
        processed / "backtest_sweep_summary.csv", index=False
    )
    pd.DataFrame([{"n_cases": 1, "pct_within_p10_p90": 100.0}]).to_csv(
        processed / "backtest_sweep_calibration.csv", index=False
    )

    summary = load_backtest_sweep_summary(config)
    calibration = load_backtest_sweep_calibration(config)

    assert summary is not None and len(summary) == 1
    assert calibration is not None and calibration.iloc[0]["n_cases"] == 1


def test_load_league_summaries_return_none_when_missing(config):
    assert load_league_regular_season_summary(config) is None
    assert load_league_playoff_summary(config) is None


def test_load_league_player_projections_returns_none_when_missing(config):
    assert load_league_player_projections(config) is None


def test_load_league_player_projections_reads_existing_csv(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Rival Player", "team_abbreviation": "BOS", "conference": "East",
          "current_age": 26, "target_age": 27, "minutes_projection": 32.0, "game_score_per36": 15.0,
          "risk_score": 0.2, "fatigue_score": 0.3, "PPG": 20.1, "PTS": 1648.2}]
    ).to_csv(processed / "league_player_projections.csv", index=False)

    result = load_league_player_projections(config)
    assert result is not None
    assert result.iloc[0]["team_abbreviation"] == "BOS"


def test_load_league_single_season_data_returns_none_when_missing(config):
    assert load_league_single_season_game_log(config) is None
    assert load_league_single_season_player_box_scores(config) is None


def test_load_league_single_season_data_reads_existing_csv_and_respects_scenario(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame([{"game_id": 0, "day": 0, "home_team_id": 1, "away_team_id": 2, "winner_team_id": 1}]).to_csv(
        processed / "league_single_season_game_log.csv", index=False
    )
    pd.DataFrame([{"game_id": 0, "team_id": 1, "player_id": 100, "PTS": 20.0}]).to_csv(
        processed / "league_single_season_player_box_scores.csv", index=False
    )

    game_log = load_league_single_season_game_log(config)
    box_scores = load_league_single_season_player_box_scores(config)
    assert game_log is not None and game_log.iloc[0]["winner_team_id"] == 1
    assert box_scores is not None and box_scores.iloc[0]["PTS"] == 20.0
    # Sin CSV del escenario "no_injuries" todavia -- None, no revienta.
    assert load_league_single_season_game_log(config, scenario="no_injuries") is None


def test_select_roster_view_totals_mode_uses_clean_column_names():
    overview = pd.DataFrame(
        [{"player_name": "Player One", "game_score_per36": 18.0, "risk_score": 0.3,
          "PTS_projected": 1600, "REB_projected": 500, "PPG": 19.5, "RPG": 6.1}]
    )
    view = select_roster_view(overview, mode="totals")
    assert "PTS" in view.columns
    assert view.iloc[0]["PTS"] == 1600
    assert "PPG" not in view.columns


def test_select_roster_view_per_game_mode_uses_per_game_columns():
    overview = pd.DataFrame(
        [{"player_name": "Player One", "game_score_per36": 18.0, "risk_score": 0.3,
          "PTS_projected": 1600, "REB_projected": 500, "PPG": 19.5, "RPG": 6.1}]
    )
    view = select_roster_view(overview, mode="per_game")
    assert "PPG" in view.columns
    assert view.iloc[0]["PPG"] == pytest.approx(19.5)
    assert "PTS" not in view.columns


def test_select_roster_view_supports_custom_meta_columns_for_league_players():
    overview = pd.DataFrame(
        [{"player_name": "Rival Player", "team_abbreviation": "BOS", "conference": "East",
          "game_score_per36": 15.0, "risk_score": 0.2, "fatigue_score": 0.3,
          "PTS_projected": 1600, "PPG": 19.5}]
    )
    view = select_roster_view(overview, mode="per_game", meta_columns=LEAGUE_PLAYER_META_COLUMNS)
    assert "team_abbreviation" in view.columns
    assert "role_expected" not in view.columns  # los jugadores de liga no tienen ese campo


def test_total_stats_and_per_game_stats_cover_the_same_totals():
    assert set(TOTAL_STATS.keys()) == set(PER_GAME_STATS.values())


def _projection_row(player_id, name, game_score_per36=15.0, mpg=30.0, games_per_season=82):
    return {
        "player_id": player_id, "player_name": name, "current_age": 25, "target_age": 26,
        "game_score_per36": game_score_per36, "projected_total_minutes": mpg * games_per_season,
        "STL_per36_projected": 1.5, "BLK_per36_projected": 1.0, "DREB_per36_projected": 5.0,
        "PF_per36_projected": 2.0,
    }


def _career_row(player_id, season, gp, gs, minutes, pts, name):
    return {
        "PLAYER_ID": player_id, "SEASON_ID": season, "GP": gp, "GS": gs, "MIN": minutes,
        "PTS": pts, "AST": 100, "REB": 200, "STL": 40, "BLK": 20, "TOV": 60,
        "FG3M": 40, "FG3A": 110, "OREB": 40, "DREB": 160, "FGM": 200, "FGA": 420,
        "FTM": 100, "FTA": 130, "PF": 120, "player_name": name,
    }


def test_compute_awards_summary_returns_none_without_any_projection_data(config):
    assert compute_awards_summary(config) is None


def test_compute_awards_summary_own_scope_has_no_coy_and_uses_roster_data(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            _projection_row(1, "Player One", game_score_per36=22.0, mpg=36.0),
            _projection_row(2, "Player Two", game_score_per36=10.0, mpg=8.0),
        ]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame(
        [
            _career_row(1, "2024-25", 70, 68, 2400, 1400, "Player One"),
            _career_row(1, "2023-24", 70, 65, 2300, 1100, "Player One"),
            _career_row(2, "2025-26", 40, 2, 300, 90, "Player Two"),
        ]
    ).to_csv(processed / "roster_career_stats.csv", index=False)

    result = compute_awards_summary(config)

    assert result is not None
    assert result["scope"] == "own"
    assert result["coy"] is None
    assert result["mvp"].iloc[0]["player_name"] == "Player One"
    assert set(result["roy"]["player_id"]) <= {2}  # Player Two: 1 sola temporada -> rookie


def test_compute_awards_summary_league_scope_computes_coy(config):
    from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID

    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {**_projection_row(1, "Star A", game_score_per36=22.0, mpg=36.0), "team_abbreviation": "ATL", "conference": "East"},
            {**_projection_row(2, "Star B", game_score_per36=20.0, mpg=34.0), "team_abbreviation": "BOS", "conference": "East"},
        ]
    ).to_csv(processed / "league_player_projections.csv", index=False)
    pd.DataFrame(
        [
            {"team_abbreviation": "ATL", "conference": "East", "wins_mean": 55.0, "wins_p10": 48, "wins_p90": 60},
            {"team_abbreviation": "BOS", "conference": "East", "wins_mean": 30.0, "wins_p10": 25, "wins_p90": 35},
        ]
    ).to_csv(processed / "league_regular_season_summary.csv", index=False)
    pd.DataFrame(
        [
            _career_row(1, "2024-25", 70, 68, 2400, 1400, "Star A"),
            _career_row(1, "2023-24", 70, 65, 2300, 1100, "Star A"),
            _career_row(2, "2024-25", 70, 68, 2300, 1200, "Star B"),
            _career_row(2, "2023-24", 70, 65, 2200, 1150, "Star B"),
        ]
    ).to_csv(processed / "league_player_career_stats.csv", index=False)
    pd.DataFrame(
        [
            {"TeamID": ABBREVIATION_TO_TEAM_ID["ATL"], "WINS": 25},
            {"TeamID": ABBREVIATION_TO_TEAM_ID["BOS"], "WINS": 45},
        ]
    ).to_csv(processed / "prior_season_standings.csv", index=False)

    result = compute_awards_summary(config)

    assert result["scope"] == "league"
    assert result["coy"] is not None
    # ATL mejoró +30 (55-25) frente a BOS que empeoró -15 (30-45).
    assert result["coy"].iloc[0]["team_abbreviation"] == "ATL"

    # team_record (para comparar candidatos entre sí): ATL ganó 55 de 82
    # -> "55-27". Se propaga por player_id, no por posición en la tabla.
    star_a = result["mvp"][result["mvp"]["player_id"] == 1].iloc[0]
    assert star_a["team_record"] == "55-27"


def test_compute_awards_summary_own_scope_team_record_comes_from_simulation_results(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [_projection_row(1, "Player One", game_score_per36=22.0, mpg=36.0)]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame(
        [_career_row(1, "2024-25", 70, 68, 2400, 1400, "Player One")]
    ).to_csv(processed / "roster_career_stats.csv", index=False)
    pd.DataFrame([{"wins": 50}, {"wins": 52}, {"wins": 48}]).to_csv(
        processed / "simulation_results.csv", index=False
    )

    result = compute_awards_summary(config)

    # Media de wins = 50 -> "50-32" (un solo equipo, mismo récord para todo el roster).
    assert result["mvp"].iloc[0]["team_record"] == "50-32"


def test_compute_awards_summary_mip_is_enriched_with_comparison_stats_and_team_record(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{**_projection_row(1, "Big Leap", game_score_per36=22.0, mpg=36.0), "PPG": 25.0, "RPG": 6.0}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame(
        [
            _career_row(1, "2023-24", 70, 68, 1800, 700, "Big Leap"),
            _career_row(1, "2024-25", 75, 70, 2200, 1400, "Big Leap"),
        ]
    ).to_csv(processed / "roster_career_stats.csv", index=False)
    pd.DataFrame([{"wins": 45}]).to_csv(processed / "simulation_results.csv", index=False)

    result = compute_awards_summary(config)

    assert not result["mip"].empty
    row = result["mip"].iloc[0]
    assert row["PPG"] == pytest.approx(25.0)
    assert row["RPG"] == pytest.approx(6.0)
    assert row["team_record"] == "45-37"


def _regular_row(team, conference, wins_mean):
    return {"team_id": 1, "team_abbreviation": team, "conference": conference, "wins_mean": wins_mean, "wins_p10": wins_mean - 8, "wins_p90": wins_mean + 8}


def _playoff_row(team, playoff_pct=50.0, championship_pct=5.0):
    return {
        "team_id": 1, "team_abbreviation": team, "playoff_pct": playoff_pct,
        "conf_semis_pct": playoff_pct / 2, "conf_finals_pct": playoff_pct / 4,
        "finals_pct": playoff_pct / 8, "championship_pct": championship_pct,
    }


def test_compute_conference_standings_splits_and_ranks_by_wins_mean():
    regular = pd.DataFrame(
        [
            _regular_row("BOS", "East", 55.0),
            _regular_row("NYK", "East", 40.0),
            _regular_row("MIA", "East", 48.0),
            _regular_row("LAL", "West", 50.0),
            _regular_row("DEN", "West", 60.0),
        ]
    )
    playoff = pd.DataFrame(
        [_playoff_row(t) for t in ["BOS", "NYK", "MIA", "LAL", "DEN"]]
    )

    standings = compute_conference_standings(regular, playoff)

    assert set(standings.keys()) == {"East", "West"}
    east = standings["East"]
    assert east["team_abbreviation"].tolist() == ["BOS", "MIA", "NYK"]  # ordenado por wins_mean desc
    assert east["seed"].tolist() == [1, 2, 3]
    west = standings["West"]
    assert west["team_abbreviation"].tolist() == ["DEN", "LAL"]
    assert west["seed"].tolist() == [1, 2]


def test_compute_conference_standings_marks_playoff_situation_by_seed():
    # 15 equipos en una sola conferencia para probar los 3 tramos de seed.
    regular = pd.DataFrame(
        [_regular_row(f"T{i}", "East", 82 - i) for i in range(15)]
    )
    playoff = pd.DataFrame([_playoff_row(f"T{i}") for i in range(15)])

    standings = compute_conference_standings(regular, playoff)
    east = standings["East"]

    assert east.iloc[0]["situacion"] == "Clasifica directo"  # seed 1
    assert east.iloc[5]["situacion"] == "Clasifica directo"  # seed 6
    assert east.iloc[6]["situacion"] == "Play-in"  # seed 7
    assert east.iloc[9]["situacion"] == "Play-in"  # seed 10
    assert east.iloc[10]["situacion"] == "Fuera"  # seed 11
    assert east.iloc[14]["situacion"] == "Fuera"  # seed 15


def test_compute_conference_standings_merges_playoff_probabilities():
    regular = pd.DataFrame([_regular_row("BOS", "East", 55.0)])
    playoff = pd.DataFrame([_playoff_row("BOS", playoff_pct=91.2, championship_pct=12.3)])

    standings = compute_conference_standings(regular, playoff)

    row = standings["East"].iloc[0]
    assert row["playoff_pct"] == pytest.approx(91.2)
    assert row["championship_pct"] == pytest.approx(12.3)


def test_load_champion_title_paths_returns_none_when_missing(config):
    assert load_champion_title_paths(config) is None


def test_compute_champion_seed_distribution_counts_and_percentages():
    paths = pd.DataFrame(
        [
            {"season": "2022-23", "team_abbreviation": "DEN", "seed": 1},
            {"season": "2023-24", "team_abbreviation": "BOS", "seed": 1},
            {"season": "2021-22", "team_abbreviation": "GSW", "seed": 3},
        ]
    )

    dist = compute_champion_seed_distribution(paths)

    assert dist.set_index("seed")["n_champions"].to_dict() == {1: 2, 3: 1}
    assert dist.set_index("seed")["pct"].to_dict()[1] == pytest.approx(200 / 3)


def test_compute_champion_seed_distribution_handles_empty_and_none():
    assert compute_champion_seed_distribution(None).empty
    assert compute_champion_seed_distribution(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# run_single_season_player_log_simulation
# ---------------------------------------------------------------------------


def test_run_single_season_player_log_simulation_returns_none_without_required_csvs(config):
    assert run_single_season_player_log_simulation(config) is None


def test_run_single_season_player_log_simulation_returns_one_row_per_roster_player(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {"player_id": 1, "player_name": "Player One", "game_score_per36": 15.0},
            {"player_id": 2, "player_name": "Player Two", "game_score_per36": 12.0},
        ]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame(
        [{"player_id": 1, "risk_score": 0.3}, {"player_id": 2, "risk_score": 0.1}]
    ).to_csv(processed / "injury_risk.csv", index=False)

    log = run_single_season_player_log_simulation(config, random_seed=1)

    assert log is not None
    assert set(log["player_name"]) == {"Player One", "Player Two"}
    assert (log["games_played"] + log["games_missed"] == 82).all()


def test_run_single_season_player_log_simulation_is_reproducible_with_same_seed(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Player One", "game_score_per36": 15.0}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "risk_score": 0.5}]).to_csv(processed / "injury_risk.csv", index=False)

    first = run_single_season_player_log_simulation(config, random_seed=123)
    second = run_single_season_player_log_simulation(config, random_seed=123)

    pd.testing.assert_frame_equal(
        first[["player_id", "games_played", "games_missed"]],
        second[["player_id", "games_played", "games_missed"]],
    )


def test_run_single_season_player_log_simulation_varies_without_an_explicit_seed(config):
    """
    Sin seed explícito debe sortear una temporada NUEVA cada vez (para el
    botón del dashboard) -- distinto del seed de
    config["simulation"]["random_seed"], que reproduce siempre
    simulation_results.csv.
    """
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"player_id": 1, "player_name": "Player One", "game_score_per36": 15.0}]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "risk_score": 0.5}]).to_csv(processed / "injury_risk.csv", index=False)

    results = [
        tuple(run_single_season_player_log_simulation(config)["games_played"]) for _ in range(20)
    ]
    assert len(set(results)) > 1


# ---------------------------------------------------------------------------
# GP / MPG simulados en select_roster_view
# ---------------------------------------------------------------------------


def test_select_roster_view_replaces_gp_mpg_with_simulated_values_when_games_per_season_given():
    overview = pd.DataFrame(
        [{
            "player_name": "Player One", "game_score_per36": 18.0, "risk_score": 0.5,
            "minutes_projection": 30.0,
            "games_played_last_season": 70, "minutes_per_game_last_season": 28.0,
            "PTS_projected": 1600, "PPG": 19.5,
        }]
    )
    view = select_roster_view(overview, mode="per_game", games_per_season=82)

    # GP simulado = 82 * (1 - 0.5) = 41, NO el histórico real (70).
    assert view.iloc[0]["GP"] == 41
    # MPG = minutes_projection tal cual (30.0), NO el histórico real (28.0)
    # y SIN descontar por risk_score (representa el ritmo cuando juega, no
    # la carga de temporada -- ver docstring de
    # _apply_simulated_games_and_minutes).
    assert view.iloc[0]["MPG"] == pytest.approx(30.0)


def test_select_roster_view_keeps_real_gp_mpg_without_games_per_season():
    """Sin games_per_season (llamante que no lo pasa) se conserva el
    comportamiento histórico -- no debe romper a nadie que no migre."""
    overview = pd.DataFrame(
        [{
            "player_name": "Player One", "game_score_per36": 18.0, "risk_score": 0.5,
            "minutes_projection": 30.0,
            "games_played_last_season": 70, "minutes_per_game_last_season": 28.0,
            "PTS_projected": 1600, "PPG": 19.5,
        }]
    )
    view = select_roster_view(overview, mode="per_game")
    assert view.iloc[0]["GP"] == 70
    assert view.iloc[0]["MPG"] == pytest.approx(28.0)


def test_select_roster_view_falls_back_to_real_values_when_risk_score_missing():
    """Si risk_score no está (injury_risk.csv no corrido todavía), se
    conserva el histórico en vez de dejarlo en blanco."""
    overview = pd.DataFrame(
        [{
            "player_name": "Player One", "game_score_per36": 18.0,
            "minutes_projection": 30.0,
            "games_played_last_season": 70, "minutes_per_game_last_season": 28.0,
            "PTS_projected": 1600, "PPG": 19.5,
        }]
    )
    view = select_roster_view(overview, mode="per_game", games_per_season=82)
    assert view.iloc[0]["GP"] == 70
    assert view.iloc[0]["MPG"] == pytest.approx(28.0)


def test_select_roster_view_falls_back_row_by_row_when_some_players_lack_risk_score():
    overview = pd.DataFrame([
        {"player_name": "A", "risk_score": 0.5, "minutes_projection": 30.0,
         "games_played_last_season": 70, "minutes_per_game_last_season": 28.0},
        {"player_name": "B", "risk_score": None, "minutes_projection": 20.0,
         "games_played_last_season": 60, "minutes_per_game_last_season": 15.0},
    ])
    view = select_roster_view(overview, mode="per_game", games_per_season=82)

    row_a = view[view["player_name"] == "A"].iloc[0]
    row_b = view[view["player_name"] == "B"].iloc[0]
    assert row_a["GP"] == 41  # simulado
    assert row_b["GP"] == 60  # sin risk_score -> histórico conservado


# ---------------------------------------------------------------------------
# Totales de temporada (PTS/REB/AST...) escalados por disponibilidad --
# regresión: cambiar de escenario "con" / "sin lesiones" solo movía
# GP/MPG, los totales y el PPG derivado se quedaban igual en ambos
# escenarios.
# ---------------------------------------------------------------------------


def test_select_roster_view_scales_season_totals_by_availability_in_totals_mode():
    overview = pd.DataFrame(
        [{
            "player_name": "Player One", "game_score_per36": 18.0, "risk_score": 0.5,
            "minutes_projection": 30.0,
            "games_played_last_season": 70, "minutes_per_game_last_season": 28.0,
            "PTS_projected": 1600, "REB_projected": 400, "PPG": 19.5,
        }]
    )
    view = select_roster_view(overview, mode="totals", games_per_season=82)

    # PTS/REB totales escalados por (1 - 0.5): un jugador de alto riesgo
    # ya no muestra los mismos puntos totales que si estuviera sano toda
    # la temporada.
    assert view.iloc[0]["PTS"] == 800
    assert view.iloc[0]["REB"] == 200


def test_select_roster_view_keeps_per_game_rate_constant_while_totals_scale():
    """PPG/RPG (ritmo cuando el jugador SÍ juega) se dejan sin escalar a
    propósito -- igual que un PPG real de la NBA no baja porque un
    jugador se pierda partidos. La relación Total = PPG * GP se mantiene
    exacta con el escalado de _apply_simulated_games_and_minutes."""
    overview = pd.DataFrame(
        [{
            "player_name": "Player One", "game_score_per36": 18.0, "risk_score": 0.5,
            "minutes_projection": 30.0,
            "games_played_last_season": 70, "minutes_per_game_last_season": 28.0,
            "PTS_projected": 1600, "PPG": 19.5122,
        }]
    )
    view = select_roster_view(overview, mode="per_game", games_per_season=82)

    assert view.iloc[0]["PPG"] == pytest.approx(19.5, abs=0.05)
    assert view.iloc[0]["GP"] == 41
    # Total implícito (PPG * GP) coincide con el total ya visto en modo
    # "totals" para el mismo jugador (800, ver test anterior).
    assert view.iloc[0]["PPG"] * view.iloc[0]["GP"] == pytest.approx(800, abs=5)


def test_select_roster_view_keeps_totals_real_without_games_per_season():
    overview = pd.DataFrame(
        [{
            "player_name": "Player One", "game_score_per36": 18.0, "risk_score": 0.5,
            "minutes_projection": 30.0,
            "games_played_last_season": 70, "minutes_per_game_last_season": 28.0,
            "PTS_projected": 1600, "PPG": 19.5,
        }]
    )
    view = select_roster_view(overview, mode="totals")
    assert view.iloc[0]["PTS"] == 1600


def test_select_roster_view_keeps_totals_real_when_risk_score_missing():
    overview = pd.DataFrame(
        [{
            "player_name": "Player One", "game_score_per36": 18.0,
            "minutes_projection": 30.0,
            "games_played_last_season": 70, "minutes_per_game_last_season": 28.0,
            "PTS_projected": 1600, "PPG": 19.5,
        }]
    )
    view = select_roster_view(overview, mode="totals", games_per_season=82)
    assert view.iloc[0]["PTS"] == 1600
