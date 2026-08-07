"""
Tests de src/champion_profiles.py. Usan DataFrames sintéticos con el
esquema de backtest_sweep_advanced_game_logs.csv / _standings.csv /
_rosters.csv / _player_career_stats.csv -- no requieren red ni el sweep
real corrido.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from champion_profiles import (  # noqa: E402
    build_champion_analysis_dataset,
    compute_champion_profiles,
    compute_roster_profile,
    compute_seed_trajectories,
    compute_title_paths,
    derive_champions,
)


def _playoff_log(season, team_id, abbrev, date, wl, matchup):
    return {
        "season": season, "TEAM_ID": team_id, "TEAM_ABBREVIATION": abbrev,
        "GAME_DATE": date, "WL": wl, "MATCHUP": matchup, "game_phase": "playoffs",
    }


def test_derive_champions_picks_the_winner_of_the_last_playoff_game():
    logs = pd.DataFrame(
        [
            _playoff_log("2023-24", 1, "BOS", "2024-06-01", "W", "BOS vs. DAL"),
            _playoff_log("2023-24", 2, "DAL", "2024-06-01", "L", "DAL @ BOS"),
            # Partido posterior: este decide el titulo.
            _playoff_log("2023-24", 1, "BOS", "2024-06-17", "W", "BOS vs. DAL"),
            _playoff_log("2023-24", 2, "DAL", "2024-06-17", "L", "DAL @ BOS"),
        ]
    )

    champs = derive_champions(logs)

    assert len(champs) == 1
    assert champs.iloc[0]["team_abbreviation"] == "BOS"
    assert champs.iloc[0]["season"] == "2023-24"


def test_derive_champions_returns_one_row_per_season():
    logs = pd.DataFrame(
        [
            _playoff_log("2022-23", 3, "DEN", "2023-06-12", "W", "DEN vs. MIA"),
            _playoff_log("2022-23", 4, "MIA", "2023-06-12", "L", "MIA @ DEN"),
            _playoff_log("2023-24", 1, "BOS", "2024-06-17", "W", "BOS vs. DAL"),
            _playoff_log("2023-24", 2, "DAL", "2024-06-17", "L", "DAL @ BOS"),
        ]
    )

    champs = derive_champions(logs)

    assert champs["season"].tolist() == ["2022-23", "2023-24"]
    assert champs["team_abbreviation"].tolist() == ["DEN", "BOS"]


def test_derive_champions_on_empty_input_returns_empty_frame():
    assert derive_champions(pd.DataFrame()).empty


def test_compute_title_paths_reports_seed_and_opponents_in_order():
    champions = pd.DataFrame(
        [{"season": "2023-24", "team_id": 1, "team_abbreviation": "BOS", "clinch_date": pd.Timestamp("2024-06-17")}]
    )
    standings = pd.DataFrame(
        [
            {"season": "2023-24", "TeamID": 1, "PlayoffRank": 1, "WINS": 64, "TeamName": "Celtics"},
            {"season": "2023-24", "TeamID": 2, "PlayoffRank": 5, "WINS": 50, "TeamName": "Mavericks"},
            {"season": "2023-24", "TeamID": 3, "PlayoffRank": 8, "WINS": 46, "TeamName": "Heat"},
        ]
    )
    logs = pd.DataFrame(
        [
            _playoff_log("2023-24", 1, "BOS", "2024-04-21", "W", "BOS vs. MIA"),
            _playoff_log("2023-24", 1, "BOS", "2024-04-24", "W", "BOS vs. MIA"),
            _playoff_log("2023-24", 1, "BOS", "2024-06-17", "W", "BOS vs. DAL"),
            _playoff_log("2023-24", 3, "MIA", "2024-04-21", "L", "MIA @ BOS"),
            _playoff_log("2023-24", 2, "DAL", "2024-06-17", "L", "DAL @ BOS"),
        ]
    )

    paths = compute_title_paths(champions, standings, logs)

    row = paths.iloc[0]
    assert row["seed"] == 1
    assert row["regular_season_wins"] == 64
    assert row["playoff_wins"] == 3
    assert row["opponents_faced"] == "MIA → DAL"  # cronologico, una entrada por serie
    assert row["seeds_beaten"] == "8 → 5"


def _roster_row(player_id, name, position, exp, age, team_id=1, season="2023-24"):
    return {
        "PLAYER_ID": player_id, "PLAYER": name, "POSITION": position,
        "EXP": exp, "AGE": age, "TeamID": team_id, "season": season,
    }


def _career_row(player_id, season, minutes):
    return {"PLAYER_ID": player_id, "SEASON_ID": season, "MIN": minutes}


def test_compute_roster_profile_measures_star_concentration():
    roster = pd.DataFrame(
        [
            _roster_row(1, "Star A", "G", 8, 28),
            _roster_row(2, "Star B", "F", 6, 27),
            _roster_row(3, "Role", "C", 3, 25),
            _roster_row(4, "Bench", "G", 1, 23),
        ]
    )
    career = pd.DataFrame(
        [
            _career_row(1, "2023-24", 3000), _career_row(2, "2023-24", 3000),
            _career_row(3, "2023-24", 2000), _career_row(4, "2023-24", 2000),
        ]
    )

    profile = compute_roster_profile(roster, career, "2023-24", star_count=2)

    # 6000 de 10000 minutos en los 2 mas usados.
    assert profile["star_minutes_share"] == pytest.approx(60.0)
    assert profile["players_with_minutes"] == 4


def test_compute_roster_profile_splits_minutes_by_position_group():
    roster = pd.DataFrame(
        [_roster_row(1, "Guard", "G", 5, 26), _roster_row(2, "Center", "C-F", 5, 26)]
    )
    career = pd.DataFrame([_career_row(1, "2023-24", 3000), _career_row(2, "2023-24", 1000)])

    profile = compute_roster_profile(roster, career, "2023-24")

    assert profile["minutes_pct_Base/Escolta"] == pytest.approx(75.0)
    assert profile["minutes_pct_Pívot"] == pytest.approx(25.0)  # "C-F" cuenta por su primera letra
    assert profile["minutes_pct_Alero/Ala-pívot"] == pytest.approx(0.0)


def test_compute_roster_profile_treats_rookie_experience_as_zero():
    roster = pd.DataFrame([_roster_row(1, "Rookie", "G", "R", 20)])
    career = pd.DataFrame([_career_row(1, "2023-24", 2000)])

    profile = compute_roster_profile(roster, career, "2023-24")

    assert profile["weighted_experience"] == pytest.approx(0.0)


def test_compute_roster_profile_returns_empty_when_no_minutes():
    roster = pd.DataFrame([_roster_row(1, "Ghost", "G", 2, 24)])
    career = pd.DataFrame([_career_row(1, "2022-23", 2000)])  # otra temporada

    assert compute_roster_profile(roster, career, "2023-24") == {}


def test_compute_champion_profiles_one_row_per_champion():
    champions = pd.DataFrame(
        [{"season": "2023-24", "team_id": 1, "team_abbreviation": "BOS", "clinch_date": pd.Timestamp("2024-06-17")}]
    )
    rosters = pd.DataFrame([_roster_row(1, "Star", "G", 5, 27), _roster_row(2, "Role", "F", 3, 25)])
    career = pd.DataFrame([_career_row(1, "2023-24", 3000), _career_row(2, "2023-24", 1000)])

    profiles = compute_champion_profiles(champions, rosters, career, star_count=1)

    assert len(profiles) == 1
    assert profiles.iloc[0]["team_abbreviation"] == "BOS"
    assert profiles.iloc[0]["star_minutes_share"] == pytest.approx(75.0)


def test_compute_seed_trajectories_pivots_franchise_by_season():
    standings = pd.DataFrame(
        [
            {"season": "2022-23", "TeamName": "Celtics", "PlayoffRank": 2, "TeamID": 1, "WINS": 57},
            {"season": "2023-24", "TeamName": "Celtics", "PlayoffRank": 1, "TeamID": 1, "WINS": 64},
            {"season": "2022-23", "TeamName": "Heat", "PlayoffRank": 8, "TeamID": 3, "WINS": 44},
        ]
    )

    traj = compute_seed_trajectories(standings)

    assert traj.loc["Celtics", "2023-24"] == 1
    assert traj.loc["Celtics", "2022-23"] == 2
    assert traj.loc["Heat", "2022-23"] == 8


def test_build_champion_analysis_dataset_returns_none_without_sweep_data(tmp_path):
    config = {"paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(tmp_path / "processed")}}
    assert build_champion_analysis_dataset(config) is None
