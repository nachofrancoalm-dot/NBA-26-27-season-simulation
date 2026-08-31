import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from league_simulation import TEAM_CONFERENCE  # noqa: E402
from league_sandbox import compute_hypothetical_awards, simulate_hypothetical_league  # noqa: E402

# Los 29 equipos reales que NO son mi equipo hipotético (PHI) -- se
# necesitan abreviaturas REALES porque simulate_hypothetical_league
# resuelve la conferencia de cada equipo vía TEAM_CONFERENCE.
OTHER_TEAMS = [abbrev for abbrev in TEAM_CONFERENCE if abbrev != "PHI"]


def _player_row(player_id, name, team_abbrev, game_score_per36, mpg, team_id):
    return {
        "player_id": player_id,
        "player_name": name,
        "team_id": team_id,
        "team_abbreviation": team_abbrev,
        "conference": TEAM_CONFERENCE[team_abbrev],
        "position": "G",
        "current_age": 25,
        "country": "USA",
        "game_score_per36": game_score_per36,
        "minutes_projection": mpg,
        "minutes_per_game_last_season": mpg,
        "risk_score": 0.2,
        "fatigue_score": 0.2,
        "games_played_last_season": 70,
        "projected_total_minutes": mpg * 82,
        "PTS_per36_projected": game_score_per36,
        "AST_per36_projected": 4.0,
        "REB_per36_projected": 5.0,
        "STL_per36_projected": 1.0,
        "BLK_per36_projected": 0.5,
        "DREB_per36_projected": 4.0,
        "TOV_per36_projected": 2.0,
        "PF_per36_projected": 2.0,
        "FG3M_per36_projected": 1.5,
        "PTS_projected": game_score_per36 * mpg / 36.0 * 82,
        "PPG": game_score_per36 * mpg / 36.0,
        "FG%": 45.0,
        "3P%": 35.0,
    }


@pytest.fixture
def league_config(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)

    rows = []
    next_id = 1
    hypothetical_player_ids = []
    for team_index, abbrev in enumerate(OTHER_TEAMS):
        team_id = 1000 + team_index
        for slot in range(2):
            player_id = next_id
            next_id += 1
            row = _player_row(player_id, f"Player {player_id}", abbrev, 12.0 + slot, 28.0 - slot * 4, team_id)
            rows.append(row)
            # Los primeros 5 jugadores creados se "prestan" al roster
            # hipotético (siguen existiendo también en su equipo real en
            # el CSV -- la decisión de diseño confirmada es no tocar los
            # otros 29 equipos).
            if len(hypothetical_player_ids) < 5:
                hypothetical_player_ids.append(player_id)

    pd.DataFrame(rows).to_csv(processed / "league_player_projections.csv", index=False)
    pd.DataFrame(columns=["player_id", "SEASON_ID", "GP", "MIN"]).to_csv(
        processed / "league_player_career_stats.csv", index=False
    )

    config = {
        "team": {"team_id": 9999, "abbreviation": "PHI", "season": "2026-27"},
        "simulation": {"games_per_season": 82, "n_seasons": 20, "random_seed": 1},
        "league_simulation": {"n_seasons": 20, "n_playoff_seasons": 20},
        "monte_carlo": {},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed)},
    }
    return config, hypothetical_player_ids


def test_simulate_hypothetical_league_returns_all_30_teams(league_config):
    config, player_ids = league_config
    result = simulate_hypothetical_league(config, player_ids, n_seasons=20, n_playoff_seasons=20)

    assert len(result["regular_season_df"]) == 30
    assert len(result["playoff_df"]) == 30
    assert config["team"]["team_id"] in result["regular_season_df"]["team_id"].to_numpy()


def test_simulate_hypothetical_league_keeps_other_29_teams_untouched(league_config):
    config, player_ids = league_config
    pool_before = pd.read_csv(Path(config["paths"]["processed_data_dir"]) / "league_player_projections.csv")

    result = simulate_hypothetical_league(config, player_ids, n_seasons=20, n_playoff_seasons=20)

    other_rows_after = result["player_projections_df"][result["player_projections_df"]["team_id"] != config["team"]["team_id"]]
    # Mismo número de jugadores en los otros 29 equipos que en el pool
    # original -- confirma que no se quitó a nadie de su equipo real por
    # haberlo "prestado" al roster hipotético.
    assert len(other_rows_after) == len(pool_before)


def test_simulate_hypothetical_league_recalculates_borrowed_player_stats(league_config):
    config, player_ids = league_config
    result = simulate_hypothetical_league(config, player_ids, n_seasons=20, n_playoff_seasons=20)

    my_rows = result["player_projections_df"][result["player_projections_df"]["team_id"] == config["team"]["team_id"]]
    assert len(my_rows) == len(player_ids)
    assert (my_rows["team_abbreviation"] == "PHI").all()
    # game_score_per36 es intrínseco al jugador -- no debe cambiar por
    # jugar en un equipo hipotético.
    original = pd.read_csv(Path(config["paths"]["processed_data_dir"]) / "league_player_projections.csv").set_index(
        "player_id"
    )
    for pid in player_ids:
        row = my_rows[my_rows["player_id"] == pid].iloc[0]
        assert row["game_score_per36"] == pytest.approx(original.loc[pid, "game_score_per36"])


def test_simulate_hypothetical_league_playoff_rounds_are_monotonic(league_config):
    config, player_ids = league_config
    result = simulate_hypothetical_league(config, player_ids, n_seasons=200, n_playoff_seasons=200)

    playoff = result["playoff_df"]
    # playoffs >= semis >= finales de conf. >= Finales >= título, para
    # cada equipo -- mismo chequeo de sanidad que ya tiene league_simulation.
    assert (playoff["playoff_pct"] >= playoff["conf_semis_pct"] - 1e-9).all()
    assert (playoff["conf_semis_pct"] >= playoff["conf_finals_pct"] - 1e-9).all()
    assert (playoff["conf_finals_pct"] >= playoff["finals_pct"] - 1e-9).all()
    assert (playoff["finals_pct"] >= playoff["championship_pct"] - 1e-9).all()


def test_compute_hypothetical_awards_returns_expected_keys(league_config):
    config, player_ids = league_config
    result = simulate_hypothetical_league(config, player_ids, n_seasons=20, n_playoff_seasons=20)

    awards = compute_hypothetical_awards(config, result)

    assert awards["scope"] == "league"
    for key in ("mvp", "dpoy", "sixth_man", "roy", "mip", "all_star", "all_nba", "all_defensive"):
        assert key in awards


def test_compute_hypothetical_awards_mvp_can_include_hypothetical_team_player(league_config):
    """Un jugador con game_score_per36 muy alto en el roster hipotético
    debe poder aparecer como candidato a MVP -- confirma que
    player_projections_df realmente se usa para los premios, no solo el
    de los otros 29 equipos."""
    config, player_ids = league_config
    result = simulate_hypothetical_league(config, player_ids, n_seasons=20, n_playoff_seasons=20)
    awards = compute_hypothetical_awards(config, result, top_n=50)

    mvp_team_abbrevs = set(awards["mvp"]["team_abbreviation"])
    assert "PHI" in mvp_team_abbrevs
