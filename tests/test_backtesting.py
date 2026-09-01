"""
Tests de backtesting.py. Usan DataFrames sintéticos con el esquema de
historical_comparables_rosters.csv / _player_career_stats.csv /
_standings.csv / _advanced_game_logs.csv -- no requieren red.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from backtesting import (  # noqa: E402
    _run_backtest_cases,
    expected_team_game_score_equivalent,
    build_real_schedule_context,
    compute_calibration_summary,
    load_league_baselines,
    filter_seasons_before,
    get_actual_season_row,
    project_backtest_team,
    project_historical_player,
    run_backtest_case,
)


def _career_seasons(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_ID": r.get("player_id", 1),
                "SEASON_ID": r["season"],
                "PLAYER_AGE": r["age"],
                "GP": r.get("gp", 80),
                "MIN": r.get("min", 2400),
                "PTS": r.get("pts", 1200),
                "AST": r.get("ast", 300),
                "REB": r.get("reb", 400),
                "STL": r.get("stl", 80),
                "BLK": r.get("blk", 40),
                "TOV": r.get("tov", 150),
                "FG3M": r.get("fg3m", 100),
                "FG3A": r.get("fg3a", 260),
                "OREB": r.get("oreb", 80),
                "DREB": r.get("dreb", 320),
                "FGM": r.get("fgm", 420),
                "FGA": r.get("fga", 880),
                "FTM": r.get("ftm", 200),
                "FTA": r.get("fta", 260),
                "PF": r.get("pf", 180),
                "TEAM_ABBREVIATION": r.get("team", "TOT"),
            }
            for r in rows
        ]
    )


def test_filter_seasons_before_excludes_target_and_future_seasons():
    seasons = _career_seasons(
        [
            {"season": "2008-09", "age": 24},
            {"season": "2009-10", "age": 25},
            {"season": "2010-11", "age": 26},  # temporada del caso -- debe excluirse
            {"season": "2011-12", "age": 27},  # futura -- debe excluirse
        ]
    )
    prior = filter_seasons_before(seasons, target_season_start_year=2010)
    assert set(prior["SEASON_ID"]) == {"2008-09", "2009-10"}


def test_get_actual_season_row_dedupes_traded_season():
    seasons = _career_seasons(
        [
            {"season": "2010-11", "age": 26, "gp": 50, "min": 1500, "team": "TOT"},
            {"season": "2010-11", "age": 26, "gp": 30, "min": 900, "team": "MIA"},
            {"season": "2010-11", "age": 26, "gp": 20, "min": 600, "team": "TOR"},
        ]
    )
    row = get_actual_season_row(seasons, "2010-11")
    assert row["GP"] == 50
    assert row["MIN"] == 1500


def test_get_actual_season_row_returns_none_when_missing():
    seasons = _career_seasons([{"season": "2009-10", "age": 25}])
    assert get_actual_season_row(seasons, "2010-11") is None


def test_project_historical_player_with_no_prior_history_returns_zeroed_baseline():
    # slice vacío de un DataFrame CON columnas -- _career_seasons([]) solo no las tiene
    empty_regular = _career_seasons([{"season": "2010-11", "age": 20}]).iloc[0:0]
    result = project_historical_player(
        empty_regular, pd.DataFrame(), "2010-11", actual_age=19, actual_minutes_per_game=15.0, games_in_season=82
    )
    assert result["game_score_per36"] == 0.0
    assert result["risk_score"] == 0.0
    assert result["fatigue_score"] == 0.0


def test_project_historical_player_only_uses_prior_seasons():
    # temporada del caso muy fuerte -- si hubiera look-ahead, el game_score sería absurdamente alto
    seasons = _career_seasons(
        [
            {"season": "2009-10", "age": 25, "pts": 800, "min": 2000},
            {"season": "2010-11", "age": 26, "pts": 3000, "min": 3000},  # no debe usarse
        ]
    )
    result = project_historical_player(
        seasons, pd.DataFrame(), "2010-11", actual_age=26, actual_minutes_per_game=36.0, games_in_season=82
    )
    assert result["game_score_per36"] < 30


def test_project_backtest_team_reads_aging_curve_config_instead_of_ignoring_it():
    # regression: project_backtest_team() no pasaba n_seasons/half_life a project_player_season()
    # y por tanto ignoraba config["aging_curve"] -- n_seasons_lookback=1 debe proyectar distinto que =2
    case = {"name": "Test Team 2026-27", "team_id": 1610612748, "season": "2026-27"}
    rosters = pd.DataFrame([{"PLAYER_ID": 1, "PLAYER": "Breakout Player", "AGE": 27,
                             "comparable_name": case["name"], "season": case["season"]}])
    player_stats = _career_seasons([
        {"player_id": 1, "season": "2024-25", "age": 25, "pts": 800, "min": 2400},   # vieja, floja
        {"player_id": 1, "season": "2025-26", "age": 26, "pts": 2000, "min": 2400},  # reciente, fuerte
    ])
    config = {
        "lineup_synergy": {}, "league_simulation": {},
        "aging_curve": {"n_seasons_lookback": 1},
    }
    only_recent = project_backtest_team(case, rosters, player_stats, pd.DataFrame(), config, games_in_season=82)

    config_diluted = {
        "lineup_synergy": {}, "league_simulation": {},
        "aging_curve": {"n_seasons_lookback": 2, "recency_half_life_seasons": 100.0},
    }
    diluted_by_old_season = project_backtest_team(
        case, rosters, player_stats, pd.DataFrame(), config_diluted, games_in_season=82
    )

    assert only_recent["game_score_per36"][0] > diluted_by_old_season["game_score_per36"][0]


def test_build_real_schedule_context_resolves_opponent_and_back_to_back():
    game_log = pd.DataFrame(
        [
            {"GAME_DATE": "2010-10-26", "MATCHUP": "MIA vs. BOS", "WL": "W"},
            {"GAME_DATE": "2010-10-27", "MATCHUP": "MIA @ ORL", "WL": "L"},  # back-to-back
            {"GAME_DATE": "2010-10-30", "MATCHUP": "MIA vs. TOR", "WL": "W"},
        ]
    )
    standings = pd.DataFrame(
        [
            {"TeamID": 1610612738, "WinPCT": 0.70},  # BOS
            {"TeamID": 1610612753, "WinPCT": 0.30},  # ORL
            {"TeamID": 1610612761, "WinPCT": 0.40},  # TOR
        ]
    )
    opponent_win_pct, is_back_to_back = build_real_schedule_context(game_log, standings)

    assert list(is_back_to_back) == [False, True, False]
    assert opponent_win_pct[0] == pytest.approx(0.70)
    assert opponent_win_pct[1] == pytest.approx(0.30)


def test_run_backtest_case_end_to_end_with_synthetic_data():
    case = {"name": "Test Team 2010-11", "team_id": 1610612748, "season": "2010-11"}

    rosters = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER": "Star Player", "AGE": 26, "comparable_name": case["name"], "season": "2010-11"},
            {"PLAYER_ID": 2, "PLAYER": "Role Player", "AGE": 24, "comparable_name": case["name"], "season": "2010-11"},
        ]
    )
    player_stats = pd.concat(
        [
            _career_seasons(
                [
                    {"player_id": 1, "season": "2008-09", "age": 24, "pts": 1800, "min": 2800},
                    {"player_id": 1, "season": "2009-10", "age": 25, "pts": 1900, "min": 2900},
                    {"player_id": 1, "season": "2010-11", "age": 26, "pts": 2000, "min": 3000, "gp": 80},
                ]
            ),
            _career_seasons(
                [
                    {"player_id": 2, "season": "2009-10", "age": 23, "pts": 500, "min": 1200},
                    {"player_id": 2, "season": "2010-11", "age": 24, "pts": 550, "min": 1300, "gp": 78},
                ]
            ),
        ],
        ignore_index=True,
    )
    standings = pd.DataFrame(
        [
            {"TeamID": 1610612738, "WinPCT": 0.60, "season": "2010-11"},
            {"TeamID": 1610612753, "WinPCT": 0.35, "season": "2010-11"},
            {"TeamID": 1610612748, "WinPCT": 0.70, "season": "2010-11"},
        ]
    )
    game_log = pd.DataFrame(
        [
            {
                "GAME_DATE": f"2010-11-{i+1:02d}",
                "MATCHUP": "MIA vs. BOS" if i % 2 == 0 else "MIA @ ORL",
                "WL": "W" if i % 3 != 0 else "L",
                "comparable_name": case["name"],
                "game_phase": "regular",
            }
            for i in range(10)
        ]
    )

    config = {
        "simulation": {"n_seasons": 20, "games_per_season": 82, "random_seed": 1},
        "lineup_synergy": {},
        "monte_carlo": {},
    }

    result = run_backtest_case(case, rosters, player_stats, pd.DataFrame(), standings, game_log, config)

    assert result["games_in_season"] == 10
    assert result["actual_wins"] == 6
    assert 0.0 <= result["actual_percentile"] <= 100.0


def test_run_backtest_case_era_baseline_lowers_wins_for_an_inflated_era():
    # con una línea base de liga más alta, el mismo equipo debe proyectar menos victorias (corrige inflación de era)
    case = {"name": "Test Team 2010-11", "team_id": 1610612748, "season": "2010-11"}
    rosters = pd.DataFrame(
        [{"PLAYER_ID": 1, "PLAYER": "Star Player", "AGE": 26, "comparable_name": case["name"], "season": "2010-11"}]
    )
    player_stats = _career_seasons(
        [
            {"player_id": 1, "season": "2009-10", "age": 25, "pts": 1900, "min": 2900},
            {"player_id": 1, "season": "2010-11", "age": 26, "pts": 2000, "min": 3000, "gp": 80},
        ]
    )
    standings = pd.DataFrame([{"TeamID": 1610612738, "WinPCT": 0.50, "season": "2010-11"}])
    game_log = pd.DataFrame(
        [
            {"GAME_DATE": f"2010-11-{i+1:02d}", "MATCHUP": "MIA vs. BOS", "WL": "W",
             "comparable_name": case["name"], "game_phase": "regular"}
            for i in range(20)
        ]
    )
    config = {
        "simulation": {"n_seasons": 300, "games_per_season": 82, "random_seed": 1},
        "lineup_synergy": {},
        "monte_carlo": {},
    }

    args = (rosters, player_stats, pd.DataFrame(), standings, game_log, config)
    low_era = run_backtest_case(case, *args, league_baseline_per36=8.0)
    high_era = run_backtest_case(case, *args, league_baseline_per36=14.0)

    assert high_era["simulated_wins_mean"] < low_era["simulated_wins_mean"]


def test_run_backtest_cases_passes_the_per_season_baseline(monkeypatch):
    # cada caso debe recibir la línea base de su propia temporada, no una global
    import backtesting as backtesting_module

    seen = {}

    def _fake_run_backtest_case(case, *args, league_baseline_per36=None, **kwargs):
        seen[case["name"]] = league_baseline_per36
        return {"comparable_name": case["name"], "actual_wins": 41}

    monkeypatch.setattr(backtesting_module, "run_backtest_case", _fake_run_backtest_case)

    cases = [
        {"name": "A 2010-11", "team_id": 1, "season": "2010-11"},
        {"name": "B 2024-25", "team_id": 2, "season": "2024-25"},
        {"name": "C 1999-00", "team_id": 3, "season": "1999-00"},  # sin baseline
    ]
    _run_backtest_cases(
        cases, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {},
        league_baseline_by_season={"2010-11": 10.5, "2024-25": 13.4},
    )

    assert seen == {"A 2010-11": 10.5, "B 2024-25": 13.4, "C 1999-00": None}


def test_run_backtest_cases_skips_a_case_that_raises_and_keeps_the_rest(monkeypatch, capsys):
    import backtesting as backtesting_module

    good_case = {"name": "Good Team 2010-11", "team_id": 1, "season": "2010-11"}
    bad_case = {"name": "Broken Team 2011-12", "team_id": 2, "season": "2011-12"}

    def _fake_run_backtest_case(case, *args, **kwargs):
        if case["name"] == bad_case["name"]:
            raise ValueError("datos incompletos simulados para el test")
        return {"comparable_name": case["name"], "actual_wins": 41}

    monkeypatch.setattr(backtesting_module, "run_backtest_case", _fake_run_backtest_case)

    result_df = _run_backtest_cases(
        [good_case, bad_case], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    )

    assert len(result_df) == 1
    assert result_df.iloc[0]["comparable_name"] == good_case["name"]
    assert "omitido" in capsys.readouterr().out


def _backtest_row(actual_wins, mean, p10, p90, percentile):
    return {
        "actual_wins": actual_wins,
        "simulated_wins_mean": mean,
        "simulated_wins_p10": p10,
        "simulated_wins_p90": p90,
        "actual_percentile": percentile,
    }


def test_compute_calibration_summary_on_empty_dataframe_returns_nan_metrics():
    summary = compute_calibration_summary(pd.DataFrame())

    assert summary["n_cases"] == 0
    assert summary["pct_within_p10_p90"] != summary["pct_within_p10_p90"]  # NaN != NaN


def test_compute_calibration_summary_well_calibrated_case():
    df = pd.DataFrame(
        [
            _backtest_row(actual_wins=50, mean=50, p10=40, p90=60, percentile=50.0),
            _backtest_row(actual_wins=45, mean=48, p10=38, p90=58, percentile=40.0),
            _backtest_row(actual_wins=55, mean=52, p10=42, p90=62, percentile=60.0),
        ]
    )

    summary = compute_calibration_summary(df)

    assert summary["n_cases"] == 3
    assert summary["pct_within_p10_p90"] == 100.0
    assert summary["mean_percentile"] == pytest.approx(50.0)
    assert abs(summary["mean_error_wins"]) < 5  # sin sesgo grande


def test_compute_calibration_summary_detects_systematic_overestimation():
    # actual_wins muy por debajo de simulated_wins_p10 -- el modelo sobreestima sistemáticamente
    df = pd.DataFrame(
        [
            _backtest_row(actual_wins=45, mean=70, p10=64, p90=77, percentile=0.05),
            _backtest_row(actual_wins=48, mean=66, p10=61, p90=71, percentile=0.25),
        ]
    )

    summary = compute_calibration_summary(df)

    assert summary["pct_within_p10_p90"] == 0.0
    assert summary["mean_error_wins"] < 0
    assert summary["mean_percentile"] < 10


def test_compute_calibration_summary_correlation_reflects_predictive_power():
    # predicciones que ordenan a los equipos correctamente, aunque el nivel absoluto esté desplazado
    df = pd.DataFrame(
        [
            _backtest_row(actual_wins=20, mean=30, p10=20, p90=40, percentile=50.0),
            _backtest_row(actual_wins=40, mean=50, p10=40, p90=60, percentile=50.0),
            _backtest_row(actual_wins=60, mean=70, p10=60, p90=80, percentile=50.0),
        ]
    )

    summary = compute_calibration_summary(df)

    assert summary["correlation_actual_vs_predicted"] > 0.99


def test_load_league_baselines_returns_empty_when_file_missing(tmp_path):
    assert load_league_baselines(tmp_path) == {}


def test_load_league_baselines_prefers_the_projected_team_column(tmp_path):
    # la columna de equipos proyectados cumple suma cero; la de jugadores deja sesgo residual
    pd.DataFrame(
        [
            {"season": "2010-11", "league_game_score_per36": 10.7, "projected_team_baseline_per36": 11.0},
            {"season": "2024-25", "league_game_score_per36": 13.4, "projected_team_baseline_per36": 12.7},
        ]
    ).to_csv(tmp_path / "league_game_score_baseline.csv", index=False)

    baselines = load_league_baselines(tmp_path)

    assert baselines == {"2010-11": 11.0, "2024-25": 12.7}


def test_load_league_baselines_falls_back_to_player_column_when_projected_missing(tmp_path):
    pd.DataFrame(
        [{"season": "2010-11", "league_game_score_per36": 10.7}]
    ).to_csv(tmp_path / "league_game_score_baseline.csv", index=False)

    assert load_league_baselines(tmp_path) == {"2010-11": 10.7}


def test_load_league_baselines_skips_seasons_without_a_value(tmp_path):
    pd.DataFrame(
        [
            {"season": "2010-11", "league_game_score_per36": 10.7, "projected_team_baseline_per36": 11.0},
            {"season": "1999-00", "league_game_score_per36": 9.9, "projected_team_baseline_per36": None},
        ]
    ).to_csv(tmp_path / "league_game_score_baseline.csv", index=False)

    assert load_league_baselines(tmp_path) == {"2010-11": 11.0}


# ---------------------------------------------------------------------------
# expected_team_game_score_equivalent -- la restricción de suma cero
# ---------------------------------------------------------------------------


def _fake_projection(n_players=10, game_score_per36=14.0, risk=0.2, synergy_value=None):
    """Proyección sintética con el esquema de project_backtest_team."""
    synergy = None
    if synergy_value is not None:
        synergy = np.full((n_players, n_players), synergy_value)
        np.fill_diagonal(synergy, 0.0)
    return {
        "player_ids": list(range(n_players)),
        "game_score_per36": np.full(n_players, game_score_per36),
        "minutes_projection": np.full(n_players, 240.0 / n_players),
        "risk_scores": np.full(n_players, risk),
        "fatigue_scores": np.zeros(n_players),
        "synergy_matrix": synergy,
        "team_game_score": game_score_per36 * 240.0 / 36.0,
    }


def test_expected_team_game_score_is_below_full_health_value():
    # regression: la línea base usaba Game Score a plena salud, pero los equipos simulan con
    # ausencias por lesión -- comparar contra una referencia sana inflaba el net rating de la liga
    config = {"simulation": {"random_seed": 42}}
    projection = _fake_projection(risk=0.25, synergy_value=None)

    expected = expected_team_game_score_equivalent(projection, config, games_in_season=82)

    assert expected < projection["team_game_score"]
    assert 0.6 < expected / projection["team_game_score"] < 0.9


def test_expected_team_game_score_absorbs_the_synergy_offset():
    # compute_game_synergy_adjustment devuelve un valor siempre positivo -- si la línea base
    # no lo absorbe, la liga entera se desplaza hacia arriba
    config = {"simulation": {"random_seed": 42}}
    without = expected_team_game_score_equivalent(
        _fake_projection(synergy_value=None), config, games_in_season=82
    )
    with_synergy = expected_team_game_score_equivalent(
        _fake_projection(synergy_value=0.05), config, games_in_season=82
    )

    assert with_synergy > without


def test_expected_team_game_score_is_deterministic_for_a_given_seed():
    # la línea base no puede bailar entre corridas -- un baseline aleatorio movería los 480 casos a la vez
    config = {"simulation": {"random_seed": 7}}
    projection = _fake_projection(synergy_value=0.05)
    first = expected_team_game_score_equivalent(projection, config, 82)
    second = expected_team_game_score_equivalent(projection, config, 82)
    assert first == pytest.approx(second)
