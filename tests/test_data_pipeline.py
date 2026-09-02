"""
Tests de build_league_schedule_dataset() en src/data_pipeline.py (filtrado de
temporada regular / partidos sin resolver), mockeando fetch_league_schedule.
El resto del módulo son wrappers finos sobre nba_api sin lógica propia.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data_pipeline  # noqa: E402


@pytest.fixture
def config(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    return {
        "team": {"season": "2026-27"},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed_dir)},
    }


def _fake_schedule_row(game_date, home, away, game_label=""):
    return {"gameDate": game_date, "gameLabel": game_label, "homeTeam_teamTricode": home, "awayTeam_teamTricode": away}


def test_build_league_schedule_dataset_drops_preseason_games(config, monkeypatch):
    fake = pd.DataFrame([
        _fake_schedule_row("2026-10-01", "AAA", "BBB", game_label="Preseason"),
        _fake_schedule_row("2026-10-20", "AAA", "BBB", game_label=""),
    ])
    monkeypatch.setattr(data_pipeline, "fetch_league_schedule", lambda season, raw_dir, force_refresh: fake)

    result = data_pipeline.build_league_schedule_dataset(config)

    assert len(result) == 1
    assert result.iloc[0]["gameDate"] == "2026-10-20"


def test_build_league_schedule_dataset_drops_games_with_unresolved_teams(config, monkeypatch):
    # plazas de la NBA Cup sin resolver (tricode nulo) se descartan, no se inventa el rival
    fake = pd.DataFrame([
        _fake_schedule_row("2026-12-15", "AAA", None),
        _fake_schedule_row("2026-12-16", "AAA", "BBB"),
    ])
    monkeypatch.setattr(data_pipeline, "fetch_league_schedule", lambda season, raw_dir, force_refresh: fake)

    result = data_pipeline.build_league_schedule_dataset(config)

    assert len(result) == 1
    assert result.iloc[0]["gameDate"] == "2026-12-16"


def test_build_league_schedule_dataset_saves_only_the_needed_columns(config, monkeypatch):
    fake = pd.DataFrame([{
        **_fake_schedule_row("2026-10-20", "AAA", "BBB"),
        "arenaCity": "Some City", "gameId": "0012600001",
    }])
    monkeypatch.setattr(data_pipeline, "fetch_league_schedule", lambda season, raw_dir, force_refresh: fake)

    result = data_pipeline.build_league_schedule_dataset(config)

    assert list(result.columns) == ["gameDate", "homeTeam_teamTricode", "awayTeam_teamTricode"]

    out_path = Path(config["paths"]["processed_data_dir"]) / "league_schedule_full.csv"
    assert out_path.exists()
    assert len(pd.read_csv(out_path)) == 1


def test_build_roster_shot_charts_dataset_covers_up_to_n_seasons_per_player(config, monkeypatch):
    processed_dir = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "SEASON_ID": "2022-23", "GP": 55},
            {"PLAYER_ID": 1, "SEASON_ID": "2023-24", "GP": 60},
            {"PLAYER_ID": 1, "SEASON_ID": "2024-25", "GP": 70},
            {"PLAYER_ID": 1, "SEASON_ID": "2026-27", "GP": 0},  # proyección futura, sin partidos -- se filtra
        ]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    calls = []

    def _fake_fetch(player_id, season, raw_dir, force_refresh):
        calls.append((player_id, season))
        return pd.DataFrame(
            [{"LOC_X": 10, "LOC_Y": 20, "SHOT_MADE_FLAG": 1, "SHOT_TYPE": "2PT Field Goal", "SHOT_ZONE_BASIC": "Mid-Range"}]
        )

    monkeypatch.setattr(data_pipeline, "fetch_player_shot_chart", _fake_fetch)

    # n_seasons=2 explícito -- las 2 más recientes de las 3 reales, nunca la de proyección.
    result = data_pipeline.build_roster_shot_charts_dataset(config, n_seasons=2)

    assert calls == [(1, "2024-25"), (1, "2023-24")]
    assert len(result) == 2
    assert set(result["season"]) == {"2023-24", "2024-25"}
    assert result.iloc[0]["shot_made"] == True  # noqa: E712
    assert result.iloc[0]["shot_zone_basic"] == "Mid-Range"


def test_build_roster_shot_charts_dataset_defaults_to_aging_curve_lookback(config, monkeypatch):
    from aging_curve import DEFAULT_N_SEASONS_LOOKBACK

    processed_dir = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [{"PLAYER_ID": 1, "SEASON_ID": f"20{20+i}-{21+i}", "GP": 60} for i in range(DEFAULT_N_SEASONS_LOOKBACK + 2)]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    calls = []
    monkeypatch.setattr(
        data_pipeline,
        "fetch_player_shot_chart",
        lambda player_id, season, raw_dir, force_refresh: (calls.append(season), pd.DataFrame())[1],
    )

    data_pipeline.build_roster_shot_charts_dataset(config)

    assert len(calls) == DEFAULT_N_SEASONS_LOOKBACK  # sin pasar n_seasons, usa el default del proyecto


def test_build_roster_shot_charts_dataset_skips_players_with_no_shots(config, monkeypatch):
    processed_dir = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame([{"PLAYER_ID": 1, "SEASON_ID": "2024-25", "GP": 60}]).to_csv(
        processed_dir / "roster_career_stats.csv", index=False
    )
    monkeypatch.setattr(data_pipeline, "fetch_player_shot_chart", lambda *a, **k: pd.DataFrame())

    result = data_pipeline.build_roster_shot_charts_dataset(config)

    assert result.empty


def test_build_league_shot_charts_dataset_requires_league_career_stats(config):
    with pytest.raises(FileNotFoundError):
        data_pipeline.build_league_shot_charts_dataset(config)


def test_build_league_shot_charts_dataset_covers_every_unique_league_player_and_season(config, monkeypatch):
    processed_dir = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "SEASON_ID": "2024-25", "GP": 60},
            {"PLAYER_ID": 1, "SEASON_ID": "2025-26", "GP": 40},
            {"PLAYER_ID": 2, "SEASON_ID": "2025-26", "GP": 70},
            {"PLAYER_ID": 3, "SEASON_ID": "2026-27", "GP": 0},  # sin partidos reales todavía -- se filtra
        ]
    ).to_csv(processed_dir / "league_player_career_stats.csv", index=False)

    calls = []

    def _fake_fetch(player_id, season, raw_dir, force_refresh):
        calls.append((player_id, season))
        return pd.DataFrame(
            [{"LOC_X": 5, "LOC_Y": 5, "SHOT_MADE_FLAG": 1, "SHOT_TYPE": "2PT Field Goal", "SHOT_ZONE_BASIC": "Restricted Area"}]
        )

    monkeypatch.setattr(data_pipeline, "fetch_player_shot_chart", _fake_fetch)

    # n_seasons=2 explícito para que el jugador 1 (2 temporadas reales) traiga ambas.
    result = data_pipeline.build_league_shot_charts_dataset(config, n_seasons=2)

    assert sorted(calls) == [(1, "2024-25"), (1, "2025-26"), (2, "2025-26")]
    assert set(result["player_id"]) == {1, 2}
    assert "shot_zone_basic" in result.columns


def test_build_league_shot_charts_dataset_skips_a_player_whose_download_fails(config, monkeypatch):
    processed_dir = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "SEASON_ID": "2025-26", "GP": 60},
            {"PLAYER_ID": 2, "SEASON_ID": "2025-26", "GP": 70},
        ]
    ).to_csv(processed_dir / "league_player_career_stats.csv", index=False)

    def _flaky_fetch(player_id, season, raw_dir, force_refresh):
        if player_id == 1:
            raise KeyError("resultSet")  # misma clase de fallo real ya visto en este endpoint
        return pd.DataFrame(
            [{"LOC_X": 5, "LOC_Y": 5, "SHOT_MADE_FLAG": 1, "SHOT_TYPE": "2PT Field Goal", "SHOT_ZONE_BASIC": "Restricted Area"}]
        )

    monkeypatch.setattr(data_pipeline, "fetch_player_shot_chart", _flaky_fetch)

    result = data_pipeline.build_league_shot_charts_dataset(config)

    # El jugador 1 se salta sin abortar la descarga del jugador 2.
    assert set(result["player_id"]) == {2}
