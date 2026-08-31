"""
Test de la parte pura de src/data_pipeline.py -- build_league_schedule_dataset()
(filtrado de temporada regular / partidos sin resolver), mockeando
fetch_league_schedule para no llamar a la API real. El resto de
data_pipeline.py son wrappers finos sobre nba_api sin lógica propia que
testear sin red -- esta función es la primera con un filtro real.
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
    """Plazas de la fase eliminatoria de la NBA Cup sin resolver todavía
    -- tricode nulo en vez de un equipo real. Se descartan, no se
    inventa a quién le toca jugar."""
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


def test_build_roster_shot_charts_dataset_uses_latest_real_season_per_player(config, monkeypatch):
    processed_dir = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "SEASON_ID": "2023-24", "GP": 60},
            {"PLAYER_ID": 1, "SEASON_ID": "2024-25", "GP": 70},
            # Temporada de proyección futura sin partidos jugados todavía -- se filtra.
            {"PLAYER_ID": 1, "SEASON_ID": "2026-27", "GP": 0},
        ]
    ).to_csv(processed_dir / "roster_career_stats.csv", index=False)

    calls = []

    def _fake_fetch(player_id, season, raw_dir, force_refresh):
        calls.append((player_id, season))
        return pd.DataFrame(
            [{"LOC_X": 10, "LOC_Y": 20, "SHOT_MADE_FLAG": 1, "SHOT_TYPE": "2PT Field Goal"}]
        )

    monkeypatch.setattr(data_pipeline, "fetch_player_shot_chart", _fake_fetch)

    result = data_pipeline.build_roster_shot_charts_dataset(config)

    assert calls == [(1, "2024-25")]  # la temporada real más reciente, no la de proyección
    assert len(result) == 1
    assert result.iloc[0]["season"] == "2024-25"
    assert result.iloc[0]["shot_made"] == True  # noqa: E712


def test_build_roster_shot_charts_dataset_skips_players_with_no_shots(config, monkeypatch):
    processed_dir = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame([{"PLAYER_ID": 1, "SEASON_ID": "2024-25", "GP": 60}]).to_csv(
        processed_dir / "roster_career_stats.csv", index=False
    )
    monkeypatch.setattr(data_pipeline, "fetch_player_shot_chart", lambda *a, **k: pd.DataFrame())

    result = data_pipeline.build_roster_shot_charts_dataset(config)

    assert result.empty
