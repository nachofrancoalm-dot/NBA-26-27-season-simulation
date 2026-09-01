"""
Tests del config_loader. No requieren conexión a stats.nba.com — validan
solo la estructura del YAML, que es donde suelen colarse errores al
adaptar el proyecto a un equipo nuevo.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config_loader import load_config, resolve_backtest_sweep_cases, _validate_config  # noqa: E402


def test_default_config_loads():
    config = load_config()
    assert "team" in config
    assert "roster" in config
    assert config["team"]["name"]


def test_default_config_has_team_id():
    config = load_config()
    assert isinstance(config["team"]["team_id"], int)


def test_validate_config_rejects_missing_keys():
    with pytest.raises(ValueError):
        _validate_config({"team": {"team_id": 1}})  # faltan roster, etc.


def test_validate_config_rejects_player_without_id():
    bad_config = {
        "team": {"team_id": 1},
        "roster": [{"name": "Jugador sin ID"}],
        "historical_comparables": [],
        "simulation": {},
        "paths": {},
    }
    with pytest.raises(ValueError):
        _validate_config(bad_config)


def test_historical_comparables_have_required_fields():
    config = load_config()
    for case in config["historical_comparables"]:
        assert "team_id" in case
        assert "season" in case


def test_resolve_backtest_sweep_cases_returns_empty_when_not_configured():
    assert resolve_backtest_sweep_cases({}) == []
    assert resolve_backtest_sweep_cases({"backtest_sweep": None}) == []


def test_resolve_backtest_sweep_cases_expands_all_30_teams_per_season():
    config = {"backtest_sweep": {"seasons": ["2019-20", "2020-21"]}}

    cases = resolve_backtest_sweep_cases(config)

    assert len(cases) == 60  # 30 equipos x 2 temporadas
    seasons_seen = {case["season"] for case in cases}
    assert seasons_seen == {"2019-20", "2020-21"}
    for case in cases:
        assert "team_id" in case
        assert "name" in case


def test_resolve_backtest_sweep_cases_uses_the_same_team_id_across_seasons():
    # team_id es estable a través de mudanzas/renombres de franquicia
    config = {"backtest_sweep": {"seasons": ["2010-11", "2024-25"]}}

    cases = resolve_backtest_sweep_cases(config)

    team_ids_2010 = {c["team_id"] for c in cases if c["season"] == "2010-11"}
    team_ids_2024 = {c["team_id"] for c in cases if c["season"] == "2024-25"}
    assert team_ids_2010 == team_ids_2024


def test_default_config_defines_a_multi_season_backtest_sweep():
    config = load_config()
    assert "backtest_sweep" in config
    assert len(config["backtest_sweep"]["seasons"]) >= 15


def test_backtest_sweep_includes_the_most_recent_completed_season():
    # regression: el sweep se quedaba una temporada corto y dejaba fuera al campeón más reciente
    config = load_config()
    seasons = config["backtest_sweep"]["seasons"]
    target_year = int(str(config["team"]["season"])[:4])

    assert int(str(seasons[-1])[:4]) == target_year - 1
