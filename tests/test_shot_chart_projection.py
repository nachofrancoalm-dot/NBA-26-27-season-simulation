import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shot_chart_projection import project_player_shot_chart  # noqa: E402


def _shot(season, loc_x, loc_y, shot_made, shot_type, shot_zone_basic):
    return {
        "season": season, "loc_x": loc_x, "loc_y": loc_y, "shot_made": shot_made,
        "shot_type": shot_type, "shot_zone_basic": shot_zone_basic,
    }


def _uniform_pool(season="2025-26", n_2pt=20, n_3pt=20, make_rate_2pt=0.5, make_rate_3pt=0.35):
    rows = []
    for i in range(n_2pt):
        rows.append(_shot(season, 0, i, i < int(n_2pt * make_rate_2pt), "2PT Field Goal", "Mid-Range"))
    for i in range(n_3pt):
        rows.append(_shot(season, 200, i, i < int(n_3pt * make_rate_3pt), "3PT Field Goal", "Above the Break 3"))
    return pd.DataFrame(rows)


def test_matches_exact_target_attempt_and_make_counts():
    shots = _uniform_pool()
    result = project_player_shot_chart(
        shots, target_fga=8, target_fg3a=3, target_fgm=4, target_fg3m=1, rng=np.random.default_rng(0)
    )
    assert len(result) == 8
    assert (result["shot_type"] == "3PT Field Goal").sum() == 3
    assert (result["shot_type"] == "2PT Field Goal").sum() == 5
    assert result["shot_made"].sum() == 4
    assert result[result["shot_type"] == "3PT Field Goal"]["shot_made"].sum() == 1
    assert result[result["shot_type"] == "2PT Field Goal"]["shot_made"].sum() == 3


def test_makes_are_weighted_toward_the_more_efficient_zone():
    # Restricted Area anota siempre en el histórico; Mid-Range nunca --
    # con volumen suficiente para que ambas zonas entren en la muestra
    # de intentos, los anotados deben concentrarse en la zona de verdad
    # eficiente (piso MIN_ZONE_MAKE_RATE deja una probabilidad mínima a
    # Mid-Range, pero muy minoritaria).
    shots = pd.DataFrame(
        [_shot("2025-26", 0, i, True, "2PT Field Goal", "Restricted Area") for i in range(20)]
        + [_shot("2025-26", 50, i, False, "2PT Field Goal", "Mid-Range") for i in range(20)]
    )
    result = project_player_shot_chart(
        shots, target_fga=200, target_fg3a=0, target_fgm=100, target_fg3m=0, rng=np.random.default_rng(0)
    )
    made = result[result["shot_made"]]
    assert len(made) == 100
    assert (made["loc_x"] == 0).mean() > 0.8  # Restricted Area está en loc_x=0


def test_recency_weighting_favors_the_most_recent_season():
    # Temporada vieja: todos los tiros en loc_x=0. Temporada reciente:
    # todos en loc_x=100. Con half_life corto, el remuestreo debe
    # acercarse mucho a la temporada reciente.
    shots = pd.DataFrame(
        [_shot("2023-24", 0, i, True, "2PT Field Goal", "Mid-Range") for i in range(30)]
        + [_shot("2025-26", 100, i, True, "2PT Field Goal", "Mid-Range") for i in range(30)]
    )
    result = project_player_shot_chart(
        shots, target_fga=200, target_fg3a=0, target_fgm=200, target_fg3m=0,
        n_seasons=2, half_life_seasons=0.5, rng=np.random.default_rng(1),
    )
    # peso_reciente=1.0 vs peso_viejo=0.5**(1/0.5)=0.25 -> ~80% esperado
    recent_share = (result["loc_x"] == 100).mean()
    assert recent_share > 0.7


def test_n_seasons_window_excludes_older_seasons():
    shots = pd.DataFrame(
        [_shot("2022-23", -999, 0, True, "2PT Field Goal", "Mid-Range")]  # fuera de la ventana
        + [_shot("2025-26", 100, i, True, "2PT Field Goal", "Mid-Range") for i in range(10)]
    )
    result = project_player_shot_chart(
        shots, target_fga=50, target_fg3a=0, target_fgm=50, target_fg3m=0,
        n_seasons=1, rng=np.random.default_rng(2),
    )
    assert (result["loc_x"] == -999).sum() == 0


def test_missing_shot_type_in_history_yields_fewer_shots_than_requested():
    # Un pívot que nunca ha tirado un triple: no se inventa una zona.
    shots = _uniform_pool(n_3pt=0)
    result = project_player_shot_chart(
        shots, target_fga=10, target_fg3a=5, target_fgm=5, target_fg3m=2, rng=np.random.default_rng(3)
    )
    assert (result["shot_type"] == "3PT Field Goal").sum() == 0
    assert (result["shot_type"] == "2PT Field Goal").sum() == 5  # target_fga - target_fg3a


def test_empty_history_returns_empty_result_with_expected_columns():
    empty = pd.DataFrame(columns=["season", "loc_x", "loc_y", "shot_made", "shot_type", "shot_zone_basic"])
    result = project_player_shot_chart(empty, target_fga=10, target_fg3a=3, target_fgm=4, target_fg3m=1)
    assert result.empty
    assert list(result.columns) == ["loc_x", "loc_y", "shot_made", "shot_type"]


def test_same_seed_is_reproducible():
    shots = _uniform_pool()
    kwargs = dict(target_fga=8, target_fg3a=3, target_fgm=4, target_fg3m=1)
    first = project_player_shot_chart(shots, rng=np.random.default_rng(42), **kwargs)
    second = project_player_shot_chart(shots, rng=np.random.default_rng(42), **kwargs)
    pd.testing.assert_frame_equal(first, second)
