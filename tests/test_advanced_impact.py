"""
Tests de la métrica de impacto compuesta (src/advanced_impact.py).

No requieren red: construyen DataFrames sintéticos con el esquema de
`league_advanced_player_stats.csv`.

El test más importante del archivo es
`test_league_average_player_gets_no_adjustment`: es el que protege la
restricción de suma cero, que ya se rompió una vez en este proyecto (ver
"INFLACIÓN DE ERA" en el docstring de simulation.py) y costó un sesgo de
-13 victorias por equipo.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from advanced_impact import (  # noqa: E402
    DEFAULT_ADVANCED_IMPACT_CONFIG,
    TOTAL_MINUTES_COLUMN,
    adjust_with_context,
    adjusted_game_score_per36,
    blend_impact_per36,
    build_advanced_context,
    compute_league_advanced_baselines,
    compute_recency_weighted_advanced,
    merge_pt_defend_stats,
    prepare_advanced_stats,
    resolve_advanced_impact_config,
)


def make_advanced(rows):
    """rows: [(season, player_id, min_per_game, gp, pie, net_rating), ...]"""
    return prepare_advanced_stats(
        pd.DataFrame(
            [
                {"season": s, "PLAYER_ID": p, "MIN": m, "GP": g, "PIE": pie, "NET_RATING": net}
                for s, p, m, g, pie, net in rows
            ]
        )
    )


def make_pt_defend(rows):
    """rows: [(season, player_id, d_fga, pct_plusminus), ...]"""
    return pd.DataFrame(
        [
            {"season": s, "PLAYER_ID": p, "D_FGA": fga, "PCT_PLUSMINUS": pct}
            for s, p, fga, pct in rows
        ]
    )


def test_prepare_advanced_stats_converts_minutes_to_totals():
    # La columna MIN del endpoint viene POR PARTIDO con measure_type
    # Advanced (ver data_pipeline.fetch_league_advanced_player_stats).
    # Si esto se rompe, el umbral min_minutes filtra a TODA la liga y el
    # módulo degrada en silencio a Game Score puro.
    df = make_advanced([("2024-25", 1, 30.0, 70, 0.12, 5.0)])
    assert df[TOTAL_MINUTES_COLUMN].iloc[0] == pytest.approx(2100.0)


def test_baselines_are_weighted_by_minutes_and_computed_per_season():
    adv = make_advanced([
        ("2023-24", 1, 30.0, 80, 0.20, 10.0),   # 2400 min
        ("2023-24", 2, 10.0, 80, 0.10, 0.0),    # 800 min
        ("2024-25", 3, 30.0, 80, 0.05, -5.0),
    ])
    baselines = compute_league_advanced_baselines(adv, min_minutes=500)

    assert set(baselines) == {"2023-24", "2024-25"}
    # Media ponderada: (0.20*2400 + 0.10*800) / 3200 = 0.175
    assert baselines["2023-24"]["PIE"] == pytest.approx(0.175)
    assert baselines["2023-24"]["NET_RATING"] == pytest.approx(7.5)
    # Cada temporada es su propio universo -- ese es el ajuste de era.
    assert baselines["2024-25"]["NET_RATING"] == pytest.approx(-5.0)


def test_baselines_exclude_players_below_the_minutes_threshold():
    adv = make_advanced([
        ("2024-25", 1, 30.0, 80, 0.15, 5.0),
        ("2024-25", 2, 2.0, 10, 0.01, -40.0),  # 20 min: ruido puro
    ])
    baselines = compute_league_advanced_baselines(adv, min_minutes=500)
    assert baselines["2024-25"]["NET_RATING"] == pytest.approx(5.0)


def test_recency_weighting_favours_the_most_recent_season():
    adv = make_advanced([
        ("2022-23", 1, 30.0, 80, 0.10, 0.0),
        ("2023-24", 1, 30.0, 80, 0.10, 0.0),
        ("2024-25", 1, 30.0, 80, 0.10, 10.0),
    ])
    result = compute_recency_weighted_advanced(adv, "2025-26", n_seasons=3, half_life=1.5)
    # La más reciente pesa más que las otras dos, pero no es el único
    # aporte -- debe quedar entre la media simple (3.33) y el valor
    # reciente (10.0).
    assert 3.33 < result["NET_RATING"] < 10.0


def test_recency_weighting_never_looks_ahead():
    """Regla de no look-ahead, la misma que backtesting.filter_seasons_before."""
    adv = make_advanced([
        ("2023-24", 1, 30.0, 80, 0.10, 0.0),
        ("2024-25", 1, 30.0, 80, 0.10, 99.0),  # la temporada que se predice
    ])
    result = compute_recency_weighted_advanced(adv, "2024-25", n_seasons=3, half_life=1.5)
    assert result["NET_RATING"] == pytest.approx(0.0)


def test_recency_weighting_returns_none_without_eligible_seasons():
    adv = make_advanced([("2024-25", 1, 30.0, 80, 0.10, 5.0)])
    assert compute_recency_weighted_advanced(adv, "2024-25", 3, 1.5) is None


def test_traded_season_collapses_to_one_minutes_weighted_row():
    # Un jugador traspasado aparece con una fila por equipo la misma
    # temporada; deben colapsarse, no contar como dos temporadas (eso
    # desplazaría la ventana de recencia).
    adv = make_advanced([
        ("2024-25", 1, 30.0, 60, 0.10, 10.0),  # 1800 min
        ("2024-25", 1, 30.0, 20, 0.10, -10.0),  # 600 min
        ("2023-24", 1, 30.0, 80, 0.10, 0.0),
    ])
    result = compute_recency_weighted_advanced(adv, "2025-26", n_seasons=1, half_life=1.5)
    # (10*1800 + -10*600) / 2400 = 5.0
    assert result["NET_RATING"] == pytest.approx(5.0)


def test_league_average_player_gets_no_adjustment():
    """
    LA PROPIEDAD CENTRAL: un jugador exactamente en la media de liga no
    recibe ajuste. Es lo que mantiene la restricción de suma cero -- si
    el ajuste tuviera media distinta de cero sobre la liga, desplazaría a
    todos los equipos a la vez y la media de victorias dejaría de ser
    games/2.
    """
    config = dict(DEFAULT_ADVANCED_IMPACT_CONFIG)
    baseline = {"PIE": 0.10, "NET_RATING": 2.0}
    result = blend_impact_per36(15.0, {"PIE": 0.10, "NET_RATING": 2.0}, baseline, config)
    assert result == pytest.approx(15.0)


def test_adjustment_is_signed_by_deviation_from_the_league():
    config = dict(DEFAULT_ADVANCED_IMPACT_CONFIG)
    baseline = {"PIE": 0.10, "NET_RATING": 0.0}

    good = blend_impact_per36(15.0, {"PIE": 0.10, "NET_RATING": 8.0}, baseline, config)
    bad = blend_impact_per36(15.0, {"PIE": 0.10, "NET_RATING": -8.0}, baseline, config)

    assert good > 15.0 > bad
    # Simétrico: el ajuste es lineal en la desviación.
    assert (good - 15.0) == pytest.approx(15.0 - bad)


def test_pie_weight_is_zero_by_default():
    """
    Medido sobre los 480 casos: PIE no aporta fuera de muestra y con
    NET_RATING presente su coeficiente sale con el signo invertido. El
    cableado se conserva, el peso no. Si alguien lo sube sin re-medir,
    este test lo obliga a pasar por aquí.
    """
    assert DEFAULT_ADVANCED_IMPACT_CONFIG["pie_weight"] == 0.0
    config = dict(DEFAULT_ADVANCED_IMPACT_CONFIG)
    baseline = {"PIE": 0.10, "NET_RATING": 0.0}
    unchanged = blend_impact_per36(15.0, {"PIE": 0.30, "NET_RATING": 0.0}, baseline, config)
    assert unchanged == pytest.approx(15.0)


def test_missing_data_degrades_to_pure_game_score():
    """Degradar al Game Score puro siempre es seguro; inventar un ajuste no."""
    config = dict(DEFAULT_ADVANCED_IMPACT_CONFIG)
    assert blend_impact_per36(15.0, None, {"NET_RATING": 0.0}, config) == 15.0
    assert blend_impact_per36(15.0, {"NET_RATING": 9.0}, None, config) == 15.0


def test_disabled_config_returns_pure_game_score():
    config = dict(DEFAULT_ADVANCED_IMPACT_CONFIG, enabled=False)
    baseline = {"PIE": 0.10, "NET_RATING": 0.0}
    assert blend_impact_per36(15.0, {"PIE": 0.3, "NET_RATING": 20.0}, baseline, config) == 15.0


def test_resolve_config_merges_over_defaults():
    resolved = resolve_advanced_impact_config({"advanced_impact": {"net_rating_weight": 0.9}})
    assert resolved["net_rating_weight"] == 0.9
    assert resolved["min_minutes_for_advanced"] == DEFAULT_ADVANCED_IMPACT_CONFIG["min_minutes_for_advanced"]


def test_build_advanced_context_returns_none_when_disabled_or_empty():
    adv = make_advanced([("2024-25", 1, 30.0, 80, 0.10, 5.0)])
    assert build_advanced_context(adv, {"advanced_impact": {"enabled": False}}) is None
    assert build_advanced_context(pd.DataFrame(), {}) is None


def test_adjust_with_context_matches_the_low_level_path():
    """
    `adjust_with_context` (la firma que usan los tres motores) y
    `adjusted_game_score_per36` (la de bajo nivel) deben dar el MISMO
    número. Este proyecto ya arrastró el mismo bug en dos módulos por
    tener lógica duplicada -- si estos dos caminos divergen, el backtest
    validaría una métrica distinta de la que simula.
    """
    adv = make_advanced([
        ("2023-24", 1, 30.0, 80, 0.15, 8.0),
        ("2023-24", 2, 30.0, 80, 0.10, 0.0),
        ("2024-25", 1, 30.0, 80, 0.15, 8.0),
        ("2024-25", 2, 30.0, 80, 0.10, 0.0),
    ])
    config = {}
    context = build_advanced_context(adv, config)

    via_context = adjust_with_context(15.0, 1, "2025-26", context)
    via_low_level = adjusted_game_score_per36(
        15.0, adv[adv["PLAYER_ID"] == 1], "2025-26", context["baselines"], context["impact_config"]
    )
    assert via_context == pytest.approx(via_low_level)
    assert via_context != pytest.approx(15.0)  # el jugador 1 está por encima de la media


def test_adjust_with_context_is_a_noop_without_context():
    assert adjust_with_context(15.0, 1, "2025-26", None) == 15.0


def test_unknown_player_falls_back_to_pure_game_score():
    adv = make_advanced([("2024-25", 1, 30.0, 80, 0.10, 5.0)])
    context = build_advanced_context(adv, {})
    assert adjust_with_context(15.0, 999, "2025-26", context) == 15.0


# ---------------------------------------------------------------------------
# PCT_PLUSMINUS (defensa por tracking) -- ver "TERCERA MÉTRICA INTEGRADA"
# en el docstring del módulo. La cobertura parcial (solo desde 2013-14) es
# justo lo que estos tests protegen: PIE/NET_RATING nunca tuvieron datos
# faltantes porque comparten endpoint, así que ese caso nunca se probó
# antes de esta métrica.
# ---------------------------------------------------------------------------


def test_merge_pt_defend_stats_adds_the_column_by_player_and_season():
    adv = make_advanced([("2023-24", 1, 30.0, 80, 0.15, 8.0)])
    pt = make_pt_defend([("2023-24", 1, 200.0, -0.03)])
    merged = merge_pt_defend_stats(adv, pt)
    assert merged["PCT_PLUSMINUS"].iloc[0] == pytest.approx(-0.03)


def test_merge_pt_defend_stats_collapses_traded_players_weighted_by_dfga():
    adv = make_advanced([("2023-24", 1, 30.0, 80, 0.15, 8.0)])
    pt = make_pt_defend([
        ("2023-24", 1, 150.0, -0.04),  # equipo A
        ("2023-24", 1, 50.0, 0.02),    # equipo B, tras el traspaso
    ])
    merged = merge_pt_defend_stats(adv, pt)
    # (-0.04*150 + 0.02*50) / 200 = -0.025
    assert merged["PCT_PLUSMINUS"].iloc[0] == pytest.approx(-0.025)


def test_merge_pt_defend_stats_leaves_nan_for_players_without_tracking_data():
    adv = make_advanced([
        ("2010-11", 1, 30.0, 80, 0.15, 8.0),  # anterior a 2013-14 -- sin tracking
    ])
    pt = make_pt_defend([("2015-16", 2, 200.0, -0.03)])  # otro jugador, otra temporada
    merged = merge_pt_defend_stats(adv, pt)
    assert merged["PCT_PLUSMINUS"].isna().all()


def test_merge_pt_defend_stats_is_a_noop_without_pt_defend_data():
    adv = make_advanced([("2023-24", 1, 30.0, 80, 0.15, 8.0)])
    assert merge_pt_defend_stats(adv, pd.DataFrame()) is adv
    assert merge_pt_defend_stats(adv, None) is adv


def test_baselines_use_only_non_null_rows_per_metric():
    """
    BUG REAL que este test evita reintroducir: promediar PCT_PLUSMINUS
    con un denominador que incluye el peso de jugadores SIN dato para esa
    métrica sesgaría la media hacia 0 en silencio. PIE/NET_RATING deben
    seguir viendo a los 3 jugadores; PCT_PLUSMINUS solo a los 2 que lo
    tienen.
    """
    adv = make_advanced([
        ("2023-24", 1, 30.0, 80, 0.20, 10.0),  # 2400 min, con tracking
        ("2023-24", 2, 30.0, 80, 0.10, 0.0),   # 2400 min, con tracking
        ("2023-24", 3, 30.0, 80, 0.30, 20.0),  # 2400 min, SIN tracking
    ])
    pt = make_pt_defend([
        ("2023-24", 1, 200.0, -0.04),
        ("2023-24", 2, 200.0, 0.02),
    ])
    adv = merge_pt_defend_stats(adv, pt)
    baselines = compute_league_advanced_baselines(adv, min_minutes=500)

    # PIE/NET_RATING: media de los 3 jugadores (7200 min totales).
    assert baselines["2023-24"]["NET_RATING"] == pytest.approx(10.0)
    # PCT_PLUSMINUS: media SOLO de los 2 con dato (4800 min), no diluida
    # por los 2400 min del jugador 3 que no tiene la métrica.
    assert baselines["2023-24"]["PCT_PLUSMINUS"] == pytest.approx((-0.04 + 0.02) / 2)


def test_recency_weighted_advanced_omits_metric_without_any_eligible_season():
    """Un jugador cuyo historial reciente no tiene NINGÚN dato de tracking
    no debe llevar PCT_PLUSMINUS en el resultado -- ni inventado como 0.0
    (que significaría "defensor exactamente promedio")."""
    adv = make_advanced([("2010-11", 1, 30.0, 80, 0.15, 8.0)])
    result = compute_recency_weighted_advanced(adv, "2011-12", n_seasons=3, half_life=1.5)
    assert "PIE" in result and "NET_RATING" in result
    assert "PCT_PLUSMINUS" not in result


def test_recency_weighted_advanced_uses_only_seasons_with_the_metric():
    # 2 temporadas previas; solo la más reciente tiene PCT_PLUSMINUS.
    adv = make_advanced([
        ("2013-14", 1, 30.0, 80, 0.10, 0.0),
        ("2014-15", 1, 30.0, 80, 0.10, 0.0),
    ])
    pt = make_pt_defend([("2014-15", 1, 200.0, -0.05)])
    adv = merge_pt_defend_stats(adv, pt)

    result = compute_recency_weighted_advanced(adv, "2015-16", n_seasons=3, half_life=1.5)
    # Con una sola temporada disponible para PCT_PLUSMINUS, su media
    # ponderada es exactamente ese valor -- no se diluye con la
    # temporada sin dato.
    assert result["PCT_PLUSMINUS"] == pytest.approx(-0.05)


def test_pct_plusminus_weight_rewards_good_defense_with_a_positive_adjustment():
    """PCT_PLUSMINUS negativo = mejor defensa que la media -- el ajuste
    debe ser POSITIVO para ese jugador (signo del peso ya invertido en
    DEFAULT_ADVANCED_IMPACT_CONFIG, ver docstring del módulo)."""
    config = dict(DEFAULT_ADVANCED_IMPACT_CONFIG)
    baseline = {"PIE": 0.10, "NET_RATING": 0.0, "PCT_PLUSMINUS": 0.0}

    good_defender = blend_impact_per36(
        15.0, {"PIE": 0.10, "NET_RATING": 0.0, "PCT_PLUSMINUS": -0.03}, baseline, config
    )
    bad_defender = blend_impact_per36(
        15.0, {"PIE": 0.10, "NET_RATING": 0.0, "PCT_PLUSMINUS": 0.03}, baseline, config
    )
    assert good_defender > 15.0 > bad_defender
