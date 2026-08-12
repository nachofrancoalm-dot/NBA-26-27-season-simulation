"""
team_quality_uncertainty.py

EXPERIMENTO, no forma parte del pipeline de producción. Calibra
`monte_carlo.team_quality_uncertainty_std` (ver
simulation.sample_team_quality_noise, añadido en esta misma sesión,
desactivado por defecto con std=0.0) contra el backtest sweep de 480
casos reales.

QUÉ PROBLEMA ATACA (y cuál NO): el usuario reportó dos síntomas --
(1) poca diferencia de victorias medias entre equipos en la predicción
26-27, y (2) 42% de los 480 casos del backtest caen en el 20% más
extremo de percentiles (debería ser ~20%). `aging_curve_shrinkage.py` ya
descartó con evidencia que (1) se deba al encogimiento de la proyección
de talento -- barrido completo del grid, dispersión prácticamente
idéntica en todo el rango. `team_quality_uncertainty_std` es DE MEDIA
CERO, así que por construcción NO puede mover wins_mean (se cancela al
promediar miles de temporadas simuladas, ver los tests de regresión en
tests/test_simulation.py) -- este experimento ataca ÚNICAMENTE el
síntoma (2), la calibración de las bandas P10-P90.

DISEÑO
------
Para cada candidato de `team_quality_uncertainty_std`, corre el backtest
completo (`backtesting._run_backtest_cases`, la MISMA función que usa
`build_backtest_sweep_dataset`, nunca reimplementada) sobre los 480
casos, con `n_seasons` REDUCIDO (ver `EXPERIMENT_N_SEASONS`) para que la
búsqueda sea barata -- la métrica de interés (`pct_within_p10_p90`,
agregada sobre 480 casos) ya tiene muestra grande a nivel población
aunque cada P10/P90 individual sea algo más ruidoso con menos
temporadas por caso. Usa la línea base de era ya calculada
(`league_game_score_baseline.csv`, de la última corrida de
`build_backtest_sweep_dataset`) en vez de recalcularla -- ligeramente
desactualizada tras la recalibración de `game_score_to_net_rating_scale`
en esta misma sesión, pero de sobra para elegir un orden de magnitud de
`std`, que es lo que pide este experimento.

Uso:
    python scripts/experiments/team_quality_uncertainty.py
    python scripts/experiments/team_quality_uncertainty.py --loso
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config, resolve_backtest_sweep_cases  # noqa: E402
from backtesting import (  # noqa: E402
    _run_backtest_cases,
    compute_calibration_summary,
    load_league_baselines,
)
from advanced_impact import build_advanced_context, load_advanced_stats  # noqa: E402

EXPERIMENT_N_SEASONS = 1000  # reducido de los 10000 de producción -- ver docstring del módulo
CANDIDATES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]  # puntos de diferencial
RESULTS_FILENAME = "experiment_team_quality_uncertainty_grid.csv"


def _load_inputs(config: Dict[str, Any]):
    paths = get_paths(config)
    required = [
        "backtest_sweep_rosters.csv", "backtest_sweep_player_career_stats.csv",
        "backtest_sweep_standings.csv", "backtest_sweep_advanced_game_logs.csv",
    ]
    for filename in required:
        if not (paths["processed"] / filename).exists():
            raise FileNotFoundError(f"No se encontró {paths['processed'] / filename}.")

    rosters = pd.read_csv(paths["processed"] / "backtest_sweep_rosters.csv")
    player_regular_stats = pd.read_csv(paths["processed"] / "backtest_sweep_player_career_stats.csv")
    playoff_path = paths["processed"] / "backtest_sweep_player_playoff_career_stats.csv"
    player_playoff_stats = (
        pd.read_csv(playoff_path) if playoff_path.exists() and playoff_path.stat().st_size > 0 else pd.DataFrame()
    )
    standings = pd.read_csv(paths["processed"] / "backtest_sweep_standings.csv")
    game_log = pd.read_csv(paths["processed"] / "backtest_sweep_advanced_game_logs.csv")
    advanced_context = build_advanced_context(load_advanced_stats(paths["processed"]), config)
    league_baseline_by_season = load_league_baselines(paths["processed"])
    cases = resolve_backtest_sweep_cases(config)
    return cases, rosters, player_regular_stats, player_playoff_stats, standings, game_log, advanced_context, league_baseline_by_season


def run_backtest_with_std(
    config: Dict[str, Any], team_quality_std: float, n_seasons: int, cases=None, inputs=None,
) -> pd.DataFrame:
    """Corre `_run_backtest_cases` (núcleo real de producción, no
    reimplementado) con `team_quality_uncertainty_std` inyectado en
    `config["monte_carlo"]` y `simulation.n_seasons` reducido para la
    búsqueda. `inputs`/`cases` opcionales para no releer los CSV en cada
    punto del grid."""
    if inputs is None:
        cases, rosters, player_regular_stats, player_playoff_stats, standings, game_log, advanced_context, league_baseline_by_season = _load_inputs(config)
    else:
        rosters, player_regular_stats, player_playoff_stats, standings, game_log, advanced_context, league_baseline_by_season = inputs

    run_config = {
        **config,
        "simulation": {**config["simulation"], "n_seasons": n_seasons},
        "monte_carlo": {**config.get("monte_carlo", {}), "team_quality_uncertainty_std": team_quality_std},
    }
    return _run_backtest_cases(
        cases, rosters, player_regular_stats, player_playoff_stats, standings, game_log, run_config,
        show_progress=False, league_baseline_by_season=league_baseline_by_season, advanced_context=advanced_context,
    )


def sweep(config: Dict[str, Any], candidates: List[float], n_seasons: int) -> pd.DataFrame:
    cases, rosters, player_regular_stats, player_playoff_stats, standings, game_log, advanced_context, league_baseline_by_season = _load_inputs(config)
    inputs = (rosters, player_regular_stats, player_playoff_stats, standings, game_log, advanced_context, league_baseline_by_season)

    rows = []
    for std in candidates:
        print(f"team_quality_uncertainty_std={std} ...")
        result_df = run_backtest_with_std(config, std, n_seasons, cases=cases, inputs=inputs)
        calibration = compute_calibration_summary(result_df)
        rows.append({"team_quality_uncertainty_std": std, **calibration})
        print(f"  pct_within_p10_p90={calibration['pct_within_p10_p90']:.1f}  "
              f"mean_absolute_error_wins={calibration['mean_absolute_error_wins']:.2f}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seasons", type=int, default=EXPERIMENT_N_SEASONS)
    parser.add_argument("--candidates", type=float, nargs="+", default=CANDIDATES)
    args = parser.parse_args()

    config = load_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        results = sweep(config, args.candidates, args.n_seasons)

    paths = get_paths(config)
    out_path = paths["processed"] / RESULTS_FILENAME
    results.to_csv(out_path, index=False)
    print("\n" + results.to_string())
    print(f"\nGuardado: {out_path}")

    results["gap_to_80"] = (results["pct_within_p10_p90"] - 80.0).abs()
    best = results.sort_values("gap_to_80").iloc[0]
    print(f"\nMejor candidato: team_quality_uncertainty_std={best['team_quality_uncertainty_std']} "
          f"(pct_within_p10_p90={best['pct_within_p10_p90']:.1f})")


if __name__ == "__main__":
    main()
