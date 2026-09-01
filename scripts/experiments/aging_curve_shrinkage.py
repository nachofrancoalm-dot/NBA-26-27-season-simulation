"""
aging_curve_shrinkage.py

EXPERIMENTO, no forma parte del pipeline de producción. Comprueba si
`config["aging_curve"]` (n_seasons_lookback / recency_half_life_seasons)
está sobre-encogiendo la proyección de cada jugador hacia su media de
varias temporadas, comprimiendo la dispersión de talento entre equipos.

Diagnóstico previo (bayesian_calibration.py): el modelo reparte talento
con la MITAD de la dispersión real (std simulado ~6.8 victorias, real
~12.25, consistente en 16 temporadas). Descontando el ruido de temporada
ya medido (K=7.23 -> std ~4.53 victorias), la dispersión de TALENTO
objetivo es sqrt(12.25^2 - 4.53^2) ~= 11.38 victorias ~= 4.59 puntos de
diferencial (1 punto = 2.48 victorias). Verificado por la vía alternativa
(std de DiffPointsPG real) con el mismo resultado.

Prerrequisito: `league_simulation.project_team_roster()` y
`backtesting.project_historical_player()` deben propagar
`config["aging_curve"]` hasta `project_player_season()`; si no, calibrar
el YAML no cambia nada en producción ni en el backtesting.

Método: para cada combinación (n_seasons_lookback, half_life_seasons) del
grid, sobre los 480 casos del backtest sweep, proyecta con
`aging_curve.project_player_season()` (misma función de producción, sin
muestreo Monte Carlo de lesión/fatiga -- no depende de estos parámetros)
y mide la dispersión del Game Score de equipo (centrada por temporada)
y su correlación con el DiffPointsPG real.

Uso:
    python scripts/experiments/aging_curve_shrinkage.py
    python scripts/experiments/aging_curve_shrinkage.py --loso
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config, resolve_backtest_sweep_cases  # noqa: E402
from backtesting import filter_seasons_before, get_actual_season_row  # noqa: E402
from aging_curve import project_player_season  # noqa: E402
from advanced_impact import adjust_with_context, build_advanced_context, load_advanced_stats  # noqa: E402
from simulation import DEFAULT_ROTATION_SIZE, normalize_rotation_minutes  # noqa: E402
from season_utils import season_start_year  # noqa: E402

CURRENT_SCALE = 0.172  # game_score_to_net_rating_scale recalibrado (ver bayesian_calibration.py)
# Ruido de temporada en victorias, convertido a puntos -- no restar victorias de puntos sin convertir.
REAL_SEASON_LUCK_STD_WINS = 4.53
WINS_PER_POINT_OF_DIFFERENTIAL = 2.48
REAL_SEASON_LUCK_STD_POINTS = REAL_SEASON_LUCK_STD_WINS / WINS_PER_POINT_OF_DIFFERENTIAL

FEATURES_FILENAME = "experiment_aging_curve_shrinkage_features.csv"
RESULTS_FILENAME = "experiment_aging_curve_shrinkage_grid.csv"


def build_grid_features(
    config: Dict[str, Any], grid: List[Tuple[int, float]],
) -> pd.DataFrame:
    """Una fila por (caso equipo-temporada, combinación del grid) con el Game Score de equipo resultante."""
    paths = get_paths(config)
    cases = resolve_backtest_sweep_cases(config)
    if not cases:
        raise ValueError("config['backtest_sweep'] no está definido -- añade el bloque en team_config.yaml.")

    rosters = pd.read_csv(paths["processed"] / "backtest_sweep_rosters.csv")
    player_regular_stats = pd.read_csv(paths["processed"] / "backtest_sweep_player_career_stats.csv")
    advanced_context = build_advanced_context(load_advanced_stats(paths["processed"]), config)
    rotation_size = config.get("league_simulation", {}).get("rotation_size", DEFAULT_ROTATION_SIZE)

    rows = []
    for i, case in enumerate(cases):
        case_roster = rosters[(rosters["comparable_name"] == case["name"]) & (rosters["season"] == case["season"])]
        if case_roster.empty:
            continue
        player_ids = case_roster["PLAYER_ID"].astype(int).tolist()
        target_year = season_start_year(case["season"])

        raw_minutes: Dict[int, float] = {}
        prior_by_player: Dict[int, pd.DataFrame] = {}
        ages_by_player: Dict[int, float] = {}
        for player_id in player_ids:
            player_regular = player_regular_stats[player_regular_stats["PLAYER_ID"] == player_id]
            season_row = get_actual_season_row(player_regular, case["season"])
            raw_minutes[player_id] = (
                float(season_row["MIN"]) / float(season_row["GP"])
                if season_row is not None and float(season_row["GP"]) > 0 else 0.0
            )
            prior_by_player[player_id] = filter_seasons_before(player_regular, target_year)
            ages_by_player[player_id] = float(case_roster[case_roster["PLAYER_ID"] == player_id].iloc[0]["AGE"])

        normalized_minutes = normalize_rotation_minutes(raw_minutes, rotation_size)

        for n_seasons, half_life in grid:
            team_gs = 0.0
            for player_id in player_ids:
                prior_regular = prior_by_player[player_id]
                minutes_per_game = normalized_minutes[player_id]
                if prior_regular.empty or minutes_per_game <= 0:
                    continue
                projection = project_player_season(
                    prior_regular, target_age=ages_by_player[player_id],
                    minutes_per_game=minutes_per_game, games_per_season=82,
                    n_seasons=n_seasons, half_life_seasons=half_life,
                )
                gs36 = projection["game_score_per36"]
                if advanced_context is not None:
                    gs36 = adjust_with_context(gs36, player_id, case["season"], advanced_context)
                team_gs += gs36 * minutes_per_game / 36.0

            rows.append({
                "comparable_name": case["name"], "team_id": case["team_id"], "season": case["season"],
                "n_seasons_lookback": n_seasons, "half_life_seasons": half_life, "team_game_score": team_gs,
            })
        if (i + 1) % 60 == 0:
            print(f"  ...{i + 1}/{len(cases)} casos")

    features = pd.DataFrame(rows)
    standings = pd.read_csv(paths["processed"] / "backtest_sweep_standings.csv")
    standings = standings[["TeamID", "season", "DiffPointsPG"]].rename(columns={"TeamID": "team_id"})
    return features.merge(standings, on=["team_id", "season"], how="inner")


def evaluate_grid(features: pd.DataFrame) -> pd.DataFrame:
    """Por cada combinación: dispersión de talento (centrada por temporada) en puntos vía CURRENT_SCALE, y correlación con el diferencial real."""
    real_season_std_points = features.groupby("season")["DiffPointsPG"].std().mean()
    target_talent_std = float(np.sqrt(real_season_std_points ** 2 - REAL_SEASON_LUCK_STD_POINTS ** 2))

    rows = []
    for (n_seasons, half_life), g in features.groupby(["n_seasons_lookback", "half_life_seasons"]):
        centered = g.groupby("season")["team_game_score"].transform(lambda s: s - s.mean())
        talent_std_gs = centered.groupby(g["season"]).std().mean()
        talent_std_pts = talent_std_gs * CURRENT_SCALE
        corr = np.corrcoef(centered, g["DiffPointsPG"])[0, 1]
        rows.append({
            "n_seasons_lookback": n_seasons, "half_life_seasons": half_life,
            "talent_std_pts": talent_std_pts, "target_talent_std_pts": target_talent_std,
            "gap_to_target": talent_std_pts - target_talent_std,
            "correlation": corr,
        })
    return pd.DataFrame(rows).sort_values("gap_to_target", key=abs)


def run_loso(config: Dict[str, Any], grid: List[Tuple[int, float]], features: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-season-out: evalúa el grid con las otras 15 temporadas y anota la combinación más cercana al objetivo, para chequear estabilidad."""
    seasons = sorted(features["season"].unique())
    rows = []
    for held_out in seasons:
        train = features[features["season"] != held_out]
        ranked = evaluate_grid(train)
        best = ranked.iloc[0]
        rows.append({
            "held_out_season": held_out,
            "best_n_seasons_lookback": int(best["n_seasons_lookback"]),
            "best_half_life_seasons": float(best["half_life_seasons"]),
            "talent_std_pts": float(best["talent_std_pts"]),
            "correlation": float(best["correlation"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--loso", action="store_true")
    args = parser.parse_args()

    config = load_config()
    paths = get_paths(config)
    features_path = paths["processed"] / FEATURES_FILENAME

    grid = list(product([1, 2, 3, 4], [0.5, 1.0, 1.5, 2.5, 4.0]))
    print(f"Grid: {len(grid)} combinaciones (n_seasons_lookback x half_life_seasons)")

    if args.refresh_features or not features_path.exists():
        features = build_grid_features(config, grid)
        features.to_csv(features_path, index=False)
        print(f"Guardado: {features_path} ({len(features)} filas)")
    else:
        features = pd.read_csv(features_path)
        print(f"Usando features cacheadas: {features_path} ({len(features)} filas)")

    if args.loso:
        loso = run_loso(config, grid, features)
        print(loso.to_string())
        loso_path = paths["processed"] / "experiment_aging_curve_shrinkage_loso.csv"
        loso.to_csv(loso_path, index=False)
        print(f"Guardado: {loso_path}")
        return

    ranked = evaluate_grid(features)
    ranked.to_csv(paths["processed"] / RESULTS_FILENAME, index=False)
    print(ranked.to_string())
    print(f"\nGuardado: {paths['processed'] / RESULTS_FILENAME}")

    best = ranked.iloc[0]
    print(f"\nMejor combinación: n_seasons_lookback={int(best['n_seasons_lookback'])}, "
          f"half_life_seasons={best['half_life_seasons']}")
    print(f"talent_std_pts={best['talent_std_pts']:.2f} (objetivo {best['target_talent_std_pts']:.2f}), "
          f"correlación={best['correlation']:.3f}")


if __name__ == "__main__":
    main()
