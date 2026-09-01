"""
bayesian_calibration.py

EXPERIMENTO, no forma parte del pipeline de producción. Explora si
`game_score_to_net_rating_scale` (constante única en team_config.yaml,
recalibrada a mano dos veces -- ver CLAUDE.md) se puede sustituir por un
modelo bayesiano jerárquico con partial pooling por temporada: cada
temporada tiene su propio slope, encogido hacia una media común en
proporción inversa a sus datos, capturando variación por era sin dejar
que una temporada rara desvíe la calibración global.

Con 480 filas (30 equipos x 16 temporadas) un modelo jerárquico con
priors débilmente informativos es el punto correcto de complejidad, y da
un RANGO con incertidumbre en vez de la constante puntual actual.

x = Game Score de equipo esperado (con lesión/fatiga/sinergia aplicados)
menos la línea base de liga esa temporada -- el mismo número que hoy se
multiplica por game_score_to_net_rating_scale. y = DiffPointsPG real de
ese equipo esa temporada. Regresión directa sobre el diferencial de
temporada (un paso antes de la logística partido a partido de
producción), suficiente para comparar slope fijo vs. partial pooling.

Requiere `python src/data_pipeline.py --backtest-sweep` corrido antes y
las dependencias de requirements-experiments.txt (pymc + arviz, pesadas,
fuera del requirements.txt principal).

Uso:
    python scripts/experiments/bayesian_calibration.py
    python scripts/experiments/bayesian_calibration.py --refresh-features
    python scripts/experiments/bayesian_calibration.py --draws 2000 --chains 4
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config, resolve_backtest_sweep_cases  # noqa: E402
from backtesting import (  # noqa: E402
    compute_projected_league_baselines,
    expected_team_game_score_equivalent,
    project_backtest_team,
)
from advanced_impact import build_advanced_context, load_advanced_stats  # noqa: E402
from simulation import TOTAL_TEAM_MINUTES_PER_GAME  # noqa: E402

FEATURES_FILENAME = "experiment_bayesian_calibration_features.csv"
SUMMARY_FILENAME = "experiment_bayesian_calibration_summary.csv"
TRACE_FILENAME = "experiment_bayesian_calibration_trace.nc"
LOSO_FILENAME = "experiment_bayesian_calibration_loso.csv"
LOSO_DIAGNOSTICS_FILENAME = "experiment_bayesian_calibration_loso_diagnostics.csv"


def build_calibration_features(config: Dict[str, Any]) -> pd.DataFrame:
    """Una fila por caso equipo-temporada con el predictor `x` y target `y` del docstring del módulo, reutilizando las funciones de producción en src/backtesting.py sin reimplementarlas."""
    paths = get_paths(config)
    cases = resolve_backtest_sweep_cases(config)
    if not cases:
        raise ValueError(
            "config['backtest_sweep'] no está definido -- añade el bloque en team_config.yaml."
        )

    required = [
        "backtest_sweep_rosters.csv",
        "backtest_sweep_player_career_stats.csv",
        "backtest_sweep_standings.csv",
    ]
    for filename in required:
        if not (paths["processed"] / filename).exists():
            raise FileNotFoundError(
                f"No se encontró {paths['processed'] / filename}. Corre "
                "`python src/data_pipeline.py --backtest-sweep` primero."
            )

    rosters = pd.read_csv(paths["processed"] / "backtest_sweep_rosters.csv")
    player_regular_stats = pd.read_csv(paths["processed"] / "backtest_sweep_player_career_stats.csv")
    playoff_path = paths["processed"] / "backtest_sweep_player_playoff_career_stats.csv"
    player_playoff_stats = (
        pd.read_csv(playoff_path) if playoff_path.exists() and playoff_path.stat().st_size > 0 else pd.DataFrame()
    )
    standings = pd.read_csv(paths["processed"] / "backtest_sweep_standings.csv")
    standings = standings[["TeamID", "season", "DiffPointsPG", "WINS", "LOSSES"]].rename(
        columns={"TeamID": "team_id"}
    )

    advanced_context = build_advanced_context(load_advanced_stats(paths["processed"]), config)

    print(f"Proyectando {len(cases)} casos (sin Monte Carlo, primera pasada barata)...")
    league_baseline_by_season = compute_projected_league_baselines(
        cases, rosters, player_regular_stats, player_playoff_stats, config,
        advanced_context=advanced_context,
    )

    minutes_scale = TOTAL_TEAM_MINUTES_PER_GAME / 36.0
    rows = []
    for i, case in enumerate(cases):
        try:
            projection = project_backtest_team(
                case, rosters, player_regular_stats, player_playoff_stats, config,
                games_in_season=82, advanced_context=advanced_context,
            )
            team_game_score = expected_team_game_score_equivalent(projection, config, games_in_season=82)
        except Exception as exc:  # noqa: BLE001 -- un caso roto no debe abortar los otros 479
            print(f"  [aviso] Caso {case['name']} saltado: {exc}")
            continue

        baseline_per36 = league_baseline_by_season.get(case["season"])
        if baseline_per36 is None:
            continue
        baseline_equivalent = baseline_per36 * minutes_scale

        rows.append({
            "comparable_name": case["name"],
            "team_id": case["team_id"],
            "season": case["season"],
            "x_game_score_vs_baseline": team_game_score - baseline_equivalent,
        })
        if (i + 1) % 60 == 0:
            print(f"  ...{i + 1}/{len(cases)}")

    features = pd.DataFrame(rows).merge(standings, on=["team_id", "season"], how="inner")
    features["y_actual_diff_points_pg"] = features["DiffPointsPG"]
    return features.drop(columns=["DiffPointsPG"])


def fit_hierarchical_model(features: pd.DataFrame, draws: int, tune: int, chains: int, seed: int):
    """
    y ~ Normal(alpha[temporada] + beta[temporada] * x, sigma), con partial
    pooling de alpha y beta hacia alpha_mu/beta_mu (beta_mu es el sustituto
    de game_score_to_net_rating_scale). alpha_mu se centra en 0 porque la
    restricción de suma cero de compute_projected_league_baselines hace
    que x=0 y el diferencial de liga medio sean 0 por construcción; beta_mu
    parte de un prior débil alrededor del valor calibrado a mano (0.21).

    Usa parametrización no centrada (alpha/beta = mu + sigma * offset)
    porque la centrada, con alpha_sigma empujado a valores pequeños por la
    restricción de suma cero, produce el embudo de Neal: 19% de muestras
    divergentes y r_hat=1.23 en alpha_mu. La no centrada es la solución
    estándar (Betancourt) para este caso.
    """
    import pymc as pm

    seasons = sorted(features["season"].unique())
    season_idx = features["season"].map({s: i for i, s in enumerate(seasons)}).to_numpy()
    x = features["x_game_score_vs_baseline"].to_numpy()
    y = features["y_actual_diff_points_pg"].to_numpy()

    with pm.Model(coords={"season": seasons}) as model:
        alpha_mu = pm.Normal("alpha_mu", mu=0.0, sigma=3.0)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=3.0)
        beta_mu = pm.Normal("beta_mu", mu=0.21, sigma=0.15)
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=0.15)

        alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, dims="season")
        beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, dims="season")
        alpha = pm.Deterministic("alpha", alpha_mu + alpha_sigma * alpha_offset, dims="season")
        beta = pm.Deterministic("beta", beta_mu + beta_sigma * beta_offset, dims="season")

        sigma = pm.HalfNormal("sigma", sigma=5.0)
        mu = alpha[season_idx] + beta[season_idx] * x
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
            progressbar=True, target_accept=0.97,
        )
    return model, idata


def summarize(idata, features: pd.DataFrame, current_scale: float) -> pd.DataFrame:
    import arviz as az

    global_summary = az.summary(idata, var_names=["alpha_mu", "beta_mu", "alpha_sigma", "beta_sigma", "sigma"])
    print("\n--- Parámetros globales (partial pooling) ---")
    print(global_summary.to_string())

    per_season = az.summary(idata, var_names=["alpha", "beta"])
    per_season = per_season.reset_index().rename(columns={"index": "param"})

    beta_mu_mean = float(idata.posterior["beta_mu"].mean())
    beta_mu_hdi = az.hdi(idata, var_names=["beta_mu"])["beta_mu"].values

    print(f"\nConstante actual (hand-tuned, config/team_config.yaml): {current_scale}")
    print(f"beta_mu posterior (media): {beta_mu_mean:.4f}")
    print(f"beta_mu posterior (94% HDI): [{beta_mu_hdi[0]:.4f}, {beta_mu_hdi[1]:.4f}]")
    if current_scale < beta_mu_hdi[0] or current_scale > beta_mu_hdi[1]:
        print("-> La constante actual queda FUERA del 94% HDI del modelo bayesiano.")
    else:
        print("-> La constante actual cae DENTRO del 94% HDI -- el modelo no la contradice.")

    # R² del slope fijo actual vs. el slope posterior medio por temporada.
    seasons = sorted(features["season"].unique())
    beta_by_season = dict(zip(seasons, idata.posterior["beta"].mean(dim=("chain", "draw")).values))
    alpha_by_season = dict(zip(seasons, idata.posterior["alpha"].mean(dim=("chain", "draw")).values))

    y = features["y_actual_diff_points_pg"].to_numpy()
    x = features["x_game_score_vs_baseline"].to_numpy()
    pred_fixed = current_scale * x
    pred_hierarchical = np.array([
        alpha_by_season[s] + beta_by_season[s] * xi
        for s, xi in zip(features["season"], x)
    ])

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot

    print(f"\nR² con la constante fija actual ({current_scale}): {r2(y, pred_fixed):.4f}")
    print(f"R² con el modelo jerárquico (partial pooling): {r2(y, pred_hierarchical):.4f}")

    return per_season


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def run_loso_validation(
    features: pd.DataFrame, current_scale: float, draws: int, tune: int, chains: int, seed: int,
) -> pd.DataFrame:
    """
    Leave-one-season-out: entrena con 15 temporadas, predice la excluida
    usando solo los hiperparámetros globales (alpha_mu/beta_mu) -- la
    predicción correcta para un grupo sin datos propios. `pred_fixed` usa
    la misma fórmula que producción (x * game_score_to_net_rating_scale,
    sin intercepto) para comparación justa. Repite el ajuste MCMC 16
    veces; a diferencia del R² in-sample de `summarize`, aquí cada
    predicción es sobre datos nunca vistos por ese ajuste.
    """
    import pymc as pm  # noqa: F401 -- solo para que un fallo de import salga aquí, no a mitad del bucle

    seasons = sorted(features["season"].unique())
    fold_rows = []
    diagnostics = []

    for i, held_out in enumerate(seasons):
        train = features[features["season"] != held_out]
        test = features[features["season"] == held_out]

        print(f"\n[{i + 1}/{len(seasons)}] Entrenando sin {held_out} ({len(train)} filas)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            _, idata = fit_hierarchical_model(train, draws=draws, tune=tune, chains=chains, seed=seed)

        n_divergences = int(idata.sample_stats["diverging"].sum())
        rhat_beta_mu = float(pm_rhat(idata, "beta_mu"))
        diagnostics.append({"held_out_season": held_out, "divergences": n_divergences, "rhat_beta_mu": rhat_beta_mu})
        if n_divergences > 0:
            print(f"  [aviso] {n_divergences} divergencias en este pliegue -- resultado sospechoso, revisar.")

        alpha_mu_fold = float(idata.posterior["alpha_mu"].mean())
        beta_mu_fold = float(idata.posterior["beta_mu"].mean())

        x_test = test["x_game_score_vs_baseline"].to_numpy()
        y_test = test["y_actual_diff_points_pg"].to_numpy()
        pred_bayes = alpha_mu_fold + beta_mu_fold * x_test
        pred_fixed = current_scale * x_test  # misma fórmula que simulation.py, sin intercepto

        for name, x_i, y_i, pb, pf in zip(test["comparable_name"], x_test, y_test, pred_bayes, pred_fixed):
            fold_rows.append({
                "held_out_season": held_out,
                "comparable_name": name,
                "x": x_i,
                "y_actual": y_i,
                "pred_bayes_loso": pb,
                "pred_fixed_current": pf,
                "beta_mu_fold": beta_mu_fold,
                "alpha_mu_fold": alpha_mu_fold,
            })

    results = pd.DataFrame(fold_rows)
    diag_df = pd.DataFrame(diagnostics)

    print("\n" + "=" * 70)
    print("RESULTADO LOSO (out-of-fold, cada temporada predicha SIN haberla visto)")
    print("=" * 70)

    y_all = results["y_actual"].to_numpy()
    r2_bayes = _r2(y_all, results["pred_bayes_loso"].to_numpy())
    r2_fixed = _r2(y_all, results["pred_fixed_current"].to_numpy())
    mae_bayes = float(np.abs(y_all - results["pred_bayes_loso"]).mean())
    mae_fixed = float(np.abs(y_all - results["pred_fixed_current"]).mean())

    print(f"\nR²  -- constante fija actual ({current_scale}): {r2_fixed:.4f}")
    print(f"R²  -- bayesiano LOSO (fuera de muestra):        {r2_bayes:.4f}")
    print(f"MAE -- constante fija actual ({current_scale}): {mae_fixed:.4f} pts/partido")
    print(f"MAE -- bayesiano LOSO (fuera de muestra):        {mae_bayes:.4f} pts/partido")

    per_season = results.groupby("held_out_season").apply(
        lambda g: pd.Series({
            "r2_fixed": _r2(g["y_actual"].to_numpy(), g["pred_fixed_current"].to_numpy()),
            "r2_bayes_loso": _r2(g["y_actual"].to_numpy(), g["pred_bayes_loso"].to_numpy()),
            "beta_mu_fold": g["beta_mu_fold"].iloc[0],
        }),
        include_groups=False,
    )
    print("\nPor temporada (beta_mu aprendido SIN ver esa temporada):")
    print(per_season.to_string())

    n_divergent_folds = int((diag_df["divergences"] > 0).sum())
    print(f"\nPliegues con divergencias: {n_divergent_folds}/{len(seasons)}")
    if n_divergent_folds == 0:
        print("Los 16 ajustes salieron limpios -- el resultado es de fiar.")
    else:
        print("Hay pliegues con divergencias -- no te fíes del resultado hasta revisarlos.")

    results.attrs["diagnostics"] = diag_df
    return results, diag_df


def pm_rhat(idata, var_name: str) -> float:
    import arviz as az

    return float(az.rhat(idata, var_names=[var_name])[var_name].values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-features", action="store_true",
                         help="Recalcula el CSV de features en vez de usar el cacheado.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loso", action="store_true",
                         help="En vez del ajuste único, valida leave-one-season-out (16 ajustes, "
                              "cada temporada predicha sin haberla visto).")
    args = parser.parse_args()

    config = load_config()
    paths = get_paths(config)
    features_path = paths["processed"] / FEATURES_FILENAME

    if args.refresh_features or not features_path.exists():
        features = build_calibration_features(config)
        features.to_csv(features_path, index=False)
        print(f"Guardado: {features_path} ({len(features)} filas)")
    else:
        features = pd.read_csv(features_path)
        print(f"Usando features cacheadas: {features_path} ({len(features)} filas) "
              f"-- pasa --refresh-features para recalcular.")

    current_scale = config.get("monte_carlo", {}).get("game_score_to_net_rating_scale", 0.21)

    if args.loso:
        results, diagnostics = run_loso_validation(
            features, current_scale, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed,
        )
        results_path = paths["processed"] / LOSO_FILENAME
        results.to_csv(results_path, index=False)
        print(f"\nGuardado: {results_path}")
        diag_path = paths["processed"] / LOSO_DIAGNOSTICS_FILENAME
        diagnostics.to_csv(diag_path, index=False)
        print(f"Guardado: {diag_path}")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        _, idata = fit_hierarchical_model(
            features, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed,
        )

    per_season_summary = summarize(idata, features, current_scale)

    summary_path = paths["processed"] / SUMMARY_FILENAME
    per_season_summary.to_csv(summary_path, index=False)
    print(f"\nGuardado: {summary_path}")

    trace_path = paths["processed"] / TRACE_FILENAME
    idata.to_netcdf(trace_path)
    print(f"Guardado: {trace_path} (traza posterior completa, para inspección con arviz)")


if __name__ == "__main__":
    main()
