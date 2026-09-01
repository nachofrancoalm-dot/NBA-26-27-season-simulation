"""
hustle_stats_signal.py

EXPERIMENTO, no forma parte del pipeline de producción. Investiga si los
hustle stats (CONTESTED_SHOTS, DEFLECTIONS, CHARGES_DRAWN,
SCREEN_ASSISTS, LOOSE_BALLS_RECOVERED, BOX_OUTS) aportan señal predictiva
que ni Game Score ni NET_RATING/PIE capturan.

Motivación: la correlación actual entre Game Score+NET_RATING y el
diferencial real (r=0.716, R²=0.513) ya explica la dispersión comprimida
de victorias -- no es un problema de calibración, hace falta una métrica
mejor. Solo 11 de las 16 temporadas del sweep tienen hustle stats
(trackeados desde 2015-16), 330 de 480 casos.

Método: agrega hustle stats a nivel de equipo (ponderado por minutos) y
compara, vía OLS, DiffPointsPG ~ x_game_score_vs_baseline + hustle_score
contra el modelo sin hustle (test F, significancia del coeficiente).

Uso:
    python scripts/experiments/hustle_stats_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config  # noqa: E402

HUSTLE_COLUMNS = [
    "CONTESTED_SHOTS", "DEFLECTIONS", "CHARGES_DRAWN",
    "SCREEN_ASSISTS", "LOOSE_BALLS_RECOVERED", "BOX_OUTS",
]


def build_team_hustle_features(config: dict) -> pd.DataFrame:
    """Una fila por caso (equipo-temporada) con HUSTLE_COLUMNS agregadas a nivel de equipo, ponderadas por minutos/partido de cada jugador."""
    paths = get_paths(config)
    rosters = pd.read_csv(paths["processed"] / "backtest_sweep_rosters.csv")
    hustle = pd.read_csv(paths["processed"] / "league_hustle_player_stats.csv")

    merged = rosters.merge(
        hustle[["PLAYER_ID", "season", "MIN", *HUSTLE_COLUMNS]],
        on=["PLAYER_ID", "season"], how="inner",
    )
    merged["weight"] = merged["MIN"]  # ponderación por minutos/partido reales

    rows = []
    for (name, season), group in merged.groupby(["comparable_name", "season"]):
        total_weight = group["weight"].sum()
        if total_weight <= 0:
            continue
        row = {"comparable_name": name, "season": season}
        for col in HUSTLE_COLUMNS:
            row[col] = float((group[col] * group["weight"]).sum() / total_weight)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    paths = get_paths(config)

    hustle_path = paths["processed"] / "league_hustle_player_stats.csv"
    if not hustle_path.exists():
        raise FileNotFoundError(
            f"No se encontró {hustle_path}. Corre "
            "`from data_pipeline import build_league_hustle_stats_dataset` primero."
        )

    hustle_features = build_team_hustle_features(config)
    print(f"Casos con hustle stats disponibles: {len(hustle_features)}")

    calibration_features = pd.read_csv(paths["processed"] / "experiment_bayesian_calibration_features.csv")
    df = hustle_features.merge(calibration_features, on=["comparable_name", "season"], how="inner")
    print(f"Casos tras cruzar con la métrica ya calibrada: {len(df)}")

    import statsmodels.api as sm

    y = df["y_actual_diff_points_pg"]
    x_base = sm.add_constant(df[["x_game_score_vs_baseline"]])
    model_base = sm.OLS(y, x_base).fit()

    print("\n--- Modelo base (solo Game Score + NET_RATING ya calibrado) ---")
    print(f"R² = {model_base.rsquared:.4f}")

    print("\n--- Cada hustle stat, añadida UNA a la vez ---")
    results = []
    for col in HUSTLE_COLUMNS:
        x_plus = sm.add_constant(df[["x_game_score_vs_baseline", col]])
        model_plus = sm.OLS(y, x_plus).fit()
        f_test = model_plus.compare_f_test(model_base)
        results.append({
            "hustle_stat": col,
            "r2_base": model_base.rsquared,
            "r2_with_stat": model_plus.rsquared,
            "delta_r2": model_plus.rsquared - model_base.rsquared,
            "coef": model_plus.params[col],
            "p_value_coef": model_plus.pvalues[col],
            "f_test_p_value": f_test[1],
        })
    results_df = pd.DataFrame(results).sort_values("delta_r2", ascending=False)
    print(results_df.to_string())

    print("\n--- Las 3 con mayor delta_r2 juntas ---")
    top3 = results_df.head(3)["hustle_stat"].tolist()
    x_top3 = sm.add_constant(df[["x_game_score_vs_baseline", *top3]])
    model_top3 = sm.OLS(y, x_top3).fit()
    f_test_top3 = model_top3.compare_f_test(model_base)
    print(f"Columnas: {top3}")
    print(f"R² = {model_top3.rsquared:.4f} (base: {model_base.rsquared:.4f}, "
          f"delta: {model_top3.rsquared - model_base.rsquared:.4f})")
    print(f"F-test p-value (vs. modelo base): {f_test_top3[1]:.4f}")

    out_path = paths["processed"] / "experiment_hustle_stats_signal.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
