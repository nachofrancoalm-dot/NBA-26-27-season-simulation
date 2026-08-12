"""
hustle_stats_signal.py

EXPERIMENTO, no forma parte del pipeline de producción. Investiga si los
hustle stats (CONTESTED_SHOTS, DEFLECTIONS, CHARGES_DRAWN,
SCREEN_ASSISTS, LOOSE_BALLS_RECOVERED, BOX_OUTS -- ver
data_pipeline.fetch_league_hustle_stats) aportan señal predictiva que ni
Game Score (puramente ofensivo) ni NET_RATING/PIE (ver
advanced_impact.py) capturan.

MOTIVACIÓN: la investigación anterior (ver CLAUDE.md, "No es un bug de
calibración") encontró que la dispersión comprimida de victorias entre
equipos es exactamente lo que predice la teoría de regresión lineal
dado el nivel de correlación actual (r=0.716, R²=0.513) entre Game
Score+NET_RATING y el diferencial de puntos real -- NO se puede arreglar
recalibrando más, hace falta una métrica que prediga MEJOR. Los hustle
stats son la métrica de "defensa sin balón" más rica que expone nba_api
y que este proyecto no había probado todavía.

LIMITACIÓN DE DATOS: la NBA solo trackea esto desde 2015-16 -- de las 16
temporadas del backtest sweep, solo 11 tienen hustle stats (330 casos de
480, no los 480 completos).

DISEÑO
------
1. Agrega hustle stats de `league_hustle_player_stats.csv` a nivel de
   EQUIPO por caso (suma ponderada por minutos de los jugadores del
   roster de ese caso, mismo criterio que Game Score/NET_RATING).
2. Reutiliza `x_game_score_vs_baseline` (la métrica compuesta YA
   calibrada, de bayesian_calibration.py) como línea base -- la pregunta
   es si añadir hustle mejora el R² MÁS ALLÁ de lo que ya aporta esa
   métrica, no si hustle solo predice bien.
3. Regresión múltiple (mínimos cuadrados, statsmodels) DiffPointsPG ~
   x_game_score_vs_baseline + hustle_score, comparado contra el modelo
   solo con x_game_score_vs_baseline -- ver si el R² sube de forma
   significativa (test F) y si el coeficiente de hustle es
   estadísticamente distinto de cero.

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
    """
    Una fila por caso (equipo-temporada) con las columnas de
    HUSTLE_COLUMNS agregadas a nivel de equipo (suma ponderada por
    minutos/partido de cada jugador del roster de ese caso -- el hustle
    stat de un suplente de garbage time no debe pesar igual que el de un
    titular).
    """
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
