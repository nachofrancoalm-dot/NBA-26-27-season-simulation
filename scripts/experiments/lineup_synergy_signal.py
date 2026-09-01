"""
lineup_synergy_signal.py

EXPERIMENTO, no forma parte del pipeline de producción. src/lineup_synergy.py
modela solo dos efectos (usage_clash, playmaking_spacing_synergy) con
pesos (0.05 y 0.02) puestos a mano y nunca validados. Comprueba si esos
dos efectos, calculados con tasas por-36 reales, predicen algo sobre el
NET_RATING real de una pareja cuando sí compartió cancha
(`leaguedashlineups`). Motivado por un caso concreto: Embiid+Maxey
(hipotético) da net_pair_score=-1.49 porque usage_clash (peso 0.05)
domina sobre su synergy alta (peso 0.02) -- ¿ese ratio 2.5x tiene apoyo
empírico?

A diferencia de pt_defend_signal.py, aquí perfil y resultado son de la
MISMA temporada (no hay tautología: el perfil por-36 no incorpora
NET_RATING de pareja) porque la pregunta es sobre la forma del modelo
(signo y magnitud relativa), no una predicción hacia el futuro.

Segunda fase (tras el resultado negativo de los dos efectos originales):
tres candidatos nuevos con datos de tracking (`leaguedashptstats`) en vez
de solo estadísticas de caja -- post_creator_synergy (poste + creador),
onball_offball_shooter_synergy (pull-up + catch-and-shoot), y
drive_interior_synergy (proxy de pick-and-roll: drives + BLK+DREB, ya que
nba_api no expone frecuencia real de bloqueo-y-continuación).

Uso:
    python scripts/experiments/lineup_synergy_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config  # noqa: E402
from lineup_synergy import (  # noqa: E402
    DEFAULT_USAGE_THRESHOLD,
    compute_playmaking_spacing_synergy,
    compute_style_profile,
    compute_usage_clash,
)
from season_utils import dedupe_traded_seasons  # noqa: E402

# Mínimo de minutos juntos para que el NET_RATING de la pareja sea fiable
# (percentil 25 real de leaguedashlineups 2-man ya está en ~250).
MIN_SHARED_MINUTES = 300.0


def build_player_style_profiles(config: dict) -> dict:
    """{(player_id, season): perfil de estilo}, por-36 desde backtest_sweep_player_career_stats.csv vía lineup_synergy.compute_style_profile(). dedupe_traded_seasons() colapsa filas 'TOT' de traspasos a mitad de temporada."""
    paths = get_paths(config)
    path = paths["processed"] / "backtest_sweep_player_career_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Corre `data_pipeline.py --backtest-sweep` primero.")

    stats = pd.read_csv(path)
    stats = dedupe_traded_seasons(stats)
    stats = stats[stats["MIN"] > 0]

    profiles = {}
    for _, row in stats.iterrows():
        minutes = float(row["MIN"])
        per36_row = pd.Series(
            {
                "FGA_per36_projected": row["FGA"] / minutes * 36.0,
                "FTA_per36_projected": row["FTA"] / minutes * 36.0,
                "TOV_per36_projected": row["TOV"] / minutes * 36.0,
                "AST_per36_projected": row["AST"] / minutes * 36.0,
                "FG3A_per36_projected": row["FG3A"] / minutes * 36.0,
                "BLK_per36_projected": row["BLK"] / minutes * 36.0,
                "DREB_per36_projected": row["DREB"] / minutes * 36.0,
            }
        )
        profiles[(int(row["PLAYER_ID"]), str(row["SEASON_ID"]))] = compute_style_profile(per36_row)
    return profiles


def build_tracking_style_features(config: dict) -> dict:
    """{(player_id, season): {post_volume, pullup_volume, catchshoot_volume, drive_volume}} por-36, desde league_tracking_stats.csv. Ya viene sin duplicados por traspaso, no hace falta dedupe."""
    paths = get_paths(config)
    path = paths["processed"] / "league_tracking_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Corre `data_pipeline.build_league_tracking_stats_dataset` primero.")

    tracking = pd.read_csv(path)
    tracking = tracking[tracking["MIN"] > 0]

    def _safe(value: float) -> float:
        # `value or 0.0` no basta: NaN es truthy en Python, se colaría sin convertir a 0.0.
        return 0.0 if pd.isna(value) else float(value)

    features = {}
    for _, row in tracking.iterrows():
        minutes = float(row["MIN"])
        features[(int(row["PLAYER_ID"]), str(row["season"]))] = {
            "post_volume": _safe(row.get("POST_TOUCH_FGA")) / minutes * 36.0,
            "pullup_volume": _safe(row.get("PULL_UP_FGA")) / minutes * 36.0,
            "catchshoot_volume": _safe(row.get("CATCH_SHOOT_FGA")) / minutes * 36.0,
            "drive_volume": _safe(row.get("DRIVES")) / minutes * 36.0,
        }
    return features


def _parse_group_id(group_id: str) -> tuple:
    """'-1628404-1628969-' -> (1628404, 1628969)."""
    parts = [p for p in group_id.split("-") if p]
    return int(parts[0]), int(parts[1])


def build_pair_dataset(config: dict, min_shared_minutes: float = MIN_SHARED_MINUTES) -> pd.DataFrame:
    """Una fila por pareja real (>= min_shared_minutes juntos) con los 2 efectos originales + 3 candidatos nuevos y el NET_RATING real. Sin datos de tracking, el candidato queda en 0 en vez de descartar la fila (0 intentos es información válida, no ausencia)."""
    paths = get_paths(config)
    lineups_path = paths["processed"] / "league_2man_lineups.csv"
    if not lineups_path.exists():
        raise FileNotFoundError(
            f"No se encontró {lineups_path}. Corre "
            "`from data_pipeline import build_league_2man_lineup_dataset` primero."
        )
    lineups = pd.read_csv(lineups_path)
    lineups = lineups[lineups["MIN"] >= min_shared_minutes]

    profiles = build_player_style_profiles(config)
    tracking = build_tracking_style_features(config)

    rows = []
    for _, row in lineups.iterrows():
        season = str(row["season"])
        try:
            id_a, id_b = _parse_group_id(str(row["GROUP_ID"]))
        except (ValueError, IndexError):
            continue
        profile_a = profiles.get((id_a, season))
        profile_b = profiles.get((id_b, season))
        if profile_a is None or profile_b is None:
            continue

        clash = compute_usage_clash(profile_a["usage"], profile_b["usage"], DEFAULT_USAGE_THRESHOLD)
        synergy = compute_playmaking_spacing_synergy(
            profile_a["playmaking"], profile_a["spacing"], profile_b["playmaking"], profile_b["spacing"]
        )

        track_a = tracking.get((id_a, season), {"post_volume": 0.0, "pullup_volume": 0.0, "catchshoot_volume": 0.0, "drive_volume": 0.0})
        track_b = tracking.get((id_b, season), {"post_volume": 0.0, "pullup_volume": 0.0, "catchshoot_volume": 0.0, "drive_volume": 0.0})
        post_creator = track_a["post_volume"] * profile_b["playmaking"] + track_b["post_volume"] * profile_a["playmaking"]
        onball_offball = track_a["pullup_volume"] * track_b["catchshoot_volume"] + track_b["pullup_volume"] * track_a["catchshoot_volume"]
        drive_interior = track_a["drive_volume"] * profile_b["interior"] + track_b["drive_volume"] * profile_a["interior"]

        rows.append(
            {
                "season": season,
                "group_name": row.get("GROUP_NAME"),
                "min_together": row["MIN"],
                "net_rating": row["NET_RATING"],
                "usage_clash": clash,
                "playmaking_spacing_synergy": synergy,
                "post_creator_synergy": post_creator,
                "onball_offball_shooter_synergy": onball_offball,
                "drive_interior_synergy": drive_interior,
            }
        )
    return pd.DataFrame(rows)


# Signo esperado de cada candidato (positivo = ayuda, negativo = perjudica).
EXPECTED_SIGNS = {
    "usage_clash": -1,
    "playmaking_spacing_synergy": 1,
    "post_creator_synergy": 1,
    "onball_offball_shooter_synergy": 1,
    "drive_interior_synergy": 1,
}


def run_regression_and_loso(df: pd.DataFrame, feature_cols: list, label: str):
    """OLS sobre todas las temporadas + leave-one-season-out, para un subconjunto de columnas de `df`. Imprime el resumen; devuelve (model, loso_df)."""
    import statsmodels.api as sm

    y = df["net_rating"]
    X = sm.add_constant(df[feature_cols])
    model = sm.OLS(y, X).fit()
    print(f"\n=== {label} ===")
    print(model.summary().tables[1])
    print(f"R²: {model.rsquared:.4f}")

    loso_rows = []
    for held_out in sorted(df["season"].unique()):
        train = df[df["season"] != held_out]
        test = df[df["season"] == held_out]
        if len(test) < 5:
            continue
        X_train = sm.add_constant(train[feature_cols])
        fold_model = sm.OLS(train["net_rating"], X_train).fit()
        X_test = sm.add_constant(test[feature_cols], has_constant="add")
        pred = fold_model.predict(X_test)
        mae = float(np.abs(pred - test["net_rating"]).mean())
        corr = float(np.corrcoef(pred, test["net_rating"])[0, 1]) if test["net_rating"].std() > 0 else float("nan")
        row = {"held_out_season": held_out, "test_mae": mae, "test_corr": corr, "n_test": len(test)}
        for col in feature_cols:
            row[f"coef_{col}"] = fold_model.params[col]
        loso_rows.append(row)
    loso_df = pd.DataFrame(loso_rows)

    for col in feature_cols:
        expected = EXPECTED_SIGNS.get(col, 0)
        correct_sign = ((loso_df[f"coef_{col}"] * expected) > 0).sum()
        print(f"  {col}: signo correcto en {correct_sign}/{len(loso_df)} pliegues LOSO")
    print(f"  MAE medio fuera de muestra: {loso_df['test_mae'].mean():.3f} | correlación media: {loso_df['test_corr'].mean():.3f}")
    return model, loso_df


def main() -> None:
    config = load_config()
    paths = get_paths(config)

    lineups_path = paths["processed"] / "league_2man_lineups.csv"
    if not lineups_path.exists():
        print("league_2man_lineups.csv no existe -- descargando (16 temporadas, ~16 llamadas a la API)...")
        from data_pipeline import build_league_2man_lineup_dataset

        build_league_2man_lineup_dataset(config)

    tracking_path = paths["processed"] / "league_tracking_stats.csv"
    if not tracking_path.exists():
        print("league_tracking_stats.csv no existe -- descargando (4 categorías x 16 temporadas)...")
        from data_pipeline import build_league_tracking_stats_dataset

        build_league_tracking_stats_dataset(config)

    df = build_pair_dataset(config)
    print(f"Parejas con >= {MIN_SHARED_MINUTES:.0f} min juntos y perfil de ambos disponible: {len(df)}")
    print(f"Temporadas cubiertas: {sorted(df['season'].unique())}")

    features_path = paths["processed"] / "experiment_lineup_synergy_features.csv"
    df.to_csv(features_path, index=False)
    print(f"Guardado: {features_path}")

    # 1) Los dos efectos originales.
    run_regression_and_loso(df, ["usage_clash", "playmaking_spacing_synergy"], "Efectos originales")

    # 2) Cada candidato nuevo, solo.
    for col in ["post_creator_synergy", "onball_offball_shooter_synergy", "drive_interior_synergy"]:
        run_regression_and_loso(df, [col], f"Candidato nuevo: {col}")

    # 3) Los 5 juntos.
    all_cols = [
        "usage_clash", "playmaking_spacing_synergy",
        "post_creator_synergy", "onball_offball_shooter_synergy", "drive_interior_synergy",
    ]
    _, loso_all = run_regression_and_loso(df, all_cols, "Los 5 efectos juntos")
    loso_path = paths["processed"] / "experiment_lineup_synergy_loso.csv"
    loso_all.to_csv(loso_path, index=False)
    print(f"\nGuardado: {loso_path}")


if __name__ == "__main__":
    main()
