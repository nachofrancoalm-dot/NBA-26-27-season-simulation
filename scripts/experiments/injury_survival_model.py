"""
injury_survival_model.py

EXPERIMENTO, no forma parte del pipeline de producción. Investiga si un
modelo de supervivencia (Cox proportional hazards, `lifelines`) predice
mejor el riesgo de lesión que el heurístico actual de
`src/context/injury_model.py` (media ponderada a mano: historical_load=
0.45, recency=0.35, age=0.20, nunca ajustados a los datos del proyecto).

RESULTADO: negativo, bien validado.

Reutiliza las mismas features del heurístico (`compute_historical_load`/
`compute_recency_score`/`compute_age_score`, sobre temporadas anteriores
a la objetivo, sin look-ahead) para preguntar si los pesos 0.45/0.35/0.20
son correctos o los datos dicen otra cosa. `historical_load_prior` y
`recency_prior` correlacionan r=0.974 (casi la misma señal dos veces) y
producen un coeficiente de Cox negativo e inestable para
historical_load_prior, así que se excluye del modelo, dejando solo
recency_prior.

nba_api no expone fecha de lesión, solo GP por temporada -- se modela
cada temporada-jugador como supervivencia con duration=GP y event=1 si
GP < duración de temporada (censurado si GP == duración completa).

Validación: leave-one-season-out sobre las 16 temporadas del sweep
(6.784 observaciones), prediciendo GP esperado y comparando contra el
heurístico en las mismas unidades (partidos).

Resultado (MAE en partidos, out-of-fold): heurístico actual 16.96 (corr
0.503) vs. Cox recency+age 18.02 (corr 0.493), OLS reajustado 17.23
(corr 0.509), Cox + minutos previos 17.79 (corr 0.513). Ninguna
alternativa mejora el MAE, aunque la correlación sube ligeramente en dos
variantes. Conclusión: el heurístico ya está razonablemente calibrado
para este problema (señal limitada, los tres componentes ya capturan la
mayor parte); se deja injury_model.py sin tocar.

Uso:
    python scripts/experiments/injury_survival_model.py
    python scripts/experiments/injury_survival_model.py --with-workload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config  # noqa: E402
from backtesting import filter_seasons_before  # noqa: E402
from context.injury_model import (  # noqa: E402
    DEFAULT_AGE_CURVE,
    DEFAULT_N_SEASONS_LOOKBACK,
    DEFAULT_RECENCY_HALF_LIFE_SEASONS,
    DEFAULT_WEIGHTS,
    compute_age_score,
    compute_historical_load,
    compute_recency_score,
    season_length,
)
from season_utils import season_start_year  # noqa: E402

FEATURES_FILENAME = "experiment_injury_survival_features.csv"


def build_survival_features(
    player_career_stats: pd.DataFrame,
    target_seasons: List[str],
    n_seasons_lookback: int = DEFAULT_N_SEASONS_LOOKBACK,
    half_life: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
) -> pd.DataFrame:
    """Una fila por (jugador, temporada objetivo) con al menos una temporada previa registrada. `duration`/`event` son el par de supervivencia; el resto son las features del heurístico, calculadas solo con temporadas anteriores. `mpg_prior` es una feature extra no usada por el heurístico."""
    rows = []
    for player_id, player_group in player_career_stats.groupby("PLAYER_ID"):
        for target_season in target_seasons:
            target_year = season_start_year(target_season)
            prior = filter_seasons_before(player_group, target_year)
            if prior.empty:
                continue

            target_row = player_group[player_group["SEASON_ID"] == target_season]
            if target_row.empty:
                continue  # no jugó esa temporada
            target_row = target_row.iloc[0]

            length = season_length(target_season)
            gp = float(target_row["GP"])
            age = float(target_row["PLAYER_AGE"])

            prior_sorted = prior.assign(_y=prior["SEASON_ID"].apply(season_start_year)).sort_values(
                "_y", ascending=False
            )
            most_recent_prior = prior_sorted.iloc[0]
            mpg_prior = (
                float(most_recent_prior["MIN"]) / float(most_recent_prior["GP"])
                if most_recent_prior["GP"] > 0 else 0.0
            )

            rows.append({
                "player_id": player_id,
                "season": target_season,
                "duration": min(gp, length),
                "event": int(gp < length),
                "historical_load_prior": compute_historical_load(prior, n_seasons_lookback),
                "recency_prior": compute_recency_score(prior, n_seasons_lookback, half_life),
                "age_score": compute_age_score(age, DEFAULT_AGE_CURVE),
                "mpg_prior": mpg_prior,
                "age": age,
                "season_length": length,
            })
    return pd.DataFrame(rows)


def heuristic_expected_games(features: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    """GP esperado del heurístico actual (pesos fijos), en las mismas unidades que Cox/OLS (partidos, no risk_score)."""
    risk = (
        weights["historical_load"] * features["historical_load_prior"]
        + weights["recency"] * features["recency_prior"]
        + weights["age"] * features["age_score"]
    ).clip(0.0, 1.0)
    return (features["season_length"] * (1 - risk)).to_numpy()


def run_loso(features: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    """Leave-one-season-out: entrena Cox con todas menos una temporada, predice GP esperado sobre la excluida, y compara contra el heurístico actual en las mismas unidades."""
    from lifelines import CoxPHFitter

    seasons = sorted(features["season"].unique())
    rows = []
    oof_cox, oof_heuristic, oof_actual = [], [], []

    for held_out in seasons:
        train = features[features["season"] != held_out]
        test = features[features["season"] == held_out]
        if test.empty or train.empty:
            continue

        cph = CoxPHFitter()
        cph.fit(train[["duration", "event", *covariates]], duration_col="duration", event_col="event")

        pred_cox = cph.predict_expectation(test[covariates]).to_numpy().ravel()
        pred_cox = np.minimum(pred_cox, test["season_length"].to_numpy())
        pred_heuristic = heuristic_expected_games(test, DEFAULT_WEIGHTS)
        actual_gp = test["duration"].to_numpy()

        oof_cox.extend(pred_cox.tolist())
        oof_heuristic.extend(pred_heuristic.tolist())
        oof_actual.extend(actual_gp.tolist())

        rows.append({
            "held_out_season": held_out,
            "n_cases": len(test),
            **{f"coef_{c}": cph.params_[c] for c in covariates},
        })

    per_season = pd.DataFrame(rows)
    print(per_season.to_string())

    oof_actual, oof_cox, oof_heuristic = np.array(oof_actual), np.array(oof_cox), np.array(oof_heuristic)
    mae_cox = np.abs(oof_actual - oof_cox).mean()
    mae_heuristic = np.abs(oof_actual - oof_heuristic).mean()
    corr_cox = np.corrcoef(oof_actual, oof_cox)[0, 1]
    corr_heuristic = np.corrcoef(oof_actual, oof_heuristic)[0, 1]

    print(f"\nMAE (partidos) -- heurístico actual: {mae_heuristic:.3f}")
    print(f"MAE (partidos) -- Cox ({'+'.join(covariates)}): {mae_cox:.3f}")
    print(f"Correlación -- heurístico actual: {corr_heuristic:.4f}")
    print(f"Correlación -- Cox: {corr_cox:.4f}")
    return per_season


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--with-workload", action="store_true",
                         help="Añade mpg_prior (minutos/partido previos) como covariable extra.")
    args = parser.parse_args()

    config = load_config()
    paths = get_paths(config)
    features_path = paths["processed"] / FEATURES_FILENAME

    if args.refresh_features or not features_path.exists():
        player_career_stats = pd.read_csv(paths["processed"] / "backtest_sweep_player_career_stats.csv")
        target_seasons = config["backtest_sweep"]["seasons"]
        features = build_survival_features(player_career_stats, target_seasons)
        features.to_csv(features_path, index=False)
        print(f"Guardado: {features_path} ({len(features)} filas)")
    else:
        features = pd.read_csv(features_path)
        print(f"Usando features cacheadas: {features_path} ({len(features)} filas)")

    covariates = ["recency_prior", "age_score"]
    if args.with_workload:
        covariates.append("mpg_prior")
    run_loso(features, covariates)


if __name__ == "__main__":
    main()
