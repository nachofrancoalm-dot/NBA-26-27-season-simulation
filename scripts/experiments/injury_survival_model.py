"""
injury_survival_model.py

EXPERIMENTO, no forma parte del pipeline de producción. Investiga si un
modelo de supervivencia (Cox proportional hazards, `lifelines`) predice
mejor el riesgo de lesión de un jugador que el heurístico actual de
`src/context/injury_model.py` (media ponderada a mano de tres
componentes: historical_load=0.45, recency=0.35, age=0.20 -- pesos de
literatura epidemiológica general, nunca ajustados a los datos de ESTE
proyecto).

RESULTADO: NEGATIVO, bien validado -- ver "CONCLUSIÓN" al final de este
docstring. Documentado igual que scripts/experiments/aging_curve_shrinkage.py
y hustle_stats_signal.py: no todas las investigaciones tienen que ganar,
pero sí tienen que quedar registradas con el mismo rigor que las que sí
aportaron algo.

MISMAS COVARIABLES DE PARTIDA, PESOS APRENDIDOS EN VEZ DE FIJADOS A MANO
-------------------------------------------------------------------------
No se reinventan las features -- se reutilizan
`injury_model.compute_historical_load`/`compute_recency_score`/
`compute_age_score` TAL CUAL sobre las temporadas ANTERIORES a la que se
predice (mismo patrón de no-look-ahead que `backtesting.filter_seasons_before`).
La pregunta que responde este experimento es específicamente: "¿son
0.45/0.35/0.20 los pesos correctos, o los datos dicen otra cosa?".

BUG DE COLINEALIDAD encontrado y corregido en el camino (MISMO patrón
que PIE/NET_RATING en advanced_impact.py): `historical_load_prior`
(media simple de partidos perdidos) y `recency_prior` (la misma media,
ponderada por recencia) correlacionan **r=0.974** entre sí -- casi la
misma señal calculada dos veces. El primer ajuste de Cox con las 3
covariables originales daba un coeficiente NEGATIVO y estable para
`historical_load_prior` (más historial de lesiones -> MENOS riesgo, al
revés de lo esperado) en los 16 pliegues -- el mismo artefacto de
colinealidad que ya hizo descartar PIE. Quitar `historical_load_prior`
del modelo (dejando solo `recency_prior`, más motivada por la
literatura) estabiliza los coeficientes sin cambiar el resultado de
fondo.

CÓMO SE ENCAJA LA SUPERVIVENCIA EN DATOS QUE SOLO TIENEN GP AGREGADO
-----------------------------------------------------------------------
nba_api no expone fecha de lesión (ver el docstring de injury_model.py)
-- solo GP (partidos jugados) por temporada. Esto no impide un modelo de
supervivencia real: se trata cada temporada de un jugador como una
observación con "tiempo de supervivencia" = GP (partidos jugados antes
de que la temporada termine para él) y "evento" = 1 si GP < duración de
la temporada (algo le impidió completar el calendario -- lesión, DNP-CD,
etc.), 0 si GP == duración de la temporada (CENSURADO -- la temporada
terminó sin observar el evento, no significa "cero riesgo", solo que no
se observó dentro del calendario). Es el mismo diseño que usan estudios
reales de epidemiología deportiva con "partidos de una temporada" como
ventana de observación. Limitación real, ya heredada del heurístico
actual: no se distingue una racha de 20 partidos perdidos y luego 62
jugados de 62 jugados y luego 20 perdidos -- ninguno de los dos modelos
tiene datos para hacerlo.

VALIDACIÓN
------------
Leave-one-season-out sobre las 16 temporadas del backtest sweep (6.784
observaciones jugador-temporada, `backtest_sweep_player_career_stats.csv`
-- mucho más rico que las 13 filas del roster propio). Para cada
temporada excluida, se entrena con todas las demás y se predice GP
esperado (`CoxPHFitter.predict_expectation`, el tiempo medio de
supervivencia restringido -- directamente comparable a
`games_per_season * (1 - risk_score)` del heurístico actual) sobre la
temporada excluida, comparado contra el GP REAL de esa temporada.

RESULTADOS (MAE en partidos, sobre las 6.784 observaciones fuera de
muestra):
    heurístico actual (0.45/0.35/0.20)                MAE 16.96  corr 0.503
    Cox, recency+age (sin la señal colineal)          MAE 18.02  corr 0.493
    OLS, mismas 2 features, pesos reajustados         MAE 17.23  corr 0.509
    Cox, + minutos/partido de la temporada previa     MAE 17.79  corr 0.513

CONCLUSIÓN: ninguna alternativa supera claramente al heurístico actual.
La correlación mejora ligeramente en dos variantes (+0.006, +0.010) pero
el error absoluto medio EMPEORA en las tres -- no hay una mejora limpia
en ningún sentido. Añadir carga de minutos (una señal genuinamente nueva,
no solo un reajuste de peso) tampoco ayuda -- su coeficiente sale
negativo y estable (jugadores con más minutos/partido juegan MÁS
partidos la temporada siguiente), que probablemente refleja selección
(los entrenadores dan minutos pesados a jugadores que ya saben que son
duraderos) más que un efecto causal de "más carga = más riesgo".

**El heurístico actual ya está razonablemente bien calibrado para este
problema concreto** -- no porque sea sofisticado, sino porque el
problema (predecir partidos perdidos el año que viene a partir del
historial reciente y la edad) tiene señal limitada y los tres
componentes que ya usa capturan la mayor parte de lo que hay que
capturar. Se deja `injury_model.py` SIN TOCAR.

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
    """
    Una fila por (jugador, temporada objetivo) para cada temporada en
    `target_seasons` en la que el jugador tiene AL MENOS una temporada
    previa registrada (sin historial previo no hay covariables que
    calcular -- mismo principio de "sin evidencia, sin inventar" que
    `compute_risk_score`). `duration`/`event` son el par de
    supervivencia; `historical_load_prior`/`recency_prior`/`age_score`
    son exactamente las mismas features que ya usa el heurístico,
    calculadas SOLO con temporadas anteriores a la objetivo.
    `mpg_prior`: minutos/partido de la temporada previa MÁS RECIENTE --
    señal de carga de trabajo, no usada por el heurístico actual.
    """
    rows = []
    for player_id, player_group in player_career_stats.groupby("PLAYER_ID"):
        for target_season in target_seasons:
            target_year = season_start_year(target_season)
            prior = filter_seasons_before(player_group, target_year)
            if prior.empty:
                continue

            target_row = player_group[player_group["SEASON_ID"] == target_season]
            if target_row.empty:
                continue  # el jugador no jugó esa temporada -- no hay duración que observar
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
    """GP esperado del heurístico ACTUAL (pesos fijos), para comparar
    contra el Cox/OLS en las mismas unidades (partidos, no risk_score)."""
    risk = (
        weights["historical_load"] * features["historical_load_prior"]
        + weights["recency"] * features["recency_prior"]
        + weights["age"] * features["age_score"]
    ).clip(0.0, 1.0)
    return (features["season_length"] * (1 - risk)).to_numpy()


def run_loso(features: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    """
    Leave-one-season-out: entrena Cox con todas las temporadas salvo una
    y predice GP esperado sobre la excluida, comparado contra el
    heurístico actual EN LAS MISMAS UNIDADES (partidos).
    `historical_load_prior` se excluye de `covariates` por defecto --
    colineal con `recency_prior` (r=0.974, ver docstring del módulo) --
    pero sigue haciendo falta en `features` porque `heuristic_expected_games`
    la usa para la comparación contra el heurístico real.
    """
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
