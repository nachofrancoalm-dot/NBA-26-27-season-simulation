"""
game_win_predictor.py

EXPERIMENTO, no forma parte del pipeline de producción. Investiga si un
modelo entrenado (gradient boosted trees) predice mejor la probabilidad
de victoria de un partido NBA individual que la fórmula fija que ya usa
`src/simulation.py` (`compute_game_net_rating_estimate` +
`compute_win_probabilities`: una logística de mano sobre un diferencial
de rating, con `outcome_variance_scale` fijado a ojo/backtest agregado,
no ajustado partido a partido).

NO es un modelo de lenguaje ni se reutiliza nada de Hugging Face --
para datos tabulares de este tamaño (decenas de miles de partidos, media
docena de features) los árboles con boosting entrenados desde cero baten
de forma consistente a redes neuronales/transfer learning, y no existe
ningún modelo preentrenado cuyos pesos codifiquen algo transferible a
este esquema de columnas concreto.

DE DÓNDE SALEN LOS DATOS -- CERO INGESTA NUEVA
------------------------------------------------
`data/processed/backtest_sweep_advanced_game_logs.csv` (generado por
`data_pipeline.py --backtest-sweep`, ya cacheado) trae, temporada regular
Y playoffs, los 30 equipos NBA de cada temporada en
`config["backtest_sweep"]["seasons"]` -- por construcción de
`resolve_backtest_sweep_cases` (30 equipos x cada temporada), AMBOS lados
de cada partido de esas temporadas están presentes, así que se puede
reconstruir el partido completo (local + visitante) sin ninguna llamada
nueva a la API. Se restringe a `game_phase == "regular"` -- playoffs
tiene dinámica distinta (eliminación, series repetidas contra el mismo
rival) y se deja fuera del alcance de este primer experimento.

FEATURES -- SIN LOOK-AHEAD
-----------------------------
Por cada partido de cada equipo, se calculan métricas PREGAME (antes de
que se juegue ese partido) con `shift(1)` + `rolling(min_periods=1)`
sobre la propia secuencia cronológica del equipo esa temporada -- mismo
principio de no-look-ahead que `backtesting.filter_seasons_before` y
`injury_survival_model.build_survival_features`, aplicado a nivel de
partido en vez de temporada:

  - `net_rating_rolling`: media EXPANDIDA (todos los partidos previos de
    la temporada, no una ventana fija -- ver "CALIDAD DE DATOS" más abajo
    para el porqué) de `net_rating_estimate` -- reutiliza
    `context.performance_curve.compute_net_rating_estimate` TAL CUAL, no
    se reinventa la estimación de Net Rating por partido.
  - `three_pt_rate_rolling`: media móvil de FG3A/FGA -- proxy de
    "playstyle" (equipos que dependen del triple son más volátiles).
  - `pace_rolling`: media móvil de posesiones estimadas -- proxy de
    ritmo de juego.
  - `rest_days`: días desde el partido anterior del mismo equipo esa
    temporada (recortado a [0, 5] -- más de 5 días de descanso no aporta
    más "frescura" marginal, es prácticamente un parón de All-Star).
    `is_back_to_back` = rest_days <= 1.

El primer partido de temporada de cada equipo no tiene historial previo
-- se descarta (mismo criterio "sin evidencia, sin inventar" que
`injury_model.compute_risk_score`), lo que de paso elimina los partidos
de apertura donde AMBOS equipos jugarían su primer partido.

Cada partido se representa como diferencial LOCAL menos VISITANTE
(`net_rating_diff`, `rest_days_diff`, `three_pt_rate_diff`, `pace_diff`)
más `is_b2b_home`/`is_b2b_away` por separado (el efecto de un
back-to-back no es simétrico simplemente restando). La ventaja de jugar
en casa NO se pasa como feature explícita -- todas las filas están
siempre en perspectiva local/visitante, así que la queda absorbida por
el intercepto del modelo (logística) o por el propio árbol (GBT), igual
que en la fórmula de producción.

BASELINE DE COMPARACIÓN
--------------------------
No se compara contra un número fijo arbitrario -- se entrena una
regresión logística de UNA sola variable (`net_rating_diff`) con el
mismo criterio de validación que el modelo completo. Es la MISMA forma
funcional que usa `compute_win_probabilities` en producción (logística
sobre un diferencial de rating), pero con el "outcome_variance_scale"
AJUSTADO a los datos reales en vez de fijado a mano -- así la
comparación aísla el efecto de dos cosas por separado: (1) ajustar la
logística a datos reales, y (2) añadir las features de descanso/
playstyle vía un modelo no lineal (GBT).

VALIDACIÓN
------------
Leave-one-season-out sobre las temporadas de `backtest_sweep`: para cada
temporada excluida, se entrena con las demás y se predice sobre la
excluida. Métricas fuera de muestra fold a fold: Brier score (foco en
calibración de probabilidad, no solo acierto binario -- es lo que de
verdad usa el simulador aguas abajo), log-loss y accuracy.

CALIDAD DE DATOS -- DOS HIPÓTESIS DE PREPARACIÓN PROBADAS
-----------------------------------------------------------
Antes de aceptar el resultado negativo de más abajo, se auditó si el
problema era de calidad/preparación de los datos en vez de falta de
señal. Dos hipótesis, ambas medidas con el mismo LOSO:

1. Tope de margen de victoria (MOV cap). Los partidos con
   |net_rating_estimate| > 25 son ~8.5% del total, y algunos son
   extremos de verdad (la paliza de 55 puntos de Lakers-Cavaliers del
   11-ene-2011, el récord de 73 puntos Grizzlies-Thunder del 2-dic-2021
   -- datos reales, no errores de captura). La intuición de sistemas de
   rating conocidos (Sagarin, ELO con tope de margen) es que estos
   blowouts sobrerrepresentan "garbage time" y conviene recortarlos.
   Resultado: recortar el margen EMPEORA el Brier score de forma
   monótona según se aprieta el tope (sin tope 0.2230 -> cap=20 0.2232
   -> cap=15 0.2235 -> cap=10 0.2241). Descartado: para este problema
   concreto, un margen de 50 puntos SÍ es información real sobre la
   diferencia de calidad entre los dos equipos, no ruido a filtrar.

2. Tamaño de la ventana de historial. `net_rating_estimate` por partido
   tiene una desviación estándar de ~14.5 puntos (partido a partido) --
   muy ruidoso para estabilizarse con pocos partidos. Barrido de ventana
   fija (Brier score OOS): 3 partidos 0.2327 -> 5: 0.2284 -> 10: 0.2230
   -> 15: 0.2203 -> 20: 0.2187 -> 30: 0.2178 -> 41: 0.2175 -> 50: 0.2177
   (empeora ligerísimamente -- la señal ya se estabilizó). La media
   EXPANDIDA (todos los partidos previos de la temporada, sin ventana
   fija) da 0.2175 -- empata con la mejor ventana fija encontrada, sin
   necesitar elegir un número mágico. Adoptado como default de
   `build_team_game_features` (`rolling_window=None`): con ventana=10
   (el valor inicial, elegido sin barrer alternativas) se estaba
   entrenando con una estimación de fuerza de equipo más ruidosa de lo
   necesario, así que este ajuste sí corregía un problema real de
   preparación de datos.

RESULTADOS (LOSO sobre las 16 temporadas del backtest sweep, 18.843
partidos fuera de muestra, home win rate real 57.2%, con media expandida
tras la mejora de arriba):

    Brier score  -- baseline (logística, net_rating_diff):  0.2175
    Brier score  -- GBT (+ descanso/B2B/playstyle):         0.2173
    Log-loss     -- baseline: 0.6264   GBT: 0.6242
    Accuracy     -- baseline: 65.36%   GBT: 64.93%

CONCLUSIÓN: la mejora de calidad de datos (media expandida) SÍ funcionó
-- Brier score baja de 0.2230 a 0.2175 (~2.5% relativo), accuracy sube
de 63.4% a 65.4% para el baseline. Pero el resultado de fondo sobre GBT
vs. logística de una sola variable NO cambia: siguen prácticamente
empatados (0.2175 vs 0.2173 de Brier, diferencia muy por debajo de la
variación fold-a-fold entre temporadas). El descanso, el back-to-back y
el playstyle siguen sin aportar señal incremental una vez que se conoce
el Net Rating reciente del equipo -- plausible: un equipo con jugadores
lesionados o cansado ya tiene ese desgaste reflejado en su propio
`net_rating_rolling`, así que esas features casi no añaden información
nueva, la mayoría ya está contenida en el rating.

Esto SÍ valida algo, aunque el GBT no gane: la forma funcional que ya
usa `compute_win_probabilities` en producción (logística sobre un
diferencial de rating) no es una simplificación ingenua -- es
prácticamente el techo de lo que estos datos permiten predecir, incluso
con la mejor preparación de datos encontrada. El 65% de accuracy fuera
de muestra queda, como se anticipó, por debajo de lo que logran las
líneas de casas de apuestas (~67-70%) -- que sí incorporan lesiones de
última hora y lineups del día, datos que este proyecto deliberadamente
no ingiere (ver discusión de RAG en llm_explainer.py).

Nota aparte, fuera del alcance de este experimento: esto NO dice que
`outcome_variance_scale` de producción esté bien calibrado -- ese
parámetro se aplica sobre Game Score de un roster HIPOTÉTICO proyectado,
no sobre `net_rating_diff` real de equipos NBA, así que no son
comparables directamente sin una capa de traducción que no existe hoy.
Se deja `simulation.py` SIN TOCAR.

¿MÁS ITERACIONES DE ENTRENAMIENTO AYUDAN? -- NO, EMPEORAN
------------------------------------------------------
Medido con la versión anterior de las features (ventana=10 fija, antes
de adoptar la media expandida de arriba); la conclusión cualitativa (más
árboles sobreajustan, no ayudan) no depende de ese cambio posterior,
pero las cifras exactas de abajo no se han vuelto a medir tras pasar a
media expandida.

Pregunta relevante: ¿el techo de arriba es un techo de información o solo
falta de entrenamiento? `HistGradientBoostingClassifier` no tiene
"épocas" (eso es terminología de redes neuronales) -- el parámetro
análogo es `max_iter` (número de árboles). Con la config por defecto
(`early_stopping="auto"`, activo automáticamente con >10k filas) subir
`max_iter` de 100 a 1000 no cambia el Brier score ni un decimal: el
modelo ya para solo en la iteración 42 porque detecta que seguir no
mejora en su validación interna -- no estaba infra-entrenado.
Forzándolo a entrenar TODAS las iteraciones sin ese freno
(`early_stopping=False`), el resultado fuera de muestra EMPEORA de
forma monótona y clara:

    max_iter forzado=  42  Brier OOS 0.2234
    max_iter forzado= 300  Brier OOS 0.2321
    max_iter forzado=1000  Brier OOS 0.2511

Sobreajuste de manual: con solo seis features, más árboles no encuentran
señal nueva, memorizan ruido de las temporadas de entrenamiento. Esto
refuerza la conclusión de arriba en vez de contradecirla -- el techo es
de información (las features no dan más de sí), no de tiempo de
entrenamiento.

Uso:
    python scripts/experiments/game_win_predictor.py
    python scripts/experiments/game_win_predictor.py --refresh-features
    python scripts/experiments/game_win_predictor.py --rolling-window 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config  # noqa: E402
from context.performance_curve import compute_net_rating_estimate, estimate_possessions  # noqa: E402

FEATURES_FILENAME = "experiment_game_win_features.csv"
# None -> media EXPANDIDA (todos los partidos previos de la temporada, sin
# tamaño de ventana fijo) -- el barrido empírico documentado en el
# docstring del módulo mostró que esto empata con la mejor ventana fija
# encontrada (41 partidos) sin necesidad de un número mágico. Un entero
# fija una ventana móvil de ese tamaño en su lugar (para comparar).
DEFAULT_ROLLING_WINDOW_GAMES: Optional[int] = None
MAX_REST_DAYS = 5

GBT_FEATURE_COLUMNS = [
    "net_rating_diff",
    "rest_days_diff",
    "is_b2b_home",
    "is_b2b_away",
    "three_pt_rate_diff",
    "pace_diff",
]
BASELINE_FEATURE_COLUMNS = ["net_rating_diff"]


def build_team_game_features(
    advanced_game_logs: pd.DataFrame, rolling_window: Optional[int] = DEFAULT_ROLLING_WINDOW_GAMES
) -> pd.DataFrame:
    """
    Una fila por (equipo, partido) de temporada regular con las features
    PREGAME descritas en el docstring del módulo. Filas sin historial
    previo esa temporada (primer partido) se descartan -- ver docstring.

    `rolling_window=None` (por defecto): media EXPANDIDA -- usa TODOS los
    partidos previos de la temporada del equipo, no una ventana fija. Un
    entero usa una ventana móvil de ese tamaño en su lugar (ver el
    barrido empírico en el docstring del módulo: por debajo de ~30
    partidos la ventana fija es claramente peor, señal de que el Net
    Rating por partido es demasiado ruidoso -- std ≈ 14.5 puntos -- para
    estabilizarse con pocos partidos).
    """
    df = advanced_game_logs[advanced_game_logs["game_phase"] == "regular"].copy()
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = compute_net_rating_estimate(df)
    df["three_pt_rate"] = (df["FG3A"] / df["FGA"].replace(0, np.nan)).fillna(0.0)
    df["possessions_estimate"] = estimate_possessions(df)
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ")
    df["opponent_abbreviation"] = df["MATCHUP"].str.split(" ").str[-1]

    frames = []
    for (_team_id, _season), group in df.groupby(["TEAM_ID", "season"]):
        group = group.sort_values("GAME_DATE").reset_index(drop=True)

        for source_col, rolling_col in [
            ("net_rating_estimate", "net_rating_rolling"),
            ("three_pt_rate", "three_pt_rate_rolling"),
            ("possessions_estimate", "pace_rolling"),
        ]:
            shifted = group[source_col].shift(1)
            if rolling_window is None:
                group[rolling_col] = shifted.expanding(min_periods=1).mean()
            else:
                group[rolling_col] = shifted.rolling(window=rolling_window, min_periods=1).mean()

        group["rest_days"] = group["GAME_DATE"].diff().dt.days

        frames.append(group)

    result = pd.concat(frames, ignore_index=True)
    result["rest_days"] = result["rest_days"].clip(0, MAX_REST_DAYS)
    result["is_back_to_back"] = (result["rest_days"] <= 1).astype(int)

    required = ["net_rating_rolling", "three_pt_rate_rolling", "pace_rolling", "rest_days"]
    result = result.dropna(subset=required)
    return result


def build_matchup_dataset(team_game_features: pd.DataFrame) -> pd.DataFrame:
    """
    Empareja las dos filas (local, visitante) de cada GAME_ID en una
    única fila por partido, con features en diferencial local-visitante
    (ver docstring del módulo) y `home_win` como etiqueta. Un GAME_ID
    donde a alguno de los dos lados le falte historial previo (dropeado
    en build_team_game_features) simplemente no aparece -- inner join.
    """
    home = team_game_features[team_game_features["is_home"]]
    away = team_game_features[~team_game_features["is_home"]]

    cols = ["GAME_ID", "season", "WL", "net_rating_rolling", "rest_days", "is_back_to_back",
            "three_pt_rate_rolling", "pace_rolling"]
    merged = home[cols].merge(away[cols], on=["GAME_ID", "season"], suffixes=("_home", "_away"))

    matchups = pd.DataFrame({
        "GAME_ID": merged["GAME_ID"],
        "season": merged["season"],
        "home_win": (merged["WL_home"] == "W").astype(int),
        "net_rating_diff": merged["net_rating_rolling_home"] - merged["net_rating_rolling_away"],
        "rest_days_diff": merged["rest_days_home"] - merged["rest_days_away"],
        "is_b2b_home": merged["is_back_to_back_home"],
        "is_b2b_away": merged["is_back_to_back_away"],
        "three_pt_rate_diff": merged["three_pt_rate_rolling_home"] - merged["three_pt_rate_rolling_away"],
        "pace_diff": merged["pace_rolling_home"] - merged["pace_rolling_away"],
    })
    return matchups


def run_loso(matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-season-out: entrena baseline (logística, solo
    net_rating_diff) y GBT (todas las features) con todas las temporadas
    salvo una, predice probabilidad de victoria local sobre la excluida.
    Devuelve la tabla por temporada y también imprime el resumen
    agregado (Brier score, log-loss, accuracy fuera de muestra).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    seasons = sorted(matchups["season"].unique())
    rows = []
    oof_actual, oof_baseline, oof_gbt = [], [], []

    for held_out in seasons:
        train = matchups[matchups["season"] != held_out]
        test = matchups[matchups["season"] == held_out]
        if test.empty or train.empty:
            continue

        baseline = LogisticRegression()
        baseline.fit(train[BASELINE_FEATURE_COLUMNS], train["home_win"])
        pred_baseline = baseline.predict_proba(test[BASELINE_FEATURE_COLUMNS])[:, 1]

        gbt = HistGradientBoostingClassifier(random_state=0)
        gbt.fit(train[GBT_FEATURE_COLUMNS], train["home_win"])
        pred_gbt = gbt.predict_proba(test[GBT_FEATURE_COLUMNS])[:, 1]

        actual = test["home_win"].to_numpy()
        oof_actual.extend(actual.tolist())
        oof_baseline.extend(pred_baseline.tolist())
        oof_gbt.extend(pred_gbt.tolist())

        rows.append({
            "held_out_season": held_out,
            "n_games": len(test),
            "brier_baseline": brier_score_loss(actual, pred_baseline),
            "brier_gbt": brier_score_loss(actual, pred_gbt),
            "accuracy_baseline": accuracy_score(actual, pred_baseline >= 0.5),
            "accuracy_gbt": accuracy_score(actual, pred_gbt >= 0.5),
        })

    per_season = pd.DataFrame(rows)
    print(per_season.to_string(index=False))

    oof_actual_arr = np.array(oof_actual)
    oof_baseline_arr = np.array(oof_baseline)
    oof_gbt_arr = np.array(oof_gbt)

    print(f"\nPartidos fuera de muestra (todas las temporadas): {len(oof_actual_arr)}")
    print(f"Home win rate real: {oof_actual_arr.mean():.3f}")
    print(f"\nBrier score -- baseline (logística, net_rating_diff): {brier_score_loss(oof_actual_arr, oof_baseline_arr):.4f}")
    print(f"Brier score -- GBT (todas las features):               {brier_score_loss(oof_actual_arr, oof_gbt_arr):.4f}")
    print(f"\nLog-loss -- baseline: {log_loss(oof_actual_arr, oof_baseline_arr):.4f}")
    print(f"Log-loss -- GBT:      {log_loss(oof_actual_arr, oof_gbt_arr):.4f}")
    print(f"\nAccuracy -- baseline: {accuracy_score(oof_actual_arr, oof_baseline_arr >= 0.5):.4f}")
    print(f"Accuracy -- GBT:      {accuracy_score(oof_actual_arr, oof_gbt_arr >= 0.5):.4f}")

    return per_season


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument(
        "--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW_GAMES,
        help="Ventana móvil fija (nº de partidos). Por defecto (sin pasar nada) usa media "
             "EXPANDIDA -- todo el historial previo de la temporada, ver docstring del módulo.",
    )
    args = parser.parse_args()

    config = load_config()
    paths = get_paths(config)
    features_path = paths["processed"] / FEATURES_FILENAME

    if args.refresh_features or not features_path.exists():
        advanced_logs_path = paths["processed"] / "backtest_sweep_advanced_game_logs.csv"
        if not advanced_logs_path.exists():
            raise FileNotFoundError(
                f"No se encontró {advanced_logs_path}. Corre "
                "`python src/data_pipeline.py --backtest-sweep` primero."
            )
        advanced_logs = pd.read_csv(advanced_logs_path)
        team_game_features = build_team_game_features(advanced_logs, args.rolling_window)
        matchups = build_matchup_dataset(team_game_features)
        matchups.to_csv(features_path, index=False)
        print(f"Guardado: {features_path} ({len(matchups)} partidos)")
    else:
        matchups = pd.read_csv(features_path)
        print(f"Usando features cacheadas: {features_path} ({len(matchups)} partidos)")

    run_loso(matchups)


if __name__ == "__main__":
    main()
