"""
game_win_predictor.py

EXPERIMENTO, no forma parte del pipeline de producción. Investiga si un
gradient boosted trees predice mejor la probabilidad de victoria de un
partido NBA individual que la logística fija de `src/simulation.py`
(`compute_game_net_rating_estimate` + `compute_win_probabilities`).

Datos: `backtest_sweep_advanced_game_logs.csv` (ya cacheado, sin ingesta
nueva), restringido a temporada regular. Features PREGAME sin look-ahead
(`shift(1)` + rolling sobre la secuencia cronológica del equipo):
net_rating_rolling, three_pt_rate_rolling (playstyle), pace_rolling, y
rest_days/is_back_to_back. Cada partido se representa como diferencial
local-visitante; la ventaja de casa queda absorbida por el intercepto,
igual que en producción. Baseline: logística de una sola variable
(net_rating_diff) con el mismo criterio de validación, para aislar el
efecto de ajustar a datos reales del efecto de las features extra.

Validación: leave-one-season-out sobre backtest_sweep, con Brier score,
log-loss y accuracy fuera de muestra.

Se auditaron dos hipótesis de calidad de datos: recortar el margen de
victoria (MOV cap) empeora el Brier de forma monótona, así que se
descartó; usar media EXPANDIDA del historial (en vez de ventana fija de
10) sí mejoró el Brier de 0.2230 a 0.2175 y quedó como default.

Resultado (LOSO, 18.843 partidos, home win rate real 57.2%): Brier
baseline 0.2175 vs GBT 0.2173, accuracy 65.36% vs 64.93% -- prácticamente
empatados. Conclusión: descanso/B2B/playstyle no aportan señal
incremental sobre el Net Rating reciente (ya la contiene), y la
logística de producción está cerca del techo de lo que estos datos
predicen (65% de accuracy, por debajo del ~67-70% de las casas de
apuestas, que sí usan lesiones de último minuto). No dice nada sobre
si `outcome_variance_scale` está bien calibrado (aplica sobre un roster
hipotético, no sobre net_rating_diff real) -- simulation.py sin tocar.

Nota aparte: más iteraciones de GBT (max_iter forzado sin early
stopping) EMPEORAN el Brier de forma monótona (0.2234 -> 0.2511) --
sobreajuste con solo seis features, confirma que el techo es de
información, no de entrenamiento.

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
# None -> media expandida (ver docstring del módulo); un entero fija una ventana móvil de ese tamaño.
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
    """Una fila por (equipo, partido) de temporada regular con las features PREGAME del docstring del módulo; descarta el primer partido de cada equipo (sin historial previo)."""
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
    """Empareja local/visitante de cada GAME_ID en una fila con features en diferencial y `home_win` como etiqueta (inner join, descarta partidos incompletos)."""
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
    """Leave-one-season-out: entrena baseline y GBT con todas menos una temporada, predice sobre la excluida; imprime Brier/log-loss/accuracy agregados."""
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
