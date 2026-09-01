"""
game_win_predictor_injury_signal.py

Seguimiento de game_win_predictor.py (GBT con seis features trailing no
mejoraba la logística baseline). Hipótesis: falta una señal de
disponibilidad REAL del día del partido. Para cada equipo-temporada, se
toman los 5 jugadores de más MPG ("jugadores clave") y se cuenta,
partido a partido, cuántos NO jugaron.

Esto mide disponibilidad RETROSPECTIVA (oráculo), no un injury report del
día del partido (nba_api no lo expone) -- por tanto no es desplegable tal
cual, pero mide el TECHO: si ni el oráculo ayuda, no compensa ingerir
injury reports en vivo.

Requiere ingesta nueva: `fetch_player_game_log` por (player, season) de
cada jugador clave (~2.250 llamadas para el sweep completo). Empieza en
modo piloto (`--seasons`, 3 temporadas por defecto) antes del sweep
completo (16 temporadas).

Método: cuenta jugadores clave ausentes por partido (`missing_key_players`)
y añade el diferencial local-visitante como séptima feature al GBT de
game_win_predictor.py, mismo LOSO.

Resultado del oráculo (sweep completo, 18.843 partidos): Brier sin señal
0.2173 vs con señal 0.2152, accuracy 64.93% vs 65.42% -- primera mejora
real de la serie, y bate al baseline logístico (0.2175).

Versión realista y desplegable (`compute_recently_missing_key_players`):
en vez del oráculo, cuenta ausencias en cualquiera de los 3 partidos
ANTERIORES (información pregame real). Conserva ~55% de la mejora de
Brier del oráculo (0.2146 vs 0.2132 vs 0.2163 sin señal) e iguala su
accuracy (65.78% vs 65.70%) sin depender de datos en vivo -- confirma que
la tendencia reciente de ausencias añade información que
net_rating_rolling no captura completamente (reacciona lento a una
lesión reciente).

No se ha llevado a producción: simulation.py proyecta un roster
hipotético, no un equipo real con historial verificable -- misma
limitación de traducción que outcome_variance_scale en
game_win_predictor.py.

Uso:
    python scripts/experiments/game_win_predictor_injury_signal.py
    python scripts/experiments/game_win_predictor_injury_signal.py --seasons 2010-11 2011-12 ... (sweep completo)
    python scripts/experiments/game_win_predictor_injury_signal.py --top-n 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from config_loader import get_paths, load_config  # noqa: E402
from data_pipeline import fetch_player_game_log  # noqa: E402
import game_win_predictor as gwp  # noqa: E402

DEFAULT_PILOT_SEASONS = ["2023-24", "2024-25", "2025-26"]
DEFAULT_TOP_N = 5
KEY_PLAYERS_FILENAME = "experiment_injury_signal_key_players.csv"
GAME_LOGS_FILENAME = "experiment_injury_signal_key_player_game_logs.csv"


def select_key_players(rosters: pd.DataFrame, career_stats: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Une roster con la fila de career stats de la temporada exacta, calcula MPG y devuelve el top-N por equipo-temporada (excluye GP=0)."""
    merged = rosters.merge(career_stats, on="PLAYER_ID")
    merged = merged[merged["SEASON_ID"] == merged["season"]]
    merged = merged[merged["GP"] > 0].copy()
    merged["mpg"] = merged["MIN"] / merged["GP"]

    key_players = (
        merged.sort_values("mpg", ascending=False)
        .groupby(["TeamID", "season"])
        .head(top_n)
        [["TeamID", "season", "PLAYER_ID", "player_name", "mpg"]]
        .reset_index(drop=True)
    )
    return key_players


def fetch_key_player_game_logs(key_players: pd.DataFrame, raw_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    """Una llamada por (PLAYER_ID, season) único, cacheada por fetch_player_game_log."""
    unique_pairs = key_players[["PLAYER_ID", "season"]].drop_duplicates()
    frames = []
    for _, row in tqdm(list(unique_pairs.iterrows()), desc="Descargando game logs de jugadores clave"):
        player_id, season = int(row["PLAYER_ID"]), row["season"]
        try:
            log = fetch_player_game_log(player_id, season, raw_dir, force_refresh)
        except Exception as e:  # noqa: BLE001 -- un jugador roto no debe abortar el resto de la ingesta
            print(f"  [omitido] player_id={player_id} season={season}: {e}")
            continue
        if log.empty:
            continue
        log = log.copy()
        log["season"] = season
        frames.append(log)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_missing_key_players(
    team_game_features: pd.DataFrame, key_players: pd.DataFrame, key_player_game_logs: pd.DataFrame
) -> pd.Series:
    """Por partido, cuenta jugadores clave sin fila en el game log de esa fecha (disponibilidad real). NaN, no 0, si el equipo-temporada no se descargó."""
    logs = key_player_game_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    played = set(zip(logs["Player_ID"], logs["season"], logs["GAME_DATE"]))

    key_players_by_team_season = key_players.groupby(["TeamID", "season"])["PLAYER_ID"].apply(list).to_dict()

    counts = []
    for _, row in team_game_features.iterrows():
        roster_key = (row["TEAM_ID"], row["season"])
        key_ids = key_players_by_team_season.get(roster_key)
        if key_ids is None:
            counts.append(np.nan)
            continue
        missing = sum(1 for pid in key_ids if (pid, row["season"], row["GAME_DATE"]) not in played)
        counts.append(missing)
    return pd.Series(counts, index=team_game_features.index)


DEFAULT_RECENT_WINDOW_GAMES = 3


def compute_recently_missing_key_players(
    team_game_features: pd.DataFrame,
    key_players: pd.DataFrame,
    key_player_game_logs: pd.DataFrame,
    window: int = DEFAULT_RECENT_WINDOW_GAMES,
) -> pd.Series:
    """Versión pregame de compute_missing_key_players: cuenta jugadores clave ausentes en al menos uno de los `window` partidos anteriores (nunca el de hoy). NaN si faltan datos."""
    logs = key_player_game_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    played = set(zip(logs["Player_ID"], logs["season"], logs["GAME_DATE"]))

    key_players_by_team_season = key_players.groupby(["TeamID", "season"])["PLAYER_ID"].apply(list).to_dict()

    result = pd.Series(np.nan, index=team_game_features.index)
    for (team_id, season), group in team_game_features.groupby(["TEAM_ID", "season"]):
        key_ids = key_players_by_team_season.get((team_id, season))
        if key_ids is None:
            continue
        group = group.sort_values("GAME_DATE")
        dates = group["GAME_DATE"].tolist()
        idxs = group.index.tolist()

        for i in range(window, len(idxs)):
            recent_dates = dates[i - window:i]
            count = sum(
                1 for pid in key_ids
                if any((pid, season, d) not in played for d in recent_dates)
            )
            result.loc[idxs[i]] = count
    return result


def build_matchups_with_injury_signal(
    team_game_features: pd.DataFrame, key_players: pd.DataFrame, key_player_game_logs: pd.DataFrame
) -> pd.DataFrame:
    tgf = team_game_features.copy()
    tgf["missing_key_players"] = compute_missing_key_players(tgf, key_players, key_player_game_logs)
    tgf["recently_missing_key_players"] = compute_recently_missing_key_players(tgf, key_players, key_player_game_logs)

    home = tgf[tgf["is_home"]]
    away = tgf[~tgf["is_home"]]
    cols = ["GAME_ID", "season", "WL", "net_rating_rolling", "rest_days", "is_back_to_back",
            "three_pt_rate_rolling", "pace_rolling", "missing_key_players", "recently_missing_key_players"]
    m = home[cols].merge(away[cols], on=["GAME_ID", "season"], suffixes=("_home", "_away"))
    m = m.dropna(subset=["missing_key_players_home", "missing_key_players_away"])

    return pd.DataFrame({
        "GAME_ID": m["GAME_ID"], "season": m["season"],
        "home_win": (m["WL_home"] == "W").astype(int),
        "net_rating_diff": m["net_rating_rolling_home"] - m["net_rating_rolling_away"],
        "rest_days_diff": m["rest_days_home"] - m["rest_days_away"],
        "is_b2b_home": m["is_back_to_back_home"], "is_b2b_away": m["is_back_to_back_away"],
        "three_pt_rate_diff": m["three_pt_rate_rolling_home"] - m["three_pt_rate_rolling_away"],
        "pace_diff": m["pace_rolling_home"] - m["pace_rolling_away"],
        "missing_key_players_diff": m["missing_key_players_home"] - m["missing_key_players_away"],
        "recently_missing_key_players_diff": (
            m["recently_missing_key_players_home"] - m["recently_missing_key_players_away"]
        ),
    })


def run_loso_three_way(matchups: pd.DataFrame) -> None:
    """Compara con el mismo LOSO: sin señal, con oráculo, y con la versión pregame realista, sobre el mismo conjunto de partidos en los tres casos."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    matchups = matchups.dropna(subset=["recently_missing_key_players_diff"])

    cols_none = gwp.GBT_FEATURE_COLUMNS
    cols_oracle = gwp.GBT_FEATURE_COLUMNS + ["missing_key_players_diff"]
    cols_realistic = gwp.GBT_FEATURE_COLUMNS + ["recently_missing_key_players_diff"]

    seasons = sorted(matchups["season"].unique())
    oof_actual, oof_none, oof_oracle, oof_realistic = [], [], [], []

    for held_out in seasons:
        train = matchups[matchups["season"] != held_out]
        test = matchups[matchups["season"] == held_out]
        if test.empty or train.empty:
            continue

        gbt_none = HistGradientBoostingClassifier(random_state=0).fit(train[cols_none], train["home_win"])
        gbt_oracle = HistGradientBoostingClassifier(random_state=0).fit(train[cols_oracle], train["home_win"])
        gbt_realistic = HistGradientBoostingClassifier(random_state=0).fit(train[cols_realistic], train["home_win"])

        oof_actual.extend(test["home_win"].tolist())
        oof_none.extend(gbt_none.predict_proba(test[cols_none])[:, 1].tolist())
        oof_oracle.extend(gbt_oracle.predict_proba(test[cols_oracle])[:, 1].tolist())
        oof_realistic.extend(gbt_realistic.predict_proba(test[cols_realistic])[:, 1].tolist())

    a = np.array(oof_actual)
    none_, oracle_, realistic_ = np.array(oof_none), np.array(oof_oracle), np.array(oof_realistic)

    print(f"\nPartidos fuera de muestra: {len(a)} (temporadas: {seasons})")
    print(f"Brier score -- sin señal:       {brier_score_loss(a, none_):.4f}")
    print(f"Brier score -- oráculo (hoy):   {brier_score_loss(a, oracle_):.4f}")
    print(f"Brier score -- realista (pregame, últimos {DEFAULT_RECENT_WINDOW_GAMES} partidos): {brier_score_loss(a, realistic_):.4f}")
    print(f"Log-loss    -- sin: {log_loss(a, none_):.4f}   oráculo: {log_loss(a, oracle_):.4f}   realista: {log_loss(a, realistic_):.4f}")
    print(f"Accuracy    -- sin: {accuracy_score(a, none_ >= 0.5):.4f}   oráculo: {accuracy_score(a, oracle_ >= 0.5):.4f}   realista: {accuracy_score(a, realistic_ >= 0.5):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_PILOT_SEASONS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    config = load_config()
    paths = get_paths(config)

    rosters = pd.read_csv(paths["processed"] / "backtest_sweep_rosters.csv")
    career = pd.read_csv(paths["processed"] / "backtest_sweep_player_career_stats.csv")
    rosters = rosters[rosters["season"].isin(args.seasons)]

    key_players = select_key_players(rosters, career, args.top_n)
    print(f"Jugadores clave seleccionados: {len(key_players)} filas "
          f"({key_players[['TeamID','season']].drop_duplicates().shape[0]} equipo-temporada)")

    key_player_game_logs = fetch_key_player_game_logs(key_players, paths["raw"], args.refresh)
    print(f"Game logs descargados: {len(key_player_game_logs)} filas")

    advanced_logs = pd.read_csv(paths["processed"] / "backtest_sweep_advanced_game_logs.csv")
    advanced_logs = advanced_logs[advanced_logs["season"].isin(args.seasons)]
    team_game_features = gwp.build_team_game_features(advanced_logs)

    matchups = build_matchups_with_injury_signal(team_game_features, key_players, key_player_game_logs)
    print(f"Partidos con señal de disponibilidad calculada: {len(matchups)}")

    run_loso_three_way(matchups)


if __name__ == "__main__":
    main()
