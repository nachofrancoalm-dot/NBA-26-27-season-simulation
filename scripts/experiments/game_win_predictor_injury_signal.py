"""
game_win_predictor_injury_signal.py

Seguimiento de game_win_predictor.py (resultado NEGATIVO: GBT con seis
features no mejora la logística de una sola variable, ver su docstring).
Ese experimento no tenía ninguna señal de disponibilidad REAL del día del
partido -- todo era pregame trailing (rating, descanso, playstyle). Este
experimento añade una: para cada equipo-temporada, se identifican sus 5
jugadores con más minutos/partido ("jugadores clave") y se cuenta,
partido a partido, cuántos de ellos NO jugaron -- lesión, descanso,
DNP-CD, lo que sea, no se distingue el motivo, igual que el resto del
proyecto no lo distingue (ver injury_model.py).

QUÉ MIDE ESTO DE VERDAD -- Y QUÉ NO
---------------------------------------
Esto usa disponibilidad REAL RETROSPECTIVA (sabemos, después del hecho,
si el jugador jugó ese partido concreto) -- NO un injury report del día
del partido, que `nba_api` no expone y este proyecto no ingiere (ver la
discusión de RAG en llm_explainer.py). Por tanto este experimento NO
simula un sistema utilizable en producción (no se puede saber ANTES del
partido si un jugador clave va a jugar sin una fuente de datos que este
proyecto no tiene). Lo que SÍ mide es el TECHO: cuánto ayudaría, en el
mejor de los casos posible, saber quién juega de verdad. Si ni siquiera
esta versión "oráculo" mejora el modelo, no compensa construir nada más
ambicioso (una ingesta de injury reports en vivo).

INGESTA NUEVA -- ESTO SÍ PEGA A LA API
------------------------------------------
A diferencia de game_win_predictor.py (cero ingesta nueva), este
experimento necesita `player_game_log` (data_pipeline.fetch_player_game_log)
de cada jugador clave -- dato que el proyecto nunca había descargado a
esta escala. Con los 5 jugadores de más MPG por cada uno de los 480
equipo-temporada del backtest sweep, hacen falta ~2.250 llamadas
(pares (player_id, season) únicos) -- del orden de 25-45 min la primera
vez, cacheado después. Por eso este script empieza en modo PILOTO
(`--seasons`, 3 temporadas por defecto, ~425 llamadas, 5-10 min) antes
de comprometerse al sweep completo (`--seasons` con las 16 temporadas de
`config["backtest_sweep"]["seasons"]`).

DISEÑO
--------
1. `select_key_players()`: cruza `backtest_sweep_rosters.csv` (roster
   REAL de cada equipo-temporada) con `backtest_sweep_player_career_stats.csv`
   (career stats YA cacheadas, cero llamadas nuevas aquí) filtrando a la
   fila de la temporada exacta, calcula MPG = MIN/GP y toma el top-N por
   equipo-temporada.
2. `fetch_key_player_game_logs()`: UNA llamada por (player_id, season)
   único de los jugadores clave seleccionados -- reutiliza
   `data_pipeline.fetch_player_game_log` tal cual, mismo caché en disco.
3. `compute_missing_key_players()`: por cada partido de cada
   equipo-temporada, cuenta cuántos jugadores clave de ESE equipo no
   tienen fila en su game log para la fecha exacta del partido.
4. Se añade `missing_key_players_diff` (local menos visitante) al
   dataset de partidos de `game_win_predictor.build_matchup_dataset` y
   se compara GBT con 6 features (sin la señal) vs. GBT con 7 (con
   ella), LOSO igual que el experimento original.

RESULTADOS
------------
Piloto (3 temporadas, 2023-24/2024-25/2025-26, 3.633 partidos, LOSO de
3 pliegues -- poca potencia estadística, solo para decidir si compensaba
el sweep completo):

    Brier score  -- GBT sin señal: 0.2373   con señal: 0.2317
    Log-loss     -- sin: 0.6856   con: 0.6721
    Accuracy     -- sin: 63.25%   con: 64.13%

Sweep completo (16 temporadas del backtest sweep, 18.843 partidos, mismo
LOSO que game_win_predictor.py -- 146.323 filas de game logs de
jugadores clave descargadas, ~1.825 llamadas nuevas a la API tras
reutilizar el caché del piloto):

    Brier score  -- GBT sin señal: 0.2173   con señal: 0.2152
    Log-loss     -- sin: 0.6242   con: 0.6192
    Accuracy     -- sin: 64.93%   con: 65.42%

CONCLUSIÓN: positiva, primera mejora real de toda esta línea de
experimentos (game_win_predictor.py, tope de MOV, más iteraciones de
GBT -- todos negativos o neutros). La mejora del sweep completo (~1%
relativo en Brier) es más MODESTA que la del piloto (~2.4%) -- esperable,
el piloto tenía poca potencia estadística y regresiona a la media con
más datos -- pero la dirección se mantiene y es consistente en las tres
métricas. Más importante: GBT CON esta señal (Brier 0.2152) bate de
forma clara al baseline logístico de `game_win_predictor.py`
(net_rating_diff, Brier 0.2175) -- por primera vez en la serie, algo
supera al baseline con un margen mayor que el ruido fold-a-fold.

Esto confirma la hipótesis que motivó el experimento: la brecha frente a
las líneas de casas de apuestas (~67-70% de accuracy, ver
game_win_predictor.py) SÍ viene en parte de disponibilidad real del día
del partido, no solo de forma reciente/descanso/playstyle -- esas
features (descanso, B2B, ritmo, triples) no aportaban nada
(game_win_predictor.py), pero disponibilidad real sí.

LIMITACIÓN DEL ORÁCULO: usa quién jugó DE VERDAD, no un injury report del
día del partido -- NO desplegable en un sistema de predicción real tal
cual, porque antes de un partido no se sabe con esta fuente de datos
quién va a jugar esta noche.

VERSIÓN REALISTA (PREGAME, SIN ORÁCULO) -- SÍ DESPLEGABLE
---------------------------------------------------------------
Pregunta de seguimiento: ¿cuánto de esa mejora sobrevive si en vez del
oráculo se usa solo información que SÍ se conoce antes del partido?
`compute_recently_missing_key_players()`: para cada partido, cuenta
cuántos jugadores clave faltaron en AL MENOS UNO de los 3 partidos
ANTERIORES del equipo (nunca el de hoy) -- una tendencia de ausencia
reciente, no confirmación del día. Usa exactamente los mismos game logs
ya descargados, cero ingesta nueva. Comparación de tres vías con el
mismo LOSO (`run_loso_three_way`), sobre el subconjunto de partidos con
al menos 3 partidos previos por equipo (18.088 de 18.843, se pierden los
de inicio de temporada -- mismo criterio que el resto del proyecto):

    Brier score -- sin señal:                            0.2163
    Brier score -- oráculo (disponibilidad de hoy):       0.2132
    Brier score -- realista (tendencia, últimos 3 partidos): 0.2146
    Log-loss    -- sin: 0.6219   oráculo: 0.6146   realista: 0.6182
    Accuracy    -- sin: 65.16%   oráculo: 65.70%   realista: 65.78%

CONCLUSIÓN FINAL: positiva y desplegable. La versión realista conserva
~55% de la mejora de Brier del oráculo (0.0017 de 0.0031) sin necesitar
ningún dato en vivo -- y en accuracy incluso iguala/supera ligeramente
al oráculo (65.78% vs. 65.70%, diferencia dentro del ruido). Es la
primera señal de disponibilidad de esta línea de experimentos que es a
la vez real (mejora medible, no ruido) y utilizable (no depende de una
fuente de datos que el proyecto no tiene). Confirma que la tendencia
reciente de ausencias de los jugadores clave de un equipo -- no solo su
Net Rating agregado -- contiene información que `net_rating_rolling` por
sí solo no captura completamente (un jugador estrella recién lesionado
arrastra el rolling del equipo hacia abajo, pero LENTO -- tarda varios
partidos en reflejarse del todo; la tendencia de ausencia lo detecta de
inmediato).

Sigue sin graduarse a `simulation.py`/`game_win_predictor.py`: ambos
necesitarían una noción de "equipo real con historial reciente
verificable" que el simulador no tiene (proyecta un roster HIPOTÉTICO,
no cruza contra partidos ya jugados de una franquicia real) -- la misma
limitación de traducción que ya se señaló para `outcome_variance_scale`
en game_win_predictor.py. Queda documentado como hallazgo validado, no
como cambio de producción.

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
    """
    Une roster REAL (TeamID, season, PLAYER_ID) con la fila de career
    stats de esa temporada EXACTA (SEASON_ID == season -- sin esto, un
    jugador con carrera larga tendría múltiples filas y el merge
    duplicaría), calcula MPG y devuelve el top-N por equipo-temporada.
    Jugadores con GP=0 esa temporada (lesión total, dos-way sin debutar)
    se excluyen -- no hay MPG que calcular.
    """
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
    """
    Una llamada por (PLAYER_ID, season) único -- reutiliza el caché en
    disco de fetch_player_game_log, así que re-ejecutar el script no
    vuelve a pegarle a la API para pares ya descargados.
    """
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
    """
    Para cada fila de `team_game_features` (un partido de un
    equipo-temporada), cuenta cuántos de sus jugadores clave NO tienen
    fila en `key_player_game_logs` para esa fecha exacta -- disponibilidad
    real, no un proxy. Equipos-temporada sin jugadores clave descargados
    (fuera del piloto) devuelven NaN, no 0 -- 0 significaría "nadie
    faltó", que no es lo mismo que "no se comprobó".
    """
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
    """
    Versión PREGAME (desplegable, sin oráculo) de compute_missing_key_players:
    para cada partido, cuenta cuántos jugadores clave del equipo faltaron
    en AL MENOS UNO de los `window` partidos ANTERIORES del EQUIPO (nunca
    el partido de hoy) -- una tendencia reciente de ausencia, no
    disponibilidad confirmada del día. Es información que SÍ se conoce
    antes de que se juegue el partido.

    NaN si el equipo-temporada no está en `key_players` o si el partido
    no tiene `window` partidos previos dentro de `team_game_features`
    (que ya excluye el primer partido de temporada -- ver
    game_win_predictor.build_team_game_features) -- mismo criterio "sin
    evidencia, sin inventar" que compute_missing_key_players.
    """
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
    """
    Compara tres configuraciones de GBT con el mismo LOSO: sin señal de
    disponibilidad, con el ORÁCULO (compute_missing_key_players, no
    desplegable), y con la versión PREGAME realista
    (compute_recently_missing_key_players, sí desplegable). Las filas sin
    `recently_missing_key_players_diff` (partidos de inicio de temporada,
    ver docstring de compute_recently_missing_key_players) se excluyen de
    LAS TRES comparaciones -- mismo conjunto de partidos en los tres
    casos, para que la comparación sea limpia.
    """
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
