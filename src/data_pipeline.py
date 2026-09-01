"""
data_pipeline.py

Capa de ingesta de datos. Responsabilidades:
  1. Descargar (con caché local) los game logs históricos de cada jugador
     del roster definido en team_config.yaml.
  2. Descargar los game logs de temporada completa de los equipos listados
     en `historical_comparables`, para usarlos como benchmark de validación.
  3. Descargar splits avanzados por jugador (usage rate, ratings on/off)
     cuando estén disponibles.

Diseño clave para reproducibilidad: NINGUNA función aquí recibe un nombre
de equipo o jugador hardcodeado. Todo viene de `config`. Así, ejecutar
este mismo script contra un config.yaml distinto simula otro equipo.

nba_api golpea stats.nba.com, que es agresivo bloqueando peticiones
rápidas -> por eso hay caché en disco y un rate limiter simple entre
llamadas. Ejecuta esto una vez y trabaja desde los CSV cacheados.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from config_loader import get_paths, load_config

# Pausa entre llamadas a la API para evitar rate-limiting / bloqueos de IP.
# stats.nba.com no publica un límite oficial; 0.6s es un valor conservador
# usado ampliamente por la comunidad de nba_api.
API_CALL_DELAY_SECONDS = 0.6
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def _cached_fetch(cache_path: Path, fetch_fn, *, force_refresh: bool = False) -> pd.DataFrame:
    """
    Envoltorio genérico de caché: si el CSV ya existe y no se pide refresco,
    lo lee de disco; si no, llama a fetch_fn(), lo guarda y lo devuelve.
    Reintenta con backoff si la API falla (timeouts/bloqueos son comunes).
    """
    if cache_path.exists() and not force_refresh:
        print(f"  [caché local] {cache_path.name} (sin llamada a la API)")
        return pd.read_csv(cache_path)

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [API stats.nba.com] descargando {cache_path.name}...")
            df = fetch_fn()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            time.sleep(API_CALL_DELAY_SECONDS)
            return df
        except Exception as e:  # nba_api lanza excepciones variadas (timeout, JSON, etc.)
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  [reintento {attempt}/{MAX_RETRIES}] fallo: {e}. Esperando {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"No se pudo obtener datos tras {MAX_RETRIES} intentos: {last_error}")


def fetch_player_game_log(
    player_id: int,
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Game log de un jugador para una temporada concreta (box scores por partido)."""
    from nba_api.stats.endpoints import playergamelog

    cache_path = raw_dir / "player_game_logs" / f"{player_id}_{season}.csv"

    def _fetch():
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return log.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_player_career_stats(
    player_id: int,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Stats de carrera por temporada de un jugador (para modelar aging curves)."""
    from nba_api.stats.endpoints import playercareerstats

    cache_path = raw_dir / "player_career_stats" / f"{player_id}.csv"

    def _fetch():
        stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        return stats.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_player_playoff_career_stats(
    player_id: int,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Stats de playoffs por temporada de un jugador (para fatigue_accumulation.py).
    Mismo endpoint que fetch_player_career_stats, distinto data frame de
    resultado (SeasonTotalsPostSeason, índice 2). Sin filas para una
    temporada sin playoffs -- correcto, no un error.
    """
    from nba_api.stats.endpoints import playercareerstats

    cache_path = raw_dir / "player_playoff_career_stats" / f"{player_id}.csv"

    def _fetch():
        stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        return stats.get_data_frames()[2]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_team_game_log(
    team_id: int,
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Game log de temporada completa de un equipo (para los comparables históricos)."""
    from nba_api.stats.endpoints import teamgamelog

    cache_path = raw_dir / "team_game_logs" / f"{team_id}_{season}.csv"

    def _fetch():
        log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
        return log.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_league_schedule(
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Calendario completo de la liga para una temporada (para
    schedule_strength.py). No se filtra por equipo aquí -- se cachea
    completo y se filtra en la capa de contexto, así el mismo caché sirve
    si cambia el equipo del config.
    """
    from nba_api.stats.endpoints import scheduleleaguev2

    cache_path = raw_dir / "league_schedule" / f"{season}.csv"

    def _fetch():
        schedule = scheduleleaguev2.ScheduleLeagueV2(season=season)
        return schedule.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_league_standings(
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Standings (récord, WinPCT, DiffPointsPG) de todos los equipos para una
    temporada. Se usa como proxy de "fuerza del rival" para el calendario
    de la temporada SIGUIENTE (ver schedule_strength.py), ya que no hay
    forma de medir la fuerza real de una temporada que aún no se jugó.
    """
    from nba_api.stats.endpoints import leaguestandingsv3

    cache_path = raw_dir / "league_standings" / f"{season}.csv"

    def _fetch():
        standings = leaguestandingsv3.LeagueStandingsV3(season=season)
        return standings.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_team_advanced_game_log(
    team_id: int,
    season: str,
    raw_dir: Path,
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Game log de un equipo vía TeamGameLogs (plural, distinto de
    fetch_team_game_log) -- incluye PLUS_MINUS, necesario para estimar
    Net Rating en performance_curve.py. season_type="Playoffs" para
    playoffs en vez de temporada regular.
    """
    from nba_api.stats.endpoints import teamgamelogs

    season_type_slug = season_type.lower().replace(" ", "_")
    cache_path = raw_dir / "team_game_logs_advanced" / f"{team_id}_{season}_{season_type_slug}.csv"

    def _fetch():
        log = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id, season_nullable=season, season_type_nullable=season_type
        )
        return log.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_league_advanced_player_stats(
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Estadísticas AVANZADAS de todos los jugadores de la liga en una
    temporada (`leaguedashplayerstats` con measure_type="Advanced"). Una
    sola llamada devuelve la liga entera -- la ingesta más barata del
    proyecto por dato obtenido.

    Trae `NET_RATING` y `PIE`, que src/advanced_impact.py mezcla con el
    Game Score (puramente ofensivo) para formar la métrica de impacto
    compuesta -- ver docstring de advanced_impact.py.
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    cache_path = raw_dir / "league_advanced_player_stats" / f"{season}.csv"

    def _fetch():
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, measure_type_detailed_defense="Advanced"
        )
        return stats.get_data_frames()[0]

    # OJO: la columna MIN de este endpoint viene en minutos POR PARTIDO
    # (no totales) y no cambia con per_mode_detailed="Totals", al revés
    # que PlayerCareerStats. La conversión a totales (MIN * GP) la hace
    # advanced_impact.load_advanced_stats, no aquí.

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_league_hustle_stats(
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Hustle stats de todos los jugadores de la liga en una temporada
    (`leaguehustlestatsplayer`): CONTESTED_SHOTS, DEFLECTIONS,
    CHARGES_DRAWN, SCREEN_ASSISTS, LOOSE_BALLS_RECOVERED, BOX_OUTS. Una
    llamada por temporada, liga entera. Cubre la señal de "defensa/juego
    en equipo sin balón" que ni Game Score ni NET_RATING/PIE capturan
    (ver scripts/experiments/hustle_stats_signal.py para el análisis
    predictivo -- esta función solo ingiere el dato).

    Disponible solo desde 2015-16 (SportVU/Second Spectrum): temporadas
    anteriores devuelven un DataFrame vacío, no una excepción --
    `build_league_hustle_stats_dataset` lo detecta y salta esa temporada.
    """
    from nba_api.stats.endpoints import leaguehustlestatsplayer

    cache_path = raw_dir / "league_hustle_stats" / f"{season}.csv"

    def _fetch():
        stats = leaguehustlestatsplayer.LeagueHustleStatsPlayer(season=season, per_mode_time="PerGame")
        return stats.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_league_pt_defend_stats(
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Defensa por tracking (`leaguedashptdefend`, Second Spectrum):
    D_FG_PCT (% de tiro real del rival cuando este jugador es el
    defensor más cercano), NORMAL_FG_PCT (% de tiro normal de esos
    mismos rivales) y PCT_PLUSMINUS = D_FG_PCT - NORMAL_FG_PCT (negativo
    = el rival tira peor de lo normal). Es la señal de impacto defensivo
    más directa que expone nba_api -- ver
    scripts/experiments/pt_defend_signal.py para el análisis predictivo.

    Una llamada por temporada, liga entera. OJO: `CLOSE_DEF_PERSON_ID` es
    el player_id del defensor, no `PLAYER_ID` como en otros endpoints.

    Disponible solo desde 2013-14 (Second Spectrum): temporadas
    anteriores devuelven un DataFrame vacío, no una excepción.
    """
    from nba_api.stats.endpoints import leaguedashptdefend

    cache_path = raw_dir / "league_pt_defend_stats" / f"{season}.csv"

    def _fetch():
        stats = leaguedashptdefend.LeagueDashPtDefend(
            season=season, defense_category="Overall", per_mode_simple="PerGame"
        )
        return stats.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_player_common_info(
    player_id: int,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Posición real de un jugador (CommonPlayerInfo, columna `POSITION`,
    p. ej. "Guard") vía player_id. Necesario para el roster HIPOTÉTICO
    propio: `CommonTeamRoster` (fetch_team_roster) devolvería el roster
    real de la franquicia, no los jugadores inventados del config.

    OJO: el formato de POSITION difiere de CommonTeamRoster ("Guard" vs.
    "G") pero comparten la primera letra, que es lo único que usa
    `champion_profiles.POSITION_GROUPS` -- no hace falta normalizar.
    """
    from nba_api.stats.endpoints import commonplayerinfo

    cache_path = raw_dir / "player_common_info" / f"{player_id}.csv"

    def _fetch():
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        return info.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_player_shot_chart(
    player_id: int,
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Ubicación real de cada tiro de un jugador en una temporada concreta
    (ShotChartDetail: LOC_X/LOC_Y, SHOT_MADE_FLAG, SHOT_TYPE, ACTION_TYPE)
    para el mapa de tiros del popup de detalle de jugador
    (webapp/static/js/court.js). `team_id=0` trae los tiros del jugador
    con cualquier equipo esa temporada.
    """
    from nba_api.stats.endpoints import shotchartdetail

    cache_path = raw_dir / "shot_charts" / f"{player_id}_{season}.csv"

    def _fetch():
        chart = shotchartdetail.ShotChartDetail(
            team_id=0, player_id=player_id, season_nullable=season, context_measure_simple="FGA"
        )
        return chart.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def build_roster_shot_charts_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Mapa de tiros de cada jugador del roster propio, para su temporada
    REAL más reciente registrada (no la temporada de proyección del
    config, que es futura y no tiene tiros de verdad todavía) -- misma
    idea que las columnas GP/MPG "reales" del roster (ver
    dashboard/data_loader.py). Guarda
    data/processed/roster_shot_charts.csv (player_id, season, loc_x,
    loc_y, shot_made, shot_type) para que el router de la webapp lo lea
    directo, sin disparar ninguna llamada a nba_api desde un request
    HTTP (mismo principio que fetch_player_common_info -- ver
    webapp/routers/players.py). Un jugador sin temporadas reales
    registradas (rookie sin GP todavía) o sin tiros en su última
    temporada real (lesión) se omite, no rompe el resto.
    """
    paths = get_paths(config)
    career_path = paths["processed"] / "roster_career_stats.csv"
    if not career_path.exists():
        raise FileNotFoundError(f"No se encontró {career_path}. Corre la ingesta de roster primero.")
    career = pd.read_csv(career_path)
    career = career[career["GP"] > 0]

    rows = []
    for player_id, group in tqdm(career.groupby("PLAYER_ID"), desc="Descargando mapas de tiro del roster"):
        latest_season = sorted(group["SEASON_ID"].astype(str))[-1]
        shots = fetch_player_shot_chart(int(player_id), latest_season, paths["raw"], force_refresh)
        if shots.empty:
            continue
        for _, shot in shots.iterrows():
            rows.append(
                {
                    "player_id": int(player_id),
                    "season": latest_season,
                    "loc_x": shot["LOC_X"],
                    "loc_y": shot["LOC_Y"],
                    "shot_made": bool(shot["SHOT_MADE_FLAG"]),
                    "shot_type": shot["SHOT_TYPE"],
                }
            )

    shots_df = pd.DataFrame(rows, columns=["player_id", "season", "loc_x", "loc_y", "shot_made", "shot_type"])
    out_path = paths["processed"] / "roster_shot_charts.csv"
    shots_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(shots_df)} tiros, {shots_df['player_id'].nunique()} jugadores)")
    return shots_df


def fetch_team_roster(
    team_id: int,
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Roster real de un equipo en una temporada concreta -- necesario para
    el backtesting (src/backtesting.py), para correr el mismo pipeline de
    proyección/riesgo sobre quién jugó realmente en cada
    `historical_comparable`.
    """
    from nba_api.stats.endpoints import commonteamroster

    cache_path = raw_dir / "team_rosters" / f"{team_id}_{season}.csv"

    def _fetch():
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        return roster.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def build_league_rosters_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Roster real de las 30 franquicias NBA para config["team"]["season"] --
    necesario para simular rivales reales (src/league_simulation.py), no
    un WinPCT genérico. Guarda data/processed/league_rosters.csv.

    ADVERTENCIA DE COSTE: 30 llamadas a CommonTeamRoster. No forma parte
    de run_full_pipeline -- se llama aparte, una vez.
    """
    from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID

    paths = get_paths(config)
    season = config["team"]["season"]
    frames: List[pd.DataFrame] = []

    for abbreviation, team_id in tqdm(ABBREVIATION_TO_TEAM_ID.items(), desc="Descargando rosters de la liga"):
        df = fetch_team_roster(team_id, season, paths["raw"], force_refresh)
        df["team_abbreviation"] = abbreviation
        df["team_id"] = team_id
        df["season"] = season
        frames.append(df)

    rosters_df = pd.concat(frames, ignore_index=True)
    out_path = paths["processed"] / "league_rosters.csv"
    rosters_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(rosters_df)} filas, {len(ABBREVIATION_TO_TEAM_ID)} equipos)")
    return rosters_df


def build_league_player_stats_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Career stats (temporada regular Y playoffs) de cada jugador en el
    roster real de las 30 franquicias -- para proyectar rivales reales
    con el mismo pipeline que el roster propio. Guarda
    data/processed/league_player_career_stats.csv y
    data/processed/league_player_playoff_career_stats.csv.

    ADVERTENCIA DE COSTE: ~450 jugadores x 2 llamadas -- la ingesta más
    cara del proyecto (20-30+ min la primera vez). Cacheado después.
    """
    paths = get_paths(config)
    rosters_path = paths["processed"] / "league_rosters.csv"
    if not rosters_path.exists():
        raise FileNotFoundError(f"No se encontró {rosters_path}. Corre `build_league_rosters_dataset` primero.")
    rosters = pd.read_csv(rosters_path)

    regular_frames: List[pd.DataFrame] = []
    playoff_frames: List[pd.DataFrame] = []

    players = rosters[["PLAYER_ID", "PLAYER", "team_id", "team_abbreviation"]].drop_duplicates()
    for _, player in tqdm(list(players.iterrows()), desc="Descargando career stats de jugadores de la liga"):
        player_id = int(player["PLAYER_ID"])

        regular = fetch_player_career_stats(player_id, paths["raw"], force_refresh)
        regular["player_name"] = player["PLAYER"]
        regular["team_id"] = player["team_id"]
        regular["team_abbreviation"] = player["team_abbreviation"]
        regular_frames.append(regular)

        playoff = fetch_player_playoff_career_stats(player_id, paths["raw"], force_refresh)
        if not playoff.empty:
            playoff["player_name"] = player["PLAYER"]
            playoff["team_id"] = player["team_id"]
            playoff["team_abbreviation"] = player["team_abbreviation"]
            playoff_frames.append(playoff)

    regular_df = pd.concat(regular_frames, ignore_index=True)
    playoff_df = pd.concat(playoff_frames, ignore_index=True) if playoff_frames else pd.DataFrame()

    regular_out = paths["processed"] / "league_player_career_stats.csv"
    playoff_out = paths["processed"] / "league_player_playoff_career_stats.csv"
    regular_df.to_csv(regular_out, index=False)
    playoff_df.to_csv(playoff_out, index=False)
    print(f"Guardado: {regular_out} ({len(regular_df)} filas)")
    print(f"Guardado: {playoff_out} ({len(playoff_df)} filas)")
    return {"regular": regular_df, "playoff": playoff_df}


def build_league_player_countries_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Nacionalidad (`COUNTRY` de CommonPlayerInfo) de cada jugador único de
    los 30 equipos reales de la liga. Guarda
    data/processed/league_player_countries.csv (player_id, country).
    `league_rosters.csv` no la trae (CommonTeamRoster solo da POSITION),
    hace falta un endpoint por jugador.

    Necesaria para el chequeo de cuota de nacionalidad del All-Star
    (16 EE.UU. / 8 internacionales de 24) -- ver
    awards_projection.check_all_star_nationality_quota; sin este CSV el
    chequeo se degrada, no falla.

    ADVERTENCIA DE COSTE: ~450-500 llamadas. Opt-in vía --league, no
    forma parte de run_full_pipeline.
    """
    paths = get_paths(config)
    rosters_path = paths["processed"] / "league_rosters.csv"
    if not rosters_path.exists():
        raise FileNotFoundError(f"No se encontró {rosters_path}. Corre `build_league_rosters_dataset` primero.")
    rosters = pd.read_csv(rosters_path)

    rows = []
    players = rosters[["PLAYER_ID"]].drop_duplicates()
    for _, player in tqdm(list(players.iterrows()), desc="Descargando nacionalidad de jugadores de la liga"):
        player_id = int(player["PLAYER_ID"])
        info = fetch_player_common_info(player_id, paths["raw"], force_refresh)
        if info.empty or "COUNTRY" not in info.columns:
            continue
        rows.append({"player_id": player_id, "country": info["COUNTRY"].iloc[0]})

    countries_df = pd.DataFrame(rows, columns=["player_id", "country"])
    out_path = paths["processed"] / "league_player_countries.csv"
    countries_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(countries_df)} filas)")
    return countries_df


def _download_rosters_for_cases(
    cases: List[Dict[str, Any]], raw_dir: Path, force_refresh: bool, desc: str
) -> pd.DataFrame:
    """Núcleo compartido por build_historical_comparables_rosters_dataset y
    build_backtest_sweep_rosters_dataset. Un caso que falle tras agotar
    reintentos se SALTA con un aviso en vez de abortar toda la ingesta."""
    frames: List[pd.DataFrame] = []
    for case in tqdm(cases, desc=desc):
        try:
            df = fetch_team_roster(case["team_id"], case["season"], raw_dir, force_refresh)
        except Exception as e:  # noqa: BLE001 -- un caso roto no debe abortar el resto de la ingesta
            print(f"  [omitido] {case['name']}: {e}")
            continue
        df["comparable_name"] = case["name"]
        df["season"] = case["season"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_historical_comparables_rosters_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> pd.DataFrame:
    """
    Roster real de cada equipo en `historical_comparables` (4 casos
    narrativos, barato, parte del pipeline normal), para el backtesting.
    Guarda data/processed/historical_comparables_rosters.csv.
    """
    paths = get_paths(config)
    rosters_df = _download_rosters_for_cases(
        config["historical_comparables"], paths["raw"], force_refresh, "Descargando rosters de comparables"
    )
    out_path = paths["processed"] / "historical_comparables_rosters.csv"
    rosters_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(rosters_df)} filas)")
    return rosters_df


def build_backtest_sweep_rosters_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Igual que build_historical_comparables_rosters_dataset pero para
    `resolve_backtest_sweep_cases(config)` -- los 30 equipos NBA × todas
    las temporadas en config["backtest_sweep"]["seasons"] (450 casos por
    defecto). Ver ADVERTENCIA DE COSTE en backtesting.py. Guarda
    data/processed/backtest_sweep_rosters.csv.
    """
    from config_loader import resolve_backtest_sweep_cases

    paths = get_paths(config)
    cases = resolve_backtest_sweep_cases(config)
    rosters_df = _download_rosters_for_cases(
        cases, paths["raw"], force_refresh, "Descargando rosters del backtest sweep"
    )
    out_path = paths["processed"] / "backtest_sweep_rosters.csv"
    rosters_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(rosters_df)} filas, {len(cases)} casos)")
    return rosters_df


def _download_player_stats_for_cases(rosters: pd.DataFrame, raw_dir: Path, force_refresh: bool, desc: str) -> Dict[str, pd.DataFrame]:
    """
    Núcleo compartido por build_historical_comparables_player_stats_dataset
    y build_backtest_sweep_player_stats_dataset. Dedupe por PLAYER_ID
    ÚNICAMENTE (no por PLAYER_ID+caso): un jugador que aparece en varios
    casos tiene la misma carrera completa, así que se descarga UNA sola
    vez. IMPORTANTE: duplicar filas por caso rompería
    `_most_recent_n_seasons` en aging_curve.py, que colapsaría el
    baseline ponderado por recencia a una sola temporada.

    Un jugador cuya descarga falle tras agotar reintentos se SALTA con un
    aviso en vez de abortar toda la ingesta.
    """
    regular_frames: List[pd.DataFrame] = []
    playoff_frames: List[pd.DataFrame] = []

    players = rosters[["PLAYER_ID", "PLAYER"]].drop_duplicates(subset="PLAYER_ID")
    for _, player in tqdm(list(players.iterrows()), desc=desc):
        player_id = int(player["PLAYER_ID"])

        try:
            regular = fetch_player_career_stats(player_id, raw_dir, force_refresh)
        except Exception as e:  # noqa: BLE001 -- un jugador roto no debe abortar el resto de la ingesta
            print(f"  [omitido] player_id={player_id} ({player['PLAYER']}): {e}")
            continue
        regular["player_name"] = player["PLAYER"]
        regular_frames.append(regular)

        try:
            playoff = fetch_player_playoff_career_stats(player_id, raw_dir, force_refresh)
        except Exception as e:  # noqa: BLE001
            print(f"  [omitido playoffs] player_id={player_id} ({player['PLAYER']}): {e}")
            playoff = pd.DataFrame()
        if not playoff.empty:
            playoff["player_name"] = player["PLAYER"]
            playoff_frames.append(playoff)

    regular_df = pd.concat(regular_frames, ignore_index=True) if regular_frames else pd.DataFrame()
    playoff_df = pd.concat(playoff_frames, ignore_index=True) if playoff_frames else pd.DataFrame()
    return {"regular": regular_df, "playoff": playoff_df}


def build_historical_comparables_player_stats_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Career stats (temporada regular Y playoffs) de cada jugador del
    roster real de cada `historical_comparable` -- para el backtesting.
    Guarda data/processed/historical_comparables_player_career_stats.csv
    y data/processed/historical_comparables_player_playoff_career_stats.csv.
    """
    paths = get_paths(config)
    rosters_path = paths["processed"] / "historical_comparables_rosters.csv"
    if not rosters_path.exists():
        raise FileNotFoundError(
            f"No se encontró {rosters_path}. Corre "
            "`build_historical_comparables_rosters_dataset` primero."
        )
    rosters = pd.read_csv(rosters_path)

    result = _download_player_stats_for_cases(
        rosters, paths["raw"], force_refresh, "Descargando career stats de jugadores de comparables"
    )
    regular_out = paths["processed"] / "historical_comparables_player_career_stats.csv"
    playoff_out = paths["processed"] / "historical_comparables_player_playoff_career_stats.csv"
    result["regular"].to_csv(regular_out, index=False)
    result["playoff"].to_csv(playoff_out, index=False)
    print(f"Guardado: {regular_out} ({len(result['regular'])} filas)")
    print(f"Guardado: {playoff_out} ({len(result['playoff'])} filas)")
    return result


def build_backtest_sweep_player_stats_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Igual que build_historical_comparables_player_stats_dataset pero
    leyendo data/processed/backtest_sweep_rosters.csv (450 casos por
    defecto). Es la parte más cara de la ingesta del sweep. Guarda
    data/processed/backtest_sweep_player_career_stats.csv y
    data/processed/backtest_sweep_player_playoff_career_stats.csv.
    """
    paths = get_paths(config)
    rosters_path = paths["processed"] / "backtest_sweep_rosters.csv"
    if not rosters_path.exists():
        raise FileNotFoundError(
            f"No se encontró {rosters_path}. Corre `build_backtest_sweep_rosters_dataset` primero."
        )
    rosters = pd.read_csv(rosters_path)

    result = _download_player_stats_for_cases(
        rosters, paths["raw"], force_refresh, "Descargando career stats de jugadores del backtest sweep"
    )
    regular_out = paths["processed"] / "backtest_sweep_player_career_stats.csv"
    playoff_out = paths["processed"] / "backtest_sweep_player_playoff_career_stats.csv"
    result["regular"].to_csv(regular_out, index=False)
    result["playoff"].to_csv(playoff_out, index=False)
    print(f"Guardado: {regular_out} ({len(result['regular'])} filas)")
    print(f"Guardado: {playoff_out} ({len(result['playoff'])} filas)")
    return result


def build_roster_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Descarga y consolida career stats de cada jugador del roster definido
    en el config. Devuelve un único DataFrame con una fila por
    jugador-temporada, listo para el modelado de aging curves y
    proyección individual.
    """
    paths = get_paths(config)
    frames: List[pd.DataFrame] = []

    roster = [p for p in config["roster"] if p.get("player_id")]
    skipped = [p for p in config["roster"] if not p.get("player_id")]
    if skipped:
        print(f"Aviso: {len(skipped)} jugador(es) sin player_id, se omiten: "
              f"{[p['name'] for p in skipped]}")

    for player in tqdm(roster, desc="Descargando career stats del roster"):
        df = fetch_player_career_stats(player["player_id"], paths["raw"], force_refresh)
        df["player_name"] = player["name"]
        df["role_expected"] = player.get("role_expected")
        df["minutes_projection"] = player.get("minutes_projection")
        frames.append(df)

    roster_df = pd.concat(frames, ignore_index=True)
    out_path = paths["processed"] / "roster_career_stats.csv"
    roster_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(roster_df)} filas)")
    return roster_df


def build_roster_positions_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Posición y nacionalidad reales de cada jugador del roster hipotético
    del config (misma llamada de fetch_player_common_info trae ambas
    columnas). Guarda data/processed/roster_positions.csv (player_id,
    position, country).

    `position` es necesaria para los quintetos All-NBA / All-Defensive
    (2 bases/escoltas + 2 aleros/ala-pívots + 1 pívot, ver
    awards_projection.py); `role_expected` de team_config.yaml es
    deliberadamente descriptivo, no una entrada de cálculo.

    `country` es necesaria para el chequeo de cuota de nacionalidad del
    All-Star (mínimo 16 jugadores de EE.UU. / 8 internacionales, regla
    real de la NBA) -- ver awards_projection.check_all_star_nationality_quota.
    """
    paths = get_paths(config)
    rows = []

    roster = [p for p in config["roster"] if p.get("player_id")]
    for player in tqdm(roster, desc="Descargando posición del roster"):
        info = fetch_player_common_info(player["player_id"], paths["raw"], force_refresh)
        if info.empty or "POSITION" not in info.columns:
            continue
        rows.append({
            "player_id": player["player_id"],
            "position": info["POSITION"].iloc[0],
            "country": info["COUNTRY"].iloc[0] if "COUNTRY" in info.columns else None,
        })

    positions_df = pd.DataFrame(rows, columns=["player_id", "position", "country"])
    out_path = paths["processed"] / "roster_positions.csv"
    positions_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(positions_df)} filas)")
    return positions_df


def build_roster_playoff_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Descarga y consolida career stats de PLAYOFFS de cada jugador del
    roster. Igual que build_roster_dataset pero para
    fetch_player_playoff_career_stats. Jugadores sin ninguna aparición en
    playoffs devuelven un DataFrame vacío para ese jugador (se omite del
    resultado consolidado, no es un error).
    """
    paths = get_paths(config)
    frames: List[pd.DataFrame] = []

    roster = [p for p in config["roster"] if p.get("player_id")]
    for player in tqdm(roster, desc="Descargando playoff career stats del roster"):
        df = fetch_player_playoff_career_stats(player["player_id"], paths["raw"], force_refresh)
        if df.empty:
            continue
        df["player_name"] = player["name"]
        frames.append(df)

    if not frames:
        playoff_df = pd.DataFrame()
    else:
        playoff_df = pd.concat(frames, ignore_index=True)

    out_path = paths["processed"] / "roster_playoff_career_stats.csv"
    playoff_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(playoff_df)} filas)")
    return playoff_df


def _previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'. Usado para saber qué temporada de standings
    pedir como proxy de fuerza de rival (ver fetch_league_standings)."""
    start_year = int(season.split("-")[0]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def resolve_advanced_stats_seasons(config: Dict[str, Any]) -> List[str]:
    """
    Temporadas para las que hace falta bajar estadísticas avanzadas: la
    unión de las del `backtest_sweep` y las N anteriores a
    `config["team"]["season"]` (para proyectar el roster hipotético).

    OJO: hace falta también el lookback anterior a la temporada más
    antigua del sweep -- sin él, los primeros casos del sweep se quedan
    sin métricas avanzadas y caen a Game Score puro, mezclando dos
    modelos distintos dentro del mismo backtest.
    """
    lookback = config.get("aging_curve", {}).get("n_seasons_lookback", 3)
    sweep = [str(s) for s in config.get("backtest_sweep", {}).get("seasons", [])]
    seasons = set(sweep)

    starting_points = [config["team"]["season"]]
    if sweep:
        starting_points.append(min(sweep))
    for start in starting_points:
        season = start
        for _ in range(lookback):
            season = _previous_season(season)
            seasons.add(season)
    return sorted(seasons)


def build_league_advanced_player_stats_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> pd.DataFrame:
    """
    Estadísticas avanzadas de jugador (NET_RATING, PIE) para todas las
    temporadas que el proyecto necesita -- ver
    `resolve_advanced_stats_seasons`. Guarda
    data/processed/league_advanced_player_stats.csv con una columna
    `season` añadida (el endpoint no la devuelve). Barato (~19 llamadas),
    pero opt-in, no forma parte de `run_full_pipeline`.
    """
    paths = get_paths(config)
    frames = []
    for season in tqdm(resolve_advanced_stats_seasons(config), desc="Stats avanzadas"):
        try:
            df = fetch_league_advanced_player_stats(season, paths["raw"], force_refresh)
        except Exception as exc:  # noqa: BLE001 -- ver _download_*_for_cases
            print(f"  AVISO: temporada {season} saltada ({type(exc).__name__}: {exc})")
            continue
        frames.append(df.assign(season=season))
        time.sleep(API_CALL_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = paths["processed"] / "league_advanced_player_stats.csv"
    combined.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(combined)} filas, {len(frames)} temporadas)")
    return combined


def build_league_hustle_stats_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Hustle stats de jugador (ver fetch_league_hustle_stats) para las
    mismas temporadas que resolve_advanced_stats_seasons. Guarda
    data/processed/league_hustle_player_stats.csv. Temporadas anteriores
    a 2015-16 devuelven un DataFrame vacío y se saltan sin abortar el resto.
    """
    paths = get_paths(config)
    frames = []
    skipped_no_data = []
    for season in tqdm(resolve_advanced_stats_seasons(config), desc="Hustle stats"):
        try:
            df = fetch_league_hustle_stats(season, paths["raw"], force_refresh)
        except Exception as exc:  # noqa: BLE001 -- ver _download_*_for_cases
            print(f"  AVISO: temporada {season} saltada ({type(exc).__name__}: {exc})")
            continue
        if df.empty:
            skipped_no_data.append(season)
            continue
        frames.append(df.assign(season=season))
        time.sleep(API_CALL_DELAY_SECONDS)

    if skipped_no_data:
        print(f"  Sin datos de hustle (anteriores a 2015-16): {', '.join(skipped_no_data)}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = paths["processed"] / "league_hustle_player_stats.csv"
    combined.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(combined)} filas, {len(frames)} temporadas)")
    return combined


def build_league_pt_defend_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Defensa por tracking (ver fetch_league_pt_defend_stats) para las
    mismas temporadas que resolve_advanced_stats_seasons. Guarda
    data/processed/league_pt_defend_stats.csv con `CLOSE_DEF_PERSON_ID`
    renombrado a `PLAYER_ID`. Temporadas anteriores a 2013-14 devuelven
    un DataFrame vacío y se saltan sin abortar el resto.
    """
    paths = get_paths(config)
    frames = []
    skipped_no_data = []
    for season in tqdm(resolve_advanced_stats_seasons(config), desc="PT defend stats"):
        try:
            df = fetch_league_pt_defend_stats(season, paths["raw"], force_refresh)
        except Exception as exc:  # noqa: BLE001 -- ver _download_*_for_cases
            print(f"  AVISO: temporada {season} saltada ({type(exc).__name__}: {exc})")
            continue
        if df.empty:
            skipped_no_data.append(season)
            continue
        frames.append(df.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"}).assign(season=season))
        time.sleep(API_CALL_DELAY_SECONDS)

    if skipped_no_data:
        print(f"  Sin datos de tracking (anteriores a 2013-14): {', '.join(skipped_no_data)}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = paths["processed"] / "league_pt_defend_stats.csv"
    combined.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(combined)} filas, {len(frames)} temporadas)")
    return combined


def fetch_league_2man_lineup_stats(
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Net rating real de cada pareja de jugadores que compartió cancha esa
    temporada (`leaguedashlineups`, group_quantity=2, measure_type
    Advanced) -- alimenta scripts/experiments/lineup_synergy_signal.py,
    que contrasta los efectos de src/lineup_synergy.py (nunca calibrados
    empíricamente) contra datos reales de pareja. Una llamada por
    temporada, liga entera. `GROUP_ID` viene como
    "-player_id_a-player_id_b-"; el experimento lo parsea, no este fetcher.
    """
    from nba_api.stats.endpoints import leaguedashlineups

    cache_path = raw_dir / "league_2man_lineups" / f"{season}.csv"

    def _fetch():
        stats = leaguedashlineups.LeagueDashLineups(
            season=season, group_quantity="2", measure_type_detailed_defense="Advanced", per_mode_detailed="Totals"
        )
        return stats.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def build_league_2man_lineup_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Net rating de parejas de jugadores para las temporadas del
    `backtest_sweep` (config["backtest_sweep"]["seasons"]). Guarda
    data/processed/league_2man_lineups.csv.
    """
    paths = get_paths(config)
    seasons = config.get("backtest_sweep", {}).get("seasons", [])
    frames = []
    for season in tqdm(seasons, desc="2-man lineup net ratings"):
        try:
            df = fetch_league_2man_lineup_stats(season, paths["raw"], force_refresh)
        except Exception as exc:  # noqa: BLE001 -- ver _download_*_for_cases
            print(f"  AVISO: temporada {season} saltada ({type(exc).__name__}: {exc})")
            continue
        if not df.empty:
            frames.append(df.assign(season=season))
        time.sleep(API_CALL_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = paths["processed"] / "league_2man_lineups.csv"
    combined.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(combined)} parejas, {len(frames)} temporadas)")
    return combined


# Cuatro categorías de tracking (`leaguedashptstats`) para probar
# candidatos nuevos de sinergia de pareja en
# scripts/experiments/lineup_synergy_signal.py, más allá de
# usage_clash/playmaking_spacing_synergy (sin apoyo empírico): tiro con
# balón vs. catch-and-shoot, penetraciones vs. presencia interior
# (proxy de pick-and-roll), y volumen de poste.
LINEUP_SYNERGY_TRACKING_MEASURE_TYPES = ["CatchShoot", "PullUpShot", "Drives", "PostTouch"]


def fetch_league_tracking_stats(
    season: str,
    pt_measure_type: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Una categoría de `leaguedashptstats` (tracking SportVU/Second
    Spectrum) para todos los jugadores de una temporada -- una llamada.
    `pt_measure_type` debe ser uno de LINEUP_SYNERGY_TRACKING_MEASURE_TYPES.
    """
    from nba_api.stats.endpoints import leaguedashptstats

    cache_path = raw_dir / "league_tracking_stats" / pt_measure_type / f"{season}.csv"

    def _fetch():
        stats = leaguedashptstats.LeagueDashPtStats(
            season=season, pt_measure_type=pt_measure_type, player_or_team="Player", per_mode_simple="Totals"
        )
        return stats.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def build_league_tracking_stats_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Las 4 categorías de LINEUP_SYNERGY_TRACKING_MEASURE_TYPES para las
    temporadas del `backtest_sweep`, unidas en un solo CSV ancho (una
    fila por PLAYER_ID+season, merge sucesivo por esa clave). Guarda
    data/processed/league_tracking_stats.csv. Temporadas o categorías sin
    datos se saltan sin abortar el resto.
    """
    paths = get_paths(config)
    seasons = config.get("backtest_sweep", {}).get("seasons", [])

    season_frames = []
    for season in tqdm(seasons, desc="Tracking stats (4 categorías/temporada)"):
        merged_season = None
        for measure_type in LINEUP_SYNERGY_TRACKING_MEASURE_TYPES:
            try:
                df = fetch_league_tracking_stats(season, measure_type, paths["raw"], force_refresh)
            except Exception as exc:  # noqa: BLE001 -- ver _download_*_for_cases
                print(f"  AVISO: {season}/{measure_type} saltada ({type(exc).__name__}: {exc})")
                continue
            if df.empty:
                continue
            keep_cols = [c for c in df.columns if c == "PLAYER_ID" or c not in ("PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "GP", "W", "L")]
            df = df[keep_cols]
            merged_season = df if merged_season is None else merged_season.merge(df, on="PLAYER_ID", how="outer", suffixes=("", f"_{measure_type}_dup"))
            time.sleep(API_CALL_DELAY_SECONDS)
        if merged_season is not None:
            season_frames.append(merged_season.assign(season=season))

    combined = pd.concat(season_frames, ignore_index=True) if season_frames else pd.DataFrame()
    out_path = paths["processed"] / "league_tracking_stats.csv"
    combined.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(combined)} filas, {len(season_frames)} temporadas)")
    return combined


def build_team_schedule_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Filtra el calendario completo de la liga a los partidos del equipo del
    config (temporada actual). Guarda data/processed/team_schedule.csv.
    """
    paths = get_paths(config)
    league_schedule = fetch_league_schedule(config["team"]["season"], paths["raw"], force_refresh)

    team_id = config["team"]["team_id"]
    is_home = league_schedule["homeTeam_teamId"] == team_id
    is_away = league_schedule["awayTeam_teamId"] == team_id
    team_schedule = league_schedule[is_home | is_away].copy()

    out_path = paths["processed"] / "team_schedule.csv"
    team_schedule.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(team_schedule)} partidos)")
    return team_schedule


def build_league_schedule_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Calendario real de la liga completa (los 30 equipos), para
    league_simulation.py -- hermana de build_team_schedule_dataset (que
    filtra a UN equipo). Guarda data/processed/league_schedule_full.csv
    con gameDate/homeTeam_teamTricode/awayTeam_teamTricode.

    Filtra partidos de preseason y filas con tricode nulo (plazas de la
    NBA Cup sin resolver, se descartan en vez de inventarse) -- por eso,
    mientras la Cup no se resuelva, cada equipo puede tener menos
    partidos que `config["simulation"]["games_per_season"]`.
    """
    paths = get_paths(config)
    league_schedule = fetch_league_schedule(config["team"]["season"], paths["raw"], force_refresh)

    is_regular_season = league_schedule["gameLabel"].fillna("") != "Preseason"
    has_both_teams = league_schedule["homeTeam_teamTricode"].notna() & league_schedule["awayTeam_teamTricode"].notna()
    schedule = league_schedule[is_regular_season & has_both_teams][
        ["gameDate", "homeTeam_teamTricode", "awayTeam_teamTricode"]
    ].copy()

    out_path = paths["processed"] / "league_schedule_full.csv"
    schedule.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(schedule)} partidos)")
    return schedule


def build_prior_season_standings_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> pd.DataFrame:
    """
    Standings de la temporada ANTERIOR a la del config -- proxy de fuerza
    de rival para el calendario de la temporada del config (ver docstring
    de fetch_league_standings). Guarda
    data/processed/prior_season_standings.csv.
    """
    paths = get_paths(config)
    prior_season = _previous_season(config["team"]["season"])
    standings = fetch_league_standings(prior_season, paths["raw"], force_refresh)

    out_path = paths["processed"] / "prior_season_standings.csv"
    standings.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(standings)} equipos, temporada {prior_season})")
    return standings


def _download_standings_for_cases(cases: List[Dict[str, Any]], raw_dir: Path, force_refresh: bool, desc: str) -> pd.DataFrame:
    """Núcleo compartido -- una llamada por TEMPORADA distinta entre los
    casos (los standings son de toda la liga de una vez). Una temporada
    que falle tras agotar reintentos se salta con un aviso."""
    frames: List[pd.DataFrame] = []
    seasons_seen = set()
    for case in tqdm(cases, desc=desc):
        season = case["season"]
        if season in seasons_seen:
            continue
        seasons_seen.add(season)
        try:
            standings = fetch_league_standings(season, raw_dir, force_refresh)
        except Exception as e:  # noqa: BLE001 -- una temporada rota no debe abortar el resto de la ingesta
            print(f"  [omitido] standings {season}: {e}")
            continue
        standings["season"] = season
        frames.append(standings)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_historical_comparables_standings_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> pd.DataFrame:
    """
    Standings de la MISMA temporada de cada caso en `historical_comparables`
    -- a diferencia de build_prior_season_standings_dataset, aquí no hace
    falta un proxy de temporada anterior porque estas temporadas ya se
    jugaron: se puede pedir la fuerza contemporánea real de cada rival.
    Usado por src/context/opponent_weighting.py para ponderar partidos
    contra contenders más que contra equipos en reconstrucción. Guarda
    data/processed/historical_comparables_standings.csv.
    """
    paths = get_paths(config)
    standings_df = _download_standings_for_cases(
        config["historical_comparables"], paths["raw"], force_refresh, "Descargando standings de comparables"
    )
    out_path = paths["processed"] / "historical_comparables_standings.csv"
    standings_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(standings_df)} filas)")
    return standings_df


def build_backtest_sweep_standings_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Igual que build_historical_comparables_standings_dataset pero para
    `resolve_backtest_sweep_cases(config)` -- solo 1 llamada por cada
    temporada DISTINTA en config["backtest_sweep"]["seasons"] (15 por
    defecto, no 450, ya que los standings son de toda la liga a la vez).
    Guarda data/processed/backtest_sweep_standings.csv.
    """
    from config_loader import resolve_backtest_sweep_cases

    paths = get_paths(config)
    cases = resolve_backtest_sweep_cases(config)
    standings_df = _download_standings_for_cases(
        cases, paths["raw"], force_refresh, "Descargando standings del backtest sweep"
    )
    out_path = paths["processed"] / "backtest_sweep_standings.csv"
    standings_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(standings_df)} filas)")
    return standings_df


def build_historical_comparables_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> pd.DataFrame:
    """
    Descarga los game logs de temporada completa de cada equipo listado en
    `historical_comparables`. Esta es la base para el backtesting del
    modelo de sinergia (ver src/lineup_synergy.py).
    """
    paths = get_paths(config)
    frames: List[pd.DataFrame] = []

    for case in tqdm(config["historical_comparables"], desc="Descargando comparables históricos"):
        df = fetch_team_game_log(case["team_id"], case["season"], paths["raw"], force_refresh)
        df["comparable_name"] = case["name"]
        df["season"] = case["season"]
        df["note"] = case.get("note", "")
        frames.append(df)

    comparables_df = pd.concat(frames, ignore_index=True)
    out_path = paths["processed"] / "historical_comparables_game_logs.csv"
    comparables_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(comparables_df)} filas)")
    return comparables_df


def _download_advanced_game_logs_for_cases(cases: List[Dict[str, Any]], raw_dir: Path, force_refresh: bool, desc: str) -> pd.DataFrame:
    """Núcleo compartido -- ver docstrings de build_historical_comparables_advanced_dataset
    / build_backtest_sweep_advanced_game_logs_dataset. Un caso roto se
    salta con un aviso en vez de abortar toda la ingesta."""
    frames: List[pd.DataFrame] = []
    for case in tqdm(cases, desc=desc):
        try:
            regular = fetch_team_advanced_game_log(
                case["team_id"], case["season"], raw_dir, "Regular Season", force_refresh
            )
            regular["game_phase"] = "regular"
            playoffs = fetch_team_advanced_game_log(
                case["team_id"], case["season"], raw_dir, "Playoffs", force_refresh
            )
            playoffs["game_phase"] = "playoffs"
        except Exception as e:  # noqa: BLE001 -- un caso roto no debe abortar el resto de la ingesta
            print(f"  [omitido] {case['name']}: {e}")
            continue

        for df in (regular, playoffs):
            if df.empty:
                continue
            df["comparable_name"] = case["name"]
            df["season"] = case["season"]
            frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_historical_comparables_advanced_dataset(
    config: Dict[str, Any], force_refresh: bool = False
) -> pd.DataFrame:
    """
    Game logs de temporada regular Y playoffs (con PLUS_MINUS) de cada
    equipo en `historical_comparables` -- para estimar Net Rating por
    partido en rolling windows (src/context/performance_curve.py) y
    detectar arranques lentos / picos de forma en playoffs. Un comparable
    sin apariciones en playoffs esa temporada simplemente aporta 0 filas
    de playoffs, no es un error.
    """
    paths = get_paths(config)
    advanced_df = _download_advanced_game_logs_for_cases(
        config["historical_comparables"], paths["raw"], force_refresh, "Descargando game logs avanzados de comparables"
    )
    out_path = paths["processed"] / "historical_comparables_advanced_game_logs.csv"
    advanced_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(advanced_df)} filas)")
    return advanced_df


def build_backtest_sweep_advanced_game_logs_dataset(config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """
    Igual que build_historical_comparables_advanced_dataset pero para
    `resolve_backtest_sweep_cases(config)` (450 casos por defecto, 2
    llamadas cada uno -- parte cara de la ingesta del sweep). Guarda
    data/processed/backtest_sweep_advanced_game_logs.csv.
    """
    from config_loader import resolve_backtest_sweep_cases

    paths = get_paths(config)
    cases = resolve_backtest_sweep_cases(config)
    advanced_df = _download_advanced_game_logs_for_cases(
        cases, paths["raw"], force_refresh, "Descargando game logs avanzados del backtest sweep"
    )
    out_path = paths["processed"] / "backtest_sweep_advanced_game_logs.csv"
    advanced_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(advanced_df)} filas)")
    return advanced_df


def run_full_pipeline(config_path: Optional[str] = None, force_refresh: bool = False) -> None:
    """Punto de entrada: ejecuta la ingesta completa para el equipo del config."""
    config = load_config(config_path) if config_path else load_config()

    print(f"=== Pipeline de datos para: {config['team']['name']} ({config['team']['season']}) ===\n")

    print("--- 1/8: Roster actual (temporada regular) ---")
    build_roster_dataset(config, force_refresh=force_refresh)

    print("\n--- 2/8: Roster actual (playoffs) ---")
    build_roster_playoff_dataset(config, force_refresh=force_refresh)

    print("\n--- 3/8: Comparables históricos ---")
    build_historical_comparables_dataset(config, force_refresh=force_refresh)

    print("\n--- 4/8: Comparables históricos (game logs avanzados, PLUS_MINUS) ---")
    build_historical_comparables_advanced_dataset(config, force_refresh=force_refresh)

    print("\n--- 5/8: Comparables históricos (standings de su propia temporada) ---")
    build_historical_comparables_standings_dataset(config, force_refresh=force_refresh)

    print("\n--- 6/8: Calendario y fuerza de rivales ---")
    build_team_schedule_dataset(config, force_refresh=force_refresh)
    build_prior_season_standings_dataset(config, force_refresh=force_refresh)

    print("\n--- 7/8: Comparables históricos (rosters reales, para backtesting) ---")
    build_historical_comparables_rosters_dataset(config, force_refresh=force_refresh)

    print("\n--- 8/8: Comparables históricos (career stats de cada jugador real) ---")
    build_historical_comparables_player_stats_dataset(config, force_refresh=force_refresh)

    print("\nPipeline completo. Datos disponibles en data/processed/.")


def run_backtest_sweep_ingestion(config_path: Optional[str] = None, force_refresh: bool = False) -> None:
    """
    Ingesta para el backtesting sistemático a gran escala (ver
    src/config_loader.py:resolve_backtest_sweep_cases y
    src/backtesting.py:build_backtest_sweep_dataset): los 30 equipos NBA
    para cada temporada en config["backtest_sweep"]["seasons"] (450 casos
    por defecto). ADVERTENCIA DE COSTE: del orden de miles de llamadas a
    la API, probablemente 1.5-3 horas la primera vez. Opt-in (flag
    --backtest-sweep), no forma parte de run_full_pipeline.
    """
    config = load_config(config_path) if config_path else load_config()
    if not config.get("backtest_sweep"):
        raise ValueError(
            "config['backtest_sweep'] no está definido en team_config.yaml -- "
            "añade un bloque 'backtest_sweep: {seasons: [...]}' primero."
        )

    print(f"=== Backtest sweep: 30 equipos x {len(config['backtest_sweep']['seasons'])} temporadas ===\n")

    print("--- 1/5: Rosters reales de cada caso ---")
    build_backtest_sweep_rosters_dataset(config, force_refresh=force_refresh)

    print("\n--- 2/5: Career stats de cada jugador único ---")
    build_backtest_sweep_player_stats_dataset(config, force_refresh=force_refresh)

    print("\n--- 3/5: Standings de cada temporada ---")
    build_backtest_sweep_standings_dataset(config, force_refresh=force_refresh)

    print("\n--- 4/5: Game logs avanzados (regular + playoffs) de cada caso ---")
    build_backtest_sweep_advanced_game_logs_dataset(config, force_refresh=force_refresh)

    # Barato (una llamada por temporada) pero va aquí y no en
    # run_full_pipeline porque las temporadas que cubre son las del sweep.
    print("\n--- 5/5: Estadísticas avanzadas de jugador (NET_RATING, PIE) ---")
    build_league_advanced_player_stats_dataset(config, force_refresh=force_refresh)

    print("\nIngesta del backtest sweep completa. Corre ahora "
          "`from src.backtesting import build_backtest_sweep_dataset` "
          "para simular y generar backtest_sweep_summary.csv.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingesta de datos NBA. Por defecto usa caché local si existe "
                     "(no llama a la API salvo que falte el archivo o se pase --refresh)."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignora la caché y vuelve a descargar todo desde stats.nba.com",
    )
    parser.add_argument(
        "--league",
        action="store_true",
        help="Además del pipeline normal, descarga rosters + career stats + nacionalidad "
             "de las 30 franquicias + calendario real de la temporada (necesario para "
             "league_simulation.py y el chequeo de cuota del All-Star). ADVERTENCIA: "
             "~1350 llamadas a la API (30-45+ min la primera vez). Opt-in.",
    )
    parser.add_argument(
        "--backtest-sweep",
        action="store_true",
        help="Además del pipeline normal, descarga los 30 equipos NBA para cada "
             "temporada en config['backtest_sweep']['seasons'] (450 casos por "
             "defecto). ADVERTENCIA: la ingesta más cara del proyecto -- del orden "
             "de miles de llamadas, 1.5-3 horas la primera vez. Opt-in.",
    )
    args = parser.parse_args()

    run_full_pipeline(force_refresh=args.refresh)

    if args.league:
        config = load_config()
        print("\n=== Ingesta de liga completa (30 equipos) ===\n")
        print("--- 1/4: Rosters de las 30 franquicias ---")
        build_league_rosters_dataset(config, force_refresh=args.refresh)
        print("\n--- 2/4: Career stats de todos los jugadores de la liga ---")
        build_league_player_stats_dataset(config, force_refresh=args.refresh)
        print("\n--- 3/4: Nacionalidad de todos los jugadores de la liga (cuota All-Star) ---")
        build_league_player_countries_dataset(config, force_refresh=args.refresh)
        print("\n--- 4/4: Calendario real de la temporada (1 llamada, barato) ---")
        build_league_schedule_dataset(config, force_refresh=args.refresh)

    if args.backtest_sweep:
        print("\n=== Ingesta del backtest sweep (30 equipos x 15 temporadas) ===\n")
        run_backtest_sweep_ingestion(force_refresh=args.refresh)
