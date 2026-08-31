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
    Stats de playoffs por temporada de un jugador (para modelar fatiga
    acumulada -- src/context/fatigue_accumulation.py). Mismo endpoint que
    fetch_player_career_stats, distinto data frame de resultado
    (SeasonTotalsPostSeason, índice 2 en get_data_frames()). Un jugador sin
    apariciones en playoffs esa temporada simplemente no tiene fila -- eso
    es correcto, no un error.
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
    Calendario completo de la liga para una temporada (todos los equipos,
    todos los partidos) -- para modelar dificultad de calendario
    (src/context/schedule_strength.py). No se filtra por equipo aquí: se
    cachea el calendario completo y se filtra al equipo del config en la
    capa de contexto, así el mismo caché sirve si se cambia de equipo en
    team_config.yaml sin volver a pegarle a la API.
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
    de LA TEMPORADA SIGUIENTE (ver schedule_strength.py) -- no hay forma
    de medir la fuerza real de un rival en una temporada que aún no se ha
    jugado, así que se usa el récord de la temporada anterior como mejor
    aproximación disponible.
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
    Game log de un equipo vía TeamGameLogs (plural, distinto endpoint de
    fetch_team_game_log) -- incluye PLUS_MINUS, que TeamGameLog (singular)
    no trae. Necesario para estimar Net Rating por partido en
    src/context/performance_curve.py. season_type="Playoffs" para traer
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
    temporada (`leaguedashplayerstats` con measure_type="Advanced").
    UNA sola llamada devuelve la liga entera -- es la ingesta más barata
    del proyecto por dato obtenido (~16 llamadas cubren todo el sweep).

    Trae `NET_RATING` y `PIE`, que src/advanced_impact.py mezcla con el
    Game Score para formar la métrica de impacto compuesta. El Game Score
    de Hollinger se calcula desde la caja (PlayerCareerStats) y es
    puramente ofensivo: no ve la defensa más allá de robos/tapones. Estas
    dos métricas sí la ven -- ver el docstring de advanced_impact.py para
    la medición que justifica integrarlas y su caveat de contaminación
    por contexto de equipo.
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    cache_path = raw_dir / "league_advanced_player_stats" / f"{season}.csv"

    def _fetch():
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, measure_type_detailed_defense="Advanced"
        )
        return stats.get_data_frames()[0]

    # OJO con la columna MIN de este endpoint: con measure_type="Advanced"
    # viene en minutos POR PARTIDO, no en totales, y NO cambia aunque se
    # pase per_mode_detailed="Totals" (comprobado contra la API: mediana
    # 19.5 con ambos valores). Es al revés que PlayerCareerStats, de donde
    # sale todo el resto del proyecto. La conversión a totales (MIN * GP)
    # la hace advanced_impact.load_advanced_stats -- se deja ahí y no aquí
    # para que el CSV crudo siga siendo un reflejo fiel de la respuesta.

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_league_hustle_stats(
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Hustle stats de todos los jugadores de la liga en una temporada
    (`leaguehustlestatsplayer`): CONTESTED_SHOTS, DEFLECTIONS,
    CHARGES_DRAWN, SCREEN_ASSISTS, LOOSE_BALLS_RECOVERED, BOX_OUTS.
    Misma ingesta barata que fetch_league_advanced_player_stats -- UNA
    llamada por temporada, liga entera.

    Motivación: investigar si aportan señal de "defensa/juego en equipo
    sin balón" que ni Game Score (puramente ofensivo, ver
    aging_curve.py) ni NET_RATING/PIE (ver advanced_impact.py) capturan
    -- ver scripts/experiments/hustle_stats_signal.py para la
    investigación en sí; esta función SOLO ingiere, no decide si el dato
    aporta.

    LIMITACIÓN DE DISPONIBILIDAD real (no un bug): la NBA solo trackea
    hustle stats desde la temporada 2015-16 (SportVU/Second Spectrum).
    Temporadas anteriores devuelven un DataFrame VACÍO (no una excepción)
    -- `build_league_hustle_stats_dataset` lo detecta y salta esa
    temporada sin abortar el resto.
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
    D_FG_PCT (% de tiro REAL del rival cuando este jugador es el
    defensor más cercano), NORMAL_FG_PCT (% de tiro normal de esos
    mismos rivales) y PCT_PLUSMINUS = D_FG_PCT - NORMAL_FG_PCT --
    negativo significa que el rival tira PEOR de lo normal defendido por
    este jugador, la señal de impacto defensivo más directa que expone
    nba_api (más que los hustle stats de fetch_league_hustle_stats, que
    miden actividad/esfuerzo, no si ese esfuerzo de verdad impide
    anotar -- ver scripts/experiments/hustle_stats_signal.py, resultado
    negativo, y scripts/experiments/pt_defend_signal.py para esta
    investigación).

    UNA llamada por temporada, liga entera -- misma ingesta barata que
    fetch_league_advanced_player_stats/fetch_league_hustle_stats.
    `CLOSE_DEF_PERSON_ID` es el player_id del defensor (no `PLAYER_ID`,
    nombre de columna distinto de cualquier otro endpoint de este
    proyecto -- OJO al combinar).

    LIMITACIÓN DE DISPONIBILIDAD real (no un bug): disponible desde
    2013-14 (Second Spectrum) -- temporadas anteriores devuelven un
    DataFrame VACÍO, no una excepción, mismo patrón que hustle stats.
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
    Posición REAL de un jugador (CommonPlayerInfo, columna `POSITION` --
    p. ej. "Guard", "Forward-Center") vía player_id, independiente de a
    qué equipo esté asignado. Necesario para el roster HIPOTÉTICO propio:
    a diferencia de un roster real, `CommonTeamRoster` (fetch_team_roster)
    no sirve aquí porque devolvería el roster REAL de esa franquicia, no
    los jugadores inventados del config -- hace falta un endpoint por
    JUGADOR, igual que fetch_player_career_stats.

    OJO: el formato de POSITION difiere entre este endpoint (palabras
    completas: "Guard", "Forward") y CommonTeamRoster (abreviado: "G",
    "F") -- ambos comparten la PRIMERA LETRA, que es lo único que usa
    `champion_profiles.POSITION_GROUPS` para agrupar, así que no hace
    falta normalizar.
    """
    from nba_api.stats.endpoints import commonplayerinfo

    cache_path = raw_dir / "player_common_info" / f"{player_id}.csv"

    def _fetch():
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        return info.get_data_frames()[0]

    return _cached_fetch(cache_path, _fetch, force_refresh=force_refresh)


def fetch_team_roster(
    team_id: int,
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Roster real de un equipo en una temporada concreta -- necesario para
    el backtesting (src/backtesting.py): a diferencia del roster
    hipotético en team_config.yaml, aquí se necesita saber quién jugó
    REALMENTE en cada `historical_comparable` para poder correr el mismo
    pipeline de proyección/riesgo retrospectivamente.
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
    un WinPCT genérico. Usa la tabla estática de 30 team_id ya definida en
    context.opponent_weighting (mismo hecho de liga, sin duplicar).
    Guarda data/processed/league_rosters.csv.

    ADVERTENCIA DE COSTE: 30 llamadas a CommonTeamRoster. Deliberadamente
    NO forma parte de run_full_pipeline (que corre cada vez que se ejecuta
    data_pipeline.py sin --refresh) -- se llama aparte, una vez, porque el
    siguiente paso (build_league_player_stats_dataset) es mucho más caro
    todavía (~450 jugadores x 2 llamadas).
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
    con el mismo pipeline (aging_curve/injury_model/fatigue_accumulation)
    que el roster propio. Reutiliza fetch_player_career_stats /
    fetch_player_playoff_career_stats, ningún endpoint nuevo. Guarda
    data/processed/league_player_career_stats.csv y
    data/processed/league_player_playoff_career_stats.csv.

    ADVERTENCIA DE COSTE: ~450 jugadores x 2 llamadas cada uno -- la
    ingesta más cara del proyecto (probablemente 20-30+ min la primera
    vez). Cacheado después, como todo lo demás.
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

    NO se puede sacar de `league_rosters.csv` (CommonTeamRoster no trae
    nacionalidad, solo POSITION) -- hace falta un endpoint POR JUGADOR
    (mismo `fetch_player_common_info` que usa build_roster_positions_dataset
    para el roster propio, y de paso trae POSITION también, aunque aquí no
    hace falta -- ya está en league_rosters.csv).

    Necesaria para el chequeo de cuota de nacionalidad del All-Star
    (mínimo real de la NBA: 16 jugadores de EE.UU. / 8 internacionales
    sobre el total de 24 seleccionados) -- ver
    awards_projection.check_all_star_nationality_quota. Sin este CSV el
    chequeo simplemente no se puede hacer sobre los 30 equipos (se
    degrada, no falla -- ver dashboard/data_loader.py).

    ADVERTENCIA DE COSTE: ~450-500 llamadas, una por jugador único de la
    liga -- del mismo orden que build_league_player_stats_dataset. Opt-in
    vía --league, no forma parte de run_full_pipeline.
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
    build_backtest_sweep_rosters_dataset -- ver docstrings de cada uno.
    Un caso que falle tras agotar reintentos (stats.nba.com devuelve
    ocasionalmente una respuesta rara para algún equipo/temporada
    concreto) se SALTA con un aviso en vez de abortar toda la ingesta --
    ver el mismo patrón en backtesting._run_backtest_cases()."""
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
    las temporadas en config["backtest_sweep"]["seasons"] (por defecto
    450 casos). ADVERTENCIA DE COSTE: ver docstring de
    build_backtest_sweep_dataset en backtesting.py y la sección
    --backtest-sweep en el CLI de este módulo. Guarda
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
    ÚNICAMENTE (no por PLAYER_ID+caso) -- un jugador que aparece en varios
    casos (misma franquicia varias temporadas, o distintas) tiene la MISMA
    carrera completa (PlayerCareerStats devuelve todas sus temporadas
    pase lo que pase), así que se descarga/concatena UNA sola vez. Esto
    importa especialmente para el sweep de 450 casos: duplicar filas por
    caso no solo desperdicia espacio, rompería `_most_recent_n_seasons`
    en aging_curve.py (que ordena por año y toma las N más recientes --
    con copias duplicadas de la MISMA temporada por aparecer en varios
    casos, tomaría copias de la temporada más reciente en vez de las N
    temporadas distintas más recientes, colapsando el baseline
    ponderado por recencia a una sola temporada).

    Un jugador cuya descarga falle tras agotar reintentos (stats.nba.com
    devuelve ocasionalmente una respuesta rara, p. ej. sin 'resultSet',
    para algún player_id concreto) se SALTA con un aviso en vez de
    abortar toda la ingesta -- con 450 casos y miles de jugadores, un
    fallo puntual de la API es mucho más probable que con el puñado de
    jugadores de los 4 comparables narrativos.
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
    Reutiliza fetch_player_career_stats / fetch_player_playoff_career_stats
    (mismas funciones que usa el roster hipotético del config, ningún
    endpoint nuevo aquí). Guarda
    data/processed/historical_comparables_player_career_stats.csv y
    data/processed/historical_comparables_player_playoff_career_stats.csv.
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
    defecto -- ver build_backtest_sweep_rosters_dataset). Es la parte más
    cara de la ingesta del sweep: cada jugador ÚNICO que apareció en
    cualquiera de los 450 rosters se descarga una vez (ver
    _download_player_stats_for_cases). Guarda
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
    del config (ver fetch_player_common_info -- misma llamada, dos
    columnas: `POSITION` y `COUNTRY`, no hace falta pedirlas por
    separado). Guarda data/processed/roster_positions.csv (player_id,
    position, country).

    `position` es necesaria para los quintetos All-NBA / All-Defensive
    (2 bases/escoltas + 2 aleros/ala-pívots + 1 pívot, ver
    awards_projection.py) -- sin posición real no hay forma de respetar
    esos cupos, y `role_expected` de team_config.yaml es deliberadamente
    descriptivo, no una entrada de cálculo (ver lineup_synergy.py).

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
    UNIÓN de las del `backtest_sweep` (para calibrar y validar) y las N
    anteriores a `config["team"]["season"]` (para proyectar el roster
    hipotético). Se resuelve aquí en vez de en cada llamante para que
    ambos caminos usen exactamente el mismo CSV -- el mismo motivo por el
    que existe `normalize_rotation_minutes` compartida.

    OJO: hace falta también el lookback ANTERIOR a la temporada más
    antigua del sweep. Proyectar 2010-11 usa las 3 temporadas previas
    (2007-08 a 2009-10); sin ellas, los 30 casos de la primera temporada
    del sweep se quedan sin métricas avanzadas y caen a Game Score puro,
    lo que mezclaría dos modelos distintos dentro del mismo backtest.
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
    `season` añadida (el endpoint no la devuelve).

    Barato: una llamada por temporada (~19), frente a las ~900 de
    `--league`. Aun así es opt-in y no forma parte de `run_full_pipeline`
    por coherencia con el resto de la ingesta de liga.
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
    mismas temporadas que resolve_advanced_stats_seasons -- reutiliza esa
    resolución para no mantener dos listas de temporadas por separado.
    Guarda data/processed/league_hustle_player_stats.csv.

    Temporadas anteriores a 2015-16 devuelven un DataFrame VACÍO (la NBA
    no trackeaba esto todavía) -- se saltan sin abortar el resto, mismo
    patrón que un caso roto en _run_backtest_cases.
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
    renombrado a `PLAYER_ID` (consistencia con el resto de datasets del
    proyecto).

    Temporadas anteriores a 2013-14 devuelven un DataFrame vacío (Second
    Spectrum no trackeaba todavía) -- se saltan sin abortar el resto.
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
    Net rating REAL de cada pareja de jugadores que compartió cancha esa
    temporada (`leaguedashlineups`, group_quantity=2, measure_type
    Advanced) -- para scripts/experiments/lineup_synergy_signal.py: la
    investigación de si los dos efectos de src/lineup_synergy.py
    (usage_clash, playmaking_spacing_synergy) predicen algo real sobre
    net rating de pareja, o si los pesos actuales (nunca calibrados
    contra datos) están adivinando. UNA llamada por temporada, liga
    entera -- mismo patrón barato que fetch_league_hustle_stats/
    fetch_league_pt_defend_stats. `GROUP_ID` viene como
    "-player_id_a-player_id_b-" (guiones incluidos); el experimento se
    encarga de parsearlo, no este fetcher.
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
    `backtest_sweep` (config["backtest_sweep"]["seasons"], 16 temporadas
    2010-11..2025-26 -- ya el rango que usa el resto de la validación
    empírica de este proyecto, no hace falta uno nuevo). Guarda
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


# Cuatro categorías de seguimiento (`leaguedashptstats`) usadas para
# probar candidatos NUEVOS de sinergia de pareja en
# scripts/experiments/lineup_synergy_signal.py, más allá de
# usage_clash/playmaking_spacing_synergy (que salieron sin apoyo
# empírico -- ver CLAUDE.md): volumen de tiro creado por uno mismo
# (PullUpShot) vs. recibido/catch-and-shoot (CatchShoot) -- pareja
# "tirador con balón + tirador sin balón"; penetraciones (Drives) de un
# manejador vs. presencia interior de un grande -- proxy de pick-and-roll
# (nba_api no expone frecuencia de bloqueo-y-continuación real); y
# volumen de poste (PostTouch) -- pareja "anotador de poste + creador".
LINEUP_SYNERGY_TRACKING_MEASURE_TYPES = ["CatchShoot", "PullUpShot", "Drives", "PostTouch"]


def fetch_league_tracking_stats(
    season: str,
    pt_measure_type: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Una categoría de `leaguedashptstats` (tracking SportVU/Second
    Spectrum) para TODOS los jugadores de una temporada -- UNA llamada,
    mismo patrón barato que fetch_league_hustle_stats/
    fetch_league_pt_defend_stats. `pt_measure_type` debe ser uno de
    LINEUP_SYNERGY_TRACKING_MEASURE_TYPES (u otro valor válido de
    `nba_api.stats.library.parameters.PtMeasureType`, pero ese es el uso
    real de este proyecto).
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
    fila por PLAYER_ID+season, una columna por estadística de cada
    categoría -- todas comparten PLAYER_ID/season como clave, así que un
    merge sucesivo es más simple que concatenar). Guarda
    data/processed/league_tracking_stats.csv. Temporadas o categorías sin
    datos (tracking no disponible antes de cierto año) se saltan sin
    abortar el resto, mismo criterio que build_league_pt_defend_dataset.
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
    Calendario REAL de la liga completa (los 30 equipos), para
    league_simulation.py -- hermana de build_team_schedule_dataset (que
    filtra a UN equipo), reutiliza fetch_league_schedule tal cual, sin
    llamada nueva a la API. Guarda data/processed/league_schedule_full.csv
    con gameDate/homeTeam_teamTricode/awayTeam_teamTricode.

    Dos filtros, ninguno inventa datos:
    - `gameLabel == "Preseason"` -- no es temporada regular.
    - Filas con tricode nulo -- plazas de la fase eliminatoria de la NBA
      Cup todavía sin resolver (los dos equipos se deciden más adelante
      en la temporada real); se descartan en vez de inventarse. Esto
      hace que, mientras la Cup no se resuelva del todo, cada equipo
      tenga menos partidos que `config["simulation"]["games_per_season"]`
      -- limitación temporal real, documentada en
      league_simulation.py, no oculta.
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
    casos (no por caso: varios equipos/casos comparten temporada, y los
    standings son de toda la liga de una vez, no hace falta repetir). Una
    temporada que falle tras agotar reintentos se salta con un aviso en
    vez de abortar toda la ingesta."""
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
    defecto, no 450: los standings son de toda la liga a la vez). Guarda
    data/processed/backtest_sweep_standings.csv.
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
    / build_backtest_sweep_advanced_game_logs_dataset. Un caso que falle
    tras agotar reintentos se salta con un aviso en vez de abortar toda
    la ingesta."""
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
    `resolve_backtest_sweep_cases(config)` (450 casos por defecto: 2
    llamadas cada uno -- regular + playoffs -- la parte más cara de la
    ingesta del sweep junto con los career stats de jugadores). Guarda
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
    la API (~450 casos x ~4 llamadas de equipo, más ~2 llamadas por
    jugador ÚNICO que haya pisado cualquiera de esos rosters) --
    probablemente 1.5-3 horas la primera vez. Opt-in a propósito (flag
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
             "league_simulation.py y para el chequeo de cuota del All-Star). ADVERTENCIA: "
             "~1350 llamadas a la API, la ingesta más cara del proyecto (30-45+ min la "
             "primera vez). Opt-in a propósito, no forma parte del pipeline normal.",
    )
    parser.add_argument(
        "--backtest-sweep",
        action="store_true",
        help="Además del pipeline normal, descarga los 30 equipos NBA para cada "
             "temporada en config['backtest_sweep']['seasons'] (450 casos por "
             "defecto: 15 temporadas). ADVERTENCIA: la ingesta MÁS CARA del "
             "proyecto -- del orden de miles de llamadas a la API, 1.5-3 horas la "
             "primera vez. Opt-in a propósito, no forma parte del pipeline normal "
             "ni de --league.",
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
