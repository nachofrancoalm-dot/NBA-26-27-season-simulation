"""
awards_projection.py

Heurísticas de premios individuales (MVP, DPOY, ROY, MIP, 6MOY) y de
equipo (COY) sobre los datos YA CALCULADOS por el resto del pipeline
(aging_curve.py, league_simulation.py, data_pipeline.py). NINGUNO de
estos premios existe todavía para la temporada simulada -- son lecturas
narrativas de las proyecciones, calibradas para ser razonables, pero NO
pretenden predecir la votación real de los medios (que pesa narrativa,
"eye test", historias de temporada, etc. que este proyecto no puede
modelar a partir de solo datos de caja). Cada función documenta
explícitamente su fórmula y sus limitaciones.

Todas las funciones son puras: reciben los DataFrames ya cargados (por
dashboard/data_loader.py u otro caller) y no leen CSV ni llaman a la API
por su cuenta -- así funcionan igual sobre el roster propio
(aging_curve_projection.csv, 10 jugadores) que sobre los 30 equipos de la
liga (league_player_projections.csv, ~500+ jugadores), y son fáciles de
testear con datos sintéticos.

MÉTRICA BASE COMPARTIDA: "valor de temporada" = game_score_per36 (Game
Score de Hollinger proyectado, ver aging_curve.py) * projected_total_minutes
/ 36 -- el Game Score total que un jugador aportaría en toda la temporada
proyectada. Combina volumen y eficiencia en una sola cifra, igual que
game_score_per36, pero recompensando también jugar muchos minutos (un
candidato a MVP que juega 20 min/partido con una tasa altísima no debería
ganarle a uno que juega 36 min/partido con una tasa ligeramente menor).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from aging_curve import compute_game_score_per36, compute_per36_stats
from champion_profiles import POSITION_GROUPS
from season_utils import dedupe_traded_seasons, season_start_year
from simulation import compute_expected_games_played

# Filtra jugadores de rotación marginal (muestra pequeña, ruido) de los
# premios "de rotación completa" (MVP, DPOY). 6MOY y ROY usan un umbral
# más bajo porque suplentes y rookies juegan estructuralmente menos.
DEFAULT_MIN_MPG_MVP_DPOY = 20.0
DEFAULT_MIN_MPG_SIXTH_MAN = 12.0
DEFAULT_MIN_MPG_ROY = 10.0
DEFAULT_MIN_MPG_MIP = 15.0

# Umbral REAL de la NBA (desde 2023-24, "Player Participation Policy")
# para optar a premios de FIN DE TEMPORADA (quintetos All-NBA/All-
# Defensive, MVP/DPOY/etc.) -- no aplica al All-Star, que se vota a mitad
# de temporada sin requisito de partidos. A petición del usuario.
DEFAULT_MIN_GAMES_SEASON_AWARDS = 65

# Cupos por quinteto, formato clásico 2 bases/escoltas + 2 aleros/ala-
# pívots + 1 pívot -- el mismo que usa la NBA real para All-NBA y All-
# Defensive (a diferencia del All-Star moderno, que ya no fuerza cupos de
# posición en las reservas). Claves = primera letra de POSITION_GROUPS.
DEFAULT_TEAM_POSITION_SLOTS: Dict[str, int] = {"G": 2, "F": 2, "C": 1}

# Nº de quintetos: la NBA real elige 3 equipos All-NBA (Primero/Segundo/
# Tercero) y 2 equipos All-Defensive (Primero/Segundo).
ALL_NBA_TEAM_NAMES = ["Primer equipo All-NBA", "Segundo equipo All-NBA", "Tercer equipo All-NBA"]
ALL_DEFENSIVE_TEAM_NAMES = ["Primer equipo All-Defensive", "Segundo equipo All-Defensive"]

# Total de seleccionados All-Star por conferencia en la NBA real (12 c/u,
# incluyendo titulares y reservas -- este proyecto no distingue entre
# ambos, ver compute_all_star_selections).
DEFAULT_ALL_STARS_PER_CONFERENCE = 12


def _season_value(df: pd.DataFrame) -> pd.Series:
    """game_score_per36 * projected_total_minutes / 36 -- ver docstring del módulo."""
    return df["game_score_per36"] * df["projected_total_minutes"] / 36.0


def _avg_mpg(df: pd.DataFrame, games_per_season: int) -> pd.Series:
    return df["projected_total_minutes"] / games_per_season


# Stats por-partido para comparar candidatos de un vistazo, a petición
# del usuario -- YA vienen calculadas en `player_df` (PER_GAME_STATS +
# FG%/3P% de dashboard/data_loader.py y del builder equivalente de
# league_simulation.py), así que aquí solo hace falta incluirlas en la
# selección de columnas de cada función, no recalcularlas.
OFFENSIVE_COMPARISON_STATS = ["PPG", "RPG", "APG", "SPG", "BPG", "FG%", "3P%"]
# DPOY es puramente defensivo -- PPG/APG/tiro no son relevantes para ese
# premio y solo añadirían ruido a la tabla. PFPG (faltas por partido) se
# añade como extra: es el único término NEGATIVO de defensive_score_per36,
# así que da contexto de por qué un jugador con buenos robos/tapones
# puede quedar penalizado.
DEFENSIVE_COMPARISON_STATS = ["RPG", "SPG", "BPG", "PFPG"]


def _attach_team_record(df: pd.DataFrame, team_record: Optional[Dict[Any, str]]) -> pd.DataFrame:
    """Añade la columna `team_record` (string "W-L") si se pasa el dict -- ver compute_awards_summary."""
    if team_record and "player_id" in df.columns:
        df["team_record"] = df["player_id"].map(team_record)
    return df


def compute_mvp_candidates(
    player_df: pd.DataFrame,
    games_per_season: int,
    team_win_pct: Optional[Dict[Any, float]] = None,
    team_record: Optional[Dict[Any, str]] = None,
    min_mpg: float = DEFAULT_MIN_MPG_MVP_DPOY,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    MVP heurístico: valor de temporada ponderado por el % de victorias
    proyectado del equipo -- el MVP real casi siempre juega en un equipo
    ganador, no solo acumula estadísticas en un equipo mediocre.

    `team_win_pct`: dict/Series player_id -> % de victorias del equipo en
    [0, 1]. Si no se pasa (o falta para un jugador), se asume 0.5 -- el
    ranking queda sin ajuste de equipo (útil para el roster propio, donde
    todos los jugadores comparten el mismo equipo y el ajuste no aporta
    nada).

    `team_record`: dict/Series player_id -> récord "W-L" del equipo (solo
    para mostrar, no afecta al ranking -- ver `_attach_team_record`).
    Incluye además OFFENSIVE_COMPARISON_STATS (PPG/RPG/APG/SPG/BPG/FG%/3P%)
    para comparar candidatos de un vistazo, a petición del usuario.
    """
    df = player_df.copy()
    df["mpg"] = _avg_mpg(df, games_per_season)
    df = df[df["mpg"] >= min_mpg]
    df["season_value"] = _season_value(df)
    if team_win_pct:
        df["team_win_pct"] = df["player_id"].map(team_win_pct).fillna(0.5)
    else:
        df["team_win_pct"] = 0.5
    df["mvp_score"] = df["season_value"] * (0.5 + df["team_win_pct"])
    df = _attach_team_record(df, team_record)

    cols = [
        c for c in
        ["player_id", "player_name", "team_abbreviation", "team_record", "mpg"]
        + OFFENSIVE_COMPARISON_STATS
        + ["season_value", "team_win_pct", "mvp_score"]
        if c in df.columns
    ]
    return df.sort_values("mvp_score", ascending=False)[cols].head(top_n).reset_index(drop=True)


def compute_dpoy_candidates(
    player_df: pd.DataFrame,
    games_per_season: int,
    team_record: Optional[Dict[Any, str]] = None,
    min_mpg: float = DEFAULT_MIN_MPG_MVP_DPOY,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    DPOY heurístico: "defensive_score_per36" = 1.5*STL_per36 +
    1.5*BLK_per36 + 0.3*DREB_per36 - 0.2*PF_per36 -- pesos de este
    proyecto, NO una métrica oficial. Limitación importante: el box score
    no captura la mayoría de la defensa individual real (contención de
    tiro, rotaciones, defensa de perímetro sin robo/tapón) -- este
    proxy solo ve robos, tapones, rebote defensivo y faltas.
    Escalado a valor de temporada completa (recompensa también jugar
    muchos minutos, no solo tasas altas en pocos minutos).

    Incluye DEFENSIVE_COMPARISON_STATS (RPG/SPG/BPG + PFPG, faltas por
    partido -- el único término NEGATIVO de la fórmula) MÁS
    OFFENSIVE_COMPARISON_STATS completo (PPG/APG/tiro incluidos) -- en un
    primer diseño se omitía PPG/APG/tiro aquí porque "DPOY es un premio
    defensivo, no aportan al RANKING", pero el ranking (dpoy_score) sigue
    siendo puramente defensivo; estas columnas son solo para que la vista
    previa al pasar el ratón en webapp/ muestre el mismo set de stats que
    el resto de premios (a petición explícita del usuario).
    """
    df = player_df.copy()
    df["mpg"] = _avg_mpg(df, games_per_season)
    df = df[df["mpg"] >= min_mpg]
    df["defensive_score_per36"] = (
        1.5 * df["STL_per36_projected"]
        + 1.5 * df["BLK_per36_projected"]
        + 0.3 * df["DREB_per36_projected"]
        - 0.2 * df["PF_per36_projected"]
    )
    df["dpoy_score"] = df["defensive_score_per36"] * df["projected_total_minutes"] / 36.0
    if "PF_projected" in df.columns:
        df["PFPG"] = (df["PF_projected"] / games_per_season).round(1)
    df = _attach_team_record(df, team_record)

    cols = [
        c for c in
        ["player_id", "player_name", "team_abbreviation", "team_record", "mpg"]
        + DEFENSIVE_COMPARISON_STATS
        + OFFENSIVE_COMPARISON_STATS
        + ["defensive_score_per36", "dpoy_score"]
        if c in df.columns
    ]
    # dict.fromkeys en vez de set(): DEFENSIVE_COMPARISON_STATS y
    # OFFENSIVE_COMPARISON_STATS comparten RPG/SPG/BPG -- deduplicar sin
    # perder el orden (pandas rechaza columnas repetidas en un selector).
    cols = list(dict.fromkeys(cols))
    return df.sort_values("dpoy_score", ascending=False)[cols].head(top_n).reset_index(drop=True)


def compute_bench_player_ids(career_stats_df: pd.DataFrame) -> Set[Any]:
    """
    Jugadores que en su temporada REAL más reciente registrada empezaron
    (GS) menos de la mitad de los partidos que jugaron (GP) -- "salieron
    del banquillo" la mayoría de las veces. Aproximación de rol de
    equipo: no distingue a un titular que perdió su puesto a mitad de
    temporada de alguien que siempre fue suplente.
    """
    if career_stats_df.empty or "GS" not in career_stats_df.columns:
        return set()
    df = career_stats_df.assign(_start_year=career_stats_df["SEASON_ID"].apply(season_start_year))
    latest = df.sort_values("_start_year", ascending=False).groupby("PLAYER_ID").head(1)
    is_bench = latest["GS"] < (latest["GP"] / 2)
    return set(latest.loc[is_bench, "PLAYER_ID"])


def compute_sixth_man_candidates(
    player_df: pd.DataFrame,
    bench_player_ids: Iterable[Any],
    games_per_season: int,
    team_record: Optional[Dict[Any, str]] = None,
    min_mpg: float = DEFAULT_MIN_MPG_SIXTH_MAN,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    6MOY heurístico: entre los jugadores marcados como "banquillo" (ver
    compute_bench_player_ids), el de mayor valor de temporada. Umbral de
    minutos más bajo que MVP/DPOY porque los suplentes juegan
    estructuralmente menos. Incluye OFFENSIVE_COMPARISON_STATS, ver
    compute_mvp_candidates.
    """
    bench_ids = set(bench_player_ids)
    df = player_df[player_df["player_id"].isin(bench_ids)].copy()
    df["mpg"] = _avg_mpg(df, games_per_season)
    df = df[df["mpg"] >= min_mpg]
    df["season_value"] = _season_value(df)
    df = _attach_team_record(df, team_record)

    cols = [
        c for c in
        ["player_id", "player_name", "team_abbreviation", "team_record", "mpg"]
        + OFFENSIVE_COMPARISON_STATS
        + ["season_value"]
        if c in df.columns
    ]
    return df.sort_values("season_value", ascending=False)[cols].head(top_n).reset_index(drop=True)


def compute_rookie_player_ids(career_stats_df: pd.DataFrame) -> Set[Any]:
    """
    Un jugador se considera rookie si TODA su historia en
    roster_career_stats.csv / league_player_career_stats.csv (pedida vía
    nba_api PlayerCareerStats, que devuelve la carrera COMPLETA del
    jugador) cae en una sola temporada -- es decir, la temporada más
    reciente es literalmente su único año en la NBA según los datos
    disponibles. Limitación: si por algún motivo los datos de carrera de
    un jugador veterano llegaran incompletos, esta heurística lo marcaría
    (incorrectamente) como rookie -- no hay forma de distinguir eso sin
    datos externos de "años de experiencia" que este proyecto no descarga.
    """
    if career_stats_df.empty:
        return set()
    seasons_per_player = career_stats_df.groupby("PLAYER_ID")["SEASON_ID"].nunique()
    return set(seasons_per_player[seasons_per_player <= 1].index)


def compute_roy_candidates(
    player_df: pd.DataFrame,
    rookie_player_ids: Iterable[Any],
    games_per_season: int,
    team_record: Optional[Dict[Any, str]] = None,
    min_mpg: float = DEFAULT_MIN_MPG_ROY,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    ROY heurístico: entre los jugadores marcados como rookies (ver
    compute_rookie_player_ids), el de mayor valor de temporada. Incluye
    OFFENSIVE_COMPARISON_STATS, ver compute_mvp_candidates.
    """
    rookie_ids = set(rookie_player_ids)
    df = player_df[player_df["player_id"].isin(rookie_ids)].copy()
    df["mpg"] = _avg_mpg(df, games_per_season)
    df = df[df["mpg"] >= min_mpg]
    df["season_value"] = _season_value(df)
    df = _attach_team_record(df, team_record)

    cols = [
        c for c in
        ["player_id", "player_name", "team_abbreviation", "team_record", "mpg"]
        + OFFENSIVE_COMPARISON_STATS
        + ["season_value"]
        if c in df.columns
    ]
    return df.sort_values("season_value", ascending=False)[cols].head(top_n).reset_index(drop=True)


def compute_mip_candidates(
    career_stats_df: pd.DataFrame,
    min_minutes_per_game: float = DEFAULT_MIN_MPG_MIP,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    MIP heurístico: compara el Game Score por-36 REAL (no proyectado) de
    las DOS temporadas más recientes de cada jugador -- el salto de
    rendimiento que YA ocurrió, no una proyección de la temporada
    simulada. MIP se vota sobre lo que un jugador ya mejoró respecto al
    año anterior, así que usar datos reales es más fiel al espíritu del
    premio que MVP/DPOY (que sí usan la proyección). Requiere al menos 2
    temporadas en los datos y un mínimo de minutos/partido en AMBAS
    (evita que un salto de bajísimo volumen a bajo volumen parezca una
    mejora enorme). Excluye implícitamente a los rookies -- solo tienen 1
    temporada, ver compute_roy_candidates.
    """
    if career_stats_df.empty:
        return pd.DataFrame(
            columns=["player_id", "player_name", "latest_season", "previous_game_score_per36", "latest_game_score_per36", "improvement"]
        )

    rows = []
    for player_id, group in career_stats_df.groupby("PLAYER_ID"):
        per36 = compute_per36_stats(group).assign(
            _start_year=lambda d: d["SEASON_ID"].apply(season_start_year)
        )
        recent_two = per36.sort_values("_start_year", ascending=False).head(2).reset_index(drop=True)
        if len(recent_two) < 2:
            continue

        latest, previous = recent_two.iloc[0], recent_two.iloc[1]
        latest_mpg = latest["MIN"] / latest["GP"] if latest["GP"] else 0.0
        previous_mpg = previous["MIN"] / previous["GP"] if previous["GP"] else 0.0
        if latest_mpg < min_minutes_per_game or previous_mpg < min_minutes_per_game:
            continue

        per36_cols = [c for c in per36.columns if c.endswith("_per36")]
        latest_gs = compute_game_score_per36({c: latest[c] for c in per36_cols})
        previous_gs = compute_game_score_per36({c: previous[c] for c in per36_cols})

        rows.append(
            {
                "player_id": player_id,
                "player_name": group["player_name"].iloc[0] if "player_name" in group.columns else None,
                "latest_season": latest["SEASON_ID"],
                "previous_game_score_per36": previous_gs,
                "latest_game_score_per36": latest_gs,
                "improvement": latest_gs - previous_gs,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("improvement", ascending=False).head(top_n).reset_index(drop=True)


PREV_SEASON_STATS_COLUMNS = ["prev_PPG", "prev_RPG", "prev_APG", "prev_SPG", "prev_BPG", "prev_FG%", "prev_3P%", "prev_season"]


def compute_latest_real_season_stats(career_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    PPG/RPG/APG/SPG/BPG/FG%/3P% REALES de la última temporada YA jugada
    de cada jugador (una fila por jugador) -- para que el popup de MIP
    (webapp/) pueda comparar la temporada PROYECTADA (mergeada aparte,
    desde player_df, ver dashboard.data_loader.compute_awards_summary)
    contra la temporada real inmediatamente anterior, a petición del
    usuario. DISTINTO de `previous_game_score_per36` de
    compute_mip_candidates: esa usa la PENÚLTIMA temporada real (para
    calcular cuánto mejoró un jugador de un año real a otro); esto usa
    la ÚLTIMA real, la que precede a la proyección.
    """
    if career_stats_df.empty:
        return pd.DataFrame(columns=["player_id"] + PREV_SEASON_STATS_COLUMNS)

    # dedupe_traded_seasons() agrupa por SEASON_ID -- BUG REAL encontrado
    # al probar esto contra los 577 jugadores reales de la liga: llamarla
    # sobre el DataFrame multi-jugador completo compara temporadas de
    # jugadores DISTINTOS entre sí (si CUALQUIER jugador de la liga fue
    # traspasado en la temporada X, "has_tot" sale True para la
    # temporada X de TODOS los jugadores, y a quienes no fueron
    # traspasados esa temporada se les descarta su única fila de esa
    # temporada por no ser 'TOT') -- para Kawhi Leonard esto acababa
    # dejando su temporada de ROOKIE (2011-12) como "la más reciente"
    # porque casi todas sus demás temporadas reales se descartaban por
    # trades de OTROS jugadores. Hay que aplicarla POR JUGADOR (mismo
    # criterio que ya usa compute_mip_candidates, que itera con
    # `.groupby("PLAYER_ID")` antes de llamar a compute_per36_stats, que
    # a su vez llama a dedupe_traded_seasons ya sobre un solo jugador).
    df = pd.concat(
        [dedupe_traded_seasons(group) for _, group in career_stats_df.groupby("PLAYER_ID")], ignore_index=True
    )
    df = df[df["GP"] > 0]
    df["_start_year"] = df["SEASON_ID"].apply(season_start_year)
    latest = df.sort_values("_start_year", ascending=False).groupby("PLAYER_ID").head(1)

    # FG_PCT/FG3_PCT no siempre están (p.ej. career_stats_df construido a
    # mano en tests, sin las columnas de tiro) -- se degradan a NaN en
    # vez de romper el resto de esta función, mismo criterio que el
    # resto del proyecto ante una columna opcional ausente.
    gp = latest["GP"]
    fg_pct = latest["FG_PCT"] * 100 if "FG_PCT" in latest.columns else np.nan
    fg3_pct = latest["FG3_PCT"] * 100 if "FG3_PCT" in latest.columns else np.nan
    return pd.DataFrame(
        {
            "player_id": latest["PLAYER_ID"],
            "prev_PPG": (latest["PTS"] / gp).round(1),
            "prev_RPG": (latest["REB"] / gp).round(1),
            "prev_APG": (latest["AST"] / gp).round(1),
            "prev_SPG": (latest["STL"] / gp).round(1),
            "prev_BPG": (latest["BLK"] / gp).round(1),
            "prev_FG%": fg_pct if isinstance(fg_pct, float) else fg_pct.round(1),
            "prev_3P%": fg3_pct if isinstance(fg3_pct, float) else fg3_pct.round(1),
            "prev_season": latest["SEASON_ID"],
        }
    )


def compute_coy_candidates(
    team_wins_df: pd.DataFrame,
    prior_wins_by_team: Dict[str, float],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    COY heurístico de EQUIPO, no de entrenador -- este proyecto no modela
    entrenadores en absoluto. Aproxima el proxy más común que usan los
    votantes reales de COY: el equipo que más mejoró respecto al récord
    REAL de la temporada anterior (`prior_season_standings.csv`). Un
    salto grande de victorias suele reflejar un cambio de sistema/gestión
    de plantilla más que solo mejores jugadores -- pero es una
    correlación, no una medición directa de la calidad de un entrenador.

    `team_wins_df`: columnas team_abbreviation, wins_mean (victorias
    medias proyectadas para la temporada simulada).
    `prior_wins_by_team`: dict team_abbreviation -> victorias REALES de
    la temporada anterior.
    """
    df = team_wins_df.copy()
    df["prior_wins"] = df["team_abbreviation"].map(prior_wins_by_team)
    df = df.dropna(subset=["prior_wins"])
    df["win_improvement"] = df["wins_mean"] - df["prior_wins"]

    cols = [c for c in ["team_abbreviation", "conference", "prior_wins", "wins_mean", "win_improvement"] if c in df.columns]
    return df.sort_values("win_improvement", ascending=False)[cols].head(top_n).reset_index(drop=True)


def _expected_games_played(df: pd.DataFrame, games_per_season: int) -> pd.Series:
    """
    Partidos jugados ESPERADOS de la temporada simulada (ver
    simulation.compute_expected_games_played) si hay `risk_score`; si
    falta (p. ej. injury_risk.csv no corrido), se asume la temporada
    completa -- degradar a "todos elegibles" es más seguro que excluir a
    todo el mundo por un dato que no llegó.
    """
    if "risk_score" not in df.columns:
        return pd.Series(float(games_per_season), index=df.index)
    risk = df["risk_score"].fillna(0.0).to_numpy()
    return pd.Series(compute_expected_games_played(risk, games_per_season), index=df.index)


def _position_group(df: pd.DataFrame) -> pd.Series:
    """
    Primera letra de `position` (Guard/Forward/Center o G/F/C, ambos
    formatos conviven -- ver data_pipeline.fetch_player_common_info),
    limitada a G/F/C. Jugadores sin `position` (roster_positions.csv no
    corrido, o liga sin ese dato) quedan como None -- no participan en
    los quintetos, que exigen cupo de posición, pero sí pueden aparecer
    en el All-Star (que no distingue posición aquí, ver
    compute_all_star_selections).
    """
    if "position" not in df.columns:
        return pd.Series(None, index=df.index, dtype=object)
    first_letter = df["position"].astype(str).str[0]
    return first_letter.where(first_letter.isin(POSITION_GROUPS.keys()))


# Titulares por conferencia (elegidos en la vida real por voto: 50% fans
# + 25% jugadores actuales + 25% panel de medios, SIN distinción de
# posición desde la temporada 2025-26) y reservas (elegidos por los
# entrenadores). 5 + 7 = 12 por conferencia, el tamaño real del roster.
DEFAULT_ALL_STAR_STARTERS_PER_CONFERENCE = 5

# Mínimos reales de nacionalidad de la NBA sobre el total de 24
# seleccionados: si la votación natural no produce al menos estos
# números, la liga añade jugadores (decisión discrecional del
# comisionado) -- ver check_all_star_nationality_quota.
DEFAULT_MIN_US_ALL_STARS = 16
DEFAULT_MIN_INTERNATIONAL_ALL_STARS = 8


def compute_all_star_selections(
    player_df: pd.DataFrame,
    games_per_season: int,
    team_win_pct: Optional[Dict[Any, float]] = None,
    conference_col: str = "conference",
    n_per_conference: int = DEFAULT_ALL_STARS_PER_CONFERENCE,
    n_starters_per_conference: int = DEFAULT_ALL_STAR_STARTERS_PER_CONFERENCE,
) -> pd.DataFrame:
    """
    All-Star heurístico: SIN restricción de partidos jugados (el All-Star
    se juega a mitad de temporada, antes de que el requisito de 65
    partidos de fin de temporada pueda siquiera evaluarse) ni de minutos
    mínimos -- solo valor de temporada, igual que MVP pero sin filtrar
    rotación marginal (un candidato a All-Star real casi siempre supera
    ese umbral de todos modos, así que el filtro no cambiaría nada en la
    práctica, y omitirlo simplifica la función). SIN distinción de
    posición -- coincide con la regla real desde esta temporada.

    METODOLOGÍA REAL vs. LO QUE ESTE PROYECTO PUEDE HACER: los 5
    titulares de cada conferencia se votan 50% fans + 25% jugadores
    actuales + 25% panel de medios; los 7 reservas los eligen los
    entrenadores. Este proyecto NO tiene forma de obtener ninguno de esos
    tres votos (no existe ese dato en `nba_api` ni en ningún sitio
    accesible) ni de modelar el criterio de un entrenador -- así que
    `season_value` actúa como proxy ÚNICO para las tres electorados y
    para el criterio de los entrenadores, algo que la vida real nunca
    hace. La columna `selection_type` ("Titular"/"Reserva") es solo una
    ETIQUETA sobre el ranking de ese proxy (los `n_starters_per_conference`
    de mayor valor = "Titular"), no una simulación de un voto real.

    Si `player_df` trae `conference_col` con más de una conferencia
    distinta (scope "league", los 30 equipos), se eligen los
    `n_per_conference` de mayor valor de temporada DE CADA conferencia
    (12 c/u = 24 en total, el tamaño real del roster de la NBA).

    Si no hay conferencia (scope "own", un solo equipo -- o si falta la
    columna), se toma un único top `n_per_conference` sobre todo
    `player_df`, sin split -- con un roster de ~13 jugadores esto en la
    práctica selecciona a casi todo el mundo, lo cual es razonable: no
    hay forma de simular "contra el resto de la NBA" con un solo equipo.
    """
    df = player_df.copy()
    df["season_value"] = _season_value(df)
    if team_win_pct:
        df["team_win_pct"] = df["player_id"].map(team_win_pct).fillna(0.5)

    cols = [
        c for c in ["player_id", "player_name", "team_abbreviation", conference_col, "country", "season_value", "team_win_pct"]
        if c in df.columns
    ]

    def _select(group: pd.DataFrame) -> pd.DataFrame:
        picked = group.sort_values("season_value", ascending=False)[cols].head(n_per_conference).copy()
        picked["selection_type"] = ["Titular"] * min(n_starters_per_conference, len(picked)) + [
            "Reserva"
        ] * max(len(picked) - n_starters_per_conference, 0)
        return picked

    has_conference = conference_col in df.columns and df[conference_col].nunique(dropna=True) > 1
    if not has_conference:
        return _select(df).reset_index(drop=True)

    selections = [_select(group) for _, group in df.groupby(conference_col)]
    return pd.concat(selections, ignore_index=True).sort_values(
        [conference_col, "season_value"], ascending=[True, False]
    ).reset_index(drop=True)


def check_all_star_nationality_quota(
    selections: pd.DataFrame,
    min_us: int = DEFAULT_MIN_US_ALL_STARS,
    min_international: int = DEFAULT_MIN_INTERNATIONAL_ALL_STARS,
) -> Dict[str, Any]:
    """
    Verifica la cuota REAL de nacionalidad de la NBA sobre una selección
    de All-Star ya calculada (`compute_all_star_selections`): al menos
    `min_us` jugadores de EE.UU. y `min_international` internacionales.

    SOLO VERIFICA, NO CORRIGE: si la selección natural no cumple el
    mínimo, en la vida real el comisionado de la NBA añade jugadores a su
    propio criterio -- eso es la decisión discrecional de una persona
    real y este proyecto, por principio, no predice decisiones
    discrecionales de personas reales (mismo motivo por el que MVP/DPOY/
    etc. son heurísticas, no una predicción de la votación real). Esta
    función se limita a informar si haría falta esa intervención.

    Requiere una columna `country` en `selections` (de
    roster_positions.csv / league_player_countries.csv, ambos vía
    CommonPlayerInfo) -- sin ella, devuelve `checked=False` en vez de
    fingir un resultado.
    """
    if "country" not in selections.columns or selections.empty:
        return {
            "checked": False, "total": int(len(selections)),
            "us_count": None, "international_count": None, "unknown_count": None,
            "meets_us_minimum": None, "meets_international_minimum": None, "meets_both": None,
        }

    is_usa = selections["country"] == "USA"
    is_unknown = selections["country"].isna()
    is_international = ~is_usa & ~is_unknown

    us_count = int(is_usa.sum())
    international_count = int(is_international.sum())
    unknown_count = int(is_unknown.sum())
    meets_us = us_count >= min_us
    meets_international = international_count >= min_international

    return {
        "checked": True,
        "total": int(len(selections)),
        "us_count": us_count,
        "international_count": international_count,
        "unknown_count": unknown_count,
        "meets_us_minimum": meets_us,
        "meets_international_minimum": meets_international,
        "meets_both": meets_us and meets_international,
    }


# Etiqueta que marca a un jugador añadido para cubrir la cuota de
# nacionalidad -- distinta de "Titular"/"Reserva" a propósito, para que
# sea imposible confundirla con una selección por mérito del ranking.
COMMISSIONER_ADDITION_SELECTION_TYPE = "Añadido por el comisionado"


def add_commissioner_picks_for_nationality_quota(
    player_df: pd.DataFrame,
    selections: pd.DataFrame,
    quota: Dict[str, Any],
    team_win_pct: Optional[Dict[Any, float]] = None,
    min_us: int = DEFAULT_MIN_US_ALL_STARS,
    min_international: int = DEFAULT_MIN_INTERNATIONAL_ALL_STARS,
) -> pd.DataFrame:
    """
    Si `quota` (de check_all_star_nationality_quota) señala que la
    selección natural no cumple el mínimo, añade jugadores de la
    nacionalidad que falta hasta cubrirlo -- A PETICIÓN EXPLÍCITA DEL
    USUARIO, que quiso simular esta parte en vez de dejarla solo como
    aviso.

    QUÉ ES ESTO REALMENTE: en la vida real, el comisionado de la NBA
    (Adam Silver) elige a su criterio quién se añade -- una decisión
    discrecional de una persona real que este proyecto NO puede predecir
    (mismo límite que el resto de premios: no hay forma de simular el
    juicio subjetivo de alguien concreto). Lo que SÍ se puede hacer es
    una aproximación razonable: coger, de entre los jugadores NO
    seleccionados de la nacionalidad que falta, los de mayor
    `season_value` (mismo proxy que el resto del All-Star) hasta llegar
    al mínimo. Cada jugador añadido así queda marcado con
    `selection_type = COMMISSIONER_ADDITION_SELECTION_TYPE` y
    `commissioner_pick = True` -- un warning explícito de que NO llegó
    por mérito del ranking natural, sino para tapar un hueco de cuota.

    Devuelve `selections` sin tocar si la cuota ya se cumple, o si falta
    la columna `country` (no se puede saber a quién añadir sin saber su
    nacionalidad).
    """
    if not quota.get("checked") or quota.get("meets_both"):
        return selections.assign(commissioner_pick=False) if not selections.empty else selections

    pool = player_df.copy()
    pool["season_value"] = _season_value(pool)
    if team_win_pct:
        pool["team_win_pct"] = pool["player_id"].map(team_win_pct).fillna(0.5)

    already_selected = set(selections["player_id"]) if "player_id" in selections.columns else set()
    pool = pool[~pool["player_id"].isin(already_selected)]

    picks = []
    if not quota["meets_us_minimum"]:
        needed = min_us - quota["us_count"]
        picks.append(pool[pool["country"] == "USA"].sort_values("season_value", ascending=False).head(needed))
    if not quota["meets_international_minimum"]:
        needed = min_international - quota["international_count"]
        is_international = pool["country"].notna() & (pool["country"] != "USA")
        picks.append(pool[is_international].sort_values("season_value", ascending=False).head(needed))

    picks = [p for p in picks if not p.empty]
    base = selections.assign(commissioner_pick=False)
    if "selection_type" not in base.columns:
        # Garantiza que la columna sobrevive al reindex de abajo incluso
        # si el llamante no pasó por compute_all_star_selections (que sí
        # la incluye siempre) -- si no, un `selections` sin esta columna
        # haría que el reindex la descartara en silencio de `added`.
        base["selection_type"] = None
    if not picks:
        # No hay candidatos elegibles de la nacionalidad que falta (caso
        # extremo con un roster pequeño/hipotético) -- se devuelve la
        # selección natural sin poder cubrir el hueco.
        return base

    added = pd.concat(picks, ignore_index=True)
    added["selection_type"] = COMMISSIONER_ADDITION_SELECTION_TYPE
    added = added.reindex(columns=base.columns, fill_value=None)
    added["commissioner_pick"] = True

    return pd.concat([base, added], ignore_index=True)


def _pick_positional_teams(
    df: pd.DataFrame,
    value_col: str,
    team_names: List[str],
    slots: Dict[str, int] = DEFAULT_TEAM_POSITION_SLOTS,
) -> pd.DataFrame:
    """
    Reparte `df` (ya filtrado por elegibilidad) en `len(team_names)`
    quintetos consecutivos, cada uno con los cupos de posición de
    `slots` (2 G + 2 F + 1 C por defecto), llenando cada cupo con el
    jugador de mayor `value_col` de esa posición que quede disponible.
    Un jugador ya elegido en un equipo no puede repetir en otro.

    LIMITACIÓN: si una posición se queda sin candidatos elegibles (p. ej.
    un roster hipotético sin ningún pívot real), ese cupo queda vacío en
    vez de forzar a alguien de otra posición -- un roster inventado no
    tiene por qué cubrir las 3 posiciones con profundidad real, y rellenar
    el hueco con un jugador de otra posición falsearía el formato.
    """
    remaining = df.dropna(subset=["_position_group"]).sort_values(value_col, ascending=False).copy()
    rows = []
    for team_name in team_names:
        for position, slot_count in slots.items():
            pool = remaining[remaining["_position_group"] == position]
            picks = pool.head(slot_count)
            remaining = remaining.drop(picks.index)
            for _, pick in picks.iterrows():
                pick_dict = pick.drop("_position_group").to_dict()
                rows.append({"team": team_name, "position_slot": position, **pick_dict})
    return pd.DataFrame(rows)


def compute_all_nba_teams(
    player_df: pd.DataFrame,
    games_per_season: int,
    team_record: Optional[Dict[Any, str]] = None,
    min_games_played: int = DEFAULT_MIN_GAMES_SEASON_AWARDS,
    team_names: List[str] = ALL_NBA_TEAM_NAMES,
) -> pd.DataFrame:
    """
    Quintetos All-NBA heurísticos: 3 equipos (Primero/Segundo/Tercero),
    2 bases/escoltas + 2 aleros/ala-pívots + 1 pívot cada uno, ordenados
    por valor de temporada (mismo `_season_value` que MVP) -- a
    diferencia del MVP, SIN ponderar por % de victorias del equipo (el
    All-NBA histórico premia producción individual incluso en equipos
    mediocres más de lo que lo hace la votación de MVP).

    ELEGIBILIDAD: partidos jugados ESPERADOS >= `min_games_played` (65
    por defecto, la política real de la NBA desde 2023-24) -- ver
    `_expected_games_played`. A petición explícita del usuario.

    Incluye OFFENSIVE_COMPARISON_STATS + team_record (a petición del
    usuario, para la vista previa al pasar el ratón en webapp/) -- mismas
    columnas que MVP/ROY/6MOY, así que el quinteto es comparable de un
    vistazo con cualquier otro premio.
    """
    df = player_df.copy()
    df["games_played_expected"] = _expected_games_played(df, games_per_season)
    df = df[df["games_played_expected"] >= min_games_played]
    df["_position_group"] = _position_group(df)
    df["season_value"] = _season_value(df)
    df = _attach_team_record(df, team_record)

    cols = [
        c for c in
        ["player_id", "player_name", "team_abbreviation", "team_record", "_position_group", "games_played_expected"]
        + OFFENSIVE_COMPARISON_STATS
        + ["season_value"]
        if c in df.columns
    ]
    return _pick_positional_teams(df[cols], "season_value", team_names)


def compute_all_defensive_teams(
    player_df: pd.DataFrame,
    games_per_season: int,
    team_record: Optional[Dict[Any, str]] = None,
    min_games_played: int = DEFAULT_MIN_GAMES_SEASON_AWARDS,
    team_names: List[str] = ALL_DEFENSIVE_TEAM_NAMES,
) -> pd.DataFrame:
    """
    Quintetos All-Defensive heurísticos: 2 equipos (Primero/Segundo),
    2 bases/escoltas + 2 aleros/ala-pívots + 1 pívot cada uno, ordenados
    por el mismo `defensive_score_per36` escalado a temporada que usa
    DPOY (ver compute_dpoy_candidates) -- MISMA limitación que el DPOY:
    el box score no ve casi nada de la defensa individual real más allá
    de robos, tapones, rebote defensivo y faltas.

    ELEGIBILIDAD: partidos jugados ESPERADOS >= `min_games_played` (65
    por defecto) -- ver `_expected_games_played`.

    Incluye OFFENSIVE_COMPARISON_STATS + team_record (a petición del
    usuario, para la vista previa al pasar el ratón en webapp/) -- mismas
    columnas que el resto de premios, aunque el ranking en sí siga siendo
    puramente defensivo.
    """
    df = player_df.copy()
    df["games_played_expected"] = _expected_games_played(df, games_per_season)
    df = df[df["games_played_expected"] >= min_games_played]
    df["_position_group"] = _position_group(df)
    df["defensive_score_per36"] = (
        1.5 * df["STL_per36_projected"]
        + 1.5 * df["BLK_per36_projected"]
        + 0.3 * df["DREB_per36_projected"]
        - 0.2 * df["PF_per36_projected"]
    )
    df["defensive_value"] = df["defensive_score_per36"] * df["projected_total_minutes"] / 36.0
    df = _attach_team_record(df, team_record)

    cols = [
        c for c in
        ["player_id", "player_name", "team_abbreviation", "team_record", "_position_group", "games_played_expected"]
        + OFFENSIVE_COMPARISON_STATS
        + ["defensive_value"]
        if c in df.columns
    ]
    return _pick_positional_teams(df[cols], "defensive_value", team_names)
