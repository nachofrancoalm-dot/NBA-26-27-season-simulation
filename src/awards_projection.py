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

# Umbral real de la NBA (desde 2023-24, "Player Participation Policy")
# para premios de fin de temporada; no aplica al All-Star.
DEFAULT_MIN_GAMES_SEASON_AWARDS = 65

# Cupos por quinteto (2 bases/escoltas + 2 aleros/ala-pívots + 1 pívot),
# igual que usa la NBA real para All-NBA/All-Defensive.
DEFAULT_TEAM_POSITION_SLOTS: Dict[str, int] = {"G": 2, "F": 2, "C": 1}

ALL_NBA_TEAM_NAMES = ["Primer equipo All-NBA", "Segundo equipo All-NBA", "Tercer equipo All-NBA"]
ALL_DEFENSIVE_TEAM_NAMES = ["Primer equipo All-Defensive", "Segundo equipo All-Defensive"]

# Seleccionados All-Star por conferencia en la NBA real (titulares + reservas).
DEFAULT_ALL_STARS_PER_CONFERENCE = 12


def _season_value(df: pd.DataFrame) -> pd.Series:
    """game_score_per36 * projected_total_minutes / 36 -- ver docstring del módulo."""
    return df["game_score_per36"] * df["projected_total_minutes"] / 36.0


def _avg_mpg(df: pd.DataFrame, games_per_season: int) -> pd.Series:
    return df["projected_total_minutes"] / games_per_season


# Stats por-partido para comparar candidatos de un vistazo -- ya vienen
# calculadas en `player_df`, solo hace falta incluirlas en la selección
# de columnas de cada función.
OFFENSIVE_COMPARISON_STATS = ["PPG", "RPG", "APG", "SPG", "BPG", "FG%", "3P%"]
# PFPG (faltas/partido) da contexto: es el único término NEGATIVO de
# defensive_score_per36.
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
    ganador. `team_win_pct` sin datos para un jugador se asume 0.5 (sin
    ajuste de equipo). `team_record` es solo para mostrar.
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
    1.5*BLK_per36 + 0.3*DREB_per36 - 0.2*PF_per36 (pesos de este proyecto,
    no una métrica oficial), escalado a valor de temporada completa. El
    box score no captura la mayoría de la defensa real (contención de
    tiro, rotaciones, defensa de perímetro sin robo/tapón), así que este
    proxy es limitado a robos/tapones/rebote defensivo/faltas.
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
    """Jugadores cuya temporada real más reciente tuvo menos partidos
    empezados (GS) que jugados/2 (GP) -- aproximación de "suplente" que no
    distingue a un titular que perdió su puesto de alguien siempre suplente."""
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
    """6MOY heurístico: mayor valor de temporada entre los jugadores marcados
    como "banquillo" (ver compute_bench_player_ids)."""
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
    """Rookie = toda su historia en career_stats (carrera COMPLETA vía
    PlayerCareerStats) cae en una sola temporada. Si los datos de carrera
    de un veterano llegaran incompletos, esta heurística lo marcaría
    (incorrectamente) como rookie."""
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
    """ROY heurístico: mayor valor de temporada entre los jugadores
    marcados como rookies (ver compute_rookie_player_ids)."""
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
    las DOS temporadas más recientes de cada jugador -- el salto que YA
    ocurrió, más fiel al espíritu del premio que usar la proyección.
    Requiere un mínimo de minutos/partido en AMBAS temporadas (evita que
    un salto de bajísimo a bajo volumen parezca una mejora enorme).
    Excluye implícitamente a los rookies, que solo tienen 1 temporada.
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
    PPG/RPG/APG/SPG/BPG/FG%/3P% REALES de la última temporada YA jugada de
    cada jugador -- para que el popup de MIP compare la temporada
    PROYECTADA contra la real inmediatamente anterior. Distinto de
    `previous_game_score_per36` de compute_mip_candidates, que usa la
    PENÚLTIMA temporada real.
    """
    if career_stats_df.empty:
        return pd.DataFrame(columns=["player_id"] + PREV_SEASON_STATS_COLUMNS)

    # dedupe_traded_seasons() agrupa por SEASON_ID y debe aplicarse POR
    # JUGADOR, no sobre el DataFrame multi-jugador completo: si se llama
    # sobre la liga entera, un trade de CUALQUIER jugador en la temporada
    # X hace que se descarte la fila de esa temporada para todos los
    # demás jugadores que no fueron traspasados (por no ser 'TOT').
    df = pd.concat(
        [dedupe_traded_seasons(group) for _, group in career_stats_df.groupby("PLAYER_ID")], ignore_index=True
    )
    df = df[df["GP"] > 0]
    df["_start_year"] = df["SEASON_ID"].apply(season_start_year)
    latest = df.sort_values("_start_year", ascending=False).groupby("PLAYER_ID").head(1)

    # FG_PCT/FG3_PCT pueden faltar (p.ej. datos de test construidos a mano)
    # -- se degradan a NaN en vez de romper el resto de la función.
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
    COY heurístico de EQUIPO, no de entrenador -- el proxy más común de
    los votantes reales: el equipo que más mejoró respecto al récord REAL
    de la temporada anterior. `team_wins_df`: columnas team_abbreviation,
    wins_mean. `prior_wins_by_team`: dict team_abbreviation -> victorias
    reales del año anterior.
    """
    df = team_wins_df.copy()
    df["prior_wins"] = df["team_abbreviation"].map(prior_wins_by_team)
    df = df.dropna(subset=["prior_wins"])
    df["win_improvement"] = df["wins_mean"] - df["prior_wins"]

    cols = [c for c in ["team_abbreviation", "conference", "prior_wins", "wins_mean", "win_improvement"] if c in df.columns]
    return df.sort_values("win_improvement", ascending=False)[cols].head(top_n).reset_index(drop=True)


def _expected_games_played(df: pd.DataFrame, games_per_season: int) -> pd.Series:
    """Partidos jugados ESPERADOS (ver simulation.compute_expected_games_played)
    si hay `risk_score`; si no, se asume la temporada completa."""
    if "risk_score" not in df.columns:
        return pd.Series(float(games_per_season), index=df.index)
    risk = df["risk_score"].fillna(0.0).to_numpy()
    return pd.Series(compute_expected_games_played(risk, games_per_season), index=df.index)


def _position_group(df: pd.DataFrame) -> pd.Series:
    """Primera letra de `position` (Guard/Forward/Center o G/F/C conviven),
    limitada a G/F/C. Sin `position`, queda None -- no participa en los
    quintetos (que exigen cupo de posición) pero sí en el All-Star."""
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
    team_record: Optional[Dict[Any, str]] = None,
    conference_col: str = "conference",
    n_per_conference: int = DEFAULT_ALL_STARS_PER_CONFERENCE,
    n_starters_per_conference: int = DEFAULT_ALL_STAR_STARTERS_PER_CONFERENCE,
) -> pd.DataFrame:
    """
    All-Star heurístico: SIN restricción de partidos jugados ni minutos
    mínimos, solo valor de temporada -- SIN distinción de posición
    (coincide con la regla real desde esta temporada).

    En la vida real los 5 titulares por conferencia se votan 50% fans +
    25% jugadores + 25% medios, y los 7 reservas los eligen los
    entrenadores; este proyecto no tiene acceso a ninguno de esos votos,
    así que `season_value` actúa como proxy único para todos ellos.
    `selection_type` ("Titular"/"Reserva") es solo una ETIQUETA sobre ese
    ranking (los de mayor valor = "Titular"), no un voto simulado.

    Con más de una conferencia en `conference_col` (scope "league"), se
    eligen los `n_per_conference` de mayor valor DE CADA conferencia. Sin
    conferencia (scope "own", un solo equipo), se toma un único top
    `n_per_conference` sobre todo `player_df`.
    """
    df = player_df.copy()
    df["season_value"] = _season_value(df)
    if team_win_pct:
        df["team_win_pct"] = df["player_id"].map(team_win_pct).fillna(0.5)
    df = _attach_team_record(df, team_record)

    cols = [
        c for c in
        ["player_id", "player_name", "team_abbreviation", "team_record", conference_col, "country"]
        + OFFENSIVE_COMPARISON_STATS
        + ["season_value", "team_win_pct"]
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
    de All-Star ya calculada: al menos `min_us` jugadores de EE.UU. y
    `min_international` internacionales. SOLO VERIFICA, NO CORRIGE -- en
    la vida real el comisionado añade jugadores a su propio criterio, una
    decisión discrecional que este proyecto no predice. Requiere una
    columna `country` en `selections`; sin ella devuelve `checked=False`.
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
    Si `quota` (de check_all_star_nationality_quota) no se cumple, añade
    jugadores de la nacionalidad que falta hasta cubrirla. En la vida
    real esto lo decide el comisionado a su criterio, algo que este
    proyecto no puede predecir; la aproximación es coger, de entre los NO
    seleccionados de la nacionalidad que falta, los de mayor
    `season_value`. Cada jugador añadido queda marcado con
    `selection_type = COMMISSIONER_ADDITION_SELECTION_TYPE` y
    `commissioner_pick = True`, para distinguirlo de una selección por
    mérito. Devuelve `selections` sin tocar si la cuota ya se cumple o si
    falta la columna `country`.
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
        # Garantiza que la columna sobrevive al reindex de abajo aunque el
        # llamante no haya pasado por compute_all_star_selections.
        base["selection_type"] = None
    if not picks:
        return base  # sin candidatos elegibles de la nacionalidad que falta

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
    quintetos consecutivos con los cupos de posición de `slots` (2 G + 2
    F + 1 C por defecto), llenando cada cupo con el jugador de mayor
    `value_col` disponible de esa posición; nadie repite entre equipos.
    Si una posición se queda sin candidatos, el cupo queda vacío en vez
    de forzar a alguien de otra posición.
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
    Quintetos All-NBA heurísticos: 3 equipos (Primero/Segundo/Tercero), 2
    bases/escoltas + 2 aleros/ala-pívots + 1 pívot cada uno, ordenados por
    valor de temporada -- a diferencia del MVP, SIN ponderar por % de
    victorias del equipo. Elegibilidad: partidos jugados ESPERADOS >=
    `min_games_played` (65 por defecto, política real de la NBA desde
    2023-24, ver `_expected_games_played`).
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
    Quintetos All-Defensive heurísticos: 2 equipos (Primero/Segundo), 2
    bases/escoltas + 2 aleros/ala-pívots + 1 pívot cada uno, ordenados por
    el mismo `defensive_score_per36` escalado a temporada que usa DPOY
    (misma limitación de box score, ver compute_dpoy_candidates).
    Elegibilidad: partidos jugados ESPERADOS >= `min_games_played`.
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
