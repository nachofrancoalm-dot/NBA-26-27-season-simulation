"""
routers/players.py

GET /api/player/{player_id} -- popup de detalle de jugador (doble clic
en el nombre, ver webapp/static/js/player-modal.js). Combina tres
fuentes YA existentes en data/processed/ y data/raw/, sin ingesta
nueva:

- Trayectoria por temporada: roster_career_stats.csv (roster propio) o
  league_player_career_stats.csv (liga) -- son el output crudo de
  PlayerCareerStats, una fila por temporada, ya generados por
  data_pipeline.py.
- Bio (altura, peso, edad, país, universidad, draft): se lee
  DIRECTAMENTE data/raw/player_common_info/{player_id}.csv si existe
  -- el cache que fetch_player_common_info() ya escribe para
  posición/nacionalidad, pero con TODAS sus columnas (nunca solo
  position/country). Esta capa NUNCA llama a fetch_player_common_info
  ni a ningún endpoint de nba_api -- si el archivo no está cacheado
  todavía, la bio simplemente se omite (bio: null), no se dispara una
  llamada de red desde un request HTTP (ver docstring de
  data_pipeline.py sobre el rate-limiting agresivo de stats.nba.com).
- "Cualidades": compute_style_profile() de src/lineup_synergy.py sobre
  la fila de aging_curve_projection.csv o league_player_projections.csv
  del jugador -- mismos 4 ejes (usage/playmaking/spacing/interior) que
  ya usa el motor de sinergia, con umbrales de PRESENTACIÓN (no
  afectan a ninguna simulación) para convertirlos en etiquetas cortas.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from config_loader import get_paths, load_config  # noqa: E402
from lineup_synergy import DEFAULT_USAGE_THRESHOLD, compute_style_profile  # noqa: E402
from simulation import compute_expected_games_played  # noqa: E402

from webapp.serializers import df_to_records

router = APIRouter()

SEASON_COLUMNS = ["SEASON_ID", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT"]

# Umbrales de PRESENTACIÓN para etiquetar el perfil de estilo en el popup
# -- puramente descriptivos, no se usan en ningún cálculo de simulación
# (a diferencia de DEFAULT_USAGE_THRESHOLD, que sí es un parámetro real
# del modelo de sinergia y se reutiliza aquí tal cual).
PLAYMAKING_LABEL_THRESHOLD = 6.0
SPACING_LABEL_THRESHOLD = 6.0
INTERIOR_LABEL_THRESHOLD = 8.0


def _find_career_rows(paths, player_id: int) -> Optional[pd.DataFrame]:
    for filename in ("roster_career_stats.csv", "league_player_career_stats.csv"):
        path = paths["processed"] / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows = df[df["PLAYER_ID"] == player_id]
        if not rows.empty:
            return rows.sort_values("SEASON_ID")
    return None


def _read_bio(paths, player_id: int) -> Optional[dict[str, Any]]:
    path = paths["raw"] / "player_common_info" / f"{player_id}.csv"
    if not path.exists():
        return None
    info = pd.read_csv(path)
    if info.empty:
        return None
    row = info.iloc[0]

    age = None
    birthdate = row.get("BIRTHDATE")
    if isinstance(birthdate, str) and birthdate[:10]:
        try:
            year, month, day = (int(part) for part in birthdate[:10].split("-"))
            today = date.today()
            age = today.year - year - ((today.month, today.day) < (month, day))
        except ValueError:
            age = None

    draft = None
    if pd.notna(row.get("DRAFT_YEAR")) and str(row.get("DRAFT_YEAR")) != "Undrafted":
        draft = f"{row.get('DRAFT_YEAR')} · ronda {row.get('DRAFT_ROUND')} · pick {row.get('DRAFT_NUMBER')}"

    return {
        "position": row.get("POSITION") if pd.notna(row.get("POSITION")) else None,
        "height": row.get("HEIGHT") if pd.notna(row.get("HEIGHT")) else None,
        "weight": int(row["WEIGHT"]) if pd.notna(row.get("WEIGHT")) else None,
        "age": age,
        "country": row.get("COUNTRY") if pd.notna(row.get("COUNTRY")) else None,
        "school": row.get("SCHOOL") if pd.notna(row.get("SCHOOL")) else None,
        "draft": draft,
    }


def _find_projection_row(paths, player_id: int) -> Optional[pd.Series]:
    """Fila de aging_curve_projection.csv (roster propio) o
    league_player_projections.csv (liga) de este jugador -- reutilizada
    tanto por _find_qualities como por _projected_season_row, un solo
    punto de búsqueda para no divergir entre las dos.

    league_player_projections.csv ya trae `risk_score` (se calcula al
    construir la liga completa). aging_curve_projection.csv NO -- es el
    output crudo de aging_curve.py, previo al merge con injury_risk.csv
    que hace dashboard/data_loader.load_roster_overview -- así que para
    el roster propio se replica ese mismo merge aquí, por player_id.
    Sin esto, `_projected_season_row` no tiene manera de saber que un
    jugador con riesgo de lesión alto no va a jugar los 82 partidos.
    """
    aging_path = paths["processed"] / "aging_curve_projection.csv"
    if aging_path.exists():
        df = pd.read_csv(aging_path)
        rows = df[df["player_id"] == player_id]
        if not rows.empty:
            row = rows.iloc[0].copy()
            if "risk_score" not in row.index:
                injury_path = paths["processed"] / "injury_risk.csv"
                if injury_path.exists():
                    injury = pd.read_csv(injury_path)
                    match = injury[injury["player_id"] == player_id]
                    if not match.empty:
                        row["risk_score"] = match["risk_score"].iloc[0]
            return row

    league_path = paths["processed"] / "league_player_projections.csv"
    if league_path.exists():
        df = pd.read_csv(league_path)
        rows = df[df["player_id"] == player_id]
        if not rows.empty:
            return rows.iloc[0]

    return None


def _find_qualities(row: Optional[pd.Series]) -> list[str]:
    if row is None:
        return []
    profile = compute_style_profile(row)
    labels = []
    if profile["usage"] > DEFAULT_USAGE_THRESHOLD:
        labels.append("Alto uso ofensivo")
    if profile["playmaking"] > PLAYMAKING_LABEL_THRESHOLD:
        labels.append("Buen creador de juego")
    if profile["spacing"] > SPACING_LABEL_THRESHOLD:
        labels.append("Amenaza desde el triple")
    if profile["interior"] > INTERIOR_LABEL_THRESHOLD:
        labels.append("Presencia interior")
    return labels


def _safe_pct(made: float, attempted: float) -> Optional[float]:
    return round(made / attempted, 3) if attempted else None


PROJECTED_TOTAL_COLUMNS = [
    "projected_total_minutes", "PTS_projected", "REB_projected", "AST_projected",
    "STL_projected", "BLK_projected", "FGM_projected", "FGA_projected",
    "FG3M_projected", "FG3A_projected", "FTM_projected", "FTA_projected",
]


def _projected_season_row(row: Optional[pd.Series], config) -> Optional[dict[str, Any]]:
    """Temporada proyectada (la que simula team_config.yaml, p.ej.
    "2026-27") como una fila más de la trayectoria -- para comparar de
    un vistazo si la proyección cuadra con la progresión de temporadas
    reales anteriores. Mismas columnas que SEASON_COLUMNS, marcada con
    `is_projection` (el frontend la resalta y no la trata como dato
    histórico real). Devuelve None si a la fila le falta alguna columna
    de totales proyectados (p.ej. una fila sintética de test que solo
    trae los *_per36 para las cualidades) -- degradar sin fallar es
    mejor que un 500 por una tabla incompleta.
    """
    if row is None or any(col not in row.index for col in PROJECTED_TOTAL_COLUMNS):
        return None
    games_per_season = config["simulation"]["games_per_season"]
    team_abbreviation = row.get("team_abbreviation")
    if team_abbreviation is None or pd.isna(team_abbreviation):
        team_abbreviation = config["team"].get("abbreviation")

    # GP refleja el riesgo de lesión real del jugador -- sin esto, todos
    # los jugadores mostraban los games_per_season completos (82), como
    # si nadie se fuera a lesionar nunca (bug real reportado). Misma
    # fórmula EXACTA que ya usa select_roster_view() para la columna
    # "GP" del resto de tablas (ver simulation.compute_expected_games_played):
    # la media de la binomial negativa que sortea sample_injury_absences,
    # no una aproximación aparte.
    #
    # PTS/REB/AST/STL/BLK/MIN de aging_curve_projection.csv son totales a
    # PLENA SALUD (minutes_projection * games_per_season, sin descontar
    # ausencias -- mismo criterio que usa hoy la tabla de roster en modo
    # "Totales"). Aquí SÍ se escalan por el mismo factor de disponibilidad
    # que GP: sin hacerlo, el modo "Por partido" del popup saldría
    # inflado -- misma producción total repartida entre menos partidos
    # jugados da un promedio irreal más alto de lo que el propio modelo
    # de riesgo predice.
    risk_score = row.get("risk_score")
    if pd.notna(risk_score):
        games = round(float(compute_expected_games_played(np.array([risk_score]), games_per_season)[0]))
        availability = games / games_per_season
    else:
        games = games_per_season
        availability = 1.0

    return {
        "SEASON_ID": f"{config['team']['season']} (proyección)",
        "TEAM_ABBREVIATION": team_abbreviation,
        "GP": games,
        "MIN": round(row["projected_total_minutes"] * availability),
        "PTS": round(row["PTS_projected"] * availability),
        "REB": round(row["REB_projected"] * availability),
        "AST": round(row["AST_projected"] * availability),
        "STL": round(row["STL_projected"] * availability),
        "BLK": round(row["BLK_projected"] * availability),
        "FG_PCT": _safe_pct(row["FGM_projected"], row["FGA_projected"]),
        "FG3_PCT": _safe_pct(row["FG3M_projected"], row["FG3A_projected"]),
        "FT_PCT": _safe_pct(row["FTM_projected"], row["FTA_projected"]),
        "is_projection": True,
    }


@router.get("/player/{player_id}")
def get_player(player_id: int):
    config = load_config()
    paths = get_paths(config)

    seasons = _find_career_rows(paths, player_id)
    if seasons is None:
        raise HTTPException(status_code=404, detail=f"No hay datos de temporada para el jugador {player_id}.")

    bio = _read_bio(paths, player_id)
    name = seasons["player_name"].iloc[-1] if "player_name" in seasons.columns else str(player_id)
    position = bio["position"] if bio else None

    view = seasons[[c for c in SEASON_COLUMNS if c in seasons.columns]]
    season_records = df_to_records(view)
    for record in season_records:
        record["is_projection"] = False

    projection_row = _find_projection_row(paths, player_id)
    projected_season = _projected_season_row(projection_row, config)
    if projected_season is not None:
        season_records.append(projected_season)

    return {
        "player_id": player_id,
        "name": name,
        "position": position,
        "bio": bio,
        "qualities": _find_qualities(projection_row),
        "seasons": season_records,
    }


@router.get("/player/{player_id}/shot-chart")
def get_player_shot_chart(player_id: int):
    """Tiros reales (LOC_X/LOC_Y) de la temporada real más reciente del
    jugador -- SOLO lee data/processed/roster_shot_charts.csv (generado
    por data_pipeline.build_roster_shot_charts_dataset, ejecutado como
    parte de la ingesta), nunca dispara una llamada a nba_api desde este
    request (mismo principio que el resto de este router). Si el CSV no
    existe o el jugador no tiene tiros cacheados, devuelve una lista
    vacía -- el frontend lo trata como "sin datos", no como error."""
    config = load_config()
    paths = get_paths(config)
    path = paths["processed"] / "roster_shot_charts.csv"
    if not path.exists():
        return {"player_id": player_id, "season": None, "shots": []}

    shots = pd.read_csv(path)
    shots = shots[shots["player_id"] == player_id]
    if shots.empty:
        return {"player_id": player_id, "season": None, "shots": []}

    season = str(shots["season"].iloc[0])
    records = shots[["loc_x", "loc_y", "shot_made", "shot_type"]].to_dict(orient="records")
    return {"player_id": player_id, "season": season, "shots": records}
