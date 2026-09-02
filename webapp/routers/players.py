"""
routers/players.py

GET /api/player/{player_id} -- popup de detalle de jugador. Combina tres
fuentes ya existentes en data/processed/ y data/raw/, sin ingesta nueva:
trayectoria por temporada (roster_career_stats.csv / league_player_career_stats.csv,
output crudo de PlayerCareerStats), bio (leída directo de
data/raw/player_common_info/{player_id}.csv si está cacheado -- nunca
dispara una llamada de red desde un request HTTP), y "cualidades"
(compute_style_profile() de src/lineup_synergy.py, mismos 4 ejes que el
motor de sinergia, con umbrales de presentación que no afectan a
ninguna simulación).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from config_loader import get_paths, load_config  # noqa: E402
from lineup_synergy import DEFAULT_USAGE_THRESHOLD, compute_style_profile  # noqa: E402
from season_utils import season_start_year  # noqa: E402
from shot_chart_projection import project_player_shot_chart  # noqa: E402
from simulation import compute_expected_games_played  # noqa: E402

from webapp.serializers import df_to_records

router = APIRouter()

SEASON_COLUMNS = ["SEASON_ID", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT"]

# Umbrales de PRESENTACIÓN para etiquetar el estilo -- descriptivos, no
# usados en simulación (a diferencia de DEFAULT_USAGE_THRESHOLD, que sí
# es del modelo de sinergia).
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
    league_player_projections.csv (liga) -- punto único de búsqueda para
    _find_qualities y _projected_season_row. aging_curve_projection.csv
    no trae `risk_score` (a diferencia del CSV de liga), así que aquí se
    replica el merge con injury_risk.csv que hace
    load_roster_overview -- sin esto no hay forma de saber que un
    jugador de alto riesgo no jugará los 82 partidos."""
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
    """Temporada proyectada como una fila más de la trayectoria, marcada
    con `is_projection`. None si falta alguna columna de totales
    proyectados (p.ej. fila sintética de test) -- degradar sin fallar es
    mejor que un 500."""
    if row is None or any(col not in row.index for col in PROJECTED_TOTAL_COLUMNS):
        return None
    games_per_season = config["simulation"]["games_per_season"]
    team_abbreviation = row.get("team_abbreviation")
    if team_abbreviation is None or pd.isna(team_abbreviation):
        team_abbreviation = config["team"].get("abbreviation")

    # GP usa la misma fórmula que select_roster_view() (compute_expected_games_played).
    # PTS/REB/AST/etc de aging_curve_projection.csv son totales a PLENA
    # SALUD -- se escalan aquí por el mismo factor de disponibilidad que
    # GP, si no el modo "Por partido" del popup saldría inflado.
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


def _load_shot_chart_history(paths, player_id: int) -> Optional[pd.DataFrame]:
    """Todas las temporadas cacheadas de un jugador (hasta
    aging_curve.DEFAULT_N_SEASONS_LOOKBACK, ver data_pipeline.py). Lee
    roster_shot_charts.csv (roster propio, curado) y cae a
    league_shot_charts.csv (los 30 equipos reales, requiere
    --league-shot-charts) para cualquier otro jugador -- nunca dispara
    una llamada a nba_api."""
    for filename in ("roster_shot_charts.csv", "league_shot_charts.csv"):
        path = paths["processed"] / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows = df[df["player_id"] == player_id]
        if not rows.empty:
            return rows
    return None


@router.get("/player/{player_id}/shot-chart")
def get_player_shot_chart(player_id: int, kind: str = Query("real", pattern="^(real|projected)$")):
    """
    `kind=real` (por defecto): tiros reales de la temporada real más
    reciente cacheada -- lo que ya había.

    `kind=projected`: shot chart SINTÉTICO de la temporada proyectada
    (ver src/shot_chart_projection.py) -- remuestrea el histórico real
    de hasta 3 temporadas para que el conteo de intentos/anotados de
    2/3 cuadre EXACTO con FGA/FG3A/FGM/FG3M ya proyectados por
    aging_curve.py, escalados por el mismo factor de disponibilidad
    (`compute_expected_games_played`) que ya usa `_projected_season_row`
    -- así el total de tiros del mapa coincide con lo que el resto del
    popup muestra para esa misma temporada. Semilla del RNG = player_id,
    no el reloj -- el mapa no debe "saltar" cada vez que se reabre el
    popup del mismo jugador.

    Lista vacía si no hay histórico cacheado para el jugador, o (en
    `projected`) si no hay proyección disponible; el frontend lo trata
    como "sin datos", nunca dispara una llamada a nba_api.
    """
    config = load_config()
    paths = get_paths(config)
    shots = _load_shot_chart_history(paths, player_id)
    if shots is None or shots.empty:
        return {"player_id": player_id, "season": None, "shots": [], "kind": kind}

    if kind == "real":
        latest_season = max(shots["season"].astype(str).unique(), key=season_start_year)
        latest = shots[shots["season"].astype(str) == latest_season]
        records = latest[["loc_x", "loc_y", "shot_made", "shot_type"]].to_dict(orient="records")
        return {"player_id": player_id, "season": latest_season, "shots": records, "kind": "real"}

    projection_row = _find_projection_row(paths, player_id)
    if projection_row is None or any(col not in projection_row.index for col in PROJECTED_TOTAL_COLUMNS):
        return {"player_id": player_id, "season": None, "shots": [], "kind": "projected"}

    games_per_season = config["simulation"]["games_per_season"]
    risk_score = projection_row.get("risk_score")
    if pd.notna(risk_score):
        availability = float(compute_expected_games_played(np.array([risk_score]), games_per_season)[0]) / games_per_season
    else:
        availability = 1.0

    projected_shots = project_player_shot_chart(
        shots,
        target_fga=projection_row["FGA_projected"] * availability,
        target_fg3a=projection_row["FG3A_projected"] * availability,
        target_fgm=projection_row["FGM_projected"] * availability,
        target_fg3m=projection_row["FG3M_projected"] * availability,
        rng=np.random.default_rng(player_id),
    )
    return {
        "player_id": player_id,
        "season": f"{config['team']['season']} (proyección)",
        "shots": projected_shots.to_dict(orient="records"),
        "kind": "projected",
    }
