"""
schedule_strength.py

Tercer submódulo de la capa de contexto de temporada (ver roadmap en
README.md): calcula un `difficulty_score` (0-1) POR PARTIDO del
calendario del equipo, combinando fuerza del rival, back-to-backs y
viaje. A diferencia de injury_model.py y fatigue_accumulation.py, este
módulo NO es por-jugador -- es por-partido del calendario del equipo.

LIMITACIONES DE DATOS IMPORTANTES
-----------------------------------
1. Fuerza del rival: `team_config.yaml` puede apuntar a una temporada que
   todavía no se ha jugado (p. ej. 2026-27). No existen resultados reales
   de esa temporada para medir qué tan bueno es cada rival. Se usa como
   proxy el WinPCT de cada rival en la temporada ANTERIOR
   (`prior_season_standings.csv`, generado por
   `data_pipeline.build_prior_season_standings_dataset`). Es la misma
   aproximación que usa cualquier preview de calendario real -- pero es
   eso, una aproximación, no un dato de la temporada en curso.
2. Viaje: `nba_api` no expone distancias. Se usa una tabla estática de
   coordenadas de las ciudades de las 30 franquicias NBA (hecho
   geográfico de la liga, no específico de ningún equipo/jugador -- no
   viola la regla de "nada hardcodeado en src/") y distancia geodésica
   (haversine) entre la ciudad del partido anterior y la de este. Partes
   neutrales fuera de las 30 ciudades conocidas (México, Londres, París)
   no tienen coordenada -- su tramo de viaje se trata como 0 y se avisa
   una vez por ejecución, en vez de fallar.

DISEÑO DEL difficulty_score
------------------------------
Tres componentes 0-1 por partido, pesos configurables en
`config["schedule_strength"]` (nunca hardcodeados):

1. `opponent_strength_score` -- WinPCT del rival en la temporada anterior
   (ya está en escala 0-1, no hace falta normalizar y evita inventar una
   escala arbitraria).
2. `back_to_back_score` -- 1.0 si el partido anterior del equipo fue el
   día previo (0 días de descanso), si no 0.0.
3. `travel_score` -- distancia (km) desde la ciudad del partido anterior,
   normalizada con un tope configurable de "viaje largo" (3000 km por
   defecto: similar a un vuelo costa a costa en EE. UU.), capado en 1.0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_paths  # noqa: E402

# Coordenadas aproximadas (lat, lon) de las ciudades de las 30 franquicias
# NBA. Hecho geográfico de la liga, no específico de ningún equipo
# simulado -- ver docstring del módulo.
ARENA_COORDS: Dict[str, Tuple[float, float]] = {
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
    "Brooklyn": (40.6782, -73.9442),
    "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298),
    "Cleveland": (41.4993, -81.6944),
    "Dallas": (32.7767, -96.7970),
    "Denver": (39.7392, -104.9903),
    "Detroit": (42.3314, -83.0458),
    "San Francisco": (37.7749, -122.4194),
    "Houston": (29.7604, -95.3698),
    "Indianapolis": (39.7684, -86.1581),
    "Los Angeles": (34.0430, -118.2673),
    "Memphis": (35.1495, -90.0490),
    "Miami": (25.7617, -80.1918),
    "Milwaukee": (43.0389, -87.9065),
    "Minneapolis": (44.9778, -93.2650),
    "New Orleans": (29.9511, -90.0715),
    "New York": (40.7128, -74.0060),
    "Oklahoma City": (35.4676, -97.5164),
    "Orlando": (28.5383, -81.3792),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "Portland": (45.5152, -122.6784),
    "Sacramento": (38.5816, -121.4944),
    "San Antonio": (29.4241, -98.4936),
    "Toronto": (43.6532, -79.3832),
    "Salt Lake City": (40.7608, -111.8910),
    "Washington": (38.9072, -77.0369),
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "opponent_strength": 0.40,
    "back_to_back": 0.30,
    "travel": 0.30,
}
DEFAULT_HIGH_TRAVEL_KM = 3000.0

_warned_missing_cities: set = set()


def haversine_km(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Distancia geodésica en km entre dos puntos (lat, lon)."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r = 6371.0  # radio medio de la Tierra en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _lookup_coords(city: str) -> Optional[Tuple[float, float]]:
    coords = ARENA_COORDS.get(city)
    if coords is None and city not in _warned_missing_cities:
        _warned_missing_cities.add(city)
        print(f"  [aviso] Sin coordenadas para '{city}' (¿partido en sede neutral?). "
              f"Ese tramo de viaje se trata como 0 km.")
    return coords


def build_team_game_calendar(schedule_df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    A partir del calendario de liga ya filtrado a los partidos de un
    equipo (team_schedule.csv), añade is_home, opponent_team_id y ordena
    cronológicamente. Un partido por fila.
    """
    df = schedule_df.copy()
    df["gameDate"] = pd.to_datetime(df["gameDate"])
    df["is_home"] = df["homeTeam_teamId"] == team_id
    df["opponent_team_id"] = df["awayTeam_teamId"].where(df["is_home"], df["homeTeam_teamId"])
    df = df.sort_values("gameDate").reset_index(drop=True)
    return df


def compute_back_to_back_scores(games: pd.DataFrame) -> pd.Series:
    """1.0 si el partido anterior del equipo fue <= 1 día antes, si no 0.0.
    El primer partido de la temporada no tiene back-to-back (no hay partido previo)."""
    gap_days = games["gameDate"].diff().dt.days
    return (gap_days <= 1).fillna(False).astype(float)


def compute_travel_scores(
    games: pd.DataFrame, high_travel_km: float = DEFAULT_HIGH_TRAVEL_KM
) -> pd.Series:
    """
    Distancia desde la ciudad del partido anterior a la de este partido,
    normalizada contra high_travel_km y capada en 1.0. El primer partido
    de la temporada no tiene viaje previo dentro del calendario (0.0).
    """
    coords = games["arenaCity"].apply(_lookup_coords)
    scores = [0.0]
    for i in range(1, len(games)):
        prev_coords = coords.iloc[i - 1]
        cur_coords = coords.iloc[i]
        if prev_coords is None or cur_coords is None:
            scores.append(0.0)
            continue
        km = haversine_km(prev_coords, cur_coords)
        scores.append(min(km / high_travel_km, 1.0) if high_travel_km > 0 else 0.0)
    return pd.Series(scores, index=games.index)


def compute_opponent_strength_scores(
    games: pd.DataFrame, standings_df: pd.DataFrame
) -> pd.Series:
    """WinPCT del rival en la temporada anterior (0-1, sin normalizar)."""
    win_pct_by_team = standings_df.set_index("TeamID")["WinPCT"]
    return games["opponent_team_id"].map(win_pct_by_team).fillna(win_pct_by_team.mean())


def compute_difficulty_scores(
    schedule_df: pd.DataFrame,
    team_id: int,
    standings_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    high_travel_km: float = DEFAULT_HIGH_TRAVEL_KM,
) -> pd.DataFrame:
    """
    Punto de entrada de cálculo: dado el calendario ya filtrado a un
    equipo y los standings de la temporada anterior, devuelve un
    DataFrame con un difficulty_score (0-1) por partido.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    games = build_team_game_calendar(schedule_df, team_id)
    games["opponent_strength_score"] = compute_opponent_strength_scores(games, standings_df)
    games["back_to_back_score"] = compute_back_to_back_scores(games)
    games["travel_score"] = compute_travel_scores(games, high_travel_km)

    games["difficulty_score"] = (
        w["opponent_strength"] * games["opponent_strength_score"]
        + w["back_to_back"] * games["back_to_back_score"]
        + w["travel"] * games["travel_score"]
    ).clip(lower=0.0, upper=1.0)

    return games


def build_schedule_difficulty_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Lee team_schedule.csv y prior_season_standings.csv (generados por
    data_pipeline.py) y guarda data/processed/schedule_difficulty.csv con
    un difficulty_score por partido.
    """
    paths = get_paths(config)
    schedule_path = paths["processed"] / "team_schedule.csv"
    standings_path = paths["processed"] / "prior_season_standings.csv"

    for path, builder in [
        (schedule_path, "build_team_schedule_dataset"),
        (standings_path, "build_prior_season_standings_dataset"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. Corre `python src/data_pipeline.py` "
                f"primero (usa data_pipeline.{builder})."
            )

    schedule_df = pd.read_csv(schedule_path)
    standings_df = pd.read_csv(standings_path)

    sched_cfg = config.get("schedule_strength", {})
    weights = {**DEFAULT_WEIGHTS, **sched_cfg.get("weights", {})}
    high_travel_km = sched_cfg.get("high_travel_km", DEFAULT_HIGH_TRAVEL_KM)

    result_df = compute_difficulty_scores(
        schedule_df, config["team"]["team_id"], standings_df, weights, high_travel_km
    )

    out_path = paths["processed"] / "schedule_difficulty.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} partidos)")
    return result_df


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config

    build_schedule_difficulty_dataset(load_config())
