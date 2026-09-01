"""
schedule_strength.py

Tercer submódulo de la capa de contexto de temporada (ver roadmap en
README.md): calcula un `difficulty_score` (0-1) POR PARTIDO del
calendario del equipo, combinando fuerza del rival, back-to-backs y
viaje -- a diferencia de injury_model.py y fatigue_accumulation.py, no
es por-jugador.

Dos aproximaciones de datos: (1) la fuerza del rival usa como proxy su
WinPCT de la temporada ANTERIOR (`prior_season_standings.csv`), porque el
config puede apuntar a una temporada que aún no se jugó; (2) el viaje usa
una tabla estática de coordenadas de las 30 ciudades NBA y distancia
geodésica (haversine) entre partidos consecutivos -- sedes neutrales sin
coordenada (México, Londres, París) cuentan como 0 km, con aviso.

difficulty_score combina tres componentes 0-1 con pesos configurables en
`config["schedule_strength"]`: opponent_strength_score (WinPCT del rival,
ya 0-1), back_to_back_score (1.0 si 0 días de descanso), travel_score
(distancia normalizada contra un tope de "viaje largo", 3000 km por
defecto, capado en 1.0).
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
