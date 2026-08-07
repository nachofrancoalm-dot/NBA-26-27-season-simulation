"""
season_utils.py

Utilidades compartidas para trabajar con las tablas de temporada de
nba_api (roster_career_stats.csv, roster_playoff_career_stats.csv, etc.).
Extraído tras aparecer duplicado de forma idéntica en injury_model.py,
fatigue_accumulation.py y aging_curve.py -- tercera repetición, el punto
en que duplicar deja de ser más simple que compartir.
"""

from __future__ import annotations

import pandas as pd


def dedupe_traded_seasons(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """
    Si un jugador fue traspasado a mitad de temporada, nba_api incluye una
    fila 'TOT' (total de esa temporada) ADEMÁS de una fila por cada equipo
    en el que jugó, todas con la misma SEASON_ID. Sin este dedupe, una
    temporada con trade contaría dos o más veces en una ventana de N
    temporadas. Se prioriza la fila 'TOT' cuando existe para esa SEASON_ID.
    """
    if "TEAM_ABBREVIATION" not in player_seasons.columns:
        return player_seasons
    has_tot = player_seasons.groupby("SEASON_ID")["TEAM_ABBREVIATION"].transform(
        lambda s: (s == "TOT").any()
    )
    is_tot = player_seasons["TEAM_ABBREVIATION"] == "TOT"
    return player_seasons[is_tot | ~has_tot].reset_index(drop=True)


def season_start_year(season_id: str) -> int:
    """Año de inicio de una temporada con formato 'YYYY-YY' (p. ej. '2023-24' -> 2023)."""
    return int(str(season_id).split("-")[0])
