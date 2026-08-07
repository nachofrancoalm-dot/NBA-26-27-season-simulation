"""
opponent_weighting.py

Quinto submódulo de la capa de contexto de temporada (ver roadmap en
README.md): pondera los partidos de cada `historical_comparable` por la
fuerza de su rival ESE MISMO partido, para que el backtesting no trate
igual una victoria contra un contender que una contra un equipo en
reconstrucción.

A diferencia de schedule_strength.py (que necesita un proxy de temporada
ANTERIOR porque el calendario que pondera aún no se ha jugado), aquí las
4 temporadas de `historical_comparables` ya se jugaron por completo: se
puede pedir la fuerza CONTEMPORÁNEA real de cada rival (mismo
season) vía `historical_comparables_standings.csv`.

DISEÑO
------
1. `resolve_opponent_team_id()` -- el game log solo trae el rival como
   abreviación de 3 letras dentro de la columna MATCHUP (p. ej.
   "MIA @ TOR"). Se resuelve a `team_id` con una tabla estática de las 30
   franquicias NBA (hecho de liga, no específico de ningún equipo/jugador
   simulado) + alias para abreviaciones históricas que cambiaron de
   nombre dentro del rango de temporadas de este proyecto (Nets "NJN"
   antes de mudarse a Brooklyn en 2012-13; Hornets/Pelicans "NOH" antes
   del cambio de nombre en 2013).
2. `compute_opponent_win_pct()` -- WinPCT del rival esa misma temporada
   (0-1, sin normalizar).
3. `compute_opponent_weight()` -- peso continuo proporcional al WinPCT
   del rival (`win_pct ** steepness`, steepness configurable). Se eligió
   una ponderación continua en vez de una categórica fija
   (contender/reconstrucción con un único umbral) porque cualquier umbral
   sería arbitrario sin evidencia que lo respalde -- un rival con 0.54 de
   WinPCT no es cualitativamente distinto de uno con 0.56.
4. `compute_weighted_net_rating()` -- Net Rating medio ponderado por ese
   peso, para comparar contra el Net Rating sin ponderar de
   performance_curve.py.
5. `classify_contender_vs_rebuilding()` -- SÍ ofrece una vista categórica
   además (contender / equipo medio / en reconstrucción), con umbrales
   configurables, porque el roadmap la pide explícitamente como resumen
   legible -- pero es descriptiva, no la base del peso numérico.

Reutiliza `compute_net_rating_estimate()` de performance_curve.py en vez
de duplicar la fórmula (import directo entre submódulos hermanos, no una
nueva capa de abstracción compartida).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_paths  # noqa: E402
from context.performance_curve import compute_net_rating_estimate  # noqa: E402

# Tabla estática de las 30 franquicias NBA (team_id de nba_api). Hecho de
# liga, no específico de ningún equipo/jugador simulado.
ABBREVIATION_TO_TEAM_ID: Dict[str, int] = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}

# Abreviaciones históricas usadas antes de mudanzas/cambios de nombre de
# franquicia, dentro del rango de temporadas que puede cubrir este
# proyecto (2010-11 en adelante).
ABBREVIATION_ALIASES: Dict[str, str] = {
    "NJN": "BKN",  # New Jersey Nets, antes de mudarse a Brooklyn (2012-13)
    "NOH": "NOP",  # New Orleans Hornets, antes de renombrarse Pelicans (2013-14)
}

DEFAULT_STEEPNESS = 1.0
DEFAULT_CONTENDER_WIN_PCT = 0.55
DEFAULT_REBUILDING_WIN_PCT = 0.45

_warned_missing_abbreviations: set = set()


def extract_opponent_abbreviation(matchup: str) -> Optional[str]:
    """'MIA @ TOR' -> 'TOR'; 'MIA vs. BOS' -> 'BOS'."""
    if not isinstance(matchup, str):
        return None
    separator = " @ " if " @ " in matchup else " vs. " if " vs. " in matchup else None
    if separator is None:
        return None
    return matchup.split(separator)[1].strip()


def resolve_opponent_team_id(matchup: str) -> Optional[int]:
    """Resuelve la columna MATCHUP a un team_id, aplicando alias históricos."""
    abbr = extract_opponent_abbreviation(matchup)
    if abbr is None:
        return None
    abbr = ABBREVIATION_ALIASES.get(abbr, abbr)
    team_id = ABBREVIATION_TO_TEAM_ID.get(abbr)
    if team_id is None and abbr not in _warned_missing_abbreviations:
        _warned_missing_abbreviations.add(abbr)
        print(f"  [aviso] Abreviación de rival desconocida: '{abbr}'. "
              f"Ese partido usará la media de la liga como fuerza de rival.")
    return team_id


def compute_opponent_win_pct(games: pd.DataFrame, standings: pd.DataFrame) -> pd.Series:
    """WinPCT del rival en la MISMA temporada del partido (0-1)."""
    win_pct_by_team = standings.set_index("TeamID")["WinPCT"]
    opponent_team_ids = games["MATCHUP"].apply(resolve_opponent_team_id)
    return opponent_team_ids.map(win_pct_by_team).fillna(win_pct_by_team.mean())


def compute_opponent_weight(win_pct: pd.Series, steepness: float = DEFAULT_STEEPNESS) -> pd.Series:
    """Peso continuo proporcional a la fuerza del rival: win_pct ** steepness."""
    return win_pct.clip(lower=0.0) ** steepness


def compute_weighted_net_rating(net_rating: pd.Series, weight: pd.Series) -> float:
    """Media ponderada de net_rating_estimate. Si todos los pesos son 0, cae a la media simple."""
    total_weight = weight.sum()
    if total_weight == 0:
        return float(net_rating.mean()) if len(net_rating) else 0.0
    return float((net_rating * weight).sum() / total_weight)


def classify_contender_vs_rebuilding(
    win_pct: pd.Series,
    contender_win_pct: float = DEFAULT_CONTENDER_WIN_PCT,
    rebuilding_win_pct: float = DEFAULT_REBUILDING_WIN_PCT,
) -> pd.Series:
    """Vista categórica descriptiva (no usada para el peso numérico): 'contender',
    'reconstruccion' o 'medio', según umbrales configurables de WinPCT del rival."""
    return pd.cut(
        win_pct,
        bins=[-float("inf"), rebuilding_win_pct, contender_win_pct, float("inf")],
        labels=["reconstruccion", "medio", "contender"],
    )


def summarize_opponent_weighting(
    games: pd.DataFrame,
    standings: pd.DataFrame,
    steepness: float = DEFAULT_STEEPNESS,
    contender_win_pct: float = DEFAULT_CONTENDER_WIN_PCT,
    rebuilding_win_pct: float = DEFAULT_REBUILDING_WIN_PCT,
) -> Dict[str, float]:
    """Resumen de UN caso histórico: Net Rating sin ponderar vs. ponderado por
    fuerza de rival, y récord/Net Rating desglosado por categoría de rival."""
    df = compute_net_rating_estimate(games)
    df["opponent_win_pct"] = compute_opponent_win_pct(df, standings)
    df["opponent_weight"] = compute_opponent_weight(df["opponent_win_pct"], steepness)
    df["opponent_tier"] = classify_contender_vs_rebuilding(
        df["opponent_win_pct"], contender_win_pct, rebuilding_win_pct
    )

    unweighted_net_rating = float(df["net_rating_estimate"].mean()) if len(df) else 0.0
    weighted_net_rating = compute_weighted_net_rating(df["net_rating_estimate"], df["opponent_weight"])

    tier_summary = {}
    for tier in ["contender", "medio", "reconstruccion"]:
        tier_games = df[df["opponent_tier"] == tier]
        tier_summary[f"{tier}_games"] = int(len(tier_games))
        tier_summary[f"{tier}_net_rating"] = (
            float(tier_games["net_rating_estimate"].mean()) if not tier_games.empty else None
        )
        if "WL" in tier_games.columns and not tier_games.empty:
            tier_summary[f"{tier}_win_pct"] = float((tier_games["WL"] == "W").mean())
        else:
            tier_summary[f"{tier}_win_pct"] = None

    return {
        "unweighted_net_rating": unweighted_net_rating,
        "weighted_net_rating": weighted_net_rating,
        **tier_summary,
    }


def build_opponent_weighting_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Lee historical_comparables_advanced_game_logs.csv y
    historical_comparables_standings.csv (generados por data_pipeline.py),
    calcula el resumen ponderado por caso histórico y guarda
    data/processed/opponent_weighting_summary.csv.
    """
    paths = get_paths(config)
    games_path = paths["processed"] / "historical_comparables_advanced_game_logs.csv"
    standings_path = paths["processed"] / "historical_comparables_standings.csv"

    for path, builder in [
        (games_path, "build_historical_comparables_advanced_dataset"),
        (standings_path, "build_historical_comparables_standings_dataset"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. Corre `python src/data_pipeline.py` "
                f"primero (usa data_pipeline.{builder})."
            )

    games_df = pd.read_csv(games_path)
    standings_df = pd.read_csv(standings_path)

    opp_cfg = config.get("opponent_weighting", {})
    steepness = opp_cfg.get("steepness", DEFAULT_STEEPNESS)
    contender_win_pct = opp_cfg.get("contender_win_pct", DEFAULT_CONTENDER_WIN_PCT)
    rebuilding_win_pct = opp_cfg.get("rebuilding_win_pct", DEFAULT_REBUILDING_WIN_PCT)

    rows = []
    for comparable_name, games in games_df.groupby("comparable_name"):
        season = games["season"].iloc[0]
        season_standings = standings_df[standings_df["season"] == season]
        summary = summarize_opponent_weighting(
            games, season_standings, steepness, contender_win_pct, rebuilding_win_pct
        )
        rows.append({"comparable_name": comparable_name, **summary})

    result_df = pd.DataFrame(rows)
    out_path = paths["processed"] / "opponent_weighting_summary.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} casos)")
    return result_df


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config

    build_opponent_weighting_dataset(load_config())
