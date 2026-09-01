"""
sandbox_simulation.py

Motor de simulación para un roster HIPOTÉTICO ensamblado a mano desde el
picker de la webapp (cualquier jugador real de `league_player_projections.csv`,
en cualquier equipo, para cualquier equipo) -- distinto de
`simulation.compute_simulation_results`, que solo conoce el roster fijo de
`config/team_config.yaml` (los 13 jugadores curados a mano del proyecto)
porque lee CSV precalculados SOLO para esos 13
(aging_curve_projection.csv/injury_risk.csv/fatigue_risk.csv). Este módulo
lee en cambio `league_player_projections.csv` (los ~577 jugadores de
rotación real de los 30 equipos, generado por `data_pipeline.py --league`),
así que cualquier combinación de jugadores es una simulación válida sin
volver a correr ningún pipeline de datos.

Reutiliza el mismo motor de simulación (`simulation.run_monte_carlo`) y la
misma normalización de minutos (`simulation.normalize_rotation_minutes`,
el fix real de "minutos sin normalizar a 240" ya documentado en
simulation.py/league_simulation.py) -- cero matemática nueva, solo un
punto de entrada distinto para construir los inputs de esa función a
partir de jugadores elegidos en vivo en vez de un roster fijo.

LIMITACIÓN DOCUMENTADA (v1, igual que otras simplificaciones ya
aceptadas en este proyecto -- ver simulation.py): SIN sinergia de
alineación. `build_synergy_matrix` necesita perfiles de estilo derivados
de las columnas de `aging_curve_projection.csv` (solo existen para el
roster propio curado a mano); adaptar esos perfiles a los 577 jugadores
de liga es una extensión futura, no un bloqueo para tener un sandbox
funcional -- el efecto de sinergia es, de todas formas, un ajuste
relativamente pequeño frente al Game Score agregado (ver
lineup_synergy.py).

`minutes_per_game_last_season` (de league_player_projections.csv, el dato
REAL de la temporada anterior de cada jugador en SU equipo real) es la
señal de "raw minutes" que se normaliza a 240 -- exactamente el mismo
input que usa `league_simulation.project_team_roster` para los 30 equipos
reales. Un jugador que casi no jugó el año pasado (novato, banca de
fondo de rotación) entra con pocos minutos "en bruto" y por tanto pesa
poco en la rotación de 240 -- no hay forma de saber, sin más contexto,
cuántos minutos le darías tú en TU roster hipotético.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config_loader import get_paths
from simulation import (
    DEFAULT_MONTE_CARLO_CONFIG,
    DEFAULT_ROTATION_SIZE,
    compute_expected_games_played,
    compute_league_average_game_score_per36,
    normalize_rotation_minutes,
    run_monte_carlo,
)

MAX_ROSTER_SIZE = 15
MIN_ROSTER_SIZE = 5
# n_seasons por defecto para una tirada "en vivo" disparada desde un clic
# HTTP -- deliberadamente menor que las 10.000 de simulation_results.csv
# (precalculado sin límite de tiempo de respuesta). Con 2.000 la
# distribución de victorias ya es estable de sobra para leer P10/mediana/P90
# a simple vista, y responde en el rango de un segundo.
DEFAULT_LIVE_N_SEASONS = 2000


class SandboxRosterError(ValueError):
    """Roster hipotético inválido -- mensaje pensado para mostrarse tal cual en la UI."""


def load_player_pool(config: Dict[str, Any]) -> pd.DataFrame:
    """Los ~577 jugadores de rotación real de los 30 equipos -- el catálogo
    completo del que se puede elegir en el picker de roster hipotético."""
    paths = get_paths(config)
    path = paths["processed"] / "league_player_projections.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Corre `python src/data_pipeline.py --league` "
            "para generar el catálogo de jugadores de los 30 equipos."
        )
    return pd.read_csv(path)


def build_roster(config: Dict[str, Any], player_ids: List[int]) -> tuple[pd.DataFrame, np.ndarray]:
    """Valida `player_ids` y calcula los minutos normalizados a 240 para
    ESTE roster hipotético -- compartido por `simulate_custom_roster` (que
    los usa para correr Monte Carlo) y `compute_roster_player_stats` (que
    los usa para recalcular PPG/RPG/... por-partido de cada jugador en
    este contexto). Si se calcularan por separado en cada función podrían
    divergir con el tiempo; un único punto de cálculo lo evita.

    Devuelve (roster_df, minutes_projection) en el MISMO orden que
    `player_ids`.
    """
    if len(player_ids) < MIN_ROSTER_SIZE:
        raise SandboxRosterError(f"El roster necesita al menos {MIN_ROSTER_SIZE} jugadores.")
    if len(player_ids) > MAX_ROSTER_SIZE:
        raise SandboxRosterError(f"Máximo {MAX_ROSTER_SIZE} jugadores en el roster hipotético.")
    if len(set(player_ids)) != len(player_ids):
        raise SandboxRosterError("Hay un jugador repetido en el roster.")

    pool = load_player_pool(config).set_index("player_id")
    missing = [pid for pid in player_ids if pid not in pool.index]
    if missing:
        raise SandboxRosterError(f"player_id(s) no encontrados en el catálogo de liga: {missing}")

    roster = pool.loc[player_ids]

    raw_minutes = roster["minutes_per_game_last_season"].fillna(0.0).to_dict()
    rotation_size = min(DEFAULT_ROTATION_SIZE, len(player_ids))
    minutes_by_player = normalize_rotation_minutes(raw_minutes, rotation_size=rotation_size)
    minutes_projection = np.array([minutes_by_player[pid] for pid in player_ids])

    return roster, minutes_projection


# Nombre de columna por-36 en league_player_projections.csv -> (nombre en
# modo "totales", nombre en modo "por partido") -- mismas siglas que ya
# usa dashboard/data_loader.TOTAL_STATS/PER_GAME_STATS, para que se lea
# igual que la tabla de "Mi equipo". FG%/3P% no están aquí porque son
# ratios (FGM/FGA), no dependen de los minutos -- se copian tal cual del
# pool en los dos modos.
PER36_TO_DISPLAY = {
    "PTS_per36_projected": ("PTS", "PPG"),
    "REB_per36_projected": ("REB", "RPG"),
    "AST_per36_projected": ("AST", "APG"),
    "STL_per36_projected": ("STL", "SPG"),
    "BLK_per36_projected": ("BLK", "BPG"),
    "TOV_per36_projected": ("TOV", "TOPG"),
    "FG3M_per36_projected": ("3PM", "3PM"),
}


def compute_roster_player_stats(config: Dict[str, Any], player_ids: List[int], mode: str = "per_game") -> pd.DataFrame:
    """Estadísticas individuales de cada jugador PARA ESTE roster
    hipotético -- lo que faltaba tras simular solo el agregado de equipo.
    `minutes_per_game_last_season`/PPG/RPG/... de `league_player_projections.csv`
    reflejan los minutos REALES de cada jugador en SU equipo real, no el
    papel que tendría en el roster que acabas de montar -- aquí se
    recalculan las tasas por-36 (que sí son independientes del equipo)
    contra los minutos NUEVOS normalizados a 240 de `build_roster`. GP =
    partidos esperados (`compute_expected_games_played`, sí depende del
    riesgo de lesión); MPG = los minutos normalizados tal cual, sin
    descontar por riesgo -- representa el ritmo cuando el jugador juega,
    no la carga de temporada (mismo criterio que
    `dashboard.data_loader._apply_simulated_games_and_minutes`). `mode`:
    "per_game" (PPG/RPG/...) o "totals" (PTS/REB/..., temporada
    completa) -- mismo toggle que `/api/roster`.
    """
    if mode not in ("per_game", "totals"):
        raise SandboxRosterError('mode debe ser "per_game" o "totals".')

    games_per_season = config["simulation"]["games_per_season"]
    roster, minutes_projection = build_roster(config, player_ids)

    risk_scores = roster["risk_score"].fillna(0.0).to_numpy()
    availability = 1 - np.clip(risk_scores, 0, 1)

    view = pd.DataFrame(
        {
            "player_id": player_ids,
            "player_name": roster["player_name"].to_numpy(),
            "team_abbreviation": roster["team_abbreviation"].to_numpy(),
            "position": roster.get("position", pd.Series([None] * len(roster))).to_numpy(),
            "game_score_per36": roster["game_score_per36"].fillna(0.0).to_numpy(),
            "risk_score": risk_scores,
            "GP": compute_expected_games_played(risk_scores, games_per_season).round(0),
            "MPG": minutes_projection.round(1),
        }
    )

    for per36_col, (total_name, per_game_name) in PER36_TO_DISPLAY.items():
        if per36_col not in roster.columns:
            continue
        per_game = roster[per36_col].fillna(0.0).to_numpy() * minutes_projection / 36.0
        if mode == "totals":
            view[total_name] = (per_game * games_per_season * availability).round(0)
        else:
            view[per_game_name] = per_game.round(1)

    for pct_col in ("FG%", "3P%"):
        if pct_col in roster.columns:
            view[pct_col] = roster[pct_col].to_numpy()

    return view.sort_values("game_score_per36", ascending=False).reset_index(drop=True)


def simulate_custom_roster(
    config: Dict[str, Any],
    player_ids: List[int],
    n_seasons: int = DEFAULT_LIVE_N_SEASONS,
    mc_overrides: Optional[Dict[str, float]] = None,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Corre `run_monte_carlo` para un roster hipotético de `player_ids`
    (cualquier jugador real de `load_player_pool`, de cualquier equipo).
    `mc_overrides` -- ajustes puntuales sobre `DEFAULT_MONTE_CARLO_CONFIG`
    (p.ej. sliders de la UI: injury_dispersion, game_variance_std...),
    aplicados POR ENCIMA de `config["monte_carlo"]` del YAML.
    """
    roster, minutes_projection = build_roster(config, player_ids)

    game_score_per36 = roster["game_score_per36"].fillna(0.0).to_numpy()
    risk_scores = roster["risk_score"].fillna(0.0).to_numpy()
    fatigue_scores = roster["fatigue_score"].fillna(0.0).to_numpy()

    paths = get_paths(config)
    standings_path = paths["processed"] / "prior_season_standings.csv"
    if not standings_path.exists():
        raise FileNotFoundError(
            f"No se encontró {standings_path}. Corre `python src/data_pipeline.py` primero."
        )
    league_win_pcts = pd.read_csv(standings_path)["WinPCT"].to_numpy()

    mc_cfg = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {}), **(mc_overrides or {})}
    # Misma recalibración automática que compute_simulation_results -- solo
    # si nadie (ni el YAML ni un slider de la UI) fijó ya un valor a mano.
    fixed_by_yaml_or_slider = "league_average_game_score_per36" in config.get("monte_carlo", {}) or (
        mc_overrides is not None and "league_average_game_score_per36" in mc_overrides
    )
    if not fixed_by_yaml_or_slider:
        league_projections = load_player_pool(config)
        # `run_monte_carlo` recibe `synergy_matrix=None` aquí (limitación
        # documentada del sandbox: sin sinergia de alineación), así que
        # este roster nunca recibe el bonus de sinergia. La línea base de
        # liga, en cambio, sí puede llevarlo incorporado vía
        # `league_mean_synergy_net_rating` (la sinergia media de los 30
        # equipos reales, ~+10.65 de net rating) -- mismo bug de "suma
        # cero rota por un término no centrado" que simulation.py/
        # backtesting.py ya documentan y evitan (ver su docstring). Con la
        # sinergia en un lado y no en el otro, el roster se compara contra
        # un rival ~10 puntos de net rating más fuerte de lo que le toca,
        # cada partido. Mientras el sandbox no module sinergia para este
        # equipo, la línea base tampoco debe llevarla -- forzar 0.0 deja a
        # los dos lados en igualdad de condiciones y alinea el resultado
        # con /sandbox/league, que tampoco aplica sinergia a ninguno de
        # los 30 equipos.
        mc_cfg["league_average_game_score_per36"] = compute_league_average_game_score_per36(
            league_projections,
            league_mean_synergy_net_rating=0.0,
            game_score_to_net_rating_scale=mc_cfg["game_score_to_net_rating_scale"],
        )

    games_per_season = config["simulation"]["games_per_season"]

    return run_monte_carlo(
        list(player_ids),
        game_score_per36,
        minutes_projection,
        risk_scores,
        fatigue_scores,
        league_win_pcts,
        n_seasons,
        games_per_season,
        mc_cfg,
        random_seed,
        synergy_matrix=None,
    )
