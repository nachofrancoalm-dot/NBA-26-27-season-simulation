"""
league_sandbox.py

Simula la LIGA COMPLETA de 30 equipos con tu roster hipotético
(sandbox_simulation.py) sustituido en el hueco de tu equipo -- para que
Liga y Playoffs / Premios individuales reflejen el escenario montado, no
solo el resultado agregado de tu equipo.

Los jugadores "tomados prestados" de otros equipos reales NO se quitan de
sus equipos reales -- esos 29 se simulan igual que en la liga real (mismo
criterio que `league_simulation.project_own_team_for_league`).

Esto es viable en vivo porque lee `league_player_projections.csv` ya
generado (~0s) para los 29 equipos sin tocar, en vez de re-derivar la
proyección de los 577 jugadores desde stats de carrera como hace
`league_simulation.load_and_project_all_teams` (~18s medidos). Regular
season (1.000 temporadas) + playoffs (1.000) suman unos pocos segundos
más, así que funciona como acción síncrona sin necesitar un job en
background.

Reutiliza `league_simulation.simulate_league_regular_season`,
`build_round_robin_schedule` y `simulate_playoffs_once` tal cual, y
`sandbox_simulation.build_roster` para el roster hipotético. Las filas de
los 29 equipos sin tocar se copian literalmente de
`league_player_projections.csv`; las de tu equipo se derivan de su fila
REAL, recalculando solo los campos que dependen de los minutos nuevos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import awards_projection as ap
from config_loader import get_paths
from league_simulation import (
    TEAM_CONFERENCE,
    build_round_robin_schedule,
    simulate_league_regular_season,
    simulate_playoffs_once,
)
from sandbox_simulation import SandboxRosterError, build_roster, load_player_pool
from simulation import DEFAULT_MONTE_CARLO_CONFIG

DEFAULT_LEAGUE_LIVE_N_SEASONS = 1000
DEFAULT_LEAGUE_LIVE_N_PLAYOFF_SEASONS = 1000

# Columna per-36 -> total, mismas que league_simulation.build_league_simulation_dataset
# recalcula. FG%/3P% quedan fuera a propósito (ratios, no dependen de minutos).
PER36_TO_TOTAL = {
    "PTS_per36_projected": "PTS_projected",
    "AST_per36_projected": "AST_projected",
    "REB_per36_projected": "REB_projected",
    "STL_per36_projected": "STL_projected",
    "BLK_per36_projected": "BLK_projected",
    "TOV_per36_projected": "TOV_projected",
    "OREB_per36_projected": "OREB_projected",
    "DREB_per36_projected": "DREB_projected",
    "FGM_per36_projected": "FGM_projected",
    "FGA_per36_projected": "FGA_projected",
    "FTM_per36_projected": "FTM_projected",
    "FTA_per36_projected": "FTA_projected",
    "PF_per36_projected": "PF_projected",
    "FG3M_per36_projected": "FG3M_projected",
    "FG3A_per36_projected": "FG3A_projected",
}

TOTAL_TO_PER_GAME_DISPLAY = {
    "PTS_projected": "PPG",
    "REB_projected": "RPG",
    "AST_projected": "APG",
    "STL_projected": "SPG",
    "BLK_projected": "BPG",
    "TOV_projected": "TOPG",
    "FG3M_projected": "3PM",
}


def _compact_projection(game_score_per36, minutes_projection, risk_scores, fatigue_scores) -> Dict[str, Any]:
    return {
        "game_score_per36": np.asarray(game_score_per36, dtype=float),
        "minutes_projection": np.asarray(minutes_projection, dtype=float),
        "risk_scores": np.asarray(risk_scores, dtype=float),
        "fatigue_scores": np.asarray(fatigue_scores, dtype=float),
        "synergy_matrix": None,
    }


def _hypothetical_player_rows(
    config: Dict[str, Any], player_ids: List[int], my_team_id: int, my_abbrev: str, my_conference: str
) -> List[Dict[str, Any]]:
    """Filas de jugador para TU equipo hipotético, mismo esquema de
    columnas que `league_player_projections.csv` -- parten de la fila
    REAL de cada jugador y solo recalculan lo que depende de los minutos
    nuevos de la rotación."""
    games_per_season = config["simulation"]["games_per_season"]
    roster, minutes_projection = build_roster(config, player_ids)

    rows = []
    for i, player_id in enumerate(player_ids):
        row = roster.loc[player_id].to_dict()
        row["player_id"] = player_id
        row["team_id"] = my_team_id
        row["team_abbreviation"] = my_abbrev
        row["conference"] = my_conference
        new_minutes = float(minutes_projection[i])
        row["minutes_projection"] = new_minutes
        row["projected_total_minutes"] = new_minutes * games_per_season

        for per36_col, total_col in PER36_TO_TOTAL.items():
            if per36_col not in row:
                continue
            per36_rate = row.get(per36_col) or 0.0
            row[total_col] = per36_rate * new_minutes / 36.0 * games_per_season

        for total_col, display_col in TOTAL_TO_PER_GAME_DISPLAY.items():
            if total_col in row:
                row[display_col] = row[total_col] / games_per_season

        rows.append(row)
    return rows


def _load_other_teams(config: Dict[str, Any], exclude_team_id: int) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """Los 29 equipos SIN tocar, leídos directo de `league_player_projections.csv`.
    Devuelve (pool_df_sin_tu_equipo, {team_id: proyección compacta})."""
    pool = load_player_pool(config)
    pool = pool[pool["team_id"] != exclude_team_id].copy()

    team_projections: Dict[int, Dict[str, Any]] = {}
    for team_id, group in pool.groupby("team_id"):
        team_projections[int(team_id)] = _compact_projection(
            group["game_score_per36"], group["minutes_projection"], group["risk_score"], group["fatigue_score"]
        )
    return pool, team_projections


def simulate_hypothetical_league(
    config: Dict[str, Any],
    player_ids: List[int],
    n_seasons: int = DEFAULT_LEAGUE_LIVE_N_SEASONS,
    n_playoff_seasons: int = DEFAULT_LEAGUE_LIVE_N_PLAYOFF_SEASONS,
    random_seed: int = 11,
) -> Dict[str, Any]:
    """Orquestador principal: temporada regular + playoffs de los 30
    equipos con tu roster hipotético sustituido en el hueco de tu equipo.
    Devuelve un dict con las mismas piezas que la API real
    (`regular_season_df`, `playoff_df`, `player_projections_df`) para que
    el router reutilice `compute_conference_standings` y
    `awards_projection.compute_*` sin duplicar lógica."""
    my_team_id = config["team"]["team_id"]
    my_abbrev = config["team"]["abbreviation"]
    my_conference = TEAM_CONFERENCE[my_abbrev]

    other_pool, team_projections = _load_other_teams(config, exclude_team_id=my_team_id)

    my_rows = _hypothetical_player_rows(config, player_ids, my_team_id, my_abbrev, my_conference)
    my_row_df = pd.DataFrame(my_rows)
    team_projections[my_team_id] = _compact_projection(
        my_row_df["game_score_per36"], my_row_df["minutes_projection"], my_row_df["risk_score"], my_row_df["fatigue_score"]
    )

    team_ids = list(team_projections.keys())
    team_abbrev_by_id = dict(zip(other_pool["team_id"], other_pool["team_abbreviation"]))
    team_abbrev_by_id.update(dict.fromkeys([my_team_id], my_abbrev))
    team_conference = {tid: TEAM_CONFERENCE[abbrev] for tid, abbrev in team_abbrev_by_id.items()}

    mc_cfg = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}
    games_per_season = config["simulation"]["games_per_season"]

    rng = np.random.default_rng(random_seed)
    schedule = build_round_robin_schedule(team_ids, games_per_season, rng)
    wins_by_team = simulate_league_regular_season(team_projections, schedule, n_seasons, games_per_season, mc_cfg, random_seed)

    regular_season_df = pd.DataFrame(
        [
            {
                "team_id": tid,
                "team_abbreviation": team_abbrev_by_id[tid],
                "conference": team_conference[tid],
                "wins_mean": float(wins_by_team[tid].mean()),
                "wins_p10": float(np.quantile(wins_by_team[tid], 0.1)),
                "wins_p90": float(np.quantile(wins_by_team[tid], 0.9)),
            }
            for tid in team_ids
        ]
    ).sort_values("wins_mean", ascending=False)

    playoff_rng = np.random.default_rng(random_seed + 1)
    made_playoffs = {tid: 0 for tid in team_ids}
    reached_conf_semis = {tid: 0 for tid in team_ids}
    reached_conf_finals = {tid: 0 for tid in team_ids}
    reached_finals = {tid: 0 for tid in team_ids}
    won_championship = {tid: 0 for tid in team_ids}

    for i in range(n_playoff_seasons):
        wins_this_season = {tid: int(wins_by_team[tid][i]) for tid in team_ids}
        result = simulate_playoffs_once(wins_this_season, team_conference, team_projections, playoff_rng, mc_cfg)
        for tid in result["east_8"] + result["west_8"]:
            made_playoffs[tid] += 1
        for tid in result["east_result"]["round1_winners"] + result["west_result"]["round1_winners"]:
            reached_conf_semis[tid] += 1
        for tid in result["east_result"]["conf_semis_winners"] + result["west_result"]["conf_semis_winners"]:
            reached_conf_finals[tid] += 1
        reached_finals[result["east_result"]["conference_champion"]] += 1
        reached_finals[result["west_result"]["conference_champion"]] += 1
        won_championship[result["nba_champion"]] += 1

    playoff_df = pd.DataFrame(
        [
            {
                "team_id": tid,
                "team_abbreviation": team_abbrev_by_id[tid],
                "conference": team_conference[tid],
                "playoff_pct": made_playoffs[tid] / n_playoff_seasons * 100,
                "conf_semis_pct": reached_conf_semis[tid] / n_playoff_seasons * 100,
                "conf_finals_pct": reached_conf_finals[tid] / n_playoff_seasons * 100,
                "finals_pct": reached_finals[tid] / n_playoff_seasons * 100,
                "championship_pct": won_championship[tid] / n_playoff_seasons * 100,
            }
            for tid in team_ids
        ]
    )

    player_projections_df = pd.concat([other_pool, my_row_df], ignore_index=True, sort=False)

    return {
        "my_team_id": my_team_id,
        "regular_season_df": regular_season_df,
        "playoff_df": playoff_df,
        "player_projections_df": player_projections_df,
    }


def compute_hypothetical_awards(config: Dict[str, Any], league_result: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """Mismos premios que `dashboard.data_loader.compute_awards_summary`
    (scope "league"), pero sobre `league_result` en memoria en vez de
    leer los CSV reales de disco -- llama a las mismas funciones puras de
    `awards_projection.py`, sin duplicar lógica de premios."""
    paths = get_paths(config)
    games_per_season = config["simulation"]["games_per_season"]
    player_df = league_result["player_projections_df"]
    regular = league_result["regular_season_df"]

    wins_by_team = (regular.set_index("team_abbreviation")["wins_mean"] / games_per_season).to_dict()
    team_win_pct = dict(zip(player_df["player_id"], player_df["team_abbreviation"].map(wins_by_team)))

    def _win_loss_record(wins_mean: float) -> str:
        wins = round(wins_mean)
        return f"{wins}-{games_per_season - wins}"

    record_by_team = {
        abbrev: _win_loss_record(wins) for abbrev, wins in regular.set_index("team_abbreviation")["wins_mean"].items()
    }
    team_record = dict(zip(player_df["player_id"], player_df["team_abbreviation"].map(record_by_team)))

    career_path = paths["processed"] / "league_player_career_stats.csv"
    career = pd.read_csv(career_path) if career_path.exists() else pd.DataFrame()

    # COY (proxy): compara victorias simuladas contra las REALES del año
    # anterior; prior_season_standings.csv no depende del roster hipotético.
    coy = None
    prior_path = paths["processed"] / "prior_season_standings.csv"
    if prior_path.exists():
        from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID

        id_to_abbrev = {v: k for k, v in ABBREVIATION_TO_TEAM_ID.items()}
        prior = pd.read_csv(prior_path)
        prior_wins_by_team = (
            prior.assign(team_abbreviation=prior["TeamID"].map(id_to_abbrev))
            .dropna(subset=["team_abbreviation"])
            .set_index("team_abbreviation")["WINS"]
            .to_dict()
        )
        coy = ap.compute_coy_candidates(regular, prior_wins_by_team, top_n=top_n)

    rookie_ids = ap.compute_rookie_player_ids(career)
    bench_ids = ap.compute_bench_player_ids(career)
    all_star = ap.compute_all_star_selections(player_df, games_per_season, team_win_pct=team_win_pct, team_record=team_record)
    all_star_quota = ap.check_all_star_nationality_quota(all_star)
    all_star_final = ap.add_commissioner_picks_for_nationality_quota(
        player_df, all_star, all_star_quota, team_win_pct=team_win_pct
    )

    mip = ap.compute_mip_candidates(career, top_n=top_n)
    if not mip.empty and "player_id" in player_df.columns:
        extra_cols = [c for c in ap.OFFENSIVE_COMPARISON_STATS + ["team_abbreviation"] if c in player_df.columns]
        mip = mip.merge(player_df[["player_id"] + extra_cols], on="player_id", how="left")
        if team_record:
            mip["team_record"] = mip["player_id"].map(team_record)
        mip = mip.merge(ap.compute_latest_real_season_stats(career), on="player_id", how="left")

    return {
        "scope": "league",
        "mvp": ap.compute_mvp_candidates(player_df, games_per_season, team_win_pct=team_win_pct, team_record=team_record, top_n=top_n),
        "dpoy": ap.compute_dpoy_candidates(player_df, games_per_season, team_record=team_record, top_n=top_n),
        "sixth_man": ap.compute_sixth_man_candidates(player_df, bench_ids, games_per_season, team_record=team_record, top_n=top_n),
        "roy": ap.compute_roy_candidates(player_df, rookie_ids, games_per_season, team_record=team_record, top_n=top_n),
        "mip": mip,
        "coy": coy,
        "all_star": all_star,
        "all_star_nationality_quota": all_star_quota,
        "all_star_final": all_star_final,
        "all_nba": ap.compute_all_nba_teams(player_df, games_per_season, team_record=team_record),
        "all_defensive": ap.compute_all_defensive_teams(player_df, games_per_season, team_record=team_record),
    }
