"""
champion_profiles.py

Análisis DESCRIPTIVO de los campeones reales de las últimas temporadas:
quién ganó, desde qué seed, a quién eliminó por el camino, cómo estaba
compuesto su roster (posiciones, experiencia, concentración de minutos en
las estrellas) y qué trayectoria de seeds ha tenido cada franquicia.

POR QUÉ ES *DESCRIPTIVO* Y NO PREDICTIVO
--------------------------------------------
Con 15 campeones (2010-11..2024-25) cualquier "perfil de campeón" que se
extraiga tiene una muestra de 15 -- insuficiente para entrenar nada. La
hipótesis de "fricción de vestuario" en superequipos, por ejemplo, parecía
sólida con 4 casos y se disolvió en artefacto de calibración al ampliar
el backtesting a 450 (ver la sección de Backtesting del README): con
muestras así de pequeñas cualquier patrón puede ser ruido. Estas
funciones se usan para CONTEXTUALIZAR y para VALIDAR el simulador (¿mi
modelo produce campeones desde los mismos seeds que la realidad?), NO
como features de predicción.

La validación sí es estadísticamente sólida en un punto concreto: en 15
temporadas NINGÚN campeón salió de un seed peor que el 3 (60% fueron seed
1). Si el simulador produce campeones de seed 4+ con frecuencia
apreciable, eso es una miscalibración medible -- y de hecho lo fue.

FUENTES DE DATOS (todas ya descargadas por `--backtest-sweep`)
-----------------------------------------------------------------
- backtest_sweep_advanced_game_logs.csv (game_phase='playoffs') -> quién
  ganó el último partido de cada temporada = campeón, y el recorrido de
  cada equipo por rondas.
- backtest_sweep_standings.csv -> PlayoffRank (el seed) y récord.
- backtest_sweep_rosters.csv -> POSITION, EXP (años de experiencia), AGE.
- backtest_sweep_player_career_stats.csv -> minutos reales de esa
  temporada, para medir la concentración de minutos.

Todo se une por TEAM_ID / TeamID (no por nombre): las franquicias cambian
de nombre y de ciudad, el id de nba_api no.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import get_paths  # noqa: E402

# Un roster NBA mezcla etiquetas simples y compuestas ("G", "G-F", "F-C").
# Para agregar por posición se usa la PRIMERA letra, que es la posición
# principal del jugador según nba_api.
POSITION_GROUPS = {"G": "Base/Escolta", "F": "Alero/Ala-pívot", "C": "Pívot"}

# Un jugador cuenta como "estrella" del equipo si está entre los N
# primeros por minutos totales jugados esa temporada. Umbral descriptivo,
# no un hecho medido -- ver docstring del módulo.
DEFAULT_STAR_COUNT = 2


def derive_champions(playoff_game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Campeón de cada temporada, derivado del ÚLTIMO partido de playoffs de
    esa temporada (el que cierra las Finales): el equipo que aparece con
    WL == 'W' en la fecha más tardía.

    `playoff_game_logs`: filas de backtest_sweep_advanced_game_logs.csv
    con game_phase == 'playoffs'. Devuelve (season, team_id,
    team_abbreviation, clinch_date).
    """
    if playoff_game_logs.empty:
        return pd.DataFrame(columns=["season", "team_id", "team_abbreviation", "clinch_date"])

    logs = playoff_game_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    rows = []
    for season, group in logs.groupby("season"):
        final_day = group[group["GAME_DATE"] == group["GAME_DATE"].max()]
        winner = final_day[final_day["WL"] == "W"]
        if winner.empty:
            continue
        row = winner.iloc[0]
        rows.append(
            {
                "season": season,
                "team_id": int(row["TEAM_ID"]),
                "team_abbreviation": row["TEAM_ABBREVIATION"],
                "clinch_date": row["GAME_DATE"],
            }
        )
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def compute_title_paths(champions: pd.DataFrame, standings: pd.DataFrame, playoff_game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Camino al título de cada campeón: desde qué seed salió, con qué
    récord, cuántos partidos de playoffs jugó y ganó, y a qué seeds
    eliminó por el camino.

    El "camino de rivales" se reconstruye de los game logs: los rivales
    distintos a los que se enfrentó en playoffs, en orden cronológico, con
    el seed que tenía cada uno esa temporada.
    """
    if champions.empty:
        return pd.DataFrame()

    seeds = standings.set_index(["season", "TeamID"])["PlayoffRank"].to_dict()
    wins = standings.set_index(["season", "TeamID"])["WINS"].to_dict()

    logs = playoff_game_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    abbrev_to_id = (
        logs.drop_duplicates("TEAM_ABBREVIATION").set_index("TEAM_ABBREVIATION")["TEAM_ID"].to_dict()
    )

    rows = []
    for champ in champions.itertuples():
        team_games = logs[(logs["season"] == champ.season) & (logs["TEAM_ID"] == champ.team_id)]
        team_games = team_games.sort_values("GAME_DATE")

        # Rivales en orden de aparición (una entrada por serie).
        opponents = []
        for matchup in team_games["MATCHUP"]:
            opponent = str(matchup).split(" @ ")[-1].split(" vs. ")[-1].strip()
            if not opponents or opponents[-1] != opponent:
                opponents.append(opponent)

        beaten_seeds = []
        for opponent in opponents:
            opponent_id = abbrev_to_id.get(opponent)
            seed = seeds.get((champ.season, opponent_id)) if opponent_id is not None else None
            beaten_seeds.append(int(seed) if pd.notna(seed) else None)

        rows.append(
            {
                "season": champ.season,
                "team_abbreviation": champ.team_abbreviation,
                "seed": int(seeds[(champ.season, champ.team_id)]) if (champ.season, champ.team_id) in seeds else None,
                "regular_season_wins": int(wins[(champ.season, champ.team_id)]) if (champ.season, champ.team_id) in wins else None,
                "playoff_games": int(len(team_games)),
                "playoff_wins": int((team_games["WL"] == "W").sum()),
                "opponents_faced": " → ".join(opponents),
                "seeds_beaten": " → ".join(str(s) for s in beaten_seeds if s is not None),
            }
        )
    return pd.DataFrame(rows)


def compute_roster_profile(
    roster: pd.DataFrame, career_stats: pd.DataFrame, season: str, star_count: int = DEFAULT_STAR_COUNT
) -> Dict[str, Any]:
    """
    Perfil de composición de UN roster en UNA temporada: reparto de
    minutos por posición, experiencia media ponderada por minutos, y qué
    porcentaje de los minutos se concentra en sus `star_count` jugadores
    más usados.

    `roster` son las filas de ese equipo-temporada de
    backtest_sweep_rosters.csv; `career_stats` las de
    backtest_sweep_player_career_stats.csv (se filtra a `season` para
    tomar los minutos REALES de esa temporada).
    """
    season_stats = career_stats[career_stats["SEASON_ID"] == season]
    minutes_by_player = season_stats.groupby("PLAYER_ID")["MIN"].sum()

    df = roster.copy()
    df["minutes"] = df["PLAYER_ID"].map(minutes_by_player).fillna(0.0)
    total_minutes = df["minutes"].sum()
    if total_minutes <= 0:
        return {}

    df["position_group"] = df["POSITION"].astype(str).str[0].map(POSITION_GROUPS)
    # "R" es como nba_api marca a un rookie -> 0 años de experiencia.
    df["experience"] = pd.to_numeric(df["EXP"].astype(str).replace("R", "0"), errors="coerce").fillna(0.0)

    minutes_share = df.groupby("position_group")["minutes"].sum() / total_minutes * 100
    top_minutes = df.nlargest(star_count, "minutes")["minutes"].sum()

    profile = {
        "star_minutes_share": float(top_minutes / total_minutes * 100),
        "weighted_experience": float((df["experience"] * df["minutes"]).sum() / total_minutes),
        "weighted_age": float((pd.to_numeric(df["AGE"], errors="coerce").fillna(0) * df["minutes"]).sum() / total_minutes),
        "players_with_minutes": int((df["minutes"] > 0).sum()),
    }
    for group in POSITION_GROUPS.values():
        profile[f"minutes_pct_{group}"] = float(minutes_share.get(group, 0.0))
    return profile


def compute_champion_profiles(
    champions: pd.DataFrame, rosters: pd.DataFrame, career_stats: pd.DataFrame,
    star_count: int = DEFAULT_STAR_COUNT,
) -> pd.DataFrame:
    """Aplica compute_roster_profile() a cada campeón. Una fila por temporada."""
    rows = []
    for champ in champions.itertuples():
        team_roster = rosters[(rosters["season"] == champ.season) & (rosters["TeamID"] == champ.team_id)]
        if team_roster.empty:
            continue
        profile = compute_roster_profile(team_roster, career_stats, champ.season, star_count)
        if not profile:
            continue
        rows.append({"season": champ.season, "team_abbreviation": champ.team_abbreviation, **profile})
    return pd.DataFrame(rows)


def compute_seed_trajectories(standings: pd.DataFrame) -> pd.DataFrame:
    """
    Trayectoria de seed de cada franquicia a lo largo de las temporadas
    disponibles: una fila por franquicia, una columna por temporada, con
    el PlayoffRank de esa temporada (1-15 dentro de su conferencia).
    Permite ver de un vistazo quién sostuvo un nivel alto y quién osciló.
    """
    if standings.empty:
        return pd.DataFrame()
    pivot = standings.pivot_table(
        index="TeamName", columns="season", values="PlayoffRank", aggfunc="first"
    )
    return pivot.sort_index()


def build_champion_analysis_dataset(config: Dict[str, Any]) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Punto de entrada: lee los datasets del backtest sweep y guarda
    - champion_title_paths.csv (camino al título de cada campeón)
    - champion_roster_profiles.csv (composición de cada roster campeón)
    - champion_seed_trajectories.csv (seed de cada franquicia por temporada)

    None si no se ha corrido `python src/data_pipeline.py --backtest-sweep`
    (este análisis reutiliza SUS datos, no descarga nada nuevo).
    """
    paths = get_paths(config)
    processed = paths["processed"]
    required = [
        "backtest_sweep_advanced_game_logs.csv",
        "backtest_sweep_standings.csv",
        "backtest_sweep_rosters.csv",
        "backtest_sweep_player_career_stats.csv",
    ]
    if any(not (processed / name).exists() for name in required):
        return None

    logs = pd.read_csv(processed / "backtest_sweep_advanced_game_logs.csv")
    standings = pd.read_csv(processed / "backtest_sweep_standings.csv")
    rosters = pd.read_csv(processed / "backtest_sweep_rosters.csv")
    career_stats = pd.read_csv(processed / "backtest_sweep_player_career_stats.csv")

    playoff_logs = logs[logs["game_phase"] == "playoffs"]
    champions = derive_champions(playoff_logs)
    star_count = config.get("champion_analysis", {}).get("star_count", DEFAULT_STAR_COUNT)

    title_paths = compute_title_paths(champions, standings, playoff_logs)
    profiles = compute_champion_profiles(champions, rosters, career_stats, star_count)
    trajectories = compute_seed_trajectories(standings)

    title_paths.to_csv(processed / "champion_title_paths.csv", index=False)
    profiles.to_csv(processed / "champion_roster_profiles.csv", index=False)
    trajectories.to_csv(processed / "champion_seed_trajectories.csv")
    print(f"Guardado: champion_title_paths.csv ({len(title_paths)} campeones)")
    print(f"Guardado: champion_roster_profiles.csv ({len(profiles)} perfiles)")
    print(f"Guardado: champion_seed_trajectories.csv ({len(trajectories)} franquicias)")

    return {"title_paths": title_paths, "profiles": profiles, "trajectories": trajectories}


if __name__ == "__main__":
    from config_loader import load_config

    build_champion_analysis_dataset(load_config())
