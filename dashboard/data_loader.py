"""
data_loader.py

Funciones puras de carga/combinación de los CSV en data/processed/ --
la capa de datos que consume webapp/ (cada router de webapp/routers/
importa de aquí en vez de leer CSV por su cuenta). Vivió originalmente
junto a un dashboard de Streamlit (dashboard/app.py, retirado del
proyecto -- webapp/ es ahora la única interfaz) precisamente porque ya
estaba separado de esa capa de renderizado: la lógica de qué datos se
muestran es testeable (ver tests/test_dashboard_data_loader.py) con
independencia de cómo se pinte.

Ninguna función aquí llama a la API -- todo lee CSV ya generados por el
pipeline (data_pipeline.py, aging_curve.py, injury_model.py,
fatigue_accumulation.py, simulation.py, backtesting.py, lineup_synergy.py,
league_simulation.py). Si un CSV no existe todavía, se devuelve None --
el caller (webapp/) decide cómo avisar al usuario, esta capa no lanza
excepciones por archivos faltantes.

CUATRO EXCEPCIONES a "solo lee CSV": run_single_bracket_simulation() sí
ejecuta código en vivo (una simulación de bracket de playoffs) -- es
rápido (proyecta 30 equipos desde CSV ya cacheados, sin red) y el botón
de la pestaña "Liga y Playoffs" lo necesita para mostrar un bracket
distinto cada vez que se pulsa. run_single_season_player_log_simulation()
simula UNA temporada concreta (no la distribución agregada de
simulation_results.csv) para el botón "Simular partidos de la temporada"
de la pestaña Simulación -- misma idea que el bracket: rápido, sin red,
sobre CSV ya cacheados. run_single_league_season_simulation() es lo
mismo pero para los 30 equipos de Liga NBA -- calendario, resultado de
cada partido y boxscore ilustrativo por jugador, para el botón "Simular
calendario de la temporada" (a diferencia del bracket y del resto de la
liga, SÍ persiste CSV propios -- league_single_season_game_log.csv /
league_single_season_player_box_scores.csv -- porque son caros de
regenerar en cada request). compute_awards_summary() no
llama a la API ni simula nada nuevo, pero sí importa lógica de negocio
de src/awards_projection.py (en vez de solo leer y reformatear un CSV)
para calcular los premios individuales sobre los CSV ya generados.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_loader import get_paths  # noqa: E402


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


# Estadísticas "por partido" mostradas en el dashboard -> columna de
# aging_curve_projection.csv / league_player_projections.csv de la que se
# derivan (total de temporada / games_per_season). Un solo diccionario
# para no repetir la lista de columnas entre los loaders y la leyenda.
PER_GAME_STATS: Dict[str, str] = {
    "PPG": "PTS_projected",
    "RPG": "REB_projected",
    "APG": "AST_projected",
    "SPG": "STL_projected",
    "BPG": "BLK_projected",
    "TOPG": "TOV_projected",
    "3PM": "FG3M_projected",
}

# Columnas de TOTAL de temporada -> nombre limpio para mostrar (vista
# "Totales" del toggle en la pestaña de roster).
TOTAL_STATS: Dict[str, str] = {
    "PTS_projected": "PTS",
    "REB_projected": "REB",
    "AST_projected": "AST",
    "STL_projected": "STL",
    "BLK_projected": "BLK",
    "TOV_projected": "TOV",
    "FG3M_projected": "3PM",
}

ROSTER_META_COLUMNS: List[str] = [
    "player_name", "unit", "role_expected", "current_age", "target_age",
    "GP", "MPG", "minutes_projection", "game_score_per36", "risk_score", "fatigue_score",
]
LEAGUE_PLAYER_META_COLUMNS: List[str] = [
    "player_name", "team_abbreviation", "conference", "current_age", "target_age",
    "GP", "MPG", "minutes_projection", "game_score_per36", "risk_score", "fatigue_score",
]

# Columnas crudas (de aging_curve.project_player_season) -> nombre limpio
# de display. El NOMBRE de columna sigue siendo "*_last_season" (así se
# llama en los CSV ya generados por el pipeline), pero el CONTENIDO que
# llega aquí ya no es necesariamente el histórico real:
# _apply_simulated_games_and_minutes() lo sustituye por la versión
# simulada (partidos/minutos esperados de la temporada que se está
# simulando) antes de este rename, cuando hay risk_score disponible --
# distinto de minutes_projection, que sigue siendo el minutaje ASUMIDO
# (input fijo, no una salida del modelo).
GAMES_MINUTES_DISPLAY_COLUMNS: Dict[str, str] = {
    "games_played_last_season": "GP",
    "minutes_per_game_last_season": "MPG",
}

# Leyenda completa de todas las columnas que puede mostrar la pestaña de
# roster (y la de liga, que comparte casi las mismas) -- una sola fuente
# de verdad para los tooltips de columna y la leyenda de texto en webapp/.
ROSTER_STAT_GLOSSARY: Dict[str, str] = {
    "PPG": "Puntos por partido (proyectados)",
    "RPG": "Rebotes por partido (proyectados)",
    "APG": "Asistencias por partido (proyectadas)",
    "SPG": "Robos de balón por partido (proyectados)",
    "BPG": "Tapones por partido (proyectados)",
    "TOPG": "Pérdidas de balón por partido (proyectadas)",
    "3PM": "Triples anotados por partido (proyectados)",
    "PTS": "Puntos totales proyectados para la temporada completa",
    "REB": "Rebotes totales proyectados para la temporada completa",
    "AST": "Asistencias totales proyectadas para la temporada completa",
    "STL": "Robos de balón totales proyectados para la temporada completa",
    "BLK": "Tapones totales proyectados para la temporada completa",
    "TOV": "Pérdidas de balón totales proyectadas para la temporada completa",
    "FG%": "Porcentaje de tiro de campo proyectado (tiros anotados / tiros intentados)",
    "3P%": "Porcentaje de triples proyectado (triples anotados / triples intentados)",
    "game_score_per36": "Game Score de Hollinger proyectado por-36 minutos -- una sola nota de impacto que combina volumen y eficiencia (ver aging_curve.py)",
    "risk_score": "Riesgo de lesión (injury_model.py), de 0 a 1 -- basado en historial de partidos perdidos, no en la edad por sí sola",
    "fatigue_score": "Desgaste acumulado por carga de minutos (fatigue_accumulation.py), de 0 a 1 -- carrera larga + uso pesado reciente + rachas sin descanso",
    "current_age": "Edad del jugador en su temporada más reciente registrada",
    "target_age": "Edad proyectada para la temporada de team_config.yaml",
    "minutes_projection": "Minutos por partido asumidos (para el roster propio: dato de team_config.yaml; para el resto de la liga: minutos/partido reales de la temporada más reciente de cada jugador)",
    "conference": "Conferencia de la franquicia (Este/Oeste)",
    "GP": "Partidos jugados ESPERADOS en la temporada simulada -- games_per_season × (1 − risk_score), la media exacta del modelo de riesgo de lesión (simulation.compute_expected_games_played), no el dato histórico de la temporada pasada",
    "MPG": "Minutos por partido asumidos los partidos en que el jugador SÍ juega (minutes_projection, el input curado del roster) -- no se descuenta por riesgo de lesión, a diferencia de GP: representa el ritmo de juego, no la carga total de temporada",
}

SIMULATION_GLOSSARY: Dict[str, str] = {
    "wins": "Victorias en esta temporada simulada",
    "losses": "Derrotas en esta temporada simulada",
    "net_rating_estimate_mean": "Diferencial de puntos estimado medio de la temporada (Game Score de equipo menos línea base de equipo promedio, menos ajuste por rival)",
    "total_games_missed": "Suma de partidos perdidos por lesión de todos los jugadores del roster en esta temporada simulada",
}

SYNERGY_GLOSSARY: Dict[str, str] = {
    "usage_clash": "Fricción cuando ambos jugadores de la pareja tienen uso alto (compiten por el balón) -- 0 si alguno está por debajo del umbral",
    "playmaking_spacing_synergy": "Bonus cuando un creador de juego comparte cancha con un tirador exterior (efecto 'gravedad')",
    "pair_weight": "Cuánto podrían compartir cancha, aproximado como min(minutos_i, minutos_j) / 48",
    "net_pair_score": "Efecto neto de la pareja: playmaking_spacing_synergy menos usage_clash, ponderado por pair_weight",
}

BACKTEST_GLOSSARY: Dict[str, str] = {
    "games_in_season": "Partidos de temporada regular de ese caso histórico real",
    "actual_wins": "Victorias REALES de ese equipo esa temporada",
    "simulated_wins_mean": "Media de victorias en las temporadas simuladas retrospectivamente",
    "simulated_wins_p10": "Percentil 10 de la distribución simulada de victorias",
    "simulated_wins_p50": "Mediana de la distribución simulada de victorias",
    "simulated_wins_p90": "Percentil 90 de la distribución simulada de victorias",
    "actual_percentile": "En qué percentil de la distribución simulada cae el resultado REAL -- percentiles extremos (cerca de 0 o 100) señalan que el modelo se aleja de lo que realmente pasó",
    "league_baseline_per36": "Game Score/36 medio de la liga en ESA temporada, usado como referencia de 'equipo promedio' -- corrige la inflación de era (el nivel de Game Score de la NBA subió de ~10.7 en 2010-11 a ~13.4 en 2024-25); ver aging_curve.compute_league_game_score_baseline",
}

CALIBRATION_GLOSSARY: Dict[str, str] = {
    "n_cases": "Número de casos (equipo-temporada) incluidos en el resumen",
    "pct_within_p10_p90": "% de casos donde el resultado REAL cae dentro del rango P10-P90 simulado -- debería rondar el 80% en un modelo bien calibrado (por construcción, P10-P90 es el 80% central de la distribución)",
    "mean_percentile": "Percentil real medio en la distribución simulada -- debería rondar 50 si el modelo no tiene sesgo sistemático",
    "median_percentile": "Percentil real mediano -- mismo chequeo que mean_percentile, menos sensible a casos extremos",
    "mean_absolute_error_wins": "Error absoluto medio entre victorias reales y la media simulada, en victorias de temporada",
    "mean_error_wins": "Error medio (real menos predicho) -- positivo indica que el modelo SUBESTIMA victorias en promedio; negativo, que las SOBREESTIMA",
    "correlation_actual_vs_predicted": "Correlación entre victorias reales y la media simulada -- cercano a 1 indica que el modelo distingue equipos buenos de malos aunque el nivel absoluto esté desplazado",
}

LEAGUE_GLOSSARY: Dict[str, str] = {
    "wins_mean": "Victorias medias simuladas de temporada regular",
    "wins_p10": "Percentil 10 de victorias simuladas",
    "wins_p90": "Percentil 90 de victorias simuladas",
    "playoff_pct": "% de temporadas simuladas en que el equipo entra a playoffs (tras resolver el play-in)",
    "conf_semis_pct": "% de temporadas simuladas en que el equipo llega a semifinales de conferencia",
    "conf_finals_pct": "% de temporadas simuladas en que el equipo llega a finales de conferencia",
    "finals_pct": "% de temporadas simuladas en que el equipo gana su conferencia y llega a las Finales de la NBA",
    "championship_pct": "% de temporadas simuladas en que el equipo gana el título",
    "seed": "Posición dentro de su conferencia, ordenada por victorias medias simuladas (1 = mejor)",
}

# "situacion" es texto, no numérico -- se documenta aparte para el texto
# de leyenda (glosario expandible en webapp/).
STANDINGS_SITUATION_GLOSSARY = (
    "**situacion** — según el seed dentro de la conferencia: **Clasifica directo** "
    "(seeds 1-6), **Play-in** (seeds 7-10, juegan por las 2 últimas plazas de playoffs), "
    "**Fuera** (seeds 11-15)."
)


def load_roster_overview(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Combina, por jugador del roster: metadata de team_config.yaml
    (role_expected, minutes_projection, unit), la proyección de
    aging_curve.py (totales de temporada, stats por partido derivadas --
    ver PER_GAME_STATS -- y game_score_per36), el risk_score de
    injury_model.py, el fatigue_score de fatigue_accumulation.py y la
    posición real de roster_positions.csv (data_pipeline.py, vía
    CommonPlayerInfo -- necesaria para los quintetos All-NBA/All-
    Defensive de awards_projection.py). None si aging_curve_projection.csv
    no existe todavía (el resto son opcionales, se dejan en blanco si
    faltan). Devuelve TODAS las columnas (totales y por partido) --
    select_roster_view() elige el subconjunto a mostrar según el modo
    activo en el dashboard.
    """
    paths = get_paths(config)
    aging = _read_csv_if_exists(paths["processed"] / "aging_curve_projection.csv")
    if aging is None:
        return None

    injury = _read_csv_if_exists(paths["processed"] / "injury_risk.csv")
    fatigue = _read_csv_if_exists(paths["processed"] / "fatigue_risk.csv")
    positions = _read_csv_if_exists(paths["processed"] / "roster_positions.csv")
    games_per_season = config["simulation"]["games_per_season"]

    roster_meta = pd.DataFrame(
        [
            {
                "player_id": p["player_id"],
                "role_expected": p.get("role_expected"),
                "minutes_projection": p.get("minutes_projection"),
                "unit": p.get("unit"),
            }
            for p in config["roster"]
            if p.get("player_id")
        ]
    )

    overview = aging.merge(roster_meta, on="player_id", how="left")
    if injury is not None:
        overview = overview.merge(injury[["player_id", "risk_score"]], on="player_id", how="left")
    if fatigue is not None:
        overview = overview.merge(fatigue[["player_id", "fatigue_score"]], on="player_id", how="left")
    if positions is not None:
        merge_cols = [c for c in ["player_id", "position", "country"] if c in positions.columns]
        overview = overview.merge(positions[merge_cols], on="player_id", how="left")

    overview = _add_per_game_and_fg_pct(overview, games_per_season)
    return overview.sort_values("game_score_per36", ascending=False).reset_index(drop=True)


def _add_per_game_and_fg_pct(df: pd.DataFrame, games_per_season: int) -> pd.DataFrame:
    df = df.copy()
    for per_game_col, total_col in PER_GAME_STATS.items():
        if total_col in df.columns:
            df[per_game_col] = df[total_col] / games_per_season
    if "FGM_projected" in df.columns and "FGA_projected" in df.columns:
        df["FG%"] = (df["FGM_projected"] / df["FGA_projected"] * 100).round(1)
    if "FG3M_projected" in df.columns and "FG3A_projected" in df.columns:
        df["3P%"] = (df["FG3M_projected"] / df["FG3A_projected"] * 100).round(1)
    return df


def _apply_simulated_games_and_minutes(
    overview: pd.DataFrame, games_per_season: Optional[int]
) -> pd.DataFrame:
    """
    Sustituye `games_played_last_season`/`minutes_per_game_last_season`
    (dato histórico REAL) por sus versiones SIMULADAS. Fila por fila:
    donde falte `risk_score` o `minutes_projection` se conserva el valor
    histórico de esa fila en vez de dejarlo en blanco -- degradar así es
    más útil que perder el dato por completo.

    GP = partidos jugados ESPERADOS en la temporada simulada
    (`simulation.compute_expected_games_played`, games_per_season ×
    (1 − risk_score)) -- SÍ depende del riesgo de lesión.

    MPG = `minutes_projection` tal cual, el input curado del roster --
    los minutos que se asumen los partidos en que el jugador SÍ juega.
    A propósito NO se descuenta por risk_score (antes se mostraba
    minutes_projection × (1 − risk_score), la carga efectiva a lo largo
    de TODA la temporada contando 0 en partidos perdidos -- cambiado
    porque en pantalla, para un jugador de alto riesgo como Embiid,
    salía una cifra como "9 MPG" que parecía un error: nadie juega 9
    minutos cuando sale a cancha, ese número mezclaba "minutos por
    partido" con "carga de temporada" en una sola cifra confusa. El
    usuario prefirió separar los dos conceptos: MPG = ritmo cuando
    juega, GP = para cuántos partidos se espera que esté disponible).

    También escala los TOTALES de temporada (columnas `TOTAL_STATS`:
    PTS_projected, REB_projected...) por el mismo factor de
    disponibilidad `(1 - risk_score)` -- BUG REAL reportado por el
    usuario: al cambiar el escenario de liga "con" / "sin lesiones" solo
    se movían GP y MPG, los puntos/rebotes/asistencias totales quedaban
    igual, como si jugar menos partidos no costara nada de producción.
    Las columnas PER_GAME_STATS (PPG/RPG/...) se dejan SIN escalar a
    propósito -- representan el ritmo del jugador cuando SÍ juega (igual
    que un PPG real de la NBA no baja porque un jugador se pierda
    partidos), y de hecho la relación Total = PPG × GP se mantiene
    exacta con este diseño (PTS_projected × disponibilidad =
    PTS_projected/games_per_season × (games_per_season ×
    disponibilidad) = PPG constante × GP simulado). Mismo factor exacto
    que ya usa `webapp/routers/players.py::_projected_season_row` para
    la fila de proyección del popup de jugador -- no una fórmula nueva.

    LIMITACIÓN CONOCIDA (no arreglada a propósito, preguntado al
    usuario): PPG/RPG/APG es un punto fijo (ritmo per-36 × minutes_projection,
    NINGUNA de las dos depende de risk_score), así que es
    matemáticamente IDÉNTICO en los escenarios "con" y "sin lesiones" --
    no hay ruido partido a partido simulado a nivel de jugador en esta
    tabla. Esto ignora un efecto real: jugar los 82 partidos sin
    descanso (escenario "sin lesiones") acumula fatiga y podría reducir
    el ritmo real, mientras que un jugador que se pierde partidos por
    lesión llega más descansado a los que sí juega. El proyecto ya mide
    ese desgaste (`fatigue_score`, ver fatigue_accumulation.py) pero hoy
    solo alimenta el resultado ganar/perder a nivel de EQUIPO, nunca se
    conecta a estos promedios individuales por partido -- conectarlo
    exigiría decidir la magnitud del efecto y recalibrarlo, no es un
    ajuste trivial.
    """
    if games_per_season is None or "risk_score" not in overview.columns:
        return overview
    from simulation import compute_expected_games_played

    overview = overview.copy()
    has_risk = overview["risk_score"].notna()

    if "games_played_last_season" in overview.columns:
        # El histórico llega como entero (partidos jugados); el simulado
        # es continuo -- convertir a float ANTES de asignar, si no pandas
        # avisa (y en el futuro fallará) al meter floats en una columna int.
        overview["games_played_last_season"] = overview["games_played_last_season"].astype(float)
        simulated_gp = compute_expected_games_played(overview["risk_score"].fillna(0).to_numpy(), games_per_season)
        overview.loc[has_risk, "games_played_last_season"] = simulated_gp[has_risk.to_numpy()]

    if "minutes_per_game_last_season" in overview.columns and "minutes_projection" in overview.columns:
        has_minutes = has_risk & overview["minutes_projection"].notna()
        overview.loc[has_minutes, "minutes_per_game_last_season"] = overview.loc[has_minutes, "minutes_projection"]

    availability = (1 - overview["risk_score"].fillna(0)).to_numpy()
    for total_col in TOTAL_STATS:
        if total_col in overview.columns:
            overview.loc[has_risk, total_col] = overview.loc[has_risk, total_col] * availability[has_risk.to_numpy()]

    return overview


def select_roster_view(
    overview: pd.DataFrame,
    mode: str,
    meta_columns: Optional[List[str]] = None,
    games_per_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Selecciona el subconjunto de columnas a mostrar según el modo del
    toggle: "totals" (temporada completa proyectada, nombres limpios vía
    TOTAL_STATS) o "per_game" (por partido, ya con nombres limpios --
    PPG/RPG/... -- vía PER_GAME_STATS). `meta_columns` por defecto es
    ROSTER_META_COLUMNS; se pasa uno distinto para league players (que no
    tienen role_expected/unit del config propio).

    GP y MPG se SUSTITUYEN por sus versiones simuladas
    (`simulation.compute_expected_games_played` /
    `compute_expected_effective_minutes_per_game`) cuando se pasa
    `games_per_season` y el DataFrame trae `risk_score` y
    `minutes_projection` -- ver `_apply_simulated_games_and_minutes`. Sin
    `games_per_season`, o si falta alguna de las otras dos columnas (p.
    ej. injury_risk.csv no se ha corrido todavía), se cae a los valores
    históricos reales sin fallar.
    """
    meta_columns = meta_columns if meta_columns is not None else ROSTER_META_COLUMNS
    overview = _apply_simulated_games_and_minutes(overview, games_per_season)
    overview = overview.rename(columns=GAMES_MINUTES_DISPLAY_COLUMNS)
    if "GP" in overview.columns:
        overview = overview.assign(GP=overview["GP"].round(0).astype("Int64"))
    if "MPG" in overview.columns:
        overview = overview.assign(MPG=overview["MPG"].round(1))
    meta_cols = [c for c in meta_columns if c in overview.columns]

    if mode == "totals":
        # Redondeado a entero: son proyecciones continuas (aging_curve.py
        # no predice un número exacto de puntos), pero "1600 PTS" se lee
        # mejor que "1600.34" -- decimales en un TOTAL de temporada no
        # aportan precisión real, solo ruido visual.
        stat_cols = [c for c in TOTAL_STATS if c in overview.columns]
        stat_df = overview[stat_cols].round(0).astype("Int64").rename(columns=TOTAL_STATS)
    else:
        stat_cols = [c for c in PER_GAME_STATS if c in overview.columns]
        stat_df = overview[stat_cols].round(1)

    parts = [overview[meta_cols].reset_index(drop=True), stat_df.reset_index(drop=True)]
    pct_cols = [c for c in ["FG%", "3P%"] if c in overview.columns]
    if pct_cols:
        parts.append(overview[pct_cols].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def load_league_player_projections(config: Dict[str, Any], scenario: str = "with_injuries") -> Optional[pd.DataFrame]:
    """
    Proyecciones por jugador de los 30 equipos (league_simulation.py).
    None si no se ha corrido. `scenario="no_injuries"` lee la variante
    sin riesgo de lesión (ver league_simulation._apply_scenario) --
    default sin cambios, mismo archivo de siempre, así que los callers
    existentes siguen funcionando igual sin tocar sus llamadas.
    """
    from league_simulation import _scenario_suffix

    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / f"league_player_projections{_scenario_suffix(scenario)}.csv")


def load_simulation_results(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / "simulation_results.csv")


def load_backtest_summary(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / "backtest_summary.csv")


def load_backtest_sweep_summary(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Resultado por caso del backtesting sistemático a gran escala (30
    equipos x N temporadas -- ver src/backtesting.build_backtest_sweep_dataset).
    None si no se ha corrido `python src/data_pipeline.py --backtest-sweep`
    seguido de `build_backtest_sweep_dataset`."""
    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / "backtest_sweep_summary.csv")


CHAMPION_GLOSSARY: Dict[str, str] = {
    "seed": "Puesto del campeón en su conferencia esa temporada (1 = mejor récord)",
    "regular_season_wins": "Victorias REALES de temporada regular del campeón",
    "playoff_games": "Partidos de playoffs jugados en su camino al título",
    "playoff_wins": "Partidos de playoffs ganados (siempre 16: cuatro rondas a 4 victorias)",
    "opponents_faced": "Rivales eliminados, en orden cronológico",
    "seeds_beaten": "Seed de cada rival eliminado, en el mismo orden",
    "star_minutes_share": "% de los minutos totales del equipo concentrados en sus 2 jugadores más usados -- mide si el título se ganó con estrellas o con reparto",
    "weighted_experience": "Años de experiencia NBA medios del roster, ponderados por minutos jugados",
    "weighted_age": "Edad media del roster, ponderada por minutos jugados",
    "players_with_minutes": "Número de jugadores del roster que jugaron algún minuto esa temporada",
    "minutes_pct_Base/Escolta": "% de minutos del equipo jugados por bases y escoltas (posición 'G')",
    "minutes_pct_Alero/Ala-pívot": "% de minutos jugados por aleros y ala-pívots (posición 'F')",
    "minutes_pct_Pívot": "% de minutos jugados por pívots (posición 'C')",
}


def load_champion_title_paths(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Camino al título de cada campeón real (src/champion_profiles.py). None si no se ha generado."""
    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / "champion_title_paths.csv")


def load_champion_roster_profiles(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Composición del roster de cada campeón real (posiciones, experiencia, concentración)."""
    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / "champion_roster_profiles.csv")


def load_champion_seed_trajectories(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Seed de cada franquicia por temporada (matriz franquicia x temporada)."""
    paths = get_paths(config)
    path = paths["processed"] / "champion_seed_trajectories.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, index_col=0)


def compute_champion_seed_distribution(title_paths: pd.DataFrame) -> pd.DataFrame:
    """
    Distribución REAL de seeds campeones -- el contraste más útil contra
    el simulador. En las temporadas cubiertas, ningún campeón salió de un
    seed peor que el 3, así que si el modelo produce campeones de seed 4+
    con frecuencia apreciable es una miscalibración medible.
    """
    if title_paths is None or title_paths.empty or "seed" not in title_paths.columns:
        return pd.DataFrame(columns=["seed", "n_champions", "pct"])
    counts = title_paths["seed"].value_counts().sort_index()
    return pd.DataFrame(
        {"seed": counts.index.astype(int), "n_champions": counts.values, "pct": counts.values / counts.sum() * 100}
    ).reset_index(drop=True)


def load_backtest_sweep_calibration(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Resumen agregado de calibración (una sola fila) del backtesting
    sistemático -- ver src/backtesting.compute_calibration_summary."""
    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / "backtest_sweep_calibration.csv")


def load_lineup_synergy_pairs(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / "lineup_synergy_pairs.csv")


def load_league_regular_season_summary(config: Dict[str, Any], scenario: str = "with_injuries") -> Optional[pd.DataFrame]:
    from league_simulation import _scenario_suffix

    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / f"league_regular_season_summary{_scenario_suffix(scenario)}.csv")


def load_league_playoff_summary(config: Dict[str, Any], scenario: str = "with_injuries") -> Optional[pd.DataFrame]:
    from league_simulation import _scenario_suffix

    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / f"league_playoff_summary{_scenario_suffix(scenario)}.csv")


def load_league_single_season_game_log(config: Dict[str, Any], scenario: str = "with_injuries") -> Optional[pd.DataFrame]:
    """Calendario y resultado de UNA temporada concreta ya simulada (ver
    run_single_league_season_simulation) -- None si todavía no se ha
    pulsado el botón que la genera."""
    from league_simulation import _scenario_suffix

    paths = get_paths(config)
    return _read_csv_if_exists(paths["processed"] / f"league_single_season_game_log{_scenario_suffix(scenario)}.csv")


def load_league_single_season_player_box_scores(
    config: Dict[str, Any], scenario: str = "with_injuries"
) -> Optional[pd.DataFrame]:
    """Boxscore ILUSTRATIVO por jugador y partido de la misma temporada
    concreta que load_league_single_season_game_log -- ver el caveat en
    league_simulation.simulate_single_league_season_player_box_scores."""
    from league_simulation import _scenario_suffix

    paths = get_paths(config)
    return _read_csv_if_exists(
        paths["processed"] / f"league_single_season_player_box_scores{_scenario_suffix(scenario)}.csv"
    )


# Estado real de playoffs NBA aplicado al seed simulado (1-6 clasifica
# directo, 7-10 juega el play-in, 11-15 queda fuera) -- mismo formato que
# league_simulation.py simula (ver resolve_play_in / simulate_conference_bracket).
PLAYOFF_SEED_CUTOFFS = {"direct": 6, "play_in": 10}


def _playoff_situation(seed: int) -> str:
    if seed <= PLAYOFF_SEED_CUTOFFS["direct"]:
        return "Clasifica directo"
    if seed <= PLAYOFF_SEED_CUTOFFS["play_in"]:
        return "Play-in"
    return "Fuera"


def compute_conference_standings(
    regular_season_summary: pd.DataFrame, playoff_summary: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Clasificación de la temporada regular simulada, dividida por
    conferencia ("East"/"West") y ordenada por victorias medias
    descendente (seed 1-15), con las probabilidades de playoffs/
    campeonato ya calculadas mergeadas. Añade una columna `situacion`
    marcando la línea real de playoffs de la NBA que league_simulation.py
    simula: seeds 1-6 clasifican directo, 7-10 juegan el play-in, 11-15
    quedan fuera. Devuelve {"East": df, "West": df}, cada uno con 15
    filas (o menos si a algún equipo le faltan datos de playoffs).
    """
    playoff_cols = ["team_abbreviation", "playoff_pct", "conf_semis_pct", "conf_finals_pct", "finals_pct", "championship_pct"]
    merged = regular_season_summary.merge(
        playoff_summary[[c for c in playoff_cols if c in playoff_summary.columns]],
        on="team_abbreviation", how="left",
    )

    standings: Dict[str, pd.DataFrame] = {}
    for conference in ("East", "West"):
        conf_df = merged[merged["conference"] == conference].sort_values(
            "wins_mean", ascending=False
        ).reset_index(drop=True)
        conf_df.insert(0, "seed", conf_df.index + 1)
        conf_df["situacion"] = conf_df["seed"].apply(_playoff_situation)
        standings[conference] = conf_df
    return standings


AWARDS_GLOSSARY: Dict[str, str] = {
    "mpg": "Minutos por partido proyectados (media de toda la temporada simulada)",
    "season_value": "Valor de temporada completa: Game Score de Hollinger proyectado por-36 minutos, escalado a los minutos totales de la temporada -- ver docstring de src/awards_projection.py",
    "team_win_pct": "% de victorias proyectado del equipo del jugador (usado para ponderar el MVP -- jugar en un equipo ganador pesa)",
    "mvp_score": "season_value ponderado por team_win_pct -- nota compuesta usada para ordenar el ranking de MVP",
    "defensive_score_per36": "Proxy de impacto defensivo por-36 min: 1.5*robos + 1.5*tapones + 0.3*rebote defensivo - 0.2*faltas (heurística de este proyecto, no una métrica oficial -- el box score no ve casi nada de la defensa individual real)",
    "dpoy_score": "defensive_score_per36 escalado a los minutos totales de la temporada",
    "latest_season": "Temporada REAL más reciente usada para el cálculo de mejora (MIP)",
    "previous_game_score_per36": "Game Score por-36 REAL (no proyectado) de la temporada anterior a la más reciente",
    "latest_game_score_per36": "Game Score por-36 REAL (no proyectado) de la temporada más reciente",
    "improvement": "latest_game_score_per36 menos previous_game_score_per36 -- la mejora ya ocurrida (MIP se basa en temporadas reales, no en la proyección simulada)",
    "prior_wins": "Victorias REALES de la temporada anterior a la simulada (prior_season_standings.csv)",
    "wins_mean": "Victorias medias proyectadas en la temporada simulada",
    "win_improvement": "wins_mean menos prior_wins -- proxy de \"equipo que más mejoró\", usado como heurística de Entrenador del Año (este proyecto no modela entrenadores)",
    "team_record": "Récord \"V-D\" proyectado del equipo del jugador en la temporada simulada",
    "prev_PPG": "Puntos por partido REALES de la última temporada ya jugada (no proyectados) -- para comparar en MIP contra la proyección",
    "prev_RPG": "Rebotes por partido REALES de la última temporada ya jugada",
    "prev_APG": "Asistencias por partido REALES de la última temporada ya jugada",
    "prev_SPG": "Robos de balón por partido REALES de la última temporada ya jugada",
    "prev_BPG": "Tapones por partido REALES de la última temporada ya jugada",
    "prev_FG%": "Porcentaje de tiro de campo REAL de la última temporada ya jugada",
    "prev_3P%": "Porcentaje de triples REAL de la última temporada ya jugada",
    "prev_season": "Última temporada REAL ya jugada -- la que precede a la proyección, no la penúltima que usa `improvement`",
}

SEASON_AWARDS_GLOSSARY: Dict[str, str] = {
    "conference": "Conferencia del jugador (Este/Oeste) -- el All-Star se selecciona por conferencia cuando hay datos de los 30 equipos",
    "season_value": "Valor de temporada completa (ver AWARDS_GLOSSARY) -- métrica de ranking del All-Star y del All-NBA",
    "team_win_pct": "% de victorias proyectado del equipo del jugador",
    "position_slot": "Cupo de posición del quinteto: G (base/escolta), F (alero/ala-pívot) o C (pívot) -- formato clásico 2-2-1",
    "games_played_expected": "Partidos jugados ESPERADOS en la temporada simulada (games_per_season × (1 − risk_score)) -- debe ser ≥ 65 para optar a All-NBA/All-Defensive, sin restricción para el All-Star",
    "defensive_value": "Proxy de impacto defensivo escalado a temporada completa (mismo defensive_score_per36 que DPOY) -- métrica de ranking del All-Defensive",
    "team": "Quinteto: Primero/Segundo/Tercero (All-NBA) o Primero/Segundo (All-Defensive)",
    "selection_type": "Titular, Reserva, o \"Añadido por el comisionado\" (cuota de nacionalidad) -- SOLO una etiqueta sobre el ranking de season_value, NO una simulación del voto real (50% fans + 25% jugadores + 25% medios para titulares, entrenadores para reservas) ni del criterio real del comisionado",
    "country": "País de nacimiento del jugador (CommonPlayerInfo) -- usado solo para el chequeo de cuota de nacionalidad del All-Star",
    "commissioner_pick": "True si el jugador se añadió para cubrir la cuota de nacionalidad, NO por mérito del ranking natural -- ver el aviso en la tabla",
    "team_record": "Récord \"V-D\" proyectado del equipo del jugador en la temporada simulada",
    "PPG": "Puntos por partido proyectados",
    "RPG": "Rebotes por partido proyectados",
    "APG": "Asistencias por partido proyectadas",
    "SPG": "Robos de balón por partido proyectados",
    "BPG": "Tapones por partido proyectados",
    "FG%": "Porcentaje de tiro de campo proyectado",
    "3P%": "Porcentaje de triples proyectado",
}


def _win_loss_record(wins_mean: float, games_per_season: int) -> str:
    """"V-D" redondeado a partir de una media continua de victorias simuladas."""
    wins = round(wins_mean)
    return f"{wins}-{games_per_season - wins}"

def compute_awards_summary(
    config: Dict[str, Any], top_n: int = 5, scenario: str = "with_injuries"
) -> Optional[Dict[str, Any]]:
    """
    Calcula los premios individuales heurísticos (ver src/awards_projection.py:
    MVP, DPOY, 6MOY, ROY, MIP y, si hay datos de liga completa, COY) sobre
    el scope disponible -- los 30 equipos de la liga si
    league_player_projections.csv existe (más significativo: compara
    contra toda la NBA), si no sobre el roster propio
    (aging_curve_projection.csv). None si ni siquiera el roster propio
    está disponible todavía.

    `scenario="no_injuries"` calcula los premios sobre la variante de
    liga sin riesgo de lesión -- afecta directamente la elegibilidad de
    All-NBA/All-Defensive (`games_played_expected >= 65`, ver
    src/awards_projection.py), que con `risk_score=0` deja de excluir a
    jugadores propensos a lesión. No afecta al scope "own" (roster
    propio), que no depende de la liga completa.
    """
    import awards_projection as ap

    paths = get_paths(config)
    games_per_season = config["simulation"]["games_per_season"]

    league_players = load_league_player_projections(config, scenario=scenario)
    team_win_pct: Optional[Dict[Any, float]] = None
    team_record: Optional[Dict[Any, str]] = None
    coy: Optional[pd.DataFrame] = None

    if league_players is not None:
        scope = "league"
        player_df = league_players
        career = _read_csv_if_exists(paths["processed"] / "league_player_career_stats.csv")
        regular = load_league_regular_season_summary(config, scenario=scenario)

        if regular is not None:
            wins_by_team = (regular.set_index("team_abbreviation")["wins_mean"] / games_per_season).to_dict()
            # keyed por player_id (no por índice posicional), que es lo
            # que esperan las funciones de awards_projection.
            team_win_pct = dict(zip(player_df["player_id"], player_df["team_abbreviation"].map(wins_by_team)))
            # Récord "V-D" del equipo -- a petición del usuario, para
            # comparar candidatos en las tablas de premios individuales.
            # Redondeado: wins_mean es una media continua de la
            # simulación, no un resultado real exacto.
            record_by_team = {
                abbrev: _win_loss_record(wins, games_per_season)
                for abbrev, wins in regular.set_index("team_abbreviation")["wins_mean"].items()
            }
            team_record = dict(zip(player_df["player_id"], player_df["team_abbreviation"].map(record_by_team)))

        prior = _read_csv_if_exists(paths["processed"] / "prior_season_standings.csv")
        if regular is not None and prior is not None:
            from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID

            id_to_abbrev = {v: k for k, v in ABBREVIATION_TO_TEAM_ID.items()}
            prior_wins_by_team = (
                prior.assign(team_abbreviation=prior["TeamID"].map(id_to_abbrev))
                .dropna(subset=["team_abbreviation"])
                .set_index("team_abbreviation")["WINS"]
                .to_dict()
            )
            coy = ap.compute_coy_candidates(regular, prior_wins_by_team, top_n=top_n)
    else:
        roster_overview = load_roster_overview(config)
        if roster_overview is None:
            return None
        scope = "own"
        player_df = roster_overview
        career = _read_csv_if_exists(paths["processed"] / "roster_career_stats.csv")

        # Un solo equipo -- el "récord" es el mismo para todo el roster,
        # tomado de la simulación Monte Carlo propia (no hay un rival
        # real por jugador con el que variar, a diferencia del scope liga).
        simulation_results = load_simulation_results(config)
        if simulation_results is not None and "player_id" in player_df.columns:
            record = _win_loss_record(simulation_results["wins"].mean(), games_per_season)
            team_record = {pid: record for pid in player_df["player_id"]}

    if career is None:
        career = pd.DataFrame()

    rookie_ids = ap.compute_rookie_player_ids(career)
    bench_ids = ap.compute_bench_player_ids(career)
    all_star = ap.compute_all_star_selections(player_df, games_per_season, team_win_pct=team_win_pct, team_record=team_record)
    all_star_quota = ap.check_all_star_nationality_quota(all_star)
    # Selección FINAL, con los añadidos del comisionado si la cuota
    # natural no llega al mínimo (a petición explícita del usuario) --
    # ver el warning "commissioner_pick" que consume webapp/routers/awards.py.
    all_star_final = ap.add_commissioner_picks_for_nationality_quota(
        player_df, all_star, all_star_quota, team_win_pct=team_win_pct
    )

    mip = ap.compute_mip_candidates(career, top_n=top_n)
    # MIP se calcula sobre career_stats_df (temporadas REALES, ver
    # docstring de compute_mip_candidates), no sobre player_df -- no
    # tiene de dónde sacar las stats proyectadas ni el récord de equipo,
    # así que se enriquece aquí, uniendo por player_id. Mismas columnas
    # que MVP/6MOY/ROY para que sea comparable de un vistazo.
    if not mip.empty and "player_id" in player_df.columns:
        extra_cols = [c for c in ap.OFFENSIVE_COMPARISON_STATS + ["team_abbreviation"] if c in player_df.columns]
        mip = mip.merge(player_df[["player_id"] + extra_cols], on="player_id", how="left")
        if team_record:
            mip["team_record"] = mip["player_id"].map(team_record)
        # PPG/RPG/APG/etc de la ÚLTIMA temporada REAL (distinto de los de
        # arriba, que son la proyección) -- para que el popup de MIP en
        # webapp/ compare "de dónde viene" vs "hacia dónde va", a
        # petición del usuario.
        mip = mip.merge(ap.compute_latest_real_season_stats(career), on="player_id", how="left")

    return {
        "scope": scope,
        "mvp": ap.compute_mvp_candidates(player_df, games_per_season, team_win_pct=team_win_pct, team_record=team_record, top_n=top_n),
        "dpoy": ap.compute_dpoy_candidates(player_df, games_per_season, team_record=team_record, top_n=top_n),
        "sixth_man": ap.compute_sixth_man_candidates(player_df, bench_ids, games_per_season, team_record=team_record, top_n=top_n),
        "roy": ap.compute_roy_candidates(player_df, rookie_ids, games_per_season, team_record=team_record, top_n=top_n),
        "mip": mip,
        "coy": coy,
        # Selección NATURAL (24, sin añadidos) -- la cuota se informa
        # siempre sobre esta, no sobre la final (que ya la cumple por
        # construcción una vez se aplican los añadidos).
        "all_star": all_star,
        "all_star_nationality_quota": all_star_quota,
        "all_star_final": all_star_final,
        "all_nba": ap.compute_all_nba_teams(player_df, games_per_season, team_record=team_record),
        "all_defensive": ap.compute_all_defensive_teams(player_df, games_per_season, team_record=team_record),
    }


def run_single_bracket_simulation(
    config: Dict[str, Any], random_seed: Optional[int] = None, scenario: str = "with_injuries"
) -> Dict[str, Any]:
    """
    ÚNICA excepción a "esta capa solo lee CSV": simula un bracket de
    playoffs concreto (league_simulation.simulate_single_bracket) para el
    botón "Simular un bracket" del dashboard. No llama a la API ni corre
    la temporada regular -- reutiliza league_regular_season_summary.csv
    (ya calculado) para el seeding y proyecta los 30 equipos desde los
    CSV de league_rosters/league_player_career_stats ya cacheados, así
    que es rápido (segundos, no minutos). `scenario="no_injuries"` juega
    el bracket con el roster de cada equipo sano.
    """
    from league_simulation import simulate_single_bracket

    return simulate_single_bracket(config, random_seed=random_seed, scenario=scenario)


def run_single_league_season_simulation(
    config: Dict[str, Any], random_seed: Optional[int] = None, scenario: str = "with_injuries"
) -> Dict[str, pd.DataFrame]:
    """
    Excepción a "esta capa solo lee CSV": simula UNA temporada regular
    concreta -- calendario, resultado de cada partido y boxscore
    ilustrativo por jugador (league_simulation.run_single_league_season_simulation)
    -- para el botón "Simular calendario de la temporada". Guarda los CSV
    en disco de paso (mismo patrón que el bracket), así que llamadas
    posteriores a load_league_single_season_game_log() los encuentran sin
    tener que volver a simular.
    """
    from league_simulation import run_single_league_season_simulation as _run

    return _run(config, scenario=scenario, random_seed=random_seed)


def run_single_season_player_log_simulation(
    config: Dict[str, Any], random_seed: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Simula UNA temporada concreta del roster propio (no la distribución
    agregada de simulation_results.csv) y devuelve, por jugador, partidos
    jugados/perdidos y el detalle de cada ausencia con su categoría
    ilustrativa (simulation.simulate_single_season_player_log -- ver ahí
    el porqué la categoría NO es un diagnóstico real). None si falta
    aging_curve_projection.csv o injury_risk.csv.

    Usa un `random_seed` distinto por defecto (basado en el reloj) al de
    `config["simulation"]["random_seed"]` -- ese seed es el que reproduce
    exactamente simulation_results.csv; usarlo aquí daría SIEMPRE la misma
    temporada individual en vez de una tirada nueva cada vez que se pulsa
    el botón del dashboard.
    """
    import numpy as np

    from simulation import DEFAULT_MONTE_CARLO_CONFIG, simulate_single_season_player_log

    paths = get_paths(config)
    aging = _read_csv_if_exists(paths["processed"] / "aging_curve_projection.csv")
    injury = _read_csv_if_exists(paths["processed"] / "injury_risk.csv")
    if aging is None or injury is None:
        return None

    player_ids = [p["player_id"] for p in config["roster"] if p.get("player_id")]
    player_names = dict(zip(aging["player_id"], aging["player_name"]))
    risk_by_player = dict(zip(injury["player_id"], injury["risk_score"]))
    risk_scores = np.array([risk_by_player.get(pid, 0.0) for pid in player_ids])

    mc_config = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}
    seed = random_seed if random_seed is not None else np.random.default_rng().integers(0, 2**31 - 1)

    return simulate_single_season_player_log(
        player_ids,
        player_names,
        risk_scores,
        config["simulation"]["games_per_season"],
        mc_config,
        random_seed=int(seed),
    )


def compute_win_distribution_summary(simulation_results: pd.DataFrame) -> Dict[str, float]:
    """Resumen legible de la distribución de victorias simuladas -- usado
    tanto por el dashboard como por cualquier test que valide el resumen."""
    wins = simulation_results["wins"]
    return {
        "mean": float(wins.mean()),
        "p10": float(wins.quantile(0.1)),
        "p50": float(wins.quantile(0.5)),
        "p90": float(wins.quantile(0.9)),
        "min": float(wins.min()),
        "max": float(wins.max()),
    }
