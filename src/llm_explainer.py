"""
llm_explainer.py

"Explicador de resultados en lenguaje natural": responde preguntas del
usuario sobre los datos YA CALCULADOS por el pipeline (roster, riesgo de
lesion/desgaste, simulacion Monte Carlo, backtesting, standings de liga),
usando la API de Groq (modelos open-weight servidos con inferencia muy
rapida) para narrar una explicacion grounded -- el LLM nunca inventa
numeros, solo redacta sobre un snapshot de texto con los datos reales ya
generados por los demas modulos (injury_model.py, fatigue_accumulation.py,
aging_curve.py, simulation.py, backtesting.py, league_simulation.py).

No ejecuta ninguna simulacion ni llama a la API de la NBA -- si el CSV
correspondiente no existe todavia, esa seccion se omite del contexto (y
se le dice explicitamente al modelo que no estan disponibles, para que
no finja tenerlos).

Requiere GROQ_API_KEY en el entorno (ver .env.example). Se carga
automaticamente desde un archivo .env en la raiz del proyecto via
python-dotenv si existe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

from config_loader import PROJECT_ROOT, get_paths

load_dotenv(PROJECT_ROOT / ".env")

MODEL_ID = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Eres un asistente que explica los resultados de un simulador Monte Carlo "
    "de baloncesto NBA a su usuario. Se te proporciona un snapshot de texto con "
    "los datos ya calculados por el pipeline (proyecciones de jugadores, riesgo "
    "de lesion y desgaste, resultados de simulacion, backtesting, standings de "
    "liga). Responde SOLO en base a esos datos -- nunca inventes cifras, "
    "jugadores, ni resultados que no aparezcan en el contexto. Si una seccion "
    "aparece marcada como NO DISPONIBLE y la pregunta depende de ella, dilo "
    "explicitamente y sugiere que parte del pipeline habria que correr para "
    "tenerla, en vez de adivinar. Responde en espanol, de forma clara y "
    "concisa, como lo haria un analista explicando un informe a alguien que no "
    "es experto en estadistica."
)


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def build_context_snapshot(config: Dict[str, Any]) -> str:
    """
    Construye el bloque de texto con los datos ya calculados que se le pasa
    al LLM como contexto grounded. Cada seccion se omite (con una nota
    explicita de "NO DISPONIBLE") si el CSV correspondiente todavia no
    existe -- así el modelo sabe que no debe fingir tener ese dato.
    """
    paths = get_paths(config)
    processed = paths["processed"]
    sections = []

    team_name = config["team"].get("name", "el equipo")
    season = config["team"].get("season", "")
    sections.append(f"# Equipo simulado: {team_name} ({season})")

    roster = _read_csv_if_exists(processed / "aging_curve_projection.csv")
    if roster is not None:
        games_per_season = config["simulation"]["games_per_season"]
        lines = ["## Proyeccion de roster (temporada completa proyectada)"]
        for _, row in roster.sort_values("game_score_per36", ascending=False).iterrows():
            ppg = row.get("PTS_projected", 0) / games_per_season
            rpg = row.get("REB_projected", 0) / games_per_season
            apg = row.get("AST_projected", 0) / games_per_season
            # minutes_projection NO es una columna de aging_curve_projection.csv
            # (esa vive en team_config.yaml, mergeada aparte por el dashboard) --
            # se deriva de projected_total_minutes, que sí está en este CSV.
            mpg = row.get("projected_total_minutes", 0) / games_per_season
            lines.append(
                f"- {row['player_name']}: {ppg:.1f} PPG, {rpg:.1f} RPG, {apg:.1f} APG, "
                f"Game Score/36 = {row['game_score_per36']:.1f}, "
                f"minutos proyectados/partido = {mpg:.1f}"
            )
        sections.append("\n".join(lines))
    else:
        sections.append("## Proyeccion de roster: NO DISPONIBLE (falta aging_curve_projection.csv)")

    injury = _read_csv_if_exists(processed / "injury_risk.csv")
    if injury is not None and roster is not None:
        merged = roster.merge(injury[["player_id", "risk_score"]], on="player_id", how="left")
        lines = ["## Riesgo de lesion por jugador (0 a 1, mayor = mas riesgo)"]
        for _, row in merged.sort_values("risk_score", ascending=False).iterrows():
            if pd.notna(row.get("risk_score")):
                lines.append(f"- {row['player_name']}: risk_score = {row['risk_score']:.2f}")
        sections.append("\n".join(lines))
    else:
        sections.append("## Riesgo de lesion: NO DISPONIBLE (falta injury_risk.csv)")

    fatigue = _read_csv_if_exists(processed / "fatigue_risk.csv")
    if fatigue is not None and roster is not None:
        merged = roster.merge(fatigue[["player_id", "fatigue_score"]], on="player_id", how="left")
        lines = ["## Desgaste acumulado por jugador (0 a 1, mayor = mas desgaste)"]
        for _, row in merged.sort_values("fatigue_score", ascending=False).iterrows():
            if pd.notna(row.get("fatigue_score")):
                lines.append(f"- {row['player_name']}: fatigue_score = {row['fatigue_score']:.2f}")
        sections.append("\n".join(lines))
    else:
        sections.append("## Desgaste acumulado: NO DISPONIBLE (falta fatigue_risk.csv)")

    sim = _read_csv_if_exists(processed / "simulation_results.csv")
    if sim is not None:
        wins = sim["wins"]
        sections.append(
            "## Simulacion Monte Carlo de temporada propia\n"
            f"- Temporadas simuladas: {len(sim)}\n"
            f"- Victorias medias: {wins.mean():.1f}\n"
            f"- Percentil 10 / 50 / 90 de victorias: {wins.quantile(0.1):.0f} / "
            f"{wins.quantile(0.5):.0f} / {wins.quantile(0.9):.0f}\n"
            f"- Net rating estimado medio: {sim['net_rating_estimate_mean'].mean():.2f}"
        )
    else:
        sections.append("## Simulacion Monte Carlo de temporada propia: NO DISPONIBLE (falta simulation_results.csv)")

    backtest = _read_csv_if_exists(processed / "backtest_summary.csv")
    if backtest is not None:
        lines = ["## Backtesting contra casos historicos comparables"]
        for _, row in backtest.iterrows():
            label = row.get("team_label", row.get("team_abbreviation", "caso historico"))
            lines.append(
                f"- {label}: victorias reales = {row['actual_wins']}, "
                f"media simulada retrospectiva = {row['simulated_wins_mean']:.1f}, "
                f"percentil real dentro de la simulacion = {row['actual_percentile']:.0f}"
            )
        sections.append("\n".join(lines))
    else:
        sections.append("## Backtesting: NO DISPONIBLE (falta backtest_summary.csv)")

    league_projections = _read_csv_if_exists(processed / "league_player_projections.csv")
    if league_projections is not None and "risk_score" in league_projections.columns:
        # Riesgo de lesion AGREGADO por equipo de los 30 equipos reales de
        # la liga -- distinto de la seccion "Riesgo de lesion por jugador"
        # de arriba, que es solo del roster HIPOTETICO propio. Sin esta
        # seccion el LLM no podia responder "que equipo tiene mas riesgo
        # de lesion" porque solo veia el propio equipo en el contexto.
        by_team = (
            league_projections.groupby("team_abbreviation")["risk_score"]
            .mean()
            .sort_values(ascending=False)
        )
        lines = ["## Riesgo de lesion medio por equipo (30 equipos reales de la liga, 0 a 1)"]
        for team_abbreviation, mean_risk in by_team.items():
            lines.append(f"- {team_abbreviation}: risk_score medio = {mean_risk:.2f}")
        sections.append("\n".join(lines))
    else:
        sections.append(
            "## Riesgo de lesion por equipo (liga completa): NO DISPONIBLE "
            "(requiere `python src/data_pipeline.py --league` y luego "
            "`build_league_simulation_dataset`, que genera league_player_projections.csv)"
        )

    regular = _read_csv_if_exists(processed / "league_regular_season_summary.csv")
    playoff = _read_csv_if_exists(processed / "league_playoff_summary.csv")
    if regular is not None and playoff is not None:
        merged = regular.merge(playoff, on="team_abbreviation", suffixes=("", "_po"))
        merged = merged.sort_values("wins_mean", ascending=False)
        lines = ["## Standings simulados de los 30 equipos de la NBA (temporada regular + playoffs)"]
        for _, row in merged.iterrows():
            lines.append(
                f"- {row['team_abbreviation']} ({row.get('conference', '?')}): "
                f"{row['wins_mean']:.1f} victorias medias, "
                f"playoffs {row['playoff_pct']:.1f}%, "
                f"campeonato {row['championship_pct']:.1f}%"
            )
        sections.append("\n".join(lines))
    else:
        sections.append(
            "## Simulacion de los 30 equipos de la liga: NO DISPONIBLE "
            "(requiere `python src/data_pipeline.py --league` y luego "
            "`build_league_simulation_dataset`)"
        )

    return "\n\n".join(sections)


def explain_question(
    question: str,
    config: Dict[str, Any],
    api_key: Optional[str] = None,
    context_snapshot: Optional[str] = None,
) -> str:
    """
    Responde `question` usando Groq, grounded en context_snapshot (si no
    se pasa, se construye uno con build_context_snapshot(config)). Lanza
    RuntimeError si no hay API key disponible -- la capa de dashboard lo
    captura para mostrar un aviso amigable en vez de un traceback.
    """
    import groq

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "No se encontro GROQ_API_KEY. Definela en un archivo .env en la "
            "raiz del proyecto (ver .env.example) o pasa api_key "
            "explicitamente para activar el explicador de resultados con IA."
        )

    snapshot = context_snapshot if context_snapshot is not None else build_context_snapshot(config)

    client = groq.Groq(api_key=key)
    response = client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + snapshot},
            {"role": "user", "content": question},
        ],
    )

    choice = response.choices[0]
    if choice.finish_reason == "content_filter":
        return (
            "El modelo no pudo generar una respuesta para esta pregunta "
            "(bloqueo de contenido). Prueba a reformularla."
        )

    return choice.message.content or ""
