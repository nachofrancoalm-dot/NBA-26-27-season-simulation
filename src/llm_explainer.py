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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from config_loader import PROJECT_ROOT, get_paths

load_dotenv(PROJECT_ROOT / ".env")

MODEL_ID = "llama-3.3-70b-versatile"

# Etiqueta obligatoria de la seccion de noticias pegadas por el usuario --
# nunca se mezcla sin marcar con las secciones de datos validados del
# pipeline (ver build_news_section). Se referencia tambien desde el
# system prompt para reforzar el criterio en la propia respuesta del modelo.
NEWS_SECTION_LABEL = (
    "## Noticias recientes (pegadas por el usuario, NO verificadas por el "
    "pipeline ni por el modelo -- pueden estar desactualizadas, incompletas "
    "o ser inexactas)"
)

SYSTEM_PROMPT = (
    "Eres un asistente que explica los resultados de un simulador Monte Carlo "
    "de baloncesto NBA a su usuario. Se te proporciona un snapshot de texto con "
    "los datos ya calculados por el pipeline (proyecciones de jugadores, riesgo "
    "de lesion y desgaste, resultados de simulacion, backtesting, standings de "
    "liga). Responde SOLO en base a esos datos -- nunca inventes cifras, "
    "jugadores, ni resultados que no aparezcan en el contexto. Si una seccion "
    "aparece marcada como NO DISPONIBLE y la pregunta depende de ella, dilo "
    "explicitamente y sugiere que parte del pipeline habria que correr para "
    "tenerla, en vez de adivinar. Si el contexto incluye la seccion "
    f"'{NEWS_SECTION_LABEL}', puedes usarla para dar contexto cualitativo "
    "reciente, pero deja siempre claro en tu respuesta que es informacion "
    "pegada por el usuario, no verificada ni calculada por el pipeline, y "
    "nunca la combines con las demas cifras como si tuviera el mismo nivel "
    "de fiabilidad. Responde en espanol, de forma clara y "
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


def _split_into_snippets(news_text: str) -> List[str]:
    """
    Trocea el texto pegado por el usuario en fragmentos recuperables.
    Primero intenta por parrafo (linea en blanco de separacion, el caso
    tipico al pegar varios articulos o titulares); si el texto es un
    unico bloque sin parrafos, cae a una linea por fragmento.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", news_text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [line.strip() for line in news_text.splitlines() if line.strip()]


def retrieve_relevant_news_snippets(news_text: str, question: str, top_k: int = 3) -> List[str]:
    """
    RAG minimo: TF-IDF + similitud coseno entre `question` y cada
    fragmento de `news_text`, sin embeddings ni dependencias nuevas
    (scikit-learn ya es dependencia principal del proyecto). Con el
    volumen de texto que un usuario pega a mano (unos pocos articulos,
    no un corpus masivo) es suficiente -- ver discusion de diseno.

    Solo devuelve fragmentos con similitud > 0 (algo de solapamiento
    lexico con la pregunta) -- si el texto pegado no tiene relacion con
    lo que se pregunta, no se inyecta nada en el contexto.
    """
    if not news_text.strip() or not question.strip():
        return []

    snippets = _split_into_snippets(news_text)
    if not snippets:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(snippets + [question])
    except ValueError:
        # Vocabulario vacio (p.ej. texto pegado son solo simbolos) -- no
        # hay nada recuperable, mejor omitir la seccion que fallar.
        return []

    similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(snippets, similarities), key=lambda pair: pair[1], reverse=True)
    return [snippet for snippet, score in ranked[:top_k] if score > 0]


def build_news_section(news_text: str, question: str) -> str:
    """
    Construye la seccion de contexto etiquetada con los fragmentos de
    `news_text` relevantes para `question`, o "" si no hay texto pegado
    o ningun fragmento resulta relevante (en ese caso se omite -- no se
    fuerza contexto irrelevante en el prompt).
    """
    snippets = retrieve_relevant_news_snippets(news_text, question)
    if not snippets:
        return ""
    lines = [NEWS_SECTION_LABEL] + [f"- {snippet}" for snippet in snippets]
    return "\n".join(lines)


def explain_question(
    question: str,
    config: Dict[str, Any],
    api_key: Optional[str] = None,
    context_snapshot: Optional[str] = None,
    news_text: Optional[str] = None,
) -> str:
    """
    Responde `question` usando Groq, grounded en context_snapshot (si no
    se pasa, se construye uno con build_context_snapshot(config)). Lanza
    RuntimeError si no hay API key disponible -- la capa de dashboard lo
    captura para mostrar un aviso amigable en vez de un traceback.

    `news_text` es opcional: texto pegado por el usuario (noticias,
    injury reports, rumores de traspaso) que el pipeline no puede ver
    porque no viene de ningun CSV calculado. Si se pasa, se recuperan
    (RAG por TF-IDF) los fragmentos relevantes a `question` y se anaden
    al contexto en una seccion claramente etiquetada como no verificada
    -- ver build_news_section y NEWS_SECTION_LABEL. Nunca se mezcla con
    context_snapshot sin esa etiqueta.
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
    if news_text:
        news_section = build_news_section(news_text, question)
        if news_section:
            snapshot = snapshot + "\n\n" + news_section

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
