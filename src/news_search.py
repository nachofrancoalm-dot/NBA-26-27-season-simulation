"""
news_search.py

Fase 2 del RAG del Explicador (ver llm_explainer.py, fase 1: TF-IDF sobre
texto pegado a mano). Busca noticias recientes en internet, solo bajo
demanda explícita del usuario -- nunca automática, mismo patrón de
opt-in que GROQ_API_KEY. Es la única llamada de red en vivo del proyecto
fuera de nba_api.

El texto devuelto se pasa como `news_text` a `explain_question()` tras
una limpieza best-effort (`_clean_content()`) y sigue el mismo camino
que el texto pegado a mano: retrieve_relevant_news_snippets() /
build_news_section(), misma etiqueta NEWS_SECTION_LABEL, mismo caveat de
"no verificado" en el prompt.

Usa la API REST de Tavily (https://tavily.com) vía `requests` en vez de
añadir tavily-python como dependencia nueva.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
REQUEST_TIMEOUT_SECONDS = 15

# Tope de caracteres por resultado tras limpiar -- evita que un resultado
# tipo tabla (roster, salarios) domine el texto y diluya el TF-IDF de fase 1.
MAX_CONTENT_CHARS = 500

# Líneas de banners publicitarios/UI que Tavily extrae junto al contenido real.
_BOILERPLATE_LINES = {"advertisement", "about our ads"}

# Separadores de tabla markdown mal extraídos (" --- | --- ", etc.), ruido puro.
_TABLE_SEPARATOR_RE = re.compile(r"^[\s|:\-]+$")


def _clean_content(text: str) -> str:
    """
    Limpieza best-effort del contenido crudo de Tavily: quita banners
    publicitarios, separadores de tabla y espacios excesivos. No es un
    parser de HTML/tablas -- contenido tipo tabla seguirá leyéndose como
    datos sueltos, no como prosa.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower() in _BOILERPLATE_LINES or _TABLE_SEPARATOR_RE.match(line):
            continue
        lines.append(line.replace("|", " "))

    cleaned = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(cleaned) > MAX_CONTENT_CHARS:
        cleaned = cleaned[:MAX_CONTENT_CHARS].rsplit(" ", 1)[0] + "…"
    return cleaned


def search_recent_news(query: str, api_key: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """
    Busca `query` en Tavily y devuelve los resultados como texto plano,
    un párrafo por resultado (título + contenido), en el mismo formato
    que build_news_section() espera de un pegado manual. Lanza
    RuntimeError si no hay API key; los errores de red/HTTP se propagan
    tal cual.
    """
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "No se encontró TAVILY_API_KEY. Definela en un archivo .env en la raíz del "
            "proyecto (ver .env.example) para activar la búsqueda de noticias recientes -- "
            "sin ella, puedes seguir pegando texto a mano en el cuadro de 'noticias recientes' "
            "del Explicador."
        )

    response = requests.post(
        TAVILY_SEARCH_URL,
        json={"api_key": key, "query": query, "max_results": max_results, "search_depth": "basic"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    paragraphs = []
    for result in data.get("results", []):
        title = (result.get("title") or "").strip()
        content = _clean_content(result.get("content") or "")
        if not content:
            continue
        paragraphs.append(f"{title}\n{content}" if title else content)

    return "\n\n".join(paragraphs)
