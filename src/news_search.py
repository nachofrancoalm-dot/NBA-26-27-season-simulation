"""
news_search.py

Fase 2 del RAG del Explicador (ver llm_explainer.py, fase 1: TF-IDF sobre
texto pegado a mano). Busca noticias recientes en internet, bajo demanda
EXPLÍCITA del usuario -- nunca automática, nunca al cargar la página,
mismo patrón de opt-in que GROQ_API_KEY. Es la ÚNICA llamada de red en
vivo del proyecto fuera de nba_api -- justificada porque el pipeline
estadístico no puede ver noticias del día (lesiones de última hora,
cambios de entrenador) y este proyecto deliberadamente no las simula.

El texto que devuelve se pasa como `news_text` a `explain_question()`
tras una limpieza BEST-EFFORT (`_clean_content()` -- quita banners
publicitarios repetidos, separadores de tabla markdown mal extraídos y
pipes sueltos; los resultados de búsqueda reales suelen venir con este
ruido) -- la búsqueda solo cambia CÓMO se rellena ese texto (antes:
pegado a mano; ahora: además, un buscador), no qué se hace con él
después. Sigue pasando por retrieve_relevant_news_snippets() /
build_news_section() (fase 1): mismo TF-IDF, misma etiqueta
NEWS_SECTION_LABEL, mismo caveat de "no verificado" en el prompt --
buscar en internet no lo hace más fiable, solo más cómodo de rellenar ni
más limpio de leer.

Usa la API REST de Tavily (https://tavily.com, pensada para dar contexto
de búsqueda a LLMs) vía `requests` -- ya es dependencia del proyecto, no
se añade tavily-python: la llamada es demasiado simple para justificar
un paquete nuevo.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
REQUEST_TIMEOUT_SECONDS = 15

# Tope de caracteres por resultado tras limpiar -- Tavily a veces
# devuelve el texto extraído de una tabla completa (roster, salarios,
# lista de entrenadores) en vez de un resumen; sin tope, un solo
# resultado así domina el texto y diluye la relevancia del TF-IDF de
# fase 1 (ver llm_explainer.retrieve_relevant_news_snippets).
MAX_CONTENT_CHARS = 500

# Líneas que Tavily extrae literalmente de banners publicitarios/UI de
# la página de origen -- no son parte de la noticia, se descartan tal
# cual (comparación exacta, insensible a mayúsculas).
_BOILERPLATE_LINES = {"advertisement", "about our ads"}

# Líneas que son solo separadores de tabla markdown mal extraídos
# (" --- | --- ", "|  |  |  |", etc.) -- ruido puro, no aportan texto.
_TABLE_SEPARATOR_RE = re.compile(r"^[\s|:\-]+$")


def _clean_content(text: str) -> str:
    """
    Limpieza BEST-EFFORT del contenido crudo que devuelve Tavily -- no es
    un parser de HTML/tablas, solo quita el ruido más obvio y más
    frecuente (banners publicitarios repetidos, separadores de tabla,
    pipes sueltos, espacios en blanco excesivos) para que el texto sea
    legible en el cuadro de noticias y no diluya el TF-IDF de fase 1 con
    basura repetida. Contenido tipo tabla (rosters, salarios) seguirá
    leyéndose como una lista de datos suelta, no como prosa -- limpiarlo
    del todo exigiría parsear la tabla real, fuera de alcance aquí.
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
    un párrafo por resultado (título + contenido) separado por línea en
    blanco -- el MISMO formato que build_news_section() espera de un
    pegado manual (fase 1), para que _split_into_snippets() lo trocee
    por párrafo sin cambios.

    Lanza RuntimeError si no hay API key -- mismo criterio que
    explain_question() con GROQ_API_KEY. Los errores de red/HTTP se
    propagan tal cual (requests.exceptions.*) -- el llamante decide cómo
    mostrarlos (ver webapp/routers/explainer.py).
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
