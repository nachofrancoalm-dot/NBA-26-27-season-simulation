"""
Tests de src/news_search.py (fase 2 del RAG de llm_explainer.py). Nunca
llamamos a la API real de Tavily -- se mockea requests.post.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from news_search import MAX_CONTENT_CHARS, _clean_content, search_recent_news  # noqa: E402


def test_search_recent_news_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        search_recent_news("lesiones Philadelphia 76ers", api_key=None)


def test_search_recent_news_formats_results_as_paragraphs(monkeypatch):
    captured = {}

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "results": [
                    {"title": "Embiid se pierde el partido", "content": "Molestias en la rodilla."},
                    {"title": "Los 76ers fichan a un ala-pivot", "content": "Acuerdo de un año."},
                ]
            },
        )

    monkeypatch.setattr("news_search.requests.post", _fake_post)

    result = search_recent_news("lesiones Philadelphia 76ers", api_key="test-key")

    # Titulo y contenido en lineas separadas dentro del mismo parrafo.
    assert "Embiid se pierde el partido\nMolestias en la rodilla." in result
    assert "Los 76ers fichan a un ala-pivot\nAcuerdo de un año." in result
    # Los DOS parrafos (uno por resultado) van separados por linea en
    # blanco -- mismo formato que espera _split_into_snippets() de
    # llm_explainer.py para trocear por parrafo.
    assert "\n\n" in result
    assert captured["json"]["query"] == "lesiones Philadelphia 76ers"
    assert captured["json"]["api_key"] == "test-key"


def test_search_recent_news_skips_results_without_content(monkeypatch):
    def _fake_post(url, json=None, timeout=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"results": [{"title": "Sin contenido", "content": ""}]},
        )

    monkeypatch.setattr("news_search.requests.post", _fake_post)

    result = search_recent_news("query cualquiera", api_key="test-key")
    assert result == ""


def test_search_recent_news_returns_empty_string_when_no_results(monkeypatch):
    def _fake_post(url, json=None, timeout=None):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"results": []})

    monkeypatch.setattr("news_search.requests.post", _fake_post)

    result = search_recent_news("query sin resultados", api_key="test-key")
    assert result == ""


def test_clean_content_removes_boilerplate_ad_lines():
    text = "Embiid se pierde el partido.\nAdvertisement\nAbout Our Ads\nMolestias en la rodilla."
    cleaned = _clean_content(text)
    assert "advertisement" not in cleaned.lower()
    assert "about our ads" not in cleaned.lower()
    assert "Embiid se pierde el partido." in cleaned
    assert "Molestias en la rodilla." in cleaned


def test_clean_content_removes_markdown_table_separator_lines():
    text = "POSITION | STARTER | RESERVE\n --- | --- | --- \n|  |  |  |\nPG | Tyrese Maxey | Anfernee Simons"
    cleaned = _clean_content(text)
    assert "---" not in cleaned
    assert "PG" in cleaned and "Tyrese Maxey" in cleaned


def test_clean_content_strips_stray_pipes():
    cleaned = _clean_content("| F | 25 | Dominick Barlow | 6 ft 9 in |")
    assert "|" not in cleaned
    assert "Dominick Barlow" in cleaned


def test_clean_content_truncates_long_content_at_word_boundary():
    text = "palabra " * 200  # muy por encima de MAX_CONTENT_CHARS
    cleaned = _clean_content(text)
    assert len(cleaned) <= MAX_CONTENT_CHARS + 1  # +1 por el "…" final
    assert cleaned.endswith("…")
    assert not cleaned[:-1].endswith(" ")  # corta en un limite de palabra, no a mitad


def test_clean_content_collapses_whitespace():
    cleaned = _clean_content("Embiid   se   pierde\n\n\nel partido")
    assert cleaned == "Embiid se pierde el partido"


def test_search_recent_news_cleans_real_world_noisy_result(monkeypatch):
    # Ejemplo real de tabla de roster mal extraida por Tavily, con
    # banners publicitarios repetidos.
    noisy_content = (
        "|  |  |  |  |\n"
        " ---  --- |\n"
        "| POSITION | STARTER | RESERVE | RESERVE |\n"
        "| PG | Tyrese Maxey | Anfernee Simons | Labaron Philon |\n"
        "About Our Ads\n"
        "Advertisement\n"
        "Advertisement\n"
        "LeBron James has finally made his decision."
    )

    def _fake_post(url, json=None, timeout=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"results": [{"title": "76ers Roster & Starting Lineup", "content": noisy_content}]},
        )

    monkeypatch.setattr("news_search.requests.post", _fake_post)

    result = search_recent_news("76ers new roster", api_key="test-key")

    assert "Advertisement" not in result
    assert "About Our Ads" not in result
    assert "---" not in result
    assert "|" not in result
    assert "Tyrese Maxey" in result
    assert "LeBron James has finally made his decision." in result


def test_search_recent_news_propagates_http_errors(monkeypatch):
    import requests

    def _fake_post(url, json=None, timeout=None):
        def _raise():
            raise requests.exceptions.HTTPError("500 Server Error")
        return SimpleNamespace(raise_for_status=_raise, json=lambda: {})

    monkeypatch.setattr("news_search.requests.post", _fake_post)

    with pytest.raises(requests.exceptions.HTTPError):
        search_recent_news("query cualquiera", api_key="test-key")
