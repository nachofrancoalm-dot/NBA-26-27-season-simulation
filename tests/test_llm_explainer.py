"""
Tests de src/llm_explainer.py. build_context_snapshot() se prueba contra
CSV sintéticos (sin red); explain_question() se prueba mockeando el
cliente de Anthropic -- nunca llamamos a la API real en los tests.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_explainer import (  # noqa: E402
    NEWS_SECTION_LABEL,
    build_context_snapshot,
    build_news_section,
    explain_question,
    retrieve_relevant_news_snippets,
)


@pytest.fixture
def config(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    return {
        "team": {"name": "Test Team", "season": "2025-26"},
        "roster": [{"player_id": 1, "name": "Player One"}],
        "simulation": {"games_per_season": 82},
        "paths": {"raw_data_dir": str(tmp_path / "raw"), "processed_data_dir": str(processed_dir)},
    }


def test_build_context_snapshot_marks_missing_sections_as_not_available(config):
    snapshot = build_context_snapshot(config)

    assert "Test Team" in snapshot
    assert "Proyeccion de roster: NO DISPONIBLE" in snapshot
    assert "Riesgo de lesion: NO DISPONIBLE" in snapshot
    assert "Desgaste acumulado: NO DISPONIBLE" in snapshot
    assert "Simulacion Monte Carlo de temporada propia: NO DISPONIBLE" in snapshot
    assert "Backtesting: NO DISPONIBLE" in snapshot
    assert "Riesgo de lesion por equipo (liga completa): NO DISPONIBLE" in snapshot
    assert "Simulacion de los 30 equipos de la liga: NO DISPONIBLE" in snapshot


def test_build_context_snapshot_includes_real_numbers_when_csvs_exist(config):
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {
                "player_id": 1, "player_name": "Player One", "game_score_per36": 20.0,
                "PTS_projected": 2050, "REB_projected": 410, "AST_projected": 656,
                "projected_total_minutes": 2788.0,  # 34.0 min/partido x 82 partidos
            }
        ]
    ).to_csv(processed / "aging_curve_projection.csv", index=False)
    pd.DataFrame([{"player_id": 1, "risk_score": 0.42}]).to_csv(processed / "injury_risk.csv", index=False)
    pd.DataFrame([{"wins": 55, "net_rating_estimate_mean": 6.2}, {"wins": 50, "net_rating_estimate_mean": 4.1}]).to_csv(
        processed / "simulation_results.csv", index=False
    )

    snapshot = build_context_snapshot(config)

    assert "Player One" in snapshot
    assert "25.0 PPG" in snapshot  # 2050 / 82
    assert "minutos proyectados/partido = 34.0" in snapshot  # 2788 / 82
    assert "risk_score = 0.42" in snapshot
    assert "Victorias medias: 52.5" in snapshot
    assert "NO DISPONIBLE" not in snapshot.split("## Simulacion Monte Carlo")[1].split("##")[0]


def test_build_context_snapshot_includes_league_wide_injury_risk_by_team(config):
    """
    Regresión: el usuario preguntó "¿qué equipo ha tenido más lesiones?"
    y el LLM no pudo responder porque solo veía el riesgo de lesión del
    roster propio, no el de los 30 equipos reales de la liga.
    """
    processed = Path(config["paths"]["processed_data_dir"])
    pd.DataFrame(
        [
            {"team_abbreviation": "AAA", "player_id": 1, "player_name": "Riesgo Alto", "risk_score": 0.80},
            {"team_abbreviation": "AAA", "player_id": 2, "player_name": "Riesgo Bajo", "risk_score": 0.20},
            {"team_abbreviation": "BBB", "player_id": 3, "player_name": "Otro Jugador", "risk_score": 0.10},
        ]
    ).to_csv(processed / "league_player_projections.csv", index=False)

    snapshot = build_context_snapshot(config)

    assert "Riesgo de lesion medio por equipo" in snapshot
    assert "Riesgo de lesion por equipo (liga completa): NO DISPONIBLE" not in snapshot
    # AAA (media 0.50) debe listarse antes que BBB (0.10) -- ordenado
    # de mayor a menor riesgo, que es justo lo que hace falta para
    # responder "qué equipo tiene más riesgo de lesión".
    section = snapshot.split("## Riesgo de lesion medio por equipo")[1].split("##")[0]
    assert section.find("AAA") < section.find("BBB")
    assert "AAA: risk_score medio = 0.50" in snapshot


def test_explain_question_raises_without_api_key(config, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        explain_question("¿Cuántas victorias medias tiene el equipo?", config, api_key=None)


def test_explain_question_grounds_the_prompt_with_the_snapshot(config, monkeypatch):
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Respuesta grounded."),
                    )
                ]
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.chat = _FakeChat()

    monkeypatch.setattr("groq.Groq", _FakeClient)

    answer = explain_question(
        "¿Cuántas victorias medias tiene el equipo?",
        config,
        api_key="test-key",
        context_snapshot="## Simulacion Monte Carlo de temporada propia\n- Victorias medias: 52.5",
    )

    assert answer == "Respuesta grounded."
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "llama-3.3-70b-versatile"
    # El snapshot va en el mensaje de system, no en el de usuario -- así el
    # modelo queda "grounded" en los datos reales ya calculados.
    system_message = captured["messages"][0]
    assert system_message["role"] == "system"
    assert "Victorias medias: 52.5" in system_message["content"]
    assert captured["messages"][1] == {
        "role": "user",
        "content": "¿Cuántas victorias medias tiene el equipo?",
    }


def test_retrieve_relevant_news_snippets_returns_empty_without_news_text():
    assert retrieve_relevant_news_snippets("", "¿Que le paso a Embiid?") == []


def test_retrieve_relevant_news_snippets_returns_empty_when_nothing_overlaps():
    # Cero solapamiento lexico (ni siquiera stopwords compartidas) --
    # no hay que forzar un fragmento irrelevante en el contexto.
    snippets = retrieve_relevant_news_snippets("Trueno relampago tormenta.", "Presupuesto anual empresa")
    assert snippets == []


def test_retrieve_relevant_news_snippets_ranks_the_relevant_paragraph_first():
    news_text = (
        "Embiid se lesiono la rodilla en el partido de ayer y es duda para esta noche.\n\n"
        "El presupuesto del equipo crecio este año fiscal segun el informe contable."
    )
    snippets = retrieve_relevant_news_snippets(news_text, "¿Que le paso a Embiid en la rodilla?")
    assert snippets
    assert "Embiid" in snippets[0]


def test_retrieve_relevant_news_snippets_falls_back_to_line_splitting():
    # Un unico bloque sin parrafos (lista de titulares, una noticia por
    # linea) -- debe trocearse por linea, no tratarse como un solo
    # fragmento gigante donde una noticia irrelevante "contamina" a la
    # relevante.
    news_text = "Embiid fuera por molestias en la rodilla\nCurry con esguince de tobillo leve"
    snippets = retrieve_relevant_news_snippets(news_text, "¿Que lesion tiene Curry?")
    assert snippets
    assert "Curry" in snippets[0]


def test_build_news_section_empty_without_news_text():
    assert build_news_section("", "¿Que le paso a Embiid?") == ""


def test_build_news_section_empty_when_nothing_relevant():
    assert build_news_section("Trueno relampago tormenta.", "Presupuesto anual empresa") == ""


def test_build_news_section_labels_the_section_explicitly():
    section = build_news_section(
        "Embiid se lesiono la rodilla en el partido de ayer.", "¿Que le paso a Embiid en la rodilla?"
    )
    assert section.startswith(NEWS_SECTION_LABEL)
    assert "Embiid" in section


def test_explain_question_appends_news_section_when_relevant(config, monkeypatch):
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))]
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.chat = _FakeChat()

    monkeypatch.setattr("groq.Groq", _FakeClient)

    explain_question(
        "¿Que le paso a Embiid en la rodilla?",
        config,
        api_key="test-key",
        context_snapshot="## Simulacion Monte Carlo de temporada propia\n- Victorias medias: 52.5",
        news_text="Embiid se lesiono la rodilla en el partido de ayer.",
    )

    system_message = captured["messages"][0]["content"]
    assert NEWS_SECTION_LABEL in system_message
    assert "Embiid" in system_message


def test_explain_question_omits_news_section_when_nothing_relevant(config, monkeypatch):
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))]
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.chat = _FakeChat()

    monkeypatch.setattr("groq.Groq", _FakeClient)

    explain_question(
        "Presupuesto anual empresa",
        config,
        api_key="test-key",
        context_snapshot="## Simulacion Monte Carlo de temporada propia\n- Victorias medias: 52.5",
        news_text="Trueno relampago tormenta.",
    )

    # El SYSTEM_PROMPT menciona la etiqueta una vez al explicar el
    # criterio -- lo que no debe pasar es que APARIEZCA COMO SECCION
    # (con las noticias pegadas debajo) cuando nada es relevante.
    system_message = captured["messages"][0]["content"]
    assert system_message.count(NEWS_SECTION_LABEL) == 1
    assert "Trueno" not in system_message


def test_explain_question_handles_content_filter_finish_reason(config, monkeypatch):
    class _FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="content_filter", message=SimpleNamespace(content=None))]
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key=None):
            self.chat = _FakeChat()

    monkeypatch.setattr("groq.Groq", _FakeClient)

    answer = explain_question("pregunta", config, api_key="test-key", context_snapshot="snapshot")

    assert "no pudo generar" in answer.lower()
