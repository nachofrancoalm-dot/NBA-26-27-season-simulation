"""
routers/explainer.py

Pestaña "Explicador (IA)". Reutiliza src/llm_explainer.py sin cambios.
explain_question() es de un solo turno (no recibe historial de
conversación) -- el historial vive solo en el frontend, igual que
st.session_state["explainer_history"] en dashboard/app.py, así que este
router no guarda sesión de ningún tipo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# A diferencia de los demás routers, este no importa dashboard.data_loader
# (que de rebote inserta src/ en sys.path) -- se hace explícito aquí para
# no depender del orden en que main.py importa los routers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from config_loader import load_config  # noqa: E402
from llm_explainer import build_context_snapshot, explain_question  # noqa: E402

router = APIRouter(prefix="/explainer")


class AskRequest(BaseModel):
    question: str


def _require_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "No se encontró la variable de entorno GROQ_API_KEY. Copia .env.example a .env "
                "en la raíz del proyecto y rellena tu API key de https://console.groq.com/keys."
            ),
        )
    return api_key


@router.get("/context")
def get_context():
    config = load_config()
    return {"snapshot": build_context_snapshot(config)}


@router.post("/ask")
def post_ask(body: AskRequest):
    api_key = _require_api_key()
    config = load_config()
    try:
        answer = explain_question(body.question, config, api_key=api_key)
    except Exception as exc:  # noqa: BLE001 -- mismo criterio que dashboard/app.py: mostrar cualquier error de API
        answer = f"Error al consultar el modelo: {exc}"
    return {"answer": answer}
