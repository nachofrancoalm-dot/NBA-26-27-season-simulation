"""
main.py

Backend FastAPI de la interfaz web -- única interfaz del proyecto (el
dashboard de Streamlit que existió en paralelo, dashboard/app.py, se
retiró; ver el aviso en el propio README). Sirve la API JSON bajo /api/*
y los archivos estáticos del frontend en /. No reimplementa ninguna
lógica de datos: cada endpoint reutiliza dashboard/data_loader.py (ya
puro, ya testeado -- ver su docstring, es la capa de datos compartida,
ya no depende de Streamlit), src/awards_projection.py,
src/champion_profiles.py y src/llm_explainer.py.

Uso:
    uvicorn webapp.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.types import Receive, Scope, Send  # noqa: E402

from webapp.routers import awards, champions, explainer, league, players, sandbox, status, team  # noqa: E402


class RevalidatingStaticFiles(StaticFiles):
    """StaticFiles fuerza al navegador a revalidar (If-Modified-Since/ETag)
    en cada carga en vez de reutilizar la caché ciegamente. Sin esto, tras
    editar un .js/.css el navegador puede seguir sirviendo la versión
    vieja desde caché heurística durante horas -- ya pasó en esta sesión
    (el bracket de playoffs se veía con el layout viejo tras una edición
    de CSS/JS porque el navegador no volvió a pedir el archivo). No usa
    no-store porque eso forzaría re-descargar el archivo entero cada vez
    -- no-cache sigue permitiendo una respuesta 304 barata cuando el
    archivo no cambió."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_no_cache(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"cache-control", b"no-cache"))
                message = {**message, "headers": headers}
            await send(message)

        await super().__call__(scope, receive, send_with_no_cache)


app = FastAPI(title="NBA Superteam Sim API")
app.include_router(status.router, prefix="/api")
app.include_router(team.router, prefix="/api")
app.include_router(league.router, prefix="/api")
app.include_router(awards.router, prefix="/api")
app.include_router(champions.router, prefix="/api")
app.include_router(explainer.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(sandbox.router, prefix="/api")

app.mount("/", RevalidatingStaticFiles(directory=PROJECT_ROOT / "webapp" / "static", html=True), name="static")
