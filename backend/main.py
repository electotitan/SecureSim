"""FastAPI application entry point.

Checkpoint 1 scope: application scaffold only.
  * GET /            -> serves the frontend (frontend/index.html)
  * GET /api/health  -> health check
  * /static/*        -> frontend static assets (CSS, JS, components)
  * /docs            -> interactive API documentation (FastAPI default)
  * /api/docs        -> alias of the interactive documentation

No mathematical pipeline stages are implemented yet.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html

APP_NAME = "Interactive Mathematical Simulator for Secure Digital Communication"
APP_VERSION = "0.1.0"

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Educational simulator. All algorithms are implemented from first "
        "principles and are NOT suitable for protecting real sensitive data."
    ),
)


@app.get("/api/health", tags=["system"])
def health() -> dict:
    """Health check used by the frontend connection indicator."""
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}


@app.get("/api/docs", include_in_schema=False, response_class=HTMLResponse)
def api_docs() -> HTMLResponse:
    """Alias for the interactive documentation (spec lists /api/docs)."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{APP_NAME} - API docs")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Static assets: /static/style.css, /static/app.js, /static/components/*.js
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
