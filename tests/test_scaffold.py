"""Checkpoint 1: application scaffold tests."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_app_imports_and_starts():
    # Entering the context manager runs FastAPI startup/shutdown events.
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200


def test_health_returns_success():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_root_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Secure Digital Communication Simulator" in r.text
    assert "/static/app.js" in r.text and "/static/style.css" in r.text


def test_static_css_loads():
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]


def test_static_js_loads():
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert "/api/health" in r.text


def test_missing_static_file_is_404():
    assert client.get("/static/does-not-exist.js").status_code == 404


def test_unknown_api_route_is_404():
    assert client.get("/api/does-not-exist").status_code == 404


def test_interactive_docs_available():
    assert client.get("/docs").status_code == 200
    assert client.get("/api/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
