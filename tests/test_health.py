"""Тесты каркаса API (Этап 2): проверки /healthz и /health."""
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Weather Forecast AI"


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert "services" in response.json()
