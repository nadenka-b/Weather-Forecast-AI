"""Интеграционные тесты health-роутов (Этапы 2–3).

TestClient с контекст-менеджером запускает lifespan: создание таблиц
в тестовой SQLite (см. conftest.py) и закрытие пула соединений.
"""
from fastapi.testclient import TestClient

from backend.api.main import app


def test_healthz() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Weather Forecast AI"


def test_health_database_healthy() -> None:
    """Lifespan создаёт таблицы, /health проверяет БД через SELECT 1."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["services"]["database"]["healthy"] is True
    assert body["services"]["database"]["error"] is None
    # LLM-проверка (Этап 11): в тестах режим stub -> здоров без сети
    assert body["services"]["llm"]["healthy"] is True
    assert body["services"]["llm"]["provider"] == "stub"


def test_lifespan_creates_tables() -> None:
    """После старта приложения в тестовой БД есть все 8 таблиц."""
    import sqlite3

    from tests.integration.conftest import TEST_DB

    with TestClient(app):
        pass

    conn = sqlite3.connect(TEST_DB)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()

    expected = {
        "users",
        "knowledge_base",
        "glossary",
        "cache",
        "evaluation_results",
        "model_comparisons",
        "reasoning_logs",
        "llm_calls",
    }
    assert expected.issubset(tables), f"нет таблиц: {expected - tables}"
