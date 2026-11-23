"""Интеграционные тесты API прогноза (Этап 12).

Схема задания MLF: POST -> 202 + task_id -> опрос статуса -> результат,
дерево рассуждений, решение человека (HITL).

Изоляция: SQLite in-memory + StaticPool; задачи выполняются сразу (eager),
поэтому ответы статуса/результата детерминированы. LLM — режим stub.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.config import get_settings
from backend.db.models import Base
from backend.services import pipeline

BASE = "/api/v1/forecast"


def series_values(n: int = 150, seed: int = 5) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 12 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.4, n)
    return [round(float(v), 4) for v in values]


@pytest.fixture
def client(monkeypatch, tmp_path):
    import backend.api.deps as deps_module
    import backend.api.main as main_module
    import backend.db.session as db_session_module

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_local = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_session_module, "AsyncSessionLocal", session_local)
    monkeypatch.setattr(deps_module, "AsyncSessionLocal", session_local)
    monkeypatch.setattr(main_module, "async_engine", engine)

    settings = get_settings()
    monkeypatch.setattr(settings, "task_eager", True)
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "forecasts_dir", tmp_path / "forecasts")

    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    import asyncio

    asyncio.run(create_tables())

    from backend.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    import asyncio as aio

    aio.run(engine.dispose())


def register(client: TestClient, email: str) -> dict:
    response = client.post(
        "/auth/register",
        json={"email": email, "username": email.split("@")[0], "password": "StrongPass1"},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def create_task(client: TestClient, headers: dict, **overrides) -> str:
    payload = {"series_values": series_values(), "horizon": 14, "city": "Jena",
               "variable": "temperature"}
    payload.update(overrides)
    response = client.post(f"{BASE}/json", json=payload, headers=headers)
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


# ============================================================================
# Аутентификация и создание задачи
# ============================================================================
class TestCreateForecast:
    def test_requires_auth(self, client):
        body = {"series_values": series_values()}
        assert client.post(f"{BASE}/json", json=body).status_code == 401
        assert client.get(f"{BASE}/any/status").status_code == 401

    def test_create_returns_202_with_task_id(self, client):
        headers = register(client, "creator@example.com")
        response = client.post(
            f"{BASE}/json",
            json={"series_values": series_values(), "horizon": 14},
            headers=headers,
        )
        assert response.status_code == 202
        body = response.json()
        assert body["task_id"]
        assert body["status"] in ("waiting_for_human", "completed")  # eager-режим
        assert body["status_url"].endswith("/status")

    def test_too_short_series_rejected(self, client):
        headers = register(client, "short@example.com")
        response = client.post(
            f"{BASE}/json", json={"series_values": [1.0] * 5}, headers=headers
        )
        assert response.status_code == 422

    def test_csv_upload(self, client):
        headers = register(client, "uploader@example.com")
        rows = "date,value\n" + "\n".join(
            f"2024-01-{(i % 28) + 1:02d},{10 + i * 0.1:.2f}" for i in range(150)
        )
        response = client.post(
            f"{BASE}/",
            files={"file": ("series.csv", rows.encode(), "text/csv")},
            data={"horizon": "7", "city": "Jena", "variable": "temperature"},
            headers=headers,
        )
        assert response.status_code == 202, response.text
        assert response.json()["task_id"]


# ============================================================================
# Статус, результат, рассуждения
# ============================================================================
class TestStatusResultReasoning:
    def test_full_flow(self, client):
        headers = register(client, "owner@example.com")
        task_id = create_task(client, headers, horizon=14)

        status = client.get(f"{BASE}/{task_id}/status", headers=headers)
        assert status.status_code == 200
        assert status.json()["status"] == "waiting_for_human"  # critic (stub) < порога

        result = client.get(f"{BASE}/{task_id}/result", headers=headers)
        assert result.status_code == 200
        body = result.json()
        assert body["model_type"] == "arima"
        assert body["metrics"]["mae"] > 0
        assert len(body["predictions"]) == 14
        assert len(body["prediction_dates"]) == 14
        assert body["predictions_path"].endswith(".csv")
        assert body["validation"]["checks"]["predictions"]["length"] == 14
        assert body["supervisor"]["stopped_reason"]

        reasoning = client.get(f"{BASE}/{task_id}/reasoning", headers=headers)
        assert reasoning.status_code == 200
        tree = reasoning.json()["tree"]
        assert tree[0]["agent_name"] == "supervisor"
        children = {node["agent_name"] for node in tree[0]["children"]}
        assert {"data_analyst", "model_selector", "validator", "critic"} <= children

    def test_result_conflict_when_not_finished(self, client, monkeypatch):
        async def broken(state):
            raise RuntimeError("pipeline down")

        monkeypatch.setattr(pipeline, "perform_forecast", broken)
        headers = register(client, "fail@example.com")
        task_id = create_task(client, headers)

        assert client.get(f"{BASE}/{task_id}/status", headers=headers).json()["status"] == "failed"
        assert client.get(f"{BASE}/{task_id}/result", headers=headers).status_code == 409

    def test_unknown_task_404(self, client):
        headers = register(client, "nobody@example.com")
        assert client.get(f"{BASE}/missing/status", headers=headers).status_code == 404

    def test_other_user_cannot_access(self, client):
        owner = register(client, "owner2@example.com")
        task_id = create_task(client, owner)
        stranger = register(client, "stranger@example.com")
        assert client.get(f"{BASE}/{task_id}/status", headers=stranger).status_code == 404
        assert client.get(f"{BASE}/{task_id}/result", headers=stranger).status_code == 404


# ============================================================================
# Human-in-the-Loop
# ============================================================================
class TestHitl:
    def test_accept_completes_task(self, client):
        headers = register(client, "hitl1@example.com")
        task_id = create_task(client, headers)
        assert client.get(f"{BASE}/{task_id}/status", headers=headers).json()["status"] == \
            "waiting_for_human"

        response = client.post(f"{BASE}/{task_id}/hitl", json={"action": "accept"}, headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

        # результат доступен, повторное решение — конфликт
        assert client.get(f"{BASE}/{task_id}/result", headers=headers).status_code == 200
        assert client.post(
            f"{BASE}/{task_id}/hitl", json={"action": "accept"}, headers=headers
        ).status_code == 409

    def test_change_model_reruns_pipeline(self, client):
        headers = register(client, "hitl2@example.com")
        task_id = create_task(client, headers)

        response = client.post(
            f"{BASE}/{task_id}/hitl",
            json={"action": "change_model", "model_type": "fit_xgboost",
                  "model_params": {"lags": 7, "n_estimators": 50}},
            headers=headers,
        )
        assert response.status_code == 200
        result = client.get(f"{BASE}/{task_id}/result", headers=headers).json()
        assert result["model_type"] == "xgboost"      # принудительно заданная модель
        assert len(result["predictions"]) == 14

    def test_change_horizon(self, client):
        headers = register(client, "hitl3@example.com")
        task_id = create_task(client, headers, horizon=14)

        response = client.post(
            f"{BASE}/{task_id}/hitl",
            json={"action": "change_horizon", "horizon": 7},
            headers=headers,
        )
        assert response.status_code == 200
        result = client.get(f"{BASE}/{task_id}/result", headers=headers).json()
        assert len(result["predictions"]) == 7

    def test_unknown_model_conflict(self, client):
        headers = register(client, "hitl4@example.com")
        task_id = create_task(client, headers)
        response = client.post(
            f"{BASE}/{task_id}/hitl",
            json={"action": "change_model", "model_type": "magic"},
            headers=headers,
        )
        assert response.status_code == 409

    def test_invalid_action_422(self, client):
        headers = register(client, "hitl5@example.com")
        task_id = create_task(client, headers)
        response = client.post(
            f"{BASE}/{task_id}/hitl", json={"action": "explode"}, headers=headers
        )
        assert response.status_code == 422
