"""Интеграционные тесты API Human-in-the-Loop и кэша (Этап 14).

Сценарий: задача прогноза (stub-режим -> waiting_for_human) попадает в очередь
HITL; эксперт подтверждает её (кейс -> база знаний) или отклоняет.
Роуты /hitl/* — только editor/admin; очистка кэша — только admin.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.config import get_settings
from backend.data.cache import cache_manager
from backend.db.models import Base, KnowledgeCase


@pytest.fixture
def api(monkeypatch, tmp_path):
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

    import asyncio

    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(create_tables())

    from backend.api.main import app

    with TestClient(app) as client:
        yield {"client": client, "session_local": session_local}
    asyncio.run(engine.dispose())


def register(api, email: str, role: str = "user") -> dict:
    client = api["client"]
    response = client.post(
        "/auth/register",
        json={"email": email, "username": email.split("@")[0], "password": "StrongPass1"},
    )
    assert response.status_code == 201, response.text
    if role != "user":
        from backend.db.models import User

        async def promote():
            async with api["session_local"]() as session:
                user = (
                    await session.execute(select(User).where(User.email == email))
                ).scalar_one()
                user.role = role
                await session.commit()

        client.portal.call(promote)
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def create_forecast(api, headers: dict) -> str:
    import numpy as np

    rng = np.random.default_rng(6)
    n = 150
    values = 12 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.5, n)
    response = api["client"].post(
        "/api/v1/forecast/json",
        json={"series_values": [round(float(v), 4) for v in values], "horizon": 7,
              "city": "Jena", "variable": "temperature"},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


# ============================================================================
# Права доступа
# ============================================================================
class TestHitlAccess:
    def test_anonymous_and_user_forbidden(self, api):
        client = api["client"]
        assert client.get("/hitl/queue").status_code == 401
        user = register(api, "user@example.com")
        assert client.get("/hitl/queue", headers=user).status_code == 403
        assert client.get("/hitl/stats", headers=user).status_code == 403
        assert client.post("/hitl/x/confirm", json={}, headers=user).status_code == 403


# ============================================================================
# Очередь и подтверждение
# ============================================================================
class TestHitlQueue:
    def test_full_hitl_cycle(self, api):
        client = api["client"]
        owner = register(api, "owner@example.com")
        editor = register(api, "editor@example.com", role="editor")

        task_id = create_forecast(api, owner)
        queue = client.get("/hitl/queue", headers=editor).json()
        assert queue["count"] == 1
        item = queue["examples"][0]
        assert item["task_id"] == task_id
        assert item["city"] == "Jena" and item["variable"] == "temperature"
        assert item["critic_score"] == 5.0            # причина попадания в очередь
        assert item["model_type"] in ("arima", "sarima")

        stats = client.get("/hitl/stats", headers=editor).json()
        assert stats == {"pending": 1, "confirmed": 0, "rejected": 0}

        # фильтр по городу
        assert client.get("/hitl/queue", params={"city": "Delhi"}, headers=editor).json()["examples"] == []

    def test_confirm_populates_knowledge_base(self, api, monkeypatch):
        # тестовый ряд (еженедельной сезонности) даёт у stub-ARIMA MASE ~11: здесь
        # проверяется само звено «подтверждение -> база знаний», поэтому порог качества
        # снимаем; фильтр по MASE проверяется юнит-тестами (test_stage14)
        monkeypatch.setattr(get_settings(), "hitl_kb_max_mase", float("inf"))
        client = api["client"]
        owner = register(api, "owner2@example.com")
        editor = register(api, "editor2@example.com", role="editor")
        task_id = create_forecast(api, owner)

        response = client.post(f"/hitl/{task_id}/confirm", json={"comment": "принято"},
                               headers=editor)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["task_id"] == task_id
        assert body["status"] == "completed"
        assert body["kb_inserted"] is True
        assert body["kb_case_id"].startswith("hitl|")

        # active learning: подтверждённый прогноз стал кейсом базы знаний
        async def kb_count():
            async with api["session_local"]() as session:
                rows = (await session.execute(select(KnowledgeCase))).scalars().all()
                return rows

        cases = client.portal.call(kb_count)
        assert len(cases) == 1 and cases[0].series_id.startswith("hitl|")

        assert client.get("/hitl/queue", headers=editor).json()["count"] == 0
        assert client.get("/hitl/stats", headers=editor).json()["confirmed"] == 1
        # задача завершена и больше не ждёт человека
        assert client.get(f"/api/v1/forecast/{task_id}/status", headers=owner).json()["status"] == \
            "completed"

    def test_confirm_with_correction(self, api):
        client = api["client"]
        owner = register(api, "owner3@example.com")
        editor = register(api, "editor3@example.com", role="editor")
        task_id = create_forecast(api, owner)

        response = client.post(
            f"/hitl/{task_id}/confirm",
            json={"model_type": "fit_xgboost", "model_params": {"lags": 7, "n_estimators": 40}},
            headers=editor,
        )
        assert response.status_code == 200, response.text
        result = client.get(f"/api/v1/forecast/{task_id}/result", headers=owner).json()
        assert result["model_type"] == "xgboost"   # прогноз перевыполнен с правкой

    def test_confirm_conflicts(self, api):
        client = api["client"]
        owner = register(api, "owner4@example.com")
        editor = register(api, "editor4@example.com", role="editor")
        assert client.post("/hitl/missing/confirm", json={}, headers=editor).status_code == 404

        task_id = create_forecast(api, owner)
        assert client.post(
            f"/hitl/{task_id}/confirm", json={"model_type": "magic"}, headers=editor
        ).status_code == 409
        client.post(f"/hitl/{task_id}/confirm", json={}, headers=editor)
        assert client.post(f"/hitl/{task_id}/confirm", json={}, headers=editor).status_code == 409


# ============================================================================
# Отклонение и обратная связь
# ============================================================================
class TestRejectAndFeedback:
    def test_reject_keeps_kb_intact(self, api):
        client = api["client"]
        owner = register(api, "owner5@example.com")
        editor = register(api, "editor5@example.com", role="editor")
        task_id = create_forecast(api, owner)

        response = client.post(f"/hitl/{task_id}/reject", json={"reason": "модель не подходит"},
                               headers=editor)
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"
        assert client.get("/hitl/stats", headers=editor).json()["rejected"] == 1

        async def kb_count():
            async with api["session_local"]() as session:
                return len((await session.execute(select(KnowledgeCase))).scalars().all())

        assert client.portal.call(kb_count) == 0
        # отклонённая задача не даёт результат
        assert client.get(f"/api/v1/forecast/{task_id}/result", headers=owner).status_code == 409

    def test_feedback_saved_and_listed(self, api):
        client = api["client"]
        editor = register(api, "editor6@example.com", role="editor")
        task_id = create_forecast(api, editor)

        response = client.post(
            "/hitl/feedback",
            json={"task_id": task_id, "correction": {"model_type": "fit_prophet"},
                  "comment": "в следующий раз Prophet"},
            headers=editor,
        )
        assert response.status_code == 200
        assert response.json() == {"task_id": task_id, "saved": True, "kb_inserted": False}

        history = client.get("/hitl/feedback", params={"task_id": task_id}, headers=editor).json()
        assert history["feedback"][0]["decision"] == "correction"
        assert history["feedback"][0]["correction"] == {"model_type": "fit_prophet"}

        assert client.post("/hitl/feedback", json={"task_id": "missing"}, headers=editor).status_code == 404


# ============================================================================
# Кэш
# ============================================================================
class TestCacheRoutes:
    def test_stats_and_clear_permissions(self, api):
        client = api["client"]
        user = register(api, "u@example.com")
        editor = register(api, "e@example.com", role="editor")
        admin = register(api, "a@example.com", role="admin")

        assert client.get("/cache/stats").status_code == 401
        assert client.get("/cache/stats", headers=user).status_code == 403
        assert client.delete("/cache/clear", headers=editor).status_code == 403  # только admin

        async def seed():
            await cache_manager.set("k1", "v", "forecast")
            await cache_manager.set("k2", "v", "llm")

        client.portal.call(seed)
        assert client.get("/cache/stats", headers=editor).json() == {"entries": 2, "total_hits": 0}

        response = client.delete("/cache/clear", params={"max_hits": 0}, headers=admin)
        assert response.status_code == 200
        assert response.json()["scheduled"] is True
        assert client.get("/cache/stats", headers=admin).json()["entries"] == 0
