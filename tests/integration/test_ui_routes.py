"""Интеграционные тесты эндпоинтов, добавленных для Streamlit-интерфейса (Этап 15)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.config import get_settings
from backend.data.cache import cache_manager
from backend.db.models import Base, User


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
        async def promote():
            async with api["session_local"]() as session:
                user = (await session.execute(select(User).where(User.email == email))).scalar_one()
                user.role = role
                await session.commit()

        client.portal.call(promote)
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def series(n=150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return [round(float(v), 4) for v in 12 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)]


# ============================================================================
# Список моих задач
# ============================================================================
class TestMyForecasts:
    def test_lists_only_own_tasks(self, api):
        client = api["client"]
        owner = register(api, "o1@example.com")
        other = register(api, "o2@example.com")
        created = client.post("/api/v1/forecast/json", json={"series_values": series()},
                              headers=owner).json()
        client.post("/api/v1/forecast/json", json={"series_values": series(seed=1)},
                    headers=other)

        tasks = client.get("/api/v1/forecast/", headers=owner).json()["tasks"]
        assert [t["task_id"] for t in tasks] == [created["task_id"]]
        assert tasks[0]["status"] == "waiting_for_human"
        assert tasks[0]["model_type"] in ("arima", "sarima")
        assert tasks[0]["city"] is None and tasks[0]["horizon"] == 14

    def test_requires_auth(self, api):
        assert api["client"].get("/api/v1/forecast/").status_code == 401

    def test_headerless_csv_is_accepted(self, api):
        """CSV без заголовка: первая строка — данные, а не имена колонок."""
        headers = register(api, "csv@example.com")
        rows = "\n".join(str(v) for v in series(40))
        response = api["client"].post(
            "/api/v1/forecast/", files={"file": ("s.csv", rows.encode(), "text/csv")},
            data={"horizon": "7"}, headers=headers,
        )
        assert response.status_code == 202, response.text


# ============================================================================
# База знаний и справочник (чтение)
# ============================================================================
class TestKnowledgeRead:
    def test_requires_auth(self, api):
        assert api["client"].get("/kb/search", params={"q": "x"}).status_code == 401
        assert api["client"].get("/kb/cases").status_code == 401
        assert api["client"].get("/glossary/terms").status_code == 401

    def test_search_ranks_relevant_case_first(self, api):
        admin = register(api, "adm1@example.com", role="admin")
        client = api["client"]
        for series_id, description in (
            ("kb|temp|1", "Дневной ряд temperature (Jena), сезонность 7, тренд слабый."),
            ("kb|humid|1", "Дневной ряд humidity (Delhi), муссонная сезонность."),
        ):
            assert client.post("/admin/kb/add", headers=admin, json={
                "series_id": series_id, "best_model": "arima", "description": description,
            }).status_code == 201
        user = register(api, "u1@example.com")
        found = client.get("/kb/search", params={"q": "temperature Jena тренд"}, headers=user).json()
        assert found["count"] == 1  # второй кейс не содержит слов запроса
        assert found["results"][0]["series_id"] == "kb|temp|1"
        assert found["results"][0]["best_model"] == "arima"
        # на крошечном корпусе IDF зануляется — запасной рейтинг по числу совпадений
        broad = client.get("/kb/search", params={"q": "Дневной ряд"}, headers=user).json()
        assert broad["count"] == 2
        assert client.get("/kb/search", params={"q": "   "}, headers=user).json()["results"] == []

    def test_list_cases_with_filters(self, api):
        admin = register(api, "adm2@example.com", role="admin")
        client = api["client"]
        client.post("/admin/kb/add", headers=admin, json={
            "series_id": "kb|t|1", "best_model": "sarima", "city": "Jena",
            "target_column": "temperature", "description": "ряд температуры",
        })
        client.post("/admin/kb/add", headers=admin, json={
            "series_id": "kb|h|1", "best_model": "xgboost", "city": "Delhi",
            "target_column": "humidity", "description": "ряд влажности",
        })
        user = register(api, "u2@example.com")

        everything = client.get("/kb/cases", headers=user).json()
        assert everything["total"] == 2
        only_jena = client.get("/kb/cases", params={"city": "Jena"}, headers=user).json()
        assert only_jena["total"] == 1 and only_jena["cases"][0]["best_model"] == "sarima"
        by_text = client.get("/kb/cases", params={"search": "влажности"}, headers=user).json()
        assert [c["series_id"] for c in by_text["cases"]] == ["kb|h|1"]
        assert client.get("/kb/cases", params={"limit": 1}).status_code == 401  # без токена

        case_id = only_jena["cases"][0]["id"]
        assert client.get(f"/kb/cases/{case_id}", headers=user).json()["city"] == "Jena"
        assert client.get("/kb/cases/missing", headers=user).status_code == 404

    def test_glossary_terms(self, api):
        admin = register(api, "adm3@example.com", role="admin")
        user = register(api, "u3@example.com")
        client = api["client"]
        for term, category in (("Адвекция", "метеорология"), ("MASE", "ML")):
            client.post("/admin/glossary/add", headers=admin,
                        json={"term": term, "definition": "определение", "category": category})

        listing = client.get("/glossary/terms", headers=user).json()
        assert listing["total"] == 2 and listing["categories"] == ["ML", "метеорология"]
        only_ml = client.get("/glossary/terms", params={"category": "ML"}, headers=user).json()
        assert [t["term"] for t in only_ml["terms"]] == ["MASE"]
        by_text = client.get("/glossary/terms", params={"search": "адвек"}, headers=user).json()
        assert [t["term"] for t in by_text["terms"]] == ["Адвекция"]


# ============================================================================
# Администрирование
# ============================================================================
class TestAdmin:
    def test_requires_admin(self, api):
        client = api["client"]
        assert client.post("/admin/kb/add", json={"series_id": "x", "best_model": "arima"}).status_code == 401
        editor = register(api, "ed1@example.com", role="editor")
        assert client.post("/admin/kb/add", headers=editor,
                           json={"series_id": "x", "best_model": "arima"}).status_code == 403
        assert client.delete("/admin/kb/delete/x", headers=editor).status_code == 403
        assert client.post("/admin/glossary/add", headers=editor,
                           json={"term": "t", "definition": "d"}).status_code == 403
        assert client.get("/admin/cache/stats", headers=editor).status_code == 403

    def test_kb_case_lifecycle(self, api):
        admin = register(api, "adm4@example.com", role="admin")
        client = api["client"]
        created = client.post("/admin/kb/add", headers=admin, json={
            "series_id": "kb|life|1", "best_model": "prophet", "city": "Jena",
            "features": {"variable": "pressure", "mean": 1013.0}, "hyperparams": {},
            "metrics": {"mae": 1.0}, "description": "давление",
        })
        assert created.status_code == 201, created.text
        assert created.json()["target_column"] == "pressure"  # из features

        duplicate = client.post("/admin/kb/add", headers=admin,
                                json={"series_id": "kb|life|1", "best_model": "arima"})
        assert duplicate.status_code == 409
        assert client.post("/admin/kb/add", headers=admin, json={"series_id": "", "best_model": "x"}
                           ).status_code == 422

        assert client.delete(f"/admin/kb/delete/{created.json()['id']}", headers=admin).status_code == 200
        assert client.delete("/admin/kb/delete/missing", headers=admin).status_code == 404
        assert client.get("/kb/cases", headers=admin).json()["total"] == 0

    def test_reindex_scheduled(self, api):
        admin = register(api, "adm5@example.com", role="admin")
        response = api["client"].post("/admin/kb/reindex", headers=admin)
        assert response.status_code == 202 and response.json()["scheduled"] is True

    def test_glossary_lifecycle_and_import(self, api):
        admin = register(api, "adm6@example.com", role="admin")
        client = api["client"]
        created = client.post("/admin/glossary/add", headers=admin,
                              json={"term": "Изотерма", "definition": "Линия равной температуры.",
                                    "category": "метеорология"})
        assert created.status_code == 201
        assert client.post("/admin/glossary/add", headers=admin,
                           json={"term": "Изотерма", "definition": "другое"}).status_code == 409
        assert client.post("/admin/glossary/add", headers=admin,
                           json={"term": "  ", "definition": "d"}).status_code == 422

        csv_text = "term;definition;category\nТермос;прибор;метеорология\nИзотерма;дубль;метео\n\nБарометр;прибор\n"
        imported = client.post("/admin/glossary/import-csv", headers=admin,
                               files={"file": ("g.csv", csv_text.encode(), "text/csv")}).json()
        assert imported["added"] == 2 and imported["skipped"] == 1  # Изотерма — дубль

        assert client.delete(f"/admin/glossary/delete/{created.json()['id']}", headers=admin).status_code == 200
        assert client.delete("/admin/glossary/delete/missing", headers=admin).status_code == 404

    def test_cache_aliases(self, api):
        admin = register(api, "adm7@example.com", role="admin")
        client = api["client"]

        async def seed():
            await cache_manager.set("k", "v", "forecast")

        client.portal.call(seed)
        assert client.get("/admin/cache/stats", headers=admin).json()["entries"] == 1
        cleared = client.delete("/admin/cache/clear", params={"max_hits": 0}, headers=admin).json()
        assert cleared["deleted_inline"] == 1
        assert client.get("/admin/cache/stats", headers=admin).json()["entries"] == 0


# ============================================================================
# Алиасы HITL по путям методички
# ============================================================================
class TestHitlAliases:
    def test_uncertain_paths(self, api):
        client = api["client"]
        owner = register(api, "own@example.com")
        editor = register(api, "ed2@example.com", role="editor")
        task_id = client.post("/api/v1/forecast/json", json={"series_values": series()},
                              headers=owner).json()["task_id"]

        queue = client.get("/hitl/uncertain", headers=editor)
        assert queue.status_code == 200
        assert queue.json()["count"] == 1 and queue.json()["examples"][0]["task_id"] == task_id
        assert client.get("/hitl/uncertain/stats", headers=editor).json()["pending"] == 1

        rejected = client.post(f"/hitl/uncertain/{task_id}/reject", json={"reason": "нет"},
                               headers=editor)
        assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"

        second = client.post("/api/v1/forecast/json", json={"series_values": series(seed=2)},
                             headers=owner).json()["task_id"]
        confirmed = client.post(f"/hitl/uncertain/{second}/confirm", json={}, headers=editor)
        assert confirmed.status_code == 200 and confirmed.json()["kb_inserted"] is True

        saved = client.post("/hitl/save-correction", headers=editor,
                            json={"task_id": second, "comment": "принято"})
        assert saved.status_code == 200 and saved.json()["saved"] is True

    def test_aliases_require_editor(self, api):
        user = register(api, "plain@example.com")
        client = api["client"]
        assert client.get("/hitl/uncertain", headers=user).status_code == 403
        assert client.get("/hitl/uncertain/stats", headers=user).status_code == 403
        assert client.post("/hitl/uncertain/x/confirm", json={}, headers=user).status_code == 403
        assert client.post("/hitl/save-correction", json={"task_id": "x"}, headers=user).status_code == 403


# ============================================================================
# /health: опциональные сервисы
# ============================================================================
class TestHealthOptionalServices:
    def test_healthy_without_optional_services(self, api):
        response = api["client"].get("/health")
        assert response.status_code == 200
        services = response.json()["services"]
        assert services["redis"]["healthy"] is True   # inline-режим: redis не нужен
        assert services["qdrant"]["healthy"] is True  # memory-режим: qdrant не нужен

    def test_unhealthy_when_enabled_service_is_down(self, api, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "task_backend", "celery")
        monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")  # заведомо закрытый порт
        monkeypatch.setattr(settings, "vector_backend", "qdrant")
        monkeypatch.setattr(settings, "qdrant_url", "http://127.0.0.1:1")
        response = api["client"].get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"
        assert body["services"]["redis"]["healthy"] is False
        assert body["services"]["qdrant"]["healthy"] is False
