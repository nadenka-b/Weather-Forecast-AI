"""Интеграционные тесты API оценки (Этап 13).

Запускать оценку (compare, batch) могут только editor/admin; читать результаты —
любой аутентифицированный пользователь. Задачи выполняются сразу (eager).
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.config import get_settings
from backend.db.models import Base
from tests.eval_helpers import write_test_set

BASE = "/api/v1/evaluate"


@pytest.fixture
def api(monkeypatch, tmp_path):
    import backend.api.deps as deps_module
    import backend.api.main as main_module
    import backend.db.session as db_session_module
    import backend.evaluation.runner as runner_module
    import backend.evaluation.service as service_module

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

    test_set = write_test_set(tmp_path / "test_set.jsonl", n_windows=4, n=160)
    monkeypatch.setattr(runner_module, "DEFAULT_TEST_SET", test_set)
    monkeypatch.setattr(service_module, "DEFAULT_TEST_SET", test_set)

    settings = get_settings()
    monkeypatch.setattr(settings, "task_eager", True)
    monkeypatch.setattr(settings, "forecasts_dir", tmp_path / "forecasts")

    import asyncio

    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(create_tables())

    from backend.api.main import app

    with TestClient(app) as client:
        yield {"client": client, "session_local": session_local, "tmp_path": tmp_path}
    asyncio.run(engine.dispose())


def register(api, email: str, role: str = "user") -> dict:
    client = api["client"]
    response = client.post(
        "/auth/register",
        json={"email": email, "username": email.split("@")[0], "password": "StrongPass1"},
    )
    assert response.status_code == 201, response.text
    if role != "user":
        from sqlalchemy import select

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


# ============================================================================
# Права доступа
# ============================================================================
class TestAccess:
    def test_anonymous_401(self, api):
        client = api["client"]
        assert client.get(f"{BASE}/compare").status_code == 401
        assert client.get(f"{BASE}/results").status_code == 401

    def test_plain_user_cannot_run(self, api):
        client = api["client"]
        user = register(api, "user@example.com")
        assert client.get(f"{BASE}/compare", headers=user).status_code == 403
        assert client.post(
            f"{BASE}/batch", json={"modes": ["baseline"], "num_series": 2}, headers=user
        ).status_code == 403
        # читать результаты обычный пользователь может
        assert client.get(f"{BASE}/results", headers=user).status_code == 200


# ============================================================================
# Сравнение моделей
# ============================================================================
class TestCompare:
    def test_editor_can_compare(self, api):
        client = api["client"]
        editor = register(api, "editor@example.com", role="editor")
        response = client.get(
            f"{BASE}/compare", params={"models": "naive,arima", "sample_size": 3, "horizon": 7},
            headers=editor,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body["results"]) == {"naive", "arima"}
        assert body["winner"] in ("naive", "arima")
        assert body["sample_size"] == 3
        assert len(body["pairs"]) == 1 and "wilcoxon" in body["pairs"][0]

        # результат сохранён и доступен через /comparisons
        saved = client.get(f"{BASE}/comparisons", headers=editor).json()["comparisons"]
        assert saved and saved[0]["models_compared"] == ["naive", "arima"]

    def test_unknown_model_422(self, api):
        editor = register(api, "editor2@example.com", role="editor")
        response = api["client"].get(f"{BASE}/compare", params={"models": "magic"}, headers=editor)
        assert response.status_code == 422


# ============================================================================
# Пакетный запуск
# ============================================================================
class TestBatch:
    def test_batch_run_lifecycle(self, api):
        client = api["client"]
        editor = register(api, "editor3@example.com", role="editor")

        response = client.post(
            f"{BASE}/batch",
            json={"modes": ["baseline"], "horizons": [7], "num_series": 2, "fast": True},
            headers=editor,
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]

        run = client.get(f"{BASE}/runs/{run_id}", headers=editor).json()
        assert run["status"] == "completed"
        assert len(run["summary"]) == 1
        assert run["summary"][0]["mode"] == "baseline"
        assert run["summary"][0]["mae"] > 0

        results = client.get(f"{BASE}/results", params={"mode": "baseline"}, headers=editor).json()
        assert len(results["results"]) == 1

    def test_invalid_params_422(self, api):
        editor = register(api, "editor4@example.com", role="editor")
        assert api["client"].post(
            f"{BASE}/batch", json={"modes": ["magic"]}, headers=editor
        ).status_code == 422
        assert api["client"].post(
            f"{BASE}/batch", json={"data_fractions": [1.5]}, headers=editor
        ).status_code == 422

    def test_unknown_run_404(self, api):
        user = register(api, "reader@example.com")
        assert api["client"].get(f"{BASE}/runs/missing", headers=user).status_code == 404
