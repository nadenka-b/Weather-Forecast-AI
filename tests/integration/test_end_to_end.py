"""End-to-end: полный путь пользователя (методичка Step_12, п. 12.18).

register → login → forecast → reasoning → повторный прогноз из кэша →
HITL confirm (кейс -> база знаний) → поиск по базе знаний → оценка → очистка кэша.

Работает без LLM и внешних сервисов: LLM — режим stub, задачи выполняются сразу.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.config import get_settings
from backend.db.models import Base, User
from tests.eval_helpers import write_test_set


@pytest.fixture
def client(monkeypatch, tmp_path):
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
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "forecasts_dir", tmp_path / "forecasts")

    import asyncio

    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(create_tables())

    from backend.api.main import app

    with TestClient(app) as test_client:
        yield {"http": test_client, "session_local": session_local}
    asyncio.run(engine.dispose())


def promote(client, email: str, role: str) -> None:
    async def _promote():
        async with client["session_local"]() as session:
            user = (await session.execute(select(User).where(User.email == email))).scalar_one()
            user.role = role
            await session.commit()

    client["http"].portal.call(_promote)


def test_full_user_journey(client):
    http = client["http"]

    # 1. Register
    reg = http.post("/auth/register", json={
        "email": "e2e@example.com", "username": "e2e_user", "password": "TestPass1234",
    })
    assert reg.status_code == 201
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    # 2. Login (те же учётные данные)
    login = http.post("/auth/login", json={
        "email": "e2e@example.com", "password": "TestPass1234",
    })
    assert login.status_code == 200
    assert http.get("/auth/me", headers=headers).json()["email"] == "e2e@example.com"

    # 3. Прогноз: загрузка ряда -> 202 + task_id -> опрос статуса
    rng = np.random.default_rng(42)
    n = 180
    series = [round(float(v), 4) for v in
              12 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.5, n)]
    payload = {"series_values": series, "horizon": 14, "city": "Jena", "variable": "temperature"}
    created = http.post("/api/v1/forecast/json", json=payload, headers=headers)
    assert created.status_code == 202, created.text
    task_id = created.json()["task_id"]

    status = http.get(f"/api/v1/forecast/{task_id}/status", headers=headers).json()
    assert status["status"] == "waiting_for_human"  # Critic (stub) ниже порога -> HITL
    result = http.get(f"/api/v1/forecast/{task_id}/result", headers=headers).json()
    assert result["model_type"] in ("arima", "sarima")
    assert len(result["predictions"]) == 14
    assert result["metrics"]["mae"] > 0

    # 4. Дерево рассуждений: Supervisor + агенты
    reasoning = http.get(f"/api/v1/forecast/{task_id}/reasoning", headers=headers).json()
    root = reasoning["tree"][0]
    assert root["agent_name"] == "supervisor"
    agents = {child["agent_name"] for child in root["children"]}
    assert {"data_analyst", "model_selector", "validator", "critic"} <= agents

    # 5. Повторный прогноз тем же рядом: результат берётся из кэша (hits растёт)
    promote(client, "e2e@example.com", "editor")
    before = http.get("/cache/stats", headers=headers).json()
    again = http.post("/api/v1/forecast/json", json=payload, headers=headers)
    assert again.status_code == 202
    after = http.get("/cache/stats", headers=headers).json()
    assert after["entries"] >= before["entries"]
    assert after["total_hits"] > before["total_hits"]  # прогноз не переобучался

    # 6. HITL: очередь -> подтверждение -> кейс в базе знаний
    queue = http.get("/hitl/queue", headers=headers).json()
    assert queue["count"] == 2  # обе задачи ждут решения
    confirmed = http.post(f"/hitl/{task_id}/confirm", json={"comment": "принято"}, headers=headers)
    assert confirmed.status_code == 200
    assert confirmed.json()["kb_inserted"] is True
    assert http.get(f"/api/v1/forecast/{task_id}/status", headers=headers).json()["status"] == "completed"
    assert http.get("/hitl/queue", headers=headers).json()["count"] == 1

    # 7. Подтверждённый прогноз стал кейсом и находится поиском
    found = http.get("/kb/search", params={"q": "Подтверждён экспертом"}, headers=headers).json()
    assert any(item["series_id"].startswith("hitl|") for item in found["results"])

    # 8. «Мои прогнозы»: завершённая и ожидающая задачи
    tasks = http.get("/api/v1/forecast/", headers=headers).json()["tasks"]
    assert {task["status"] for task in tasks} == {"completed", "waiting_for_human"}

    # 9. Оценка качества: сравнение моделей и пакетный запуск
    compare = http.get("/api/v1/evaluate/compare",
                       params={"models": "naive,arima", "sample_size": 3, "horizon": 7},
                       headers=headers)
    assert compare.status_code == 200, compare.text
    assert compare.json()["winner"] in ("naive", "arima")

    batch = http.post("/api/v1/evaluate/batch", headers=headers,
                      json={"modes": ["baseline"], "horizons": [7], "num_series": 2, "fast": True})
    assert batch.status_code == 202
    run = http.get(f"/api/v1/evaluate/runs/{batch.json()['run_id']}", headers=headers).json()
    assert run["status"] == "completed" and run["summary"][0]["mae"] > 0
    assert len(http.get("/api/v1/evaluate/results", headers=headers).json()["results"]) == 1

    # 10. История решений и очистка кэша
    feedback = http.get("/hitl/feedback", headers=headers).json()["feedback"]
    assert [item["decision"] for item in feedback] == ["confirm"]
    promote(client, "e2e@example.com", "admin")
    http.delete("/admin/cache/clear", params={"max_hits": 10_000}, headers=headers)
    assert http.get("/admin/cache/stats", headers=headers).json()["entries"] == 0


def test_health_through_api(client):
    body = client["http"].get("/health")
    assert body.status_code == 200
    services = body.json()["services"]
    assert services["database"]["healthy"] and services["llm"]["provider"] == "stub"
    assert client["http"].get("/healthz").json()["status"] == "ok"
