"""Тесты Этапа 12 (unit): метрики классификации/диагностики, финальный прогноз,
репозиторий задач, пайплайн, HITL-резолюция, раннеры задач."""
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.metrics import (
    accuracy,
    classification_metrics,
    f1_score,
    ljung_box_pvalue,
    roc_auc,
    shapiro_pvalue,
)
from backend.core.models import ForecastState
from backend.data.tasks import TaskRepository
from backend.db.models import Base
from backend.services import pipeline
from backend.timeseries.classical import predict_final
from backend.worker import tasks as worker_tasks


def seasonal_series(n: int = 220, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 15 + 6 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=n, freq="D"))


@pytest.fixture
async def memory_db(monkeypatch):
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
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield session_local
    await engine.dispose()


# ============================================================================
# Метрики
# ============================================================================
class TestClassificationMetrics:
    def test_accuracy(self):
        assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75

    def test_f1_binary(self):
        assert f1_score([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(2 / 3)

    def test_roc_auc_perfect_and_single_class(self):
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert np.isnan(roc_auc([1, 1], [0.5, 0.7]))

    def test_classification_summary(self):
        result = classification_metrics(
            [0, 0, 1, 1], [0, 1, 1, 1], y_scores=[0.2, 0.6, 0.7, 0.9]
        )
        assert result["accuracy"] == 0.75
        assert "f1" in result and result["roc_auc"] == 1.0

    def test_residual_tests(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, 60)
        assert ljung_box_pvalue(noise) > 0.05       # белый шум — нет автокорреляции
        assert ljung_box_pvalue(np.linspace(-3, 3, 30)) < 0.05  # тренд в остатках
        assert ljung_box_pvalue([0.1, 0.2]) is None  # слишком мало точек
        assert shapiro_pvalue(noise) > 0.01
        assert shapiro_pvalue([1.0, 2.0]) is None


# ============================================================================
# Финальный прогноз
# ============================================================================
class TestPredictFinal:
    def test_naive_length(self):
        preds = predict_final(seasonal_series(), 14, "naive_seasonal", {"season": 7})
        assert len(preds) == 14
        assert all(np.isfinite(preds))

    def test_arima_explicit_order(self):
        preds = predict_final(
            seasonal_series(), 7, "sarima",
            {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7]}, season=7,
        )
        assert len(preds) == 7

    def test_xgboost(self):
        preds = predict_final(seasonal_series(), 10, "xgboost", {"lags": 7, "n_estimators": 40})
        assert len(preds) == 10

    def test_unknown_model_falls_back_to_naive(self):
        preds = predict_final(seasonal_series(), 5, "mystery_model", None)
        assert len(preds) == 5
        assert preds == [round(float(seasonal_series().iloc[-1]), 4)] * 5


# ============================================================================
# Репозиторий задач
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestTaskRepository:
    def test_save_input_series(self, tmp_path, monkeypatch):
        from backend.core.config import get_settings

        monkeypatch.setattr(get_settings(), "uploads_dir", tmp_path)
        repo = TaskRepository()
        path = repo.save_input_series("t1", [1.0, 2.0, 3.0], None)
        assert Path(path).exists()
        frame = pd.read_csv(path)
        assert list(frame["value"]) == [1.0, 2.0, 3.0]

    async def test_lifecycle(self, memory_db):
        repo = TaskRepository()
        state = ForecastState(series_id="s", horizon=7, metadata={"a": 1})
        task = await repo.create(user_id="u1", state=state, input_path=None)
        assert task["status"] == "pending"
        assert task["id"] == state.task_id == state.log_id

        await repo.set_status(task["id"], "running")
        assert (await repo.get(task["id"]))["started_at"] is not None

        state.hitl_requested = True
        state.iteration = 2
        await repo.save_state(task["id"], state, {"model_type": "arima"}, status="waiting_for_human")
        updated = await repo.get(task["id"])
        assert updated["status"] == "waiting_for_human"
        assert updated["hitl_requested"] is True and updated["iterations"] == 2
        assert updated["result"]["model_type"] == "arima"
        assert updated["finished_at"] is not None

        await repo.set_status(task["id"], "failed", error="boom")
        assert (await repo.get(task["id"]))["error"] == "boom"

        with pytest.raises(LookupError):
            await repo.set_status("missing", "running")

    async def test_list_for_user(self, memory_db):
        repo = TaskRepository()
        for i in range(3):
            await repo.create(
                user_id="u1" if i < 2 else "u2",
                state=ForecastState(series_id=f"s{i}", horizon=7),
                input_path=None,
            )
        assert len(await repo.list_for_user("u1")) == 2
        assert len(await repo.list_for_user("u2")) == 1


# ============================================================================
# Пайплайн
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestPipeline:
    async def test_perform_forecast_end_to_end(self, tmp_path, monkeypatch):
        """Полный цикл на реальных агентах (LLM — stub)."""
        from backend.core.config import get_settings

        monkeypatch.setattr(get_settings(), "forecasts_dir", tmp_path)
        series = seasonal_series(200)
        state = ForecastState(
            series_id="pipe", horizon=14, freq="D", task_id="task-pipe",
            metadata={"series_values": series.tolist(), "city": "Jena", "variable": "temperature"},
        )
        state = await pipeline.perform_forecast(state)

        # stub-выбор ARIMA(1,1,1); Editor может усилить модель до SARIMA
        assert state.model_type in ("arima", "sarima")
        assert state.metadata["supervisor"]["stopped_reason"] in (
            "converged", "editor_not_registered", "no_change",
            "max_iterations", "rollback_regression",
        )
        # финальный прогноз построен и провалидирован
        assert len(state.metadata["predictions"]) == 14
        assert len(state.metadata["prediction_dates"]) == 14
        assert Path(state.predictions_path).exists()
        report = state.diagnostics["validation"]
        assert report["checks"]["predictions"]["length"] == 14

    async def test_run_forecast_statuses(self, memory_db, monkeypatch):
        repo = TaskRepository()
        state = ForecastState(series_id="s", horizon=7, metadata={})
        task = await repo.create(user_id="u1", state=state, input_path=None)

        async def fake_perform(st):
            st.model_type = "arima"
            st.metadata["predictions"] = [1.0] * st.horizon
            st.hitl_requested = True  # не сошлось — нужно решение человека
            return st

        monkeypatch.setattr(pipeline, "perform_forecast", fake_perform)
        result_state = await pipeline.run_forecast(task["id"])
        updated = await repo.get(task["id"])
        assert updated["status"] == "waiting_for_human"
        assert updated["hitl_requested"] is True
        assert updated["result"]["model_type"] == "arima"

    async def test_run_forecast_completed_when_converged(self, memory_db, monkeypatch):
        repo = TaskRepository()
        task = await repo.create(
            user_id="u1", state=ForecastState(series_id="s", horizon=7), input_path=None
        )

        async def converged(st):
            st.hitl_requested = False
            return st

        monkeypatch.setattr(pipeline, "perform_forecast", converged)
        await pipeline.run_forecast(task["id"])
        assert (await repo.get(task["id"]))["status"] == "completed"

    async def test_run_forecast_failure(self, memory_db, monkeypatch):
        repo = TaskRepository()
        task = await repo.create(
            user_id="u1", state=ForecastState(series_id="s", horizon=7), input_path=None
        )

        async def broken(st):
            raise RuntimeError("model crashed")

        monkeypatch.setattr(pipeline, "perform_forecast", broken)
        assert await pipeline.run_forecast(task["id"]) is None
        updated = await repo.get(task["id"])
        assert updated["status"] == "failed"
        assert "model crashed" in updated["error"]

    async def test_run_forecast_unknown_task(self, memory_db):
        with pytest.raises(LookupError):
            await pipeline.run_forecast("no-such-task")


# ============================================================================
# HITL-резолюция
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestHitlResolution:
    async def _waiting_task(self) -> str:
        repo = TaskRepository()
        state = ForecastState(
            series_id="s", horizon=7, model_type="arima",
            metadata={"predictions": [1.0] * 7, "supervisor": {"x": 1}},
        )
        state.hitl_requested = True
        task = await repo.create(user_id="u1", state=state, input_path=None)
        await repo.save_state(task["id"], state, {"model_type": "arima"}, status="waiting_for_human")
        return task["id"]

    async def test_accept_completes(self, memory_db):
        task_id = await self._waiting_task()
        task = await pipeline.apply_hitl_resolution(task_id, {"action": "accept"})
        assert task["status"] == "completed"

    async def test_change_model_resets_and_pending(self, memory_db):
        task_id = await self._waiting_task()
        task = await pipeline.apply_hitl_resolution(
            task_id, {"action": "change_model", "model_type": "fit_xgboost",
                      "model_params": {"lags": 14}}
        )
        assert task["status"] == "pending"
        state = task["state"]
        assert state["metadata"]["forced_model"] == {"tool": "fit_xgboost", "arguments": {"lags": 14}}
        assert state["model_type"] is None
        assert state["hitl_requested"] is False
        assert "predictions" not in state["metadata"]  # прошлый результат сброшен

    async def test_change_horizon(self, memory_db):
        task_id = await self._waiting_task()
        task = await pipeline.apply_hitl_resolution(task_id, {"action": "change_horizon", "horizon": 3})
        assert task["status"] == "pending"
        assert task["state"]["horizon"] == 3

    async def test_invalid_cases(self, memory_db):
        task_id = await self._waiting_task()
        with pytest.raises(ValueError, match="Неизвестная модель"):
            await pipeline.apply_hitl_resolution(task_id, {"action": "change_model", "model_type": "magic"})
        with pytest.raises(ValueError, match="horizon"):
            await pipeline.apply_hitl_resolution(task_id, {"action": "change_horizon"})
        with pytest.raises(ValueError, match="Неизвестное действие"):
            await pipeline.apply_hitl_resolution(task_id, {"action": "nope"})

        await pipeline.apply_hitl_resolution(task_id, {"action": "accept"})
        with pytest.raises(ValueError, match="не ожидает решения"):
            await pipeline.apply_hitl_resolution(task_id, {"action": "accept"})
        with pytest.raises(LookupError):
            await pipeline.apply_hitl_resolution("missing", {"action": "accept"})


# ============================================================================
# Раннеры задач и Celery
# ============================================================================
class TestTaskRunners:
    async def test_inline_eager_runs_immediately(self, monkeypatch):
        calls = []

        async def fake_run(task_id):
            calls.append(task_id)

        monkeypatch.setattr(pipeline, "run_forecast", fake_run)
        await worker_tasks.InlineTaskRunner(eager=True).enqueue("t-eager")
        assert calls == ["t-eager"]

    async def test_inline_background(self, monkeypatch):
        calls = []

        async def fake_run(task_id):
            calls.append(task_id)

        monkeypatch.setattr(pipeline, "run_forecast", fake_run)
        await worker_tasks.InlineTaskRunner(eager=False).enqueue("t-bg")
        await asyncio.sleep(0.05)
        assert calls == ["t-bg"]

    async def test_celery_runner_sends_task(self, monkeypatch):
        sent = []

        class FakeCelery:
            def send_task(self, name, args=None, kwargs=None, queue=None):
                sent.append((name, args, queue))

        import backend.worker.celery_app as celery_module

        monkeypatch.setattr(celery_module, "celery_app", FakeCelery())
        await worker_tasks.CeleryTaskRunner().enqueue("t-celery")
        assert sent == [("forecast.forecast_task", ["t-celery"], "forecast")]

    def test_get_task_runner_by_settings(self, monkeypatch):
        from backend.core.config import get_settings

        monkeypatch.setattr(get_settings(), "task_backend", "celery")
        assert isinstance(worker_tasks.get_task_runner(), worker_tasks.CeleryTaskRunner)
        monkeypatch.setattr(get_settings(), "task_backend", "inline")
        assert isinstance(worker_tasks.get_task_runner(), worker_tasks.InlineTaskRunner)

    def test_persistent_loop_is_single(self):
        loop_a = worker_tasks.get_persistent_loop()
        loop_b = worker_tasks.get_persistent_loop()
        assert loop_a is loop_b and loop_a.is_running()

    def test_forecast_task_runs_in_persistent_loop(self, monkeypatch):
        seen = {}

        async def fake_run(task_id):
            seen["loop"] = asyncio.get_running_loop()
            return f"done:{task_id}"

        monkeypatch.setattr(pipeline, "run_forecast", fake_run)
        assert worker_tasks.forecast_task("t1") == "done:t1"
        assert seen["loop"] is worker_tasks.get_persistent_loop()

    def test_celery_task_registered(self):
        from backend.worker.celery_app import celery_app

        assert "forecast.forecast_task" in celery_app.tasks


# ============================================================================
# Схема БД: таблицы создаются в любой точке входа (регрессия: сводка оценки
# не сохранилась, потому что в старой БД не было evaluation_runs)
# ============================================================================
class TestEnsureTables:
    async def test_ensure_tables_once_creates_missing_tables(self, monkeypatch):
        import backend.db.session as db_session_module
        from sqlalchemy import text
        from sqlalchemy.exc import OperationalError

        from backend.db.session import ensure_tables_once

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_local = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )
        monkeypatch.setattr(db_session_module, "AsyncSessionLocal", session_local)
        monkeypatch.setattr(db_session_module, "async_engine", engine)
        monkeypatch.setattr(db_session_module, "_ready_engine_ref", None)

        async def table_names() -> set:
            async with session_local() as session:
                rows = await session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                return {row[0] for row in rows.all()}

        assert await table_names() == set()  # пустая БД
        await ensure_tables_once()
        tables = await table_names()
        assert {
            "users", "knowledge_base", "glossary", "cache", "evaluation_results",
            "model_comparisons", "reasoning_logs", "llm_calls", "forecast_tasks",
            "evaluation_runs",
        } <= tables
        await engine.dispose()
