"""Тесты Этапа 14: кэш, агент Editor, HITL/Active Learning, обратная связь,
задачи обслуживания (reindex RAG, очистка кэша)."""
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.agents.editor import Editor
from backend.agents.supervisor import Supervisor
from backend.core.llm.manager import LLMManager
from backend.core.models import ForecastState
from backend.data import knowledge_base
from backend.data.cache import CacheManager, cache_manager
from backend.data.tasks import TaskRepository
from backend.db.models import Base, ForecastCache, ForecastTask, HitlFeedback, KnowledgeCase
from backend.hitl.active_learning import ActiveLearningManager, build_kb_case
from backend.hitl.feedback import FeedbackHandler
from backend.services import pipeline
from backend.worker import tasks as worker_tasks


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


def series_state(n: int = 160, **meta) -> ForecastState:
    rng = np.random.default_rng(31)
    t = np.arange(n)
    values = 12 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)
    metadata = {"series_values": values.tolist(), "city": "Jena", "variable": "temperature"}
    metadata.update(meta)
    return ForecastState(series_id="s14", horizon=7, freq="D", metadata=metadata)


async def waiting_task(memory_db, with_diagnostics: bool = True) -> str:
    """Задача в статусе waiting_for_human (как оставляет её пайплайн)."""
    state = ForecastState(series_id="wait", horizon=7, model_type="arima",
                          model_params={"order": [1, 1, 1]}, hitl_requested=True)
    state.critic_score = 5.0
    state.metadata = {
        "city": "Jena", "variable": "temperature",
        "critic_suggestions": ["проверить сезонность"], "validator_errors": [],
    }
    if with_diagnostics:
        state.diagnostics = {
            "stats": {"length": 160, "mean": 12.0, "std": 3.6, "min": 4.0, "max": 20.0,
                      "acf_lag1": 0.8},
            "adf": {"adf_pvalue": 0.01, "stationary": True},
            "seasonal_period": 7, "stl": {"trend_strength": 0.1, "seasonal_strength": 0.6},
            "raw_missing_rate": 0.0,
        }
        state.metadata["val_metrics"] = {"mae": 1.2, "mase": 0.8}
    task = await TaskRepository().create(user_id="editor-1", state=state, input_path=None)
    await TaskRepository().save_state(task["id"], state, {"model_type": "arima"},
                                      status="waiting_for_human")
    return task["id"]


# ============================================================================
# Кэш
# ============================================================================
class TestCacheManager:
    def test_make_key_deterministic_and_order_insensitive(self):
        a = CacheManager.make_key("llm", "m", [{"role": "user", "content": "x"}], 0.0)
        b = CacheManager.make_key("llm", "m", [{"content": "x", "role": "user"}], 0.0)
        assert a == b and len(a) == 64
        assert a != CacheManager.make_key("llm", "m", [{"role": "user", "content": "y"}], 0.0)

    async def test_set_get_and_hit_counter(self, memory_db):
        key = CacheManager.make_key("forecast", [1.0, 2.0], 7)
        assert await cache_manager.get(key, model_used="forecast") is None
        assert await cache_manager.set(key, "[1, 2]", model_used="forecast") is True
        assert await cache_manager.set(key, "[9, 9]", model_used="forecast") is False  # не перезапись

        assert await cache_manager.get(key, model_used="forecast") == "[1, 2]"
        assert await cache_manager.get(key, model_used="forecast") == "[1, 2]"
        async with memory_db() as session:
            row = (await session.execute(select(ForecastCache))).scalar_one()
        assert row.hits == 2  # счётчик попаданий растёт

    async def test_clear_by_age_and_hits(self, memory_db):
        now = datetime.now(timezone.utc)
        await cache_manager.set("k1", "v", "forecast")
        await cache_manager.set("k2", "v", "forecast")
        async with memory_db() as session:
            rows = (await session.execute(select(ForecastCache))).scalars().all()
            rows[0].created_at = now - timedelta(days=40)
            rows[1].created_at = now - timedelta(days=40)
            rows[1].hits = 100  # старая, но популярная
            await session.commit()

        # старше 30 дней И непопулярная: уйдёт только rows[0]
        assert await cache_manager.clear(older_than_days=30, max_hits=10) == 1
        stats = await cache_manager.stats()
        assert stats["entries"] == 1 and stats["total_hits"] == 100

        assert await cache_manager.clear() == 1  # без условий — всё
        assert (await cache_manager.stats())["entries"] == 0


class TestLLMCache:
    def _provider(self, calls: list) -> "object":
        from backend.core.llm.providers.openai_compatible import OpenAICompatibleProvider

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json={
                "model": "test-model",
                "choices": [{"message": {"role": "assistant",
                                         "content": '{"score": 8, "reasoning": "ok"}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return OpenAICompatibleProvider(base_url="http://t/v1", model="test-model",
                                        client=client, retry_delay=0)

    async def test_identical_calls_hit_cache(self, memory_db):
        calls: list = []
        manager = LLMManager(provider=self._provider(calls))

        first = await manager.complete("оцени ряд", system="судья")
        second = await manager.complete("оцени ряд", system="судья")
        assert first == second
        assert len(calls) == 1  # второй вызов — из кэша
        assert (await cache_manager.stats())["total_hits"] == 1

    async def test_different_calls_miss(self, memory_db):
        calls: list = []
        manager = LLMManager(provider=self._provider(calls))
        await manager.complete("оцени ряд A")
        await manager.complete("оцени ряд B")
        assert len(calls) == 2

    async def test_cache_can_be_disabled(self, memory_db, monkeypatch):
        from backend.core.config import get_settings

        monkeypatch.setattr(get_settings(), "cache_enabled", False)
        calls: list = []
        manager = LLMManager(provider=self._provider(calls))
        await manager.complete("тот же промпт")
        await manager.complete("тот же промпт")
        assert len(calls) == 2


async def test_forecast_cache_skips_refit(memory_db, tmp_path, monkeypatch):
    from backend.core.config import get_settings

    monkeypatch.setattr(get_settings(), "forecasts_dir", tmp_path)
    calls: list = []

    def fake_predict(series, horizon, model_type, params, season):
        calls.append(1)
        return [1.0] * horizon

    monkeypatch.setattr(pipeline, "predict_final", fake_predict)
    state = series_state(120, series_path=None)
    state.metadata.pop("series_path", None)
    state.task_id = "fc1"
    await pipeline.finalize_forecast(state)
    state2 = series_state(120)
    state2.task_id = "fc2"
    await pipeline.finalize_forecast(state2)  # тот же ряд/модель/горизонт
    assert len(calls) == 1  # второй прогноз взят из кэша
    assert state2.metadata["predictions"] == [1.0] * 7


# ============================================================================
# Агент Editor
# ============================================================================
class TestEditor:
    def _state(self, model="arima", params=None, **meta) -> ForecastState:
        state = ForecastState(series_id="ed", horizon=7, model_type=model,
                              model_params=params or {"order": [1, 1, 1]})
        state.diagnostics = {"seasonal_period": 7, "stats": {}, "adf": {}, "stl": {}}
        state.metadata.update(meta)
        return state

    def test_rule_adds_seasonal_order(self):
        plan = Editor().rule_based_plan(self._state("arima", {"order": [1, 1, 1]}))
        assert plan == ("fit_arima", {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7]})

    def test_rule_raises_ar_order_on_ljung_box(self):
        plan = Editor().rule_based_plan(self._state(
            "sarima", {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7]},
            validator_errors=["Ljung-Box: остатки автокоррелированы"],
        ))
        assert plan == ("fit_arima", {"order": [2, 1, 1], "seasonal_order": [1, 1, 1, 7]})

    def test_rule_doubles_xgboost_lags(self):
        plan = Editor().rule_based_plan(self._state("xgboost", {"lags": 7, "n_estimators": 100}))
        assert plan == ("fit_xgboost", {"lags": 14, "n_estimators": 100})

    def test_rule_fallback_to_auto(self):
        state = self._state(None, None, used_fallback=True)
        assert Editor().rule_based_plan(state) == ("fit_arima", {})

    def test_rule_none_for_prophet(self):
        assert Editor().rule_based_plan(self._state("prophet", {"seasonality_mode": "additive"})) is None

    async def test_no_issues_means_no_change(self):
        state = series_state(120)
        state.validation_passed = True
        state.critic_score = 9.0
        result = await Editor().process(state)
        assert result.editor_changed is False
        assert result.metadata["editor_note"] == "Нет замечаний — правка не нужна"

    async def test_editor_refits_and_records_change(self):
        state = series_state(150)
        state.diagnostics = {"seasonal_period": 7, "stats": {}, "adf": {}, "stl": {}}
        state.critic_score = 4.0
        state.metadata["critic_suggestions"] = ["учесть сезонность"]
        result = await Editor().process(state)

        assert result.editor_changed is True
        change = result.metadata["editor_change"]
        assert change["source"] == "rules"
        assert change["before"]["model"] is None
        assert result.model_type == "sarima"               # модель переобучена
        assert result.model_params["seasonal_order"] == [1, 1, 1, 7]
        assert len(result.metadata["val_pred"]) > 0        # свежий валидационный прогноз

    async def test_llm_advice_is_used_when_valid(self):
        state = series_state(150)
        state.diagnostics = {"seasonal_period": 7, "stats": {}, "adf": {}, "stl": {}}
        state.critic_score = 4.0
        prompts: list[str] = []

        async def advisor(prompt):
            prompts.append(prompt)
            return '{"tool": "fit_xgboost", "arguments": {"lags": 14, "n_estimators": 40}}'

        result = await Editor(advisor=advisor).process(state)
        assert result.model_type == "xgboost"
        assert result.metadata["editor_change"]["source"] == "llm"
        assert "Замечания" in prompts[0] and "fit_xgboost" in prompts[0]

    async def test_invalid_llm_advice_falls_back_to_rules(self):
        state = series_state(150)
        state.diagnostics = {"seasonal_period": 7, "stats": {}, "adf": {}, "stl": {}}
        state.model_type = "arima"
        state.model_params = {"order": [1, 1, 1]}
        state.critic_score = 4.0

        async def advisor(prompt):
            return '{"tool": "fit_magic", "arguments": {}}'

        result = await Editor(advisor=advisor).process(state)
        assert result.metadata["editor_change"]["source"] == "rules"
        assert result.model_type == "sarima"

    async def test_no_applicable_rule(self):
        state = series_state(150)
        state.model_type = "prophet"
        state.model_params = {"seasonality_mode": "additive"}
        state.critic_score = 4.0
        result = await Editor().process(state)
        assert result.editor_changed is False
        assert result.metadata["editor_note"] == "Правила не нашли, что менять"


class TestSupervisorWithEditor:
    async def test_editor_improves_validation_then_converges(self):
        from tests.unit.test_supervisor import MockModelSelector, ScriptedCritic

        validations = iter([False, True, True])

        class FlipValidator(MockModelSelector):
            async def process(self, state):
                state.validation_passed = next(validations)
                return state

        class RealEditor(Editor):
            async def process(self, state):
                state.model_type = "sarima"
                state.editor_changed = True
                return state

        supervisor = Supervisor(agents={
            "model_selector": MockModelSelector(),
            "validator": FlipValidator(),
            "critic": ScriptedCritic([8.0, 8.0, 8.0]),
            "editor": RealEditor(),
        })
        result = await supervisor.process(series_state(80))
        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "converged"
        assert info["iterations"] == 1
        assert result.model_type == "sarima"   # правка Editor принята


# ============================================================================
# HITL / Active Learning
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestActiveLearning:
    async def test_queue_lists_waiting_tasks(self, memory_db):
        task_id = await waiting_task(memory_db)
        queue = await ActiveLearningManager.get_pending_examples()
        assert [e["task_id"] for e in queue] == [task_id]
        assert queue[0]["critic_score"] == 5.0
        assert queue[0]["critic_suggestions"] == ["проверить сезонность"]
        assert await ActiveLearningManager.count_pending() == 1
        assert await ActiveLearningManager.get_pending_examples(city="Delhi") == []

    async def test_confirm_adds_case_to_kb_atomically(self, memory_db):
        task_id = await waiting_task(memory_db)
        result = await ActiveLearningManager.confirm_correction(task_id, user_id="editor-1",
                                                                 comment="ок")
        assert result["status"] == "completed" and result["kb_inserted"] is True

        task = await TaskRepository().get(task_id)
        assert task["status"] == "completed" and task["hitl_requested"] is False
        async with memory_db() as session:
            cases = (await session.execute(select(KnowledgeCase))).scalars().all()
            feedback = (await session.execute(select(HitlFeedback))).scalars().all()
        assert len(cases) == 1 and cases[0].best_model == "arima"
        assert cases[0].features["seasonal_period"] == 7
        assert "Подтверждён экспертом" in cases[0].description
        assert len(feedback) == 1 and feedback[0].decision == "confirm"
        assert feedback[0].user_id == "editor-1"

    async def test_confirm_twice_conflicts(self, memory_db):
        task_id = await waiting_task(memory_db)
        await ActiveLearningManager.confirm_correction(task_id)
        with pytest.raises(ValueError, match="не ожидает подтверждения"):
            await ActiveLearningManager.confirm_correction(task_id)

    async def test_confirm_with_correction_reruns_forecast(self, memory_db, monkeypatch):
        task_id = await waiting_task(memory_db)
        seen = {}

        async def fake_perform(state, supervisor=None):
            seen["forced"] = state.metadata.get("forced_model")
            state.model_type = "xgboost"
            state.metadata["predictions"] = [1.0] * state.horizon
            return state

        monkeypatch.setattr(pipeline, "perform_forecast", fake_perform)
        await ActiveLearningManager.confirm_correction(
            task_id, correction={"model_type": "fit_xgboost", "model_params": {"lags": 14}}
        )
        assert seen["forced"] == {"tool": "fit_xgboost", "arguments": {"lags": 14}}
        task = await TaskRepository().get(task_id)
        assert task["result"]["model_type"] == "xgboost"
        assert task["state"]["hitl_resolution"]["action"] == "accept"

    async def test_confirm_with_unknown_model(self, memory_db):
        task_id = await waiting_task(memory_db)
        with pytest.raises(ValueError, match="Неизвестная модель"):
            await ActiveLearningManager.confirm_correction(task_id, correction={"model_type": "magic"})

    async def test_reject_keeps_kb_intact(self, memory_db):
        task_id = await waiting_task(memory_db)
        result = await ActiveLearningManager.reject_example(task_id, user_id="editor-1",
                                                            reason="модель не подходит")
        assert result["status"] == "rejected"
        assert (await TaskRepository().get(task_id))["status"] == "rejected"
        async with memory_db() as session:
            assert (await session.execute(select(KnowledgeCase))).scalars().all() == []
            feedback = (await session.execute(select(HitlFeedback))).scalars().all()
        assert feedback[0].decision == "reject"
        with pytest.raises(ValueError, match="не ожидает решения"):
            await ActiveLearningManager.reject_example(task_id)

    async def test_confirm_without_diagnostics_skips_kb(self, memory_db):
        task_id = await waiting_task(memory_db, with_diagnostics=False)
        result = await ActiveLearningManager.confirm_correction(task_id)
        assert result["kb_inserted"] is False
        assert (await TaskRepository().get(task_id))["status"] == "completed"

    async def test_stats_single_result(self, memory_db):
        first = await waiting_task(memory_db)
        second = await waiting_task(memory_db)
        await waiting_task(memory_db)  # остаётся в очереди
        await ActiveLearningManager.confirm_correction(first)
        await ActiveLearningManager.reject_example(second)
        assert await ActiveLearningManager.get_stats() == {
            "pending": 1, "confirmed": 1, "rejected": 1,
        }

    async def test_unknown_task(self, memory_db):
        with pytest.raises(LookupError):
            await ActiveLearningManager.confirm_correction("missing")


class TestFeedbackHandler:
    async def test_save_and_list(self, memory_db):
        task_id = await waiting_task(memory_db)
        result = await FeedbackHandler.save_correction(
            task_id, user_id="editor-1", correction={"model_type": "fit_prophet"},
            comment="лучше Prophet", add_to_kb=True,
        )
        assert result["saved"] is True and result["kb_inserted"] is True
        feedback = await FeedbackHandler.list_feedback(task_id=task_id)
        assert feedback[0]["decision"] == "correction"
        assert feedback[0]["correction"] == {"model_type": "fit_prophet"}
        # статус задачи не менялся
        assert (await TaskRepository().get(task_id))["status"] == "waiting_for_human"

    async def test_unknown_task(self, memory_db):
        with pytest.raises(LookupError):
            await FeedbackHandler.save_correction("missing")


def test_build_kb_case_without_diagnostics_returns_none():
    task = ForecastTask(id="t-00000000-xx", series_id="s", horizon=7, status="waiting_for_human",
                        state={"metadata": {}})
    assert build_kb_case(task, ForecastState(series_id="s", horizon=7)) is None


# ============================================================================
# Задачи обслуживания
# ============================================================================
class TestMaintenanceTasks:
    async def test_reindex_rag(self, memory_db):
        assert await knowledge_base.reindex_rag() == {
            "cases": 0, "target": "memory", "note": "empty",
        }
        await knowledge_base.import_cases([
            {"series_id": f"c{i}", "best_model": "arima",
             "features": {"length": 300, "mean": float(i), "std": 1.0, "variable": "temperature"},
             "description": f"кейс {i}"} for i in range(5)
        ])
        assert await knowledge_base.reindex_rag() == {"cases": 5, "target": "memory"}

    async def test_inline_runner_reindex_and_clear(self, memory_db, monkeypatch):
        await knowledge_base.import_cases([{"series_id": "c1", "best_model": "arima",
                                            "description": "d"}])
        runner = worker_tasks.InlineTaskRunner(eager=True)
        await runner.enqueue_reindex()  # выполняется сразу, без исключений

        await cache_manager.set("k", "v", "forecast")
        await runner.enqueue_clear_cache(older_than_days=0)
        assert (await cache_manager.stats())["entries"] == 0

    async def test_celery_runner_sends_maintenance(self, monkeypatch):
        sent = []

        class FakeCelery:
            def send_task(self, name, args=None, kwargs=None, queue=None):
                sent.append((name, args, kwargs))

        import backend.worker.celery_app as celery_module

        monkeypatch.setattr(celery_module, "celery_app", FakeCelery())
        runner = worker_tasks.CeleryTaskRunner()
        await runner.enqueue_reindex()
        await runner.enqueue_clear_cache(older_than_days=10, max_hits=3)
        assert sent == [
            ("rag.reindex_rag_task", [], {}),
            ("cache.clear_cache_task", [], {"older_than_days": 10, "max_hits": 3}),
        ]

    def test_celery_tasks_and_beat_registered(self):
        from backend.worker.celery_app import celery_app

        for name in ("forecast.forecast_task", "evaluate.evaluate_model_task",
                     "rag.reindex_rag_task", "cache.clear_cache_task"):
            assert name in celery_app.tasks
        schedule = celery_app.conf.beat_schedule["clear-old-cache-daily"]
        assert schedule["task"] == "cache.clear_cache_task"
        assert schedule["schedule"] == 60 * 60 * 24

    async def test_clear_cache_task_body(self, memory_db, monkeypatch):
        await cache_manager.set("k", "v", "forecast")
        deleted = worker_tasks.clear_cache_task(older_than_days=0)  # через постоянный loop
        assert deleted == 1
