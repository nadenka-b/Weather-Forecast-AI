"""Юнит-тесты Supervisor (Этап 5): сценарии методички Step_5, адаптированные
под ForecastState: converged, rollback_regression, no_change, max_iterations
(+ HITL по заданию MLF), пропущенные агенты, ошибки агентов.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.agents.base import BaseAgent
from backend.agents.reasoning import InMemoryLogStore, ReasoningLogger
from backend.agents.supervisor import Supervisor
from backend.core.models import ForecastState
from backend.db.models import Base, ReasoningLog


# ============================================================================
# Mock-агенты
# ============================================================================
class MockDataAnalyst(BaseAgent):
    """Проставляет диагностику (имитация Этапа 6)."""

    async def process(self, state: ForecastState) -> ForecastState:
        state.diagnostics = {"adf_pvalue": 0.01, "seasonal_period": 7}
        self._last_result = state.diagnostics
        return state


class MockModelSelector(BaseAgent):
    """Выбирает SARIMA (имитация Этапа 7)."""

    async def process(self, state: ForecastState) -> ForecastState:
        state.model_type = "sarima"
        state.model_params = {"order": [1, 1, 1]}
        self._last_result = state.model_type
        return state


class ScriptedValidator(BaseAgent):
    """validation_passed = фиксированное значение."""

    def __init__(self, passed: bool = True):
        super().__init__()
        self.passed = passed

    async def process(self, state: ForecastState) -> ForecastState:
        state.validation_passed = self.passed
        self._last_result = {"passed": self.passed}
        return state


class ScriptedCritic(BaseAgent):
    """Выдаёт оценки по сценарию (список), последняя повторяется."""

    def __init__(self, scores: list[float]):
        super().__init__()
        self.scores = list(scores)

    async def process(self, state: ForecastState) -> ForecastState:
        state.critic_score = self.scores.pop(0) if self.scores else state.critic_score
        state.metadata["critic_suggestions"] = []
        self._last_result = {"score": state.critic_score}
        return state


class ScriptedEditor(BaseAgent):
    """Меняет гиперпараметры (или нет / падает — по флагам)."""

    def __init__(self, changed: bool = True, fail: bool = False):
        super().__init__()
        self.changed = changed
        self.fail = fail

    async def process(self, state: ForecastState) -> ForecastState:
        if self.fail:
            raise RuntimeError("editor LLM timeout")
        state.editor_changed = self.changed
        if self.changed:
            state.model_params = {"order": [2, 1, 1], "seasonal_order": [1, 1, 1, 7]}
        self._last_result = {"changed": self.changed}
        return state


def make_state() -> ForecastState:
    return ForecastState(series_id="jena_temp_2013", horizon=14, freq="D")


def full_agents(**overrides) -> dict:
    agents = {
        "data_analyst": MockDataAnalyst(),
        "model_selector": MockModelSelector(),
        "validator": ScriptedValidator(passed=True),
        "critic": ScriptedCritic([8.0]),
        "editor": ScriptedEditor(),
    }
    agents.update(overrides)
    return agents


# ============================================================================
# Сценарий: сходимость без итераций
# ============================================================================
class TestConverged:
    async def test_converged_zero_iterations(self):
        sup = Supervisor(agents=full_agents())
        result = await sup.process(make_state())

        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "converged"
        assert info["iterations"] == 0
        assert info["final"] is True
        assert result.hitl_requested is False
        assert result.validation_passed is True
        assert result.critic_score == 8.0
        assert result.model_type == "sarima"


# ============================================================================
# Сценарий: Editor улучшает за одну итерацию
# ============================================================================
class TestEditorImproves:
    async def test_converged_after_one_iteration(self):
        agents = full_agents(critic=ScriptedCritic([5.0, 8.0]))
        sup = Supervisor(agents=agents)
        result = await sup.process(make_state())

        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "converged"
        assert info["iterations"] == 1
        assert result.iteration == 1
        assert result.critic_score == 8.0
        assert result.hitl_requested is False


# ============================================================================
# Сценарий: max_iterations исчерпан, оценка ниже порога -> HITL
# ============================================================================
class TestMaxIterationsHitl:
    async def test_improving_but_below_threshold_requests_hitl(self):
        # 4.0 -> 5.0 -> 6.0 -> 6.4: улучшение каждую итерацию, но порог 7.0 не достигнут
        agents = full_agents(critic=ScriptedCritic([4.0, 5.0, 6.0, 6.4]))
        sup = Supervisor(agents=agents, max_iterations=3)
        result = await sup.process(make_state())

        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "max_iterations"
        assert info["iterations"] == 3
        assert info["final"] is False
        assert result.hitl_requested is True  # задание MLF: несходимость -> HITL


# ============================================================================
# Сценарий: rollback при регрессии
# ============================================================================
class TestRollbackRegression:
    async def test_regression_rolls_back_params(self):
        agents = full_agents(critic=ScriptedCritic([6.0, 3.0]))
        sup = Supervisor(agents=agents)
        result = await sup.process(make_state())

        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "rollback_regression"
        assert result.metadata["editor_rolled_back"] is True
        # гиперпараметры восстановлены из снапшота (Editor менял на [2,1,1])
        assert result.model_params == {"order": [1, 1, 1]}
        assert result.hitl_requested is True


# ============================================================================
# Сценарий: Editor ничего не изменил
# ============================================================================
class TestNoChange:
    async def test_no_change_stops_without_rollback(self):
        agents = full_agents(
            critic=ScriptedCritic([5.0, 5.0]), editor=ScriptedEditor(changed=False)
        )
        sup = Supervisor(agents=agents)
        result = await sup.process(make_state())

        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "no_change"
        assert result.metadata.get("editor_rolled_back") is not True
        assert result.hitl_requested is True


# ============================================================================
# Сценарий: ошибка агента -> rollback_error
# ============================================================================
class TestAgentError:
    async def test_editor_failure_rolls_back_and_logs(self):
        agents = full_agents(
            critic=ScriptedCritic([5.0]), editor=ScriptedEditor(fail=True)
        )
        sup = Supervisor(agents=agents)
        result = await sup.process(make_state())

        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "rollback_error"
        # agent_errors откатывается вместе со снапшотом — факт сбоя живёт в трассе
        trace_statuses = {
            e["agent_name"]: e["status"] for e in result.metadata["reasoning_trace"]
        }
        assert trace_statuses.get("editor") == "error"
        assert result.model_params == {"order": [1, 1, 1]}  # откат к снапшоту
        assert result.hitl_requested is True


# ============================================================================
# Сценарий: агенты не зарегистрированы (Этапы 6–8 ещё не реализованы)
# ============================================================================
class TestMissingAgents:
    async def test_all_skipped_still_finalizes(self):
        sup = Supervisor(agents={})
        result = await sup.process(make_state())

        info = result.metadata["supervisor"]
        assert info["stopped_reason"] == "editor_not_registered"
        assert result.log_id is not None
        # все шаги пайплайна в трассе со статусом skipped
        statuses = {
            e["agent_name"]: e["status"] for e in result.metadata["reasoning_trace"]
        }
        assert statuses["data_analyst"] == "skipped"
        assert statuses["model_selector"] == "skipped"

    async def test_log_id_reused_when_provided(self):
        state = make_state()
        state.log_id = "req-fixed"
        sup = Supervisor(agents={})
        result = await sup.process(state)
        assert result.log_id == "req-fixed"


# ============================================================================
# Reasoning log: иерархия внутри прогона Supervisor
# ============================================================================
class TestSupervisorReasoning:
    async def test_trace_hierarchy_and_status(self):
        store = InMemoryLogStore()
        sup = Supervisor(
            agents=full_agents(critic=ScriptedCritic([5.0, 8.0])),
            logger_=ReasoningLogger("req-trace", store=store),
        )
        await sup.process(make_state())

        tree = store.tree()
        assert len(tree) == 1
        root = tree[0]
        assert root["agent_name"] == "supervisor"
        child_names = {c["agent_name"] for c in root["children"]}
        assert {"data_analyst", "model_selector", "validator", "critic", "editor"} <= child_names
        # все дети ссылаются на root
        assert all(c["parent_id"] == root["id"] for c in root["children"])
        # все шаги завершены успешно
        assert all(c["status"] == "success" for c in root["children"])

    async def test_state_gets_reasoning_trace(self):
        sup = Supervisor(agents=full_agents())
        result = await sup.process(make_state())
        trace = result.metadata["reasoning_trace"]
        assert trace  # непустой
        assert any(e["agent_name"] == "supervisor" for e in trace)


# ============================================================================
# DbLogStore: запись в таблицу reasoning_logs
# ============================================================================
class TestDbLogStore:
    async def test_write_and_update_rows(self, monkeypatch):
        import backend.db.session as db_session_module
        from backend.agents.reasoning import DbLogStore

        test_engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        test_session_local = async_sessionmaker(
            bind=test_engine, class_=AsyncSession, expire_on_commit=False
        )
        monkeypatch.setattr(db_session_module, "AsyncSessionLocal", test_session_local)
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        store = DbLogStore()
        logger = ReasoningLogger(request_id="req-db", store=store)

        root_id = await logger.start("supervisor", input_summary="series=jena")
        child_id = await logger.start("critic", parent_id=root_id)
        await logger.finish(child_id, output_summary="score=8.0")
        await logger.finish(root_id, output_summary="final ok")

        async with test_session_local() as session:
            rows = (await session.execute(select(ReasoningLog))).scalars().all()
        by_agent = {r.agent_name: r for r in rows}
        assert set(by_agent) == {"supervisor", "critic"}
        assert by_agent["critic"].parent_id == by_agent["supervisor"].id
        assert by_agent["critic"].request_id == "req-db"
        assert by_agent["critic"].status == "success"
        assert by_agent["supervisor"].started_at is not None
        assert by_agent["supervisor"].finished_at is not None


# ============================================================================
# Критерий улучшения: при равной оценке Critic решает метрика валидации
# ============================================================================
class TestImprovementByMetric:
    @staticmethod
    def _state(score=5.0, passed=True, mase=None, mae=None) -> ForecastState:
        state = ForecastState(series_id="s", horizon=7)
        state.critic_score = score
        state.validation_passed = passed
        state.metadata["val_metrics"] = {"mase": mase, "mae": mae}
        return state

    def test_lower_mase_is_improvement(self):
        prev = self._state(mase=1.10, mae=2.0)
        curr = self._state(mase=1.05, mae=2.0)  # хуже по MAE, лучше по MASE
        assert Supervisor._is_improvement(prev, curr) is True

    def test_higher_mase_is_regression(self):
        prev = self._state(mase=1.00, mae=1.0)
        curr = self._state(mase=1.20, mae=1.0)
        assert Supervisor._is_improvement(prev, curr) is False

    def test_mae_breaks_tie_when_mase_equal(self):
        prev = self._state(mase=0.95, mae=1.50)
        assert Supervisor._is_improvement(prev, self._state(mase=0.95, mae=1.40)) is True
        assert Supervisor._is_improvement(prev, self._state(mase=0.95, mae=1.60)) is False
        assert Supervisor._is_improvement(prev, self._state(mase=0.95, mae=1.50)) is False

    def test_no_metrics_means_no_improvement(self):
        prev = self._state(mase=None, mae=None)
        assert Supervisor._is_improvement(prev, self._state(mase=None, mae=None)) is False

    def test_broken_validation_is_regression_despite_metric(self):
        prev = self._state(passed=True, mase=1.2, mae=2.0)
        curr = self._state(passed=False, mase=0.5, mae=1.0)  # метрика лучше, валидация сломалась
        assert Supervisor._is_improvement(prev, curr) is False

    def test_critic_score_still_dominates(self):
        prev = self._state(score=5.0, mase=1.0, mae=1.0)
        assert Supervisor._is_improvement(prev, self._state(score=6.0, mase=2.0, mae=9.0)) is True
        assert Supervisor._is_improvement(prev, self._state(score=4.0, mase=0.1, mae=0.1)) is False
