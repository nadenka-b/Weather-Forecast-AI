"""Юнит-тесты базового агента и логгера рассуждений (Этап 5)."""
from backend.agents.base import BaseAgent, LLMAgent
from backend.agents.reasoning import InMemoryLogStore, ReasoningLogger, new_request_id
from backend.core.models import ForecastState


# ============================================================================
# BaseAgent / LLMAgent
# ============================================================================
class TestBaseAgent:
    def test_process_is_abstract(self):
        assert BaseAgent.__abstractmethods__ == frozenset({"process"})

    def test_cannot_instantiate_abstract(self):
        import pytest

        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_concrete_subclass(self):
        class Dummy(BaseAgent):
            async def process(self, state: ForecastState) -> ForecastState:
                self._last_result = "ok"
                return state

        dummy = Dummy()
        assert dummy.name == "Dummy"
        state = ForecastState(series_id="jena_temp", horizon=7)
        import asyncio

        result = asyncio.run(dummy.process(state))
        assert result.series_id == "jena_temp"
        assert dummy.get_last_result() == "ok"


class TestLLMAgent:
    def test_defaults_deterministic(self):
        class Critic(LLMAgent):
            async def process(self, state: ForecastState) -> ForecastState:
                return state

        critic = Critic(name="critic")
        assert critic.temperature == 0.0  # детерминированность по умолчанию
        assert critic.model_name is None


# ============================================================================
# ReasoningLogger + InMemoryLogStore
# ============================================================================
class TestReasoningLogger:
    def test_hierarchy_parent_child(self):
        import asyncio

        store = InMemoryLogStore()
        logger = ReasoningLogger(request_id="req-1", store=store)

        async def run():
            root = await logger.start("supervisor")
            child = await logger.start("data_analyst", parent_id=root)
            await logger.finish(child, output_summary="eda done")
            grand = await logger.start("model_selector", parent_id=child, tool_name="fit_sarima")
            await logger.finish(grand, output_summary="sarima chosen")
            await logger.finish(root, output_summary="final ok")
            return root, child

        root, child = asyncio.run(run())

        assert len(store.entries) == 3
        entries = {e["agent_name"]: e for e in store.entries}
        assert entries["supervisor"]["parent_id"] is None
        assert entries["data_analyst"]["parent_id"] == root
        assert entries["model_selector"]["parent_id"] == child
        assert entries["model_selector"]["tool_name"] == "fit_sarima"

        # все шаги закрыты
        assert all(e["status"] == "success" for e in store.entries)
        assert all(e["finished_at"] is not None for e in store.entries)

    def test_tree_structure(self):
        import asyncio

        store = InMemoryLogStore()
        logger = ReasoningLogger(request_id="req-2", store=store)

        async def run():
            root = await logger.start("supervisor")
            a = await logger.start("data_analyst", parent_id=root)
            await logger.finish(a)
            b = await logger.start("critic", parent_id=root)
            await logger.finish(b)
            await logger.finish(root)

        asyncio.run(run())

        tree = store.tree()
        assert len(tree) == 1
        assert tree[0]["agent_name"] == "supervisor"
        assert {c["agent_name"] for c in tree[0]["children"]} == {"data_analyst", "critic"}

    def test_update_unknown_id_does_not_raise(self):
        import asyncio

        store = InMemoryLogStore()
        logger = ReasoningLogger(request_id="req-3", store=store)
        asyncio.run(logger.finish("missing-id"))

    def test_new_request_id_unique(self):
        assert new_request_id() != new_request_id()

    def test_error_status(self):
        import asyncio

        store = InMemoryLogStore()
        logger = ReasoningLogger(request_id="req-4", store=store)

        async def run():
            step = await logger.start("editor")
            await logger.finish(step, output_summary="boom", status="error")

        asyncio.run(run())
        assert store.entries[0]["status"] == "error"
