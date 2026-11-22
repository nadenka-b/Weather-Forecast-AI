"""Юнит-тесты агента ModelSelector (Этап 7) и интеграция с Supervisor."""
import numpy as np
import pandas as pd

from backend.agents.data_analyst import DataAnalyst
from backend.agents.model_selector import ModelSelector
from backend.agents.supervisor import Supervisor
from backend.core.llm.manager import LLMManager
from backend.core.models import ForecastState


def weekly_state(n: int = 220) -> ForecastState:
    rng = np.random.default_rng(3)
    t = np.arange(n)
    values = 12 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.4, n)
    return ForecastState(
        series_id="climate_weekly",
        horizon=7,
        freq="D",
        metadata={"series_values": values.tolist()},
    )


class ValidJSONLLM:
    """Возвращает заданный валидный Tool Call."""

    def __init__(self, payload: str):
        self.payload = payload

    async def __call__(self, messages):
        return self.payload


class GarbageLLM:
    """Имитация слабой модели: всегда невалидный ответ."""

    async def __call__(self, messages):
        return "э... кажется, лучше спросить кого-то ещё"


ARIMA_CALL = '{"tool": "fit_arima", "arguments": {"order": [1, 1, 1]}}'
XGB_CALL = '{"tool": "fit_xgboost", "arguments": {"lags": 7, "n_estimators": 50}}'


# ============================================================================
# Ветка timeseries
# ============================================================================
class TestModelSelectorTimeseries:
    async def test_explicit_tool_call(self):
        selector = ModelSelector(
            llm_manager=LLMManager(llm_call=ValidJSONLLM(ARIMA_CALL))
        )
        state = await selector.process(weekly_state())

        assert state.model_type == "arima"
        assert state.model_params["order"] == [1, 1, 1]
        assert state.metadata["selected_tool"] == "fit_arima"
        assert state.metadata["tool_attempts"] == 1
        assert state.metadata["used_fallback"] is False
        assert np.isfinite(state.metadata["val_metrics"]["mae"])

    async def test_xgboost_tool_call(self):
        selector = ModelSelector(llm_manager=LLMManager(llm_call=ValidJSONLLM(XGB_CALL)))
        state = await selector.process(weekly_state())
        assert state.model_type == "xgboost"
        assert state.model_params["lags"] == 7

    async def test_fallback_to_sarima_auto(self):
        selector = ModelSelector(llm_manager=LLMManager(llm_call=GarbageLLM()))
        state = await selector.process(weekly_state())

        assert state.metadata["used_fallback"] is True
        assert state.metadata["tool_attempts"] == 3
        assert state.model_type in ("arima", "sarima")
        assert state.model_params.get("auto") is True
        # fallback фиксируется как замечание для Critic/HITL
        assert state.metadata["critic_suggestions"]


# ============================================================================
# Ветка tabular
# ============================================================================
class TestModelSelectorTabular:
    async def test_tabular_selection(self):
        rng = np.random.default_rng(8)
        rows = [
            {
                "a": float(a),
                "b": float(b),
                "city": "jena" if i % 2 else "delhi",
                "temp": float(a * 2 + b + 10),
            }
            for i, (a, b) in enumerate(rng.normal(size=(120, 2)))
        ]
        state = ForecastState(
            series_id="tabular_sample",
            horizon=0,
            task_type="tabular",
            target_column="temp",
            metadata={"records": rows},
        )
        selector = ModelSelector(
            llm_manager=LLMManager(
                llm_call=ValidJSONLLM(
                    '{"tool": "fit_tabular_xgboost", "arguments": {"target_column": "temp"}}'
                )
            )
        )
        state = await selector.process(state)
        assert state.model_type == "xgboost_tabular"
        assert np.isfinite(state.metadata["val_metrics"]["mae"])


# ============================================================================
# Интеграция: DataAnalyst + ModelSelector в Supervisor
# ============================================================================
class TestSelectorInPipeline:
    async def test_full_pipeline_converged(self):
        from tests.unit.test_supervisor import ScriptedCritic, ScriptedValidator

        sup = Supervisor(
            agents={
                "data_analyst": DataAnalyst(),
                "model_selector": ModelSelector(
                    llm_manager=LLMManager(llm_call=ValidJSONLLM(ARIMA_CALL))
                ),
                "validator": ScriptedValidator(passed=True),
                "critic": ScriptedCritic([8.0]),
            }
        )
        result = await sup.process(weekly_state())

        assert result.model_type == "arima"
        assert result.metadata["supervisor"]["stopped_reason"] == "converged"
        statuses = {
            e["agent_name"]: e["status"] for e in result.metadata["reasoning_trace"]
        }
        assert statuses["data_analyst"] == "success"
        assert statuses["model_selector"] == "success"
