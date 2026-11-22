"""ModelSelector — выбор модели через Tool Calling (Этап 7).

Пайплайн агента:
1. EDA-сводка из state.diagnostics (результат DataAnalyst);
2. RAG-подсказка похожих кейсов (пока заглушка, реальный гибридный поиск — Этап 10);
3. LLMManager.select_tool: промпт со схемами инструментов -> JSON {tool, arguments};
4. ToolRegistry.run: обучение/валидация выбранной модели;
5. state.model_type / model_params / metadata (метрики, попытки, fallback).

Если LLM не дала валидный вызов за 3 попытки — LLMManager возвращает
fallback-правило (fit_arima с автоподбором порядка).
"""
import logging
from typing import Any

from backend.agents.base import LLMAgent
from backend.core.llm.manager import LLMManager
from backend.core.llm.tools import ToolRegistry, build_default_registry
from backend.core.models import ForecastState
from backend.timeseries.preprocessing import (
    load_records,
    load_series,
    prepare_tabular,
    prepare_timeseries,
)

logger = logging.getLogger(__name__)

# Какие ключи результата инструмента попадают в state.model_params
_PARAM_KEYS = (
    "order",
    "seasonal_order",
    "auto",
    "seasonality_mode",
    "lags",
    "n_estimators",
    "test_size",
    "aic",
)


def _eda_summary(state: ForecastState) -> str:
    """Компактная EDA-сводка для промпта."""
    d = state.diagnostics or {}
    stats = d.get("stats", {})
    lines = [
        f"task_type={state.task_type}",
        f"series_id={state.series_id}, horizon={state.horizon}, freq={state.freq}",
        f"n_observations={stats.get('length')}, missing_rate={d.get('raw_missing_rate')}",
        f"seasonal_period={d.get('seasonal_period')}",
        f"stationary={d.get('adf', {}).get('stationary')}",
    ]
    stl = d.get("stl") or {}
    if stl.get("seasonal_strength") is not None:
        lines.append(
            f"stl: trend={stl.get('trend_strength'):.2f}, "
            f"seasonal={stl.get('seasonal_strength'):.2f}"
        )
    if state.task_type == "tabular":
        lines.append(
            f"tabular: num_features={state.metadata.get('num_features')}, "
            f"target_column={state.target_column}"
        )
    return "\n".join(str(line) for line in lines)


def _rag_stub(state: ForecastState) -> list[str]:
    """Заглушка RAG до Этапа 10: пустой список похожих кейсов."""
    return []


class ModelSelector(LLMAgent):
    """Выбирает модель и гиперпараметры через реестр инструментов."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        llm_manager: LLMManager | None = None,
    ):
        super().__init__(
            name="model_selector",
            description="Tool Calling: выбор модели прогнозирования",
        )
        self.registry = registry or build_default_registry()
        self.manager = llm_manager or LLMManager()

    async def process(self, state: ForecastState) -> ForecastState:
        if state.task_type == "tabular":
            return await self._process_tabular(state)
        return await self._process_timeseries(state)

    # ------------------------------------------------------------------
    # Временные ряды
    # ------------------------------------------------------------------
    async def _process_timeseries(self, state: ForecastState) -> ForecastState:
        eda = _eda_summary(state)
        decision = await self.manager.select_tool(
            eda, self.registry, fewshot=_rag_stub(state)
        )

        values, dates = load_series(state.metadata)
        prepared = prepare_timeseries(
            values, dates=dates, target_freq=state.freq or "D"
        )
        context: dict[str, Any] = {
            "series": prepared["series"],
            "metadata": state.metadata,
            "horizon": state.horizon,
            "season": (state.diagnostics or {}).get("seasonal_period"),
        }

        result = self.registry.run(decision.tool, decision.arguments, context)
        return self._apply_decision(state, decision, result)

    # ------------------------------------------------------------------
    # Табличные данные
    # ------------------------------------------------------------------
    async def _process_tabular(self, state: ForecastState) -> ForecastState:
        records = load_records(state.metadata)
        prepared = prepare_tabular(records, target_column=state.target_column)
        decision = await self.manager.select_tool(
            _eda_summary(state), self.registry, fewshot=_rag_stub(state)
        )
        context = {
            "frame": prepared["frame"],
            "target_column": state.target_column,
        }
        result = self.registry.run(decision.tool, decision.arguments, context)
        return self._apply_decision(state, decision, result)

    # ------------------------------------------------------------------
    # Общее
    # ------------------------------------------------------------------
    def _apply_decision(
        self,
        state: ForecastState,
        decision: Any,
        result: dict[str, Any],
    ) -> ForecastState:
        state.model_type = result.get("model", decision.tool)
        state.model_params = {
            key: result[key] for key in _PARAM_KEYS if key in result
        }
        state.metadata["val_metrics"] = result.get("metrics")
        state.metadata["selected_tool"] = decision.tool
        state.metadata["tool_attempts"] = decision.attempts
        state.metadata["used_fallback"] = decision.used_fallback
        if decision.used_fallback:
            state.metadata.setdefault(
                "critic_suggestions",
                [{"type": "fallback_used", "detail": "LLM не дала валидный Tool Call"}],
            )

        self._last_result = {
            "tool": decision.tool,
            "attempts": decision.attempts,
            "used_fallback": decision.used_fallback,
            "metrics": result.get("metrics"),
        }
        logger.info(
            "ModelSelector: tool=%s attempts=%s fallback=%s mae=%s",
            decision.tool,
            decision.attempts,
            decision.used_fallback,
            (result.get("metrics") or {}).get("mae"),
        )
        return state
