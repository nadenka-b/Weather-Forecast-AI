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
from backend.core.llm.manager import LLMManager, ToolDecision
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


def _query_features(state: ForecastState) -> dict[str, Any]:
    """Признаки текущего ряда в формате кейсов БЗ (для векторного поиска)."""
    diagnostics = state.diagnostics or {}
    stats = diagnostics.get("stats", {})
    stl = diagnostics.get("stl") or {}
    return {
        "length": stats.get("length"),
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "min": stats.get("min"),
        "max": stats.get("max"),
        "acf_lag1": stats.get("acf_lag1"),
        "adf_pvalue": (diagnostics.get("adf") or {}).get("adf_pvalue"),
        "stationary": (diagnostics.get("adf") or {}).get("stationary"),
        "seasonal_period": diagnostics.get("seasonal_period"),
        "trend_strength": stl.get("trend_strength"),
        "seasonal_strength": stl.get("seasonal_strength"),
        "missing_rate": diagnostics.get("raw_missing_rate"),
        "variable": state.metadata.get("variable") or state.target_column,
        "city": state.metadata.get("city"),
    }


def _query_text(state: ForecastState) -> str:
    """Текстовое описание текущего ряда (для BM25), в стиле descriptions БЗ."""
    diagnostics = state.diagnostics or {}
    stl = diagnostics.get("stl") or {}
    variable = state.metadata.get("variable") or state.target_column or "ряд"
    city = state.metadata.get("city") or "город"
    period = diagnostics.get("seasonal_period")
    return (
        f"Дневной ряд {variable} ({city}), сезонность {period or 'не выражена'}, "
        f"тренд {stl.get('trend_strength', 0):.2f}."
    )


class ModelSelector(LLMAgent):
    """Выбирает модель и гиперпараметры через реестр инструментов.

    retriever (Этап 10): гибридный RAG-поиск похожих кейсов базы знаний;
    если не задан — промпт идёт без few-shot примеров.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        llm_manager: LLMManager | None = None,
        retriever: Any | None = None,
    ):
        super().__init__(
            name="model_selector",
            description="Tool Calling: выбор модели прогнозирования",
        )
        self.registry = registry or build_default_registry()
        self.manager = llm_manager or LLMManager()
        self.retriever = retriever

    async def _few_shot(self, state: ForecastState) -> list[str]:
        """Few-shot примеры: топ похожих кейсов из базы знаний (RAG)."""
        if self.retriever is None:
            return []
        return self.retriever.few_shot_prompts(
            _query_features(state), _query_text(state), top_k=3
        )

    async def process(self, state: ForecastState) -> ForecastState:
        if state.task_type == "tabular":
            return await self._process_tabular(state)
        return await self._process_timeseries(state)

    # ------------------------------------------------------------------
    # Временные ряды
    # ------------------------------------------------------------------
    async def _process_timeseries(self, state: ForecastState) -> ForecastState:
        eda = _eda_summary(state)
        forced = state.metadata.get("forced_model")
        if forced:
            # Решение человека (HITL): модель задана явно, LLM не спрашиваем
            decision = ToolDecision(
                tool=forced["tool"],
                arguments=forced.get("arguments") or {},
                attempts=0,
                used_fallback=False,
            )
        else:
            decision = await self.manager.select_tool(
                eda, self.registry, fewshot=await self._few_shot(state)
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
        if result.get("val_pred"):
            # даты валидационного отрезка — для климатических проверок Validator
            n_val = len(result["val_pred"])
            result["val_dates"] = [
                str(ts.date()) for ts in prepared["series"].index[-n_val:]
            ]
        return self._apply_decision(state, decision, result)

    # ------------------------------------------------------------------
    # Табличные данные
    # ------------------------------------------------------------------
    async def _process_tabular(self, state: ForecastState) -> ForecastState:
        records = load_records(state.metadata)
        prepared = prepare_tabular(records, target_column=state.target_column)
        decision = await self.manager.select_tool(
            _eda_summary(state), self.registry, fewshot=await self._few_shot(state)
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
        # валидационный прогноз/факт/даты — вход для Validator (Этап 8)
        state.metadata["val_pred"] = result.get("val_pred")
        state.metadata["val_true"] = result.get("val_true")
        state.metadata["val_dates"] = result.get("val_dates")
        state.metadata["selected_tool"] = decision.tool
        state.metadata["tool_attempts"] = decision.attempts
        state.metadata["used_fallback"] = decision.used_fallback
        # расход LLM — для агентных метрик run_eval (Этап 13)
        state.metadata["llm_usage"] = {
            "tokens_in": decision.tokens_in,
            "tokens_out": decision.tokens_out,
            "latency_ms": decision.latency_ms,
            "attempts": decision.attempts,
            "model": decision.model,
        }
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
