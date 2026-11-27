"""Реестр инструментов (Tool Calling) — Этап 7.

ModelSelector НЕ генерирует сырой Python-код (антипаттерн из задания MLF):
LLM возвращает JSON {tool, arguments}, а инструмент выполняется реестром.

Минимум по заданию: fit_arima, fit_prophet, fit_xgboost + fit_tabular_xgboost
(демонстрация расширяемости на табличные задачи).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from backend.timeseries import classical
from backend.timeseries.preprocessing import prepare_timeseries

logger = logging.getLogger(__name__)


class UnknownToolError(KeyError):
    """Запрошен инструмент, которого нет в реестре."""


class ToolArgumentError(ValueError):
    """Аргументы вызова не прошли валидацию по схеме."""


@dataclass(frozen=True)
class ToolSpec:
    """Спецификация инструмента: описание + JSON-схема параметров + обработчик."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    """Хранилище инструментов; run() валидирует аргументы и вызывает обработчик."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name) from exc

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Схемы в формате OpenAI function calling (для промпта LLM)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in sorted(self._tools.values(), key=lambda s: s.name)
        ]

    def run(self, name: str, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Валидирует обязательные аргументы и выполняет инструмент."""
        spec = self.get(name)
        required = spec.parameters.get("required", [])
        missing = [key for key in required if key not in (arguments or {})]
        if missing:
            raise ToolArgumentError(f"{name}: не хватает аргументов {missing}")
        return spec.handler(arguments or {}, context)


# ============================================================================
# Обработчики: маппинг аргументов на classical.fit_*
# ============================================================================
def _prepared_series(context: dict[str, Any]):
    """Ряд из контекста (уже подготовленный DataAnalyst'ом) или из metadata."""
    if context.get("series") is not None:
        return context["series"]
    meta = context.get("metadata") or {}
    result = prepare_timeseries(
        meta["series_values"], dates=meta.get("series_dates"),
        target_freq=(meta.get("freq") or "D"),
    )
    return result["series"]


def _handle_fit_arima(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    series = _prepared_series(context)
    return classical.fit_arima(
        series,
        horizon=int(args.get("horizon") or context.get("horizon") or 14),
        order=args.get("order"),
        seasonal_order=args.get("seasonal_order"),
        season=context.get("season"),
    )


def _handle_fit_prophet(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    series = _prepared_series(context)
    return classical.fit_prophet(
        series,
        horizon=int(args.get("horizon") or context.get("horizon") or 14),
        seasonality_mode=args.get("seasonality_mode", "additive"),
        season=context.get("season"),
    )


def _handle_fit_xgboost(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    series = _prepared_series(context)
    return classical.fit_xgboost(
        series,
        horizon=int(args.get("horizon") or context.get("horizon") or 14),
        lags=int(args.get("lags", 7)),
        n_estimators=int(args.get("n_estimators", 200)),
        season=context.get("season"),
    )


def _handle_fit_tabular(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    frame = context.get("frame")
    target = context.get("target_column") or args.get("target_column")
    if frame is None or target is None:
        raise ToolArgumentError("fit_tabular_xgboost: нужны context['frame'] и target_column")
    return classical.fit_tabular_xgboost(
        frame,
        target_column=str(target),
        n_estimators=int(args.get("n_estimators", 200)),
        test_size=float(args.get("test_size", 0.2)),
    )


def build_default_registry() -> ToolRegistry:
    """Реестр по умолчанию (минимум трёх классических методов + tabular)."""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="fit_arima",
        description=(
            "ARIMA/SARIMA для временного ряда. Если order не задан — "
            "автоподбор по валидации (fallback-правило системы)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order": {
                    "type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3,
                    "description": "(p, d, q), например [1, 1, 1]",
                },
                "seasonal_order": {
                    "type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
                    "description": "(P, D, Q, s), например [1, 1, 1, 7] для недельной сезонности",
                },
            },
            "required": [],
        },
        handler=_handle_fit_arima,
    ))
    registry.register(ToolSpec(
        name="fit_prophet",
        description="Prophet: тренд + сезонность, устойчив к пропускам.",
        parameters={
            "type": "object",
            "properties": {
                "seasonality_mode": {"type": "string", "enum": ["additive", "multiplicative"]},
            },
            "required": [],
        },
        handler=_handle_fit_prophet,
    ))
    registry.register(ToolSpec(
        name="fit_xgboost",
        description="XGBoost на лаговых признаках с рекурсивным прогнозом.",
        parameters={
            "type": "object",
            "properties": {
                "lags": {"type": "integer", "minimum": 1, "maximum": 60},
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000},
            },
            "required": [],
        },
        handler=_handle_fit_xgboost,
    ))
    registry.register(ToolSpec(
        name="fit_tabular_xgboost",
        description="XGBoost для табличных задач (регрессия) — расширяемость архитектуры.",
        parameters={
            "type": "object",
            "properties": {
                "target_column": {"type": "string"},
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000},
                "test_size": {"type": "number", "minimum": 0.1, "maximum": 0.5},
            },
            "required": ["target_column"],
        },
        handler=_handle_fit_tabular,
    ))
    return registry
