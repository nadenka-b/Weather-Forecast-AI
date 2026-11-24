"""Pydantic-схемы запросов/ответов API (Этап 12).

Эндпоинты задания MLF:
    POST /api/v1/forecast/                  -> 202 + task_id
    GET  /api/v1/forecast/{task_id}/status  -> pending|running|waiting_for_human|completed|failed
    GET  /api/v1/forecast/{task_id}/result
    GET  /api/v1/forecast/{task_id}/reasoning
    POST /api/v1/forecast/{task_id}/hitl
"""
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskStatus = Literal["pending", "running", "waiting_for_human", "completed", "failed"]


class ForecastCreateJson(BaseModel):
    """Запуск прогноза по ряду в теле запроса (JSON)."""

    series_values: list[float] = Field(..., min_length=30, description="Значения ряда")
    series_dates: list[str] | None = Field(
        None, description="Даты (ISO); без них — ежедневный ряд с 2000-01-01"
    )
    horizon: int = Field(14, ge=1, le=60, description="Горизонт, дней (7–14 по теме)")
    freq: str = Field("D", description="Частота ряда (pandas: D, h, ...)")
    task_type: Literal["timeseries", "tabular"] = "timeseries"
    target_column: str | None = None
    city: str | None = Field(None, description="Город (для климатических проверок)")
    variable: str | None = Field(
        None, description="Переменная: temperature, humidity, pressure, wind_speed, precipitation"
    )
    series_id: str | None = None


class ForecastAccepted(BaseModel):
    """Ответ на создание задачи (HTTP 202)."""

    task_id: str
    status: TaskStatus
    status_url: str


class ForecastStatus(BaseModel):
    task_id: str
    status: TaskStatus
    hitl_requested: bool = False
    iterations: int = 0
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ForecastResult(BaseModel):
    """Результат прогноза: модель, метрики, прогноз, путь к файлу."""

    task_id: str
    status: TaskStatus
    model_type: str | None = None
    model_params: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    critic_score: float | None = None
    critic_reasoning: str | None = None
    validation_passed: bool = False
    validation: dict[str, Any] | None = None
    predictions_path: str | None = None
    predictions: list[float] | None = None
    prediction_dates: list[str] | None = None
    hitl_requested: bool = False
    supervisor: dict[str, Any] | None = None
    critic_suggestions: list[Any] = Field(default_factory=list)


class ReasoningResponse(BaseModel):
    """Дерево рассуждений агентов + трассировка вызовов LLM."""

    task_id: str
    tree: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]


class HitlRequest(BaseModel):
    """Решение человека для задачи в статусе waiting_for_human."""

    action: Literal["accept", "change_model", "change_horizon"]
    model_type: str | None = Field(
        None, description="Для change_model: fit_arima | fit_prophet | fit_xgboost"
    )
    model_params: dict[str, Any] | None = Field(None, description="Гиперпараметры новой модели")
    horizon: int | None = Field(None, ge=1, le=60, description="Для change_horizon")
    comment: str | None = None


# ============================================================================
# Оценка качества (Этап 13)
# ============================================================================
EvalMode = Literal["baseline", "rag", "agents", "foundation"]


class EvaluateBatchRequest(BaseModel):
    """Параметры пакетного запуска оценки (POST /api/v1/evaluate/batch)."""

    modes: list[EvalMode] = Field(default_factory=lambda: ["baseline", "rag", "agents"])
    horizons: list[int] = Field(default_factory=lambda: [14], min_length=1)
    data_fractions: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    num_series: int = Field(10, ge=1, le=50, description="Число тестовых окон (20/50/100)")
    fast: bool = Field(True, description="Сокращённая сетка baseline (naive + ARIMA)")
    seed: int = 42


class EvaluationRunResponse(BaseModel):
    """Запуск оценки: статус, параметры, сводка, сравнения."""

    run_id: str
    status: str
    params: dict[str, Any] | None = None
    summary: list[dict[str, Any]] | None = None
    comparisons: list[dict[str, Any]] | None = None
    n_records: int | None = None
    error: str | None = None


class EvaluationAccepted(BaseModel):
    run_id: str
    status: str
    status_url: str


# ============================================================================
# HITL и Active Learning (Этап 14)
# ============================================================================
class ConfirmRequest(BaseModel):
    """Подтверждение прогноза из очереди (опционально — с правкой модели)."""

    model_type: str | None = Field(
        None, description="Правка эксперта: fit_arima | fit_prophet | fit_xgboost"
    )
    model_params: dict[str, Any] | None = None
    comment: str | None = None


class RejectRequest(BaseModel):
    reason: str | None = None


class CorrectionRequest(BaseModel):
    """Сохранить правку/комментарий эксперта без смены статуса задачи."""

    task_id: str
    correction: dict[str, Any] | None = None
    comment: str | None = None
    add_to_kb: bool = Field(False, description="Добавить кейс задачи в базу знаний")
