"""Pydantic-модели состояния пайплайна (State).

State передаётся между агентами ТОЛЬКО по ссылке (ID задачи в БД) —
передача тяжёлых объектов (DataFrame) в payload запрещена (задание MLF).

Погодная специфика хранится в metadata:
  city              — город прогноза;
  target_variable   — "mean_temp" | "precipitation";
  climate_bounds    — климатические пределы для Validator (по месяцу/городу);
  extreme_flags     — признаки экстремальных условий (HITL-триггер).
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ForecastState(BaseModel):
    """Состояние задачи прогноза, которым обмениваются агенты (Этапы 5–14)."""

    task_id: Optional[str] = None
    series_id: str
    horizon: int
    freq: Optional[str] = None
    # Расширяемость на табличные данные (регрессия/классификация)
    task_type: Literal["timeseries", "tabular"] = "timeseries"
    metadata: dict[str, Any] = Field(default_factory=dict)
    target_column: Optional[str] = None

    # Выбор модели (ModelSelector, Tool Calling)
    model_type: Optional[str] = None  # "arima" | "sarima" | "prophet" | "xgboost" | ...
    model_params: Optional[dict] = None  # {"order": [1, 1, 1]}, {"seasonality_mode": "additive"}, ...

    # Результаты
    predictions_path: Optional[str] = None
    metrics: Optional[dict] = None
    diagnostics: Optional[dict] = None  # ACF/PACF, STL, тесты остатков

    # Контроль качества
    validation_passed: bool = False
    critic_score: Optional[float] = None
    critic_reasoning: Optional[str] = None
    editor_changed: bool = False

    # Цикл Supervisor
    iteration: int = 0
    max_iterations: int = 3
    log_id: Optional[str] = None  # ссылка на дерево reasoning_logs

    # Human-in-the-Loop
    hitl_requested: bool = False
    hitl_resolution: Optional[dict] = None  # решение пользователя (смена модели, горизонт, accept)
