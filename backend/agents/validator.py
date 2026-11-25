"""Validator — формальная проверка прогноза (Этап 8).

Rule-based агент (без LLM). Проверяет (задание MLF + тема проекта):

1. Прогноз без NaN и имеет ожидаемую длину;
2. Физическая реалистичность: значения не выходят за климатические пределы
   (помесячные нормы города/месяца от DataAnalyst, с допуском);
3. Статистические тесты остатков: Ljung-Box (автокорреляция — ошибка модели)
   и Shapiro-Wilk (нормальность — предупреждение);
4. Для task_type="tabular": форма предсказаний и конечность метрик.

Пока валидируется валидационный отрезок (val_pred/val_true от ModelSelector);
финальный прогноз горизонта появится с фоновыми задачами (Этап 12) — формат
проверок тот же (metadata["predictions"]).
"""
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from backend.agents.base import BaseAgent
from backend.core.metrics import ljung_box_pvalue, shapiro_pvalue
from backend.core.models import ForecastState

logger = logging.getLogger(__name__)

LJUNG_BOX_ALPHA = 0.05
SHAPIRO_ALPHA = 0.05


class Validator(BaseAgent):
    """Формальная валидация: NaN, климат, остатки."""

    def __init__(self):
        super().__init__(name="validator", description="Формальные проверки прогноза")

    @staticmethod
    def _all_finite(values: list) -> bool:
        """Все значения — конечные числа (None/NaN недопустимы)."""
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(array).all())

    async def process(self, state: ForecastState) -> ForecastState:
        if state.task_type == "tabular":
            return self._validate_tabular(state)
        return self._validate_timeseries(state)

    # ------------------------------------------------------------------
    # Временные ряды
    # ------------------------------------------------------------------
    def _validate_timeseries(self, state: ForecastState) -> ForecastState:
        checks: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        meta = state.metadata

        is_final = meta.get("predictions") is not None
        predictions = meta.get("predictions") or meta.get("val_pred")
        actuals = meta.get("val_true")
        # даты: для финального прогноза — будущие даты горизонта, для валидации — val_dates
        dates = meta.get("prediction_dates") if is_final else meta.get("val_dates")

        # --- 1. NaN и длина
        if not predictions:
            checks["predictions"] = "skipped: нет прогноза (ранний этап пайплайна)"
        else:
            # финальный прогноз горизонта (Этап 12) — строгая длина;
            # валидационный отрезок — достаточна одна точка
            checks["predictions"] = self._check_predictions(
                predictions,
                state.horizon,
                errors,
                expected_len=state.horizon if is_final else None,
            )

        # --- 2. Климатические пределы
        bounds = meta.get("climate_bounds")
        if predictions and bounds:
            checks["climate"] = self._check_climate(
                predictions, dates, bounds, errors
            )
        elif predictions:
            checks["climate"] = "skipped: нет climate_bounds в metadata"

        # --- 3. Тесты остатков
        if (
            predictions
            and actuals
            and len(predictions) == len(actuals)
            and self._all_finite(predictions)
            and self._all_finite(actuals)
        ):
            checks["residuals"] = self._check_residuals(
                predictions, actuals, errors, warnings
            )

        state.validation_passed = not errors
        state.diagnostics = state.diagnostics or {}
        state.diagnostics["validation"] = {
            "passed": state.validation_passed,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }
        state.metadata["validator_errors"] = errors
        state.metadata["validator_warnings"] = warnings

        self._last_result = {"passed": state.validation_passed, "errors": errors}
        logger.info(
            "Validator: passed=%s errors=%d warnings=%d",
            state.validation_passed,
            len(errors),
            len(warnings),
        )
        return state

    @staticmethod
    def _check_predictions(
        predictions: list[float],
        horizon: int,
        errors: list[str],
        expected_len: int | None = None,
    ) -> dict[str, Any]:
        """NaN — всегда ошибка; длина: для финального прогноза строго == horizon,
        для валидационного отрезка — не меньше одной точки."""
        nan_count = sum(1 for v in predictions if v is None or math.isnan(float(v)))
        if nan_count:
            errors.append(f"Прогноз содержит {nan_count} NaN")
        if expected_len is not None and len(predictions) != expected_len:
            errors.append(
                f"Длина финального прогноза {len(predictions)} != горизонту {expected_len}"
            )
        elif len(predictions) < 1:
            errors.append("Прогноз пуст")
        return {"length": len(predictions), "nan_count": nan_count}

    @staticmethod
    def _check_climate(
        predictions: list[float],
        dates: list | None,
        bounds: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any]:
        """Прогноз в климатических пределах: помесячно, иначе глобально."""
        monthly = bounds.get("monthly") or {}
        global_bounds = bounds.get("global")
        violations: list[dict[str, Any]] = []

        parsed_months: list[int | None] = [None] * len(predictions)
        if dates:
            try:
                parsed_months = [pd.Timestamp(d).month for d in dates[: len(predictions)]]
            except (ValueError, TypeError):
                parsed_months = [None] * len(predictions)

        for i, value in enumerate(predictions):
            month = parsed_months[i] if i < len(parsed_months) else None
            try:
                number = float(value)
            except (TypeError, ValueError):
                violations.append({"index": i, "value": None, "month": month})
                continue
            window = monthly.get(str(month)) if month else None
            if window is None:
                window = global_bounds
            if window is None:
                continue
            lo, hi = float(window[0]), float(window[1])
            if not (lo <= number <= hi):
                violations.append(
                    {
                        "index": i,
                        "value": round(number, 2),
                        "bounds": [lo, hi],
                        "month": month,
                    }
                )

        if violations:
            worst = violations[0]
            bounds = worst.get("bounds")
            window_text = (
                f"вне [{bounds[0]}, {bounds[1]}]" if bounds else "(не число)"
            )
            errors.append(
                f"Климатические пределы нарушены ({len(violations)} точек): "
                f"например {worst['value']} {window_text}"
            )
        return {"violations": violations, "checked": len(predictions)}

    @staticmethod
    def _check_residuals(
        predictions: list[float],
        actuals: list[float],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any]:
        residuals = np.asarray(actuals, dtype=float) - np.asarray(
            predictions, dtype=float
        )
        result: dict[str, Any] = {
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
        }

        # Ljung-Box: автокорреляция остатков означает непойманную структуру
        lb_pvalue = ljung_box_pvalue(residuals)
        if lb_pvalue is not None:
            result["ljung_box_pvalue"] = round(lb_pvalue, 4)
            if lb_pvalue < LJUNG_BOX_ALPHA:
                errors.append(
                    f"Ljung-Box: остатки автокоррелированы (p={lb_pvalue:.3f})"
                )

        # Shapiro-Wilk: нормальность остатков (менее критична -> warning)
        sh_pvalue = shapiro_pvalue(residuals)
        if sh_pvalue is not None:
            result["shapiro_pvalue"] = round(sh_pvalue, 4)
            if sh_pvalue < SHAPIRO_ALPHA:
                warnings.append(
                    f"Shapiro-Wilk: остатки не нормальны (p={sh_pvalue:.3f})"
                )
        return result

    # ------------------------------------------------------------------
    # Табличные данные
    # ------------------------------------------------------------------
    def _validate_tabular(self, state: ForecastState) -> ForecastState:
        checks: dict[str, Any] = {}
        errors: list[str] = []
        metrics = state.metadata.get("val_metrics")

        if metrics is None:
            checks["metrics"] = "skipped: нет метрик (ранний этап пайплайна)"
        else:
            for key in ("mae", "rmse"):
                value = metrics.get(key)
                if value is None or not math.isfinite(float(value)):
                    errors.append(f"Метрика {key} отсутствует или не конечна")
            checks["metrics"] = {k: metrics.get(k) for k in ("mae", "rmse", "wape")}
            if not state.metadata.get("num_features"):
                errors.append("Нет признаков в metadata['num_features']")

        state.validation_passed = not errors
        state.diagnostics = state.diagnostics or {}
        state.diagnostics["validation"] = {
            "passed": state.validation_passed,
            "errors": errors,
            "warnings": [],
            "checks": checks,
        }
        state.metadata["validator_errors"] = errors
        self._last_result = {"passed": state.validation_passed, "errors": errors}
        return state
