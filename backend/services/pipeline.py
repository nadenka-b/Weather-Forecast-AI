"""Центральный модуль пайплайна прогноза (Этап 12) — по методичке Step_9 (services/pipeline).

Единственная точка сборки всех компонентов: perform_forecast — полный цикл
агентов (Supervisor) + финальный прогноз горизонта + строгая валидация.
Её вызывают роуты API (через задачу), Celery-воркер (forecast_task) и
run_eval (Этап 13). Логика не дублируется.

Жизненный цикл задачи (задание MLF):
    pending -> running -> waiting_for_human | completed | failed
State хранится по ссылке (task_id): снимок ForecastState в forecast_tasks.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.agents.supervisor import build_supervisor
from backend.agents.validator import Validator
from backend.core.config import get_settings
from backend.core.llm.tools import build_default_registry
from backend.core.metrics import naive_forecast, regression_metrics
from backend.core.models import ForecastState
from backend.data.tasks import TaskRepository
from backend.db.session import ensure_tables_once
from backend.timeseries.classical import predict_final, time_split
from backend.timeseries.preprocessing import load_series, prepare_timeseries

logger = logging.getLogger(__name__)


# ============================================================================
# Основной сценарий
# ============================================================================
async def perform_forecast(
    state: ForecastState, supervisor: Any | None = None
) -> ForecastState:
    """Полный цикл: агенты -> финальный прогноз -> валидация финального прогноза.

    supervisor можно передать извне (run_eval переиспользует один экземпляр
    с уже собранным RAG на серию прогонов вместо сборки на каждую задачу).
    """
    if supervisor is None:
        supervisor = await build_supervisor()
    state = await supervisor.process(state)

    if state.task_type == "timeseries":
        await finalize_forecast(state)
        # строгая проверка финального прогноза: длина == horizon, NaN, климат
        state = await Validator().process(state)
    return state


async def finalize_forecast(state: ForecastState) -> None:
    """Переобучает выбранную модель на всём ряду и строит прогноз на горизонт.

    Результат: metadata['predictions'], ['prediction_dates'] и CSV-файл
    data/forecasts/<task_id>.csv (путь — в state.predictions_path).
    """
    values, dates = load_series(state.metadata)
    freq = state.freq or "D"
    prepared = prepare_timeseries(values, dates=dates, target_freq=freq)
    series: pd.Series = prepared["series"]

    season = (state.diagnostics or {}).get("seasonal_period")

    # Кэш прогнозов (Этап 14): тот же ряд + горизонт + модель + параметры
    predictions: list[float] | None = None
    cache_key = None
    if get_settings().cache_enabled:
        from backend.data.cache import cache_manager

        cache_key = cache_manager.make_key(
            "forecast", [round(float(v), 6) for v in series.values], state.horizon,
            state.model_type, state.model_params, season,
        )
        cached = await cache_manager.get(cache_key, model_used="forecast")
        if cached:
            try:
                candidate = json.loads(cached)
                if isinstance(candidate, list) and len(candidate) == state.horizon:
                    predictions = candidate
                    logger.debug("Forecast cache hit for task %s", state.task_id)
            except (json.JSONDecodeError, TypeError):
                predictions = None

    if predictions is None:
        predictions = predict_final(
            series, state.horizon, state.model_type, state.model_params, season
        )
        if cache_key is not None:
            await cache_manager.set(cache_key, json.dumps(predictions), model_used="forecast")
    offset = pd.tseries.frequencies.to_offset(freq)
    future_dates = pd.date_range(
        series.index[-1] + offset, periods=state.horizon, freq=freq
    )

    forecasts_dir = Path(get_settings().forecasts_dir)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    path = forecasts_dir / f"{state.task_id or state.log_id or 'forecast'}.csv"
    pd.DataFrame(
        {
            "date": future_dates.strftime("%Y-%m-%d"),
            "prediction": predictions,
            "model": state.model_type,
        }
    ).to_csv(path, index=False)

    state.predictions_path = str(path)
    state.metadata["predictions"] = predictions
    state.metadata["prediction_dates"] = [str(d.date()) for d in future_dates]

    # Наивный сезонный прогноз на том же контрольном отрезке, что и метрики
    # модели (val_metrics из fit_*): сравнение «модель vs наивный» (задание
    # MLF: результат с метриками и сравнением с наивным прогнозом) — для UI/отчёта
    train, val = time_split(series, state.horizon)
    naive_pred = naive_forecast(train.values, len(val), season or 1)
    state.metadata["naive_metrics"] = regression_metrics(
        val.values, naive_pred, train.values, season=season or 1
    )


def build_result(state: ForecastState) -> dict[str, Any]:
    """Словарь результата для GET /result."""
    return {
        "model_type": state.model_type,
        "model_params": state.model_params,
        "metrics": state.metadata.get("val_metrics"),
        "naive_metrics": state.metadata.get("naive_metrics"),
        "critic_score": state.critic_score,
        "critic_reasoning": state.critic_reasoning,
        "validation_passed": state.validation_passed,
        "validation": (state.diagnostics or {}).get("validation"),
        "predictions_path": state.predictions_path,
        "predictions": state.metadata.get("predictions"),
        "prediction_dates": state.metadata.get("prediction_dates"),
        "hitl_requested": state.hitl_requested,
        "supervisor": state.metadata.get("supervisor"),
        "critic_suggestions": state.metadata.get("critic_suggestions") or [],
        "used_fallback": state.metadata.get("used_fallback"),
    }


# ============================================================================
# Задача: запуск по task_id
# ============================================================================
async def run_forecast(task_id: str) -> ForecastState | None:
    """Выполняет задачу прогноза (вызывается inline-раннером или Celery-воркером).

    Ошибки не пробрасывает: задача переводится в failed с текстом ошибки,
    чтобы фоновый раннер не падал.
    """
    repo = TaskRepository()
    await ensure_tables_once()  # воркер/скрипт могли стартовать раньше обновления схемы
    task = await repo.get(task_id)
    if task is None:
        raise LookupError(f"Задача {task_id} не найдена")

    await repo.set_status(task_id, "running")
    state = ForecastState(**task["state"])
    try:
        state = await perform_forecast(state)
    except Exception as exc:  # noqa: BLE001 — любая ошибка = failed
        logger.exception("Forecast task %s failed", task_id)
        await repo.set_status(task_id, "failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        return None

    status = "waiting_for_human" if state.hitl_requested else "completed"
    await repo.save_state(task_id, state, build_result(state), status=status)
    logger.info("Forecast task %s -> %s", task_id, status)
    return state


# ============================================================================
# Human-in-the-Loop: решение пользователя (Этап 12/14)
# ============================================================================
_RESET_METADATA_KEYS = (
    "critic_suggestions", "critic_adequacy", "critic_used_fallback",
    "validator_errors", "validator_warnings", "agent_errors",
    "supervisor", "reasoning_trace", "predictions", "prediction_dates",
    "val_pred", "val_true", "val_dates", "val_metrics", "naive_metrics",
    "selected_tool", "tool_attempts", "used_fallback",
)


async def apply_hitl_resolution(task_id: str, resolution: dict[str, Any]) -> dict[str, Any]:
    """Применяет решение человека; возвращает обновлённую задачу.

    - accept — принять текущий результат (completed);
    - change_model — выбрать другую модель (перезапуск с forced_model);
    - change_horizon — изменить горизонт (перезапуск).
    Перезапуск ставит задачу в pending — роут отправляет её в очередь.
    """
    repo = TaskRepository()
    task = await repo.get(task_id)
    if task is None:
        raise LookupError(f"Задача {task_id} не найдена")
    if task["status"] != "waiting_for_human":
        raise ValueError(f"Задача {task_id} не ожидает решения человека (статус {task['status']})")

    state = ForecastState(**task["state"])
    state.hitl_resolution = resolution
    action = resolution.get("action")

    if action == "accept":
        await repo.save_state(task_id, state, status="completed")
        return await repo.get(task_id)  # type: ignore[return-value]

    if action == "change_model":
        tool = resolution.get("model_type") or ""
        registry = build_default_registry()
        if tool not in registry.names:
            raise ValueError(f"Неизвестная модель {tool!r}; доступны: {registry.names}")
        state.metadata["forced_model"] = {
            "tool": tool,
            "arguments": resolution.get("model_params") or {},
        }
        state.model_type = None
        state.model_params = None
    elif action == "change_horizon":
        horizon = resolution.get("horizon")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Для change_horizon нужен horizon >= 1")
        state.horizon = horizon
    else:
        raise ValueError(f"Неизвестное действие {action!r}: accept | change_model | change_horizon")

    # сброс результатов предыдущего прогона
    state.hitl_requested = False
    state.critic_score = None
    state.critic_reasoning = None
    state.validation_passed = False
    state.iteration = 0
    state.predictions_path = None
    for key in _RESET_METADATA_KEYS:
        state.metadata.pop(key, None)

    await repo.save_state(task_id, state, status="pending")
    return await repo.get(task_id)  # type: ignore[return-value]
