"""Фоновые задачи и раннеры (Этап 12).

Два бэкенда по принципу задания MLF «Local Dev / Production»:

- InlineTaskRunner — задача выполняется в процессе API: сразу (eager, тесты
  и отладка) или как asyncio-задача. Не требует Redis — локальная разработка.
- CeleryTaskRunner — задача уходит в очередь Redis, её выполняет воркер
  backend.worker.celery_app (продакшен, docker-compose).

Ловушка «Future attached to a different loop» (методичка Step_3/Step_5):
в Celery-воркере нельзя создавать новый event loop на каждую задачу —
async engine привязан к loop'у. Поэтому один постоянный loop на процесс
в отдельном потоке, задачи выполняются через run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Coroutine, Protocol

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

FORECAST_TASK_NAME = "forecast.forecast_task"


class TaskRunner(Protocol):
    """Контракт раннера: поставить задачу на выполнение."""

    async def enqueue(self, task_id: str) -> None: ...


class InlineTaskRunner:
    """Выполнение в процессе API (локальный режим без Redis)."""

    def __init__(self, eager: bool | None = None):
        self.eager = eager if eager is not None else get_settings().task_eager
        self._background: set[asyncio.Task] = set()

    async def enqueue(self, task_id: str) -> None:
        from backend.services.pipeline import run_forecast

        if self.eager:
            await run_forecast(task_id)
            return
        task = asyncio.create_task(self._run(task_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _run(self, task_id: str) -> None:
        from backend.services.pipeline import run_forecast

        try:
            await run_forecast(task_id)
        except Exception:  # noqa: BLE001 — фоновая задача не должна ронять loop
            logger.exception("Background forecast task %s crashed", task_id)


class CeleryTaskRunner:
    """Отправка задачи в очередь (выполняет Celery-воркер)."""

    async def enqueue(self, task_id: str) -> None:
        from backend.worker.celery_app import celery_app

        celery_app.send_task(FORECAST_TASK_NAME, args=[task_id], queue="forecast")
        logger.info("Task %s sent to Celery queue 'forecast'", task_id)


def get_task_runner() -> TaskRunner:
    """Раннер по настройке TASK_BACKEND: inline (по умолчанию) | celery."""
    if get_settings().task_backend.lower() == "celery":
        return CeleryTaskRunner()
    return InlineTaskRunner()


# ============================================================================
# Celery: постоянный event loop на процесс воркера
# ============================================================================
_persistent_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def get_persistent_loop() -> asyncio.AbstractEventLoop:
    """Один event loop на процесс (в отдельном потоке) — для Celery-воркера."""
    global _persistent_loop, _loop_thread
    with _loop_lock:
        if _persistent_loop is None or _persistent_loop.is_closed():
            _persistent_loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_persistent_loop.run_forever, name="weather-async-loop", daemon=True
            )
            _loop_thread.start()
        return _persistent_loop


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Выполняет корутину в постоянном loop'е и ждёт результат (синхронно)."""
    future = asyncio.run_coroutine_threadsafe(coro, get_persistent_loop())
    return future.result()


def forecast_task(task_id: str) -> Any:
    """Тело Celery-задачи: выполнить прогноз по task_id (регистрируется в celery_app)."""
    from backend.services.pipeline import run_forecast

    return run_async(run_forecast(task_id))
