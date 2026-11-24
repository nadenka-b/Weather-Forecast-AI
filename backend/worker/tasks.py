"""Фоновые задачи и раннеры (Этапы 12–14).

Два бэкенда по принципу задания MLF «Local Dev / Production»:

- InlineTaskRunner — задача выполняется в процессе API: сразу (eager, тесты
  и отладка) или как asyncio-задача. Не требует Redis — локальная разработка.
- CeleryTaskRunner — задача уходит в очередь Redis, её выполняет воркер
  backend.worker.celery_app (продакшен, docker-compose).

Ловушка «Future attached to a different loop» (методичка Step_3/Step_5):
в Celery-воркере нельзя создавать новый event loop на каждую задачу —
async engine привязан к loop'у. Поэтому один постоянный loop на процесс
в отдельном потоке, задачи выполняются через run_coroutine_threadsafe.

Задачи: forecast_task, evaluate_model_task, reindex_rag_task (после
подтверждения HITL), clear_cache_task (по расписанию beat — очистка кэша).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Coroutine, Protocol

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

FORECAST_TASK_NAME = "forecast.forecast_task"
EVALUATE_TASK_NAME = "evaluate.evaluate_model_task"
REINDEX_TASK_NAME = "rag.reindex_rag_task"
CLEAR_CACHE_TASK_NAME = "cache.clear_cache_task"


class TaskRunner(Protocol):
    """Контракт раннера: поставить задачу на выполнение."""

    async def enqueue(self, task_id: str) -> None: ...

    async def enqueue_eval(self, run_id: str) -> None: ...

    async def enqueue_reindex(self) -> None: ...

    async def enqueue_clear_cache(
        self, older_than_days: int = 30, max_hits: int | None = None
    ) -> None: ...


class InlineTaskRunner:
    """Выполнение в процессе API (локальный режим без Redis)."""

    def __init__(self, eager: bool | None = None):
        self.eager = eager if eager is not None else get_settings().task_eager
        self._background: set[asyncio.Task] = set()

    def _spawn(self, coro: Coroutine[Any, Any, Any], label: str) -> asyncio.Task:
        task = asyncio.create_task(self._guard(coro, label))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    async def _guard(self, coro: Coroutine[Any, Any, Any], label: str) -> None:
        try:
            await coro
        except Exception:  # noqa: BLE001 — фоновая задача не должна ронять loop
            logger.exception("Background task %s crashed", label)

    async def enqueue(self, task_id: str) -> None:
        from backend.services.pipeline import run_forecast

        if self.eager:
            await run_forecast(task_id)
        else:
            self._spawn(run_forecast(task_id), f"forecast:{task_id}")

    async def enqueue_eval(self, run_id: str) -> None:
        from backend.evaluation.service import run_evaluation_run

        if self.eager:
            await run_evaluation_run(run_id)
        else:
            self._spawn(run_evaluation_run(run_id), f"evaluate:{run_id}")

    async def enqueue_reindex(self) -> None:
        from backend.data.knowledge_base import reindex_rag

        if self.eager:
            await reindex_rag()
        else:
            self._spawn(reindex_rag(), "reindex_rag")

    async def enqueue_clear_cache(
        self, older_than_days: int = 30, max_hits: int | None = None
    ) -> None:
        from backend.data.cache import cache_manager

        if self.eager:
            await cache_manager.clear(older_than_days=older_than_days, max_hits=max_hits)
        else:
            self._spawn(
                cache_manager.clear(older_than_days=older_than_days, max_hits=max_hits),
                "clear_cache",
            )


class CeleryTaskRunner:
    """Отправка задачи в очередь (выполняет Celery-воркер)."""

    async def _send(self, name: str, args: tuple = (), kwargs: dict | None = None) -> None:
        from backend.worker.celery_app import celery_app

        celery_app.send_task(name, args=list(args), kwargs=kwargs or {}, queue="forecast")
        logger.info("Task %s %s sent to Celery queue 'forecast'", name, args or kwargs or "")

    async def enqueue(self, task_id: str) -> None:
        await self._send(FORECAST_TASK_NAME, args=(task_id,))

    async def enqueue_eval(self, run_id: str) -> None:
        await self._send(EVALUATE_TASK_NAME, args=(run_id,))

    async def enqueue_reindex(self) -> None:
        await self._send(REINDEX_TASK_NAME)

    async def enqueue_clear_cache(
        self, older_than_days: int = 30, max_hits: int | None = None
    ) -> None:
        await self._send(
            CLEAR_CACHE_TASK_NAME,
            kwargs={"older_than_days": older_than_days, "max_hits": max_hits},
        )


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


# ============================================================================
# Тела задач (регистрируются в celery_app)
# ============================================================================
def forecast_task(task_id: str) -> Any:
    """Выполнить прогноз по task_id."""
    from backend.services.pipeline import run_forecast

    return run_async(run_forecast(task_id))


def evaluate_model_task(run_id: str) -> Any:
    """Выполнить запуск оценки по run_id."""
    from backend.evaluation.service import run_evaluation_run

    return run_async(run_evaluation_run(run_id))


def reindex_rag_task() -> Any:
    """Переиндексировать базу знаний после подтверждения HITL."""
    from backend.data.knowledge_base import reindex_rag

    return run_async(reindex_rag())


def clear_cache_task(older_than_days: int = 30, max_hits: int | None = None) -> Any:
    """Очистить старый и/или непопулярный кэш (по расписанию beat — раз в сутки)."""
    from backend.data.cache import cache_manager

    return run_async(
        cache_manager.clear(older_than_days=older_than_days, max_hits=max_hits)
    )
