"""Репозиторий задач прогноза (Этап 12): таблица forecast_tasks + файлы входов.

State хранится «по ссылке»: в БД лежит снимок ForecastState по task_id,
входной ряд — CSV-файл (data/uploads/<task_id>.csv), тяжёлые объекты
в payload запросов не передаются (требование задания MLF).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from backend.core.config import get_settings
from backend.core.models import ForecastState
from backend.db.models import ForecastTask

logger = logging.getLogger(__name__)

# pending -> running -> waiting_for_human | completed | failed
TASK_STATUSES: tuple[str, ...] = (
    "pending", "running", "waiting_for_human", "completed", "failed",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def task_to_dict(task: ForecastTask) -> dict[str, Any]:
    """ORM-строка -> словарь для API (даты в ISO)."""
    def iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "id": task.id,
        "user_id": task.user_id,
        "status": task.status,
        "series_id": task.series_id,
        "horizon": task.horizon,
        "task_type": task.task_type,
        "target_column": task.target_column,
        "input_path": task.input_path,
        "state": task.state,
        "result": task.result,
        "error": task.error,
        "hitl_requested": task.hitl_requested,
        "iterations": task.iterations,
        "created_at": iso(task.created_at),
        "started_at": iso(task.started_at),
        "finished_at": iso(task.finished_at),
    }


class TaskRepository:
    """Async-CRUD задач прогноза."""

    def save_input_series(self, task_id: str, values: list[float], dates: list | None) -> str:
        """Сохраняет входной ряд в CSV (date,value); возвращает путь."""
        uploads = Path(get_settings().uploads_dir)
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f"{task_id}.csv"
        if dates is not None:
            index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        else:
            index = pd.date_range("2000-01-01", periods=len(values), freq="D")
        frame = pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "value": values})
        frame.to_csv(path, index=False)
        return str(path)

    def save_input_file(self, task_id: str, content: bytes, filename: str) -> str:
        """Сохраняет загруженный файл таблицы как есть (tabular-задачи)."""
        uploads = Path(get_settings().uploads_dir)
        uploads.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".csv"
        path = uploads / f"{task_id}{suffix}"
        path.write_bytes(content)
        return str(path)

    async def create(
        self,
        *,
        user_id: str | None,
        state: ForecastState,
        input_path: str | None,
    ) -> dict[str, Any]:
        """Создаёт задачу в статусе pending со снимком state."""
        from backend.db.session import AsyncSessionLocal

        task_id = state.task_id or str(uuid.uuid4())
        state.task_id = task_id
        state.log_id = task_id  # reasoning_logs и llm_calls сшиваются с задачей
        async with AsyncSessionLocal() as session:
            task = ForecastTask(
                id=task_id,
                user_id=user_id,
                status="pending",
                series_id=state.series_id,
                horizon=state.horizon,
                task_type=state.task_type,
                target_column=state.target_column,
                input_path=input_path,
                state=state.model_dump(mode="json"),
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task_to_dict(task)

    async def get(self, task_id: str) -> dict[str, Any] | None:
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            task = await session.get(ForecastTask, task_id)
            return task_to_dict(task) if task else None

    async def set_status(
        self,
        task_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        """Обновляет статус и отметки времени (started/finished)."""
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            task = await session.get(ForecastTask, task_id)
            if task is None:
                raise LookupError(f"Задача {task_id} не найдена")
            task.status = status
            if status == "running":
                task.started_at = _now()
            if status in ("waiting_for_human", "completed", "failed"):
                task.finished_at = _now()
            if error is not None:
                task.error = error
            await session.commit()

    async def save_state(
        self,
        task_id: str,
        state: ForecastState,
        result: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> None:
        """Сохраняет снимок state, результат и (опционально) статус."""
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            task = await session.get(ForecastTask, task_id)
            if task is None:
                raise LookupError(f"Задача {task_id} не найдена")
            task.state = state.model_dump(mode="json")
            if result is not None:
                task.result = result
            task.hitl_requested = state.hitl_requested
            task.iterations = state.iteration
            if status:
                task.status = status
                if status in ("waiting_for_human", "completed", "failed"):
                    task.finished_at = _now()
            await session.commit()

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            rows = await session.execute(
                select(ForecastTask)
                .where(ForecastTask.user_id == user_id)
                .order_by(ForecastTask.created_at.desc())
                .limit(limit)
            )
            return [task_to_dict(task) for task in rows.scalars()]
