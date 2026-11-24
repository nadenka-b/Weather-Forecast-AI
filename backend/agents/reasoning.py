"""Иерархический лог рассуждений агентов (Этап 5).

Требование задания MLF: reasoning_log с parent_id (parent-child между шагами),
чтобы можно было восстановить дерево рассуждений, а не плоский список.

Структура дерева для задачи прогноза:

    supervisor (root)
    ├── data_analyst
    ├── model_selector
    │   └── tool: fit_sarima          (внуки — вызовы инструментов, Этап 7)
    ├── validator
    ├── critic
    └── editor (итерация 1..N)

Хранилища:
- InMemoryLogStore — для тестов и локальных прогонов без БД;
- DbLogStore — таблица reasoning_logs (запись при старте шага, обновление
  при завершении; финальное дерево доступно через API на Этапе 12).
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from backend.db.models import ReasoningLog

logger = logging.getLogger(__name__)


def new_request_id() -> str:
    """ID задачи прогноза — корень дерева рассуждений."""
    return str(uuid.uuid4())


class LogStore(Protocol):
    """Контракт хранилища записей лога рассуждений."""

    async def write(self, record: dict[str, Any]) -> None:
        """Создаёт запись (status=running)."""

    async def update(self, log_id: str, fields: dict[str, Any]) -> None:
        """Обновляет запись по завершении шага."""


class InMemoryLogStore:
    """Хранилище в памяти: записи-словари, иерархия по parent_id."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    async def write(self, record: dict[str, Any]) -> None:
        self.entries.append(record)

    async def update(self, log_id: str, fields: dict[str, Any]) -> None:
        for entry in self.entries:
            if entry["id"] == log_id:
                entry.update(fields)
                return
        logger.warning("Log entry %s not found for update", log_id)

    def tree(self) -> list[dict[str, Any]]:
        """Дерево: корневые записи с вложенными children."""
        by_id = {e["id"]: {**e, "children": []} for e in self.entries}
        roots: list[dict[str, Any]] = []
        for node in by_id.values():
            parent = node.get("parent_id")
            if parent and parent in by_id:
                by_id[parent]["children"].append(node)
            else:
                roots.append(node)
        return roots


class DbLogStore:
    """Хранилище в таблице reasoning_logs (async SQLAlchemy).

    На старте шага — INSERT (status='running'), на завершении — UPDATE.
    Сбой трассировки (БД недоступна, нет таблицы) не роняет пайплайн:
    ошибка пишется в лог, основной путь продолжает работать.
    """

    async def write(self, record: dict[str, Any]) -> None:
        from sqlalchemy import select

        from backend.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                exists = await session.execute(
                    select(ReasoningLog.id).where(ReasoningLog.id == record["id"])
                )
                if exists.scalar_one_or_none():
                    return
                session.add(ReasoningLog(**record))
                await session.commit()
        except Exception as exc:  # noqa: BLE001 — трассировка не должна ломать пайплайн
            logger.error("reasoning_logs write failed: %s", exc)

    async def update(self, log_id: str, fields: dict[str, Any]) -> None:
        from sqlalchemy import select

        from backend.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ReasoningLog).where(ReasoningLog.id == log_id)
                )
                log = result.scalar_one_or_none()
                if log is None:
                    logger.warning("reasoning_logs entry %s not found", log_id)
                    return
                for key, value in fields.items():
                    setattr(log, key, value)
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("reasoning_logs update failed: %s", exc)


class ReasoningLogger:
    """Ведёт дерево шагов: start -> (работа агента) -> finish."""

    def __init__(self, request_id: str, store: LogStore | None = None):
        self.request_id = request_id
        self.store = store or InMemoryLogStore()
        self._entries: list[dict[str, Any]] = []

    async def start(
        self,
        agent_name: str,
        parent_id: str | None = None,
        tool_name: str | None = None,
        input_summary: str | None = None,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Открывает шаг (status='running'), возвращает его log_id."""
        log_id = str(uuid.uuid4())
        record = {
            "id": log_id,
            "request_id": self.request_id,
            "parent_id": parent_id,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "input_summary": input_summary,
            "model_name": model_name,
            "parameters": parameters,
            "status": "running",
        }
        self._entries.append(record)
        await self.store.write(dict(record))
        return log_id

    async def finish(
        self,
        log_id: str,
        output_summary: str | None = None,
        status: str = "success",
    ) -> None:
        """Закрывает шаг: статус, время завершения, краткий результат."""
        fields = {
            "output_summary": output_summary,
            "status": status,
            "finished_at": datetime.now(timezone.utc),
        }
        for entry in self._entries:
            if entry["id"] == log_id:
                entry.update(fields)
        await self.store.update(log_id, fields)

    @property
    def entries(self) -> list[dict[str, Any]]:
        """Плоский список записей текущего прогона (для state.metadata)."""
        return list(self._entries)


async def fetch_reasoning_tree(request_id: str) -> list[dict[str, Any]]:
    """Дерево рассуждений задачи из reasoning_logs (Этап 12, GET /reasoning)."""
    from sqlalchemy import select

    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ReasoningLog)
                .where(ReasoningLog.request_id == request_id)
                .order_by(ReasoningLog.started_at)
            )
        ).scalars().all()

    def iso(value: Any) -> str | None:
        return value.isoformat() if value else None

    nodes = {
        row.id: {
            "id": row.id,
            "parent_id": row.parent_id,
            "agent_name": row.agent_name,
            "tool_name": row.tool_name,
            "status": row.status,
            "input_summary": row.input_summary,
            "output_summary": row.output_summary,
            "started_at": iso(row.started_at),
            "finished_at": iso(row.finished_at),
            "children": [],
        }
        for row in rows
    }
    roots: list[dict[str, Any]] = []
    for node in nodes.values():
        parent = nodes.get(node["parent_id"]) if node["parent_id"] else None
        (parent["children"] if parent else roots).append(node)
    return roots
