"""Обратная связь эксперта (Этап 14, методичка Step_10 — FeedbackHandler).

save_correction — сохраняет решение/правку по задаче в hitl_feedback без
смены статуса (например, комментарий к уже завершённой задаче или правка
модели «на будущее»). Опционально добавляет кейс в базу знаний.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from backend.core.models import ForecastState
from backend.db.models import ForecastTask, HitlFeedback, KnowledgeCase

logger = logging.getLogger(__name__)


class FeedbackHandler:
    """Сохранение решений и правок эксперта."""

    @staticmethod
    async def save_correction(
        task_id: str,
        user_id: str | None = None,
        correction: dict[str, Any] | None = None,
        comment: str | None = None,
        add_to_kb: bool = False,
    ) -> dict[str, Any]:
        """Записывает правку; при add_to_kb=True добавляет кейс в базу знаний."""
        from backend.db.session import AsyncSessionLocal
        from backend.hitl.active_learning import build_kb_case

        kb_inserted = False
        async with AsyncSessionLocal() as session:
            task = await session.get(ForecastTask, task_id)
            if task is None:
                raise LookupError(f"Задача {task_id} не найдена")
            session.add(
                HitlFeedback(
                    task_id=task_id, user_id=user_id, decision="correction",
                    correction=correction, comment=comment,
                )
            )
            if add_to_kb and task.state:
                kb_case = build_kb_case(task, ForecastState(**task.state))
                if kb_case is not None:
                    exists = await session.execute(
                        select(KnowledgeCase.id).where(
                            KnowledgeCase.series_id == kb_case["series_id"]
                        )
                    )
                    if exists.scalar_one_or_none() is None:
                        session.add(KnowledgeCase(**kb_case))
                        kb_inserted = True
            await session.commit()
        return {"task_id": task_id, "saved": True, "kb_inserted": kb_inserted}

    @staticmethod
    async def list_feedback(task_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Последние записи обратной связи (опционально по задаче)."""
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            query = select(HitlFeedback).order_by(HitlFeedback.created_at.desc()).limit(limit)
            if task_id:
                query = query.where(HitlFeedback.task_id == task_id)
            rows = (await session.execute(query)).scalars().all()
            return [
                {
                    "id": row.id,
                    "task_id": row.task_id,
                    "user_id": row.user_id,
                    "decision": row.decision,
                    "correction": row.correction,
                    "comment": row.comment,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
