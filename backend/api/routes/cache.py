"""Роуты администрирования кэша (Этап 14).

    GET    /cache/stats                          записи и число попаданий (editor/admin)
    DELETE /cache/clear?older_than_days&max_hits очистка (только admin)

Очистка выполняется как фоновая задача (inline или Celery clear_cache_task);
по расписанию та же задача запускается beat'ом раз в сутки.
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from backend.api.deps import require_admin, require_editor
from backend.data.cache import cache_manager
from backend.db.models import User
from backend.worker.tasks import get_task_runner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats")
async def cache_stats(current_user: User = Depends(require_editor)) -> dict[str, Any]:
    return await cache_manager.stats()


@router.delete("/clear")
async def clear_cache(
    older_than_days: int | None = Query(None, ge=1, description="Удалить записи старше N дней"),
    max_hits: int | None = Query(
        None, ge=0, description="Удалить только записи с числом попаданий <= max_hits"
    ),
    current_user: User = Depends(require_admin),
) -> dict[str, Any]:
    """Очистка кэша: по возрасту и/или популярности (фоновая задача)."""
    if older_than_days is None and max_hits is None:
        older_than_days = 30  # безопасное значение по умолчанию
    before = await cache_manager.stats()
    await get_task_runner().enqueue_clear_cache(
        older_than_days=older_than_days, max_hits=max_hits
    )
    after = await cache_manager.stats()
    return {
        "scheduled": True,
        "older_than_days": older_than_days,
        "max_hits": max_hits,
        "entries_before": before["entries"],
        "entries_after_inline": after["entries"],
    }

