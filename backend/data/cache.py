"""Кэш результатов LLM и прогнозов (Этап 14, методичка Step_11).

Хранилище — таблица cache (input_key, model_used, output, hits). Уникальность
(input_key, model_used); при попадании счётчик hits увеличивается в той же
транзакции. Любой сбой кэша (БД недоступна) логируется и не влияет на основной
путь: кэш — оптимизация, а не зависимость.

Использование:
    key = CacheManager.make_key("llm", model, messages, temperature)
    if (cached := await cache_manager.get(key, model_used="llm")) is not None: ...
    await cache_manager.set(key, payload_json, model_used="llm")
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager:
    """CRUD над таблицей cache с подсчётом попаданий."""

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Детерминированный ключ: SHA-256 от канонического JSON аргументов."""
        canonical = json.dumps(
            parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def get(self, key: str, model_used: str | None = None) -> str | None:
        """Возвращает закэшированное значение или None; увеличивает hits."""
        from sqlalchemy import select, update

        from backend.db.models import ForecastCache
        from backend.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(
                        select(ForecastCache).where(
                            ForecastCache.input_key == key,
                            ForecastCache.model_used == (model_used or "generic"),
                        )
                    )
                ).scalar_one_or_none()
                if row is None:
                    return None
                await session.execute(
                    update(ForecastCache)
                    .where(ForecastCache.id == row.id)
                    .values(hits=ForecastCache.hits + 1)
                )
                await session.commit()
                return row.output
        except Exception as exc:  # noqa: BLE001 — кэш не должен ломать основной путь
            logger.warning("cache get failed: %s", exc)
            return None

    async def set(self, key: str, output: str, model_used: str | None = None) -> bool:
        """Сохраняет значение; False, если запись уже есть (перезаписи нет)."""
        from sqlalchemy import select

        from backend.db.models import ForecastCache
        from backend.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                existing = (
                    await session.execute(
                        select(ForecastCache.id).where(
                            ForecastCache.input_key == key,
                            ForecastCache.model_used == (model_used or "generic"),
                        )
                    )
                ).scalar_one_or_none()
                if existing:
                    return False
                session.add(
                    ForecastCache(
                        input_key=key, model_used=model_used or "generic",
                        output=output, hits=0,
                    )
                )
                await session.commit()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache set failed: %s", exc)
            return False

    async def clear(
        self, older_than_days: int | None = None, max_hits: int | None = None
    ) -> int:
        """Очистка: старше N дней И (опционально) с числом попаданий <= max_hits.

        В отличие от референса (Step_11, п. 11.8) учитывает популярность:
        часто используемая запись не удаляется только из-за возраста.
        """
        from sqlalchemy import delete, select

        from backend.db.models import ForecastCache
        from backend.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                query = select(ForecastCache.id)
                conditions = []
                if older_than_days is not None:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
                    conditions.append(ForecastCache.created_at < cutoff)
                if max_hits is not None:
                    conditions.append(ForecastCache.hits <= max_hits)
                for condition in conditions:
                    query = query.where(condition)
                ids = [row[0] for row in (await session.execute(query)).all()]
                if ids:
                    await session.execute(
                        delete(ForecastCache).where(ForecastCache.id.in_(ids))
                    )
                    await session.commit()
                return len(ids)
        except Exception as exc:  # noqa: BLE001
            logger.error("cache clear failed: %s", exc)
            return 0

    async def stats(self) -> dict[str, Any]:
        """entries, total_hits."""
        from sqlalchemy import func, select

        from backend.db.models import ForecastCache
        from backend.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                entries, total_hits = (
                    await session.execute(
                        select(
                            func.count(ForecastCache.id),
                            func.coalesce(func.sum(ForecastCache.hits), 0),
                        )
                    )
                ).one()
            return {
                "entries": int(entries or 0),
                "total_hits": int(total_hits or 0),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache stats failed: %s", exc)
            return {"entries": 0, "total_hits": 0, "error": str(exc)[:200]}


cache_manager = CacheManager()
