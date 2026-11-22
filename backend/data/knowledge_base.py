"""Доступ к базе знаний метеокейсов и справочнику терминов (Этап 9).

Тонкий async-слой над таблицами knowledge_base и glossary: используется
scripts/import_data.py, а с Этапа 10 — гибридным RAG-поиском.
Дубликаты отбрасываются по (series_id, task_type) и term.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select

from backend.db.models import GlossaryTerm, KnowledgeCase

logger = logging.getLogger(__name__)


async def import_cases(cases: list[dict[str, Any]]) -> tuple[int, int]:
    """Импорт кейсов: (вставлено, пропущено-дубликатов)."""
    from backend.db.session import AsyncSessionLocal

    inserted = skipped = 0
    async with AsyncSessionLocal() as session:
        existing = set(
            (await session.execute(select(KnowledgeCase.series_id, KnowledgeCase.task_type))).all()
        )
        for case in cases:
            key = (case["series_id"], case.get("task_type", "timeseries"))
            if key in existing:
                skipped += 1
                continue
            session.add(
                KnowledgeCase(
                    series_id=case["series_id"],
                    task_type=case.get("task_type", "timeseries"),
                    city=case.get("city"),
                    freq=case.get("freq"),
                    target_column=case.get("target_column"),
                    features=case.get("features"),
                    best_model=case["best_model"],
                    hyperparams=case.get("hyperparams"),
                    metrics=case.get("metrics"),
                    description=case.get("description"),
                )
            )
            existing.add(key)
            inserted += 1
        await session.commit()
    logger.info("knowledge_base: inserted=%d skipped=%d", inserted, skipped)
    return inserted, skipped


async def import_terms(terms: list[dict[str, Any]]) -> tuple[int, int]:
    """Импорт терминов справочника: (вставлено, пропущено-дубликатов)."""
    from backend.db.session import AsyncSessionLocal

    inserted = skipped = 0
    async with AsyncSessionLocal() as session:
        existing = set(
            (await session.execute(select(GlossaryTerm.term))).scalars().all()
        )
        for term in terms:
            name = term["term"]
            if name in existing:
                skipped += 1
                continue
            session.add(
                GlossaryTerm(
                    term=name,
                    definition=term["definition"],
                    category=term.get("category"),
                )
            )
            existing.add(name)
            inserted += 1
        await session.commit()
    logger.info("glossary: inserted=%d skipped=%d", inserted, skipped)
    return inserted, skipped


async def get_counts() -> dict[str, int]:
    """Количество записей в таблицах базы знаний и справочника."""
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        cases = await session.scalar(select(func.count(KnowledgeCase.id)))
        terms = await session.scalar(select(func.count(GlossaryTerm.id)))
    return {"knowledge_base": int(cases or 0), "glossary": int(terms or 0)}


async def sample_cases(limit: int = 5) -> list[dict[str, Any]]:
    """Последние кейсы (для быстрой проверки/отладки RAG)."""
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            select(KnowledgeCase).order_by(KnowledgeCase.created_at.desc()).limit(limit)
        )
        return [
            {
                "series_id": row.series_id,
                "city": row.city,
                "best_model": row.best_model,
                "metrics": row.metrics,
                "description": row.description,
            }
            for row in rows.scalars()
        ]
