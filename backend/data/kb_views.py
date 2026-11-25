"""Просмотр и правка базы знаний и справочника (Этап 15 — для страниц UI).

read-операции доступны любому аутентифицированному пользователю,
add/delete — только admin (роуты /admin/kb/*, /admin/glossary/*).
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import delete, func, select

from backend.db.models import GlossaryTerm, KnowledgeCase

logger = logging.getLogger(__name__)


def _case_to_dict(row: KnowledgeCase) -> dict[str, Any]:
    return {
        "id": row.id,
        "series_id": row.series_id,
        "task_type": row.task_type,
        "city": row.city,
        "freq": row.freq,
        "target_column": row.target_column,
        "features": row.features,
        "best_model": row.best_model,
        "hyperparams": row.hyperparams,
        "metrics": row.metrics,
        "description": row.description,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ============================================================================
# База знаний
# ============================================================================
async def list_cases(
    city: str | None = None,
    variable: str | None = None,
    best_model: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Кейсы с фильтрами (город, переменная, модель, поиск по описанию)."""
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        query = select(KnowledgeCase)
        if city:
            query = query.where(KnowledgeCase.city == city)
        if variable:
            query = query.where(KnowledgeCase.target_column == variable)
        if best_model:
            query = query.where(KnowledgeCase.best_model == best_model)
        if search:
            # ILIKE в SQLite нечувствителен к регистру только для ASCII — фильтруем в Python
            needle = search.lower()
            matching = [
                row.id for row in (await session.execute(
                    select(KnowledgeCase.id, KnowledgeCase.description)
                )).all()
                if row.description and needle in row.description.lower()
            ]
            query = query.where(KnowledgeCase.id.in_(matching))
        total = await session.scalar(select(func.count()).select_from(query.subquery()))
        rows = (
            await session.execute(
                query.order_by(KnowledgeCase.created_at.desc()).offset(offset).limit(limit)
            )
        ).scalars().all()
        return {"total": int(total or 0), "cases": [_case_to_dict(row) for row in rows]}


async def get_case(case_id: str) -> dict[str, Any] | None:
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        row = await session.get(KnowledgeCase, case_id)
        return _case_to_dict(row) if row else None


async def add_case(case: dict[str, Any]) -> dict[str, Any]:
    """Добавляет кейс (защита от дубля по series_id + task_type)."""
    from backend.db.session import AsyncSessionLocal

    if not case.get("series_id") or not case.get("best_model"):
        raise ValueError("Для кейса нужны series_id и best_model")
    async with AsyncSessionLocal() as session:
        exists = await session.execute(
            select(KnowledgeCase.id).where(
                KnowledgeCase.series_id == case["series_id"],
                KnowledgeCase.task_type == case.get("task_type", "timeseries"),
            )
        )
        if exists.scalar_one_or_none():
            raise ValueError(f"Кейс {case['series_id']!r} уже существует")
        features = case.get("features") or {}
        row = KnowledgeCase(
            series_id=case["series_id"],
            task_type=case.get("task_type", "timeseries"),
            city=case.get("city"),
            freq=case.get("freq"),
            target_column=case.get("target_column") or features.get("variable"),
            features=features,
            best_model=case["best_model"],
            hyperparams=case.get("hyperparams"),
            metrics=case.get("metrics"),
            description=case.get("description"),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _case_to_dict(row)


async def delete_case(case_id: str) -> bool:
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(delete(KnowledgeCase).where(KnowledgeCase.id == case_id))
        await session.commit()
        return bool(result.rowcount)


async def search_cases(text: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Лексический поиск похожих кейсов (BM25 по описаниям)."""
    from backend.retrieval.hybrid_retriever import load_cases_from_db
    from backend.retrieval.bm25_index import BM25Index

    cases = await load_cases_from_db()
    if not cases or not text.strip():
        return []
    index = BM25Index(cases)
    by_id = {case["series_id"]: case for case in cases}
    return [
        {
            "series_id": series_id,
            "score": round(score, 4),
            "city": by_id[series_id].get("city"),
            "variable": (by_id[series_id].get("features") or {}).get("variable"),
            "best_model": by_id[series_id].get("best_model"),
            "hyperparams": by_id[series_id].get("hyperparams"),
            "metrics": by_id[series_id].get("metrics"),
            "description": by_id[series_id].get("description"),
        }
        for series_id, score in index.search(text, top_k=top_k)
    ]


# ============================================================================
# Справочник терминов
# ============================================================================
async def list_terms(
    category: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Термины с фильтрами + список категорий."""
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        query = select(GlossaryTerm)
        if category:
            query = query.where(GlossaryTerm.category == category)
        if search:
            # кириллица: регистронезависимый поиск в Python (ILIKE в SQLite — только ASCII)
            needle = search.lower()
            matching = [
                row.id for row in (await session.execute(
                    select(GlossaryTerm.id, GlossaryTerm.term, GlossaryTerm.definition)
                )).all()
                if needle in row.term.lower() or needle in (row.definition or "").lower()
            ]
            query = query.where(GlossaryTerm.id.in_(matching))
        total = await session.scalar(select(func.count()).select_from(query.subquery()))
        rows = (
            await session.execute(query.order_by(GlossaryTerm.term).offset(offset).limit(limit))
        ).scalars().all()
        categories = sorted(
            row[0]
            for row in (
                await session.execute(
                    select(GlossaryTerm.category)
                    .where(GlossaryTerm.category.isnot(None))
                    .distinct()
                )
            ).all()
        )
        return {
            "total": int(total or 0),
            "categories": categories,
            "terms": [
                {"id": r.id, "term": r.term, "definition": r.definition, "category": r.category}
                for r in rows
            ],
        }


async def add_term(term: str, definition: str, category: str | None = None) -> dict[str, Any]:
    from backend.db.session import AsyncSessionLocal

    if not term.strip() or not definition.strip():
        raise ValueError("Нужны термин и определение")
    async with AsyncSessionLocal() as session:
        exists = await session.execute(
            select(GlossaryTerm.id).where(GlossaryTerm.term == term.strip())
        )
        if exists.scalar_one_or_none():
            raise ValueError(f"Термин {term!r} уже есть в справочнике")
        row = GlossaryTerm(
            term=term.strip(), definition=definition.strip(), category=category
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return {"id": row.id, "term": row.term, "definition": row.definition,
                "category": row.category}


async def delete_term(term_id: str) -> bool:
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            delete(GlossaryTerm).where(GlossaryTerm.id == term_id)
        )
        await session.commit()
        return bool(result.rowcount)
