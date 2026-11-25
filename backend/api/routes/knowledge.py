"""Роуты чтения базы знаний и справочника (Этап 15 — для страниц UI).

Доступны любому аутентифицированному пользователю:
    GET /kb/search?q=...                 похожие кейсы (BM25 по описаниям)
    GET /kb/cases?city&variable&best_model&search&limit&offset
    GET /kb/cases/{case_id}
    GET /glossary/terms?category&search&limit&offset

Правка (add/delete) — в backend/api/routes/admin.py, только admin.
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import get_current_user
from backend.data import kb_views
from backend.db.models import User

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])


@router.get("/kb/search")
async def search_knowledge(
    q: str = Query(..., min_length=1, description="Текст запроса (например, «температура Jena сезонность 7»)"),
    top_k: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    results = await kb_views.search_cases(q, top_k=top_k)
    return {"query": q, "count": len(results), "results": results}


@router.get("/kb/cases")
async def list_kb_cases(
    city: str | None = Query(None),
    variable: str | None = Query(None),
    best_model: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return await kb_views.list_cases(
        city=city, variable=variable, best_model=best_model, search=search,
        limit=limit, offset=offset,
    )


@router.get("/kb/cases/{case_id}")
async def get_kb_case(
    case_id: str, current_user: User = Depends(get_current_user)
) -> dict[str, Any]:
    case = await kb_views.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Кейс не найден")
    return case


@router.get("/glossary/terms")
async def list_glossary_terms(
    category: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return await kb_views.list_terms(
        category=category, search=search, limit=limit, offset=offset
    )
