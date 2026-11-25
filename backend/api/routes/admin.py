"""Администрирование (Этап 15 — для страниц UI), только admin.

    POST   /admin/kb/add                 добавить кейс в базу знаний
    DELETE /admin/kb/delete/{case_id}
    POST   /admin/kb/reindex             переиндексация RAG (фоновая задача)
    POST   /admin/glossary/add           добавить термин
    DELETE /admin/glossary/delete/{term_id}
    POST   /admin/glossary/import-csv    CSV: term;definition;category
    GET    /admin/cache/stats            (пути как в методичке Step_12)
    DELETE /admin/cache/clear?older_than_days&max_hits

Роли назначаются скриптом scripts/create_admin.py или напрямую в БД.
"""
import csv
import io
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from backend.api.deps import require_admin
from backend.api.schemas import KbAddRequest, TermRequest
from backend.data import kb_views
from backend.data.cache import cache_manager
from backend.db.models import User
from backend.worker.tasks import get_task_runner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# База знаний
# ============================================================================
@router.post("/kb/add", status_code=201)
async def add_kb_case(
    request: KbAddRequest, current_user: User = Depends(require_admin)
) -> dict[str, Any]:
    try:
        case = await kb_views.add_case(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await get_task_runner().enqueue_reindex()
    return case


@router.delete("/kb/delete/{case_id}")
async def delete_kb_case(
    case_id: str, current_user: User = Depends(require_admin)
) -> dict[str, Any]:
    if not await kb_views.delete_case(case_id):
        raise HTTPException(status_code=404, detail="Кейс не найден")
    await get_task_runner().enqueue_reindex()
    return {"deleted": case_id}


@router.post("/kb/reindex", status_code=202)
async def reindex_knowledge(current_user: User = Depends(require_admin)) -> dict[str, Any]:
    await get_task_runner().enqueue_reindex()
    return {"scheduled": True, "task": "rag.reindex_rag_task"}


# ============================================================================
# Справочник
# ============================================================================
@router.post("/glossary/add", status_code=201)
async def add_glossary_term(
    request: TermRequest, current_user: User = Depends(require_admin)
) -> dict[str, Any]:
    try:
        return await kb_views.add_term(
            request.term, request.definition, request.category
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/glossary/delete/{term_id}")
async def delete_glossary_term(
    term_id: str, current_user: User = Depends(require_admin)
) -> dict[str, Any]:
    if not await kb_views.delete_term(term_id):
        raise HTTPException(status_code=404, detail="Термин не найден")
    return {"deleted": term_id}


@router.post("/glossary/import-csv")
async def import_glossary_csv(
    file: UploadFile = File(...), current_user: User = Depends(require_admin)
) -> dict[str, Any]:
    """CSV с колонками term;definition;category (разделитель ; или ,)."""
    content = (await file.read()).decode("utf-8-sig")
    sample = content[:2048]
    delimiter = ";" if sample.count(";") >= sample.count(",") else ","
    added = skipped = 0
    errors: list[str] = []
    for number, row in enumerate(csv.DictReader(io.StringIO(content), delimiter=delimiter), start=2):
        term = (row.get("term") or "").strip()
        definition = (row.get("definition") or "").strip()
        category = (row.get("category") or "").strip() or None
        if not term or not definition:
            errors.append(f"строка {number}: нет term или definition")
            continue
        try:
            await kb_views.add_term(term, definition, category)
            added += 1
        except ValueError:
            skipped += 1  # дубликат
        except Exception as exc:  # noqa: BLE001
            errors.append(f"строка {number}: {exc}")
    return {"added": added, "skipped": skipped, "errors": errors[:20]}


# ============================================================================
# Кэш (пути из методички Step_12)
# ============================================================================
@router.get("/cache/stats")
async def admin_cache_stats(current_user: User = Depends(require_admin)) -> dict[str, Any]:
    return await cache_manager.stats()


@router.delete("/cache/clear")
async def admin_cache_clear(
    older_than_days: int | None = Query(None, ge=1),
    max_hits: int | None = Query(None, ge=0),
    current_user: User = Depends(require_admin),
) -> dict[str, Any]:
    """Очистка кэша: по возрасту и/или числу попаданий (условия объединяются через И).

    Без параметров — записи старше 30 дней.
    """
    if older_than_days is None and max_hits is None:
        older_than_days = 30
    before = await cache_manager.stats()
    await get_task_runner().enqueue_clear_cache(
        older_than_days=older_than_days, max_hits=max_hits
    )
    after = await cache_manager.stats()
    return {
        "deleted_inline": before["entries"] - after["entries"],
        "entries_before": before["entries"],
        "entries_after_inline": after["entries"],
    }
