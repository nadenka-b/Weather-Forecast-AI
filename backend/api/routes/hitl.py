"""Роуты Human-in-the-Loop (Этап 14) — очередь на проверку, подтверждение, отклонение.

Все эндпоинты доступны только ролям editor/admin (require_editor):
    GET  /hitl/queue?limit&offset&city&variable  задачи в waiting_for_human
    GET  /hitl/stats                             {pending, confirmed, rejected}
    POST /hitl/{task_id}/confirm                 подтвердить (кейс -> база знаний, переиндексация)
    POST /hitl/{task_id}/reject                  отклонить (база знаний не меняется)
    POST /hitl/feedback                          сохранить правку/комментарий
    GET  /hitl/feedback?task_id&limit            история решений

Задача может быть подтверждена с правкой (другая модель/параметры): тогда
прогноз перевыполняется, и в базу знаний попадает уже исправленный кейс.
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import require_editor
from backend.api.schemas import ConfirmRequest, CorrectionRequest, RejectRequest
from backend.db.models import User
from backend.hitl.active_learning import ActiveLearningManager
from backend.hitl.feedback import FeedbackHandler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/hitl", tags=["hitl"])


@router.get("/queue")
async def get_queue(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    city: str | None = Query(None),
    variable: str | None = Query(None),
    current_user: User = Depends(require_editor),
) -> dict[str, Any]:
    """Очередь задач, ожидающих решения человека (сначала новые)."""
    examples = await ActiveLearningManager.get_pending_examples(
        limit=limit, offset=offset, city=city, variable=variable
    )
    return {"count": await ActiveLearningManager.count_pending(), "examples": examples}


@router.get("/stats")
async def get_stats(current_user: User = Depends(require_editor)) -> dict[str, int]:
    return await ActiveLearningManager.get_stats()


@router.post("/{task_id}/confirm")
async def confirm(
    task_id: str,
    request: ConfirmRequest,
    current_user: User = Depends(require_editor),
) -> dict[str, Any]:
    """Подтвердить прогноз: кейс попадает в базу знаний (active learning)."""
    correction = None
    if request.model_type or request.model_params:
        correction = {"model_type": request.model_type, "model_params": request.model_params}
    try:
        return await ActiveLearningManager.confirm_correction(
            task_id, user_id=current_user.id, correction=correction, comment=request.comment
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{task_id}/reject")
async def reject(
    task_id: str,
    request: RejectRequest,
    current_user: User = Depends(require_editor),
) -> dict[str, Any]:
    """Отклонить прогноз: задача закрыта, база знаний не меняется."""
    try:
        return await ActiveLearningManager.reject_example(
            task_id, user_id=current_user.id, reason=request.reason
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/feedback")
async def save_feedback(
    request: CorrectionRequest,
    current_user: User = Depends(require_editor),
) -> dict[str, Any]:
    """Сохранить правку/комментарий эксперта (без смены статуса задачи)."""
    try:
        return await FeedbackHandler.save_correction(
            request.task_id,
            user_id=current_user.id,
            correction=request.correction,
            comment=request.comment,
            add_to_kb=request.add_to_kb,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/feedback")
async def list_feedback(
    task_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(require_editor),
) -> dict[str, Any]:
    return {"feedback": await FeedbackHandler.list_feedback(task_id=task_id, limit=limit)}

