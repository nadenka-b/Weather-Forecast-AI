"""Роуты оценки качества (Этап 13) — контракт задания MLF.

GET  /api/v1/evaluate/compare?models=...&sample_size=...&horizon=...  сравнение моделей
POST /api/v1/evaluate/batch                                            пакетный запуск (202)
GET  /api/v1/evaluate/runs/{run_id}                                    статус запуска + сводка
GET  /api/v1/evaluate/results                                          сохранённые результаты
GET  /api/v1/evaluate/comparisons                                      сохранённые сравнения

Запускать оценку могут editor/admin (тяжёлые вычисления); читать — любой
аутентифицированный пользователь. Результаты сохраняются в evaluation_results
и model_comparisons.
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from backend.api.deps import get_current_user, require_editor
from backend.api.schemas import EvaluateBatchRequest, EvaluationAccepted, EvaluationRunResponse
from backend.db.models import ModelComparison, User
from backend.evaluation.service import (
    compare_models,
    create_run,
    get_run,
    list_results,
)
from backend.worker.tasks import get_task_runner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/evaluate", tags=["evaluate"])


@router.get("/compare")
async def compare(
    models: str = Query("arima,xgboost", description="Модели через запятую"),
    sample_size: int = Query(20, ge=1, le=50),
    horizon: int = Query(14, ge=1, le=60),
    current_user: User = Depends(require_editor),
) -> dict[str, Any]:
    """Сравнение именованных моделей на одной выборке тестовых окон."""
    names = [name.strip() for name in models.split(",") if name.strip()]
    try:
        return await compare_models(names, sample_size=sample_size, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/batch", response_model=EvaluationAccepted, status_code=status.HTTP_202_ACCEPTED)
async def run_batch(
    payload: EvaluateBatchRequest,
    current_user: User = Depends(require_editor),
) -> EvaluationAccepted:
    """Пакетная оценка по режимам/горизонтам/долям данных; результат — по run_id."""
    try:
        run = await create_run(payload.model_dump(), user_id=current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await get_task_runner().enqueue_eval(run["id"])
    run = await get_run(run["id"]) or run
    return EvaluationAccepted(
        run_id=run["id"],
        status=run["status"],
        status_url=f"/api/v1/evaluate/runs/{run['id']}",
    )


@router.get("/runs/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation_run(
    run_id: str, current_user: User = Depends(get_current_user)
) -> EvaluationRunResponse:
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Запуск оценки не найден")
    return EvaluationRunResponse(
        run_id=run["id"],
        status=run["status"],
        params={k: v for k, v in (run["params"] or {}).items()},
        summary=run["summary"],
        comparisons=run["comparisons"],
        n_records=run["n_records"],
        error=run["error"],
    )


@router.get("/results")
async def results(
    mode: str | None = Query(None, description="Фильтр по режиму"),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return {"results": await list_results(mode=mode, limit=limit)}


@router.get("/comparisons")
async def comparisons(
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ModelComparison).order_by(ModelComparison.created_at.desc()).limit(limit)
            )
        ).scalars().all()
    return {
        "comparisons": [
            {
                "id": row.id,
                "models_compared": row.models_compared,
                "sample_size": row.sample_size,
                "results": row.results,
                "winner": row.winner,
                "significance": row.significance,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }
