"""Роуты прогноза (Этап 12) — контракт задания MLF.

POST /api/v1/forecast/                 загрузка ряда (CSV multipart) + параметры -> 202 + task_id
POST /api/v1/forecast/json             то же, ряд в JSON (inline-значения)
GET  /api/v1/forecast/{task_id}/status статус задачи
GET  /api/v1/forecast/{task_id}/result результат, метрики, модель, путь к файлу
GET  /api/v1/forecast/{task_id}/reasoning дерево рассуждений + вызовы LLM
POST /api/v1/forecast/{task_id}/hitl   решение человека (accept / change_model / change_horizon)

State передаётся по ссылке (task_id): роуты только читают/меняют задачу,
весь тяжёлый пайплайн выполняет раннер (inline или Celery-воркер).
"""
import io
import logging
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select

from backend.api.deps import get_current_user
from backend.api.schemas import (
    ForecastAccepted,
    ForecastCreateJson,
    ForecastResult,
    ForecastStatus,
    HitlRequest,
    ReasoningResponse,
)
from backend.core.models import ForecastState
from backend.data.tasks import TaskRepository
from backend.db.models import LLMCall, User
from backend.services.pipeline import apply_hitl_resolution
from backend.agents.reasoning import fetch_reasoning_tree
from backend.worker.tasks import get_task_runner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])

_FINISHED = ("completed", "waiting_for_human")


# ============================================================================
# Вспомогательное
# ============================================================================
def _ensure_access(task: dict[str, Any] | None, user: User) -> dict[str, Any]:
    """Задача доступна владельцу и ролям editor/admin; иначе 404."""
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if task["user_id"] != user.id and user.role not in ("admin", "editor"):
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task


def _status_response(task: dict[str, Any]) -> ForecastStatus:
    return ForecastStatus(
        task_id=task["id"],
        status=task["status"],
        hitl_requested=task["hitl_requested"],
        iterations=task["iterations"],
        error=task["error"],
        created_at=task["created_at"],
        started_at=task["started_at"],
        finished_at=task["finished_at"],
    )


async def _create_and_run(
    *,
    user: User,
    series_id: str,
    horizon: int,
    freq: str,
    task_type: str,
    target_column: str | None,
    metadata: dict[str, Any],
    input_path: str,
) -> dict[str, Any]:
    """Создаёт задачу и ставит её в раннер; возвращает свежий снимок задачи."""
    repo = TaskRepository()
    task_id = str(uuid.uuid4())
    state = ForecastState(
        task_id=task_id,
        series_id=series_id,
        horizon=horizon,
        freq=freq,
        task_type=task_type,  # type: ignore[arg-type]
        target_column=target_column,
        metadata=metadata,
    )
    await repo.create(user_id=user.id, state=state, input_path=input_path)
    await get_task_runner().enqueue(task_id)
    task = await repo.get(task_id)
    assert task is not None
    return task


def _parse_series_csv(content: bytes) -> tuple[list[float], list[Any]]:
    """CSV ряда: колонки (date, value) или только value.

    Заголовок определяется по первой строке: у CSV без заголовка первая точка
    ряда не должна превращаться в имена колонок.
    """
    first_line = content.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace")
    try:
        [float(cell) for cell in first_line.replace(";", ",").split(",") if cell.strip()]
        header = None  # все ячейки первой строки — числа: заголовка нет
    except ValueError:
        header = 0
    frame = pd.read_csv(io.BytesIO(content), header=header)
    if frame.shape[1] >= 2:
        dates = frame.iloc[:, 0].tolist()
        values = frame.iloc[:, 1].astype(float).tolist()
    else:
        dates, values = None, frame.iloc[:, 0].astype(float).tolist()
    if len(values) < 30:
        raise HTTPException(
            status_code=422, detail="Ряд слишком короткий: нужно не менее 30 точек"
        )
    return values, dates


# ============================================================================
# POST — создание задачи
# ============================================================================
@router.get("/")
async def list_my_forecasts(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Мои задачи: последние с статусом и выбранной моделью (страница «Мои прогнозы»)."""
    tasks = await TaskRepository().list_for_user(current_user.id, limit=limit)
    items = []
    for task in tasks:
        result = task.get("result") or {}
        state = task.get("state") or {}
        items.append(
            {
                "task_id": task["id"],
                "status": task["status"],
                "series_id": task["series_id"],
                "horizon": task["horizon"],
                "model_type": result.get("model_type") or state.get("model_type"),
                "mae": (result.get("metrics") or {}).get("mae"),
                "naive_mae": (result.get("naive_metrics") or {}).get("mae"),
                "city": (state.get("metadata") or {}).get("city"),
                "variable": (state.get("metadata") or {}).get("variable"),
                "hitl_requested": task["hitl_requested"],
                "iterations": task["iterations"],
                "error": task["error"],
                "created_at": task["created_at"],
                "finished_at": task["finished_at"],
            }
        )
    return {"tasks": items}


@router.post("/", response_model=ForecastAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_forecast_from_file(
    file: UploadFile = File(..., description="CSV ряда (date,value) или таблицы"),
    horizon: int = Form(14, ge=1, le=60),
    freq: str = Form("D"),
    task_type: str = Form("timeseries"),
    target_column: str | None = Form(None),
    city: str | None = Form(None),
    variable: str | None = Form(None),
    current_user: User = Depends(get_current_user),
) -> ForecastAccepted:
    """Загрузка ряда файлом: возвращает 202 и task_id для опроса статуса."""
    content = await file.read()
    repo = TaskRepository()
    task_id = str(uuid.uuid4())

    if task_type == "tabular":
        if not target_column:
            raise HTTPException(422, "Для tabular-задачи нужен target_column")
        input_path = repo.save_input_file(task_id, content, file.filename or "table.csv")
        metadata: dict[str, Any] = {"records_path": input_path}
    else:
        values, dates = _parse_series_csv(content)
        input_path = repo.save_input_series(task_id, values, dates)
        metadata = {"series_path": input_path, "city": city, "variable": variable}

    task = await _create_and_run(
        user=current_user,
        series_id=f"{(file.filename or 'series').split('.')[0]}_{task_id[:8]}",
        horizon=horizon,
        freq=freq,
        task_type=task_type,
        target_column=target_column,
        metadata=metadata,
        input_path=input_path,
    )
    return ForecastAccepted(
        task_id=task["id"],
        status=task["status"],
        status_url=f"/api/v1/forecast/{task['id']}/status",
    )


@router.post("/json", response_model=ForecastAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_forecast_inline(
    payload: ForecastCreateJson,
    current_user: User = Depends(get_current_user),
) -> ForecastAccepted:
    """Запуск прогноза по ряду в JSON (удобно для UI и тестов)."""
    repo = TaskRepository()
    task_id = str(uuid.uuid4())
    input_path = repo.save_input_series(task_id, payload.series_values, payload.series_dates)
    task = await _create_and_run(
        user=current_user,
        series_id=payload.series_id or f"inline_{task_id[:8]}",
        horizon=payload.horizon,
        freq=payload.freq,
        task_type=payload.task_type,
        target_column=payload.target_column,
        metadata={
            "series_path": input_path,
            "city": payload.city,
            "variable": payload.variable,
        },
        input_path=input_path,
    )
    return ForecastAccepted(
        task_id=task["id"],
        status=task["status"],
        status_url=f"/api/v1/forecast/{task['id']}/status",
    )


# ============================================================================
# GET — статус, результат, рассуждения
# ============================================================================
@router.get("/{task_id}/status", response_model=ForecastStatus)
async def get_forecast_status(
    task_id: str, current_user: User = Depends(get_current_user)
) -> ForecastStatus:
    task = _ensure_access(await TaskRepository().get(task_id), current_user)
    return _status_response(task)


@router.get("/{task_id}/result", response_model=ForecastResult)
async def get_forecast_result(
    task_id: str, current_user: User = Depends(get_current_user)
) -> ForecastResult:
    task = _ensure_access(await TaskRepository().get(task_id), current_user)
    if task["status"] not in _FINISHED:
        raise HTTPException(
            status_code=409,
            detail=f"Результат ещё не готов: статус задачи — {task['status']}",
        )
    result = dict(task.get("result") or {})
    result.setdefault("hitl_requested", task["hitl_requested"])
    return ForecastResult(task_id=task["id"], status=task["status"], **result)


@router.get("/{task_id}/reasoning", response_model=ReasoningResponse)
async def get_forecast_reasoning(
    task_id: str, current_user: User = Depends(get_current_user)
) -> ReasoningResponse:
    _ensure_access(await TaskRepository().get(task_id), current_user)
    tree = await fetch_reasoning_tree(task_id)

    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(LLMCall).where(LLMCall.request_id == task_id).order_by(LLMCall.created_at)
            )
        ).scalars().all()
    llm_calls = [
        {
            "agent_name": row.agent_name,
            "model_name": row.model_name,
            "provider": row.provider,
            "tokens_in": row.tokens_in,
            "tokens_out": row.tokens_out,
            "latency_ms": row.latency_ms,
            "status": row.status,
        }
        for row in rows
    ]
    return ReasoningResponse(task_id=task_id, tree=tree, llm_calls=llm_calls)


# ============================================================================
# POST — решение человека (HITL)
# ============================================================================
@router.post("/{task_id}/hitl", response_model=ForecastStatus)
async def resolve_hitl(
    task_id: str,
    request: HitlRequest,
    current_user: User = Depends(get_current_user),
) -> ForecastStatus:
    """Решение по задаче в waiting_for_human: принять, сменить модель или горизонт."""
    _ensure_access(await TaskRepository().get(task_id), current_user)
    try:
        task = await apply_hitl_resolution(task_id, request.model_dump())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if task["status"] == "pending":  # change_model / change_horizon — перезапуск
        await get_task_runner().enqueue(task_id)
        task = await TaskRepository().get(task_id)
        assert task is not None
    return _status_response(task)
