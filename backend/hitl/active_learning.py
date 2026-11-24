"""HITL-очередь и Active Learning (Этап 14, методичка Step_10).

Очередь «неуверенных» примеров — это задачи прогноза в статусе
waiting_for_human (Supervisor ставит hitl_requested, когда цикл не сошёлся
или Critic ниже порога). Эксперт (editor/admin) видит очередь и решает:

- confirm_correction — атомарно в одной транзакции: кейс добавляется в базу
  знаний (active learning: подтверждённый прогноз становится примером для
  RAG), задача переводится в completed, пишется строка hitl_feedback;
  опционально эксперт может указать другую модель/параметры — тогда прогноз
  сначала перевыполняется с ней;
- reject_example — задача отклоняется (rejected), база знаний не меняется.

Подтверждение запускает переиндексацию RAG (inline-раннер или Celery).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from backend.core.models import ForecastState
from backend.db.models import ForecastTask, HitlFeedback, KnowledgeCase

logger = logging.getLogger(__name__)


def _task_summary(task: ForecastTask) -> dict[str, Any]:
    """Компактное описание задачи для очереди (что нужно эксперту для решения)."""
    state = task.state or {}
    metadata = state.get("metadata") or {}
    supervisor = metadata.get("supervisor") or {}
    return {
        "task_id": task.id,
        "series_id": task.series_id,
        "horizon": task.horizon,
        "city": metadata.get("city"),
        "variable": metadata.get("variable"),
        "model_type": state.get("model_type"),
        "model_params": state.get("model_params"),
        "critic_score": state.get("critic_score"),
        "critic_reasoning": state.get("critic_reasoning"),
        "critic_suggestions": metadata.get("critic_suggestions") or [],
        "validator_errors": metadata.get("validator_errors") or [],
        "validation_passed": state.get("validation_passed"),
        "used_fallback": metadata.get("used_fallback"),
        "stopped_reason": supervisor.get("stopped_reason"),
        "iterations": supervisor.get("iterations", 0),
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }


def build_kb_case(task: ForecastTask, state: ForecastState) -> dict[str, Any] | None:
    """Кейс базы знаний из подтверждённой задачи (признаки — как в generate_knowledge_base).

    None, если у задачи нет диагностики ряда (нечего индексировать).
    """
    diagnostics = state.diagnostics or {}
    stats = diagnostics.get("stats") or {}
    adf = diagnostics.get("adf") or {}
    stl = diagnostics.get("stl") or {}
    if not stats:
        return None
    metadata = state.metadata or {}
    variable = metadata.get("variable") or state.target_column
    city = metadata.get("city")
    metrics = metadata.get("val_metrics") or {}
    return {
        "series_id": f"hitl|{task.series_id}|{task.id[:8]}",
        "task_type": state.task_type,
        "city": city,
        "freq": state.freq,
        "target_column": variable,
        "features": {
            "length": stats.get("length"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "acf_lag1": stats.get("acf_lag1"),
            "adf_pvalue": adf.get("adf_pvalue"),
            "stationary": adf.get("stationary"),
            "seasonal_period": diagnostics.get("seasonal_period"),
            "trend_strength": stl.get("trend_strength"),
            "seasonal_strength": stl.get("seasonal_strength"),
            "missing_rate": diagnostics.get("raw_missing_rate"),
            "variable": variable,
            "city": city,
        },
        "best_model": state.model_type or "naive_seasonal",
        "hyperparams": state.model_params or {},
        "metrics": metrics,
        "description": (
            f"Подтверждён экспертом (HITL): ряд {variable} ({city}), горизонт {state.horizon}; "
            f"модель {state.model_type}, MASE {metrics.get('mase')}, MAE {metrics.get('mae')}."
        ),
    }


class ActiveLearningManager:
    """Очередь задач на проверку человеком: просмотр, подтверждение, отклонение."""

    # ------------------------------------------------------------------
    # Очередь
    # ------------------------------------------------------------------
    @staticmethod
    async def get_pending_examples(
        limit: int = 50, offset: int = 0, city: str | None = None, variable: str | None = None
    ) -> list[dict[str, Any]]:
        """Задачи в статусе waiting_for_human (сначала новые)."""
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            query = (
                select(ForecastTask)
                .where(ForecastTask.status == "waiting_for_human")
                .order_by(ForecastTask.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            rows = (await session.execute(query)).scalars().all()
        examples = [_task_summary(row) for row in rows]
        if city:
            examples = [e for e in examples if e["city"] == city]
        if variable:
            examples = [e for e in examples if e["variable"] == variable]
        return examples

    @staticmethod
    async def count_pending() -> int:
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            return int(
                await session.scalar(
                    select(func.count(ForecastTask.id)).where(
                        ForecastTask.status == "waiting_for_human"
                    )
                )
                or 0
            )

    # ------------------------------------------------------------------
    # Подтверждение (active learning)
    # ------------------------------------------------------------------
    @staticmethod
    async def confirm_correction(
        task_id: str,
        user_id: str | None = None,
        correction: dict[str, Any] | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Подтвердить прогноз: кейс -> база знаний + completed + feedback (одна транзакция).

        correction: {"model_type": "fit_xgboost", "model_params": {...}} — прогноз
        перевыполняется с указанной моделью до сохранения кейса.
        """
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            task = await session.get(ForecastTask, task_id)
            if task is None:
                raise LookupError(f"Задача {task_id} не найдена")
            if task.status != "waiting_for_human":
                raise ValueError(
                    f"Задача {task_id} не ожидает подтверждения (статус {task.status})"
                )

        state = ForecastState(**(task.state or {}))

        if correction and (correction.get("model_type") or correction.get("model_params")):
            from backend.core.llm.tools import build_default_registry
            from backend.services.pipeline import perform_forecast

            tool = correction.get("model_type") or state.metadata.get("selected_tool")
            if tool and tool not in build_default_registry().names:
                raise ValueError(f"Неизвестная модель {tool!r}")
            if tool:
                state.metadata["forced_model"] = {
                    "tool": tool, "arguments": correction.get("model_params") or {}
                }
            state.hitl_requested = False
            state = await perform_forecast(state)  # перевыполнить с корректировкой
        state.hitl_requested = False  # решение человека: результат принят
        state.hitl_resolution = {"action": "accept", "correction": correction}

        kb_case = build_kb_case(task, state)
        kb_inserted = False
        async with AsyncSessionLocal() as session:
            try:
                if kb_case is not None:
                    exists = await session.execute(
                        select(KnowledgeCase.id).where(
                            KnowledgeCase.series_id == kb_case["series_id"]
                        )
                    )
                    if exists.scalar_one_or_none() is None:
                        session.add(KnowledgeCase(**kb_case))
                        kb_inserted = True
                task = await session.get(ForecastTask, task_id)
                assert task is not None
                task.status = "completed"
                task.hitl_requested = False
                task.finished_at = datetime.now(timezone.utc)
                task.state = state.model_dump(mode="json")
                task.result = {**(task.result or {}), "model_type": state.model_type,
                               "model_params": state.model_params,
                               "metrics": state.metadata.get("val_metrics"),
                               "predictions": state.metadata.get("predictions"),
                               "prediction_dates": state.metadata.get("prediction_dates"),
                               "hitl_requested": False}
                session.add(
                    HitlFeedback(
                        task_id=task_id, user_id=user_id, decision="confirm",
                        correction=correction, comment=comment,
                    )
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise

        # Переиндексация RAG — в фоне (inline asyncio-задача или Celery)
        try:
            from backend.worker.tasks import get_task_runner

            await get_task_runner().enqueue_reindex()
        except Exception as exc:  # noqa: BLE001 — переиндексация не должна ломать подтверждение
            logger.warning("RAG reindex not scheduled: %s", exc)

        logger.info("HITL confirm: task=%s kb_inserted=%s", task_id, kb_inserted)
        return {
            "task_id": task_id,
            "status": "completed",
            "kb_inserted": kb_inserted,
            "kb_case_id": kb_case["series_id"] if kb_case else None,
        }

    # ------------------------------------------------------------------
    # Отклонение
    # ------------------------------------------------------------------
    @staticmethod
    async def reject_example(
        task_id: str, user_id: str | None = None, reason: str | None = None
    ) -> dict[str, Any]:
        """Отклонить прогноз: статус rejected, база знаний не меняется."""
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            task = await session.get(ForecastTask, task_id)
            if task is None:
                raise LookupError(f"Задача {task_id} не найдена")
            if task.status != "waiting_for_human":
                raise ValueError(
                    f"Задача {task_id} не ожидает решения (статус {task.status})"
                )
            task.status = "rejected"
            task.finished_at = datetime.now(timezone.utc)
            session.add(
                HitlFeedback(task_id=task_id, user_id=user_id, decision="reject", comment=reason)
            )
            await session.commit()
        logger.info("HITL reject: task=%s", task_id)
        return {"task_id": task_id, "status": "rejected"}

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------
    @staticmethod
    async def get_stats() -> dict[str, int]:
        """{pending, confirmed, rejected} — одним запросом с CASE."""
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            confirmed = select(func.count(HitlFeedback.id)).where(
                HitlFeedback.decision == "confirm"
            ).scalar_subquery()
            rejected = select(func.count(HitlFeedback.id)).where(
                HitlFeedback.decision == "reject"
            ).scalar_subquery()
            pending = (
                select(func.count(ForecastTask.id))
                .where(ForecastTask.status == "waiting_for_human")
                .scalar_subquery()
            )
            row = (
                await session.execute(select(pending, confirmed, rejected))
            ).one()
        return {"pending": int(row[0]), "confirmed": int(row[1]), "rejected": int(row[2])}
