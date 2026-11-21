"""Healthcheck-роуты.

/healthz — liveness: жив ли процесс.
/health  — readiness: готов ли принимать запросы (проверяет БД через SELECT 1).
"""
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.core.config import get_settings
from backend.db.session import async_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness-проба: приложение запустилось."""
    settings = get_settings()
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


@router.get("/health")
async def health() -> JSONResponse:
    """Readiness-проба: доступна ли БД (200 — да, 503 — нет)."""
    settings = get_settings()

    db_ok, db_error = await check_database()

    payload = {
        "status": "healthy" if db_ok else "unhealthy",
        "app": settings.app_name,
        "debug": settings.debug,
        "services": {
            "database": {"healthy": db_ok, "error": db_error},
            # redis / qdrant / llm появятся на Этапах 10–11
        },
    }
    return JSONResponse(
        status_code=200 if db_ok else 503,
        content=payload,
    )


async def check_database() -> tuple[bool, str | None]:
    """Проверяет, что БД отвечает на простой запрос."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001 — любой сбой = нездоров
        logger.error("Database health check failed: %s", exc)
        return False, str(exc)
