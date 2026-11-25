"""Healthcheck-роуты.

/healthz — liveness: жив ли процесс.
/health  — readiness: готов ли принимать запросы (БД через SELECT 1 + LLM).
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
    """Readiness-проба: БД и LLM доступны (200 — да, 503 — нет)."""
    settings = get_settings()

    db_ok, db_error = await check_database()
    llm_ok, llm_error, llm_provider = await check_llm()
    # redis и qdrant — опциональные сервисы: проверяются только если включены
    redis_ok, redis_error = await check_redis()
    qdrant_ok, qdrant_error = await check_qdrant()

    healthy = db_ok and llm_ok and redis_ok and qdrant_ok
    payload = {
        "status": "healthy" if healthy else "unhealthy",
        "app": settings.app_name,
        "debug": settings.debug,
        "services": {
            "database": {"healthy": db_ok, "error": db_error},
            "llm": {"provider": llm_provider, "healthy": llm_ok, "error": llm_error},
            "redis": {"healthy": redis_ok, "error": redis_error},
            "qdrant": {"healthy": qdrant_ok, "error": qdrant_error},
            "tasks": {"backend": settings.task_backend, "healthy": True},
        },
    }
    return JSONResponse(status_code=200 if healthy else 503, content=payload)


async def check_database() -> tuple[bool, str | None]:
    """Проверяет, что БД отвечает на простой запрос."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001 — любой сбой = нездоров
        logger.error("Database health check failed: %s", exc)
        return False, str(exc)


async def check_llm() -> tuple[bool, str | None, str]:
    """Доступность LLM-провайдера: (healthy, error, имя провайдера)."""
    provider_name = get_settings().llm_provider.lower()
    if provider_name == "stub":
        return True, None, "stub"
    try:
        from backend.core.llm.manager import get_provider

        provider = get_provider(provider_name)
        if provider is None:
            return False, "unknown provider", provider_name
        ok, error = await provider.check_health()
        return ok, error, provider_name
    except Exception as exc:  # noqa: BLE001 — отсутствие пакета/сети = нездоров
        logger.error("LLM health check failed: %s", exc)
        return False, str(exc), provider_name


async def check_redis() -> tuple[bool, str | None]:
    """Redis: проверяется только при TASK_BACKEND=celery, иначе считаетсяhealthy."""
    if get_settings().task_backend.lower() != "celery":
        return True, None
    try:
        from redis import asyncio as aioredis

        client = aioredis.from_url(get_settings().redis_url, socket_connect_timeout=2)
        try:
            await client.ping()
        finally:
            await client.aclose()
        return True, None
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis health check failed: %s", exc)
        return False, str(exc)[:200]


async def check_qdrant() -> tuple[bool, str | None]:
    """Qdrant: проверяется только при VECTOR_BACKEND=qdrant."""
    if get_settings().vector_backend.lower() != "qdrant":
        return True, None
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=get_settings().qdrant_url, timeout=2)
        client.get_collections()
        return True, None
    except Exception as exc:  # noqa: BLE001
        logger.error("Qdrant health check failed: %s", exc)
        return False, str(exc)[:200]
