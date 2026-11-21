"""Точка входа FastAPI.

Этап 2: каркас + CORS. Этап 3: lifespan с create_all и health с проверкой БД.
Этап 4: роутер auth (register/login/me). Прогнозные роуты /api/v1/* — Этапы 12–14.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.auth import router as auth_router
from backend.api.routes.health import router as health_router
from backend.core.config import get_settings
from backend.db.models import Base
from backend.db.session import async_engine
from backend.utils.logging import setup_logging

settings = get_settings()
logger = logging.getLogger(settings.app_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управляет ресурсами приложения: создаёт таблицы, закрывает пул соединений."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("%s starting...", settings.app_name)
    logger.info("Debug mode: %s", settings.debug)
    logger.info("Database: %s", settings.database_url.split("://")[0])

    # Создание таблиц БД при старте (идемпотентно; Alembic — при смене схемы)
    logger.info("Creating database tables...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")

    yield

    logger.info("%s stopped", settings.app_name)
    await async_engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Мультиагентная платформа прогнозирования погодных временных рядов",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(health_router)
