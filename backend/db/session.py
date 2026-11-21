"""Асинхронное подключение к базе данных через SQLAlchemy.

- async_engine — пул соединений с БД (создаётся один раз при импорте);
- AsyncSessionLocal — фабрика сессий (новая сессия на каждый запрос);
- get_async_session — FastAPI-зависимость для получения сессии в роутах.

Ловушка «Future attached to a different loop»: engine привязан к event loop,
в котором создан. Один engine на процесс — при подключении Celery (Этап 11)
не создавать новый loop на каждую задачу.
"""
import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Для SQLite пул настраивать не нужно, для PostgreSQL — пул с запасом.
engine_kwargs: dict = {"echo": settings.debug, "pool_pre_ping": True}
if settings.database_url.startswith("postgresql"):
    engine_kwargs.update({"pool_size": 10, "max_overflow": 20})

async_engine = create_async_engine(settings.database_url, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-зависимость: выдаёт сессию и закрывает её после ответа."""
    async with AsyncSessionLocal() as session:
        yield session
