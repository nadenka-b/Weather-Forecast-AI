"""Асинхронное подключение к базе данных через SQLAlchemy.

- async_engine — пул соединений с БД (создаётся один раз при импорте);
- AsyncSessionLocal — фабрика сессий (новая сессия на каждый запрос);
- get_async_session — FastAPI-зависимость для получения сессии в роутах.

Ловушка «Future attached to a different loop»: engine привязан к event loop,
в котором создан. Один engine на процесс — при подключении Celery (Этап 11)
не создавать новый loop на каждую задачу.
"""
import logging
import weakref
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
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


_ready_engine_ref: "weakref.ReferenceType[AsyncEngine] | None" = None


async def ensure_tables() -> None:
    """Создаёт отсутствующие таблицы (идемпотентно)."""
    from backend.db.models import Base

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ensure_tables_once() -> None:
    """ensure_tables один раз на каждый engine.

    Вызывается на старте API (lifespan), в воркере и в скриптах: схема БД
    дополняется новыми таблицами, даже если сама БД создана раньше.
    Флаг привязан к engine через слабую ссылку — смена engine (тесты,
    переподключение) запускает проверку заново.
    """
    global _ready_engine_ref
    if _ready_engine_ref is None or _ready_engine_ref() is not async_engine:
        await ensure_tables()
        _ready_engine_ref = weakref.ref(async_engine)
