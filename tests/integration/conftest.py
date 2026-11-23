"""Настройки окружения для интеграционных тестов API.

DATABASE_URL задаётся ДО импорта приложения, чтобы таблицы создавались
в отдельном тестовом файле, а не в рабочей БД. Файл удаляется после прогона.
"""
import asyncio
import os
from pathlib import Path

TEST_DB = Path(__file__).resolve().parents[2] / "test_weather.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB}"
os.environ.setdefault("DEBUG", "false")

import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _cleanup_test_db():
    """Удаляет тестовую БД после всех тестов.

    Сначала закрываем пул соединений: на Windows открытый в пуле SQLite-файл
    заблокирован, и unlink падает с PermissionError.
    """
    yield
    try:
        asyncio.run(_dispose_engine())
    except Exception:  # noqa: BLE001 — teardown не должен ронять прогон
        pass
    try:
        TEST_DB.unlink(missing_ok=True)
    except PermissionError:
        pass


async def _dispose_engine() -> None:
    from backend.db import session as db_session_module

    await db_session_module.async_engine.dispose()
