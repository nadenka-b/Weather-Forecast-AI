"""Настройки окружения для интеграционных тестов API.

DATABASE_URL задаётся ДО импорта приложения, чтобы таблицы создавались
в отдельном тестовом файле, а не в рабочей БД. Файл удаляется после прогона.
"""
import os
from pathlib import Path

TEST_DB = Path(__file__).resolve().parents[2] / "test_weather.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB}"
os.environ.setdefault("DEBUG", "false")

import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _cleanup_test_db():
    """Удаляет тестовую БД после всех тестов."""
    yield
    if TEST_DB.exists():
        TEST_DB.unlink()
