"""Общие настройки тестового окружения.

- LLM_PROVIDER=stub: тесты не ходят в сеть — LLMManager и Critic работают
  на заглушках. Провайдеры тестируются отдельно через httpx.MockTransport.
- DATABASE_URL: любая связь с БД вне in-memory фикстур идёт в отдельный
  тестовый файл, а не в рабочую БД проекта (защита от загрязнения при
  запуске отдельных файлов).

Обе переменные выставляются ДО импорта backend (конфиг читается при импорте).
"""
import os
from pathlib import Path

TEST_DB = Path(__file__).resolve().parent.parent / "test_weather.db"

os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{TEST_DB}")
