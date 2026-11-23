"""Конфигурация приложения (pydantic-settings).

Все параметры читаются из переменных окружения / файла .env (см. .env.example).
Локальная разработка по умолчанию использует SQLite и Qdrant/Redis по необходимости;
продакшен-значения задаются в infrastructure/docker-compose.yml (Этап 15).
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Настройки Weather Forecast AI."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Приложение
    app_name: str = "Weather Forecast AI"
    app_version: str = "0.1.0"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    debug: bool = True

    # База данных (local: SQLite, prod: PostgreSQL)
    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR / 'weather_forecast.db'}"

    # LLM: lm_studio | vllm | hf_local | api | stub (Этап 11)
    # LM Studio — локальный OpenAI-совместимый сервер (порт 1234 по умолчанию)
    llm_provider: str = "lm_studio"
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = "qwen2.5-7b-instruct"
    llm_max_fallback_attempts: int = 3
    # vLLM — OpenAI-совместимый сервер для инференса (бонус, продвинутый режим)
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model: str = "qwen2.5-7b-instruct"
    # HuggingFace Transformers — запасной локальный режим (Causal LM)
    hf_local_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Внешний API (OpenRouter и др., бонус): OpenAI-совместимый
    llm_api_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_api_model: str = "qwen/qwen2.5-7b-instruct"
    # Таймаут LLM-запроса: холодный старт модели может занять минуты
    llm_timeout: float = 300.0
    llm_max_retries: int = 3
    llm_retry_delay: float = 1.0

    # RAG (Этап 10)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "weather_cases"
    embedding_dim: int = 128
    rrf_k: int = 60

    # Фоновые задачи / кэш (Этапы 11–14)
    redis_url: str = "redis://127.0.0.1:6379/0"
    cache_enabled: bool = True

    # Агентный цикл
    max_iterations: int = 3
    critic_pass_threshold: float = 7.0  # Critic ниже порога -> итерация Editor / HITL

    # Данные
    data_dir: Path = BASE_DIR / "data"
    knowledge_base_min_records: int = 1000
    glossary_min_terms: int = 500

    # JWT (Этап 4)
    jwt_secret: str = "change-me-in-.env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60


@lru_cache
def get_settings() -> Settings:
    """Кэшированный доступ к настройкам."""
    return Settings()
