"""Celery-приложение (Этапы 12–14): брокер и backend результатов — Redis.

Запуск воркера и планировщика (продакшен, docker-compose):
    celery -A backend.worker.celery_app worker --loglevel=info --queues forecast
    celery -A backend.worker.celery_app beat --loglevel=info

beat: ежедневная очистка кэша (записи старше 30 дней; часто используемые
можно сохранить через max_hits — см. clear_cache_task).
"""
from celery import Celery

from backend.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "weather_forecast",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_default_queue="forecast",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # тяжёлые задачи (обучение моделей + LLM): мягкий и жёсткий лимит времени
    task_soft_time_limit=1800,
    task_time_limit=2100,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

from backend.worker.tasks import (  # noqa: E402
    CLEAR_CACHE_TASK_NAME,
    EVALUATE_TASK_NAME,
    FORECAST_TASK_NAME,
    REINDEX_TASK_NAME,
    clear_cache_task,
    evaluate_model_task,
    forecast_task,
    reindex_rag_task,
)

celery_app.task(name=FORECAST_TASK_NAME)(forecast_task)
celery_app.task(name=EVALUATE_TASK_NAME)(evaluate_model_task)
celery_app.task(name=REINDEX_TASK_NAME)(reindex_rag_task)
celery_app.task(name=CLEAR_CACHE_TASK_NAME)(clear_cache_task)

# Расписание (задание MLF: «переиндексация RAG, очистка кэша» как фоновые задачи)
celery_app.conf.beat_schedule = {
    "clear-old-cache-daily": {
        "task": CLEAR_CACHE_TASK_NAME,
        "schedule": 60 * 60 * 24,  # раз в сутки
        "kwargs": {"older_than_days": 30},
    },
}
