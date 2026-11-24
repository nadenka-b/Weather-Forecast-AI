"""Celery-приложение (Этап 12): брокер и backend результатов — Redis.

Запуск воркера (продакшен, docker-compose):
    celery -A backend.worker.celery_app worker --loglevel=info --queues forecast
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
    EVALUATE_TASK_NAME,
    FORECAST_TASK_NAME,
    evaluate_model_task,
    forecast_task,
)

celery_app.task(name=FORECAST_TASK_NAME)(forecast_task)
celery_app.task(name=EVALUATE_TASK_NAME)(evaluate_model_task)
