"""Точка входа FastAPI (Этап 2: каркас приложения).

Роуты /api/v1/forecast, /hitl, /evaluate подключаются на Этапах 12–14.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Мультиагентная платформа прогнозирования погодных временных рядов",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness-проба: приложение отвечает."""
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


@app.get("/health")
async def health() -> dict:
    """Readiness-проба: статусы зависимостей (наполняется на Этапах 3–11)."""
    return {
        "status": "ok",
        "services": {
            "database": "not_configured",
            "redis": "not_configured",
            "qdrant": "not_configured",
            "llm": "not_configured",
        },
    }
