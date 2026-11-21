"""Локальный запуск API: python run.py (uvicorn с reload при DEBUG=true)."""
import uvicorn

from backend.core.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
