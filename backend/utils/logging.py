"""Логирование приложения: консоль + файлы в logs/ (Этап 2, доопределено на Этапе 3).

- app_YYYYMMDD.log — все записи уровня INFO и выше;
- errors_YYYYMMDD.log — только ERROR и выше;
- DEBUG включается флагом DEBUG в .env (шумно, только для отладки).
"""
import logging
import sys
from datetime import date

from backend.core.config import BASE_DIR, get_settings

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging() -> logging.Logger:
    """Настраивает root-логгер (консоль + файлы). Идемпотентно."""
    settings = get_settings()

    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    if root.handlers:  # уже настроено
        return root

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    app_file = logging.FileHandler(
        logs_dir / f"app_{date.today():%Y%m%d}.log", encoding="utf-8"
    )
    app_file.setFormatter(formatter)
    root.addHandler(app_file)

    error_file = logging.FileHandler(
        logs_dir / f"errors_{date.today():%Y%m%d}.log", encoding="utf-8"
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    root.addHandler(error_file)

    return logging.getLogger(settings.app_name)
