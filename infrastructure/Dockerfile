# Базовый образ приложения (финализируется на Этапе 15, методичка Step_12).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry install --with dev --no-root

COPY . .

CMD ["poetry", "run", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
