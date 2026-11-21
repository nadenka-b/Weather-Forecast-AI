# Образ приложения Weather Forecast AI (Этап 3; финализируется на Этапе 15).
# Сборка из корня проекта: docker compose -f infrastructure/docker-compose.yml up -d --build

FROM python:3.12-slim

WORKDIR /app

# Системные зависимости: gcc для сборки C-расширений, curl для healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry без виртуального окружения внутри контейнера
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# Зависимости отдельным слоем — кэшируется, пока не меняется pyproject.toml
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --only main --no-interaction --no-ansi

# Код приложения
COPY backend/ ./backend/
COPY run.py ./

RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
