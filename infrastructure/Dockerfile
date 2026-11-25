# Образ приложения (методичка Step_3; финализирован на Этапе 15).
# Сборка из корня проекта: docker compose -f infrastructure/docker-compose.yml up -d --build

FROM python:3.12-slim

# build-essential: gcc/g++/make — для сборки C-расширений и CmdStan (Prophet);
# curl — для healthcheck (/health)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Poetry без виртуального окружения внутри контейнера
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# Зависимости отдельным слоем — кэшируется, пока не меняется pyproject.toml/poetry.lock
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --only main --no-interaction --no-ansi

# CmdStan для Prophet (prophet >= 1.2 не содержит скомпилированных моделей).
# Тяжёлый шаг (~2–3 минуты): отключается сборкой с --build-arg INSTALL_CMDSTAN=false
ARG INSTALL_CMDSTAN=true
RUN if [ "$INSTALL_CMDSTAN" = "true" ]; then \
        poetry run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"; \
    fi

# Код приложения
COPY backend/ ./backend/
COPY run.py ./

RUN mkdir -p /app/logs /app/data/processed

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
