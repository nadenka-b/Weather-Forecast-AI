# 🌤 Weather Forecast AI — мультиагентная платформа прогнозирования погоды

**Интеллектуальная система прогнозирования погодных временных рядов с мультиагентной архитектурой, Agentic RAG и Human-in-the-Loop.**

Пользователь загружает метеорологический ряд, указывает город, переменную и горизонт (7–14 дней).
Система анализирует данные, подбирает модель через Tool Calling (без генерации кода), строит прогноз,
проверяет его физическую реалистичность и объясняет выбор; сомнительные прогнозы уходят эксперту.

## 📋 О проекте

Курсовой проект по заданию MLF-2026 («Разработка интеллектуальной системы прогнозирования
временных рядов с мультиагентной архитектурой, Agentic RAG и оценкой качества»), тема —
прогноз погоды (температура, осадки).

Оркестрация — чистый Python без тяжёлых агентных фреймворков: состояние (`ForecastState`,
Pydantic) передаётся между агентами только по ссылке (`task_id`), тяжёлые объекты в payload
не передаются. LLM используется через **Tool Calling**: `ModelSelector` выбирает инструмент
и гиперпараметры из реестра, но никогда не пишет код.

## ✨ Возможности

- **Мультиагентный пайплайн**: DataAnalyst → ModelSelector → Validator → Critic → (Editor) под
  управлением Supervisor (адаптивный бюджет итераций, rollback при регрессии, HITL при несходимости).
- **Agentic RAG**: гибридный поиск похожих кейсов (BM25 + векторы статистических признаков,
  RRF) по базе знаний из 1000 кейсов; few-shot примеры попадают в промпт ModelSelector.
- **Реестр инструментов**: ARIMA/SARIMA (автоподбор порядка), Prophet, XGBoost на лагах,
  XGBoost для табличных задач; fallback — 3 попытки Tool Calling, затем SARIMA-автоподбор.
- **Валидация**: климатические пределы, NaN и длина прогноза, Ljung-Box и Shapiro-Wilk на остатках.
- **Critic (LLM-as-a-Judge)** и **Editor**: правка гиперпараметров по замечаниям с переобучением.
- **Human-in-the-Loop**: очередь на проверку, подтверждение (кейс → база знаний, active learning)
  и отклонение; решение с правкой модели перевыполняет прогноз.
- **Оценка качества**: `run_eval.py` (режимы × горизонты × доли данных, агентные метрики,
  Diebold-Mariano / Wilcoxon / bootstrap-CI / t-test), API сравнения моделей.
- **Observability**: иерархическое дерево рассуждений (`reasoning_logs`, `parent_id`) и
  трассировка каждого вызова LLM (`llm_calls`: модель, токены, latency).
- **Инфраструктура**: REST API (FastAPI, 202 + опрос статуса), фоновые задачи (inline или
  Celery + Redis + beat), кэш LLM и прогнозов, JWT-аутентификация с ролями, Streamlit UI, Docker Compose + Nginx.

## 🏗️ Архитектура

```text
                ┌─────────────────────────────────────────────────────────────┐
                │                     FastAPI (app)                           │
                │  /auth  /api/v1/forecast  /hitl  /api/v1/evaluate  /admin   │
                └───────────────┬─────────────────────────────┬───────────────┘
                                │ task_id (202)               │
                     ┌──────────▼──────────┐      ┌──────────▼──────────┐
                     │  TaskRunner         │      │  Streamlit UI       │
                     │  inline | Celery    │      │  (на хосте, :8501)  │
                     └──────────┬──────────┘      └─────────────────────┘
                                │ run_forecast(task_id)
                ┌───────────────▼───────────────────────────────────────────┐
                │                Supervisor (до 3 итераций)                 │
                │                                                               │
                │  DataAnalyst ─► ModelSelector ─► Validator ─► Critic       │
                │       (EDA,        │      ▲         (NaN,      (LLM-        │
                │        STL, ADF)   │      │          климат,    Judge)     │
                │                   │      │          остатки)      │        │
                │              Tool Calling  │                        ▼        │
                │                   │      └────────── Editor (правка +      │
                │                   ▼                 переобучение)           │
                │           Реестр инструментов: fit_arima / fit_prophet /    │
                │           fit_xgboost / fit_tabular_xgboost                │
                └──────┬──────────────────┬──────────────────┬────────────────┘
                       │                  │                  │
             ┌─────────▼──────┐  ┌────────▼───────┐  ┌───────▼─────────────┐
             │ LLM (LM Studio)│  │ Agentic RAG    │  │ PostgreSQL / SQLite │
             │ tool_calls     │  │ BM25 + векторы │  │ reasoning_logs,     │
             │ llm_calls      │  │ + RRF          │  │ llm_calls, БЗ, кэш  │
             └────────────────┘  └────────────────┘  └─────────────────────┘
                                                       │
                                 ┌─────────────────────▼─────────────────────┐
                                 │ waiting_for_human → HITL-очередь →        │
                                 │ confirm: кейс → база знаний → reindex RAG │
                                 └───────────────────────────────────────────┘
```

Ключевые файлы:

- `backend/agents/` — `BaseAgent`, `Supervisor`, DataAnalyst, ModelSelector, Validator, Critic, Editor
- `backend/services/pipeline.py` — центральный `perform_forecast` (агенты → финальный прогноз → валидация)
- `backend/core/llm/` — `LLMManager` (роутинг провайдеров, Tool Calling), реестр инструментов, провайдеры
- `backend/retrieval/` — BM25, векторный индекс (in-memory / Qdrant), RRF
- `backend/hitl/` — `ActiveLearningManager`, `FeedbackHandler`
- `backend/evaluation/` — ядро оценки, статистические тесты, сервис запусков
- `backend/worker/` — Celery-приложение, 4 задачи, постоянный event loop
- `streamlit_app/` — интерфейс (8 страниц)
- `infrastructure/` — Docker Compose (7 сервисов), Dockerfile, Nginx
- `scripts/` — загрузка данных, генерация базы знаний и справочника, `run_eval.py`, `report_tables.py`

## 🚀 Быстрый старт

### Требования

- Python 3.12, Poetry 2.x
- LM Studio с моделью Qwen2.5-7B-Instruct (или внешний API); без LLM доступен режим `stub`
- Docker Desktop / Docker Compose v2 — для полного стека (вариант B)

### Вариант A — локальный запуск

```bash
git clone https://github.com/nadenka-b/Weather-Forecast-AI.git
cd Weather-Forecast-AI
poetry install
cp .env.example .env                        # LM_STUDIO_BASE_URL, JWT_SECRET и др.

# данные, база знаний и справочник
poetry run python scripts/download_data.py                              # Jena Climate, Daily Climate
poetry run python scripts/generate_knowledge_base.py --target-cases 1000  # окна -> сетка моделей -> кейсы
poetry run python scripts/generate_glossary.py                          # 578 метеотерминов
poetry run python scripts/import_data.py                                # в БД (без дублей)

poetry run python run.py                     # API: http://localhost:8000/docs
```

Без запущенного LM Studio: `LLM_PROVIDER=stub poetry run python run.py` — LLM заменена
заглушками (выбирается ARIMA(1,1,1), Critic ставит 5.0, задачи попадают в HITL-очередь).

### Вариант B — Docker Compose (полный стек)

```bash
cp infrastructure/.env.example infrastructure/.env   # задайте JWT_SECRET, LM_STUDIO_BASE_URL
cd infrastructure
docker compose up -d --build
```

Сервисы: `db` (PostgreSQL 16), `redis`, `qdrant`, `app`, `worker`, `beat`, `nginx`.
LM Studio остаётся на хосте (Developer → Start Server, порт 1234) и доступна контейнерам
через `host.docker.internal`. После сборки:

- API и Swagger: http://localhost/docs (через Nginx)
- Здоровье сервисов: http://localhost/health

### Первый администратор

```bash
poetry run python scripts/create_admin.py --email admin@example.com --username admin --password Admin1234
```

### Интерфейс

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py --server.port 8501
```

## 📊 Данные

| Источник | Объём | Частота | Переменные |
|---|---|---|---|
| **Jena Climate** (UCI, 2009–2016) | 420 451 наблюдений, 14 переменных → 2921 день | 10 мин → D | temperature, pressure, humidity, wind_speed, dew_point |
| **Daily Climate** (Kaggle, Дели) | 1462 наблюдения, 4 переменные | D | temperature, humidity |
| **База знаний** | 1000 кейсов (Jena 910, Delhi 90): окна 365 дней с шагом 10 → сетка моделей → лучший по MASE | D | 14 признаков на кейс |
| **Справочник** | 578 метеорологических, климатологических и статистических терминов | — | — |
| **Holdout-набор** | 238 окон, не пересекается с базой знаний (зазор по времени) | D | те же ряды |

## 🧪 Тесты

```bash
poetry run pytest tests -q --cov=backend        # покрытие ≥ 70%
poetry run ruff check backend tests scripts streamlit_app
```

Тесты не требуют LLM, Redis и Qdrant: LLM работает в режиме `stub`, провайдеры проверяются
через `httpx.MockTransport`, Qdrant — через in-memory клиент. CI (GitHub Actions) ставит
лёгкое подмножество зависимостей из `requirements-ci.txt` и запускает линтер и тесты
с порогом покрытия 70%.

## 🎯 Режимы оценки

| Режим | LLM (Tool Calling) | RAG | Цикл агентов | Что измеряет |
|---|---|---|---|---|
| `baseline` | ✗ | ✗ | ✗ | классика без LLM: сетка моделей, лучшая по валидации |
| `rag` | ✗ | ✓ | ✗ | выбор модели по ближайшему кейсу базы знаний |
| `agents` | ✓ | ✓ | ✓ | полный цикл агентов + финальный прогноз |
| `foundation` | — | — | — | TS Foundation Models (Chronos/TimesFM; пока не установлены — пропуск) |

```bash
poetry run python scripts/run_eval.py --num-series 50 --horizons 7,14 --data-fractions 1.0
poetry run python scripts/report_tables.py          # таблицы и графики в reports/
```

Сводные таблицы (`reports/tables/*.md`) и графики (`reports/plots/*.png`) генерируются скриптом `report_tables.py`; результаты прогона и их анализ приведены в отчёте по проекту.

## 🔌 API

| Метод | Путь | Роль | Назначение |
|---|---|---|---|
| POST | `/auth/register`, `/auth/login` | — | регистрация, вход (JWT) |
| GET | `/auth/me` | user | данные пользователя |
| POST | `/api/v1/forecast/` (CSV), `/api/v1/forecast/json` | user | запуск прогноза → 202 + `task_id` |
| GET | `/api/v1/forecast/` | user | мои задачи |
| GET | `/api/v1/forecast/{id}/status`, `/result` | user | статус и результат |
| GET | `/api/v1/forecast/{id}/reasoning` | user | дерево рассуждений + вызовы LLM |
| POST | `/api/v1/forecast/{id}/hitl` | user | accept / change_model / change_horizon |
| GET | `/hitl/queue`, `/hitl/stats`, `/hitl/feedback` | editor | очередь на проверку |
| POST | `/hitl/{id}/confirm`, `/hitl/{id}/reject` | editor | подтверждение (кейс → БЗ) / отклонение |
| GET | `/api/v1/evaluate/compare` | editor | сравнение моделей на одной выборке |
| POST | `/api/v1/evaluate/batch` | editor | пакетная оценка → 202 + `run_id` |
| GET | `/api/v1/evaluate/runs/{id}`, `/results`, `/comparisons` | user | результаты оценки |
| GET | `/kb/search`, `/kb/cases`, `/glossary/terms` | user | база знаний и справочник |
| POST/DELETE | `/admin/kb/*`, `/admin/glossary/*` | admin | правка базы знаний и справочника |
| GET/DELETE | `/admin/cache/stats`, `/admin/cache/clear` | admin | кэш |
| GET | `/health`, `/healthz` | — | состояние сервисов |

Полная интерактивная документация — Swagger UI (`/docs`).

## 🖥️ Веб-интерфейс (Streamlit)

Запускается отдельно на хосте: это dev-инструмент, в Docker Compose его нет. Меню зависит от роли:

| Страница | user | editor | admin | Что на ней |
|---|---|---|---|---|
| Главная | ✓ | ✓ | ✓ | состояние сервисов, быстрый прогноз, навигация |
| Прогноз | ✓ | ✓ | ✓ | CSV или ввод ряда, статус задачи, график факт/прогноз, метрики, дерево рассуждений |
| Мои прогнозы | ✓ | ✓ | ✓ | список задач, переход к результату и логам |
| Логи рассуждений | ✓ | ✓ | ✓ | пошаговое дерево агентов, трассировка LLM-вызовов |
| Оценка качества | ✓ | ✓ | ✓ | запуск (editor+), таблицы режимов, графики Plotly, значимость |
| База знаний | ✓ | ✓ | ✓ | поиск похожих кейсов, справочник (правка — admin) |
| Сравнение моделей | — | ✓ | ✓ | MAE/RMSE моделей, победитель, Wilcoxon и bootstrap-CI |
| HITL-очередь | — | ✓ | ✓ | очередь, правка, подтверждение/отклонение |
| Состояние системы | — | — | ✓ | сервисы, кэш и его очистка, переиндексация RAG |

## ⚠️ Известные ограничения

- **Prophet требует CmdStan** (`poetry run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`).
  Без него (а на Windows с нелатинской локалью `cmdstanpy` не может загрузить бэкенд) инструмент
  возвращает понятную ошибку, остальные модели работают.
- **TS Foundation Models не установлены** — режим `foundation` пропускается.
- **Векторный индекс**: в режиме `memory` индекс пересобирается на каждую задачу (~1 с на 1000 кейсов);
  в режиме `qdrant` переиндексация полная (пересоздание коллекции), не инкрементальная.
- **Celery-воркер**: `--concurrency=1` — одна тяжёлая задача одновременно.
- **BM25** лексический: совпадают точные формы слов, без стемминга.
- **MAPE** нестабилен для рядов около нуля (точка росы, зимняя температура) — опираться на WAPE и MASE.
- **Климатические пределы** считаются по истории самого ряда, а не по климатическим нормам станции.
- **Оценка в offline-режиме**: без LLM (`stub`) режим `agents` всегда выбирает ARIMA(1,1,1),
  поэтому его метрики отражают работу пайплайна, а не качество выбора модели LLM.
- **Streamlit не в Docker Compose** — запускается на хосте отдельно.

## 🛠 Стек

Python 3.12 · Poetry · FastAPI · SQLAlchemy (async) · PostgreSQL / SQLite · Celery + Redis + beat ·
Qdrant (или in-memory) + rank-bm25 · statsmodels (ARIMA/SARIMA, STL) · Prophet · XGBoost · scikit-learn (PCA) ·
LM Studio (OpenAI-совместимый API) / HuggingFace Transformers / внешние API · Streamlit · Plotly ·
Docker Compose + Nginx · pytest · GitHub Actions.

## 👥 Команда проекта

| Участник | Направление |
|---|---|
| **Балабанова Надежда** | ML-инжиниринг и данные: датасеты, предобработка рядов, база знаний и справочник, признаки и эмбеддинги, гибридный RAG, метрики, `run_eval.py`, статистические тесты, отчёт |
| **Саттарова Аделина** | Бэкенд и DevOps: FastAPI и REST API, БД, JWT и роли, агентный слой (Supervisor, Tool Calling, Critic, Editor), фоновые задачи, HITL-роуты, кэш, Streamlit UI, Docker Compose + Nginx, CI |
