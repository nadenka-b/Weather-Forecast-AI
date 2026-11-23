# 🌤 Weather Forecast AI

Интеллектуальная система прогнозирования погодных временных рядов с **мультиагентной архитектурой**, **Agentic RAG** и строгой оценкой качества. Курсовой проект по дисциплине MLF-2026.

> **Статус:** Этапы 1–11 выполнены: каркас, БД (8 таблиц), Docker (db + app), JWT-аутентификация, мультиагентная система (DataAnalyst / ModelSelector c Tool Calling / Validator / Critic + Supervisor), база знаний 1000 кейсов из Jena Climate и Daily Climate, справочник 578 терминов, гибридный RAG (BM25 + векторы + RRF), реальные LLM через LM Studio с трассировкой вызовов. 200+ тестов зелёные.

---

## 📖 О проекте

Пользователь загружает метеорологический временной ряд (или выбирает встроенный датасет), указывает **город**, **целевую переменную** (среднесуточная температура или количество осадков) и **горизонт** (7–14 дней). Система:

1. анализирует данные (пропуски, сезонность, стационарность);
2. подбирает оптимальную модель прогнозирования через **Tool Calling** (от классических статистических до градиентного бустинга; бонус — TS Foundation Models);
3. обучает/инференсит, диагностирует остатки;
4. валидирует физическую реалистичность (климатические пределы) и возвращает обоснованный прогноз с метриками.

---

## 🤖 Как работает

### Мультиагентный пайплайн

`Supervisor` оркестрирует цикл (до `max_iterations = 3`), ведёт иерархический **Reasoning Log** (записи с `parent_id`, доступны через API) и при необходимости приостанавливает задачу для HITL:

| Агент | Ответственность |
|---|---|
| **DataAnalyst** | парсинг дат, проверка регулярности, ресемплинг, заполнение пропусков метеоданных, STL-декомпозиция, ADF-тест, выделение сезонности |
| **ModelSelector** | анализирует EDA + похожие кейсы из RAG, вызывает LLM с **Tool Calling** (не генерирует сырой код!) и выбирает модель + гиперпараметры из реестра инструментов |
| **Validator** | формальная проверка: NaN, длина горизонта, **климатические пределы** (температура не выходит за рамки климатической нормы города/месяца), статистические тесты остатков (Ljung-Box, Shapiro-Wilk) |
| **Critic** | LLM-as-a-Judge: адекватность выбора модели; особая проверка на **исторических аномалиях** (волны жары, заморозки) |
| **Editor** *(опц.)* | если Validator/Critic недовольны — корректирует гиперпараметры, добавляет лаги давления, влажности, ветра |

**Реестр инструментов** (`backend/core/llm/tools.py`): минимум `fit_arima`, `fit_prophet`, `fit_xgboost` + `fit_tabular_xgboost` (демонстрация расширяемости на табличные данные); бонус — `infer_chronos`. Fallback Tool Calling: 3 попытки, затем SARIMA с автоподбором порядка (`auto_arima`).

### Agentic RAG

- **Гибридный поиск**: BM25 (по текстовым описаниям кейсов) + векторный поиск (Qdrant; локально — FAISS), слияние через **Reciprocal Rank Fusion**.
- **Эмбеддинги рядов**: статистические признаки (тренд, сезонность, энтропия, автокорреляции; `tsfresh`) + снижение размерности PCA до 128 dims. Бонус — contextual retrieval.
- **База знаний**: 1000+ «успешных кейсов» (ряд → сетка моделей → лучшая модель + гиперпараметры + метрики + описание) и **справочник 500+ метеорологических терминов**.

### Human-in-the-Loop

Если цикл не сошёлся, Critic выставил критические оценки или обнаружены **экстремальные погодные условия** — задача уходит в очередь модерации. Эксперт может выбрать другую модель, изменить горизонт или принять результат; подтверждённые кейсы пополняют базу знаний (active learning) и запускают переиндексацию RAG.

### Режимы оценки и метрики

- Режимы: `baseline` / `+rag` / `+agents` / `+foundation` (ablation).
- Точность: **MAE, RMSE, MAPE, MASE, WAPE**; сравнение с **наивным (сезонным) прогнозом**.
- Статистика: Diebold-Mariano, Wilcoxon signed-rank, bootstrap-CI (бонус).
- Агентные метрики: среднее число итераций Supervisor, доля HITL, время, токены LLM.
- Анализ подвыборок 10/20/50/100% — кривая «точность vs размер обучающей выборки».

### Инженерия

- Кэш результатов LLM/прогнозов (Redis / таблица в БД), счётчик попаданий.
- Фоновые задачи: Celery + Redis (прогноз, batch-оценка, переиндексация RAG, очистка кэша).
- JWT-аутентификация с ролями (guest / user / editor / admin).
- Observability: каждый LLM-вызов логируется в `llm_calls` (модель, токены, latency, статус).

---

## 🧰 Стек

Python 3.12 + Poetry · FastAPI · SQLAlchemy (async) + PostgreSQL/SQLite · Alembic · Celery + Redis · Qdrant/FAISS + rank-bm25 · sentence-transformers / tsfresh / PCA · statsmodels (ARIMA/SARIMA, STL) · Prophet · XGBoost/LightGBM · pmdarima · LM Studio (OpenAI-совместимый сервер, Qwen2.5-7B-Instruct) / HuggingFace / внешние API · pytest (покрытие ≥ 70%) · Chainlit/Gradio · Docker Compose + Nginx · GitHub Actions (бонус).

---

## 👥 Команда и распределение работ

| Участник | Направление |
|---|---|
| **Балабанова Надежда** — ML-инжиниринг и данные | метеодатасеты, предобработка (пропуски, сезонность), feature extraction и эмбеддинги, генерация базы знаний и справочника метеотерминов, качество RAG, метрики и `run_eval.py`, отчёт с анализом |
| **Саттарова Аделина** — бэкенд и DevOps | FastAPI и REST API, БД и миграции, JWT/роли, агентный слой (base/Supervisor/Tool Calling/LLMManager), Celery/Redis, Docker Compose + Nginx, CI, UI |

Детальное распределение по этапам команда ведёт во внутреннем плане работ.

---

## 📁 Структура проекта

```text
Weather-Forecast-AI/
├── backend/
│   ├── agents/            # DataAnalyst, ModelSelector, Validator, Critic, Editor, Supervisor
│   ├── api/               # FastAPI: main, routes, schemas
│   ├── core/              # config, models (ForecastState), metrics, llm/ (manager, tools, providers)
│   ├── timeseries/        # preprocessing, classical (ARIMA/Prophet/XGBoost), foundation (бонус)
│   ├── data/              # knowledge_base, cache, embeddings
│   ├── db/                # SQLAlchemy, модели таблиц
│   ├── evaluation/        # прогоны оценки, агентные метрики
│   ├── hitl/              # active_learning, feedback
│   ├── retrieval/         # bm25_index, vector_index, hybrid_retriever (RRF)
│   ├── utils/             # logging, security (JWT), llm_logger
│   └── worker/            # Celery-задачи
├── frontend/              # Chainlit/Gradio (Этап 15)
├── scripts/               # download_data, generate_knowledge_base, import_data, run_eval
├── tests/                 # pytest
├── infrastructure/        # docker-compose, Dockerfile, nginx.conf
├── data/                  # датасеты (raw/processed, не в git)
├── pyproject.toml
└── run.py
```

---

## 🚀 Быстрый старт (локальный режим)

```bash
git clone https://github.com/nadenka-b/Weather-Forecast-AI.git && cd Weather-Forecast-AI
poetry install                          # зависимости (бонус-группа: poetry install --with foundation)
cp .env.example .env                    # настройки (по умолчанию SQLite, без внешних сервисов)
poetry run python run.py                # API на http://127.0.0.1:8000
```

Проверки: `GET /healthz` — liveness, `GET /health` — статусы зависимостей (БД + LLM),
`poetry run pytest` — тесты.

**Режимы LLM** (`LLM_PROVIDER` в `.env`): `lm_studio` — основной, локальный OpenAI-совместимый
сервер (LM Studio → Developer → Start Server, порт 1234, модель Qwen2.5-7B-Instruct);
`vllm`, `hf_local` (transformers), `api` (OpenRouter и др.) — альтернативы;
`stub` — без сети (тесты и офлайн-разработка: заглушки возвращают валидный Tool Call).
Каждый вызов LLM пишется в таблицу `llm_calls` (модель, токены, latency, статус).

Docker-стек Этапа 3 — PostgreSQL + приложение (оба сервиса проходят healthcheck):

```bash
cd infrastructure
cp .env.example .env                 # задайте JWT_SECRET
docker compose up -d --build
curl http://localhost:8000/health    # {"status":"healthy", ...}
```

LM Studio (локальный OpenAI-совместимый сервер: приложение LM Studio → Developer → Start Server,
порт 1234, модель Qwen2.5-7B-Instruct), а также Redis и Qdrant подключаются на Этапах 10–11;
полный стек (worker + beat + nginx) собирается на Этапе 15.

---

## 📊 Датасеты

| Датасет | Тип | Признаки |
|---|---|---|
| **Jena Climate** (Kaggle/UCI) | 10-мин метеоряд, 2009–2016 | T, давление, влажность, ветер и др. (~14) |
| `sumanthvrao/daily-climate-time-series-data` | дневной ряд (Дели) | meantemp, humidity, wind_speed, pressure (4 + лаги/календарь) |
| **Monash TSF** (weather) | наборы погодных рядов | зависит от набора |

Загрузка: `poetry run python scripts/download_data.py` (kagglehub). Для каждого датасета число признаков фиксируется в отчёте (минимум 5; при нехватке — лаги, скользящие статистики, календарные признаки).
