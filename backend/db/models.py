"""SQLAlchemy-модели базы данных Weather Forecast AI.

Каждый класс = таблица, атрибуты = колонки. Восемь таблиц по заданию MLF:
users, knowledge_base, glossary, cache, evaluation_results, model_comparisons,
reasoning_logs (иерархия через parent_id), llm_calls.

JSON-поля используют переносимый тип sqlalchemy.JSON (работает и в SQLite,
и в PostgreSQL).
"""
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


# ============================================================================
# Users — пользователи системы
# ============================================================================
class User(Base):
    """Пользователь системы.

    Роли:
    - user   — может запускать прогнозы;
    - editor — дополнительно обрабатывает HITL-очередь и пополняет базу знаний;
    - admin  — полный доступ.
    """

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_username", "username"),
        Index("ix_users_role", "role"),
    )

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role})>"


# ============================================================================
# KnowledgeBase — база знаний «успешных кейсов» прогнозирования
# ============================================================================
class KnowledgeCase(Base):
    """Успешный кейс: ряд -> лучшая модель + гиперпараметры + метрики.

    Погодная специализация: поля city и freq позволяют фильтровать кейсы
    по городу и частоте наблюдений при поиске похожих рядов (Этап 10).
    """

    __tablename__ = "knowledge_base"

    id = Column(String, primary_key=True, default=_uuid)
    series_id = Column(String, nullable=False)  # например "jena_2013_temp"
    task_type = Column(String, default="timeseries", nullable=False)  # timeseries | tabular
    city = Column(String, nullable=True)        # метео: город наблюдений
    freq = Column(String, nullable=True)        # например "D", "10min"
    target_column = Column(String, nullable=True)  # для tabular-кейсов
    features = Column(JSON, nullable=True)      # статистические признаки ряда
    embedding = Column(JSON, nullable=True)     # вектор 128 dims (PCA)
    best_model = Column(String, nullable=False)  # "arima" | "prophet" | "xgboost" | ...
    hyperparams = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)       # {"mase": 0.85, "smape": 0.12}
    description = Column(Text, nullable=True)   # текстовое описание для BM25

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("series_id", "task_type", name="uq_kb_series"),
        Index("ix_kb_task_type", "task_type"),
        Index("ix_kb_city_freq", "city", "freq"),
        Index("ix_kb_best_model", "best_model"),
    )


# ============================================================================
# Glossary — справочник метеорологических терминов (>=500 записей)
# ============================================================================
class GlossaryTerm(Base):
    """Термин справочника: определение, формула, советы по тюнингу."""

    __tablename__ = "glossary"

    id = Column(String, primary_key=True, default=_uuid)
    term = Column(String, nullable=False)
    definition = Column(Text, nullable=False)
    category = Column(String, nullable=True)  # например "метеорология", "статистика"

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("term", name="uq_glossary_term"),
        Index("ix_glossary_category", "category"),
    )


# ============================================================================
# Cache — кэш результатов LLM и прогнозов
# ============================================================================
class ForecastCache(Base):
    """Кэш: ключ (hash входных данных + параметры задачи) -> результат.

    Уникальность включает model_used: один и тот же вход может быть
    прокэширован для разных моделей (ollama / api / arima / prophet).
    """

    __tablename__ = "cache"

    id = Column(String, primary_key=True, default=_uuid)
    input_key = Column(String, nullable=False)  # hash(series_id + horizon + params)
    model_used = Column(String, nullable=False)
    output = Column(Text, nullable=False)       # сериализованный результат
    hits = Column(Integer, default=1, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("input_key", "model_used", name="uq_cache_key_model"),
        Index("ix_cache_created", "created_at"),
        Index("ix_cache_hits", "hits"),
    )


# ============================================================================
# EvaluationResults — результаты run_eval по одной конфигурации
# ============================================================================
class EvaluationResult(Base):
    """Результат оценки: модель + режим + размер выборки -> метрики."""

    __tablename__ = "evaluation_results"

    id = Column(String, primary_key=True, default=_uuid)
    model_name = Column(String, nullable=False)
    mode = Column(String, nullable=False)        # baseline | rag | agents | foundation
    sample_size = Column(Integer, nullable=True)  # число рядов
    horizon = Column(Integer, nullable=True)
    data_fraction = Column(Float, nullable=True)  # доля обучающих данных (0.1..1.0)
    metrics = Column(JSON, nullable=True)
    agent_metrics = Column(JSON, nullable=True)   # итерации, HITL, токены, время
    evaluation_time = Column(Float, nullable=True)  # секунды

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_eval_mode", "mode"),
        Index("ix_eval_model", "model_name"),
    )


# ============================================================================
# ModelComparisons — попарные/групповые сравнения моделей
# ============================================================================
class ModelComparison(Base):
    """Сравнение набора моделей на одной выборке: результаты + победитель."""

    __tablename__ = "model_comparisons"

    id = Column(String, primary_key=True, default=_uuid)
    models_compared = Column(JSON, nullable=False)  # ["arima", "prophet", ...]
    sample_size = Column(Integer, nullable=True)
    results = Column(JSON, nullable=True)           # метрики каждой модели
    winner = Column(String, nullable=True)
    significance = Column(JSON, nullable=True)      # DM/Wilcoxon/bootstrap (бонус)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_comparison_created", "created_at"),
    )


# ============================================================================
# ForecastTasks — задачи прогноза (Этап 12; расширение схемы сверх 8 таблиц задания)
# ============================================================================
class ForecastTask(Base):
    """Задача прогноза: state хранится по ссылке (task_id), не в payload.

    Статусы (задание MLF): pending -> running -> waiting_for_human | completed | failed.
    """

    __tablename__ = "forecast_tasks"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status = Column(String, default="pending", nullable=False)
    series_id = Column(String, nullable=False)
    horizon = Column(Integer, nullable=False)
    task_type = Column(String, default="timeseries", nullable=False)
    target_column = Column(String, nullable=True)
    input_path = Column(String, nullable=True)   # CSV ряда (date,value)
    state = Column(JSON, nullable=True)          # снимок ForecastState
    result = Column(JSON, nullable=True)         # результат для /result
    error = Column(Text, nullable=True)
    hitl_requested = Column(Boolean, default=False, nullable=False)
    iterations = Column(Integer, default=0, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_forecast_tasks_status", "status"),
    )


# ============================================================================
# EvaluationRuns — фоновые запуски оценки (Этап 13; POST /api/v1/evaluate/batch)
# ============================================================================
class EvaluationRun(Base):
    """Пакетный запуск оценки: параметры, сводка, сравнения, статус."""

    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True, default=_uuid)
    status = Column(String, default="pending", nullable=False)  # pending|running|completed|failed
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    params = Column(JSON, nullable=True)
    summary = Column(JSON, nullable=True)
    comparisons = Column(JSON, nullable=True)
    n_records = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_evaluation_runs_status", "status"),
    )


# ============================================================================
# ReasoningLogs — иерархический лог рассуждений агентов
# ============================================================================
class ReasoningLog(Base):
    """Запись дерева рассуждений: parent_id задаёт иерархию шагов.

    Supervisor (родитель) -> вызовы агентов (дети) -> tool-вызовы (внуки).
    Дерево восстанавливается запросом по request_id.
    """

    __tablename__ = "reasoning_logs"

    id = Column(String, primary_key=True, default=_uuid)
    request_id = Column(String, nullable=False, index=True)  # ID задачи прогноза
    parent_id = Column(
        String, ForeignKey("reasoning_logs.id", ondelete="SET NULL"), nullable=True
    )
    agent_name = Column(String, nullable=False)  # supervisor | data_analyst | ...
    tool_name = Column(String, nullable=True)    # fit_arima | retrieve_cases | ...
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    model_name = Column(String, nullable=True)
    parameters = Column(JSON, nullable=True)
    status = Column(String, default="running", nullable=False)  # running|success|error
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_reasoning_request", "request_id"),
        Index("ix_reasoning_parent", "parent_id"),
        Index("ix_reasoning_agent", "agent_name"),
    )


# ============================================================================
# LLMCalls — трассировка всех вызовов LLM (Observability)
# ============================================================================
class LLMCall(Base):
    """Каждый вызов LLM: агент, модель, провайдер, токены, latency, статус.

    Хэши промпта/ответа позволяют искать дубликаты без хранения текстов.
    """

    __tablename__ = "llm_calls"

    id = Column(String, primary_key=True, default=_uuid)
    request_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    provider = Column(String, nullable=False)  # ollama | hf_local | api
    prompt_hash = Column(String, nullable=True)
    response_hash = Column(String, nullable=True)
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    status = Column(String, default="success", nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_llm_request", "request_id"),
        Index("ix_llm_created", "created_at"),
        Index("ix_llm_model", "model_name"),
    )
