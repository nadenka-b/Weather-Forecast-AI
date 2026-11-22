"""Юнит-тесты RAG-слоя (Этап 10): BM25, векторизатор, RRF, интеграция."""
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.agents.supervisor import build_supervisor
from backend.core.models import ForecastState
from backend.data.embeddings import CaseVectorizer, case_to_vector_features
from backend.db.models import Base, KnowledgeCase, ReasoningLog
from backend.retrieval.bm25_index import BM25Index, tokenize
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.vector_index import InMemoryVectorIndex


def make_case(
    series_id: str,
    variable: str,
    period: float | None,
    model: str,
    mean: float,
    std: float,
    description: str,
    city: str = "Jena",
) -> dict:
    return {
        "series_id": series_id,
        "task_type": "timeseries",
        "city": city,
        "freq": "D",
        "features": {
            "length": 365,
            "mean": mean,
            "std": std,
            "min": mean - 3 * std,
            "max": mean + 3 * std,
            "acf_lag1": 0.8,
            "adf_pvalue": 0.01,
            "stationary": True,
            "seasonal_period": period,
            "trend_strength": 0.1,
            "seasonal_strength": 0.6,
            "missing_rate": 0.0,
            "variable": variable,
        },
        "best_model": model,
        "hyperparams": {"order": [1, 1, 1]},
        "metrics": {"mase": 0.7, "mae": 1.1},
        "description": description,
    }


CORPUS = [
    make_case("c_temp_week", "temperature", 7, "sarima", 15.0, 5.0,
              "Дневной ряд temperature (Jena), сезонность 7, SARIMA лучшая."),
    make_case("c_temp_year", "temperature", None, "arima", 10.0, 8.0,
              "Дневной ряд temperature (Jena), годовой ход, ARIMA лучшая."),
    make_case("c_humid", "humidity", 7, "xgboost", 70.0, 10.0,
              "Дневной ряд humidity (Jena), сезонность 7, XGBoost лучшая."),
    make_case("c_pressure", "pressure", None, "naive_seasonal", 1010.0, 8.0,
              "Дневной ряд pressure (Jena), без сезонности, наивная лучшая."),
    make_case("c_delhi_temp", "temperature", 7, "prophet", 30.0, 6.0,
              "Дневной ряд temperature (Delhi), муссонная сезонность, Prophet.",
              city="Delhi"),
]


# ============================================================================
# Токенизация и BM25
# ============================================================================
class TestBM25:
    def test_tokenize_russian_and_latin(self):
        assert tokenize("Дневной ряд 7 дней, SARIMA (лучшая)!") == [
            "дневной", "ряд", "7", "дней", "sarima", "лучшая",
        ]

    def test_finds_relevant_description(self):
        index = BM25Index(CORPUS)
        hits = index.search("humidity Jena сезонность", top_k=3)
        assert hits[0][0] == "c_humid"

    def test_empty_query_returns_nothing(self):
        assert BM25Index(CORPUS).search("!!! ???") == []


# ============================================================================
# Векторизатор кейсов
# ============================================================================
class TestVectorizer:
    def test_case_to_vector_features_fixed_order(self):
        row = case_to_vector_features(CORPUS[0]["features"])
        assert len(row) == 12 + 7  # числовые + one-hot переменных
        assert row[0] == 365
        assert row[12] == 1.0  # temperature
        assert sum(row[12:]) == 1

    def test_none_values_become_zero(self):
        row = case_to_vector_features({"variable": None})
        assert row[0] == 0.0 and sum(row[12:]) == 0

    def test_fit_transform_l2_normalized(self):
        rows = [case_to_vector_features(case["features"]) for case in CORPUS]
        vectorizer = CaseVectorizer(target_dim=128).fit(rows)
        assert vectorizer.dim <= min(19, len(CORPUS))
        vectors = vectorizer.transform(rows)
        assert vectors.shape == (len(CORPUS), vectorizer.dim)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-8)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            CaseVectorizer().transform([[1.0, 2.0]])


# ============================================================================
# Векторный индекс
# ============================================================================
class TestVectorIndex:
    def test_search_nearest_with_filters(self):
        rows = [case_to_vector_features(case["features"]) for case in CORPUS]
        vectorizer = CaseVectorizer().fit(rows)
        vectors = vectorizer.transform(rows)

        index = InMemoryVectorIndex()
        index.add(CORPUS, vectors)

        # запрос — кейс про температуру Jena: похожие temp-кейсы выше humidity
        query = vectorizer.transform_one(CORPUS[0]["features"])
        hits = index.search(query, top_k=5)
        assert hits[0][0] == "c_temp_week"
        assert {"c_temp_week", "c_temp_year"} <= {sid for sid, _ in hits[:3]}

        # фильтр по городу: Delhi-кейс остаётся один
        delhi_hits = index.search(query, top_k=5, filters={"city": "Delhi"})
        assert [sid for sid, _ in delhi_hits] == ["c_delhi_temp"]

    def test_empty_index(self):
        assert InMemoryVectorIndex().search(np.zeros(8), top_k=3) == []


# ============================================================================
# Гибридный ретривер и RRF
# ============================================================================
class TestHybridRetriever:
    def test_rrf_prefers_case_found_by_both(self):
        retriever = HybridRetriever(CORPUS)
        # текст ведёт к humidity, признаки — тоже ближе к humidity
        results = retriever.retrieve(
            query_features=CORPUS[2]["features"],
            query_text=" humidity сезонность 7 XGBoost",
            top_k=3,
        )
        assert results[0].case["series_id"] == "c_humid"
        assert results[0].bm25_rank == 1
        assert results[0].vector_rank == 1

    def test_few_shot_prompts_contain_description(self):
        retriever = HybridRetriever(CORPUS)
        prompts = retriever.few_shot_prompts(
            CORPUS[0]["features"], "temperature Jena сезонность", top_k=2
        )
        assert len(prompts) == 2
        assert "temperature" in prompts[0]
        assert "mase" in prompts[0]  # метрики кейса приложены


# ============================================================================
# ModelSelector с RAG + build_supervisor (reasoning в БД)
# ============================================================================
@pytest.fixture
async def memory_db(monkeypatch):
    import backend.db.session as db_session_module

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_local = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_session_module, "AsyncSessionLocal", session_local)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield session_local
    await engine.dispose()


class TestSelectorWithRAG:
    async def test_fewshot_in_prompt(self):
        from backend.agents.model_selector import ModelSelector
        from backend.core.llm.manager import LLMManager

        captured: list[list[dict]] = []

        async def spy(messages):
            captured.append(messages)
            return '{"tool": "fit_arima", "arguments": {"order": [1, 1, 1]}}'

        retriever = HybridRetriever(CORPUS)
        selector = ModelSelector(llm_manager=LLMManager(llm_call=spy), retriever=retriever)

        rng = np.random.default_rng(5)
        n = 220
        values = 15 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.5, n)
        state = ForecastState(
            series_id="rag_query",
            horizon=7,
            freq="D",
            metadata={
                "series_values": values.tolist(),
                "city": "Jena",
                "variable": "temperature",
            },
        )
        state = await selector.process(state)

        user_text = captured[0][1]["content"]
        assert "Кейс:" in user_text  # few-shot из БЗ попал в промпт
        assert "temperature" in user_text


@pytest.mark.usefixtures("memory_db")
class TestBuildSupervisor:
    async def test_factory_wires_rag_and_db_logs(self, memory_db):
        # база знаний для RAG
        from backend.data import knowledge_base

        await knowledge_base.import_cases(CORPUS)

        supervisor = await build_supervisor()
        assert set(supervisor.agents) == {
            "data_analyst", "model_selector", "validator", "critic",
        }
        assert supervisor.agents["model_selector"].retriever is not None

        # прогон — reasoning-лог должен попасть в таблицу reasoning_logs
        rng = np.random.default_rng(9)
        n = 180
        values = 12 + 4 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.4, n)
        state = ForecastState(
            series_id="dblog_check",
            horizon=7,
            freq="D",
            metadata={"series_values": values.tolist(), "city": "Jena",
                      "variable": "temperature"},
        )
        result = await supervisor.process(state)

        async with memory_db() as session:
            logs = (
                await session.execute(
                    select(ReasoningLog).where(ReasoningLog.request_id == state.log_id)
                )
            ).scalars().all()
        agents_logged = {log.agent_name for log in logs}
        assert "supervisor" in agents_logged
        assert {"data_analyst", "model_selector"} <= agents_logged
        # состояние обработано полным пайплайном
        assert result.model_type in ("arima", "sarima", "naive_seasonal", "xgboost")
        assert result.diagnostics is not None
