"""Юнит-тесты Этапа 9: генерация справочника/БЗ и импорт в БД без дубликатов."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.data import knowledge_base
from backend.db.models import Base, KnowledgeCase

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


glossary_script = load_script("generate_glossary")
kb_script = load_script("generate_knowledge_base")


# ============================================================================
# Справочник терминов
# ============================================================================
class TestGlossary:
    def test_at_least_500_terms(self):
        glossary = glossary_script.build_glossary(min_terms=500)
        assert len(glossary) >= 500

    def test_terms_unique(self):
        glossary = glossary_script.build_glossary()
        terms = [entry["term"] for entry in glossary]
        assert len(terms) == len(set(terms))

    def test_definitions_and_categories(self):
        glossary = glossary_script.build_glossary()
        for entry in glossary:
            assert entry["definition"].strip()
            assert entry["category"]
        categories = {entry["category"] for entry in glossary}
        assert {"метеорология", "статистика", "ML"} <= categories

    def test_contains_core_terms(self):
        glossary = glossary_script.build_glossary()
        terms = {entry["term"] for entry in glossary}
        assert "ARIMA" in terms
        assert "Тест Льюнга-Бокса" in terms
        assert any(t.startswith("Аномалия") for t in terms)
        assert any("Лаг" in t and "k=7" in t for t in terms)

    def test_deterministic(self):
        first = glossary_script.build_glossary()
        second = glossary_script.build_glossary()
        assert first == second


# ============================================================================
# Генератор базы знаний
# ============================================================================
class TestKnowledgeBaseGeneration:
    def test_synthetic_sources(self):
        sources = kb_script.synthetic_sources()
        assert len(sources) == 8
        for source in sources:
            assert len(source.series) == 730
            assert source.series.index.freqstr == "D"

    def test_make_windows_counts_and_holdout_logic(self):
        series = pd.Series(
            np.arange(100, dtype=float),
            index=pd.date_range("2024-01-01", periods=100, freq="D"),
        )
        windows = kb_script.make_windows(series, window=30, step=10)
        assert len(windows) == 8  # (100 - 30) // 10 + 1
        assert all(len(w) == 30 for w in windows)

    def test_evaluate_window_produces_case(self):
        source = kb_script.synthetic_sources(seed=7)[0]
        window = source.series.iloc[:120]
        case = kb_script.evaluate_window(window, source, fast=True)

        assert case is not None
        assert case["task_type"] == "timeseries"
        assert case["city"] == source.city
        assert case["best_model"] in ("naive_seasonal", "arima", "sarima", "xgboost")
        assert case["hyperparams"]
        for key in ("mae", "rmse", "mape", "wape"):
            assert key in case["metrics"]
        for key in ("length", "mean", "std", "seasonal_period", "variable"):
            assert key in case["features"]
        assert source.variable in case["description"]
        assert "MASE" in case["description"]

    def test_evaluate_window_short_series_returns_none(self):
        source = kb_script.synthetic_sources()[0]
        assert kb_script.evaluate_window(source.series.iloc[:40], source, True) is None

    def test_series_id_unique_per_window(self):
        source = kb_script.synthetic_sources()[0]
        ids = set()
        for start in (0, 30):
            case = kb_script.evaluate_window(
                source.series.iloc[start : start + 120], source, True
            )
            ids.add(case["series_id"])
        assert len(ids) == 2


# ============================================================================
# Импорт в БД без дубликатов
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


@pytest.mark.usefixtures("memory_db")
class TestKnowledgeBaseImport:
    async def test_import_and_dedupe_cases(self):
        cases = [
            {
                "series_id": "jena|temperature|20130101",
                "best_model": "sarima",
                "metrics": {"mase": 0.7},
                "description": "кейс 1",
            },
            {
                "series_id": "jena|humidity|20130101",
                "best_model": "xgboost",
                "metrics": {"mase": 0.9},
                "description": "кейс 2",
            },
        ]
        inserted, skipped = await knowledge_base.import_cases(cases)
        assert (inserted, skipped) == (2, 0)

        # повторный импорт — всё дубликаты
        inserted, skipped = await knowledge_base.import_cases(cases)
        assert (inserted, skipped) == (0, 2)

        counts = await knowledge_base.get_counts()
        assert counts["knowledge_base"] == 2

    async def test_import_and_dedupe_terms(self):
        terms = [
            {"term": "Адвекция", "definition": "перенос воздуха", "category": "метеорология"},
            {"term": "ARIMA", "definition": "модель", "category": "ML"},
        ]
        inserted, _ = await knowledge_base.import_terms(terms)
        assert inserted == 2
        inserted, skipped = await knowledge_base.import_terms(terms)
        assert (inserted, skipped) == (0, 2)

    async def test_case_fields_persisted(self, memory_db):
        await knowledge_base.import_cases(
            [
                {
                    "series_id": "delhi|meantemp|20130101",
                    "task_type": "timeseries",
                    "city": "Delhi",
                    "freq": "D",
                    "target_column": "meantemp",
                    "features": {"seasonal_period": 365},
                    "best_model": "arima",
                    "hyperparams": {"order": [1, 1, 1]},
                    "metrics": {"mase": 0.85},
                    "description": "описание",
                }
            ]
        )
        async with memory_db() as session:
            row = (
                await session.execute(select(KnowledgeCase))
            ).scalars().one()
        assert row.city == "Delhi"
        assert row.best_model == "arima"
        assert row.features["seasonal_period"] == 365
        assert row.hyperparams["order"] == [1, 1, 1]

    async def test_sample_cases(self):
        await knowledge_base.import_terms(
            [{"term": "Термин", "definition": "определение"}]
        )
        await knowledge_base.import_cases(
            [{"series_id": "s|v|1", "best_model": "arima", "description": "d"}]
        )
        samples = await knowledge_base.sample_cases(limit=3)
        assert len(samples) == 1
        assert samples[0]["best_model"] == "arima"
