"""Юнит-тесты моделей БД (backend/db/models.py) на SQLite in-memory.

Проверяем: создание и чтение записей, уникальные ограничения,
дефолты, иерархию reasoning_logs (parent_id).
"""
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.db.models import (
    Base,
    EvaluationResult,
    ForecastCache,
    GlossaryTerm,
    KnowledgeCase,
    LLMCall,
    ModelComparison,
    ReasoningLog,
    User,
)


@pytest.fixture
def session():
    """Свежая in-memory БД с созданными таблицами для каждого теста."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


# ============================================================================
# User
# ============================================================================
class TestUser:
    def test_create_user(self, session):
        u = User(
            email="nadine@example.com",
            username="nadenka",
            hashed_password="hash",
            role="editor",
        )
        session.add(u)
        session.commit()

        loaded = session.execute(select(User).where(User.username == "nadenka")).scalar_one()
        assert loaded.email == "nadine@example.com"
        assert loaded.role == "editor"
        assert loaded.is_active is True

    def test_email_unique(self, session):
        session.add(User(id="u1", email="a@b.c", username="a", hashed_password="h"))
        session.commit()
        session.add(User(id="u2", email="a@b.c", username="b", hashed_password="h"))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_username_unique(self, session):
        session.add(User(id="u1", email="a@b.c", username="alice", hashed_password="h"))
        session.commit()
        session.add(User(id="u2", email="b@c.d", username="alice", hashed_password="h"))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_role_default(self, session):
        u = User(email="a@b.c", username="alice", hashed_password="hash")
        session.add(u)
        session.commit()
        assert u.role == "user"


# ============================================================================
# KnowledgeCase
# ============================================================================
class TestKnowledgeCase:
    def test_create_weather_case(self, session):
        case = KnowledgeCase(
            series_id="jena_2013_mean_temp",
            task_type="timeseries",
            city="Jena",
            freq="D",
            features={"trend": 0.3, "seasonality": "yearly", "entropy": 1.2},
            embedding=[0.1] * 128,
            best_model="sarima",
            hyperparams={"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7]},
            metrics={"mase": 0.85, "mape": 0.12},
            description="Дневная температура с годовой сезонностью, SARIMA обошла Prophet.",
        )
        session.add(case)
        session.commit()

        loaded = session.execute(
            select(KnowledgeCase).where(KnowledgeCase.series_id == "jena_2013_mean_temp")
        ).scalar_one()
        assert loaded.best_model == "sarima"
        assert loaded.metrics["mase"] == 0.85
        assert len(loaded.embedding) == 128
        assert loaded.task_type == "timeseries"

    def test_series_unique_per_task_type(self, session):
        session.add(KnowledgeCase(series_id="s1", task_type="timeseries", best_model="arima"))
        session.commit()
        session.add(KnowledgeCase(series_id="s1", task_type="timeseries", best_model="prophet"))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_same_series_different_task_ok(self, session):
        session.add(KnowledgeCase(series_id="s1", task_type="timeseries", best_model="arima"))
        session.add(
            KnowledgeCase(series_id="s1", task_type="tabular", best_model="xgboost")
        )
        session.commit()
        assert len(session.execute(select(KnowledgeCase)).scalars().all()) == 2


# ============================================================================
# GlossaryTerm
# ============================================================================
class TestGlossaryTerm:
    def test_create_term(self, session):
        g = GlossaryTerm(
            term="Индекс адаптации (MASE)",
            definition="Отношение MAE модели к MAE наивного сезонного прогноза.",
            category="статистика",
        )
        session.add(g)
        session.commit()

        loaded = session.execute(
            select(GlossaryTerm).where(GlossaryTerm.term == "Индекс адаптации (MASE)")
        ).scalar_one()
        assert loaded.category == "статистика"

    def test_term_unique(self, session):
        session.add(GlossaryTerm(term="MLAPE", definition="d1"))
        session.commit()
        session.add(GlossaryTerm(term="MLAPE", definition="d2"))
        with pytest.raises(IntegrityError):
            session.commit()


# ============================================================================
# ForecastCache
# ============================================================================
class TestForecastCache:
    def test_two_models_same_key(self, session):
        session.add(ForecastCache(input_key="k1", model_used="arima", output="o1"))
        session.add(ForecastCache(input_key="k1", model_used="prophet", output="o2"))
        session.commit()
        assert len(session.execute(select(ForecastCache)).scalars().all()) == 2

    def test_same_key_same_model_conflict(self, session):
        session.add(ForecastCache(input_key="k1", model_used="arima", output="o1"))
        session.commit()
        session.add(ForecastCache(input_key="k1", model_used="arima", output="o2"))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_hits_default(self, session):
        c = ForecastCache(input_key="k1", model_used="arima", output="o1")
        session.add(c)
        session.commit()
        assert c.hits == 1


# ============================================================================
# EvaluationResult и ModelComparison
# ============================================================================
class TestEvaluation:
    def test_create_evaluation_result(self, session):
        r = EvaluationResult(
            model_name="sarima",
            mode="agents",
            sample_size=50,
            horizon=14,
            data_fraction=1.0,
            metrics={"mae": 1.2, "rmse": 1.9, "mape": 0.08},
            agent_metrics={"iterations": 2, "hitl": 0, "tokens": 1500},
            evaluation_time=3.5,
        )
        session.add(r)
        session.commit()

        loaded = session.execute(select(EvaluationResult).where(EvaluationResult.mode == "agents")).scalar_one()
        assert loaded.metrics["mae"] == 1.2
        assert loaded.agent_metrics["iterations"] == 2

    def test_create_model_comparison(self, session):
        c = ModelComparison(
            models_compared=["arima", "prophet", "xgboost"],
            sample_size=100,
            results={"arima": {"mase": 1.0}, "prophet": {"mase": 0.9}},
            winner="prophet",
        )
        session.add(c)
        session.commit()
        assert c.winner == "prophet"
        assert "xgboost" in c.models_compared


# ============================================================================
# ReasoningLog — иерархия
# ============================================================================
class TestReasoningLog:
    def test_parent_child_hierarchy(self, session):
        root = ReasoningLog(request_id="req-1", agent_name="supervisor", status="success")
        session.add(root)
        session.flush()  # получаем root.id до коммита

        child = ReasoningLog(
            request_id="req-1",
            parent_id=root.id,
            agent_name="model_selector",
            tool_name="fit_sarima",
            parameters={"order": [1, 1, 1]},
            status="success",
        )
        session.add(child)
        session.flush()  # получаем child.id до создания внука

        grandchild = ReasoningLog(
            request_id="req-1",
            parent_id=child.id,
            agent_name="model_selector",
            tool_name="retrieve_cases",
            status="success",
        )
        session.add(grandchild)
        session.commit()

        logs = session.execute(
            select(ReasoningLog).where(ReasoningLog.request_id == "req-1")
        ).scalars().all()
        assert len(logs) == 3

        # ребёнок ссылается на родителя, внук — на ребёнка
        assert child.parent_id == root.id
        assert grandchild.parent_id == child.id
        # у корня нет родителя
        assert root.parent_id is None

    def test_status_default(self, session):
        log = ReasoningLog(request_id="req-2", agent_name="critic")
        session.add(log)
        session.commit()
        assert log.status == "running"


# ============================================================================
# LLMCall
# ============================================================================
class TestLLMCall:
    def test_create_call(self, session):
        call = LLMCall(
            request_id="req-1",
            agent_name="model_selector",
            model_name="qwen2.5:7b-instruct",
            provider="ollama",
            prompt_hash="abc123",
            response_hash="def456",
            tokens_in=800,
            tokens_out=120,
            latency_ms=1500.0,
            status="success",
        )
        session.add(call)
        session.commit()

        loaded = session.execute(select(LLMCall).where(LLMCall.request_id == "req-1")).scalar_one()
        assert loaded.provider == "ollama"
        assert loaded.tokens_in + loaded.tokens_out == 920
