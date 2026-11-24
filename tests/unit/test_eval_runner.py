"""Тесты ядра оценки, сервиса и скрипта run_eval (Этап 13)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, EvaluationResult, ModelComparison
from backend.evaluation import service
from backend.evaluation.runner import (
    EvalConfig,
    aggregate,
    check_sufficiency,
    compare_modes,
    evaluate_window,
    forecast_baseline,
    forecast_rag,
    load_test_windows,
    run_evaluation,
    split_window,
    summary_frame,
)
from backend.retrieval.hybrid_retriever import HybridRetriever
from tests.eval_helpers import make_window, write_test_set

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


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


# ============================================================================
# Данные и проверки
# ============================================================================
class TestEvalData:
    def test_load_test_windows_sampling_is_deterministic(self, tmp_path):
        path = write_test_set(tmp_path / "test_set.jsonl", n_windows=10)
        first = load_test_windows(path, num_series=4, seed=7)
        second = load_test_windows(path, num_series=4, seed=7)
        assert [w["series_id"] for w in first] == [w["series_id"] for w in second]
        assert len(first) == 4
        assert len(load_test_windows(path)) == 10

    def test_missing_test_set(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate_knowledge_base"):
            load_test_windows(tmp_path / "missing.jsonl")

    def test_split_window(self):
        window = make_window(0, n=200)
        full = split_window(window, horizon=14, data_fraction=1.0)
        assert len(full["test_values"]) == 14
        assert full["n_train"] == 186
        assert full["test_values"] == window["values"][-14:]

        half = split_window(window, horizon=14, data_fraction=0.5)
        assert half["n_train"] == 93
        assert half["train_values"] == window["values"][:186][-93:]

        small = split_window(window, horizon=14, data_fraction=0.1)
        assert small["n_train"] == 60  # не меньше MIN_TRAIN

    def test_sufficiency(self):
        assert check_sufficiency(200, 14, 7)["ok"] is True
        assert check_sufficiency(200, 14, 7)["seasonal_ok"] is True
        short = check_sufficiency(40, 14, 7)
        assert short["ok"] is False and short["reasons"]
        long_horizon = check_sufficiency(100, 50, None)
        assert long_horizon["ok"] is False  # короче 3 горизонтов
        assert check_sufficiency(100, 14, 60)["seasonal_ok"] is False


# ============================================================================
# Прогноз по режимам
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestModeForecasts:
    def test_baseline_fast(self):
        import pandas as pd

        window = make_window(1, n=200)
        series = pd.Series(window["values"], index=pd.to_datetime(window["dates"]))
        result = forecast_baseline(series, horizon=7, season=7, fast=True)
        assert len(result["predictions"]) == 7
        assert result["model"] in ("naive_seasonal", "arima")
        assert "mase" in result["val_metrics"]

    def test_rag_uses_nearest_case(self):
        import pandas as pd

        corpus = [
            {
                "series_id": "kb|x|1", "task_type": "timeseries", "city": "TestCity",
                "features": {"length": 200, "mean": 12.0, "std": 3.5, "min": 4.0, "max": 20.0,
                             "acf_lag1": 0.8, "adf_pvalue": 0.01, "stationary": True,
                             "seasonal_period": 7, "trend_strength": 0.1,
                             "seasonal_strength": 0.6, "missing_rate": 0.0,
                             "variable": "temperature"},
                "best_model": "xgboost", "hyperparams": {"lags": 7, "n_estimators": 30},
                "metrics": {"mase": 0.8},
                "description": "Дневной ряд temperature (TestCity), сезонность 7.",
            }
        ]
        window = make_window(2, n=200)
        series = pd.Series(window["values"], index=pd.to_datetime(window["dates"]))
        result = forecast_rag(
            HybridRetriever(corpus), series, 5, 7,
            diagnostics={"seasonal_period": 7, "stats": {"length": 200, "mean": 12.0, "std": 3.5,
                                                          "min": 4.0, "max": 20.0, "acf_lag1": 0.8},
                          "adf": {"adf_pvalue": 0.01, "stationary": True},
                          "stl": {"trend_strength": 0.1, "seasonal_strength": 0.6},
                          "raw_missing_rate": 0.0},
            city="TestCity", variable="temperature",
        )
        assert result["model"] == "xgboost"      # модель из ближайшего кейса
        assert result["retrieved"] == "kb|x|1"
        assert len(result["predictions"]) == 5

    async def test_evaluate_window_modes(self):
        window = make_window(3, n=200)
        for mode in ("baseline", "agents"):
            record = await evaluate_window(window, EvalConfig(mode=mode, horizon=7, fast=True))
            assert "error" not in record, record
            assert record["mode"] == mode
            for key in ("mae", "rmse", "mape", "wape", "mase"):
                assert key in record["metrics"]
            assert len(record["predictions"]) == len(record["actuals"]) == 7
        agents = await evaluate_window(window, EvalConfig(mode="agents", horizon=7))
        assert agents["agent_metrics"]["duration_s"] > 0
        assert "hitl_requested" in agents["agent_metrics"]
        assert "iterations" in agents["agent_metrics"]

    async def test_foundation_mode_marked_unavailable(self):
        record = await evaluate_window(make_window(0, n=120), EvalConfig(mode="foundation", horizon=7))
        assert "не установлены" in record["error"]

    async def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Неизвестный режим"):
            await evaluate_window(make_window(0, n=120), EvalConfig(mode="magic", horizon=7))

    async def test_insufficient_window_is_skipped(self):
        window = make_window(0, n=70)  # 70 - 50 = 20 точек обучения < MIN_TRAIN
        record = await evaluate_window(window, EvalConfig(mode="baseline", horizon=50, fast=True))
        assert "error" in record and record["sufficiency"]["ok"] is False


# ============================================================================
# Агрегация и сравнение
# ============================================================================
def _record(series_id: str, mode: str, mae: float, offset: float = 0.0) -> dict:
    actuals = [float(i) for i in range(7)]
    return {
        "series_id": series_id, "mode": mode, "horizon": 7, "data_fraction": 1.0,
        "metrics": {"mae": mae, "rmse": mae * 1.1, "mape": 10.0, "wape": 8.0, "mase": 0.9},
        "predictions": [a + offset for a in actuals], "actuals": actuals,
        "model": "arima",
    }


class TestAggregation:
    def test_aggregate_groups_by_mode(self):
        records = [
            _record("s1", "baseline", 2.0, 0.2), _record("s1", "agents", 1.0, 0.1),
            _record("s2", "baseline", 4.0, 0.2), _record("s2", "agents", 3.0, 0.1),
            {"series_id": "s3", "mode": "baseline", "horizon": 7, "data_fraction": 1.0, "error": "x"},
        ]
        summary = aggregate(records)
        by_mode = {row["mode"]: row for row in summary}
        assert by_mode["baseline"]["n_series"] == 2      # ошибочная запись не учтена
        assert by_mode["baseline"]["mae"] == 3.0
        assert by_mode["baseline"]["mae_median"] == 3.0
        assert by_mode["baseline"]["models_used"] == {"arima": 2}

    def test_aggregate_agent_metrics(self):
        record = _record("s1", "agents", 1.0, 0.1)
        record["agent_metrics"] = {
            "iterations": 0, "hitl_requested": True, "used_fallback": False,
            "tokens_in": 100, "tokens_out": 20, "duration_s": 1.5, "critic_score": 5.0,
        }
        row = aggregate([record])[0]
        agent = row["agent_metrics"]
        assert agent["hitl_share"] == 1.0
        assert agent["mean_tokens_in"] == 100
        assert agent["mean_duration_s"] == 1.5

    def test_compare_modes_pairs_and_tests(self):
        records = [
            _record(f"s{i}", "baseline", 3.0, 0.3) for i in range(12)
        ] + [
            _record(f"s{i}", "agents", 1.0, 0.1) for i in range(12)
        ]
        pairs = compare_modes(records)
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["models_compared"] == ["agents", "baseline"]
        assert pair["winner"] == "agents"
        sig = pair["significance"]
        assert {"wilcoxon", "bootstrap", "paired_ttest", "dm_mean_pvalue"} <= set(sig)
        assert sig["wilcoxon"]["significant"] is True
        assert sig["bootstrap"]["significant"] is True

    def test_summary_frame_flattens_agent_metrics(self):
        record = _record("s1", "agents", 1.0, 0.1)
        record["agent_metrics"] = {"iterations": 0, "hitl_requested": True, "tokens_in": 5,
                                   "tokens_out": 1, "duration_s": 0.5, "critic_score": 5.0}
        frame = summary_frame(aggregate([record]))
        assert "hitl_share" in frame.columns and "agent_metrics" not in frame.columns


# ============================================================================
# Полный прогон и сервис
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestRunEvaluation:
    async def test_small_run(self, tmp_path):
        path = write_test_set(tmp_path / "test_set.jsonl", n_windows=2, n=160)
        windows = load_test_windows(path)
        configs = [EvalConfig(mode="baseline", horizon=7, fast=True),
                   EvalConfig(mode="agents", horizon=7)]
        outcome = await run_evaluation(configs, windows)
        assert outcome["n_windows"] == 2 and outcome["n_records"] == 4
        assert outcome["n_errors"] == 0
        assert {row["mode"] for row in outcome["summary"]} == {"baseline", "agents"}
        assert len(outcome["comparisons"]) == 1
        assert outcome["total_time_s"] > 0

    async def test_parse_params(self):
        configs = service.parse_params({"modes": ["baseline", "rag"], "horizons": [7, 14],
                                        "data_fractions": [0.5, 1.0], "fast": False})
        assert len(configs) == 8  # декартово произведение
        assert configs[0] == EvalConfig("baseline", 7, 0.5, fast=False)
        with pytest.raises(ValueError, match="Неизвестный режим"):
            service.parse_params({"modes": ["magic"]})
        with pytest.raises(ValueError, match="Доля данных"):
            service.parse_params({"data_fractions": [1.5]})

    async def test_run_lifecycle_persists_results(self, tmp_path, monkeypatch, memory_db):
        test_set = write_test_set(tmp_path / "test_set.jsonl", n_windows=2, n=160)
        monkeypatch.setattr(service, "DEFAULT_TEST_SET", test_set)
        run = await service.create_run({"modes": ["baseline"], "horizons": [7], "num_series": 2})
        assert run["status"] == "pending"

        outcome = await service.run_evaluation_run(run["id"])
        assert outcome is not None
        finished = await service.get_run(run["id"])
        assert finished["status"] == "completed"
        assert len(finished["summary"]) == 1

        async with memory_db() as session:
            results = (await session.execute(select(EvaluationResult))).scalars().all()
        assert len(results) == 1
        assert results[0].mode == "baseline" and results[0].sample_size == 2
        assert results[0].metrics["mae"] > 0

    async def test_run_failure_recorded(self, monkeypatch):
        run = await service.create_run({"modes": ["baseline"], "horizons": [7], "num_series": 2})
        monkeypatch.setattr(service, "load_test_windows",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
        assert await service.run_evaluation_run(run["id"]) is None
        failed = await service.get_run(run["id"])
        assert failed["status"] == "failed" and "no data" in failed["error"]

    async def test_unknown_run(self):
        with pytest.raises(LookupError):
            await service.run_evaluation_run("missing")

    async def test_compare_models(self, monkeypatch, tmp_path, memory_db):
        import backend.evaluation.runner as runner_module

        monkeypatch.setattr(runner_module, "DEFAULT_TEST_SET",
                            write_test_set(tmp_path / "test_set.jsonl", n_windows=4, n=160))
        outcome = await service.compare_models(["naive", "arima"], sample_size=4, horizon=7)
        assert set(outcome["results"]) == {"naive", "arima"}
        assert outcome["winner"] in ("naive", "arima")
        assert len(outcome["pairs"]) == 1
        assert "wilcoxon" in outcome["pairs"][0]

        async with memory_db() as session:
            saved = (await session.execute(select(ModelComparison))).scalars().all()
        assert len(saved) == 1 and saved[0].models_compared == ["naive", "arima"]

        with pytest.raises(ValueError, match="Неизвестные модели"):
            await service.compare_models(["magic"])


# ============================================================================
# Скрипт run_eval.py
# ============================================================================
class TestRunEvalScript:
    def test_script_writes_outputs(self, tmp_path, monkeypatch):
        spec = importlib.util.spec_from_file_location(
            "run_eval", SCRIPTS / "run_eval.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["run_eval"] = module
        spec.loader.exec_module(module)

        test_set = write_test_set(tmp_path / "test_set.jsonl", n_windows=2, n=160)
        out_dir = tmp_path / "eval_out"
        monkeypatch.setattr(
            sys, "argv",
            ["run_eval", "--num-series", "2", "--horizons", "7", "--data-fractions", "1.0",
             "--mode", "baseline", "--fast", "--no-save-db",
             "--test-set", str(test_set), "--out-dir", str(out_dir)],
        )
        assert module.main() == 0
        for name in ("results.jsonl", "summary.csv", "summary.json", "comparisons.json",
                     "meta.json", "eval.log"):
            assert (out_dir / name).exists(), name
        records = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(
            encoding="utf-8").splitlines()]
        assert len(records) == 2 and all("metrics" in r for r in records)
        assert "run_eval" in (out_dir / "eval.log").read_text(encoding="utf-8")
