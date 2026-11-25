"""Сервис оценки (Этап 13): запуски, сохранение результатов, сравнение моделей.

Используется scripts/run_eval.py (файл + БД), API /api/v1/evaluate/*
и Celery-задачей evaluate_model_task. Результаты пишутся в таблицы
evaluation_results (сводка по режиму × горизонт × доля) и model_comparisons
(попарные сравнения со стат. значимостью).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import select

from backend.core.metrics import regression_metrics
from backend.db.models import EvaluationResult, EvaluationRun, ModelComparison
from backend.evaluation.runner import (
    DEFAULT_TEST_SET,
    MODES,
    EvalConfig,
    load_test_windows,
    split_window,
)
from backend.evaluation.stats import bootstrap_mean_diff_ci, wilcoxon_test

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "modes": ["baseline", "rag", "agents"],
    "horizons": [14],
    "data_fractions": [1.0],
    "num_series": 10,
    "fast": True,
    "metric": "mae",
}


def parse_params(params: dict[str, Any] | None) -> list[EvalConfig]:
    """Параметры запуска -> список конфигураций (декартово произведение)."""
    merged = {**DEFAULT_PARAMS, **(params or {})}
    configs: list[EvalConfig] = []
    for mode in merged["modes"]:
        if mode not in MODES:
            raise ValueError(f"Неизвестный режим {mode!r}; доступны {list(MODES)}")
        for horizon in merged["horizons"]:
            for fraction in merged["data_fractions"]:
                if not 0 < float(fraction) <= 1:
                    raise ValueError(f"Доля данных должна быть в (0, 1]: {fraction}")
                configs.append(
                    EvalConfig(
                        mode=mode, horizon=int(horizon),
                        data_fraction=float(fraction), fast=bool(merged["fast"]),
                    )
                )
    return configs


# ============================================================================
# Запуски оценки
# ============================================================================
def _run_to_dict(run: EvaluationRun) -> dict[str, Any]:
    def iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "id": run.id,
        "status": run.status,
        "params": run.params,
        "summary": run.summary,
        "comparisons": run.comparisons,
        "n_records": run.n_records,
        "error": run.error,
        "created_at": iso(run.created_at),
        "started_at": iso(run.started_at),
        "finished_at": iso(run.finished_at),
    }


async def create_run(params: dict[str, Any] | None, user_id: str | None = None) -> dict[str, Any]:
    """Создаёт запуск в статусе pending (валидация параметров — здесь же)."""
    from backend.db.session import AsyncSessionLocal

    from backend.db.session import ensure_tables_once

    configs = parse_params(params)  # ValueError при некорректных параметрах
    await ensure_tables_once()
    async with AsyncSessionLocal() as session:
        run = EvaluationRun(
            status="pending",
            user_id=user_id,
            params={"n_configs": len(configs), **(params or {})},
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return _run_to_dict(run)


async def get_run(run_id: str) -> dict[str, Any] | None:
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        run = await session.get(EvaluationRun, run_id)
        return _run_to_dict(run) if run else None


async def _update_run(run_id: str, **fields: Any) -> None:
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        run = await session.get(EvaluationRun, run_id)
        if run is None:
            raise LookupError(f"Запуск оценки {run_id} не найден")
        for key, value in fields.items():
            setattr(run, key, value)
        await session.commit()


async def run_evaluation_run(run_id: str) -> dict[str, Any] | None:
    """Выполняет запуск оценки по run_id (inline-раннер или Celery-воркер)."""
    from backend.evaluation.runner import run_evaluation

    from backend.db.session import ensure_tables_once

    await ensure_tables_once()  # воркер/скрипт могли стартовать раньше обновления схемы
    run = await get_run(run_id)
    if run is None:
        raise LookupError(f"Запуск оценки {run_id} не найдена")
    params = run["params"] or {}
    await _update_run(run_id, status="running", started_at=datetime.now(timezone.utc))
    try:
        configs = parse_params(params)
        windows = load_test_windows(
            params.get("test_set") or DEFAULT_TEST_SET,
            num_series=int(params.get("num_series", DEFAULT_PARAMS["num_series"])),
            seed=int(params.get("seed", 42)),
        )
        outcome = await run_evaluation(configs, windows)
        await save_results(run_id, outcome)
        await _update_run(
            run_id, status="completed", finished_at=datetime.now(timezone.utc),
            summary=outcome["summary"], comparisons=outcome["comparisons"],
            n_records=outcome["n_records"],
        )
        return outcome
    except Exception as exc:  # noqa: BLE001 — ошибка фиксируется в запуске
        logger.exception("Evaluation run %s failed", run_id)
        await _update_run(
            run_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:1000],
            finished_at=datetime.now(timezone.utc),
        )
        return None


async def save_results(run_id: str | None, outcome: dict[str, Any]) -> tuple[int, int]:
    """Сводка -> evaluation_results, сравнения -> model_comparisons.

    Возвращает (строк результатов, строк сравнений).
    """
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        for row in outcome["summary"]:
            session.add(
                EvaluationResult(
                    model_name=row["mode"],
                    mode=row["mode"],
                    sample_size=row["n_series"],
                    horizon=row["horizon"],
                    data_fraction=row["data_fraction"],
                    metrics={k: row.get(k) for k in ("mae", "rmse", "mape", "wape", "mase")},
                    agent_metrics=row.get("agent_metrics"),
                    evaluation_time=outcome.get("total_time_s"),
                )
            )
        for pair in outcome["comparisons"]:
            session.add(
                ModelComparison(
                    models_compared=pair["models_compared"],
                    sample_size=pair["n_series"],
                    results={
                        "horizon": pair["horizon"],
                        "data_fraction": pair["data_fraction"],
                        "metric": pair["metric"],
                        pair["models_compared"][0]: pair[f"mean_{pair['metric']}_{pair['models_compared'][0]}"],
                        pair["models_compared"][1]: pair[f"mean_{pair['metric']}_{pair['models_compared'][1]}"],
                    },
                    winner=pair["winner"],
                    significance=pair["significance"],
                )
            )
        await session.commit()
        return len(outcome["summary"]), len(outcome["comparisons"])


async def list_results(mode: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Последние строки evaluation_results (опционально по режиму)."""
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        query = select(EvaluationResult).order_by(EvaluationResult.created_at.desc()).limit(limit)
        if mode:
            query = query.where(EvaluationResult.mode == mode)
        rows = (await session.execute(query)).scalars().all()
        return [
            {
                "id": r.id, "mode": r.mode, "model_name": r.model_name,
                "sample_size": r.sample_size, "horizon": r.horizon,
                "data_fraction": r.data_fraction, "metrics": r.metrics,
                "agent_metrics": r.agent_metrics, "evaluation_time": r.evaluation_time,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


# ============================================================================
# Сравнение моделей по именам (GET /api/v1/evaluate/compare)
# ============================================================================
# имя модели -> (model_type, hyperparams или callable(season))
_MODEL_SPECS: dict[str, Any] = {
    "naive": ("naive_seasonal", lambda season: {"season": season or 1}),
    "arima": ("arima", lambda season: {"order": [1, 1, 1]}),
    "sarima": ("sarima", lambda season: {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, season or 7]}),
    "xgboost": ("xgboost", lambda season: {"lags": 7, "n_estimators": 100}),
    "prophet": ("prophet", lambda season: {"seasonality_mode": "additive"}),
}


async def compare_models(
    models: list[str],
    sample_size: int = 20,
    horizon: int = 14,
    seed: int = 42,
    save: bool = True,
) -> dict[str, Any]:
    """Сравнение именованных моделей на одной выборке тестовых окон.

    Каждая модель обучается на всём обучающем отрезке окна и прогнозирует
    горизонт; метрики усредняются по окнам, пары — Wilcoxon + bootstrap-CI.
    """
    from backend.timeseries.classical import predict_final
    from backend.timeseries.preprocessing import prepare_timeseries

    unknown = [m for m in models if m not in _MODEL_SPECS]
    if unknown:
        raise ValueError(f"Неизвестные модели {unknown}; доступны {sorted(_MODEL_SPECS)}")
    if not models:
        raise ValueError("Список моделей пуст")
    sample_size = max(1, min(int(sample_size), 50))
    horizon = max(1, min(int(horizon), 60))

    windows = load_test_windows(num_series=sample_size, seed=seed)
    per_model: dict[str, dict[str, list]] = {m: {"mae": [], "rmse": []} for m in models}
    errors: dict[str, str] = {}

    for window in windows:
        train = split_window(window, horizon, 1.0)
        prepared = prepare_timeseries(train["train_values"], dates=train["train_dates"])
        series, season = prepared["series"], prepared["diagnostics"].get("seasonal_period")
        actuals = np.asarray(train["test_values"], dtype=float)
        for name in models:
            model_type, hyper = _MODEL_SPECS[name]
            try:
                predictions = np.asarray(
                    predict_final(series, horizon, model_type, hyper(season), season), dtype=float
                )
                metrics = regression_metrics(actuals, predictions, series.values, season=season or 1)
                per_model[name]["mae"].append(metrics["mae"])
                per_model[name]["rmse"].append(metrics["rmse"])
            except Exception as exc:  # noqa: BLE001 — модель может не обучиться
                errors[name] = f"{type(exc).__name__}: {exc}"[:200]

    def stats(values: list[float]) -> dict[str, float | None]:
        array = np.asarray(values, dtype=float)
        return {
            "mean": round(float(array.mean()), 4) if array.size else None,
            "median": round(float(np.median(array)), 4) if array.size else None,
            "n": int(array.size),
        }

    results = {
        name: {
            "mae": stats(per_model[name]["mae"]),
            "rmse": stats(per_model[name]["rmse"]),
            "model_type": _MODEL_SPECS[name][0],
            **({"error": errors[name]} if name in errors else {}),
        }
        for name in models
    }
    scored = [m for m in models if per_model[m]["mae"]]
    pairs = []
    for i, a in enumerate(scored):
        for b in scored[i + 1 :]:
            n = min(len(per_model[a]["mae"]), len(per_model[b]["mae"]))
            va, vb = per_model[a]["mae"][:n], per_model[b]["mae"][:n]
            pairs.append(
                {
                    "models_compared": [a, b],
                    "mean_mae": {a: stats(va)["mean"], b: stats(vb)["mean"]},
                    "winner": a if np.mean(va) < np.mean(vb) else b,
                    "wilcoxon": wilcoxon_test(va, vb),
                    "bootstrap": bootstrap_mean_diff_ci(va, vb),
                    "n_series": n,
                }
            )
    winner = (
        min(scored, key=lambda m: np.mean(per_model[m]["mae"])) if scored else None
    )
    outcome = {
        "models": models,
        "sample_size": len(windows),
        "horizon": horizon,
        "results": results,
        "pairs": pairs,
        "winner": winner,
    }

    if save:
        from backend.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            session.add(
                ModelComparison(
                    models_compared=models,
                    sample_size=len(windows),
                    results={"horizon": horizon, **results},
                    winner=winner,
                    significance={f"{p['models_compared'][0]}_vs_{p['models_compared'][1]}": {
                        "wilcoxon": p["wilcoxon"], "bootstrap": p["bootstrap"]
                    } for p in pairs},
                )
            )
            await session.commit()
    return outcome
