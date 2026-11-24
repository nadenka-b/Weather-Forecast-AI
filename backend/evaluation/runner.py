"""Ядро оценки качества (Этап 13) — используется scripts/run_eval.py и API /evaluate.

Режимы (задание MLF):
    baseline  — классические модели без LLM: сетка (наивный сезонный, ARIMA,
                SARIMA при сезонности <= 31, XGBoost), лучшая по валидации;
    rag       — выбор модели по ближайшему кейсу базы знаний (RAG без LLM);
    agents    — полный цикл агентов (Supervisor: DataAnalyst -> ModelSelector
                c Tool Calling -> Validator -> Critic) + финальный прогноз;
    foundation — TS Foundation Models (бонус; если не установлены — пропуск).

Один тестовый ряд = окно из test_set.jsonl: последние `horizon` точек — факт,
остальное — обучение (берётся последняя доля `data_fraction`).

Метрики на горизонте: MAE, RMSE, MAPE, WAPE, MASE. Агентные метрики (agents):
итерации Supervisor, доля HITL, токены LLM, время. Сравнение режимов:
Diebold-Mariano, Wilcoxon, bootstrap-CI, paired t-test (backend/evaluation/stats.py).
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.core.metrics import naive_forecast, regression_metrics
from backend.core.models import ForecastState
from backend.evaluation.stats import (
    bootstrap_mean_diff_ci,
    diebold_mariano,
    paired_ttest,
    wilcoxon_test,
)
from backend.timeseries.classical import fit_arima, fit_xgboost, predict_final, time_split
from backend.timeseries.preprocessing import prepare_timeseries

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEST_SET = BASE_DIR / "data" / "processed" / "test_set.jsonl"

MODES: tuple[str, ...] = ("baseline", "rag", "agents", "foundation")
METRIC_KEYS = ("mae", "rmse", "mape", "wape", "mase")
MIN_TRAIN = 60
EVAL_HORIZON_VAL = 14  # горизонт валидационного сплита при выборе модели


@dataclass
class EvalConfig:
    mode: str
    horizon: int
    data_fraction: float = 1.0
    fast: bool = False  # сетка baseline только naive + ARIMA(1,1,1)


# ============================================================================
# Данные
# ============================================================================
def load_test_windows(
    path: Path | str = DEFAULT_TEST_SET, num_series: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    """Тестовые окна из test_set.jsonl (holdout, не пересекаются с базой знаний).

    num_series: случайная (детерминированная по seed) выборка окон, чтобы
    охватить разные переменные и города.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Тестовый набор не найден: {path}. Запустите scripts/generate_knowledge_base.py"
        )
    with path.open(encoding="utf-8") as handle:
        windows = [json.loads(line) for line in handle if line.strip()]
    if num_series is not None and num_series < len(windows):
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(windows), size=num_series, replace=False).tolist())
        windows = [windows[i] for i in indices]
    return windows


def split_window(window: dict[str, Any], horizon: int, data_fraction: float) -> dict[str, Any]:
    """Последние horizon точек — факт; обучение — последняя доля остальных."""
    values, dates = window["values"], window["dates"]
    train_part, train_dates = values[:-horizon], dates[:-horizon]
    n_train = max(MIN_TRAIN, min(len(train_part), int(round(len(train_part) * data_fraction))))
    return {
        "train_values": train_part[-n_train:],
        "train_dates": train_dates[-n_train:],
        "test_values": values[-horizon:],
        "test_dates": dates[-horizon:],
        "n_train": n_train,
    }


def check_sufficiency(n_train: int, horizon: int, season: int | None) -> dict[str, Any]:
    """Достаточно ли данных для MASE и сезонных моделей (задание MLF)."""
    reasons: list[str] = []
    if n_train < MIN_TRAIN:
        reasons.append(f"обучающих точек меньше {MIN_TRAIN}")
    if n_train < 3 * horizon:
        reasons.append(f"обучающая часть короче 3 горизонтов ({n_train} < {3 * horizon})")
    mase_ok = n_train > max(1, season or 1)
    seasonal_ok = season is None or n_train >= 2 * season
    return {
        "ok": not reasons,
        "mase_ok": mase_ok,
        "seasonal_ok": seasonal_ok,
        "reasons": reasons,
    }


# ============================================================================
# Прогноз по режиму
# ============================================================================
def _prepare(train_values: list[float], train_dates: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    prepared = prepare_timeseries(train_values, dates=train_dates, target_freq="D")
    return prepared["series"], prepared["diagnostics"]


def forecast_baseline(
    series: pd.Series, horizon: int, season: int | None, fast: bool
) -> dict[str, Any]:
    """Классика без LLM: сетка моделей, лучшая по валидации (MASE)."""
    candidates: list[dict[str, Any]] = []

    train, val = time_split(series, EVAL_HORIZON_VAL)
    naive_pred = naive_forecast(train.values, len(val), season=season or 1)
    candidates.append(
        {
            "model": "naive_seasonal",
            "hyperparams": {"season": season or 1},
            "metrics": regression_metrics(val.values, naive_pred, train.values, season=season or 1),
        }
    )
    try:
        arima = fit_arima(series, EVAL_HORIZON_VAL, order=None if not fast else [1, 1, 1], season=season)
        candidates.append(
            {
                "model": arima["model"],
                "hyperparams": {"order": arima["order"], "seasonal_order": arima["seasonal_order"]},
                "metrics": arima["metrics"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("baseline arima failed: %s", exc)
    if not fast:
        try:
            xgb = fit_xgboost(series, EVAL_HORIZON_VAL, lags=7, n_estimators=100, season=season)
            candidates.append(
                {
                    "model": "xgboost",
                    "hyperparams": {"lags": 7, "n_estimators": 100},
                    "metrics": xgb["metrics"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("baseline xgboost failed: %s", exc)

    def rank(case: dict[str, Any]) -> float:
        value = case["metrics"].get("mase")
        return value if value is not None and np.isfinite(value) else case["metrics"]["mae"] * 10

    best = min(candidates, key=rank)
    predictions = predict_final(
        series, horizon, best["model"], best["hyperparams"], season
    )
    return {
        "predictions": predictions,
        "model": best["model"],
        "hyperparams": best["hyperparams"],
        "val_metrics": best["metrics"],
    }


def forecast_rag(
    retriever: Any,
    series: pd.Series,
    horizon: int,
    season: int | None,
    diagnostics: dict[str, Any],
    city: str | None,
    variable: str | None,
) -> dict[str, Any]:
    """Выбор модели по ближайшему кейсу базы знаний (RAG без LLM)."""
    from backend.agents.model_selector import _query_features, _query_text

    probe = ForecastState(
        series_id="rag_probe", horizon=horizon, freq="D", diagnostics=diagnostics,
        metadata={"city": city, "variable": variable},
    )
    top = retriever.retrieve(_query_features(probe), _query_text(probe), top_k=1)
    if not top:
        raise RuntimeError("RAG не нашёл похожих кейсов")
    case = top[0].case
    model = case.get("best_model")
    hyperparams = case.get("hyperparams") or {}
    predictions = predict_final(series, horizon, model, hyperparams, season)
    return {
        "predictions": predictions,
        "model": model,
        "hyperparams": hyperparams,
        "retrieved": case.get("series_id"),
        "rrf_score": top[0].rrf_score,
    }


async def forecast_agents(
    train: dict[str, Any],
    horizon: int,
    meta: dict[str, Any],
    supervisor: Any | None = None,
) -> dict[str, Any]:
    """Полный цикл агентов; агентные метрики из state."""
    from backend.services.pipeline import perform_forecast

    state = ForecastState(
        series_id=meta["series_id"],
        horizon=horizon,
        freq="D",
        metadata={
            "series_values": train["train_values"],
            "series_dates": train["train_dates"],
            "city": meta.get("city"),
            "variable": meta.get("variable"),
        },
    )
    start = time.perf_counter()
    state = await perform_forecast(state, supervisor=supervisor)
    duration = time.perf_counter() - start

    usage = state.metadata.get("llm_usage") or {}
    critic_usage = state.metadata.get("critic_usage") or {}
    supervisor_info = state.metadata.get("supervisor") or {}
    return {
        "predictions": state.metadata.get("predictions") or [],
        "model": state.model_type,
        "hyperparams": state.model_params,
        "agent_metrics": {
            "iterations": supervisor_info.get("iterations", 0),
            "stopped_reason": supervisor_info.get("stopped_reason"),
            "hitl_requested": bool(state.hitl_requested),
            "used_fallback": bool(state.metadata.get("used_fallback")),
            "critic_score": state.critic_score,
            "validation_passed": state.validation_passed,
            "tokens_in": int(usage.get("tokens_in") or 0) + int(critic_usage.get("tokens_in") or 0),
            "tokens_out": int(usage.get("tokens_out") or 0) + int(critic_usage.get("tokens_out") or 0),
            "duration_s": round(duration, 3),
        },
    }


# ============================================================================
# Оценка одного окна
# ============================================================================
async def evaluate_window(
    window: dict[str, Any],
    config: EvalConfig,
    retriever: Any | None = None,
    supervisor: Any | None = None,
) -> dict[str, Any]:
    """Прогноз по режиму на одном тестовом окне + метрики на горизонте."""
    record: dict[str, Any] = {
        "series_id": window["series_id"],
        "city": window.get("city"),
        "variable": window.get("variable"),
        "mode": config.mode,
        "horizon": config.horizon,
        "data_fraction": config.data_fraction,
    }
    if config.mode == "foundation":
        record["error"] = "TS Foundation Models не установлены (бонус-группа poetry --with foundation)"
        return record
    if config.mode not in MODES:
        raise ValueError(f"Неизвестный режим {config.mode!r}; доступны {MODES}")

    train = split_window(window, config.horizon, config.data_fraction)
    series, diagnostics = _prepare(train["train_values"], train["train_dates"])
    season = diagnostics.get("seasonal_period")
    record["sufficiency"] = check_sufficiency(len(series), config.horizon, season)
    if not record["sufficiency"]["ok"]:
        record["error"] = "; ".join(record["sufficiency"]["reasons"])
        return record

    start = time.perf_counter()
    if config.mode == "baseline":
        result = forecast_baseline(series, config.horizon, season, config.fast)
    elif config.mode == "rag":
        result = forecast_rag(
            retriever, series, config.horizon, season, diagnostics,
            window.get("city"), window.get("variable"),
        )
    else:  # agents
        result = await forecast_agents(
            train, config.horizon,
            {"series_id": window["series_id"], "city": window.get("city"),
             "variable": window.get("variable")},
            supervisor=supervisor,
        )
    record["duration_s"] = round(time.perf_counter() - start, 3)

    predictions = np.asarray(result["predictions"], dtype=float)
    actuals = np.asarray(train["test_values"], dtype=float)
    if len(predictions) != len(actuals):
        record["error"] = f"длина прогноза {len(predictions)} != горизонта {len(actuals)}"
        return record

    record.update(
        {
            "model": result.get("model"),
            "hyperparams": result.get("hyperparams"),
            "metrics": regression_metrics(actuals, predictions, np.asarray(train["train_values"]), season=season or 1),
            "agent_metrics": result.get("agent_metrics"),
            "predictions": [round(float(v), 4) for v in predictions],
            "actuals": [round(float(v), 4) for v in actuals],
        }
    )
    return record


# ============================================================================
# Агрегация и сравнение режимов
# ============================================================================
def _nanmean(values: list[float]) -> float | None:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    return round(float(np.nanmean(array)), 4) if array.size else None


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Сводка: одна строка на (mode, horizon, data_fraction)."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "metrics" not in record:
            continue
        key = (record["mode"], record["horizon"], record["data_fraction"])
        groups[key].append(record)

    summary: list[dict[str, Any]] = []
    for (mode, horizon, fraction), rows in sorted(groups.items()):
        row: dict[str, Any] = {
            "mode": mode, "horizon": horizon, "data_fraction": fraction,
            "n_series": len(rows),
        }
        for metric in METRIC_KEYS:
            values = [r["metrics"].get(metric) for r in rows]
            row[metric] = _nanmean(values)  # type: ignore[arg-type]
            row[f"{metric}_median"] = (
                round(float(np.nanmedian([v for v in values if v is not None])), 4)
                if any(v is not None for v in values) else None
            )
        models: dict[str, int] = defaultdict(int)
        for r in rows:
            models[str(r.get("model"))] += 1
        row["models_used"] = dict(models)

        agent_rows = [r["agent_metrics"] for r in rows if r.get("agent_metrics")]
        if agent_rows:
            row["agent_metrics"] = {
                "mean_iterations": _nanmean([a.get("iterations", 0) for a in agent_rows]),  # type: ignore[list-item]
                "hitl_share": round(
                    sum(1 for a in agent_rows if a.get("hitl_requested")) / len(agent_rows), 4
                ),
                "fallback_share": round(
                    sum(1 for a in agent_rows if a.get("used_fallback")) / len(agent_rows), 4
                ),
                "mean_tokens_in": _nanmean([a.get("tokens_in", 0) for a in agent_rows]),  # type: ignore[list-item]
                "mean_tokens_out": _nanmean([a.get("tokens_out", 0) for a in agent_rows]),  # type: ignore[list-item]
                "mean_duration_s": _nanmean([a.get("duration_s", 0.0) for a in agent_rows]),  # type: ignore[list-item]
                "mean_critic_score": _nanmean([a.get("critic_score") for a in agent_rows]),  # type: ignore[list-item]
            }
        summary.append(row)
    return summary


def compare_modes(records: list[dict[str, Any]], metric: str = "mae") -> list[dict[str, Any]]:
    """Попарное сравнение режимов на одних рядах (стат. тесты — бонус задания)."""
    by_key: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if "metrics" in record:
            key = (record["series_id"], record["horizon"], record["data_fraction"])
            by_key[key][record["mode"]] = record

    pairs: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for key, modes in by_key.items():
        names = sorted(modes)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                seen.add((a, b, key[1], key[2]))

    for a, b, horizon, fraction in sorted(seen):
        common = [
            modes for modes in by_key.values()
            if a in modes and b in modes
            and modes[a]["horizon"] == horizon and modes[a]["data_fraction"] == fraction
        ]
        if not common:
            continue
        values_a = [m[a]["metrics"][metric] for m in common]
        values_b = [m[b]["metrics"][metric] for m in common]
        dm_pvalues = [
            diebold_mariano(
                m[a]["actuals"], m[a]["predictions"], m[b]["predictions"], horizon=horizon
            )["pvalue"]
            for m in common
        ]
        mean_a, mean_b = _nanmean(values_a), _nanmean(values_b)  # type: ignore[arg-type]
        pairs.append(
            {
                "models_compared": [a, b],
                "horizon": horizon,
                "data_fraction": fraction,
                "metric": metric,
                f"mean_{metric}_{a}": mean_a,
                f"mean_{metric}_{b}": mean_b,
                "winner": a if (mean_a or 0) < (mean_b or 0) else b,
                "significance": {
                    "wilcoxon": wilcoxon_test(values_a, values_b),  # type: ignore[arg-type]
                    "bootstrap": bootstrap_mean_diff_ci(values_a, values_b),  # type: ignore[arg-type]
                    "paired_ttest": paired_ttest(values_a, values_b),  # type: ignore[arg-type]
                    "dm_mean_pvalue": _nanmean(dm_pvalues),  # type: ignore[arg-type]
                    "dm_significant_share": (
                        round(sum(1 for p in dm_pvalues if p < 0.05) / len(dm_pvalues), 4)
                    ),
                },
                "n_series": len(common),
            }
        )
    return pairs


# ============================================================================
# Полный прогон
# ============================================================================
async def run_evaluation(
    configs: list[EvalConfig],
    windows: list[dict[str, Any]],
    retriever: Any | None = None,
    supervisor: Any | None = None,
) -> dict[str, Any]:
    """Прогон всех (режим × горизонт × доля) по тестовым окнам."""
    if any(c.mode == "rag" for c in configs) and retriever is None:
        from backend.retrieval.hybrid_retriever import build_hybrid_retriever

        retriever = await build_hybrid_retriever()
    if any(c.mode == "agents" for c in configs) and supervisor is None:
        from backend.agents.supervisor import build_supervisor

        # лог рассуждений только в памяти: оценка не пишет в reasoning_logs
        supervisor = await build_supervisor(use_db_logs=False)

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        for config in configs:
            try:
                record = await evaluate_window(window, config, retriever, supervisor)
            except Exception as exc:  # noqa: BLE001 — одно окно не должно ронять прогон
                logger.exception("eval failed: %s / %s", window.get("series_id"), config)
                record = {
                    "series_id": window.get("series_id"), "mode": config.mode,
                    "horizon": config.horizon, "data_fraction": config.data_fraction,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            records.append(record)
        logger.info("eval: %d / %d окон", index, len(windows))

    summary = aggregate(records)
    return {
        "records": records,
        "summary": summary,
        "comparisons": compare_modes(records),
        "n_windows": len(windows),
        "n_records": len(records),
        "n_errors": sum(1 for r in records if "error" in r),
        "total_time_s": round(time.perf_counter() - started, 2),
        "configs": [asdict(c) for c in configs],
    }


def summary_frame(summary: list[dict[str, Any]]) -> pd.DataFrame:
    """Сводная таблица для CSV/отчёта."""
    rows = []
    for row in summary:
        flat = {k: v for k, v in row.items() if k != "agent_metrics"}
        flat.update(row.get("agent_metrics") or {})
        rows.append(flat)
    return pd.DataFrame(rows)
