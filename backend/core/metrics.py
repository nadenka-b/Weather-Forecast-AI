"""Метрики качества прогнозов (реализованы на Этапе 7, расширются на Этапе 12).

Точность (регрессия/временные ряды): MAE, RMSE, MAPE, WAPE, MASE
(относительно наивного сезонного прогноза). Диагностика остатков
(Ljung-Box, Shapiro-Wilk) и классификационные метрики — Этап 12.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff**2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE в процентах; точки с y_true около нуля пропускаются."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE (взвешенный APE): сумма ошибок / сумма факта, в процентах."""
    y_true = np.asarray(y_true, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - np.asarray(y_pred))) / denom * 100)


def naive_forecast(y_train: np.ndarray, horizon: int, season: int = 1) -> np.ndarray:
    """Наивный сезонный прогноз: повтор последних `season` значений."""
    y_train = np.asarray(y_train)
    if season > 1 and len(y_train) >= season:
        tail = y_train[-season:]
        reps = int(np.ceil(horizon / season))
        return np.tile(tail, reps)[:horizon]
    return np.full(horizon, y_train[-1])


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 1,
) -> float:
    """MASE: MAE модели / MAE наивного сезонного прогноза (in-sample).

    < 1 — модель лучше наива, 1 — не хуже/не лучше, > 1 — хуже наива.
    """
    y_train = np.asarray(y_train)
    if season > 1 and len(y_train) > season:
        naive_errors = np.abs(y_train[season:] - y_train[:-season])
    else:
        naive_errors = np.abs(np.diff(y_train))
    denom = float(np.mean(naive_errors)) if len(naive_errors) else 0.0
    if denom == 0:
        return float("nan")
    return mae(y_true, y_pred) / denom


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    season: int = 1,
) -> dict[str, float | None]:
    """Сводка метрик: mae, rmse, mape, wape (+ mase, если задан train)."""
    result: dict[str, float | None] = {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
        "wape": round(wape(y_true, y_pred), 4),
    }
    if y_train is not None:
        result["mase"] = round(mase(y_true, y_pred, y_train, season=season), 4)
    return result
