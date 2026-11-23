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


# ============================================================================
# Классификация (tabular-задачи: Accuracy, F1, ROC-AUC)
# ============================================================================
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля верных ответов."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary"
) -> float:
    """F1: 2*P*R/(P+R); average — 'binary' | 'macro' | 'weighted'."""
    from sklearn.metrics import f1_score as _sk_f1

    return float(_sk_f1(y_true, y_pred, average=average, zero_division=0))


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """ROC-AUC по вероятностям положительного класса; NaN, если один класс."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_scores))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Сводка для классификации: accuracy, f1 (+ roc_auc при наличии score)."""
    result: dict[str, float | None] = {
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }
    if y_scores is not None:
        value = roc_auc(y_true, y_scores)
        result["roc_auc"] = None if np.isnan(value) else round(value, 4)
    return result


# ============================================================================
# Диагностика остатков (задание MLF: Ljung-Box, Shapiro-Wilk, ACF/PACF)
# ============================================================================
def ljung_box_pvalue(residuals: np.ndarray, lags: int | None = None) -> float | None:
    """p-value теста Льюнга-Бокса на автокорреляцию остатков.

    p < 0.05 — остатки автокоррелированы, модель не поймала структуру ряда.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n < 4:
        return None
    lag = lags or min(10, max(1, n // 2))
    lag = min(lag, n - 1)
    result = acorr_ljungbox(residuals, lags=[lag], return_df=True)
    return float(result["lb_pvalue"].iloc[0])


def shapiro_pvalue(residuals: np.ndarray) -> float | None:
    """p-value теста Шапиро-Уилка на нормальность (3 <= n <= 5000)."""
    from scipy.stats import shapiro

    n = len(residuals)
    if not 3 <= n <= 5000:
        return None
    return float(shapiro(np.asarray(residuals, dtype=float)).pvalue)
