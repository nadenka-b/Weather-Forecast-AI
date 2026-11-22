"""Классические модели прогнозирования (Этап 7).

Каждая fit_* функция — это «инструмент» реестра Tool Calling: принимает ряд,
горизонт и гиперпараметры, обучает на train (последние `horizon` точек —
валидация) и возвращает словарь с метриками и итоговой спецификацией модели.

ARIMA/SARIMA — statsmodels (при order=None небольшой grid search по валидации —
fallback-правило «SARIMA с автоподбором порядка» из задания MLF).
Prophet и XGBoost импортируются лениво: если пакет не установлен,
инструмент возвращает понятную ошибку ToolExecutionError.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from backend.core.metrics import regression_metrics

logger = logging.getLogger(__name__)

# Небольшой grid для автоподбора (fallback-правило задания MLF)
_AUTO_ORDERS: tuple[tuple[int, int, int], ...] = ((1, 1, 1), (0, 1, 1), (2, 1, 1))


class ToolExecutionError(RuntimeError):
    """Ошибка выполнения инструмента (нет пакета, не сошлась модель и т.п.)."""


def time_split(series: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Временной сплит: последние `horizon` точек — валидация."""
    horizon = max(1, min(horizon, len(series) // 5))
    return series.iloc[:-horizon], series.iloc[-horizon:]


def _fit_sarimax(
    train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None,
):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order or (0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=100)


def fit_arima(
    series: pd.Series,
    horizon: int,
    order: list[int] | None = None,
    seasonal_order: list[int] | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """ARIMA/SARIMA. order=None -> автоподбор по grid (fallback-правило)."""
    train, val = time_split(series, horizon)
    auto = order is None
    seasonal_candidates: list[tuple[int, int, int, int] | None] = [None]
    if season and season > 1 and len(train) > 2 * season:
        seasonal_candidates = [(1, 1, 1, season), None]

    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int] | None]] = (
        [(tuple(order), tuple(seasonal_order) if seasonal_order else None)]
        if not auto
        else [(o, s) for s in seasonal_candidates for o in _AUTO_ORDERS]
    )

    best: dict[str, Any] | None = None
    last_error: Exception | None = None
    for cand_order, cand_seasonal in candidates:
        try:
            fitted = _fit_sarimax(train, cand_order, cand_seasonal)
            pred = fitted.forecast(len(val))
            metrics = regression_metrics(
                val.values, np.asarray(pred), train.values, season=season or 1
            )
        except Exception as exc:  # noqa: BLE001 — пробуем следующий кандидат
            last_error = exc
            logger.debug("SARIMAX %s/%s failed: %s", cand_order, cand_seasonal, exc)
            continue
        candidate_result = {
            "model": "sarima" if cand_seasonal else "arima",
            "order": list(cand_order),
            "seasonal_order": list(cand_seasonal) if cand_seasonal else None,
            "auto": auto,
            "metrics": metrics,
            "n_train": int(len(train)),
            "aic": round(float(fitted.aic), 2),
        }
        if best is None or metrics["mae"] < best["metrics"]["mae"]:
            best = candidate_result

    if best is None:
        raise ToolExecutionError(f"SARIMAX не обучилась ни на одном кандидате: {last_error}")
    return best


def fit_prophet(
    series: pd.Series,
    horizon: int,
    seasonality_mode: str = "additive",
    season: int | None = None,
) -> dict[str, Any]:
    """Prophet (ленивый импорт — тяжёлая зависимость)."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ToolExecutionError("Пропущен prophet: poetry install prophet") from exc

    train, val = time_split(series, horizon)
    freq = series.index.freqstr or "D"
    if freq.upper().startswith("24H"):
        freq = "D"

    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality=False,
    )
    frame = pd.DataFrame({"ds": train.index, "y": train.values})
    model.fit(frame)

    future = model.make_future_dataframe(periods=len(val), freq=freq)
    forecast = model.predict(future)
    pred = forecast.iloc[-len(val):]["yhat"].values

    return {
        "model": "prophet",
        "seasonality_mode": seasonality_mode,
        "metrics": regression_metrics(val.values, pred, train.values, season=season or 1),
        "n_train": int(len(train)),
    }


def make_lag_features(series: pd.Series, lags: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Матрица лаговых признаков: лаги 1..lags + скользящее среднее окна 7."""
    values = series.values
    rows, targets = [], []
    for i in range(max(lags, 7), len(values)):
        window = values[i - max(lags, 7) : i]
        rows.append([values[i - lag] for lag in range(1, lags + 1)] + [window[-7:].mean()])
        targets.append(values[i])
    return np.asarray(rows, dtype="float64"), np.asarray(targets, dtype="float64")


def fit_xgboost(
    series: pd.Series,
    horizon: int,
    lags: int = 7,
    n_estimators: int = 200,
    season: int | None = None,
) -> dict[str, Any]:
    """XGBoost на лаговых признаках с рекурсивным прогнозом на горизонт."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ToolExecutionError("Пропущен xgboost: poetry install xgboost") from exc

    n_val = max(1, min(horizon, len(series) // 5))
    train_series = series.iloc[:-n_val]
    val = series.iloc[-n_val:]

    X, y = make_lag_features(train_series, lags=lags)
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    # рекурсивный прогноз: каждая предсказанная точка становится лагом
    history = list(series.values[: -n_val])
    predictions: list[float] = []
    for _ in range(n_val):
        window = np.asarray(history[-max(lags, 7) :], dtype="float64")
        row = [window[-lag] for lag in range(1, lags + 1)] + [window[-7:].mean()]
        pred = float(model.predict(np.asarray([row]))[0])
        predictions.append(pred)
        history.append(pred)

    return {
        "model": "xgboost",
        "lags": lags,
        "n_estimators": n_estimators,
        "metrics": regression_metrics(
            val.values, np.asarray(predictions), train_series.values, season=season or 1
        ),
        "n_train": int(len(train_series)),
    }


def fit_tabular_xgboost(
    frame: pd.DataFrame,
    target_column: str,
    n_estimators: int = 200,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """XGBoost для табличных задач (расширяемость архитектуры на tabular)."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ToolExecutionError("Пропущен xgboost: poetry install xgboost") from exc

    X = frame.drop(columns=[target_column]).values
    y = frame[target_column].values

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(y))
    n_test = max(1, int(len(y) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])

    return {
        "model": "xgboost_tabular",
        "n_estimators": n_estimators,
        "metrics": regression_metrics(y[test_idx], pred, y[train_idx]),
        "n_train": int(len(train_idx)),
    }
