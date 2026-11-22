"""Предобработка погодных временных рядов (Этап 6).

Метеоспецифика: ряды приходят с разной частотой (10-мин Jena Climate, дневные
наблюдения), с пропусками и нерегулярными датами. Здесь ряд приводится к
регулярному виду, заполняются пропуски, оцениваются сезонность (STL),
стационарность (ADF) и базовые статистики для диагностики.

Для tabular-задач — prepare_tabular: типы, импутация, one-hot, статистика.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

ADF_PVALUE_THRESHOLD = 0.05
SEASONAL_ACF_THRESHOLD = 0.2


# ============================================================================
# Базовые операции над рядом
# ============================================================================
def to_series(values: list[float] | pd.Series, dates: list | None = None) -> pd.Series:
    """Строит pd.Series с DatetimeIndex (сортировка, дубликаты усредняются)."""
    s = pd.Series(values, dtype="float64")
    if dates is not None:
        if len(dates) != len(s):
            raise ValueError(
                f"Длины не совпадают: values={len(s)}, dates={len(dates)}"
            )
        s.index = pd.to_datetime(pd.Series(dates))
    else:
        s.index = pd.date_range("2000-01-01", periods=len(s), freq="D")
    s = s.sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    return s.astype("float64")


def _mode_step(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    """Модальный шаг между датами — устойчив к редким пропускам."""
    if len(index) < 3:
        return None
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return None
    return diffs.mode().iloc[0]


def _canonical_freq(step: pd.Timedelta) -> str:
    """Каноничные частоты: Timedelta(1 день) -> 'D' (а не '24h')."""
    canonical = {
        pd.Timedelta(minutes=1): "min",
        pd.Timedelta(hours=1): "h",
        pd.Timedelta(days=1): "D",
        pd.Timedelta(weeks=1): "W",
    }
    if step in canonical:
        return canonical[step]
    return pd.tseries.frequencies.to_offset(step).freqstr


def check_regularity(s: pd.Series) -> dict[str, Any]:
    """Регулярность индекса: шаг (частота) и число пропущенных дат.

    pd.infer_freq здесь не подходит: он требует подряд идущих дат и
    возвращает None при любой дыре. Считаем модальную разность дат.
    """
    step = _mode_step(s.index)
    if step is None:
        return {"inferred_freq": None, "gaps": 0, "is_regular": True}
    diffs = s.index.to_series().diff().dropna()
    gaps = int((diffs != step).sum())
    return {
        "inferred_freq": _canonical_freq(step),
        "gaps": gaps,
        "is_regular": gaps == 0,
    }


def fill_missing(s: pd.Series, method: str = "time") -> pd.Series:
    """Заполнение пропусков (NaN и отсутствующих дат) интерполяцией по времени."""
    step = _mode_step(s.index)
    freq = step if step is not None else (s.index.freq or pd.infer_freq(s.index))
    full_index = pd.date_range(s.index[0], s.index[-1], freq=freq)
    s = s.reindex(full_index)
    return s.interpolate(method=method).ffill().bfill()


def resample(s: pd.Series, freq: str = "D", agg: str = "mean") -> pd.Series:
    """Ресемплинг (например, 10-мин Jena -> среднесуточные значения)."""
    return s.resample(freq).agg(agg)


def adf_test(s: pd.Series) -> dict[str, Any]:
    """Тест Дики-Фуллера: стационарен ли ряд (p < 0.05)."""
    clean = s.dropna()
    if len(clean) < 20:
        return {"adf_pvalue": None, "stationary": None, "note": "insufficient data"}
    result = adfuller(clean, autolag="AIC")
    pvalue = float(result[1])
    return {"adf_pvalue": round(pvalue, 4), "stationary": pvalue < ADF_PVALUE_THRESHOLD}


def detect_seasonal_period(s: pd.Series) -> int | None:
    """Период сезонности по максимуму положительной автокорреляции.

    Полупериодные лаги дают сильную отрицательную корреляцию (синус в
    противофазе) — их не считаем сезонностью, ищем повторение сигнала.
    """
    clean = s.dropna()
    max_lag = min(len(clean) // 2, 400)
    if max_lag < 10:
        return None
    acf = {lag: clean.autocorr(lag=lag) for lag in range(2, max_lag + 1)}
    positive = {lag: value for lag, value in acf.items() if value > SEASONAL_ACF_THRESHOLD}
    if not positive:
        return None
    best_lag = max(positive, key=positive.get)
    return int(best_lag)


def stl_decompose(s: pd.Series, period: int) -> dict[str, Any]:
    """STL-декомпозиция: доли дисперсии тренда/сезонности/остатка."""
    if len(s) < 2 * period:
        return {"period": period, "note": "insufficient data for STL"}
    result = STL(s.dropna(), period=period, robust=True).fit()
    var = float(np.var(s.dropna()))
    return {
        "period": period,
        "trend_strength": float(np.var(result.trend) / var) if var else 0.0,
        "seasonal_strength": float(np.var(result.seasonal) / var) if var else 0.0,
        "resid_strength": float(np.var(result.resid) / var) if var else 0.0,
    }


def summarize(s: pd.Series) -> dict[str, Any]:
    """Базовые статистики ряда для diagnostics/metadata."""
    clean = s.dropna()
    return {
        "length": int(len(s)),
        "missing_rate": round(float(s.isna().mean()), 4),
        "mean": round(float(clean.mean()), 3) if len(clean) else None,
        "std": round(float(clean.std()), 3) if len(clean) else None,
        "min": round(float(clean.min()), 3) if len(clean) else None,
        "max": round(float(clean.max()), 3) if len(clean) else None,
        "acf_lag1": round(float(s.autocorr(lag=1)), 3) if len(clean) > 2 else None,
    }


# ============================================================================
# Загрузка данных из metadata состояния (общая для DataAnalyst/ModelSelector)
# ============================================================================
def load_series(metadata: dict[str, Any]) -> tuple[list[float], list | None]:
    """Ряд из metadata: inline-значения или CSV (первая колонка — дата)."""
    if metadata.get("series_values") is not None:
        values = [float(v) for v in metadata["series_values"]]
        return values, metadata.get("series_dates")

    path = metadata.get("series_path")
    if path and Path(path).exists():
        df = pd.read_csv(path)
        date_col = df.columns[0]
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        return df[value_col].astype(float).tolist(), df[date_col].tolist()

    raise ValueError("Ряд не найден: задайте metadata['series_values'] или metadata['series_path']")


def load_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Табличные данные из metadata: records или CSV."""
    if metadata.get("records") is not None:
        return metadata["records"]
    path = metadata.get("records_path")
    if path and Path(path).exists():
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("Таблица не найдена: задайте metadata['records'] или metadata['records_path']")


# ============================================================================
# Полный пайплайн предобработки ряда
# ============================================================================
def prepare_timeseries(
    values: list[float] | pd.Series,
    dates: list | None = None,
    target_freq: str = "D",
) -> dict[str, Any]:
    """Приводит ряд к регулярному виду и считает диагностику.

    Возвращает {"series": подготовленный pd.Series, "diagnostics": {...}}.
    """
    s = to_series(values, dates=dates)

    regularity = check_regularity(s)
    raw_missing = float(s.isna().mean())

    if regularity["inferred_freq"] and regularity["inferred_freq"] != target_freq:
        s = resample(s, freq=target_freq)
    s = fill_missing(s)

    period = detect_seasonal_period(s)
    diagnostics: dict[str, Any] = {
        "raw_freq": regularity["inferred_freq"],
        "raw_gaps": regularity["gaps"],
        "raw_missing_rate": round(raw_missing, 4),
        "freq": target_freq,
        "adf": adf_test(s),
        "seasonal_period": period,
        "stl": stl_decompose(s, period) if period else None,
        "stats": summarize(s),
    }
    return {"series": s, "diagnostics": diagnostics}


# ============================================================================
# Tabular: подготовка таблиц (расширяемость архитектуры)
# ============================================================================
def prepare_tabular(
    records: list[dict[str, Any]] | pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Типы, импутация (медиана/мода), one-hot, базовая статистика.

    Возвращает {"frame": подготовленный pd.DataFrame, "diagnostics": {...}}.
    """
    df = pd.DataFrame(records)
    n_rows = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_rate = float(df.isna().mean().mean())

    # Импутация: числовые — медиана, категориальные — мода
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

    # one-hot для категориальных признаков
    encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
    encoded = encoded.astype("float64", copy=False)

    diagnostics: dict[str, Any] = {
        "num_rows": int(n_rows),
        "num_features_raw": int(len(numeric_cols) + len(categorical_cols)),
        "num_features_encoded": int(encoded.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_rate": round(missing_rate, 4),
    }

    if target_column and target_column in df.columns and target_column in numeric_cols:
        correlations = encoded.corr(numeric_only=True).get(target_column)
        if correlations is not None:
            top = correlations.drop(labels=[target_column], errors="ignore")
            top = top.reindex(top.abs().sort_values(ascending=False).index)
            diagnostics["target_correlations"] = {
                str(k): round(float(v), 3) for k, v in top.head(5).items()
            }

    return {"frame": encoded, "diagnostics": diagnostics}
