"""Статистические тесты для сравнения прогнозов (Этап 13, бонус +3% по заданию MLF).

- Diebold-Mariano — сравнение точности двух прогнозов на одном ряду
  (дифференциал потерь, дисперсия по Ньюи-Уэсту с лагом горизонта);
- Wilcoxon signed-rank — сравнение распределений ошибок по нескольким рядам;
- Bootstrap-доверительные интервалы для средней разности метрик;
- paired t-test — парное сравнение по рядам.

Применение обязательно при сравнении режимов (baseline vs +rag vs +agents)
в отчёте (задание MLF, раздел «Оценка качества»).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def _loss(errors: np.ndarray, kind: str = "squared") -> np.ndarray:
    errors = np.abs(np.asarray(errors, dtype=float))
    return errors**2 if kind == "squared" else errors


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
) -> dict[str, float]:
    """Тест Диеболда-Мариано: H0 — прогнозы равноточны.

    Возвращает dm-статистику, p-value и флаг значимости (p < 0.05).
    dm > 0 — прогноз a хуже b (средняя потеря a больше).
    """
    y_true = np.asarray(y_true, dtype=float)
    d = _loss(y_true - np.asarray(pred_a), loss) - _loss(
        y_true - np.asarray(pred_b), loss
    )
    n = len(d)
    d_mean = float(np.mean(d))
    if n < 3 or np.allclose(d, d[0]):
        return {"dm": 0.0, "pvalue": 1.0, "significant": False, "n": n}

    # дисперсия с автокорреляцией до лага h-1 (Newey-West)
    centered = d - d_mean
    var_d = float(np.sum(centered**2) / n)
    for lag in range(1, min(horizon, n - 1)):
        weight = 1.0 - lag / horizon
        cov = float(np.sum(centered[lag:] * centered[:-lag]) / n)
        var_d += 2.0 * weight * cov
    if var_d <= 0:
        return {"dm": 0.0, "pvalue": 1.0, "significant": False, "n": n}

    dm = d_mean / np.sqrt(var_d / n)
    # поправка Харви-Лейборна-Ньюболда для малых выборок
    factor = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm *= factor
    pvalue = float(2 * (1 - scipy_stats.norm.cdf(abs(dm))))
    return {
        "dm": round(float(dm), 4),
        "pvalue": round(pvalue, 4),
        "significant": bool(pvalue < 0.05),
        "n": n,
    }


def wilcoxon_test(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    """Знаковых рангов Уилкоксона для пар (a, b) по рядам (например, MAE)."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b
    if len(diff) < 1 or np.allclose(diff, 0):
        return {"statistic": 0.0, "pvalue": 1.0, "significant": False, "n": int(len(diff))}
    result = scipy_stats.wilcoxon(a, b, zero_method="wilcox")
    return {
        "statistic": round(float(result.statistic), 4),
        "pvalue": round(float(result.pvalue), 4),
        "significant": bool(result.pvalue < 0.05),
        "n": int(len(diff)),
    }


def bootstrap_mean_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap-CI средней разности mean(a) - mean(b) по рядам."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return {
        "mean_diff": round(float((a - b).mean()), 6),
        "ci_low": round(float(np.quantile(diffs, alpha / 2)), 6),
        "ci_high": round(float(np.quantile(diffs, 1 - alpha / 2)), 6),
        "significant": bool(np.quantile(diffs, alpha / 2) > 0 or np.quantile(diffs, 1 - alpha / 2) < 0),
        "n": int(n),
    }


def paired_ttest(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    """Парный t-тест по рядам."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) < 2 or np.allclose(a, b):
        return {"statistic": 0.0, "pvalue": 1.0, "significant": False, "n": int(len(a))}
    result = scipy_stats.ttest_rel(a, b)
    return {
        "statistic": round(float(result.statistic), 4),
        "pvalue": round(float(result.pvalue), 4),
        "significant": bool(result.pvalue < 0.05),
        "n": int(len(a)),
    }
