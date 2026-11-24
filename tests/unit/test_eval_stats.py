"""Тесты статистических тестов сравнения (Этап 13, бонус задания MLF)."""
import numpy as np
import pytest

from backend.evaluation.stats import (
    bootstrap_mean_diff_ci,
    diebold_mariano,
    paired_ttest,
    wilcoxon_test,
)


class TestDieboldMariano:
    def test_identical_forecasts_not_significant(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 40)
        result = diebold_mariano(y, y + 0.1, y + 0.1, horizon=7)
        assert result["pvalue"] == 1.0
        assert result["significant"] is False

    def test_clearly_worse_forecast_is_significant(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 60)
        good = y + rng.normal(0, 0.1, 60)
        bad = y + rng.normal(0, 5.0, 60)
        result = diebold_mariano(y, bad, good, horizon=7)
        assert result["dm"] > 0          # a (bad) хуже b (good)
        assert result["significant"] is True

    def test_too_short_series(self):
        assert diebold_mariano([1.0, 2.0], [1.0, 2.0], [1.5, 2.5])["pvalue"] == 1.0


class TestWilcoxon:
    def test_consistent_difference_significant(self):
        rng = np.random.default_rng(2)
        b = rng.uniform(0.5, 1.5, 30)
        a = b + 2.0
        result = wilcoxon_test(a, b)
        assert result["significant"] is True and result["n"] == 30

    def test_equal_values(self):
        values = np.ones(10)
        assert wilcoxon_test(values, values)["pvalue"] == 1.0


class TestBootstrap:
    def test_ci_excludes_zero_for_constant_diff(self):
        a = np.arange(10, dtype=float) + 5.0
        b = np.arange(10, dtype=float)
        result = bootstrap_mean_diff_ci(a, b, n_boot=200)
        assert result["mean_diff"] == pytest.approx(5.0)
        assert result["ci_low"] > 0 and result["ci_high"] > 0
        assert result["significant"] is True

    def test_ci_contains_zero_for_same_distribution(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0, 1, 200)
        result = bootstrap_mean_diff_ci(a, b, n_boot=300)
        assert result["ci_low"] < 0 < result["ci_high"]
        assert result["significant"] is False

    def test_deterministic(self):
        a, b = np.arange(20.0), np.arange(20.0) + 1
        assert bootstrap_mean_diff_ci(a, b) == bootstrap_mean_diff_ci(a, b)


class TestPairedTTest:
    def test_significant_and_not(self):
        rng = np.random.default_rng(4)
        b = rng.normal(0, 1, 40)
        a = b + 3.0
        assert paired_ttest(a, b)["significant"] is True
        assert paired_ttest(b, b)["significant"] is False
        assert paired_ttest([1.0], [1.0])["pvalue"] == 1.0
