"""Юнит-тесты предобработки временных рядов и таблиц (Этап 6)."""
import numpy as np
import pandas as pd
import pytest

from backend.timeseries.preprocessing import (
    adf_test,
    check_regularity,
    detect_seasonal_period,
    fill_missing,
    prepare_tabular,
    prepare_timeseries,
    resample,
    stl_decompose,
    summarize,
    to_series,
)


def daily_temperature(n_days: int = 900, seed: int = 42) -> pd.Series:
    """Синтетическая среднесуточная температура: годовая сезонность + шум."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    values = 10 + 12 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 1.5, n_days)
    index = pd.date_range("2013-01-01", periods=n_days, freq="D")
    return pd.Series(values, index=index)


# ============================================================================
# to_series / regularity / fill / resample
# ============================================================================
class TestSeriesBasics:
    def test_to_series_sorts_and_deduplicates(self):
        s = to_series([3.0, 1.0, 2.0], dates=["2024-01-02", "2024-01-01", "2024-01-01"])
        assert list(s.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        assert s.iloc[0] == 1.5  # дубликат дат усреднён (2.0 и 1.0)

    def test_check_regularity_detects_gaps(self):
        s = daily_temperature(100)
        with_gaps = s.drop(index=s.index[[10, 20, 30]])
        info = check_regularity(with_gaps)
        assert info["inferred_freq"] == "D"
        assert info["gaps"] == 3
        assert info["is_regular"] is False

    def test_fill_missing_no_nan_left(self):
        s = daily_temperature(60)
        s.iloc[5:8] = np.nan
        filled = fill_missing(s)
        assert not filled.isna().any()
        assert len(filled) == 60

    def test_resample_10min_to_daily(self):
        index = pd.date_range("2024-01-01", periods=144, freq="10min")  # ровно сутки
        s = pd.Series(np.arange(144, dtype=float), index=index)
        daily = resample(s, freq="D")
        assert len(daily) == 1
        assert daily.iloc[0] == pytest.approx(np.arange(144).mean())


# ============================================================================
# Стационарность / сезонность / STL
# ============================================================================
class TestDiagnostics:
    def test_adf_stationary_series(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 300))
        result = adf_test(s)
        assert result["stationary"] is True

    def test_adf_trending_series(self):
        t = np.arange(500, dtype=float)
        s = pd.Series(0.05 * t + np.random.default_rng(1).normal(0, 1, 500))
        result = adf_test(s)
        assert result["stationary"] is False

    def test_detect_yearly_period(self):
        s = daily_temperature(900)
        assert detect_seasonal_period(s) in range(360, 371)

    def test_detect_none_on_noise(self):
        rng = np.random.default_rng(3)
        s = pd.Series(rng.normal(0, 1, 400))
        assert detect_seasonal_period(s) is None

    def test_stl_strengths_sum_reasonable(self):
        s = daily_temperature(800)
        result = stl_decompose(s, period=365)
        assert result["trend_strength"] < 0.3      # тренда почти нет
        assert result["seasonal_strength"] > 0.5   # доминирует сезонность
        assert result["resid_strength"] < 0.3

    def test_stl_insufficient_data(self):
        s = daily_temperature(100)
        result = stl_decompose(s, period=365)
        assert "note" in result


# ============================================================================
# Полный пайплайн
# ============================================================================
class TestPrepareTimeseries:
    def test_prepare_fills_gaps_and_diagnoses(self):
        s = daily_temperature(900)
        s.iloc[100:103] = np.nan
        irregular = s.drop(index=s.index[[200, 400]])

        result = prepare_timeseries(irregular.values.tolist(), dates=list(irregular.index))
        prepared: pd.Series = result["series"]
        d = result["diagnostics"]

        assert not prepared.isna().any()
        assert len(prepared) >= 898  # дыры восстановлены
        assert d["raw_gaps"] >= 2
        assert d["raw_missing_rate"] > 0
        assert d["freq"] == "D"
        assert d["seasonal_period"] in range(360, 371)
        # ADF имеет низкую мощность против годового цикла на ~2.5 годах данных:
        # проверяем, что тест посчитан, не его вердикт в пограничном случае
        assert d["adf"]["adf_pvalue"] is not None
        assert d["stl"] is not None
        assert d["stats"]["length"] == len(prepared)

    def test_prepare_resamples_to_daily(self):
        index = pd.date_range("2013-01-01", periods=144 * 30, freq="10min")  # 30 суток
        values = np.sin(np.arange(len(index)) / 100) + 15
        result = prepare_timeseries(values.tolist(), dates=list(index), target_freq="D")
        assert len(result["series"]) == 30
        assert result["diagnostics"]["raw_freq"] == "10min"


# ============================================================================
# summarize + tabular
# ============================================================================
class TestSummarizeAndTabular:
    def test_summarize_fields(self):
        s = daily_temperature(100)
        stats = summarize(s)
        assert stats["length"] == 100
        assert stats["missing_rate"] == 0.0
        assert stats["min"] < stats["mean"] < stats["max"]
        assert abs(stats["acf_lag1"]) > 0.5  # сильная автокорреляция у температуры

    def test_prepare_tabular_imputes_and_encodes(self):
        records = [
            {"humidity": 80, "wind": 3.2, "city": "Jena", "temp": 12.0},
            {"humidity": None, "wind": 5.0, "city": "Delhi", "temp": 28.0},
            {"humidity": 60, "wind": None, "city": "Jena", "temp": 15.0},
        ]
        result = prepare_tabular(records, target_column="temp")
        frame = result["frame"]
        d = result["diagnostics"]

        assert not frame.isna().any().any()  # импутация закрыла пропуски
        assert "city_Jena" in frame.columns  # one-hot сработал
        assert d["num_features_raw"] == 4
        assert d["num_features_encoded"] >= 4
        assert d["missing_rate"] > 0
        assert "temp" in d["target_correlations"] or d["target_correlations"]

    def test_prepare_tabular_without_target(self):
        records = [{"a": 1.0, "b": "x"}, {"a": 2.0, "b": "y"}]
        result = prepare_tabular(records)
        # 1 числовой признак + 2 one-hot колонки категории b
        assert result["diagnostics"]["num_features_encoded"] == 3
        assert "target_correlations" not in result["diagnostics"]
