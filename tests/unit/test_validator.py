"""Юнит-тесты агента Validator (Этап 8)."""
import numpy as np
import pandas as pd

from backend.agents.validator import Validator
from backend.core.models import ForecastState


def make_state(**overrides) -> ForecastState:
    rng = np.random.default_rng(21)
    n = 30
    true = 20 + 4 * np.sin(np.arange(n) / 5) + rng.normal(0, 0.3, n)
    pred = true + rng.normal(0, 0.3, n)
    dates = list(pd.date_range("2024-07-01", periods=n, freq="D"))

    meta = {
        "val_pred": [round(float(v), 4) for v in pred],
        "val_true": [round(float(v), 4) for v in true],
        "val_dates": [str(d.date()) for d in dates],
        "climate_bounds": {
            "global": [10.0, 30.0],
            "monthly": {"7": [14.0, 28.0], "8": [14.0, 28.0]},
        },
    }
    state = ForecastState(series_id="temp_july", horizon=14, freq="D", metadata=meta)
    for key, value in overrides.items():
        if key == "metadata":
            state.metadata.update(value)
        else:
            setattr(state, key, value)
    return state


# ============================================================================
# Успешная валидация
# ============================================================================
class TestValidatorPass:
    async def test_good_forecast_passes(self):
        state = await Validator().process(make_state())
        assert state.validation_passed is True
        report = state.diagnostics["validation"]
        assert report["errors"] == []
        assert report["checks"]["predictions"]["nan_count"] == 0
        assert report["checks"]["climate"]["violations"] == []
        assert "ljung_box_pvalue" in report["checks"]["residuals"]
        assert state.metadata["validator_errors"] == []

    async def test_no_predictions_skips_checks(self):
        state = ForecastState(series_id="early", horizon=7, metadata={})
        state = await Validator().process(state)
        assert state.validation_passed is True
        assert "skipped" in state.diagnostics["validation"]["checks"]["predictions"]

    async def test_no_bounds_skips_climate(self):
        state = make_state()
        state.metadata.pop("climate_bounds")
        state = await Validator().process(state)
        assert "skipped" in state.diagnostics["validation"]["checks"]["climate"]


# ============================================================================
# NaN и длина
# ============================================================================
class TestValidatorNaN:
    async def test_nan_in_predictions_fails(self):
        state = make_state()
        pred = state.metadata["val_pred"]
        pred[3] = None
        state = await Validator().process(state)
        assert state.validation_passed is False
        assert any("NaN" in e for e in state.metadata["validator_errors"])


# ============================================================================
# Климатические пределы
# ============================================================================
class TestValidatorClimate:
    async def test_physically_impossible_temperature_fails(self):
        state = make_state()
        state.metadata["val_pred"] = [60.0] * 30  # +60°C в июле
        state = await Validator().process(state)
        assert state.validation_passed is False
        assert any(
            "Климатические" in e for e in state.metadata["validator_errors"]
        )

    async def test_monthly_bounds_used_when_dates_present(self):
        state = make_state()
        # августовские даты, узкие августовские пределы [24, 26]
        state.metadata["climate_bounds"]["monthly"]["8"] = [24.0, 26.0]
        state.metadata["val_dates"] = [
            str(d.date()) for d in pd.date_range("2024-08-01", periods=30, freq="D")
        ]
        state.metadata["val_pred"] = [30.0] * 30  # выше августовской нормы
        state = await Validator().process(state)
        violations = state.diagnostics["validation"]["checks"]["climate"]["violations"]
        assert violations and violations[0]["month"] == 8

    async def test_global_bounds_when_no_dates(self):
        state = make_state()
        state.metadata.pop("val_dates")
        state.metadata["val_pred"] = [40.0] * 30
        state = await Validator().process(state)
        assert state.validation_passed is False


# ============================================================================
# Тесты остатков
# ============================================================================
class TestValidatorResiduals:
    async def test_autocorrelated_residuals_fail_ljung_box(self):
        state = make_state()
        true = np.asarray(state.metadata["val_true"])
        # прогноз систематически отстаёт: остатки = нарастающий тренд
        ramp = np.linspace(-3.0, 3.0, len(true))
        state.metadata["val_pred"] = [round(float(v), 4) for v in (true - ramp)]
        state = await Validator().process(state)
        assert state.validation_passed is False
        assert any("Ljung-Box" in e for e in state.metadata["validator_errors"])

    async def test_outlier_residuals_warn_shapiro_only(self):
        # детерминированные остатки: тихий ряд + один выброс
        residuals = [0.0] * 19 + [25.0]
        true = np.asarray([20.0] * 20)
        pred = true - np.asarray(residuals)
        state = ForecastState(
            series_id="resid_case",
            horizon=14,
            metadata={
                "val_pred": [round(float(v), 4) for v in pred],
                "val_true": [20.0] * 20,
            },
        )
        state = await Validator().process(state)
        report = state.diagnostics["validation"]
        assert any("Shapiro" in w for w in report["warnings"])
        assert report["errors"] == []  # автокорреляции нет — только warning
        assert state.validation_passed is True


# ============================================================================
# Tabular-ветка
# ============================================================================
class TestValidatorTabular:
    async def test_tabular_metrics_ok(self):
        state = ForecastState(
            series_id="tab",
            horizon=0,
            task_type="tabular",
            target_column="y",
            metadata={
                "val_metrics": {"mae": 1.2, "rmse": 2.0, "wape": 15.0},
                "num_features": 6,
            },
        )
        state = await Validator().process(state)
        assert state.validation_passed is True

    async def test_tabular_non_finite_metric_fails(self):
        state = ForecastState(
            series_id="tab",
            horizon=0,
            task_type="tabular",
            metadata={
                "val_metrics": {"mae": float("nan"), "rmse": 2.0},
                "num_features": 3,
            },
        )
        state = await Validator().process(state)
        assert state.validation_passed is False

    async def test_tabular_no_metrics_skips(self):
        state = ForecastState(
            series_id="tab", horizon=0, task_type="tabular", metadata={}
        )
        state = await Validator().process(state)
        assert state.validation_passed is True
