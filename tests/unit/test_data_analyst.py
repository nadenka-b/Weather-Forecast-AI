"""Юнит-тесты агента DataAnalyst (Этап 6) и его интеграция с Supervisor."""
import numpy as np
import pandas as pd
import pytest

from backend.agents.data_analyst import DataAnalyst
from backend.agents.supervisor import Supervisor
from backend.core.models import ForecastState


def make_temperature_state(n_days: int = 900, with_gaps: bool = True) -> ForecastState:
    rng = np.random.default_rng(7)
    t = np.arange(n_days)
    values = 8 + 11 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 1.5, n_days)
    index = pd.date_range("2013-01-01", periods=n_days, freq="D")
    if with_gaps:
        # пропуск в данных = удалённая строка целиком (и дата, и значение)
        keep = [i for i in range(n_days) if i not in (50, 51, 300)]
        values, dates = values[keep], index[keep]
    else:
        dates = index
    return ForecastState(
        series_id="jena_mean_temp",
        horizon=14,
        freq="D",
        metadata={"series_values": values.tolist(), "series_dates": list(dates)},
    )


# ============================================================================
# Ветка timeseries
# ============================================================================
class TestDataAnalystTimeseries:
    async def test_diagnostics_filled(self):
        agent = DataAnalyst()
        state = await agent.process(make_temperature_state())

        assert state.diagnostics is not None
        assert state.diagnostics["freq"] == "D"
        assert state.diagnostics["raw_gaps"] >= 2  # удалённые даты обнаружены
        assert state.diagnostics["seasonal_period"] in range(360, 371)
        assert state.diagnostics["adf"]["adf_pvalue"] is not None
        assert state.diagnostics["stl"] is not None

        assert state.metadata["num_observations"] == 900
        assert state.metadata["seasonal_period"] in range(360, 371)
        assert state.metadata["stationary"] is not None
        assert agent.get_last_result()["length"] == 900

    async def test_freq_defaults_to_daily(self):
        state = make_temperature_state()
        state.freq = None
        agent = DataAnalyst()
        state = await agent.process(state)
        assert state.freq == "D"

    async def test_missing_series_raises(self):
        state = ForecastState(series_id="empty", horizon=7, metadata={})
        agent = DataAnalyst()
        with pytest.raises(ValueError, match="series_values|series_path"):
            await agent.process(state)


# ============================================================================
# Ветка tabular
# ============================================================================
class TestDataAnalystTabular:
    async def test_tabular_diagnostics(self):
        state = ForecastState(
            series_id="housing_sample",
            horizon=0,
            task_type="tabular",
            target_column="price",
            metadata={
                "records": [
                    {"area": 120, "rooms": 3, "district": "north", "price": 310},
                    {"area": 80, "rooms": None, "district": "south", "price": 190},
                    {"area": 150, "rooms": 4, "district": "north", "price": 400},
                ]
            },
        )
        agent = DataAnalyst()
        state = await agent.process(state)

        assert state.diagnostics["num_features_raw"] == 4
        assert state.diagnostics["num_features_encoded"] >= 4
        assert state.metadata["num_rows"] == 3
        assert state.diagnostics["missing_rate"] > 0
        assert state.diagnostics.get("target_correlations")

    async def test_tabular_missing_records_raises(self):
        state = ForecastState(
            series_id="no_data", horizon=0, task_type="tabular", metadata={}
        )
        agent = DataAnalyst()
        with pytest.raises(ValueError, match="records"):
            await agent.process(state)


# ============================================================================
# Интеграция с Supervisor
# ============================================================================
class TestDataAnalystInPipeline:
    async def test_real_analyst_with_mock_rest(self):
        from tests.unit.test_supervisor import (
            MockModelSelector,
            ScriptedCritic,
            ScriptedValidator,
        )

        sup = Supervisor(
            agents={
                "data_analyst": DataAnalyst(),
                "model_selector": MockModelSelector(),
                "validator": ScriptedValidator(passed=True),
                "critic": ScriptedCritic([8.0]),
            }
        )
        result = await sup.process(make_temperature_state(with_gaps=False))

        # реальный DataAnalyst отработал внутри пайплайна
        assert result.diagnostics is not None
        assert result.metadata["num_observations"] == 900
        assert result.metadata["supervisor"]["stopped_reason"] == "converged"

        statuses = {
            e["agent_name"]: e["status"] for e in result.metadata["reasoning_trace"]
        }
        assert statuses["data_analyst"] == "success"
        # Editor участвует только в цикле коррекции: при converged его в трассе нет
        assert "editor" not in statuses
