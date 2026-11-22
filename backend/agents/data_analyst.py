"""DataAnalyst — агент анализа данных (Этап 6).

task_type="timeseries": загрузка ряда (inline-значения или CSV-путь),
приведение к DatetimeIndex, регулярность, ресемплинг, заполнение пропусков,
STL-декомпозиция, ADF-тест, сезонность. Результат — state.diagnostics.

task_type="tabular": проверка типов, импутация, one-hot, базовая статистика,
корреляции с целевой переменной (задание MLF, ветка расширяемости).

Ряд передаётся по ссылке (metadata) — тяжёлых объектов в state нет:
- "series_values": list[float] (+ опционально "series_dates"),
- "series_path": путь к CSV (колонка даты + значение),
- "records": list[dict] для tabular (+ target_column).
"""
import logging
from typing import Any

from backend.agents.base import BaseAgent
from backend.core.models import ForecastState
from backend.timeseries.preprocessing import (
    load_records,
    load_series,
    prepare_tabular,
    prepare_timeseries,
)

logger = logging.getLogger(__name__)


class DataAnalyst(BaseAgent):
    """Анализирует данные и готовит диагностику для ModelSelector."""

    def __init__(self, target_freq: str = "D"):
        super().__init__(name="data_analyst", description="EDA и предобработка ряда/таблицы")
        self.target_freq = target_freq

    async def process(self, state: ForecastState) -> ForecastState:
        if state.task_type == "tabular":
            return self._process_tabular(state)
        return self._process_timeseries(state)

    # ------------------------------------------------------------------
    # Временные ряды
    # ------------------------------------------------------------------
    def _process_timeseries(self, state: ForecastState) -> ForecastState:
        values, dates = load_series(state.metadata)
        target_freq = state.freq or self.target_freq

        result = prepare_timeseries(values, dates=dates, target_freq=target_freq)
        diagnostics: dict[str, Any] = result["diagnostics"]
        series: pd.Series = result["series"]

        state.diagnostics = diagnostics
        state.freq = target_freq
        state.metadata.update(
            {
                "num_observations": diagnostics["stats"]["length"],
                "seasonal_period": diagnostics["seasonal_period"],
                "stationary": diagnostics["adf"]["stationary"],
                "prepared_mean": diagnostics["stats"]["mean"],
                "prepared_std": diagnostics["stats"]["std"],
            }
        )
        self._last_result = {
            "length": diagnostics["stats"]["length"],
            "seasonal_period": diagnostics["seasonal_period"],
            "stationary": diagnostics["adf"]["stationary"],
        }
        logger.info(
            "DataAnalyst: series=%s n=%s period=%s stationary=%s",
            state.series_id,
            diagnostics["stats"]["length"],
            diagnostics["seasonal_period"],
            diagnostics["adf"]["stationary"],
        )
        return state

    # ------------------------------------------------------------------
    # Табличные данные
    # ------------------------------------------------------------------
    def _process_tabular(self, state: ForecastState) -> ForecastState:
        records = load_records(state.metadata)
        result = prepare_tabular(records, target_column=state.target_column)
        state.diagnostics = result["diagnostics"]
        state.metadata.update(
            {
                "num_features": result["diagnostics"]["num_features_encoded"],
                "num_rows": result["diagnostics"]["num_rows"],
                "target_correlations": result["diagnostics"].get("target_correlations"),
            }
        )
        self._last_result = result["diagnostics"]
        return state
