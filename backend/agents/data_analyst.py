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
from pathlib import Path
from typing import Any

import pandas as pd

from backend.agents.base import BaseAgent
from backend.core.models import ForecastState
from backend.timeseries.preprocessing import prepare_tabular, prepare_timeseries

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
        values, dates = self._load_series(state)
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

    @staticmethod
    def _load_series(state: ForecastState) -> tuple[list[float], list | None]:
        """Ряд из metadata: inline-значения или CSV-файл."""
        meta = state.metadata
        if meta.get("series_values") is not None:
            values = [float(v) for v in meta["series_values"]]
            dates = meta.get("series_dates")
            return values, dates

        path = meta.get("series_path")
        if path and Path(path).exists():
            df = pd.read_csv(path)
            date_col = df.columns[0]
            value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            return df[value_col].astype(float).tolist(), df[date_col].tolist()

        raise ValueError(
            "Ряд не найден: задайте metadata['series_values'] или metadata['series_path']"
        )

    # ------------------------------------------------------------------
    # Табличные данные
    # ------------------------------------------------------------------
    def _process_tabular(self, state: ForecastState) -> ForecastState:
        records = state.metadata.get("records")
        if records is None:
            path = state.metadata.get("records_path")
            if path and Path(path).exists():
                records = pd.read_csv(path).to_dict(orient="records")
            else:
                raise ValueError(
                    "Таблица не найдена: задайте metadata['records'] или metadata['records_path']"
                )

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
