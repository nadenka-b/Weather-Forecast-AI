"""Юнит-тесты реестра инструментов, LLMManager и классических моделей (Этап 7)."""
import numpy as np
import pandas as pd
import pytest

from backend.core.llm.manager import LLMManager, parse_tool_json
from backend.core.llm.tools import (
    ToolArgumentError,
    UnknownToolError,
    build_default_registry,
)
from backend.timeseries.classical import (
    ToolExecutionError,
    fit_arima,
    fit_prophet,
    fit_tabular_xgboost,
    fit_xgboost,
    make_lag_features,
    make_prophet,
    time_split,
)


def seasonal_series(n: int = 220, season: int = 7, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 15 + 6 * np.sin(2 * np.pi * t / season) + 0.01 * t + rng.normal(0, 0.5, n)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=n, freq="D"))


class ScriptedLLM:
    """Асинхронная заглушка LLM: ответы по сценарию, последний повторяется."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self._last = responses[-1] if responses else ""
        self.calls = 0

    async def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.responses:
            self._last = self.responses.pop(0)
        return self._last


# ============================================================================
# Реестр инструментов
# ============================================================================
class TestToolRegistry:
    def test_default_registry_has_minimum_tools(self):
        registry = build_default_registry()
        assert {"fit_arima", "fit_prophet", "fit_xgboost", "fit_tabular_xgboost"} <= set(
            registry.names
        )

    def test_schemas_openai_format(self):
        schemas = build_default_registry().schemas()
        for schema in schemas:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert fn["name"] and fn["description"]
            assert fn["parameters"]["type"] == "object"

    def test_unknown_tool_raises(self):
        with pytest.raises(UnknownToolError):
            build_default_registry().get("fit_magic")

    def test_run_validates_required_args(self):
        registry = build_default_registry()
        with pytest.raises(ToolArgumentError):
            registry.run("fit_tabular_xgboost", {}, context={})  # нет target_column


# ============================================================================
# Парсинг JSON от LLM
# ============================================================================
class TestParseToolJson:
    def test_plain_json(self):
        parsed = parse_tool_json('{"tool": "fit_arima", "arguments": {"order": [1,1,1]}}')
        assert parsed == {"tool": "fit_arima", "arguments": {"order": [1, 1, 1]}}

    def test_fenced_json(self):
        text = '```json\n{"tool": "fit_prophet", "arguments": {}}\n```'
        assert parse_tool_json(text) == {"tool": "fit_prophet", "arguments": {}}

    def test_json_inside_text(self):
        text = 'Думаю, лучше всего подойдёт {"tool": "fit_xgboost", "arguments": {"lags": 14}} как вариант'
        assert parse_tool_json(text)["tool"] == "fit_xgboost"

    def test_invalid_returns_none(self):
        assert parse_tool_json("никакого json тут нет") is None
        assert parse_tool_json("{broken json}}") is None
        assert parse_tool_json("") is None

    def test_missing_tool_returns_none(self):
        assert parse_tool_json('{"arguments": {}}') is None

    def test_arguments_default_empty(self):
        assert parse_tool_json('{"tool": "fit_prophet"}') == {
            "tool": "fit_prophet",
            "arguments": {},
        }


# ============================================================================
# LLMManager: попытки и fallback
# ============================================================================
class TestLLMManager:
    async def test_valid_first_attempt(self):
        llm = ScriptedLLM('{"tool": "fit_arima", "arguments": {"order": [1,1,1]}}')
        decision = await LLMManager(llm_call=llm).select_tool("EDA", build_default_registry())
        assert decision.tool == "fit_arima"
        assert decision.arguments == {"order": [1, 1, 1]}
        assert decision.attempts == 1
        assert decision.used_fallback is False

    async def test_recovers_on_second_attempt(self):
        llm = ScriptedLLM("мусор", '{"tool": "fit_prophet", "arguments": {}}')
        decision = await LLMManager(llm_call=llm).select_tool("EDA", build_default_registry())
        assert decision.tool == "fit_prophet"
        assert decision.attempts == 2
        assert decision.used_fallback is False
        assert llm.calls == 2

    async def test_fallback_after_three_failures(self):
        llm = ScriptedLLM("мусор1", "мусор2", "мусор3")
        decision = await LLMManager(llm_call=llm).select_tool("EDA", build_default_registry())
        assert decision.used_fallback is True
        assert decision.tool == "fit_arima"  # fallback-правило: SARIMA с автоподбором
        assert decision.arguments == {}
        assert decision.attempts == 3
        assert llm.calls == 3

    async def test_unknown_tool_counts_as_failure(self):
        llm = ScriptedLLM('{"tool": "fit_magic", "arguments": {}}')
        decision = await LLMManager(llm_call=llm, max_attempts=2).select_tool(
            "EDA", build_default_registry()
        )
        assert decision.used_fallback is True
        assert llm.calls == 2

    async def test_prompt_contains_eda_and_tools(self):
        captured: list[list[dict[str, str]]] = []

        async def spy(messages):
            captured.append(messages)
            return '{"tool": "fit_arima", "arguments": {}}'

        await LLMManager(llm_call=spy).select_tool(
            "seasonal_period=7", build_default_registry(), fewshot=["кейс: SARIMA(1,1,1)"]
        )
        user_text = captured[0][1]["content"]
        assert "seasonal_period=7" in user_text
        assert "fit_arima" in user_text and "fit_xgboost" in user_text
        assert "SARIMA(1,1,1)" in user_text


# ============================================================================
# Классические модели
# ============================================================================
class TestClassicalModels:
    def test_time_split(self):
        s = seasonal_series(100)
        train, val = time_split(s, 14)
        assert len(val) == 14
        assert len(train) == 86
        assert val.index[0] > train.index[-1]

    def test_fit_arima_explicit_order(self):
        result = fit_arima(seasonal_series(220), horizon=14, order=[1, 1, 1])
        assert result["model"] == "arima"
        assert result["order"] == [1, 1, 1]
        assert result["auto"] is False
        for key in ("mae", "rmse", "mape", "wape", "mase"):
            assert key in result["metrics"]

    def test_fit_arima_auto_grid(self):
        result = fit_arima(seasonal_series(220), horizon=14, season=7)
        assert result["auto"] is True
        assert result["order"] is not None
        assert np.isfinite(result["metrics"]["mae"])

    def test_fit_arima_sarima_when_seasonal(self):
        result = fit_arima(
            seasonal_series(220), horizon=14, order=[1, 1, 1], seasonal_order=[1, 1, 1, 7]
        )
        assert result["model"] == "sarima"
        assert result["seasonal_order"] == [1, 1, 1, 7]

    def test_fit_xgboost_lags(self):
        result = fit_xgboost(seasonal_series(220), horizon=14, lags=7, n_estimators=50)
        assert result["model"] == "xgboost"
        assert np.isfinite(result["metrics"]["mae"])
        assert result["lags"] == 7

    def test_make_lag_features_shapes(self):
        X, y = make_lag_features(seasonal_series(100), lags=7)
        assert X.shape == (93, 8)  # 7 лагов + скользящее среднее
        assert y.shape == (93,)

    def test_fit_tabular_xgboost(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 5))
        y = X @ np.array([1.0, -2.0, 3.0, 0.5, 0.0]) + 2.0
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        frame["target"] = y

        result = fit_tabular_xgboost(frame, target_column="target", n_estimators=60)
        assert result["model"] == "xgboost_tabular"
        assert result["metrics"]["mae"] < 1.0  # линейная зависимость легко ловится

    def test_fit_prophet(self):
        pytest.importorskip("prophet", reason="prophet не установлен в этом окружении")
        # Prophet >= 1.2 требует CmdStan; без него (или на Windows с нелатинской
        # локалью cmdstanpy не читает вывод where.exe) бэкенд не загружается
        try:
            make_prophet()
        except ToolExecutionError as exc:
            pytest.skip(f"Stan-бэкенд Prophet недоступен: {exc}")

        result = fit_prophet(seasonal_series(220), horizon=14)
        assert result["model"] == "prophet"
        assert np.isfinite(result["metrics"]["mae"])


# ============================================================================
# Реестр + обработчики: сквозной вызов run()
# ============================================================================
class TestRegistryRun:
    def test_run_fit_arima_with_context_series(self):
        registry = build_default_registry()
        series = seasonal_series(220)
        result = registry.run(
            "fit_arima", {"order": [1, 1, 1]}, context={"series": series, "horizon": 14}
        )
        assert result["model"] == "arima"

    def test_run_fit_arima_from_metadata(self):
        registry = build_default_registry()
        values = seasonal_series(220)
        result = registry.run(
            "fit_arima",
            {"order": [1, 1, 1]},
            context={
                "metadata": {
                    "series_values": values.values.tolist(),
                    "series_dates": list(values.index),
                    "freq": "D",
                },
                "horizon": 14,
            },
        )
        assert result["n_train"] == 206

    def test_run_fit_tabular(self):
        registry = build_default_registry()
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
        frame["y"] = frame["a"] * 3.0
        result = registry.run(
            "fit_tabular_xgboost",
            {"target_column": "y"},
            context={"frame": frame, "target_column": "y"},
        )
        assert result["model"] == "xgboost_tabular"
