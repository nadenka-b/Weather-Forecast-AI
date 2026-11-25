"""Юнит-тесты агента Critic (Этап 8) и сквозная цепочка агентов."""
import numpy as np

from backend.agents.critic import (
    Critic,
    extract_judgment_text,
    parse_judgment_json,
)
from backend.core.models import ForecastState


# ============================================================================
# Парсинг ответов
# ============================================================================
class TestParseJudgment:
    def test_plain_json(self):
        parsed = parse_judgment_json('{"score": 8, "reasoning": "ok", "suggestions": []}')
        assert parsed["score"] == 8

    def test_fenced_json(self):
        text = '```json\n{"score": 6.5, "suggestions": ["добавить лаги"]}\n```'
        parsed = parse_judgment_json(text)
        assert parsed["score"] == 6.5
        assert parsed["suggestions"] == ["добавить лаги"]

    def test_json_inside_text(self):
        text = 'Оценка: {"score": 3, "reasoning": "плохо"} — вот так.'
        assert parse_judgment_json(text)["score"] == 3

    def test_missing_score_returns_none(self):
        assert parse_judgment_json('{"reasoning": "нет оценки"}') is None
        assert parse_judgment_json("совсем не json") is None
        assert parse_judgment_json("") is None

    def test_text_fallback_score(self):
        judgment = extract_judgment_text("Итоговая оценка.\nSCORE: 7.5\nСпасибо.")
        assert judgment["score"] == 7.5

    def test_text_fallback_with_suggestion(self):
        judgment = extract_judgment_text("SCORE: 4\nSUGGESTION: добавить сезонный порядок")
        assert judgment["score"] == 4.0
        assert "сезонный порядок" in judgment["suggestions"][0]

    def test_text_fallback_no_score(self):
        assert extract_judgment_text("никаких цифр") is None


# ============================================================================
# Агент Critic
# ============================================================================
class JudgeStub:
    def __init__(self, response: str | Exception):
        self.response = response
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def judged_state(**meta_overrides) -> ForecastState:
    state = ForecastState(
        series_id="jena_temp",
        horizon=14,
        freq="D",
        model_type="sarima",
        model_params={"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7]},
        metadata={
            "val_metrics": {"mae": 1.1, "rmse": 1.5, "mase": 0.8},
            "used_fallback": False,
            "validator_errors": [],
            "seasonal_period": 7,
        },
        diagnostics={"seasonal_period": 7, "adf": {"stationary": True}},
    )
    state.metadata.update(meta_overrides)
    return state


class TestCriticAgent:
    async def test_json_judgment_applied(self):
        judge = JudgeStub('{"score": 8, "adequacy": 9, "reasoning": "хороший выбор", "suggestions": ["попробовать Prophet"]}')
        state = await Critic(judge=judge).process(judged_state())

        assert state.critic_score == 8.0
        assert state.metadata["critic_adequacy"] == 9.0
        assert "хороший выбор" in state.critic_reasoning
        assert state.metadata["critic_suggestions"] == ["попробовать Prophet"]
        assert state.metadata["critic_used_fallback"] is False

    async def test_prompt_contains_context(self):
        judge = JudgeStub('{"score": 5}')
        await Critic(judge=judge).process(judged_state())
        prompt = judge.prompts[0]
        assert "sarima" in prompt
        assert "0.8" in prompt  # MASE
        assert "SARIMA(1,1,1)" in prompt  # few-shot кейс
        assert "Validator" in prompt

    async def test_fenced_json_parsed(self):
        judge = JudgeStub('```json\n{"score": 3, "suggestions": ["добавить лаги давления"]}\n```')
        state = await Critic(judge=judge).process(judged_state())
        assert state.critic_score == 3.0
        assert state.metadata["critic_suggestions"] == ["добавить лаги давления"]

    async def test_text_fallback_parsed(self):
        judge = JudgeStub("SCORE: 6.5\nSUGGESTION: проверить WAPE")
        state = await Critic(judge=judge).process(judged_state())
        assert state.critic_score == 6.5
        assert state.metadata["critic_used_fallback"] is False

    async def test_judge_failure_gives_neutral_score(self):
        judge = JudgeStub(RuntimeError("LM Studio недоступна"))
        state = await Critic(judge=judge).process(judged_state())
        assert state.critic_score == 5.0
        assert state.metadata["critic_used_fallback"] is True
        assert any("fallback" in s for s in state.metadata["critic_suggestions"])

    async def test_score_clamped_to_0_10(self):
        judge = JudgeStub('{"score": 42}')
        state = await Critic(judge=judge).process(judged_state())
        assert state.critic_score == 10.0

    async def test_extreme_flags_in_prompt(self):
        judge = JudgeStub('{"score": 5}')
        state = judged_state(extreme_flags={"heat_wave": True})
        await Critic(judge=judge).process(state)
        assert "heat_wave" in judge.prompts[0]


# ============================================================================
# Сквозная цепочка: DataAnalyst -> ModelSelector -> Validator -> Critic
# ============================================================================
class TestFullAgentChain:
    async def test_chain_runs_all_agents(self):
        from backend.agents.data_analyst import DataAnalyst
        from backend.agents.model_selector import ModelSelector
        from backend.agents.validator import Validator
        from backend.core.llm.manager import LLMManager

        rng = np.random.default_rng(13)
        n = 220
        t = np.arange(n)
        values = 12 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.4, n)

        state = ForecastState(
            series_id="chain_weekly",
            horizon=7,
            freq="D",
            metadata={"series_values": values.tolist()},
        )

        selector = ModelSelector(
            llm_manager=LLMManager(
                llm_call=ValidCall('{"tool": "fit_arima", "arguments": {"order": [1, 1, 1]}}')
            )
        )
        judge = JudgeStub('{"score": 9, "adequacy": 9, "reasoning": "ок", "suggestions": []}')

        state = await DataAnalyst().process(state)
        assert state.metadata["climate_bounds"]["global"]

        state = await selector.process(state)
        assert state.model_type == "arima"
        assert state.metadata["val_pred"] and state.metadata["val_dates"]

        state = await Validator().process(state)
        assert isinstance(state.validation_passed, bool)
        assert "validation" in state.diagnostics

        state = await Critic(judge=judge).process(state)
        assert state.critic_score == 9.0
        assert "arima" in judge.prompts[0]


class ValidCall:
    def __init__(self, payload: str):
        self.payload = payload

    async def __call__(self, messages):
        return self.payload
