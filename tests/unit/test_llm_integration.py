"""Тесты интеграции LLM (Этап 11): провайдеры через httpx.MockTransport,
retry, трассировка llm_calls, роутинг, Critic на реальной LLM."""
import asyncio
import copy
import json

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.llm.manager import LLMManager, get_llm_manager, get_provider
from backend.core.llm.providers.openai_compatible import (
    LLMProviderError,
    LLMResponse,
    OpenAICompatibleProvider,
    normalize_message,
)
from backend.core.llm.tools import build_default_registry
from backend.db.models import Base, LLMCall
from backend.utils.async_utils import retry
from backend.utils.llm_logger import (
    current_agent_name,
    current_request_id,
    estimate_tokens,
    hash_text,
)

TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-test",
    "model": "qwen2.5-7b-instruct",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "fit_arima",
                            "arguments": '{"order": [1, 1, 1]}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 120, "completion_tokens": 18},
}

TEXT_RESPONSE = {
    "model": "qwen2.5-7b-instruct",
    "choices": [
        {"message": {"role": "assistant", "content": ' {"score": 8, "reasoning": "ok"} '},
         "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 90, "completion_tokens": 12},
}


def make_provider(handler, **kwargs) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("retry_delay", 0)
    kwargs.setdefault("max_retries", 3)
    return OpenAICompatibleProvider(
        base_url="http://llm-test/v1", model="test-model", client=client, **kwargs
    )


# ============================================================================
# Утилиты
# ============================================================================
class TestUtils:
    def test_hash_text_stable_and_none(self):
        assert hash_text("abc") == hash_text("abc")
        assert len(hash_text("abc")) == 64
        assert hash_text(None) is None

    def test_estimate_tokens_positive(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("a" * 35) == 10

    async def test_retry_retries_then_succeeds(self):
        calls = {"n": 0}

        @retry(max_attempts=3, delay=0, exceptions=(ValueError,))
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("temporary")
            return "ok"

        assert await flaky() == "ok"
        assert calls["n"] == 3

    async def test_retry_exhausts_and_reraises(self):
        @retry(max_attempts=2, delay=0, exceptions=(ValueError,))
        async def broken():
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await broken()

    async def test_retry_ignores_other_exceptions(self):
        calls = {"n": 0}

        @retry(max_attempts=3, delay=0, exceptions=(ValueError,))
        async def wrong_type():
            calls["n"] += 1
            raise KeyError("bug")

        with pytest.raises(KeyError):
            await wrong_type()
        assert calls["n"] == 1  # ошибки логики не ретраятся

    async def test_http_client_per_loop(self):
        from backend.utils.http_client import close_http_clients, get_http_client

        first = get_http_client()
        assert get_http_client() is first  # один клиент внутри loop

        def client_in_fresh_loop():
            return asyncio.run(get_async_client())

        async def get_async_client():
            return get_http_client()

        # другой loop (в отдельном потоке) — другой клиент
        other = await asyncio.to_thread(client_in_fresh_loop)
        assert other is not first
        await close_http_clients()


# ============================================================================
# OpenAI-совместимый провайдер
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestOpenAICompatibleProvider:
    def test_normalize_tool_calls(self):
        message = TOOL_CALL_RESPONSE["choices"][0]["message"]
        parsed = json.loads(normalize_message(message))
        assert parsed == {"tool": "fit_arima", "arguments": {"order": [1, 1, 1]}}

    def test_normalize_plain_content(self):
        assert normalize_message({"content": "просто текст"}) == "просто текст"

    def test_normalize_broken_arguments(self):
        message = {"tool_calls": [{"function": {"name": "t", "arguments": "{bad"}}]}
        parsed = json.loads(normalize_message(message))
        assert parsed["tool"] == "t"
        assert "_raw" in parsed["arguments"]

    async def test_tool_call_success_with_usage(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=TOOL_CALL_RESPONSE)

        provider = make_provider(handler)
        response = await provider.complete(
            [{"role": "user", "content": "выбери"}],
            tools=build_default_registry().schemas(),
        )
        assert isinstance(response, LLMResponse)
        assert json.loads(response.text)["tool"] == "fit_arima"
        assert response.tokens_in == 120 and response.tokens_out == 18
        assert response.model == "qwen2.5-7b-instruct"
        assert response.finish_reason == "tool_calls"

        body = json.loads(seen[0].content)
        assert body["tools"][0]["type"] == "function"  # схемы ушли в запрос
        assert body["tool_choice"] == "auto"
        assert body["seed"] == 42  # воспроизводимость
        assert body["temperature"] == 0.0

    async def test_text_response(self):
        provider = make_provider(lambda req: httpx.Response(200, json=TEXT_RESPONSE))
        response = await provider.complete([{"role": "user", "content": "оцени"}])
        assert response.text.strip() == '{"score": 8, "reasoning": "ok"}'
        assert response.tokens_in == 90

    async def test_retries_on_500_then_succeeds(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json=TEXT_RESPONSE)

        response = await make_provider(handler).complete(
            [{"role": "user", "content": "x"}]
        )
        assert calls["n"] == 2
        assert response.text.strip()

    async def test_no_retry_on_400(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad request"})

        with pytest.raises(LLMProviderError):
            await make_provider(handler).complete([{"role": "user", "content": "x"}])
        assert calls["n"] == 1  # ошибка запроса — без повторов

    async def test_connect_error_exhausts_retries(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            raise httpx.ConnectError("connection refused", request=request)

        provider = make_provider(handler, max_retries=2)
        with pytest.raises(LLMProviderError, match="2 попыток"):
            await provider.complete([{"role": "user", "content": "x"}])
        assert calls["n"] == 2

    async def test_api_key_sent_as_bearer(self):
        seen: list[httpx.Request] = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=TEXT_RESPONSE)

        provider = make_provider(handler, api_key="secret-key")
        await provider.complete([{"role": "user", "content": "x"}])
        assert seen[0].headers["Authorization"] == "Bearer secret-key"

    async def test_check_health_ok_and_fail(self):
        ok_provider = make_provider(lambda req: httpx.Response(200, json={"data": []}))
        assert await ok_provider.check_health() == (True, None)

        def down(request):
            raise httpx.ConnectError("refused", request=request)

        healthy, error = await make_provider(down).check_health()
        assert healthy is False and error


# ============================================================================
# Трассировка llm_calls
# ============================================================================
@pytest.fixture
async def memory_db(monkeypatch):
    import backend.db.session as db_session_module

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_local = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_session_module, "AsyncSessionLocal", session_local)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield session_local
    await engine.dispose()


@pytest.mark.usefixtures("memory_db")
class TestLLMCallLogging:
    async def test_success_call_logged_with_context(self, memory_db):
        request_token = current_request_id.set("req-llm-1")
        agent_token = current_agent_name.set("model_selector")
        try:
            provider = make_provider(lambda req: httpx.Response(200, json=TOOL_CALL_RESPONSE))
            await provider.complete([{"role": "user", "content": "промпт"}])
        finally:
            current_request_id.reset(request_token)
            current_agent_name.reset(agent_token)

        async with memory_db() as session:
            rows = (await session.execute(select(LLMCall))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.request_id == "req-llm-1"       # сшито с задачей
        assert row.agent_name == "model_selector"   # агент из contextvar
        assert row.provider == "openai_compatible"
        assert row.tokens_in == 120 and row.tokens_out == 18
        assert row.status == "success"
        assert row.prompt_hash == hash_text("промпт")
        assert row.latency_ms is not None

    async def test_error_call_logged(self, memory_db):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        provider = make_provider(handler, max_retries=1)
        with pytest.raises(LLMProviderError):
            await provider.complete([{"role": "user", "content": "x"}])

        async with memory_db() as session:
            rows = (await session.execute(select(LLMCall))).scalars().all()
        assert len(rows) == 1
        assert rows[0].status == "error"
        assert rows[0].request_id == "unattributed"
        assert rows[0].tokens_in == estimate_tokens("x")  # usage нет — оценка


# ============================================================================
# Роутинг и LLMManager
# ============================================================================
class TestRouting:
    def test_stub_mode_by_default_in_tests(self):
        assert get_provider("stub") is None
        assert LLMManager().uses_stub is True

    def test_routing_by_name(self):
        from backend.core.llm.providers.api import ExternalAPIProvider
        from backend.core.llm.providers.hf_local import HFLocalProvider
        from backend.core.llm.providers.lm_studio import LMStudioProvider
        from backend.core.llm.providers.vllm import VLLMProvider

        assert isinstance(get_provider("lm_studio"), LMStudioProvider)
        assert isinstance(get_provider("vllm"), VLLMProvider)
        assert isinstance(get_provider("api"), ExternalAPIProvider)
        assert isinstance(get_provider("hf_local"), HFLocalProvider)
        assert get_provider("no-such-provider") is None

    def test_lm_studio_defaults_from_settings(self):
        from backend.core.llm.providers.lm_studio import LMStudioProvider

        provider = LMStudioProvider()
        assert provider.base_url.endswith("/v1")
        assert provider.model
        assert provider.name == "lm_studio"


@pytest.mark.usefixtures("memory_db")
class TestManagerWithProvider:
    async def test_tool_call_via_provider(self):
        provider = make_provider(lambda req: httpx.Response(200, json=TOOL_CALL_RESPONSE))
        decision = await LLMManager(provider=provider).select_tool(
            "EDA: seasonal_period=7", build_default_registry()
        )
        assert decision.tool == "fit_arima"
        assert decision.arguments == {"order": [1, 1, 1]}
        assert decision.used_fallback is False
        assert decision.tokens_in == 120 and decision.tokens_out == 18
        assert decision.model == "qwen2.5-7b-instruct"

    async def test_text_json_via_provider(self):
        payload = copy.deepcopy(TEXT_RESPONSE)
        payload["choices"][0]["message"] = {
            "role": "assistant",
            "content": '{"tool": "fit_prophet", "arguments": {"seasonality_mode": "additive"}}',
        }
        provider = make_provider(lambda req: httpx.Response(200, json=payload))
        decision = await LLMManager(provider=provider).select_tool("EDA", build_default_registry())
        assert decision.tool == "fit_prophet"
        assert decision.arguments["seasonality_mode"] == "additive"

    async def test_provider_down_falls_back_after_three_attempts(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            raise httpx.ConnectError("LM Studio не запущена", request=request)

        provider = make_provider(handler, max_retries=1)
        decision = await LLMManager(provider=provider).select_tool(
            "EDA", build_default_registry()
        )
        assert decision.used_fallback is True
        assert decision.tool == "fit_arima" and decision.arguments == {}
        assert decision.attempts == 3
        assert calls["n"] == 3  # по одному вызову провайдера на попытку

    async def test_complete_via_provider(self):
        provider = make_provider(lambda req: httpx.Response(200, json=TEXT_RESPONSE))
        text = await LLMManager(provider=provider).complete("оцени", system="ты судья")
        assert json.loads(text)["score"] == 8

    def test_get_llm_manager_singleton(self):
        get_llm_manager.cache_clear()
        assert get_llm_manager() is get_llm_manager()
        get_llm_manager.cache_clear()


# ============================================================================
# HF-провайдер (фейковый pipeline, без transformers)
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestHFLocalProvider:
    async def test_complete_with_fake_pipeline(self):
        from backend.core.llm.providers import hf_local

        class FakeTokenizer:
            chat_template = None

            def encode(self, text):
                return text.split()

        class FakePipeline:
            def __call__(self, prompt, **kwargs):
                assert kwargs["return_full_text"] is False
                return [{"generated_text": '{"score": 7}'}]

        hf_local._loaded.clear()
        provider = hf_local.HFLocalProvider(model="fake/model")
        provider._pipeline, provider._tokenizer = FakePipeline(), FakeTokenizer()
        hf_local._loaded[("fake/model", "cpu")] = (provider._pipeline, provider._tokenizer)

        response = await provider.complete(
            [{"role": "user", "content": "user: оцени"}, {"role": "system", "content": "x"}]
        )
        assert response.text == '{"score": 7}'
        assert response.tokens_in > 0 and response.tokens_out > 0
        hf_local.unload_hf_models()


# ============================================================================
# Critic через реальный LLMManager (провайдер подменён моком)
# ============================================================================
@pytest.mark.usefixtures("memory_db")
class TestCriticWithRealLLM:
    async def test_judge_via_llm_manager(self, monkeypatch):
        import backend.core.llm.manager as manager_module
        from backend.agents.critic import Critic, _judge_via_llm
        from backend.core.models import ForecastState

        judgment = copy.deepcopy(TEXT_RESPONSE)
        judgment["choices"][0]["message"] = {
            "role": "assistant",
            "content": '{"score": 6, "adequacy": 7, "reasoning": "сезонность учтена", "suggestions": ["проверить MASE"]}',
        }
        provider = make_provider(lambda req: httpx.Response(200, json=judgment))
        manager = LLMManager(provider=provider)
        monkeypatch.setattr(manager_module, "get_llm_manager", lambda: manager)

        state = ForecastState(
            series_id="s", horizon=7, model_type="sarima",
            metadata={"val_metrics": {"mae": 1.0}},
        )
        state = await Critic(judge=_judge_via_llm).process(state)
        assert state.critic_score == 6.0
        assert state.metadata["critic_adequacy"] == 7.0
        assert state.metadata["critic_used_fallback"] is False

    async def test_stub_mode_default_judge(self):
        from backend.agents.critic import Critic
        from backend.core.models import ForecastState

        state = await Critic().process(ForecastState(series_id="s", horizon=7, metadata={}))
        assert state.critic_score == 5.0
        assert state.metadata["critic_used_fallback"] is False  # заглушка отработала штатно
