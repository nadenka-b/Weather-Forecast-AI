"""LLMManager — роутинг провайдеров и Tool Calling (Этапы 7 и 11).

Роутинг по настройке LLM_PROVIDER (методичка Step_8):
    lm_studio  — локальный OpenAI-совместимый сервер (основной режим);
    vllm       — OpenAI-совместимый инференс-сервер;
    hf_local   — transformers (Causal LM), запасной локальный режим;
    api        — внешний API (OpenRouter и др.);
    stub       — заглушка без сети (тесты, офлайн-разработка).

Tool Calling: в запрос передаются JSON-схемы инструментов реестра; ответ
(tool_calls или JSON в тексте) нормализуется и парсится. Fallback по заданию
MLF: максимум 3 неудачные попытки (в т.ч. недоступный провайдер), затем
правило по умолчанию — SARIMA с автоподбором порядка.

Каждый вызов провайдера трассируется в llm_calls (токены, latency, статус);
суммарные токены возвращаются в ToolDecision для агентных метрик.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Awaitable, Callable

from backend.core.config import get_settings
from backend.core.llm.tools import UnknownToolError, ToolRegistry

logger = logging.getLogger(__name__)

# Фиксированный seed для воспроизводимости ответов LLM
DEFAULT_LLM_SEED = 42

# Ответ LLM должен быть строго таким JSON (однострочно или в блоке кода)
_RESPONSE_FORMAT = (
    '{"tool": "<имя инструмента>", "arguments": {<гиперпараметры>}}'
)

_LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]


@dataclass
class ToolDecision:
    """Результат выбора инструмента: имя, аргументы, статистика попыток и LLM."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    used_fallback: bool = False
    raw_response: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    model: str | None = None


def parse_tool_json(text: str) -> dict[str, Any] | None:
    """Достаёт {tool, arguments} из ответа LLM.

    Понимает: чистый JSON, JSON в ```json ...```, JSON внутри прочего текста.
    Возвращает None, если ответ не парсится или не содержит нужных ключей.
    """
    if not text:
        return None

    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None
    tool = parsed.get("tool")
    arguments = parsed.get("arguments", {})
    if not isinstance(tool, str) or not tool:
        return None
    if not isinstance(arguments, dict):
        return None
    return {"tool": tool, "arguments": arguments}


async def _stub_llm_call(messages: list[dict[str, str]]) -> str:
    """Заглушка (LLM_PROVIDER=stub): валидный вызов ARIMA(1,1,1).

    Явный order (а не автоподбор) — чтобы прогоны были быстрыми;
    fallback-правило (fit_arima без order) проверяется отдельными тестами.
    """
    return '{"tool": "fit_arima", "arguments": {"order": [1, 1, 1]}}'


def get_provider(provider: str | None = None, agent_name: str | None = None):
    """Роутинг по LLM_PROVIDER. None — заглушка (stub или неизвестное имя)."""
    name = (provider or get_settings().llm_provider).lower()
    if name == "lm_studio":
        from backend.core.llm.providers.lm_studio import LMStudioProvider

        return LMStudioProvider(agent_name=agent_name)
    if name == "vllm":
        from backend.core.llm.providers.vllm import VLLMProvider

        return VLLMProvider(agent_name=agent_name)
    if name == "api":
        from backend.core.llm.providers.api import ExternalAPIProvider

        return ExternalAPIProvider(agent_name=agent_name)
    if name == "hf_local":
        from backend.core.llm.providers.hf_local import HFLocalProvider

        return HFLocalProvider(agent_name=agent_name)
    if name != "stub":
        logger.warning("Неизвестный LLM_PROVIDER=%r — используется заглушка", name)
    return None


class LLMManager:
    """Диалог с LLM: промпт со схемами инструментов -> ToolDecision."""

    def __init__(
        self,
        llm_call: _LLMCall | None = None,
        max_attempts: int | None = None,
        provider: Any | None = None,
    ):
        settings = get_settings()
        self.max_attempts = max_attempts or settings.llm_max_fallback_attempts
        # расход последнего вызова (токены/latency) — для агентных метрик
        self.last_usage: dict[str, Any] = {"tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0}
        # llm_call — внедрённая заглушка (тесты); иначе провайдер: явный
        # объект или роутинг по настройкам
        self._injected = llm_call
        self._provider = (
            provider
            if provider is not None
            else (None if llm_call is not None else get_provider())
        )

    @property
    def uses_stub(self) -> bool:
        return self._injected is not None or self._provider is None

    # ------------------------------------------------------------------
    # Вызов LLM
    # ------------------------------------------------------------------
    async def _provider_call(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = DEFAULT_LLM_SEED,
        max_tokens: int | None = None,
    ):
        """Вызов провайдера с кэшированием (CACHE_ENABLED, таблица cache).

        При попадании LLM не вызывается: токены 0, latency 0, finish_reason='cache'.
        Ключ — хэш модели, сообщений, схем инструментов и параметров генерации.
        """
        assert self._provider is not None
        cache_key = None
        if get_settings().cache_enabled:
            from backend.data.cache import cache_manager

            cache_key = cache_manager.make_key(
                "llm", self._provider.name, getattr(self._provider, "model", None),
                messages, tools, temperature, seed, max_tokens,
            )
            cached = await cache_manager.get(cache_key, model_used="llm")
            if cached is not None:
                try:
                    payload = json.loads(cached)
                except (json.JSONDecodeError, TypeError):
                    payload = None
                if isinstance(payload, dict) and "text" in payload:
                    from backend.core.llm.providers.openai_compatible import LLMResponse

                    logger.debug("LLM cache hit for %s", self._provider.name)
                    return LLMResponse(
                        text=payload["text"], model=payload.get("model") or getattr(self._provider, "model", "cache"),
                        latency_ms=0.0, finish_reason="cache",
                    )

        response = await self._provider.complete(
            messages, tools=tools, temperature=temperature, seed=seed, max_tokens=max_tokens
        )
        if cache_key is not None:
            from backend.data.cache import cache_manager

            await cache_manager.set(
                cache_key,
                json.dumps({"text": response.text, "model": response.model}, ensure_ascii=False),
                model_used="llm",
            )
        return response

    async def _invoke(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ):
        """Возвращает (текст, ответ провайдера | None для заглушки)."""
        if self._injected is not None:
            return await self._injected(messages), None
        if self._provider is None:  # режим stub / неизвестный провайдер
            return await _stub_llm_call(messages), None
        response = await self._provider_call(messages, tools=tools)
        return response.text, response

    def _build_messages(
        self,
        eda_summary: str,
        registry: ToolRegistry,
        fewshot: list[str] | None = None,
    ) -> list[dict[str, str]]:
        schemas = "\n".join(
            f"- {schema['function']['name']}: {schema['function']['description']}"
            for schema in registry.schemas()
        )
        examples = "\n".join(f"- {example}" for example in (fewshot or []))
        system = (
            "Ты — агент выбора модели прогнозирования временных рядов. "
            "Ответь СТРОГО одним JSON без пояснений: "
            + _RESPONSE_FORMAT
        )
        user = (
            f"Данные (EDA):\n{eda_summary}\n\n"
            f"Доступные инструменты:\n{schemas}\n"
            + (f"\nПохожие успешные кейсы:\n{examples}\n" if examples else "")
            + "\nВыбери инструмент и гиперпараметры. Если не уверен в порядке "
            "ARIMA — не задавай order, сработает автоподбор."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def select_tool(
        self,
        eda_summary: str,
        registry: ToolRegistry,
        fewshot: list[str] | None = None,
    ) -> ToolDecision:
        """До max_attempts попыток получить валидный вызов; затем fallback."""
        messages = self._build_messages(eda_summary, registry, fewshot)
        last_raw: str | None = None
        tokens_in = tokens_out = 0
        latency_ms = 0.0
        model: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw, response = await self._invoke(messages, tools=registry.schemas())
            except Exception as exc:  # noqa: BLE001 — недоступный провайдер = неудачная попытка
                logger.warning(
                    "Попытка %d: провайдер LLM не ответил (%s: %s)",
                    attempt, type(exc).__name__, exc,
                )
                raw, response = "", None

            last_raw = raw
            if response is not None:
                tokens_in += response.tokens_in or 0
                tokens_out += response.tokens_out or 0
                latency_ms += response.latency_ms
                model = response.model

            parsed = parse_tool_json(raw)
            if parsed is not None:
                try:
                    registry.get(parsed["tool"])  # валидация имени инструмента
                    logger.info(
                        "Tool calling ok: %s (попытка %d)", parsed["tool"], attempt
                    )
                    return ToolDecision(
                        tool=parsed["tool"],
                        arguments=parsed["arguments"],
                        attempts=attempt,
                        used_fallback=False,
                        raw_response=raw,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        latency_ms=round(latency_ms, 2),
                        model=model,
                    )
                except UnknownToolError:
                    logger.warning("Попытка %d: неизвестный инструмент %r", attempt, parsed["tool"])
            else:
                logger.warning("Попытка %d: ответ LLM не парсится как JSON", attempt)

        # Fallback-правило задания MLF: SARIMA с автоподбором порядка
        logger.warning(
            "Tool Calling не удался за %d попыток — применяется fallback fit_arima(auto)",
            self.max_attempts,
        )
        return ToolDecision(
            tool="fit_arima",
            arguments={},
            attempts=self.max_attempts,
            used_fallback=True,
            raw_response=last_raw,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=round(latency_ms, 2),
            model=model,
        )

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Простое текстовое дополнение без tool-протокола (для Critic)."""
        messages: list[dict[str, str]] = (
            [{"role": "system", "content": system}] if system else []
        )
        messages.append({"role": "user", "content": prompt})
        if self._injected is not None:
            return await self._injected(messages)
        if self._provider is None:  # режим stub / неизвестный провайдер
            return await _stub_llm_call(messages)
        response = await self._provider_call(messages, max_tokens=max_tokens)
        self.last_usage = {
            "tokens_in": response.tokens_in or 0,
            "tokens_out": response.tokens_out or 0,
            "latency_ms": response.latency_ms,
        }
        return response.text


@lru_cache
def get_llm_manager() -> LLMManager:
    """Один менеджер на процесс (роутинг по настройкам)."""
    return LLMManager()
