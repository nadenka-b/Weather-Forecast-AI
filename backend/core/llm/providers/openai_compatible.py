"""OpenAI-совместимый провайдер LLM (Этап 11) — общая база для LM Studio, vLLM и внешних API.

Все три говорят на одном протоколе: POST {base_url}/chat/completions с
messages и (опционально) tools (function calling). От ветки провайдера
отличаются только адресом, моделью и заголовком авторизации.

Контракт ответа: tool_calls нормализуются в текст
    {"tool": "<name>", "arguments": {...}}
— тот же формат, который парсит LLMManager.parse_tool_json.

Каждый вызов трассируется в llm_calls (модель, токены, latency, статус).
Retry — только на сетевые сбои и 5xx/429 (методичка Step_8): 4xx не ретраим.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.core.config import get_settings
from backend.utils.llm_logger import current_agent_name, log_llm_call

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Провайдер не смог ответить (после всех повторов)."""


@dataclass
class LLMResponse:
    """Нормализованный ответ провайдера."""

    text: str
    model: str
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def normalize_message(message: dict[str, Any]) -> str:
    """tool_calls -> JSON {"tool", "arguments"}; иначе content как есть."""
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        function = call.get("function", {})
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
        except (json.JSONDecodeError, TypeError):
            arguments = {"_raw": raw_arguments}
        return json.dumps(
            {"tool": function.get("name"), "arguments": arguments}, ensure_ascii=False
        )
    return str(message.get("content") or "")


class OpenAICompatibleProvider:
    """POST {base_url}/chat/completions с retry и трассировкой."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        agent_name: str | None = None,
        client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or None
        self.agent_name = agent_name
        self._client = client
        settings = get_settings()
        self.max_retries = max_retries or settings.llm_max_retries
        self.retry_delay = retry_delay if retry_delay is not None else settings.llm_retry_delay

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        from backend.utils.http_client import get_http_client

        return get_http_client(timeout=get_settings().llm_timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        seed: int | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed  # воспроизводимость (DEFAULT_LLM_SEED=42)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _parse_usage(data: dict[str, Any]) -> tuple[int | None, int | None]:
        usage = data.get("usage") or {}
        return usage.get("prompt_tokens"), usage.get("completion_tokens")

    async def check_health(self) -> tuple[bool, str | None]:
        """Доступность сервера: GET {base_url}/models с коротким таймаутом."""
        try:
            response = await self._get_client().get(
                f"{self.base_url}/models", headers=self._headers(), timeout=3.0
            )
            response.raise_for_status()
            return True, None
        except Exception as exc:  # noqa: BLE001 — любой сбой = нездоров
            return False, str(exc)[:200]

    # ------------------------------------------------------------------
    # Основной вызов
    # ------------------------------------------------------------------
    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST chat/completions; любой не-2xx -> HTTPStatusError (решение о
        повторе принимает complete: 5xx/429 ретраим, остальные 4xx — нет)."""
        response = await self._get_client().post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Один вызов chat/completions с retry; трассируется в llm_calls."""
        payload = self._build_payload(messages, tools, temperature, seed, max_tokens)
        prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
        start = time.perf_counter()

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                data = await self._request(payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                pause = self.retry_delay * attempt
                logger.warning(
                    "[%s] %s: попытка %d/%d не удалась (%s) — повтор через %.1f с",
                    self.agent_name, self.name, attempt, self.max_retries,
                    exc.__class__.__name__, pause,
                )
                await asyncio.sleep(pause)
            except httpx.HTTPStatusError as exc:
                # 5xx/429 ретраим, остальные 4xx — сразу ошибка
                status = exc.response.status_code
                if status >= 500 or status == 429:
                    last_error = exc
                    if attempt >= self.max_retries:
                        break
                    pause = self.retry_delay * attempt
                    logger.warning(
                        "[%s] %s: HTTP %d, попытка %d/%d — повтор через %.1f с",
                        self.agent_name, self.name, status, attempt,
                        self.max_retries, pause,
                    )
                    await asyncio.sleep(pause)
                else:
                    last_error = exc
                    break
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                message = (data.get("choices") or [{}])[0].get("message") or {}
                text = normalize_message(message)
                tokens_in, tokens_out = self._parse_usage(data)
                await log_llm_call(
                    agent_name=self.agent_name or current_agent_name.get(),
                    model_name=data.get("model") or self.model,
                    provider=self.name,
                    prompt=prompt_text,
                    response=text,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_ms=latency_ms,
                    status="success",
                )
                return LLMResponse(
                    text=text,
                    model=data.get("model") or self.model,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    finish_reason=(data.get("choices") or [{}])[0].get("finish_reason"),
                    raw=data,
                )

        latency_ms = (time.perf_counter() - start) * 1000
        await log_llm_call(
            agent_name=self.agent_name or current_agent_name.get(),
            model_name=self.model,
            provider=self.name,
            prompt=prompt_text,
            response=None,
            latency_ms=latency_ms,
            status="error",
        )
        raise LLMProviderError(
            f"{self.name}: не удалось получить ответ за {self.max_retries} "
            f"попыток ({last_error})"
        ) from last_error
