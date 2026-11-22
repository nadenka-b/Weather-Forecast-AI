"""LLMManager — выбор инструмента через Tool Calling (Этапы 7 и 11).

Этап 7: протокол диалога с LLM и надёжный парсинг ответа с fallback.
Реальный вызов LLM (LM Studio / HF / внешний API, retry, подсчёт токенов,
таблица llm_calls) подключается на Этапе 11 — до этого используется
заглушка _stub_llm_call, которую тесты подменяют своей функцией.

Правило fallback (задание MLF): максимум 3 неудачные попытки получить
валидный вызов инструмента, затем выбор по умолчанию — SARIMA
с автоподбором порядка (fit_arima без order).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from backend.core.config import get_settings
from backend.core.llm.tools import UnknownToolError, ToolRegistry

logger = logging.getLogger(__name__)

# Ответ LLM должен быть строго таким JSON (однострочно или в блоке кода)
_RESPONSE_FORMAT = (
    '{"tool": "<имя инструмента>", "arguments": {<гиперпараметры>}}'
)

_LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]


@dataclass
class ToolDecision:
    """Результат выбора инструмента: имя, аргументы, статистика попыток."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    used_fallback: bool = False
    raw_response: str | None = None


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
    """Заглушка LLM до Этапа 11: валидный вызов ARIMA(1,1,1).

    Явный order (а не автоподбор) — чтобы прогоны с заглушкой были быстрыми;
    fallback-правило (fit_arima без order) проверяется отдельными тестами.
    """
    return '{"tool": "fit_arima", "arguments": {"order": [1, 1, 1]}}'


class LLMManager:
    """Диалог с LLM: промпт со схемами инструментов -> ToolDecision."""

    def __init__(self, llm_call: _LLMCall | None = None, max_attempts: int | None = None):
        settings = get_settings()
        self.llm_call = llm_call or _stub_llm_call
        self.max_attempts = max_attempts or settings.llm_max_fallback_attempts

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

        for attempt in range(1, self.max_attempts + 1):
            raw = await self.llm_call(messages)
            last_raw = raw
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
        )
