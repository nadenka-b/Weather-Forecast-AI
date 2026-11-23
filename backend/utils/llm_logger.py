"""Трассировка всех вызовов LLM в таблицу llm_calls (Этап 11, Observability).

Задание MLF: логировать модель, параметры, токены, latency, статус каждого
вызова; хэши промпта/ответа позволяют искать дубликаты без хранения текстов.

request_id берётся из ContextVar: Supervisor выставляет его на время задачи,
поэтому вызовы LLM сшиваются с деревом reasoning_logs того же запроса.
"""
import hashlib
import logging
import time
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

current_request_id: ContextVar[str | None] = ContextVar(
    "llm_request_id", default=None
)
current_agent_name: ContextVar[str] = ContextVar("llm_agent_name", default="unknown")


def hash_text(text: str | None) -> str | None:
    """SHA-256 от текста (хэш промпта/ответа вместо самого текста)."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Приблизительный подсчёт токенов, если API не вернул usage.

    Эвристика ~3.5 символа на токен (латиница с кириллицей): нужна только
    для провайдеров без usage — OpenAI-совместимые API отдают точные числа,
    HF-провайдер считает собственным токенизатором.
    """
    return max(1, round(len(text) / 3.5))


class Timer:
    """Контекстный таймер: latency_ms с момента входа."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.latency_ms = (time.perf_counter() - self.start) * 1000


async def log_llm_call(
    *,
    agent_name: str,
    model_name: str,
    provider: str,
    prompt: str | None,
    response: str | None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    latency_ms: float | None = None,
    status: str = "success",
) -> None:
    """Пишет строку в llm_calls. Сбой логирования не роняет вызов LLM."""
    try:
        from backend.db.models import LLMCall
        from backend.db.session import AsyncSessionLocal

        row = LLMCall(
            request_id=current_request_id.get() or "unattributed",
            agent_name=agent_name or current_agent_name.get(),
            model_name=model_name,
            provider=provider,
            prompt_hash=hash_text(prompt),
            response_hash=hash_text(response),
            tokens_in=tokens_in if tokens_in is not None else (
                estimate_tokens(prompt) if prompt else None
            ),
            tokens_out=tokens_out if tokens_out is not None else (
                estimate_tokens(response) if response else None
            ),
            latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
            status=status,
        )
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — трассировка не должна ломать основной путь
        logger.error("llm_calls logging failed: %s", exc)
