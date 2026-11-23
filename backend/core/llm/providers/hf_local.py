"""HuggingFace Transformers — запасной локальный режим (Этап 11).

По заданию MLF: только Causal LM (генеративные), Encoder-only/Seq2Seq
запрещены. Асинхронная обёртка: тяжёлый `pipeline` грузится один раз
(синглтон на процесс) и вызывается через asyncio.to_thread, чтобы не
блокировать event loop. Токены считаются токенизатором модели.
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from typing import Any

from backend.core.config import get_settings
from backend.core.llm.providers.openai_compatible import LLMResponse
from backend.utils.llm_logger import current_agent_name, log_llm_call

logger = logging.getLogger(__name__)

# Синглтон пайплайна: (model_name, device) -> (pipeline, tokenizer)
_loaded: dict[tuple[str, str], tuple[Any, Any]] = {}


class HFLocalProvider:
    """Causal LM через transformers.pipeline (текст без tool-протокола:
    JSON запрашивается в самом промпте и парсится менеджером)."""

    name = "hf_local"

    def __init__(self, agent_name: str | None = None, model: str | None = None, device: str = "cpu"):
        self.model = model or get_settings().hf_local_model
        self.device = device
        self.agent_name = agent_name
        self._pipeline = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Ленивая загрузка (синглтон на процесс)
    # ------------------------------------------------------------------
    def _load(self) -> None:
        key = (self.model, self.device)
        if key in _loaded:
            self._pipeline, self._tokenizer = _loaded[key]
            return
        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError("Пропущен transformers: poetry install transformers") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=tokenizer,
            device=self.device,
        )
        _loaded[key] = (pipe, tokenizer)
        self._pipeline, self._tokenizer = pipe, tokenizer
        logger.info("HF pipeline loaded: %s on %s", self.model, self.device)

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Синхронная генерация (вызывается в потоке через to_thread)."""
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )
        return outputs[0]["generated_text"] if outputs else ""

    async def check_health(self) -> tuple[bool, str | None]:
        try:
            await asyncio.to_thread(self._load)
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]

    # ------------------------------------------------------------------
    # Основной вызов
    # ------------------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        await asyncio.to_thread(self._load)

        # chat-шаблон модели; если его нет — конкатенация ролей
        if self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        start = time.perf_counter()
        try:
            text = await asyncio.to_thread(
                partial(self._generate, prompt, max_tokens or 512, temperature)
            )
        except Exception as exc:  # noqa: BLE001
            await log_llm_call(
                agent_name=self.agent_name or current_agent_name.get(), model_name=self.model,
                provider=self.name, prompt=prompt, response=None,
                latency_ms=(time.perf_counter() - start) * 1000, status="error",
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        tokens_in = len(self._tokenizer.encode(prompt))
        tokens_out = len(self._tokenizer.encode(text))
        await log_llm_call(
            agent_name=self.agent_name or current_agent_name.get(), model_name=self.model, provider=self.name,
            prompt=prompt, response=text, tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        return LLMResponse(
            text=text, model=self.model, latency_ms=latency_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )


def unload_hf_models() -> None:
    """Освобождает загруженные модели (shutdown, фикстуры тестов)."""
    _loaded.clear()
