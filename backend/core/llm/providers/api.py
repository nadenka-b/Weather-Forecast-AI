"""Внешние API (OpenRouter, OpenAI, Groq) через OpenAI-совместимый интерфейс (Этап 11).

Бонус по заданию: унифицированный интерфейс, retry и fallback общие с
локальными провайдерами. Ключ — LLM_API_KEY (не коммитится).
"""
from backend.core.config import get_settings
from backend.core.llm.providers.openai_compatible import OpenAICompatibleProvider


class ExternalAPIProvider(OpenAICompatibleProvider):
    name = "api"

    def __init__(self, **kwargs):
        settings = get_settings()
        kwargs.setdefault("base_url", settings.llm_api_base_url)
        kwargs.setdefault("model", settings.llm_api_model)
        kwargs.setdefault("api_key", settings.llm_api_key or None)
        super().__init__(**kwargs)
