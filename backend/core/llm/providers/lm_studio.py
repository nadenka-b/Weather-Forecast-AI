"""LM Studio — локальный OpenAI-совместимый сервер (Этап 11).

Приложение LM Studio -> Developer -> Start Server (порт 1234). Та же модель
Qwen2.5-7B-Instruct, что и в референс-проекте; supports tool calling.
"""
from backend.core.config import get_settings
from backend.core.llm.providers.openai_compatible import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    name = "lm_studio"

    def __init__(self, **kwargs):
        settings = get_settings()
        kwargs.setdefault("base_url", settings.lm_studio_base_url)
        kwargs.setdefault("model", settings.lm_studio_model)
        super().__init__(**kwargs)
