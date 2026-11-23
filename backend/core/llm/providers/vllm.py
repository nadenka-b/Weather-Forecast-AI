"""vLLM — OpenAI-совместимый сервер для продвинутого инференса (Этап 11).

Тот же протокол, что LM Studio, другой адрес (порт 8000) — переключается
через LLM_PROVIDER=vllm без изменения кода агентов.
"""
from backend.core.config import get_settings
from backend.core.llm.providers.openai_compatible import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    name = "vllm"

    def __init__(self, **kwargs):
        settings = get_settings()
        kwargs.setdefault("base_url", settings.vllm_base_url)
        kwargs.setdefault("model", settings.vllm_model)
        super().__init__(**kwargs)
