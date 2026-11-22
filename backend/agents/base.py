"""Базовые классы агентов мультиагентной системы (Этап 5).

- BaseAgent(ABC) — контракт агента: строго типизированный process(state),
  работающий с ForecastState и возвращающий ForecastState.
- LLMAgent(BaseAgent) — агент, которому нужна LLM. Реальный вызов LLM
  (LM Studio / HF / API через LLMManager) подключается на Этапе 11;
  до этого наследники используют заглушки с логированием.

State передаётся между агентами только по ссылке (ID задачи) — тяжёлые
объекты (DataFrame) в payload запрещены (задание MLF).
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

from backend.core.models import ForecastState

logger = logging.getLogger(__name__)

# Фиксированный seed для воспроизводимости LLM-ответов агентов
DEFAULT_LLM_SEED = 42


class BaseAgent(ABC):
    """Абстрактный агент пайплайна прогнозирования.

    Контракт, через который Supervisor гоняет все этапы:
    ``process(state)`` и ``get_last_result()``.
    """

    def __init__(self, name: str | None = None, description: str = ""):
        self.name = name or self.__class__.__name__
        self.description = description
        self._last_result: Any = None

    @abstractmethod
    async def process(self, state: ForecastState) -> ForecastState:
        """Обрабатывает состояние и возвращает обновлённое состояние."""

    def get_last_result(self) -> Any:
        """Результат последнего вызова process (для статуса и тестов)."""
        return self._last_result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class LLMAgent(BaseAgent):
    """Агент с доступом к LLM.

    temperature=0.0 по умолчанию: Critic и ModelSelector должны быть
    детерминированными; Editor переопределит на 0.3.
    Вызов LLM (_call_llm) появится на Этапе 11 — LLMManager с роутингом
    lm_studio | hf_local | api, retry и подсчётом токенов.
    """

    temperature: float = 0.0
    max_tokens: int = 512

    def __init__(
        self,
        name: str | None = None,
        description: str = "",
        model_name: str | None = None,
    ):
        super().__init__(name=name, description=description)
        self.model_name = model_name
