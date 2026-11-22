"""Critic — LLM-as-a-Judge (Этап 8).

Оценивает адекватность выбора модели и качества прогноза от 0 до 10:
- строит промпт: выбранная модель + гиперпараметры, метрики валидации,
  EDA (сезонность/стационарность), ошибки Validator'а, fallback-флаг,
  few-shot примеры метеокейсов (пока статические; RAG — Этап 10);
- вызывает judge-функцию (реальная LLM подключается на Этапе 11 через
  LM Studio; сейчас заглушка с логированием — как требует план Этапа 8);
- парсит JSON-ответ (в т.ч. из блока кода или внутри текста), с текстовым
  fallback «SCORE: N»; при полном сбое — нейтральная оценка 5.0.

Раздел «исторические аномалии» (волны жары/заморозки — тема проекта):
в промпт попадают экстремальные остатки и metadata['extreme_flags'];
реальные кейсы аномалий наполнятся базой знаний (Этапы 9–10).
"""
import json
import logging
import re
from typing import Any, Awaitable, Callable

from backend.agents.base import LLMAgent
from backend.core.models import ForecastState

logger = logging.getLogger(__name__)

JudgeCall = Callable[[str], Awaitable[str]]

# Few-shot примеры из метеодомена (заменит гибридный RAG на Этапе 10)
_FEWSHOT_CASES = [
    "Кейс: дневная температура с годовой сезонностью без тренда, SARIMA(1,1,1)(1,1,1,7) "
    "дала MASE 0.7 — выбор адекватен, score 9.",
    "Кейс: осадки — сильно скошенный ряд с нулями; MAPE у XGBoost раздут из-за "
    "деления на близкие к нулю значения — смотреть WAPE/MASE, score 6, "
    "предложение: переходить на WAPE-метрики.",
    "Кейс: выбрана ARIMA без сезонного порядка при явной недельной сезонности "
    "(seasonal_period=7) — Ljung-Box завален, score 3, "
    "предложение: добавить seasonal_order=[1,1,1,7].",
    "Кейс: fallback автоподбор SARIMA на стационарном ряду с MASE 0.9 — приемлемо, "
    "но проверить, не лучше ли Prophet при годовой сезонности, score 7.",
]

_RESPONSE_FORMAT = (
    '{"score": <0..10>, "adequacy": <0..10>, "reasoning": "<кратко>", '
    '"suggestions": ["<конкретное улучшение>", ...]}'
)


def parse_judgment_json(text: str) -> dict[str, Any] | None:
    """JSON-ответ LLM: чистый, в ```блоке``` или внутри текста."""
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
    if not isinstance(parsed, dict) or not isinstance(parsed.get("score"), (int, float)):
        return None
    return parsed


def extract_judgment_text(text: str) -> dict[str, Any] | None:
    """Текстовый fallback: «SCORE: 7.5» (+ опционально SUGGESTION: ...)."""
    if not text:
        return None
    score_match = re.search(r"SCORE\s*[:=]\s*([0-9](?:\.\d+)?)", text, re.IGNORECASE)
    if not score_match:
        return None
    judgment: dict[str, Any] = {"score": float(score_match.group(1)), "suggestions": []}
    suggestion = re.search(r"SUGGEST(?:ION)?\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if suggestion:
        judgment["suggestions"] = [suggestion.group(1).strip()]
    return judgment


async def _default_judge_stub(prompt: str) -> str:
    """Заглушка LLM до Этапа 11 (заменяется реальным вызовом LM Studio)."""
    logger.info("Critic: judge-заглушка (реальная LLM — Этап 11), промпт %d симв.",
                len(prompt))
    return json.dumps(
        {
            "score": 5.0,
            "adequacy": 5.0,
            "reasoning": "LLM не подключена (Этап 11) — нейтральная оценка",
            "suggestions": [],
        },
        ensure_ascii=False,
    )


class Critic(LLMAgent):
    """Оценивает выбор модели и прогноз по шкале 0–10."""

    def __init__(self, judge: JudgeCall | None = None):
        super().__init__(name="critic", description="LLM-as-a-Judge для выбора модели")
        self.judge = judge or _default_judge_stub

    # ------------------------------------------------------------------
    # Промпт
    # ------------------------------------------------------------------
    def build_prompt(self, state: ForecastState) -> str:
        meta = state.metadata
        metrics = meta.get("val_metrics") or {}
        eda = state.diagnostics or {}
        lines = [
            "Ты — строгий судья (LLM-as-a-Judge) системы прогноза погоды.",
            "Оцени выбор модели и качество прогноза по шкале 0–10.",
            "",
            f"Задача: ряд {state.series_id}, горизонт {state.horizon}, "
            f"частота {state.freq}, сезонность {eda.get('seasonal_period')}, "
            f"стационарность (ADF): {eda.get('adf', {}).get('stationary') if eda else None}.",
            f"Выбрано: модель {state.model_type}, параметры {state.model_params}.",
            f"Метрики валидации: {metrics}.",
            f"Tool Calling fallback использован: {meta.get('used_fallback')}.",
            f"Ошибки Validator: {meta.get('validator_errors') or 'нет'}.",
            f"Предупреждения Validator: {meta.get('validator_warnings') or 'нет'}.",
        ]
        extreme = meta.get("extreme_flags")
        if extreme:
            lines.append(f"Экстремальные условия в валидации: {extreme}.")
        lines += [
            "",
            "Похожие успешные кейсы:",
            *[f"- {case}" for case in _FEWSHOT_CASES],
            "",
            f"Ответь СТРОГО одним JSON: {_RESPONSE_FORMAT}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------
    async def process(self, state: ForecastState) -> ForecastState:
        prompt = self.build_prompt(state)

        try:
            raw = await self.judge(prompt)
            judgment = parse_judgment_json(raw) or extract_judgment_text(raw)
            used_fallback = False
        except Exception as exc:  # noqa: BLE001 — сбой LLM не должен ронять пайплайн
            logger.warning("Critic: judge call failed: %s", exc)
            judgment, raw, used_fallback = None, None, True

        if judgment is None:
            # Fallback: нейтральная средняя оценка (методичка Step_5)
            judgment = {
                "score": 5.0,
                "adequacy": 5.0,
                "reasoning": "Оценка LLM недоступна — нейтральная оценка (fallback)",
                "suggestions": [],
            }
            used_fallback = True

        score = max(0.0, min(10.0, float(judgment.get("score", 5.0))))
        adequacy = judgment.get("adequacy")
        suggestions = [
            str(s) for s in (judgment.get("suggestions") or []) if str(s).strip()
        ]

        state.critic_score = score
        state.critic_reasoning = str(judgment.get("reasoning", ""))[:1000]
        state.metadata["critic_adequacy"] = (
            max(0.0, min(10.0, float(adequacy))) if adequacy is not None else None
        )
        state.metadata["critic_suggestions"] = suggestions
        state.metadata["critic_used_fallback"] = used_fallback
        if used_fallback:
            state.metadata.setdefault(
                "critic_suggestions", []
            ).append("LLM-оценка недоступна (fallback Critic)")

        self._last_result = {
            "score": score,
            "used_fallback": used_fallback,
            "suggestions": len(suggestions),
        }
        logger.info(
            "Critic: score=%s fallback=%s suggestions=%d",
            score,
            used_fallback,
            len(suggestions),
        )
        return state
