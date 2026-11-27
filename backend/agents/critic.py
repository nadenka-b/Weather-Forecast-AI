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
from backend.core.config import get_settings
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


def _neutral_judgment(reasoning: str) -> str:
    """Нейтральная оценка 5.0 (fallback / режим stub)."""
    return json.dumps(
        {"score": 5.0, "adequacy": 5.0, "reasoning": reasoning, "suggestions": []},
        ensure_ascii=False,
    )


async def _judge_via_llm(prompt: str) -> str:
    """Реальный вызов LLM (LM Studio / vLLM / HF / API) через LLMManager.

    Вызов трассируется в llm_calls; Tool Calling здесь не нужен — формат JSON
    задан в промпте и парсится parse_judgment_json.
    """
    from backend.core.llm.manager import get_llm_manager

    return await get_llm_manager().complete(prompt, max_tokens=400)


async def _stub_judge(prompt: str) -> str:
    """LLM_PROVIDER=stub: нейтральная оценка без сети (тесты, офлайн)."""
    logger.info("Critic: режим stub — нейтральная оценка (промпт %d симв.)", len(prompt))
    return _neutral_judgment("LLM не подключена (режим stub) — нейтральная оценка")


class Critic(LLMAgent):
    """Оценивает выбор модели и прогноз по шкале 0–10."""

    def __init__(self, judge: JudgeCall | None = None):
        super().__init__(name="critic", description="LLM-as-a-Judge для выбора модели")
        if judge is not None:
            self.judge = judge
        elif get_settings().llm_provider.lower() == "stub":
            self.judge = _stub_judge
        else:
            self.judge = _judge_via_llm

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
            # без порогов LLM ставит «безопасную» 7 при любом качестве
            # (проверено на живых прогонах: MASE 0.46 и 2.69 получали одинаковые 7)
            "Оценивай по совокупности критериев (не по одному MASE):",
            "- MASE — главный ориентир: < 0.9 — заметно лучше наивного, 0.9–1.0 — "
            "на уровне, > 1.0 — хуже наивного;",
            "- WAPE и MAE — масштаб ошибки в единицах ряда (для температуры — °C);",
            "- ошибки Validator (серьёзно) и предупреждения (умеренно);",
            "- адекватность выбора характеру ряда: сезонный порядок при "
            "seasonal_period > 1, осознанный выбор против fallback.",
            "Пороги: 9-10 — ошибок нет, MASE < 0.9 и выбор адекватен; 7-8 — ошибок "
            "нет и MASE около 1; 4-6 — предупреждения, или MASE 1.0-1.3, или "
            "выбор неадекватен; 1-3 — ошибки валидатора или MASE > 1.3.",
            "MAPE не использовать как главный критерий (нестабилен около нуля).",
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
        # расход LLM судьи — для агентных метрик run_eval (Этап 13)
        if self.judge is _judge_via_llm:
            from backend.core.llm.manager import get_llm_manager

            state.metadata["critic_usage"] = dict(get_llm_manager().last_usage)
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
