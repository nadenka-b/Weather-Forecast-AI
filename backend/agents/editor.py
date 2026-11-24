"""Editor — агент-редактор (Этап 14; опциональный агент задания MLF).

Если Validator/Critic недовольны, Editor корректирует гиперпараметры и
ПЕРЕОБУЧАЕТ модель тем же инструментом из реестра (Tool Calling, без генерации
кода), после чего Supervisor заново прогоняет Validator и Critic.

Стратегия предложения правки:
1. advisor (LLM) — если задан: промпт с замечаниями и схемами инструментов,
   ответ — JSON {"tool", "arguments"}; невалидный совет отбрасывается;
2. rule_based_plan — детерминированные правила метеодомена:
   - ARIMA при сезонности <= 31 без сезонного порядка -> SARIMA;
   - автокорреляция остатков (Ljung-Box) -> рост порядка AR;
   - XGBoost -> удвоение числа лагов (давление/влажность как признаки);
   - fallback/неизвестная модель -> автоподбор ARIMA.

Цикл Editor -> Validator -> Critic организует Supervisor (rollback при
регрессии). В режиме stub LLM-совет не используется — работают правила.
"""
import logging
from typing import Any, Awaitable, Callable

from backend.agents.base import LLMAgent
from backend.agents.model_selector import (
    _eda_summary,
    apply_tool_result,
    attach_val_dates,
    prepare_tool_context,
)
from backend.core.config import get_settings
from backend.core.llm.manager import parse_tool_json
from backend.core.llm.tools import ToolRegistry, UnknownToolError, build_default_registry
from backend.core.models import ForecastState

logger = logging.getLogger(__name__)

Advisor = Callable[[str], Awaitable[str]]


class Editor(LLMAgent):
    """Корректирует гиперпараметры модели и переобучает её."""

    def __init__(self, registry: ToolRegistry | None = None, advisor: Advisor | None = None):
        super().__init__(
            name="editor",
            description="Коррекция гиперпараметров по замечаниям Validator/Critic",
        )
        self.registry = registry or build_default_registry()
        self.advisor = advisor

    # ------------------------------------------------------------------
    # Замечания и план
    # ------------------------------------------------------------------
    @staticmethod
    def issues(state: ForecastState) -> list[str]:
        """Замечания Validator и Critic (+ низкая оценка) как список строк."""
        metadata = state.metadata
        issues = [str(item) for item in (metadata.get("validator_errors") or [])]
        issues += [str(item) for item in (metadata.get("critic_suggestions") or [])]
        threshold = get_settings().critic_pass_threshold
        if state.critic_score is not None and state.critic_score < threshold:
            issues.append(
                f"Critic: оценка {state.critic_score} ниже порога {threshold}"
            )
        return issues

    def rule_based_plan(
        self, state: ForecastState
    ) -> tuple[str, dict[str, Any]] | None:
        """Детерминированная правка по правилам; None — правил нет."""
        model = (state.model_type or "").lower()
        params = dict(state.model_params or {})
        season = (state.diagnostics or {}).get("seasonal_period")
        errors = " ".join(str(e) for e in (state.metadata.get("validator_errors") or [])).lower()

        if model == "arima" and season and 1 < season <= 31 and not params.get("seasonal_order"):
            return (
                "fit_arima",
                {"order": params.get("order") or [1, 1, 1],
                 "seasonal_order": [1, 1, 1, int(season)]},
            )
        if "ljung-box" in errors and model in ("arima", "sarima"):
            order = list(params.get("order") or [1, 1, 1])
            if order[0] < 5:  # рост порядка AR, пока остатки автокоррелированы
                order[0] += 1
                arguments: dict[str, Any] = {"order": order}
                if params.get("seasonal_order"):
                    arguments["seasonal_order"] = params["seasonal_order"]
                return "fit_arima", arguments
        if model == "xgboost":
            lags = min(int(params.get("lags", 7)) * 2, 60)
            return (
                "fit_xgboost",
                {"lags": lags, "n_estimators": int(params.get("n_estimators", 200))},
            )
        if state.metadata.get("used_fallback") or not model:
            return "fit_arima", {}  # автоподбор порядка
        return None

    def build_prompt(self, state: ForecastState, issues: list[str]) -> str:
        """Промпт для LLM-советника: контекст, замечания, схемы инструментов."""
        schemas = "\n".join(
            f"- {s['function']['name']}: {s['function']['description']} "
            f"параметры: {s['function']['parameters'].get('properties', {})}"
            for s in self.registry.schemas()
        )
        return (
            "Ты — агент-редактор системы прогноза погоды. Модель выбрана, но проверка "
            "выявила замечания. Предложи ОДНУ правку: инструмент и гиперпараметры.\n\n"
            f"Данные:\n{_eda_summary(state)}\n"
            f"Текущая модель: {state.model_type}, параметры: {state.model_params}\n"
            f"Метрики валидации: {state.metadata.get('val_metrics')}\n"
            f"Замечания:\n" + "\n".join(f"- {issue}" for issue in issues) + "\n\n"
            f"Инструменты:\n{schemas}\n\n"
            'Ответь СТРОГО одним JSON: {"tool": "<имя>", "arguments": {<гиперпараметры>}}'
        )

    async def _propose(
        self, state: ForecastState, issues: list[str]
    ) -> tuple[str, dict[str, Any], str] | None:
        """Возвращает (tool, arguments, source) — совет LLM или правило."""
        if self.advisor is not None:
            try:
                raw = await self.advisor(self.build_prompt(state, issues))
            except Exception as exc:  # noqa: BLE001 — сбой LLM: работаем по правилам
                logger.warning("Editor advisor failed: %s", exc)
                raw = None
            parsed = parse_tool_json(raw or "")
            if parsed:
                try:
                    self.registry.get(parsed["tool"])
                    return parsed["tool"], parsed["arguments"], "llm"
                except UnknownToolError:
                    logger.warning("Editor: совет LLM ссылается на неизвестный инструмент")
        plan = self.rule_based_plan(state)
        return (*plan, "rules") if plan else None

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------
    async def process(self, state: ForecastState) -> ForecastState:
        state.editor_changed = False
        if state.task_type != "timeseries":
            state.metadata["editor_note"] = "Пропущено: tabular-задача"
            return state

        issues = self.issues(state)
        if not issues:
            state.metadata["editor_note"] = "Нет замечаний — правка не нужна"
            return state

        proposal = await self._propose(state, issues)
        if proposal is None:
            state.metadata["editor_note"] = "Правила не нашли, что менять"
            return state

        tool, arguments, source = proposal
        context = prepare_tool_context(state)
        before = {
            "model": state.model_type, "params": state.model_params,
            "mase": (state.metadata.get("val_metrics") or {}).get("mase"),
        }
        # исключение пробрасывается: Supervisor откатит состояние (rollback_error)
        result = self.registry.run(tool, arguments, context)
        attach_val_dates(result, context)
        apply_tool_result(state, result)
        state.editor_changed = True
        state.metadata["editor_change"] = {
            "source": source,
            "tool": tool,
            "arguments": arguments,
            "before": before,
            "after": {
                "model": state.model_type, "params": state.model_params,
                "mase": (state.metadata.get("val_metrics") or {}).get("mase"),
            },
        }
        self._last_result = state.metadata["editor_change"]
        logger.info(
            "Editor: %s -> %s (%s): %s", before["model"], state.model_type, source, arguments
        )
        return state
