"""Supervisor — оркестратор мультиагентного пайплайна прогноза (Этап 5).

Пайплайн: data_analyst -> model_selector -> validator -> critic, затем цикл
editor -> validator -> critic до сходимости или max_iterations (по методичке
Step_5: адаптивный бюджет, rollback при регрессии, stopped_reason).

Отличия от методички (по заданию MLF):
- state — строго типизированный ForecastState (Pydantic), не dict;
- reasoning_log — иерархический, с parent_id, пишется в БД (reasoning_logs);
- при несходимости после max_iterations — state.hitl_requested = True
  (Human-in-the-Loop, решение через API на Этапе 14).

Агенты внедряются извне (DI): пока реальные агенты не реализованы (Этапы 6–8),
Supervisor получает mock-агенты или частичный набор — отсутствующие
пропускаются с записью в лог. Это позволяет добавлять агентов поэтапно.
"""
import logging
from typing import Any, Mapping

from backend.agents.base import BaseAgent
from backend.agents.reasoning import (
    DbLogStore,
    InMemoryLogStore,
    LogStore,
    ReasoningLogger,
    new_request_id,
)
from backend.core.config import get_settings
from backend.core.models import ForecastState
from backend.utils.llm_logger import current_agent_name, current_request_id

logger = logging.getLogger(__name__)

# Порядок основного пайплайна (до цикла коррекции)
PIPELINE: tuple[str, ...] = ("data_analyst", "model_selector", "validator", "critic")

# Адаптивный бюджет итераций Editor (методичка Step_5):
# score None/<5.0 -> 3 попытки; <6.5 -> 2; иначе 1 при замечаниях, 0 если чисто
ADAPTIVE_BUDGET_LOW = 5.0
ADAPTIVE_BUDGET_MID = 6.5


class Supervisor:
    """Оркестрирует цикл агентов и ведёт дерево рассуждений.

    log_store задаёт хранилище reasoning-лога для каждого запроса:
    InMemoryLogStore (по умолчанию — тесты и локальные прогоны) или
    DbLogStore (таблица reasoning_logs, требование задания MLF —
    используется фабрикой build_supervisor для продакшена).
    """

    def __init__(
        self,
        agents: Mapping[str, BaseAgent] | None = None,
        max_iterations: int | None = None,
        log_store: LogStore | None = None,
        logger_: ReasoningLogger | None = None,
    ):
        settings = get_settings()
        self.agents: dict[str, BaseAgent] = dict(agents or {})
        self.max_iterations = max_iterations or settings.max_iterations
        self.critic_threshold = settings.critic_pass_threshold
        self.log_store = log_store
        self._injected_logger = logger_
        self._reasoning: ReasoningLogger | None = logger_
        self._iteration = 0

    # ------------------------------------------------------------------
    # Служебные проверки (по методичке Step_5)
    # ------------------------------------------------------------------
    def _needs_correction(self, state: ForecastState) -> bool:
        """Нужна ли итерация Editor: валидация не прошла или Critic низко оценил."""
        if state.critic_score is None:
            return not state.validation_passed
        if not state.validation_passed:
            return True
        if state.critic_score < self.critic_threshold:
            return True
        return bool(state.metadata.get("critic_suggestions"))

    def _adaptive_budget(self, state: ForecastState) -> int:
        score = state.critic_score
        if score is None or score < ADAPTIVE_BUDGET_LOW:
            return 3
        if score < ADAPTIVE_BUDGET_MID:
            return 2
        if not state.validation_passed:
            return 1
        if state.metadata.get("critic_suggestions"):
            return 1
        return 0

    @staticmethod
    def _is_improvement(prev: ForecastState, curr: ForecastState) -> bool:
        """Стало ли лучше: рост critic_score, затем validation_passed."""
        prev_score, curr_score = prev.critic_score, curr.critic_score
        if curr_score is None and prev_score is not None:
            return False
        if curr_score is not None and prev_score is None:
            return True
        if curr_score is not None and prev_score is not None:
            if curr_score > prev_score:
                return True
            if curr_score < prev_score:
                return False
        if curr.validation_passed and not prev.validation_passed:
            return True
        return False

    def _compute_final(self, state: ForecastState) -> bool:
        """Финальная сходимость: валидация прошла и Critic не ниже порога."""
        if not state.validation_passed:
            return False
        if state.critic_score is None:
            return True  # Critic не вызывался — прошли все выполненные проверки
        return state.critic_score >= self.critic_threshold

    # ------------------------------------------------------------------
    # Логирование шагов
    # ------------------------------------------------------------------
    async def _run_agent(
        self,
        name: str,
        state: ForecastState,
        parent_id: str | None,
        parameters: dict[str, Any] | None = None,
    ) -> ForecastState:
        """Вызывает агента с обёрткой в лог; отсутствующего — пропускает."""
        agent = self.agents.get(name)
        if agent is None:
            logger.info("Agent '%s' not registered — skipping", name)
            if self._reasoning:
                skip_id = await self._reasoning.start(
                    name, parent_id=parent_id, input_summary="not_implemented"
                )
                await self._reasoning.finish(skip_id, status="skipped")
            return state

        step_id = None
        if self._reasoning:
            step_id = await self._reasoning.start(
                name,
                parent_id=parent_id,
                model_name=getattr(agent, "model_name", None),
                parameters=parameters,
                input_summary=f"series={state.series_id} iter={state.iteration}",
            )
        agent_token = current_agent_name.set(name)
        try:
            state = await agent.process(state)
        except Exception as exc:  # noqa: BLE001 — сбой агента не роняет цикл
            logger.exception("Agent '%s' failed: %s", name, exc)
            if self._reasoning and step_id:
                await self._reasoning.finish(
                    step_id, output_summary=str(exc)[:500], status="error"
                )
            state.metadata.setdefault("agent_errors", {})[name] = str(exc)[:200]
            return state
        finally:
            current_agent_name.reset(agent_token)

        if self._reasoning and step_id:
            await self._reasoning.finish(
                step_id, output_summary=self._summarize(state)
            )
        return state

    @staticmethod
    def _summarize(state: ForecastState) -> str:
        return (
            f"model={state.model_type} passed={state.validation_passed} "
            f"critic={state.critic_score}"
        )

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------
    async def process(self, state: ForecastState) -> ForecastState:
        """Прогоняет пайплайн и цикл коррекции, возвращает финальный state."""
        self._iteration = 0
        # Свежий (пустой) внедрённый логгер задаёт request_id; иначе генерируем новый
        if not state.log_id:
            reusable = self._reasoning is not None and not self._reasoning.entries
            state.log_id = self._reasoning.request_id if reusable else new_request_id()

        needs_new_logger = (
            self._reasoning is None
            or self._reasoning.request_id != state.log_id
            or self._reasoning.entries  # повторный прогон — новое дерево
        )
        if needs_new_logger:
            store = self.log_store or getattr(self._reasoning, "store", None)
            self._reasoning = ReasoningLogger(state.log_id, store=store)
        reasoning = self._reasoning

        root_id = await reasoning.start(
            "supervisor",
            input_summary=f"series={state.series_id} horizon={state.horizon}",
            parameters={"max_iterations": self.max_iterations},
        )

        state.editor_changed = False
        state.hitl_requested = False
        state.iteration = 0

        # request_id для трассировки LLM-вызовов (llm_calls сшивается с reasoning_logs)
        request_token = current_request_id.set(state.log_id)
        try:
            return await self._run_pipeline(state, root_id)
        finally:
            current_request_id.reset(request_token)

    async def _run_pipeline(self, state: ForecastState, root_id: str) -> ForecastState:
        """Основной пайплайн и цикл коррекции (вызывается из process)."""
        reasoning = self._reasoning
        assert reasoning is not None

        # --- Основной пайплайн: analyst -> selector -> validator -> critic
        for name in PIPELINE:
            state = await self._run_agent(name, state, root_id)

        editor = self.agents.get("editor")
        needs_correction = self._needs_correction(state)
        adaptive_budget = self._adaptive_budget(state) if needs_correction else 0
        effective_max = min(self.max_iterations, adaptive_budget)
        initial_score = state.critic_score

        if not needs_correction:
            stopped_reason = "converged"
        elif editor is None:
            stopped_reason = "editor_not_registered"
        elif effective_max == 0:
            stopped_reason = "budget_exhausted"
        else:
            stopped_reason = "max_iterations"

        snapshot = state.model_copy(deep=True)

        # --- Цикл коррекции: editor -> validator -> critic
        while editor is not None and needs_correction and self._iteration < effective_max:
            self._iteration += 1
            state.iteration = self._iteration

            state = await self._run_agent(
                "editor", state, root_id, parameters={"iteration": self._iteration}
            )
            state = await self._run_agent("validator", state, root_id)
            state = await self._run_agent("critic", state, root_id)

            agent_failed = bool(state.metadata.get("agent_errors"))
            if agent_failed:
                state = snapshot.model_copy(deep=True)
                state.metadata["editor_rolled_back"] = True
                stopped_reason = "rollback_error"
                break

            if not state.editor_changed:
                stopped_reason = "no_change"
                break

            if not self._is_improvement(snapshot, state):
                state = snapshot.model_copy(deep=True)
                state.metadata["editor_rolled_back"] = True
                stopped_reason = "rollback_regression"
                break

            snapshot = state.model_copy(deep=True)
            state.metadata["editor_rolled_back"] = False
            needs_correction = self._needs_correction(state)
            if not needs_correction:
                stopped_reason = "converged"

        # --- Финализация
        final_passed = self._compute_final(state)
        # HITL (задание MLF): не сошлись после бюджета — нужен человек
        if not final_passed and stopped_reason in (
            "max_iterations",
            "budget_exhausted",
            "no_change",
            "rollback_regression",
            "rollback_error",
            "editor_not_registered",
        ):
            state.hitl_requested = True

        state.max_iterations = self.max_iterations
        state.metadata["supervisor"] = {
            "iterations": self._iteration,
            "final": final_passed,
            "stopped_reason": stopped_reason,
            "budget": effective_max,
            "initial_critic_score": initial_score,
        }
        state.metadata["reasoning_trace"] = [
            {k: v for k, v in e.items() if k != "parameters"} for e in reasoning.entries
        ]

        await reasoning.finish(
            root_id,
            output_summary=(
                f"final={final_passed} reason={stopped_reason} "
                f"iterations={self._iteration} hitl={state.hitl_requested}"
            ),
        )
        logger.info(
            "Supervisor final: iterations=%s passed=%s reason=%s hitl=%s",
            self._iteration,
            final_passed,
            stopped_reason,
            state.hitl_requested,
        )
        return state

    def get_status(self) -> dict[str, Any]:
        """Краткий статус последнего прогона (для /status и тестов)."""
        return {
            "iterations": self._iteration,
            "agents_registered": sorted(self.agents.keys()),
        }


async def build_supervisor(max_iterations: int | None = None) -> Supervisor:
    """Продакшен-сборка (Этапы 10–12): все реальные агенты + RAG-поиск кейсов
    + reasoning-лог в таблице reasoning_logs (требование задания MLF).

    Editor подключается на Этапе 14 (HITL-цикл коррекции).
    """
    from backend.agents.critic import Critic
    from backend.agents.data_analyst import DataAnalyst
    from backend.agents.model_selector import ModelSelector
    from backend.agents.validator import Validator

    agents: dict[str, BaseAgent] = {
        "data_analyst": DataAnalyst(),
        "model_selector": ModelSelector(),
        "validator": Validator(),
        "critic": Critic(),
    }
    try:
        from backend.retrieval.hybrid_retriever import build_hybrid_retriever

        retriever = await build_hybrid_retriever()
        agents["model_selector"] = ModelSelector(retriever=retriever)
        logger.info("RAG-поиск кейсов подключён к ModelSelector")
    except Exception as exc:  # noqa: BLE001 — без БЗ пайплайн работает без few-shot
        logger.warning("RAG недоступен (%s) — ModelSelector без few-shot кейсов", exc)

    return Supervisor(
        agents=agents,
        max_iterations=max_iterations,
        log_store=DbLogStore(),
    )
