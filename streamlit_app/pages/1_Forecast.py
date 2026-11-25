# streamlit_app/pages/1_Forecast.py
"""Прогноз: загрузка ряда, запуск задачи, статус, результат и рассуждения агентов."""

import time

import streamlit as st

from utils import api_client, charts
from utils.auth_guard import logout_button, require_login
from utils.helpers import (
    format_score,
    parse_series,
    render_llm_calls,
    render_metrics,
    render_naive_comparison,
    render_tree,
    status_label,
)

st.set_page_config(page_title="Прогноз — Weather Forecast AI", page_icon="📈", layout="wide")
require_login()
logout_button()

st.title("📈 Прогноз погодного ряда")
FINISHED = ("completed", "waiting_for_human", "failed", "rejected")


def wait_for_task(task_id: str, max_seconds: int = 90) -> dict | None:
    """Опрашивает статус задачи, пока она не завершится (или не истечёт время)."""
    deadline = time.time() + max_seconds
    status = api_client.get_forecast_status(task_id)
    with st.status("Ожидание задачи…", expanded=True) as holder:
        while status and status.get("status") not in FINISHED and time.time() < deadline:
            holder.update(label=f"Статус: {status_label(status.get('status'))}")
            time.sleep(2)
            status = api_client.get_forecast_status(task_id)
        if status:
            holder.update(label=f"Готово: {status_label(status.get('status'))}",
                          state="complete" if status.get("status") in FINISHED else "running")
    return status


def show_result(task_id: str) -> None:
    """Результат: статус, метрики, график, проверки, дерево рассуждений."""
    status = wait_for_task(task_id)
    if status is None:
        st.warning("Не удалось получить статус задачи.")
        return
    state = status.get("status")
    st.info(f"Задача `{task_id[:8]}` — {status_label(state)}"
            + (f" · итераций: {status.get('iterations')}" if status.get("iterations") else ""))
    if state == "failed":
        st.error(status.get("error") or "Задача завершилась с ошибкой")
        return
    if state not in ("completed", "waiting_for_human"):
        st.info("Задача ещё выполняется — обновите страницу через несколько секунд.")
        return

    result = api_client.get_forecast_result(task_id)
    if not result:
        return
    if state == "waiting_for_human":
        st.warning("🧑‍⚖️ Прогноз требует решения эксперта (Critic ниже порога или замечания "
                   "валидатора). Задача передана в HITL-очередь.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Модель", result.get("model_type") or "—")
    c2.metric("Оценка Critic", format_score(result.get("critic_score")))
    c3.metric("Валидация", "пройдена" if result.get("validation_passed") else "есть замечания")

    predictions = result.get("predictions") or []
    dates = result.get("prediction_dates") or []
    if predictions:
        st.plotly_chart(
            charts.forecast_chart(None, [], dates, predictions,
                                  title=f"Прогноз на {len(predictions)} шагов ({result.get('model_type')})"),
            use_container_width=True,
        )
    render_metrics(result.get("metrics") or {})
    if result.get("naive_metrics"):
        render_naive_comparison(result.get("metrics") or {}, result["naive_metrics"])
    st.caption("Метрики считаются на контрольном отрезке ряда (последние `horizon` точек "
               "до прогноза) · модель: " + f"{result.get('model_params')}")

    with st.expander("🔍 Проверка Validator"):
        st.json(result.get("validation") or {})
    if result.get("critic_reasoning"):
        with st.expander("🧑‍⚖️ Рассуждение Critic"):
            st.write(result["critic_reasoning"])
            for suggestion in result.get("critic_suggestions") or []:
                st.markdown(f"- {suggestion}")
    with st.expander("🧠 Ход выполнения агентов"):
        reasoning = api_client.get_forecast_reasoning(task_id)
        if reasoning:
            render_tree(reasoning.get("tree") or [])
            render_llm_calls(reasoning.get("llm_calls") or [])


# ============================================================================
# Форма запуска
# ============================================================================
tab_file, tab_manual, tab_existing = st.tabs(["📂 Загрузить CSV", "⌨️ Ввести ряд", "🔎 По ID задачи"])

with tab_file:
    file = st.file_uploader("CSV ряда (date,value) или (value)", type="csv", key="fc_file")
    c1, c2, c3 = st.columns(3)
    horizon = c1.slider("Горизонт, дней", 1, 60, 14, key="fc_horizon")
    city = c2.selectbox("Город", ["Jena", "Delhi"], key="fc_city")
    variable = c3.selectbox(
        "Переменная", ["temperature", "humidity", "pressure", "wind_speed",
                       "precipitation", "dew_point"], key="fc_variable",
    )
    # ключ результата: при изменении входа старый результат не показывается
    fc_key = f"{getattr(file, 'name', '')}|{horizon}|{city}|{variable}"
    # сброс только при РЕАЛЬНОМ изменении входа (не на первом запуске — иначе
    # задача, переданная с главной страницы, потерялась бы)
    if st.session_state.get("fc_key") not in (None, fc_key):
        st.session_state["forecast_result"] = None
    st.session_state["fc_key"] = fc_key
    if st.button("🚀 Запустить прогноз", type="primary", disabled=file is None, key="fc_run"):
        accepted = api_client.create_forecast_file(file, horizon=horizon, city=city, variable=variable)
        if accepted and accepted.get("task_id"):
            st.session_state["forecast_result"] = accepted["task_id"]
            st.rerun()

with tab_manual:
    text = st.text_area("Значения ряда (через запятую, пробел или новую строку)",
                        height=150, key="fc_text")
    c1, c2, c3 = st.columns(3)
    m_horizon = c1.slider("Горизонт, дней", 1, 60, 14, key="m_horizon")
    m_city = c2.selectbox("Город ", ["Jena", "Delhi"], key="m_city")
    m_variable = c3.selectbox(
        "Переменная ", ["temperature", "humidity", "pressure", "wind_speed",
                        "precipitation", "dew_point"], key="m_variable",
    )
    m_key = f"{text}|{m_horizon}|{m_city}|{m_variable}"
    if st.session_state.get("m_key") not in (None, m_key):
        st.session_state["forecast_result"] = None
    st.session_state["m_key"] = m_key
    if st.button("🚀 Запустить прогноз ", type="primary", key="m_run"):
        try:
            values = parse_series(text)
        except ValueError:
            st.error("Не удалось разобрать числа — проверьте формат")
        else:
            if len(values) < 30:
                st.error("Нужно не менее 30 точек")
            else:
                accepted = api_client.create_forecast_json(
                    values, horizon=m_horizon, city=m_city, variable=m_variable
                )
                if accepted and accepted.get("task_id"):
                    st.session_state["forecast_result"] = accepted["task_id"]
                    st.rerun()

with tab_existing:
    task_input = st.text_input("ID задачи", key="fc_task_input",
                               value=st.session_state.get("forecast_result") or "")
    if st.button("Открыть", key="fc_open") and task_input.strip():
        st.session_state["forecast_result"] = task_input.strip()
        st.rerun()

st.divider()
if st.session_state.get("forecast_result"):
    show_result(st.session_state["forecast_result"])
else:
    st.caption("Загрузите ряд и запустите прогноз — результат появится здесь.")
