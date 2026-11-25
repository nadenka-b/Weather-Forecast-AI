# streamlit_app/pages/3_Reasoning.py
"""Логи рассуждений: выбор задачи, пошаговое дерево действий агентов и трассировка LLM."""

import streamlit as st

from utils import api_client
from utils.auth_guard import logout_button, require_login
from utils.helpers import render_llm_calls, render_tree, status_label

st.set_page_config(page_title="Логи рассуждений — Weather Forecast AI", page_icon="🧠", layout="wide")
require_login()
logout_button()

st.title("🧠 Логи рассуждений агентов")
st.markdown(
    "Дерево шагов из `reasoning_logs` (Supervisor → агенты → инструменты) и трассировка "
    "вызовов LLM из `llm_calls` (модель, токены, latency)."
)

tasks = (api_client.list_my_forecasts(limit=50) or {}).get("tasks") or []
options = {task["task_id"]: f"{task['task_id'][:8]} · {status_label(task['status'])}" for task in tasks}
preselected = st.session_state.get("reasoning_task_id")

col_input, col_or = st.columns([3, 1])
with col_input:
    if options:
        index = list(options.keys()).index(preselected) if preselected in options else 0
        task_id = st.selectbox("Задача", list(options.keys()), index=index,
                               format_func=lambda value: options[value])
    else:
        task_id = st.text_input("ID задачи", value=preselected or "")
        if not task_id.strip():
            st.info("У вас пока нет задач — запустите прогноз.")

if not task_id or not task_id.strip():
    st.stop()

reasoning = api_client.get_forecast_reasoning(task_id.strip())
if not reasoning:
    st.stop()

tree = reasoning.get("tree") or []
calls = reasoning.get("llm_calls") or []


def count_steps(nodes: list) -> int:
    return sum(1 + count_steps(node.get("children") or []) for node in nodes)


c1, c2, c3, c4 = st.columns(4)
c1.metric("Шагов агентов", count_steps(tree))
c2.metric("Вызовов LLM", len(calls))
c3.metric("Токены (вход)", sum((call.get("tokens_in") or 0) for call in calls))
c4.metric("Токены (выход)", sum((call.get("tokens_out") or 0) for call in calls))

st.subheader("Дерево рассуждений")
if tree:
    render_tree(tree)
else:
    st.info("Лог рассуждений пуст — возможно, задача ещё не запускалась.")

st.subheader("Трассировка вызовов LLM")
render_llm_calls(calls)
