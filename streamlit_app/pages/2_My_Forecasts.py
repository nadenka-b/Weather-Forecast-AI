# streamlit_app/pages/2_My_Forecasts.py
"""Мои прогнозы: список задач пользователя, статусы, переход к результату."""

import pandas as pd
import streamlit as st

from utils import api_client
from utils.auth_guard import logout_button, require_login
from utils.helpers import format_number, short, status_label

st.set_page_config(page_title="Мои прогнозы — Weather Forecast AI", page_icon="📁", layout="wide")
require_login()
logout_button()

st.title("📁 Мои прогнозы")

# кэш списка: перезапрос только при изменении фильтра или по кнопке «Обновить»
limit = st.selectbox("Показать", [10, 20, 50], index=1)
list_key = f"my_tasks|{limit}"
if st.button("🔄 Обновить") or st.session_state.get("my_tasks_key") != list_key:
    st.session_state["my_tasks_key"] = list_key
    st.session_state["my_tasks_data"] = api_client.list_my_forecasts(limit=limit)

data = st.session_state.get("my_tasks_data") or {}
tasks = data.get("tasks") or []
if not tasks:
    st.info("Задач пока нет — запустите прогноз на странице «Прогноз».")
    st.stop()

frame = pd.DataFrame([
    {
        "ID": task["task_id"][:8],
        "Статус": status_label(task["status"]),
        "Ряд": short(task.get("series_id") or "", 28),
        "Город": task.get("city") or "—",
        "Переменная": task.get("variable") or "—",
        "Горизонт": task.get("horizon"),
        "Модель": task.get("model_type") or "—",
        "MAE": format_number(task.get("mae")),
        "MAE наива": format_number(task.get("naive_mae")),
        "HITL": "да" if task.get("hitl_requested") else "",
        "Создана": (task.get("created_at") or "")[:19].replace("T", " "),
    }
    for task in tasks
])
st.dataframe(frame, use_container_width=True, hide_index=True)

options = {task["task_id"]: f"{task['task_id'][:8]} · {status_label(task['status'])}" for task in tasks}
selected = st.selectbox("Задача", list(options.keys()),
                        format_func=lambda task_id: options[task_id])
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📈 Открыть результат", use_container_width=True):
        st.session_state["forecast_result"] = selected
        st.switch_page("pages/1_Forecast.py")
with col2:
    if st.button("🧠 Логи рассуждений", use_container_width=True):
        st.session_state["reasoning_task_id"] = selected
        st.switch_page("pages/3_Reasoning.py")

task = next(t for t in tasks if t["task_id"] == selected)
if task.get("error"):
    st.error(task["error"])
result = api_client.get_forecast_result(selected)
if result:
    mae = (result.get("metrics") or {}).get("mae")
    naive_mae = (result.get("naive_metrics") or {}).get("mae")
    comparison = (f" · **MAE:** {format_number(mae)} против наива {format_number(naive_mae)}"
                  if mae is not None else "")
    st.markdown(f"**Модель:** `{result.get('model_type')}` · **Critic:** {result.get('critic_score')} "
                f"· **Валидация:** {'пройдена' if result.get('validation_passed') else 'есть замечания'}"
                + comparison)
    predictions = result.get("predictions") or []
    if predictions:
        st.markdown("**Прогноз:** " + ", ".join(f"{v:.2f}" for v in predictions[:14])
                    + (" …" if len(predictions) > 14 else ""))
else:
    st.caption("Результат ещё не готов или задача завершена с ошибкой.")
