# streamlit_app/main.py
"""Главная страница: приветствие, здоровье сервисов, быстрый прогноз, навигация."""

import streamlit as st

from utils import api_client
from utils.auth_guard import logout_button, require_login
from utils.constants import CITIES
from utils.helpers import city_value, csv_to_series, status_label

st.set_page_config(page_title="Главная — Weather Forecast AI", page_icon="🌤", layout="wide")
require_login()
logout_button()

user = st.session_state.get("user") or {}
role = user.get("role", "user")
st.title("🌤 Weather Forecast AI")
st.markdown(f"Добро пожаловать, **{user.get('email', '')}** (роль: `{role}`)")
st.markdown(
    "Загрузите метеоряд — мультиагентная система (DataAnalyst → ModelSelector → Validator → "
    "Critic → Editor) подберёт модель через Tool Calling, построит прогноз на 7–14 дней "
    "и объяснит свой выбор."
)

# --- Здоровье сервисов
health = api_client.get_health() or {}
services = health.get("services") or {}
if services:
    with st.expander(f"🔧 Состояние сервисов: {health.get('status', '—')}", expanded=False):
        cols = st.columns(len(services))
        for col, (name, info) in zip(cols, services.items()):
            healthy = info.get("healthy", True)
            col.metric(name, "healthy" if healthy else "unhealthy",
                       delta=None if healthy else (info.get("error") or "")[:120])
        st.caption(
            f"LLM: {services.get('llm', {}).get('provider', '—')} · "
            f"задачи: {services.get('tasks', {}).get('backend', '—')}"
        )

st.divider()

# --- Быстрый прогноз: результат привязан к ключу входа (quick_key), как в методичке
left, right = st.columns([3, 2])
with left:
    st.subheader("⚡ Быстрый прогноз")
    file = st.file_uploader("CSV ряда (date,value)", type="csv", key="quick_file")
    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.slider("Горизонт, дней", 1, 60, 14, key="quick_horizon")
    with c2:
        city = st.selectbox("Город", CITIES, key="quick_city")
    with c3:
        variable = st.selectbox(
            "Переменная",
            ["temperature", "humidity", "pressure", "wind_speed", "precipitation", "dew_point"],
            key="quick_variable",
        )

    quick_key = f"{getattr(file, 'name', '')}|{horizon}|{city}|{variable}"
    if st.session_state.get("quick_key") != quick_key:
        st.session_state["quick_key"] = quick_key
        st.session_state["quick_result"] = None

    if st.button("🚀 Запустить прогноз", type="primary", disabled=file is None,
                 use_container_width=True):
        values, _dates = csv_to_series(file.getvalue())
        if len(values) < 30:
            st.error("Ряд слишком короткий: нужно не менее 30 точек")
        else:
            result = api_client.create_forecast_json(
                values, horizon=horizon, city=city_value(city), variable=variable
            )
            if result and result.get("task_id"):
                st.session_state["quick_result"] = result["task_id"]
                st.session_state["forecast_task_id"] = result["task_id"]
                st.switch_page("pages/1_Forecast.py")

    task_id = st.session_state.get("quick_result")
    if task_id:
        st.success(f"Задача запущена: `{task_id[:8]}`")
        status = api_client.get_forecast_status(task_id)
        if status:
            st.caption(
                f"Статус: {status_label(status.get('status'))} — детали на странице «Прогноз»"
            )
        else:
            st.caption("Статус уточняется…")

with right:
    st.subheader("📑 Страницы")
    st.page_link("pages/1_Forecast.py", label="📈 Прогноз — загрузка ряда и результат", icon="📈")
    st.page_link("pages/2_My_Forecasts.py", label="📁 Мои прогнозы", icon="📁")
    st.page_link("pages/3_Reasoning.py", label="🧠 Логи рассуждений и LLM-вызовов", icon="🧠")
    st.page_link("pages/4_Evaluation.py", label="📊 Оценка качества", icon="📊")
    st.page_link("pages/7_Knowledge_Base.py", label="🔍 База знаний и справочник", icon="🔍")
    if role in ("editor", "admin"):
        st.page_link("pages/5_Model_Compare.py", label="⚖️ Сравнение моделей", icon="⚖️")
        st.page_link("pages/6_HITL_Queue.py", label="🛠 HITL-очередь", icon="🛠")
    if role == "admin":
        st.page_link("pages/8_System_Health.py", label="👑 Состояние системы", icon="👑")
