# streamlit_app/pages/6_HITL_Queue.py
"""HITL-очередь: просмотр, правка, подтверждение/отклонение (editor/admin)."""

import pandas as pd
import streamlit as st

from utils import api_client
from utils.auth_guard import logout_button, require_role
from utils.helpers import ALL, format_score, short

st.set_page_config(page_title="HITL-очередь — Weather Forecast AI", page_icon="🛠", layout="wide")
require_role("editor", "admin")
logout_button()

st.title("🛠 Очередь экспертной проверки")
st.markdown(
    "Здесь оказываются прогнозы, где цикл агентов не сошёлся. Подтверждение закрывает задачу; "
    "кейс попадает в базу знаний (active learning), только если прогноз лучше наивного "
    "(MASE < 1). Отклонение закрывает задачу без изменения базы."
)

stats = api_client.get_hitl_stats()
if stats:
    c1, c2, c3 = st.columns(3)
    c1.metric("В очереди", stats["pending"])
    c2.metric("Подтверждено", stats["confirmed"])
    c3.metric("Отклонено", stats["rejected"])

c_city, c_var, _ = st.columns(3)
city = c_city.selectbox("Город", [ALL, "Jena", "Delhi"])
variable = c_var.selectbox(
    "Переменная", [ALL, "temperature", "humidity", "pressure", "wind_speed", "precipitation",
                   "dew_point"],
)
queue_key = f"hitl|{city}|{variable}"
# перевыгрузка при смене фильтров ИЛИ если список стёрт (после подтверждения/
# отклонения данные обнуляются — без второго условия очередь показывалась пустой)
if (st.session_state.get("hitl_queue_key") != queue_key
        or st.session_state.get("hitl_queue_data") is None):
    st.session_state["hitl_queue_key"] = queue_key
    st.session_state["hitl_queue_data"] = api_client.get_hitl_queue(
        city=None if city == ALL else city, variable=None if variable == ALL else variable
    )

queue = st.session_state.get("hitl_queue_data") or {}
examples = queue.get("examples") or []
if not examples:
    st.success("Очередь пуста — все прогнозы хорошего качества 👌")
else:
    for item in examples:
        task_id = item["task_id"]
        header = (f"⏳ `{task_id[:8]}` · {item.get('city') or '—'} / {item.get('variable') or '—'} · "
                  f"модель: {item.get('model_type') or '—'} · Critic: {format_score(item.get('critic_score'))}")
        with st.expander(header):
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Почему в очереди:**")
                for error in item.get("validator_errors") or []:
                    st.markdown(f"- ⚠️ Validator: {error}")
                for suggestion in item.get("critic_suggestions") or []:
                    st.markdown(f"- 💬 Critic: {suggestion}")
                if not (item.get("validator_errors") or item.get("critic_suggestions")):
                    st.caption(f"Critic ниже порога: {item.get('critic_reasoning') or 'без пояснения'}")
                st.markdown(
                    f"**Остановка цикла:** {item.get('stopped_reason')} · итераций: {item.get('iterations')}"
                )
            with right:
                st.markdown("**Правка (опционально):**")
                override = st.selectbox(
                    "Модель", ["(оставить текущую)", "fit_arima", "fit_prophet", "fit_xgboost"],
                    key=f"override_{task_id}",
                )
                comment = st.text_input("Комментарий", key=f"comment_{task_id}")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✅ Подтвердить", key=f"ok_{task_id}", type="primary",
                                 use_container_width=True):
                        result = api_client.confirm_hitl(
                            task_id,
                            model_type=None if override.startswith("(") else override,
                            comment=comment or None,
                        )
                        if result:
                            st.session_state["hitl_queue_data"] = None
                            st.success(f"Подтверждено · кейс в базу знаний: {result['kb_inserted']}")
                            st.rerun()
                with b2:
                    if st.button("❌ Отклонить", key=f"no_{task_id}", use_container_width=True):
                        if api_client.reject_hitl(task_id, reason=comment or None):
                            st.session_state["hitl_queue_data"] = None
                            st.rerun()

st.divider()

# --- Ручное сохранение правки (без смены статуса)
with st.expander("📝 Сохранить правку по задаче (без смены статуса)"):
    with st.form("correction_form"):
        corr_task = st.text_input("ID задачи")
        corr_comment = st.text_input("Комментарий / правка")
        add_to_kb = st.checkbox("Добавить кейс задачи в базу знаний")
        if st.form_submit_button("Сохранить", use_container_width=True):
            if not corr_task.strip():
                st.error("Укажите ID задачи")
            else:
                api_client.save_correction(corr_task.strip(), comment=corr_comment or None,
                                           add_to_kb=add_to_kb)

feedback = api_client.list_feedback(limit=20)
rows = (feedback or {}).get("feedback") or []
if rows:
    st.subheader("История решений")
    st.dataframe(pd.DataFrame([
        {"Задача": row["task_id"][:8], "Решение": row["decision"],
         "Комментарий": short(row.get("comment") or "", 60),
         "Когда": (row.get("created_at") or "")[:19].replace("T", " ")}
        for row in rows
    ]), use_container_width=True, hide_index=True)
