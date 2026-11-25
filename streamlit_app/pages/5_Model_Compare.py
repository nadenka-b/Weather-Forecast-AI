# streamlit_app/pages/5_Model_Compare.py
"""Сравнение именованных моделей на одной выборке тестовых окон (editor/admin)."""

import pandas as pd
import streamlit as st

from utils import api_client, charts
from utils.auth_guard import logout_button, require_role
from utils.helpers import format_number

st.set_page_config(page_title="Сравнение моделей — Weather Forecast AI", page_icon="⚖️", layout="wide")
require_role("editor", "admin")
logout_button()

st.title("⚖️ Сравнение моделей")
st.markdown(
    "Каждая модель обучается на обучающем отрезке тестовых окон и прогнозирует горизонт; "
    "метрики усредняются по окнам, пары моделей сравниваются тестом Уилкоксона и bootstrap-CI."
)

with st.form("compare_form"):
    c1, c2, c3 = st.columns(3)
    models = c1.multiselect("Модели", ["naive", "arima", "sarima", "xgboost", "prophet"],
                            default=["naive", "arima", "xgboost"])
    sample_size = c2.select_slider("Число окон", [5, 10, 20, 30, 50], value=10)
    horizon = c3.selectbox("Горизонт", [7, 14, 20], index=1)
    submitted = st.form_submit_button("▶️ Сравнить", use_container_width=True)

compare_key = f"{sorted(models)}|{sample_size}|{horizon}"
if st.session_state.get("compare_key") != compare_key:
    st.session_state["compare_key"] = compare_key
    st.session_state["compare_data"] = None

if submitted:
    if len(models) < 1:
        st.error("Выберите хотя бы одну модель")
    else:
        st.session_state["compare_data"] = api_client.compare_models(models, sample_size, horizon)

result = st.session_state.get("compare_data")
if result:
    st.metric("Лучшая модель", result.get("winner") or "—")
    results = result.get("results") or {}
    st.dataframe(pd.DataFrame([
        {
            "Модель": name,
            "Тип": info.get("model_type"),
            "MAE (среднее)": format_number((info.get("mae") or {}).get("mean"), ".3f"),
            "MAE (медиана)": format_number((info.get("mae") or {}).get("median"), ".3f"),
            "RMSE (среднее)": format_number((info.get("rmse") or {}).get("mean"), ".3f"),
            "Окон": (info.get("mae") or {}).get("n"),
            **({"Ошибка": info["error"]} if info.get("error") else {}),
        }
        for name, info in results.items()
    ]), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    left.plotly_chart(charts.models_bar_chart(results, "mae"), use_container_width=True)
    left.plotly_chart(charts.models_bar_chart(results, "rmse"), use_container_width=True)

    pairs = result.get("pairs") or []
    if pairs:
        right.subheader("Попарные сравнения")
        right.dataframe(pd.DataFrame([
            {
                "Пара": " vs ".join(pair["models_compared"]),
                "Победитель": pair["winner"],
                "Разность MAE": format_number((pair.get("bootstrap") or {}).get("mean_diff"), ".3f"),
                "95% CI": f"[{format_number((pair.get('bootstrap') or {}).get('ci_low'), '.3f')}, "
                          f"{format_number((pair.get('bootstrap') or {}).get('ci_high'), '.3f')}]",
                "Wilcoxon p": format_number((pair.get("wilcoxon") or {}).get("pvalue"), ".3f"),
            }
            for pair in pairs
        ]), use_container_width=True, hide_index=True)
else:
    st.caption("Выберите модели и нажмите «Сравнить».")
