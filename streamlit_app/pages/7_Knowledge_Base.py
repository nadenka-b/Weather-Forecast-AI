# streamlit_app/pages/7_Knowledge_Base.py
"""База знаний: поиск похожих кейсов, просмотр и правка кейсов, справочник терминов."""

import pandas as pd
import streamlit as st

from utils import api_client
from utils.auth_guard import has_role, logout_button, require_login
from utils.helpers import ALL, format_number, short

st.set_page_config(page_title="База знаний — Weather Forecast AI", page_icon="🔍", layout="wide")
require_login()
logout_button()

is_admin = has_role("admin")
st.title("🔍 База знаний и справочник")

tab_search, tab_kb, tab_glossary = st.tabs(["🔎 Поиск кейсов", "📚 Кейсы", "📖 Справочник"])

# ============================================================================
# Поиск похожих кейсов (BM25 по описаниям)
# ============================================================================
with tab_search:
    query = st.text_input(
        "Запрос", placeholder="например: дневной ряд temperature Jena сезонность 7",
    )
    if query.strip():
        found = api_client.search_kb(query)
        results = (found or {}).get("results") or []
        if not results:
            st.info("Ничего не найдено.")
        for item in results:
            with st.expander(f"{item['score']:.2f} · {item['best_model']} · {item['series_id']}"):
                st.write(item.get("description") or "")
                c1, c2, c3 = st.columns(3)
                c1.metric("Город", item.get("city") or "—")
                c2.metric("Переменная", item.get("variable") or "—")
                c3.metric("MASE", format_number((item.get("metrics") or {}).get("mase")))
                st.caption(f"Гиперпараметры: {item.get('hyperparams')}")

# ============================================================================
# Кейсы базы знаний
# ============================================================================
with tab_kb:
    c1, c2, c3, c4 = st.columns(4)
    f_city = c1.selectbox("Город", [ALL, "Jena", "Delhi"], key="kb_city")
    f_model = c2.selectbox("Модель", [ALL, "arima", "sarima", "xgboost", "naive_seasonal",
                                      "prophet"], key="kb_model")
    f_search = c3.text_input("Поиск по описанию", key="kb_search")
    f_limit = c4.selectbox("Записей", [20, 50, 100], index=1, key="kb_limit")

    kb_key = f"kb|{f_city}|{f_model}|{f_search}|{f_limit}"
    if st.session_state.get("kb_filter_key") != kb_key:
        st.session_state["kb_filter_key"] = kb_key
        st.session_state["kb_data"] = api_client.list_kb_cases(
            city=None if f_city == ALL else f_city, best_model=None if f_model == ALL else f_model,
            search=f_search or None, limit=f_limit,
        )
    data = st.session_state.get("kb_data") or {}
    st.caption(f"Всего кейсов по фильтру: {data.get('total', 0)}")

    cases = data.get("cases") or []
    if cases:
        st.dataframe(pd.DataFrame([
            {
                "ID": case["id"][:8],
                "Series ID": case["series_id"],
                "Город": case.get("city") or "—",
                "Переменная": case.get("target_column") or "—",
                "Модель": case["best_model"],
                "MASE": format_number((case.get("metrics") or {}).get("mase")),
                "Описание": short(case.get("description") or "", 80),
            }
            for case in cases
        ]), use_container_width=True, hide_index=True)
        selected = st.selectbox("Кейс", [c["id"] for c in cases],
                                format_func=lambda cid: next(
                                    c["series_id"] for c in cases if c["id"] == cid))
        case = api_client.api_request("GET", f"/kb/cases/{selected}")
        if case:
            left, right = st.columns(2)
            with left:
                st.json(case.get("features") or {})
            with right:
                st.write(case.get("description") or "")
                st.write(f"**Модель:** {case['best_model']} · **Параметры:** {case.get('hyperparams')}")
                st.write(f"**Метрики:** {case.get('metrics')}")

    if is_admin:
        with st.expander("➕ Добавить кейс (admin)"):
            with st.form("kb_add_form"):
                a1, a2 = st.columns(2)
                series_id = a1.text_input("series_id")
                best_model = a2.selectbox("best_model", ["arima", "sarima", "xgboost",
                                                         "naive_seasonal", "prophet"])
                a3, a4, a5 = st.columns(3)
                city = a3.text_input("city")
                variable = a4.text_input("variable (target_column)")
                description = a5.text_input("description")
                if st.form_submit_button("Добавить", use_container_width=True):
                    if series_id.strip():
                        api_client.add_kb_case({
                            "series_id": series_id.strip(), "best_model": best_model,
                            "city": city or None, "freq": "D",
                            "target_column": variable or None,
                            "features": {"variable": variable or None, "city": city or None},
                            "description": description or None,
                        })
                        st.session_state["kb_filter_key"] = None
                        st.rerun()
                    else:
                        st.error("Укажите series_id")

        if cases:
            with st.expander("🗑 Удалить кейс (admin)"):
                delete_id = st.selectbox("Кейс для удаления", [c["id"] for c in cases], key="kb_del")
                st.warning("Удаление кейса необратимо и влияет на поиск похожих кейсов (RAG).")
                confirm = st.checkbox("Я понимаю, что действие нельзя отменить",
                                      key=f"kb_del_confirm_{delete_id}")
                if st.button("Удалить", disabled=not confirm):
                    if api_client.delete_kb_case(delete_id):
                        st.session_state["kb_filter_key"] = None
                        st.rerun()

# ============================================================================
# Справочник терминов
# ============================================================================
with tab_glossary:
    g1, g2, g3 = st.columns(3)
    categories = (api_client.list_terms(limit=1) or {}).get("categories") or []
    g_category = g1.selectbox("Категория", [ALL] + categories, key="gl_cat")
    g_search = g2.text_input("Поиск по термины и определению", key="gl_search")
    g_limit = g3.selectbox("Записей", [20, 50, 100], index=1, key="gl_limit")

    gl_key = f"glossary|{g_category}|{g_search}|{g_limit}"
    if st.session_state.get("glossary_filter_key") != gl_key:
        st.session_state["glossary_filter_key"] = gl_key
        st.session_state["glossary_data"] = api_client.list_terms(
            category=None if g_category == ALL else g_category, search=g_search or None,
            limit=g_limit,
        )
    glossary = st.session_state.get("glossary_data") or {}
    st.caption(f"Всего терминов: {glossary.get('total', 0)}")
    terms = glossary.get("terms") or []
    if terms:
        st.dataframe(pd.DataFrame(terms), use_container_width=True, hide_index=True)

    if is_admin:
        c_left, c_right = st.columns(2)
        with c_left:
            with st.form("term_form"):
                t1, t2 = st.columns(2)
                term = t1.text_input("Термин")
                category = t2.text_input("Категория (например, метеорология)")
                definition = st.text_area("Определение")
                if st.form_submit_button("Добавить термин", use_container_width=True):
                    if term.strip() and definition.strip():
                        api_client.add_term(term, definition, category or None)
                        st.session_state["glossary_filter_key"] = None
                        st.rerun()
                    else:
                        st.error("Нужны термин и определение")
        with c_right:
            uploaded = st.file_uploader("Импорт CSV (term;definition;category)", type="csv",
                                        key="gl_csv")
            if uploaded is not None and st.button("Импортировать"):
                result = api_client.import_terms_csv(uploaded)
                if result:
                    st.success(f"Добавлено: {result['added']} · пропущено (дубли): {result['skipped']}")
                    st.session_state["glossary_filter_key"] = None

        if terms:
            del_term = st.selectbox("Термин для удаления", [t["id"] for t in terms], key="gl_del",
                                    format_func=lambda tid: next(
                                        t["term"] for t in terms if t["id"] == tid))
            confirm = st.checkbox("Подтверждаю удаление", key=f"gl_del_confirm_{del_term}")
            if st.button("Удалить термин", disabled=not confirm):
                if api_client.delete_term(del_term):
                    st.session_state["glossary_filter_key"] = None
                    st.rerun()
