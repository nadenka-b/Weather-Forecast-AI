# streamlit_app/pages/8_System_Health.py
"""Состояние системы: сервисы, кэш, переиндексация RAG (admin)."""

import streamlit as st

from utils import api_client
from utils.auth_guard import logout_button, require_role

st.set_page_config(page_title="Состояние системы — Weather Forecast AI", page_icon="👑", layout="wide")
require_role("admin")
logout_button()

st.title("👑 Состояние системы")

# --- Сервисы
health = api_client.get_health() or {}
services = health.get("services") or {}
st.metric("Общий статус", health.get("status", "—"))
if services:
    cols = st.columns(len(services))
    for col, (name, info) in zip(cols, services.items()):
        healthy = info.get("healthy", True)
        col.metric(f"{name}{' · ' + info['provider'] if info.get('provider') else ''}",
                   "healthy" if healthy else "unhealthy")
        if not healthy and info.get("error"):
            col.caption(str(info["error"])[:200])
        if name == "tasks":
            col.caption(f"backend: {info.get('backend')}")

st.divider()

# --- Кэш
left, right = st.columns(2)
with left:
    st.subheader("Кэш")
    stats = api_client.get_cache_stats()
    if stats:
        c1, c2 = st.columns(2)
        c1.metric("Записей", stats.get("entries", 0))
        c2.metric("Попаданий", stats.get("total_hits", 0))

with right:
    st.subheader("Очистка кэша")
    older_than_days = st.number_input("Старше, дней (пусто = 30)", min_value=1, max_value=365,
                                      value=30)
    max_hits = st.number_input("Число попаданий ≤ (—1 = не учитывать)", min_value=-1,
                               max_value=1000, value=-1)
    if older_than_days and max_hits < 0 and max_hits != -1:
        st.warning("⚠️ Будут удалены все записи кэша, старше указанного срока.")
    confirm = st.checkbox("Я понимаю, что действие нельзя отменить", key="clear_cache_confirm")
    if st.button("🗑 Очистить кэш", disabled=not confirm):
        result = api_client.clear_cache(
            older_than_days=int(older_than_days) if older_than_days else None,
            max_hits=int(max_hits) if max_hits >= 0 else None,
        )
        if result:
            st.success(f"Очистка запущена · удалено сразу: {result.get('deleted_inline', 0)}")

st.divider()

# --- RAG
st.subheader("База знаний и RAG")
kb = api_client.list_kb_cases(limit=1)
glossary = api_client.list_terms(limit=1)
c1, c2 = st.columns(2)
c1.metric("Кейсов в базе знаний", (kb or {}).get("total", 0))
c2.metric("Терминов в справочнике", (glossary or {}).get("total", 0))
if st.button("🔄 Переиндексировать RAG"):
    result = api_client.reindex_kb()
    if result:
        st.success("Переиндексация запущена (фоновая задача)")
