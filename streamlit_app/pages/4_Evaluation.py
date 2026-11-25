# streamlit_app/pages/4_Evaluation.py
"""Оценка качества: пакетный запуск (режимы × горизонты × доли), таблицы и графики."""

import time

import pandas as pd
import streamlit as st

from utils import api_client, charts
from utils.auth_guard import has_role, logout_button, require_login
from utils.helpers import ALL, format_number

st.set_page_config(page_title="Оценка качества — Weather Forecast AI", page_icon="📊", layout="wide")
require_login()
logout_button()

st.title("📊 Оценка качества")
st.markdown(
    "Режимы: `baseline` — классика без LLM, `rag` — выбор модели по кейсам базы знаний, "
    "`agents` — полный цикл агентов, `foundation` — TS Foundation Models (если установлены). "
    "Оценка идёт на holdout-наборе (`data/processed/test_set.jsonl`)."
)
can_run = has_role("editor", "admin")

# ============================================================================
# Запуск оценки (editor/admin)
# ============================================================================
with st.form("eval_form"):
    c1, c2 = st.columns(2)
    modes = c1.multiselect("Режимы", ["baseline", "rag", "agents", "foundation"],
                           default=["baseline", "rag", "agents"])
    horizons = c2.multiselect("Горизонты", [7, 10, 14, 20, 30], default=[7, 14])
    c3, c4 = st.columns(2)
    fractions = c3.multiselect("Доли обучающих данных", [0.1, 0.2, 0.5, 1.0],
                               default=[1.0], format_func=lambda v: f"{v:.1f}")
    num_series = c4.select_slider("Число тестовых окон", [10, 20, 30, 50], value=20)
    fast = st.checkbox("Сокращённая сетка baseline (быстрее)", value=True)
    submitted = st.form_submit_button("▶️ Запустить оценку", disabled=not can_run,
                                      use_container_width=True)

if not can_run:
    st.caption("Запускать оценку могут роли editor/admin — ниже доступны сохранённые результаты.")

if submitted and can_run:
    if not (modes and horizons and fractions):
        st.error("Выберите хотя бы один режим, горизонт и долю данных")
    else:
        accepted = api_client.run_evaluation_batch(modes, horizons, fractions,
                                                   num_series=num_series, fast=fast)
        if accepted and accepted.get("run_id"):
            st.session_state["eval_run_id"] = accepted["run_id"]
            st.rerun()

# ============================================================================
# Результат запуска: опрос статуса
# ============================================================================
run_id = st.session_state.get("eval_run_id")
if run_id:
    with st.spinner("Оценка выполняется…"):
        deadline = time.time() + 300
        run = api_client.get_evaluation_run(run_id)
        while run and run["status"] in ("pending", "running") and time.time() < deadline:
            time.sleep(3)
            run = api_client.get_evaluation_run(run_id)

    if run:
        st.markdown(f"Запуск `{run_id[:8]}` — **{run['status']}**"
                    + (f" · {run['n_records']} записей" if run.get("n_records") else ""))
        if run["status"] == "failed":
            st.error(run.get("error") or "Ошибка выполнения")
        elif run["status"] == "completed":
            summary = run.get("summary") or []
            if summary:
                frame = pd.DataFrame(summary)
                st.dataframe(
                    frame[[c for c in ("mode", "horizon", "data_fraction", "n_series", "mae",
                                       "rmse", "mape", "wape", "mase") if c in frame.columns]],
                    use_container_width=True, hide_index=True,
                )
                left, right = st.columns(2)
                left.plotly_chart(charts.modes_bar_chart(summary, "mae"), use_container_width=True)
                right.plotly_chart(charts.fractions_line_chart(summary, "mae"),
                                   use_container_width=True)

                agents = frame[frame["mode"] == "agents"] if "mode" in frame.columns else frame
                if not agents.empty:
                    st.subheader("Агентные метрики (режим agents)")
                    st.dataframe(
                        agents[[c for c in ("horizon", "data_fraction", "n_series") if c in agents]],
                        use_container_width=True, hide_index=True,
                    )
                    raw = [row.get("agent_metrics") or {} for row in summary
                           if row.get("mode") == "agents"]
                    agent_frame = pd.DataFrame(raw)
                    if not agent_frame.empty:
                        st.dataframe(agent_frame, use_container_width=True, hide_index=True)

            comparisons = run.get("comparisons") or []
            if comparisons:
                st.subheader("Статистическое сравнение режимов")
                rows = []
                for pair in comparisons:
                    sig = pair.get("significance") or {}
                    boot = sig.get("bootstrap") or {}
                    rows.append({
                        "Сравнение": " vs ".join(pair["models_compared"]),
                        "Горизонт": pair["horizon"],
                        "Доля данных": pair["data_fraction"],
                        "Победитель": pair["winner"],
                        "Разность MAE": format_number(boot.get("mean_diff"), ".3f"),
                        "Bootstrap 95% CI": f"[{format_number(boot.get('ci_low'), '.3f')}, "
                                            f"{format_number(boot.get('ci_high'), '.3f')}]",
                        "Wilcoxon p": format_number((sig.get("wilcoxon") or {}).get("pvalue"), ".3f"),
                        "t-test p": format_number((sig.get("paired_ttest") or {}).get("pvalue"), ".3f"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ============================================================================
# Сохранённые результаты (для всех)
# ============================================================================
st.subheader("Сохранённые результаты")
mode_filter = st.selectbox("Режим", [ALL, "baseline", "rag", "agents", "foundation"])
saved = api_client.get_evaluation_results(
    mode=None if mode_filter == ALL else mode_filter, limit=50
)
results = (saved or {}).get("results") or []
if results:
    st.dataframe(pd.DataFrame(results)[
        [c for c in ("mode", "horizon", "data_fraction", "sample_size", "metrics",
                     "agent_metrics", "created_at") if c in results[0]]
    ], use_container_width=True, hide_index=True)
else:
    st.caption("Сохранённых результатов нет.")
