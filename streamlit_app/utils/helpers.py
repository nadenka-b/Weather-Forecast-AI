# streamlit_app/utils/helpers.py
"""Утилиты отображения (методичка Step_12, п. 12.4) и разбор входных данных."""
import io
from typing import Optional

import pandas as pd
import streamlit as st

from utils import constants
from utils.constants import ALL  # noqa: F401  (реэкспорт для страниц)


def filter_value(label: Optional[str]) -> Optional[str]:
    """«All» (sentinel фильтра) -> None («не фильтровать» для API)."""
    return None if not label or label == constants.ALL else label


def short(text: str, n: int = 60) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "..."


def format_number(value, spec: str = ".2f") -> str:
    """Число по формату; «N/A», если метрика не посчитана (None)."""
    if value is None:
        return "N/A"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def format_score(score) -> str:
    return "нет оценки" if score is None else f"{score:.2f}"


def status_label(status: Optional[str]) -> str:
    return constants.TASK_STATUSES.get(status or "", status or "—")


def render_metrics(metrics: dict, keys=("mae", "rmse", "mape", "wape", "mase"),
                   labels: Optional[dict] = None, columns: int = 5) -> None:
    """Метрики точности как st.metric; отсутствующие — серые «N/A»."""
    metrics = metrics or {}
    labels = labels or {"mae": "MAE", "rmse": "RMSE", "mape": "MAPE, %", "wape": "WAPE, %",
                        "mase": "MASE"}
    cols = st.columns(min(columns, len(keys)))
    for col, key in zip(cols, keys):
        col.metric(labels.get(key, key), format_number(metrics.get(key)))


def render_tree(nodes: list, depth: int = 0) -> None:
    """Пошаговое дерево рассуждений агентов (supervisor -> агенты -> tools)."""
    for node in nodes or []:
        icon = {"success": "✅", "error": "❌", "running": "🔄", "skipped": "⏭"}.get(
            node.get("status"), "•")
        title = f"{icon} **{node.get('agent_name')}**"
        if node.get("tool_name"):
            title += f" · `{node['tool_name']}`"
        duration = ""
        if node.get("started_at") and node.get("finished_at"):
            duration = f" · {_duration_ms(node['started_at'], node['finished_at'])}"
        body = ""
        if node.get("input_summary"):
            body += f"\n\n_Вход:_ {node['input_summary']}"
        if node.get("output_summary"):
            body += f"\n\n_Выход:_ {node['output_summary']}"
        with st.expander(title, expanded=depth == 0):
            if body:
                st.markdown(body.strip())
            if duration:
                st.caption(duration)
            if not body and not duration:
                st.caption("нет деталей")
            if node.get("children"):
                render_tree(node["children"], depth + 1)


def _duration_ms(start: str, finish: str) -> str:
    try:
        delta = pd.Timestamp(finish) - pd.Timestamp(start)
        return f"{max(0.0, delta.total_seconds() * 1000):.0f} мс"
    except (ValueError, TypeError):
        return ""


def render_llm_calls(calls: list) -> None:
    """Трассировка вызовов LLM: таблица + суммарные токены."""
    if not calls:
        st.caption("Вызовы LLM отсутствуют (режим stub или LLM не использовалась).")
        return
    frame = pd.DataFrame(calls)
    st.dataframe(
        frame[["agent_name", "model_name", "provider", "tokens_in", "tokens_out",
               "latency_ms", "status"]],
        use_container_width=True, hide_index=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Вызовов LLM", len(calls))
    c2.metric("Токенов на входе", int(frame["tokens_in"].fillna(0).sum()))
    c3.metric("Токенов на выходе", int(frame["tokens_out"].fillna(0).sum()))


def parse_series(text: str) -> list:
    """Ряд из текста: числа через запятую, пробел или новую строку."""
    tokens = text.replace("\n", " ").replace(";", " ").split(",")
    values = []
    for chunk in tokens:
        values.extend(chunk.split())
    return [float(v.strip()) for v in values if v.strip()]


def csv_to_series(content: bytes) -> tuple:
    """CSV ряда (date,value) или (value) -> (values, dates|None).

    Заголовок определяется по первой строке: если все её ячейки — числа,
    заголовка нет (иначе pd.read_csv съест первую точку ряда).
    """
    frame = pd.read_csv(io.BytesIO(content), header=_sniff_header(content))
    if frame.shape[1] >= 2:
        return frame.iloc[:, 1].astype(float).tolist(), frame.iloc[:, 0].astype(str).tolist()
    return frame.iloc[:, 0].astype(float).tolist(), None


def validate_password(password: str) -> Optional[str]:
    """Зеркалит серверную валидацию (мгновенная обратная связь до отправки)."""
    if len(password) < 8:
        return "Пароль: минимум 8 символов"
    if not any(c.isdigit() for c in password):
        return "Пароль: нужна хотя бы одна цифра"
    if not any(c.isupper() for c in password):
        return "Пароль: нужна заглавная буква"
    if len(password.encode("utf-8")) > 72:
        return "Пароль: не более 72 байт (ограничение bcrypt)"
    return None


def _sniff_header(content: bytes) -> int | None:
    """0 — первая строка является заголовком, None — заголовка нет."""
    first_line = content.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace")
    for delimiter in (";", ",", "\t"):
        if delimiter in first_line:
            cells = first_line.split(delimiter)
            break
    else:
        cells = [first_line]
    try:
        [float(cell.strip()) for cell in cells if cell.strip() != ""]
    except ValueError:
        return 0  # есть нечисловая ячейка — это заголовок
    return None
