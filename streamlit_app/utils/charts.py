# streamlit_app/utils/charts.py
"""Графики Plotly для страниц (задание MLF, Этап 15: «графики (Plotly)»)."""
import pandas as pd
import plotly.graph_objects as go


def forecast_chart(history_dates, history_values, forecast_dates, forecast_values,
                   title: str = "Прогноз", history_points: int = 90) -> go.Figure:
    """Хвост истории ряда + прогноз на горизонт."""
    figure = go.Figure()
    if history_dates:
        figure.add_trace(go.Scatter(
            x=list(history_dates[-history_points:]), y=list(history_values[-history_points:]),
            name="Факт", mode="lines+markers",
        ))
    if forecast_dates:
        # соединяем с последней фактической точкой, чтобы линия не рвалась
        x = list(forecast_dates)
        y = list(forecast_values)
        if history_dates:
            x = [history_dates[-1]] + x
            y = [history_values[-1]] + y
        figure.add_trace(go.Scatter(x=x, y=y, name="Прогноз", mode="lines+markers",
                                    line=dict(dash="dash")))
    figure.update_layout(
        title=title, xaxis_title="Дата", yaxis_title="Значение",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=50, b=10), height=420,
    )
    return figure


def modes_bar_chart(summary: list, metric: str = "mae") -> go.Figure:
    """MAE (или другая метрика) по режимам, с разбивкой по горизонтам."""
    frame = pd.DataFrame(summary)
    if frame.empty:
        return go.Figure().update_layout(title="Нет данных")
    figure = go.Figure()
    for horizon, group in frame.groupby("horizon"):
        figure.add_trace(go.Bar(
            x=group["mode"], y=pd.to_numeric(group[metric], errors="coerce"),
            name=f"Горизонт {horizon}",
        ))
    figure.update_layout(
        title=f"{metric.upper()} по режимам", barmode="group",
        xaxis_title="Режим", yaxis_title=metric.upper(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=50, b=10), height=380,
    )
    return figure


def fractions_line_chart(summary: list, metric: str = "mae") -> go.Figure:
    """Кривая «точность vs доля обучающих данных»."""
    frame = pd.DataFrame(summary)
    if frame.empty or "data_fraction" not in frame.columns:
        return go.Figure().update_layout(title="Нет данных")
    figure = go.Figure()
    for (mode, horizon), group in frame.groupby(["mode", "horizon"]):
        group = group.sort_values("data_fraction")
        figure.add_trace(go.Scatter(
            x=group["data_fraction"], y=pd.to_numeric(group[metric], errors="coerce"),
            name=f"{mode}, h={horizon}", mode="lines+markers",
        ))
    figure.update_layout(
        title=f"{metric.upper()} vs доля обучающих данных",
        xaxis_title="Доля данных", yaxis_title=metric.upper(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=50, b=10), height=380,
    )
    return figure


def models_bar_chart(results: dict, metric_key: str = "mae") -> go.Figure:
    """Средняя метрика по именованным моделям (/evaluate/compare)."""
    names, means = [], []
    for name, info in (results or {}).items():
        value = (info.get(metric_key) or {}).get("mean")
        if value is not None:
            names.append(name)
            means.append(value)
    figure = go.Figure(go.Bar(x=names, y=means, marker_color="#636efa"))
    figure.update_layout(
        title=f"Средний {metric_key.upper()} по моделям",
        xaxis_title="Модель", yaxis_title=metric_key.upper(),
        margin=dict(l=10, r=10, t=50, b=10), height=360,
    )
    return figure
