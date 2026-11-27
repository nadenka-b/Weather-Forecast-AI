# streamlit_app/utils/constants.py
"""Справочники для UI — копии констант backend (методичка Step_12, п. 12.1).

Streamlit не импортирует backend/ (запускается отдельно на хосте, backend —
отдельный процесс). Если добавляете переменную или город в backend —
синхронизируйте здесь.

Источник в backend: KNOWN_VARIABLES (data/embeddings.py), MODES
(evaluation/runner.py), _MODEL_SPECS (evaluation/service.py).
"""

# Код переменной -> название (как в features кейсов базы знаний)
VARIABLES = {
    "temperature": "Температура",
    "humidity": "Влажность",
    "pressure": "Давление",
    "wind_speed": "Скорость ветра",
    "precipitation": "Осадки",
    "dew_point": "Точка росы",
}

# Города датасетов (Jena Climate, Daily Climate Delhi). NO_CITY — ряд без привязки
# к городу: в API уходит city=None, фильтр города в RAG отключается и поиск
# похожих кейсов идёт по всей базе знаний.
NO_CITY = "Другой / без города"
CITIES = ["Jena", "Delhi", NO_CITY]

# Именованные модели для GET /api/v1/evaluate/compare
MODELS = ["naive", "arima", "sarima", "xgboost", "prophet"]

# Режимы оценки (backend/evaluation/runner.py: MODES)
EVAL_MODES = ["baseline", "rag", "agents", "foundation"]

# Sentinel для фильтров «не фильтровать» (helpers.filter_value)
ALL = "All"

# Статусы задачи прогноза (задание MLF)
TASK_STATUSES = {
    "pending": "⏳ В очереди",
    "running": "🔄 Выполняется",
    "waiting_for_human": "🧑‍⚖️ Ждёт решения эксперта",
    "completed": "✅ Завершена",
    "failed": "❌ Ошибка",
    "rejected": "🚫 Отклонена",
}
