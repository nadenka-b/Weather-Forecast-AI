"""Тесты интерфейса Streamlit (Этап 15, методичка Step_12).

Каждая страница прогоняется через AppTest с подменённым API (utils.api_client.api_request),
поэтому тесты не требуют ни backend, ни LLM.
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI_DIR = Path(__file__).resolve().parents[2] / "streamlit_app"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import utils.api_client as api_client  # noqa: E402
from utils import charts, helpers  # noqa: E402

USER = {"email": "user@test.local", "role": "user", "access_token": "tok"}
EDITOR = {"email": "editor@test.local", "role": "editor", "access_token": "tok"}
ADMIN = {"email": "admin@test.local", "role": "admin", "access_token": "tok"}


class FakeAPI:
    """Подмена api_request: маршруты (METHOD, префикс endpoint) -> ответ или функция."""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []

    def __call__(self, method, endpoint, **kwargs):
        self.calls.append((method, endpoint, kwargs))
        # самый специфичный маршрут побеждает (…/{id}/result раньше, чем список /)
        matches = [
            (prefix, response) for (expected_method, prefix), response in self.routes.items()
            if method == expected_method and endpoint.startswith(prefix)
        ]
        if matches:
            prefix, response = max(matches, key=lambda item: len(item[0]))
            return response(endpoint, kwargs) if callable(response) else response
        return None

    def called(self, method, prefix):
        return [call for call in self.calls if call[0] == method and call[1].startswith(prefix)]


@pytest.fixture
def fake_api(monkeypatch):
    fake = FakeAPI()
    monkeypatch.setattr(api_client, "api_request", fake)
    return fake


def run_page(name: str, user: dict | None = None, **session):
    at = AppTest.from_file(str(UI_DIR / name), default_timeout=20)
    if user:
        at.session_state["access_token"] = "tok"
        at.session_state["user"] = user
    for key, value in session.items():
        at.session_state[key] = value
    at.run()
    return at


def values(n=40, seed=0):
    import numpy as np

    return [round(float(v), 4) for v in 12 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + seed]


TASK = {"task_id": "task-0001-abcd", "status": "completed", "hitl_requested": False,
        "iterations": 1, "created_at": "2026-11-25T10:00:00", "finished_at": None,
        "series_id": "jena_temp", "horizon": 7, "model_type": "sarima", "city": "Jena",
        "variable": "temperature", "error": None, "mae": 1.2, "naive_mae": 1.4}
RESULT = {"task_id": "task-0001-abcd", "status": "completed", "model_type": "sarima",
          "model_params": {"order": [1, 1, 1]}, "metrics": {"mae": 1.2, "rmse": 1.5, "mape": 10.0,
                                                           "wape": 8.0, "mase": 0.9},
          "naive_metrics": {"mae": 1.4, "rmse": 1.8, "mape": 12.0, "wape": 9.5, "mase": 1.05},
          "critic_score": 8.0, "critic_reasoning": "хорошо", "validation_passed": True,
          "validation": {"errors": [], "warnings": []}, "predictions": [1.0] * 7,
          "prediction_dates": [f"2026-12-0{i + 1}" for i in range(7)],
          "hitl_requested": False, "critic_suggestions": []}
REASONING = {
    "tree": [{"id": "r1", "agent_name": "supervisor", "status": "success",
              "started_at": "2026-11-25T10:00:00", "finished_at": "2026-11-25T10:00:02",
              "input_summary": "series=jena", "output_summary": "final=True",
              "children": [{"id": "c1", "agent_name": "data_analyst", "status": "success",
                            "children": []}]}],
    "llm_calls": [{"agent_name": "model_selector", "model_name": "qwen", "provider": "lm_studio",
                   "tokens_in": 120, "tokens_out": 18, "latency_ms": 900.0, "status": "success"}],
}


# ============================================================================
# Точка входа: вход / регистрация / навигация
# ============================================================================
class TestAuth:
    def test_anonymous_sees_login(self):
        at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=20).run()
        assert any("Weather Forecast AI" in t.value for t in at.title)
        assert [tab.label for tab in at.tabs] == ["Вход", "Регистрация"]
        assert not at.exception

    def test_login_success(self, fake_api):
        fake_api.routes[("POST", "/auth/login")] = {"access_token": "tok", "email": "a@b.c",
                                                    "role": "user"}
        at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=20).run()
        at.text_input[0].input("a@b.c")
        at.text_input[1].input("StrongPass1")
        at.button[0].click().run()
        assert at.session_state["access_token"] == "tok"
        assert not [e.value for e in at.error]

    def test_login_failure_shows_error(self, fake_api):
        at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=20).run()
        at.text_input[0].input("a@b.c")
        at.text_input[1].input("WrongPass1")
        at.button[0].click().run()
        assert "Неверный email или пароль" in [e.value for e in at.error]
        assert not at.session_state["access_token"]

    def test_register_weak_password_rejected_before_api(self, fake_api):
        at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=20).run()
        at.text_input[2].input("new@b.c")
        at.text_input[3].input("newbie")
        at.text_input[4].input("weak")
        at.button[1].click().run()
        assert any("8 символов" in e.value for e in at.error)
        assert fake_api.calls == []  # клиентская валидация — без запроса

    def test_logout_button_clears_session(self, fake_api):
        at = run_page("main.py", USER)
        assert "Добро пожаловать" in " ".join(m.value for m in at.markdown)
        at.sidebar.button[0].click().run()
        assert not at.session_state["access_token"]
        assert not at.session_state["user"]
        assert any("нужно войти" in e.value for e in at.error)


class TestDashboard:
    def test_requires_login(self, fake_api):
        at = run_page("main.py")
        assert any("нужно войти" in e.value for e in at.error)
        assert not at.title

    def test_links_depend_on_role(self, fake_api):
        fake_api.routes[("GET", "/health")] = {"status": "healthy", "services": {
            "database": {"healthy": True}, "llm": {"healthy": True, "provider": "stub"}}}
        user_links = {link.label for link in run_page("main.py", USER).page_link}
        assert any("HITL" in label for label in user_links) is False  # обычному пользователю скрыто
        editor_links = {link.label for link in run_page("main.py", EDITOR).page_link}
        assert any("HITL" in label for label in editor_links)
        admin_links = {link.label for link in run_page("main.py", ADMIN).page_link}
        assert any("Состояние системы" in label for label in admin_links)


# ============================================================================
# Страницы
# ============================================================================
class TestForecastPage:
    def test_requires_login(self, fake_api):
        at = run_page("pages/1_Forecast.py")
        assert any("нужно войти" in e.value for e in at.error)

    def test_manual_input_creates_task(self, fake_api):
        fake_api.routes[("POST", "/api/v1/forecast/json")] = {"task_id": "new-task", "status": "pending"}
        at = run_page("pages/1_Forecast.py", USER)
        at.text_area[0].input(", ".join(str(v) for v in values(40)))
        at.button(key="m_run").click().run()
        calls = fake_api.called("POST", "/api/v1/forecast/json")
        assert len(calls) == 1
        assert len(calls[0][2]["json"]["series_values"]) == 40
        assert at.session_state["forecast_result"] == "new-task"

    def test_short_series_rejected(self, fake_api):
        at = run_page("pages/1_Forecast.py", USER)
        at.text_area[0].input("1, 2, 3")
        at.button(key="m_run").click().run()
        assert any("30 точек" in e.value for e in at.error)
        assert fake_api.called("POST", "/api/v1/forecast/json") == []

    def test_shows_result_metrics_and_chart(self, fake_api):
        fake_api.routes.update({
            ("GET", "/api/v1/forecast/task-0001-abcd/status"): TASK,
            ("GET", "/api/v1/forecast/task-0001-abcd/result"): RESULT,
            ("GET", "/api/v1/forecast/task-0001-abcd/reasoning"): REASONING,
        })
        at = run_page("pages/1_Forecast.py", USER, forecast_result="task-0001-abcd")
        assert not at.exception
        metric_values = {m.label: m.value for m in at.metric}
        assert metric_values["Модель"] == "sarima"
        assert metric_values["MAE"] == "1.20"
        assert any("наивным" in m.value for m in at.markdown)  # сравнение с наивным прогнозом
        comparison = next(f for f in (d.value for d in at.dataframe)
                          if "Наивный прогноз" in f.columns)
        assert len(comparison) == 5  # все 5 метрик
        assert comparison.iloc[0]["Итог"] == "лучше наива ✅"
        assert any("🧠" in exp.label for exp in at.expander)   # дерево рассуждений
        assert any("supervisor" in exp.label for exp in at.expander)
        assert not [e.value for e in at.error]

    def test_waiting_for_human_shows_warning(self, fake_api):
        fake_api.routes.update({
            ("GET", "/api/v1/forecast/t/status"): {**TASK, "status": "waiting_for_human",
                                                    "hitl_requested": True},
            ("GET", "/api/v1/forecast/t/result"): {**RESULT, "status": "waiting_for_human",
                                                    "hitl_requested": True},
        })
        at = run_page("pages/1_Forecast.py", USER, forecast_result="t")
        assert any("HITL" in w.value for w in at.warning)

    def test_failed_task_shows_error(self, fake_api):
        fake_api.routes[("GET", "/api/v1/forecast/t/status")] = {**TASK, "status": "failed",
                                                                  "error": "model crashed"}
        at = run_page("pages/1_Forecast.py", USER, forecast_result="t")
        assert any("model crashed" in e.value for e in at.error)


class TestMyForecastsPage:
    def test_empty_state(self, fake_api):
        fake_api.routes[("GET", "/api/v1/forecast/")] = {"tasks": []}
        at = run_page("pages/2_My_Forecasts.py", USER)
        assert any("Задач пока нет" in i.value for i in at.info)

    def test_lists_tasks(self, fake_api):
        fake_api.routes[("GET", "/api/v1/forecast/")] = {"tasks": [TASK]}
        fake_api.routes[("GET", "/api/v1/forecast/task-0001-abcd/result")] = RESULT
        at = run_page("pages/2_My_Forecasts.py", USER)
        assert len(at.dataframe[0].value) == 1
        assert at.dataframe[0].value.iloc[0]["Модель"] == "sarima"
        assert at.dataframe[0].value.iloc[0]["MAE"] == "1.20"      # метрики в списке
        assert at.dataframe[0].value.iloc[0]["MAE наива"] == "1.40"


class TestReasoningPage:
    def test_renders_tree_and_llm_calls(self, fake_api):
        fake_api.routes[("GET", "/api/v1/forecast/")] = {"tasks": [TASK]}
        fake_api.routes[("GET", "/api/v1/forecast/task-0001-abcd/reasoning")] = REASONING
        at = run_page("pages/3_Reasoning.py", USER, reasoning_task_id="task-0001-abcd")
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Шагов агентов"] == "2"      # supervisor + data_analyst
        assert metrics["Вызовов LLM"] == "1"
        assert metrics["Токены (вход)"] == "120"
        labels = [exp.label for exp in at.expander]
        assert any("supervisor" in label for label in labels)
        assert any("data_analyst" in label for label in labels)


class TestEvaluationPage:
    def test_plain_user_cannot_run(self, fake_api):
        at = run_page("pages/4_Evaluation.py", USER)
        assert fake_api.called("POST", "/api/v1/evaluate/batch") == []
        assert not at.exception

    def test_completed_run_renders_tables(self, fake_api):
        fake_api.routes[("GET", "/api/v1/evaluate/runs/run-1")] = {
            "run_id": "run-1", "status": "completed", "n_records": 6,
            "summary": [
                {"mode": "baseline", "horizon": 7, "data_fraction": 1.0, "n_series": 2,
                 "mae": 4.0, "rmse": 4.5, "mape": 100.0, "wape": 28.0, "mase": 0.97},
                {"mode": "agents", "horizon": 7, "data_fraction": 1.0, "n_series": 2,
                 "mae": 3.6, "rmse": 4.1, "mape": 120.0, "wape": 27.0, "mase": 0.9,
                 "agent_metrics": {"mean_iterations": 1.0, "hitl_share": 1.0}},
            ],
            "comparisons": [{
                "models_compared": ["agents", "baseline"], "horizon": 7, "data_fraction": 1.0,
                "winner": "agents", "n_series": 2,
                "significance": {"wilcoxon": {"pvalue": 0.18},
                                 "bootstrap": {"mean_diff": -0.4, "ci_low": -0.9, "ci_high": -0.02},
                                 "paired_ttest": {"pvalue": 0.07}},
            }],
        }
        fake_api.routes[("GET", "/api/v1/evaluate/results")] = {"results": []}
        at = run_page("pages/4_Evaluation.py", EDITOR, eval_run_id="run-1")
        assert not at.exception
        assert "agents" in at.dataframe[0].value["mode"].tolist()
        pairs = at.dataframe[-1].value
        assert pairs.iloc[0]["Победитель"] == "agents"

    def test_failed_run_shows_error(self, fake_api):
        fake_api.routes[("GET", "/api/v1/evaluate/runs/run-2")] = {
            "run_id": "run-2", "status": "failed", "error": "no test set"}
        at = run_page("pages/4_Evaluation.py", EDITOR, eval_run_id="run-2")
        assert any("no test set" in e.value for e in at.error)


class TestModelComparePage:
    def test_requires_editor(self, fake_api):
        at = run_page("pages/5_Model_Compare.py", USER)
        assert any("editor" in e.value for e in at.error)

    def test_compare_renders_winner(self, fake_api):
        fake_api.routes[("GET", "/api/v1/evaluate/compare")] = {
            "winner": "arima",
            "results": {"naive": {"model_type": "naive_seasonal", "mae": {"mean": 5.0, "median": 5.0, "n": 10}},
                        "arima": {"model_type": "arima", "mae": {"mean": 3.0, "median": 3.0, "n": 10}}},
            "pairs": [{"models_compared": ["arima", "naive"], "winner": "arima",
                       "wilcoxon": {"pvalue": 0.01},
                       "bootstrap": {"mean_diff": -2.0, "ci_low": -3.0, "ci_high": -1.0}}],
        }
        at = run_page("pages/5_Model_Compare.py", EDITOR)
        at.multiselect[0].select(["naive", "arima"])
        at.form_submit_button[0].click().run()
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Лучшая модель"] == "arima"
        assert "arima" in at.dataframe[0].value["Модель"].tolist()


class TestHitlPage:
    def test_requires_editor(self, fake_api):
        at = run_page("pages/6_HITL_Queue.py", USER)
        assert any("editor" in e.value for e in at.error)

    def test_queue_and_confirm(self, fake_api):
        item = {"task_id": "task-9", "status": "waiting_for_human", "city": "Jena",
                "variable": "temperature", "model_type": "arima", "critic_score": 5.0,
                "critic_reasoning": "сезонность", "critic_suggestions": ["проверить сезонность"],
                "validator_errors": [], "stopped_reason": "no_change", "iterations": 1}
        fake_api.routes.update({
            ("GET", "/hitl/stats"): {"pending": 1, "confirmed": 0, "rejected": 0},
            ("GET", "/hitl/queue"): {"count": 1, "examples": [item]},
            ("POST", "/hitl/task-9/confirm"): {"task_id": "task-9", "status": "completed",
                                               "kb_inserted": True},
            ("GET", "/hitl/feedback"): {"feedback": []},
        })
        at = run_page("pages/6_HITL_Queue.py", EDITOR)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics == {"В очереди": "1", "Подтверждено": "0", "Отклонено": "0"}
        assert any("Critic: 5.00" in exp.label for exp in at.expander)

        at.button(key="ok_task-9").click().run()
        assert len(fake_api.called("POST", "/hitl/task-9/confirm")) == 1

    def test_empty_queue(self, fake_api):
        fake_api.routes.update({
            ("GET", "/hitl/stats"): {"pending": 0, "confirmed": 0, "rejected": 0},
            ("GET", "/hitl/queue"): {"count": 0, "examples": []},
            ("GET", "/hitl/feedback"): {"feedback": []},
        })
        at = run_page("pages/6_HITL_Queue.py", EDITOR)
        assert any("Очередь пуста" in s.value for s in at.success)


class TestKnowledgePage:
    def test_requires_login(self, fake_api):
        at = run_page("pages/7_Knowledge_Base.py")
        assert any("нужно войти" in e.value for e in at.error)

    def test_search_shows_results(self, fake_api):
        fake_api.routes[("GET", "/kb/search")] = {"results": [{
            "series_id": "jena|temperature|2013", "score": 3.2, "best_model": "sarima",
            "city": "Jena", "variable": "temperature", "metrics": {"mase": 0.8},
            "hyperparams": {"order": [1, 1, 1]}, "description": "Дневной ряд температуры",
        }]}
        at = run_page("pages/7_Knowledge_Base.py", USER)
        at.text_input[0].input("температура Jena").run()
        assert any("sarima" in exp.label for exp in at.expander)

    def test_admin_sees_add_form(self, fake_api):
        fake_api.routes[("GET", "/kb/cases")] = {"total": 0, "cases": []}
        fake_api.routes[("GET", "/glossary/terms")] = {"total": 0, "categories": [], "terms": []}
        at = run_page("pages/7_Knowledge_Base.py", ADMIN)
        assert any("Добавить кейс" in exp.label for exp in at.expander)
        assert not at.exception


class TestHealthPage:
    def test_requires_admin(self, fake_api):
        at = run_page("pages/8_System_Health.py", EDITOR)
        assert any("admin" in e.value for e in at.error)

    def test_renders_services_and_cache(self, fake_api):
        fake_api.routes.update({
            ("GET", "/health"): {"status": "healthy", "services": {
                "database": {"healthy": True}, "llm": {"healthy": True, "provider": "stub"},
                "tasks": {"healthy": True, "backend": "inline"}}},
            ("GET", "/admin/cache/stats"): {"entries": 12, "total_hits": 34},
            ("GET", "/kb/cases"): {"total": 1000, "cases": []},
            ("GET", "/glossary/terms"): {"total": 578, "categories": [], "terms": []},
        })
        at = run_page("pages/8_System_Health.py", ADMIN)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Общий статус"] == "healthy"
        assert metrics["Записей"] == "12"
        assert metrics["Кейсов в базе знаний"] == "1000"
        assert not at.exception


# ============================================================================
# Утилиты
# ============================================================================
class TestUtils:
    def test_parse_series(self):
        assert helpers.parse_series("1, 2\n3 4;5") == [1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.raises(ValueError):
            helpers.parse_series("1, abc")

    def test_csv_to_series(self):
        content = b"date,value\n2024-01-01,1.5\n2024-01-02,2.5\n"
        values, dates = helpers.csv_to_series(content)
        assert values == [1.5, 2.5] and len(dates) == 2
        only_values, no_dates = helpers.csv_to_series(b"3\n4\n")
        assert only_values == [3.0, 4.0] and no_dates is None

    def test_formatting(self):
        assert helpers.format_number(None) == "N/A"
        assert helpers.format_number(1.23456) == "1.23"
        assert helpers.format_score(None) == "нет оценки"
        assert helpers.filter_value("All") is None and helpers.filter_value("Jena") == "Jena"
        assert helpers.short("x" * 70, 10) == "x" * 10 + "..."
        assert helpers.status_label("completed") == "✅ Завершена"

    def test_validate_password(self):
        assert helpers.validate_password("StrongPass1") is None
        assert helpers.validate_password("short1A") is not None
        assert helpers.validate_password("longpassword") is not None
        assert helpers.validate_password("longpassword1") is not None  # нет заглавной

    def test_charts(self):
        figure = charts.forecast_chart(["d1", "d2"], [1, 2], ["d3"], [3.0])
        assert len(figure.data) == 2 and figure.data[1].y[-1] == 3.0
        assert len(charts.modes_bar_chart([{"mode": "a", "horizon": 7, "mae": 1.0}]).data) == 1
        assert len(charts.fractions_line_chart(
            [{"mode": "a", "horizon": 7, "data_fraction": 0.5, "mae": 1.0}]).data) == 1
        assert len(charts.models_bar_chart({"a": {"mae": {"mean": 1.0}}}).data) == 1


class TestApiClient:
    def test_extract_detail_variants(self):
        class Response:
            def __init__(self, payload=None, text=""):
                self._payload, self.text = payload, text

            def json(self):
                if self._payload is None:
                    raise ValueError("no json")
                return self._payload

        assert api_client._extract_detail(Response({"detail": "Not authenticated"})) == "Not authenticated"
        assert api_client._extract_detail(Response({"detail": [
            {"loc": ["body", "password"], "msg": "too short"}]})) == "body.password: too short"
        assert api_client._extract_detail(Response(None, "raw text")) == "raw text"

    def test_health_503_is_parsed_not_error(self, monkeypatch):
        import requests

        class Response:
            status_code = 503
            content = b"{}"
            text = "{}"

            def json(self):
                return {"status": "unhealthy", "services": {"database": {"healthy": False}}}

            def raise_for_status(self):
                raise requests.exceptions.HTTPError(response=self)

        monkeypatch.setattr(requests, "request", lambda *a, **k: Response())

        def page():
            import streamlit as st

            import utils.api_client as api_client
            result = api_client.api_request("GET", "/health")
            st.write(f"status={result.get('status')}")

        at = AppTest.from_function(page).run()
        assert any("status=unhealthy" in m.value for m in at.markdown)
        assert not at.error

    def test_unauthorized_clears_session(self, monkeypatch):
        import requests

        class Response:
            status_code = 401
            content = b'{"detail": "Token expired"}'
            text = '{"detail": "Token expired"}'

            def json(self):
                return {"detail": "Token expired"}

            def raise_for_status(self):
                raise requests.exceptions.HTTPError(response=self)

        monkeypatch.setattr(requests, "request", lambda *a, **k: Response())

        def page():
            import streamlit as st

            import utils.api_client as api_client
            st.session_state["access_token"] = "expired"
            st.session_state["user"] = {"role": "user"}
            st.session_state["forecast_result"] = "stale"
            api_client.get_forecast_status("t")
            st.write(f"token={st.session_state['access_token']}")

        at = AppTest.from_function(page).run()
        assert "API Error (401): Token expired" in [e.value for e in at.error]
        assert any("token=None" in m.value for m in at.markdown)
        assert "forecast_result" not in at.session_state  # эфемерный ключ удалён

    def test_unreachable_api_hint(self, monkeypatch):
        import requests

        def boom(*args, **kwargs):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "request", boom)

        def page():
            import utils.api_client as api_client
            api_client.get_healthz()

        at = AppTest.from_function(page).run()
        assert any("API недоступен" in e.value for e in at.error)
