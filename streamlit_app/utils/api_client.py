# streamlit_app/utils/api_client.py
"""Единый клиент API для всех страниц Streamlit (методичка Step_12, п. 12.2).

- API_URL из st.secrets (.streamlit/secrets.toml), fallback http://localhost:8000;
- 503 от /health — диагностический ответ (парсим JSON, а не ошибка);
- 401 — сброс сессии (access_token, user, эфемерные ключи *_result/_data/_key);
- detail и строка (HTTPException), и список (Pydantic 422) разворачиваются в текст;
- для файлов Content-Type не ставится (requests сам формирует multipart).

Streamlit не импортирует backend/ — только HTTP.
"""
import requests
import streamlit as st
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"

_EPHEMERAL_SUFFIXES = ("_result", "_data", "_key")


# ============================================================================
# Базовый запрос
# ============================================================================
def get_headers() -> Dict[str, str]:
    token = st.session_state.get("access_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _extract_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:500]

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in item.get("loc", []))
                msg = item.get("msg", "")
                parts.append(f"{loc}: {msg}" if loc else msg)
            else:
                parts.append(str(item))
        return "; ".join(p for p in parts if p)
    return ""


def api_request(method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
    """HTTP-запрос к API; при ошибке показывает st.error и возвращает None."""
    url = f"{API_URL}{endpoint}"
    headers = get_headers()
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    if "files" not in kwargs:
        headers.setdefault("Content-Type", "application/json")

    try:
        response = requests.request(method, url, headers=headers, **kwargs)

        # /health отдаёт 503 при недоступности сервиса — это диагностика, не ошибка
        if endpoint == "/health" and response.status_code == 503:
            try:
                return response.json()
            except ValueError:
                return {}

        response.raise_for_status()
        return response.json() if response.content else {}

    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else "??"
        detail = _extract_detail(e.response) if e.response is not None else ""
        message = detail or str(e)
        if e.response is None:  # API недоступен — подсказка, как запустить
            message = (
                "❌ API недоступен. Запустите backend: poetry run python run.py "
                f"(должен отвечать на {API_URL})."
            )
        st.error(f"API Error ({status}): {message}")

        if e.response is not None and e.response.status_code == 401:
            _clear_session_state()
        return None


def _clear_session_state() -> None:
    st.session_state.access_token = None
    st.session_state.user = None
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.endswith(_EPHEMERAL_SUFFIXES):
            del st.session_state[key]


# ============================================================================
# Аутентификация
# ============================================================================
def login(email: str, password: str) -> bool:
    result = api_request("POST", "/auth/login", json={"email": email, "password": password})
    if result and "access_token" in result:
        st.session_state.access_token = result["access_token"]
        st.session_state.user = result
        return True
    return False


def register(email: str, username: str, password: str, full_name: str = "") -> bool:
    result = api_request(
        "POST", "/auth/register",
        json={"email": email, "username": username, "password": password,
              "full_name": full_name},
    )
    if result and "access_token" in result:
        st.session_state.access_token = result["access_token"]
        st.session_state.user = result
        return True
    return False


def logout():
    _clear_session_state()


def get_me() -> Optional[Dict]:
    return api_request("GET", "/auth/me")


# ============================================================================
# Прогноз
# ============================================================================
def create_forecast_json(
    series_values: List[float],
    series_dates: Optional[List[str]] = None,
    horizon: int = 14,
    freq: str = "D",
    city: Optional[str] = None,
    variable: Optional[str] = None,
) -> Optional[Dict]:
    payload: Dict[str, Any] = {
        "series_values": series_values, "horizon": horizon, "freq": freq,
        "city": city, "variable": variable,
    }
    if series_dates:
        payload["series_dates"] = series_dates
    return api_request("POST", "/api/v1/forecast/json", json=payload)


def create_forecast_file(file, horizon: int = 14, freq: str = "D",
                         city: Optional[str] = None, variable: Optional[str] = None) -> Optional[Dict]:
    data: Dict[str, Any] = {"horizon": str(horizon), "freq": freq}
    if city:
        data["city"] = city
    if variable:
        data["variable"] = variable
    return api_request(
        "POST", "/api/v1/forecast/",
        files={"file": (file.name, file.getvalue(), "text/csv")},
        data=data,
    )


def list_my_forecasts(limit: int = 20) -> Optional[Dict]:
    return api_request("GET", "/api/v1/forecast/", params={"limit": limit})


def get_forecast_status(task_id: str) -> Optional[Dict]:
    return api_request("GET", f"/api/v1/forecast/{quote(task_id, safe='')}/status")


def get_forecast_result(task_id: str) -> Optional[Dict]:
    return api_request("GET", f"/api/v1/forecast/{quote(task_id, safe='')}/result")


def get_forecast_reasoning(task_id: str) -> Optional[Dict]:
    return api_request("GET", f"/api/v1/forecast/{quote(task_id, safe='')}/reasoning")


def resolve_hitl(task_id: str, action: str, model_type: Optional[str] = None,
                 model_params: Optional[Dict] = None, horizon: Optional[int] = None) -> Optional[Dict]:
    payload: Dict[str, Any] = {"action": action}
    if model_type:
        payload["model_type"] = model_type
    if model_params:
        payload["model_params"] = model_params
    if horizon:
        payload["horizon"] = horizon
    return api_request("POST", f"/api/v1/forecast/{quote(task_id, safe='')}/hitl", json=payload)


# ============================================================================
# Оценка качества
# ============================================================================
def compare_models(models: List[str], sample_size: int = 20, horizon: int = 14) -> Optional[Dict]:
    return api_request(
        "GET", "/api/v1/evaluate/compare",
        params={"models": ",".join(models), "sample_size": sample_size, "horizon": horizon},
    )


def run_evaluation_batch(
    modes: List[str],
    horizons: List[int],
    data_fractions: List[float],
    num_series: int = 10,
    fast: bool = True,
) -> Optional[Dict]:
    return api_request(
        "POST", "/api/v1/evaluate/batch",
        json={"modes": modes, "horizons": horizons, "data_fractions": data_fractions,
              "num_series": num_series, "fast": fast},
    )

def get_evaluation_run(run_id: str) -> Optional[Dict]:
    return api_request("GET", f"/api/v1/evaluate/runs/{quote(run_id, safe='')}")


def get_evaluation_results(mode: Optional[str] = None, limit: int = 50) -> Optional[Dict]:
    params: Dict[str, Any] = {"limit": limit}
    if mode:
        params["mode"] = mode
    return api_request("GET", "/api/v1/evaluate/results", params=params)


def get_evaluation_comparisons(limit: int = 50) -> Optional[Dict]:
    return api_request("GET", "/api/v1/evaluate/comparisons", params={"limit": limit})


# ============================================================================
# HITL
# ============================================================================
def get_hitl_queue(city: Optional[str] = None, variable: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Optional[Dict]:
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if city:
        params["city"] = city
    if variable:
        params["variable"] = variable
    return api_request("GET", "/hitl/queue", params=params)


def get_hitl_stats() -> Optional[Dict]:
    return api_request("GET", "/hitl/stats")


def confirm_hitl(task_id: str, model_type: Optional[str] = None,
                 model_params: Optional[Dict] = None, comment: Optional[str] = None) -> Optional[Dict]:
    payload: Dict[str, Any] = {}
    if model_type:
        payload["model_type"] = model_type
    if model_params:
        payload["model_params"] = model_params
    if comment:
        payload["comment"] = comment
    return api_request("POST", f"/hitl/{quote(task_id, safe='')}/confirm", json=payload)


def reject_hitl(task_id: str, reason: Optional[str] = None) -> Optional[Dict]:
    return api_request(
        "POST", f"/hitl/{quote(task_id, safe='')}/reject", json={"reason": reason},
    )


def save_correction(task_id: str, correction: Optional[Dict] = None,
                    comment: Optional[str] = None, add_to_kb: bool = False) -> Optional[Dict]:
    return api_request(
        "POST", "/hitl/feedback",
        json={"task_id": task_id, "correction": correction, "comment": comment,
              "add_to_kb": add_to_kb},
    )


def list_feedback(task_id: Optional[str] = None, limit: int = 50) -> Optional[Dict]:
    params: Dict[str, Any] = {"limit": limit}
    if task_id:
        params["task_id"] = task_id
    return api_request("GET", "/hitl/feedback", params=params)


# ============================================================================
# База знаний и справочник
# ============================================================================
def search_kb(query: str, top_k: int = 10) -> Optional[Dict]:
    return api_request("GET", "/kb/search", params={"q": query, "top_k": top_k})


def list_kb_cases(city: Optional[str] = None, variable: Optional[str] = None,
                  best_model: Optional[str] = None, search: Optional[str] = None,
                  limit: int = 50, offset: int = 0) -> Optional[Dict]:
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if city:
        params["city"] = city
    if variable:
        params["variable"] = variable
    if best_model:
        params["best_model"] = best_model
    if search:
        params["search"] = search
    return api_request("GET", "/kb/cases", params=params)


def add_kb_case(case: Dict) -> Optional[Dict]:
    return api_request("POST", "/admin/kb/add", json=case)


def delete_kb_case(case_id: str) -> Optional[Dict]:
    return api_request("DELETE", f"/admin/kb/delete/{quote(case_id, safe='')}")


def reindex_kb() -> Optional[Dict]:
    return api_request("POST", "/admin/kb/reindex")


def list_terms(category: Optional[str] = None, search: Optional[str] = None,
               limit: int = 50, offset: int = 0) -> Optional[Dict]:
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if category:
        params["category"] = category
    if search:
        params["search"] = search
    return api_request("GET", "/glossary/terms", params=params)


def add_term(term: str, definition: str, category: Optional[str] = None) -> Optional[Dict]:
    return api_request("POST", "/admin/glossary/add",
                       json={"term": term, "definition": definition, "category": category})


def delete_term(term_id: str) -> Optional[Dict]:
    return api_request("DELETE", f"/admin/glossary/delete/{quote(term_id, safe='')}")


def import_terms_csv(file) -> Optional[Dict]:
    return api_request("POST", "/admin/glossary/import-csv",
                       files={"file": (file.name, file.getvalue(), "text/csv")})


# ============================================================================
# Кэш и здоровье сервисов
# ============================================================================
def get_cache_stats() -> Optional[Dict]:
    return api_request("GET", "/admin/cache/stats")


def clear_cache(older_than_days: Optional[int] = None, max_hits: Optional[int] = None) -> Optional[Dict]:
    params: Dict[str, Any] = {}
    if older_than_days is not None:
        params["older_than_days"] = older_than_days
    if max_hits is not None:
        params["max_hits"] = max_hits
    return api_request("DELETE", "/admin/cache/clear", params=params)


def get_health() -> Optional[Dict]:
    return api_request("GET", "/health")


def get_healthz() -> Optional[Dict]:
    return api_request("GET", "/healthz")
