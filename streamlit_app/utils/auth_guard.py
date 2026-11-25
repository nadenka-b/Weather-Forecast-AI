# streamlit_app/utils/auth_guard.py
"""Проверка авторизации и ролей для страниц (методичка Step_12, п. 12.3).

- require_login / require_role вызываются в начале каждой защищённой страницы;
- st.stop() — единственный способ прервать top-level скрипт страницы;
- сессия живёт в st.session_state (access_token, user); токен дублируется
  в query params, чтобы пережить F5 (UI локальный — допустимо, как в референсе).
"""
import streamlit as st

from utils import api_client

_SESSION_KEYS = ("access_token", "user")


def current_user() -> dict:
    return st.session_state.get("user") or {}


def has_role(*allowed_roles: str) -> bool:
    """True, если роль пользователя входит в allowed_roles (без остановки страницы)."""
    role = current_user().get("role", "user")
    return role in allowed_roles


def require_login() -> dict:
    """Останавливает страницу, если пользователь не вошёл."""
    user = current_user()
    if not user:
        st.error("🔒 Для доступа к этой странице нужно войти.")
        st.page_link("main.py", label="→ На главную (вход)", icon="🚀")
        st.stop()
    return user


def require_role(*allowed_roles: str) -> dict:
    """Сначала логин, затем проверка роли (каскад: «войдите» → «нет прав»)."""
    user = require_login()
    role = user.get("role", "user")
    if role not in allowed_roles:
        st.error(f"🔒 Эта страница требует одну из ролей: {', '.join(allowed_roles)}.")
        st.page_link("main.py", label="→ На главную", icon="🚀")
        st.stop()
    return user


def logout_button() -> None:
    """Кнопка «Выйти» в сайдбаре (JWT stateless — сервер не хранит сессии)."""
    if st.session_state.get("access_token"):
        if st.sidebar.button("Выйти", use_container_width=True):
            drop_session()
            st.rerun()


def save_session(token_response: dict) -> None:
    """Сохраняет токен; роль/имя — из ответа (TokenResponse) или /auth/me."""
    st.session_state["access_token"] = token_response["access_token"]
    st.session_state["user"] = token_response
    st.query_params["token"] = token_response["access_token"]


def drop_session() -> None:
    """Полный сброс сессии (session_state + query params + эфемерные ключи)."""
    for key in _SESSION_KEYS:
        st.session_state[key] = None
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.endswith(api_client._EPHEMERAL_SUFFIXES):
            del st.session_state[key]
    st.query_params.clear()


def restore_session() -> None:
    """Восстанавливает сессию после перезагрузки страницы (токен из query params)."""
    if st.session_state.get("access_token"):
        return
    token = st.query_params.get("token")
    if not token:
        return
    st.session_state["access_token"] = token
    me = api_client.get_me()
    if me and me.get("id"):
        st.session_state["user"] = me
    else:
        drop_session()  # токен истёк или отозван
