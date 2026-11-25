# streamlit_app/app.py
"""
Weather Forecast AI — роутер UI (Этап 15, методичка Step_12).

До входа — форма входа/регистрации. После входа — навигация, вид которой
зависит от роли:
- user:   Главная, Прогноз, Мои прогнозы, Логи рассуждений, Оценка качества,
          База знаний;
- editor: + Сравнение моделей, HITL-очередь;
- admin:  + Состояние системы.

Важно (Streamlit >= 1.36): st.navigation() только регистрирует страницы,
скрипт текущей страницы нужно запустить самому через .run() у возвращённой
страницы — иначе после входа будет пустой экран.

Запуск (отдельно от Docker, на хосте — методичка Step_12, п. 12.0):
    pip install -r streamlit_app/requirements.txt
    streamlit run streamlit_app/app.py --server.port 8501
"""

import streamlit as st

from utils import api_client
from utils.auth_guard import drop_session, restore_session, save_session
from utils.helpers import validate_password

st.set_page_config(
    page_title="Weather Forecast AI",
    page_icon="🌤",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in (("access_token", None), ("user", None)):
    st.session_state.setdefault(key, default)


def render_auth():
    """Форма входа/регистрации (2 таба)."""
    st.title("🌤 Weather Forecast AI")
    st.markdown(
        "Интеллектуальная система прогнозирования погодных временных рядов с "
        "мультиагентной архитектурой, Agentic RAG и контуром экспертской проверки (HITL)."
    )

    tab_login, tab_register = st.tabs(["Вход", "Регистрация"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", use_container_width=True)
        if submitted:
            if api_client.login(email, password):
                save_session(st.session_state["user"])
                st.rerun()
            else:
                st.error("Неверный email или пароль")

    with tab_register:
        with st.form("register_form"):
            email = st.text_input("Email ", key="reg_email")
            username = st.text_input("Имя пользователя (мин. 3 символа)")
            password = st.text_input(
                "Пароль (мин. 8 символов, цифра и заглавная буква)", type="password",
            )
            submitted = st.form_submit_button("Зарегистрироваться", use_container_width=True)
        if submitted:
            # клиентская проверка — мгновенная обратная связь до отправки
            problem = validate_password(password)
            if problem:
                st.error(problem)
            elif len(username.strip()) < 3:
                st.error("Имя пользователя: минимум 3 символа")
            elif api_client.register(email.strip(), username.strip(), password):
                save_session(st.session_state["user"])
                st.success("Регистрация успешна")
                st.rerun()


def build_navigation():
    """Навигация с учётом роли (st.navigation отключает авто-список pages/)."""
    role = (st.session_state.get("user") or {}).get("role", "user")

    pages = [
        st.Page("main.py", title="Главная", icon="🌤", default=True),
        st.Page("pages/1_Forecast.py", title="Прогноз", icon="📈"),
        st.Page("pages/2_My_Forecasts.py", title="Мои прогнозы", icon="📁"),
        st.Page("pages/3_Reasoning.py", title="Логи рассуждений", icon="🧠"),
        st.Page("pages/4_Evaluation.py", title="Оценка качества", icon="📊"),
        st.Page("pages/7_Knowledge_Base.py", title="База знаний", icon="🔍"),
    ]
    if role in ("editor", "admin"):
        pages.append(st.Page("pages/5_Model_Compare.py", title="Сравнение моделей", icon="⚖️"))
        pages.append(st.Page("pages/6_HITL_Queue.py", title="HITL-очередь", icon="🛠"))
    if role == "admin":
        pages.append(st.Page("pages/8_System_Health.py", title="Состояние системы", icon="👑"))
    return st.navigation(pages)


# ============================================================================
# Точка входа
# ============================================================================
restore_session()

if st.session_state.get("access_token"):
    # Токен в query params переживает F5 на любой странице
    st.query_params["token"] = st.session_state["access_token"]
    build_navigation().run()
else:
    drop_session()
    st.navigation([st.Page(render_auth, title="Вход", icon="🔒", default=True)]).run()
