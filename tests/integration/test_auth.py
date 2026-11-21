"""Интеграционные тесты /auth/*: register -> login -> /me.

Изоляция через SQLite in-memory + StaticPool (одно соединение на фикстуру,
иначе таблицы «пропадают» между запросами). Подменяем engine и фабрику сессий
во всех модулях, где они импортированы напрямую: db.session, api.deps, api.main.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from backend.db.models import Base


@pytest.fixture
def client(monkeypatch):
    """TestClient с изолированной in-memory SQLite."""
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session_local = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    import backend.api.deps as deps_module
    import backend.api.main as main_module
    import backend.db.session as db_session_module

    monkeypatch.setattr(db_session_module, "async_engine", test_engine)
    monkeypatch.setattr(db_session_module, "AsyncSessionLocal", test_session_local)
    monkeypatch.setattr(deps_module, "AsyncSessionLocal", test_session_local)
    monkeypatch.setattr(main_module, "async_engine", test_engine)

    asyncio.run(_create_tables(test_engine))

    from backend.api.main import app

    with TestClient(app) as c:
        yield c


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@pytest.fixture
def registered_user(client):
    """Зарегистрированный пользователь: данные + токен."""
    r = client.post(
        "/auth/register",
        json={
            "email": "user@example.com",
            "username": "user1",
            "password": "StrongPass1",
            "full_name": "User One",
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    return {
        "email": "user@example.com",
        "username": "user1",
        "password": "StrongPass1",
        "token": data["access_token"],
        "user_id": data["user_id"],
    }


# ============================================================================
# Регистрация
# ============================================================================
class TestRegister:
    def test_register_success(self, client):
        r = client.post(
            "/auth/register",
            json={
                "email": "new@example.com",
                "username": "newuser",
                "password": "StrongPass1",
                "full_name": "New User",
            },
        )
        assert r.status_code == 201
        data = r.json()
        assert "access_token" in data
        assert data["email"] == "new@example.com"
        assert data["role"] == "user"
        assert data["token_type"] == "bearer"

    def test_register_duplicate_email(self, client, registered_user):
        r = client.post(
            "/auth/register",
            json={
                "email": registered_user["email"],
                "username": "different_username",
                "password": "StrongPass1",
            },
        )
        assert r.status_code == 400
        assert r.json()["detail"] == "Email already registered"

    def test_register_duplicate_username(self, client, registered_user):
        r = client.post(
            "/auth/register",
            json={
                "email": "different@example.com",
                "username": registered_user["username"],
                "password": "StrongPass1",
            },
        )
        assert r.status_code == 400
        assert r.json()["detail"] == "Username already taken"

    def test_register_weak_password_too_short(self, client):
        r = client.post(
            "/auth/register",
            json={"email": "weak@example.com", "username": "w", "password": "Short1"},
        )
        assert r.status_code == 422
        assert "at least 8 characters" in r.text

    def test_register_weak_password_no_digit(self, client):
        r = client.post(
            "/auth/register",
            json={"email": "weak@example.com", "username": "w", "password": "NoDigitsHere"},
        )
        assert r.status_code == 422
        assert "digit" in r.text

    def test_register_weak_password_no_uppercase(self, client):
        r = client.post(
            "/auth/register",
            json={"email": "weak@example.com", "username": "w", "password": "nouppercase1"},
        )
        assert r.status_code == 422
        assert "uppercase" in r.text

    def test_register_invalid_email(self, client):
        r = client.post(
            "/auth/register",
            json={"email": "not-an-email", "username": "someuser", "password": "StrongPass1"},
        )
        assert r.status_code == 422


# ============================================================================
# Логин
# ============================================================================
class TestLogin:
    def test_login_success(self, client, registered_user):
        r = client.post(
            "/auth/login",
            json={"email": registered_user["email"], "password": registered_user["password"]},
        )
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_login_wrong_password(self, client, registered_user):
        r = client.post(
            "/auth/login",
            json={"email": registered_user["email"], "password": "WrongPass1"},
        )
        assert r.status_code == 401
        assert r.json()["detail"] == "Invalid credentials"

    def test_login_unknown_email(self, client):
        r = client.post(
            "/auth/login",
            json={"email": "nobody@example.com", "password": "StrongPass1"},
        )
        assert r.status_code == 401
        assert r.json()["detail"] == "Invalid credentials"


# ============================================================================
# /auth/me
# ============================================================================
class TestMe:
    def test_me_with_valid_token(self, client, registered_user):
        r = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["email"] == registered_user["email"]
        assert data["role"] == "user"
        assert data["is_active"] is True

    def test_me_without_token(self, client):
        assert client.get("/auth/me").status_code == 401

    def test_me_with_invalid_token(self, client):
        r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-valid-token"})
        assert r.status_code == 401


# ============================================================================
# Полный цикл
# ============================================================================
class TestFullCycle:
    def test_register_login_me(self, client):
        r = client.post(
            "/auth/register",
            json={
                "email": "cycle@example.com",
                "username": "cycleuser",
                "password": "StrongPass1",
            },
        )
        assert r.status_code == 201
        token_register = r.json()["access_token"]

        r = client.get("/auth/me", headers={"Authorization": f"Bearer {token_register}"})
        assert r.status_code == 200
        assert r.json()["email"] == "cycle@example.com"

        r = client.post(
            "/auth/login",
            json={"email": "cycle@example.com", "password": "StrongPass1"},
        )
        assert r.status_code == 200
        token_login = r.json()["access_token"]

        r = client.get("/auth/me", headers={"Authorization": f"Bearer {token_login}"})
        assert r.status_code == 200
        assert r.json()["email"] == "cycle@example.com"
