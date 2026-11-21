"""Роуты аутентификации:

- POST /auth/register — регистрация, возвращает JWT;
- POST /auth/login    — вход, возвращает JWT;
- GET  /auth/me       — текущий пользователь (нужен JWT).
"""
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user
from backend.db.models import User
from backend.db.session import get_async_session
from backend.utils.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================================
# Pydantic-модели запросов и ответов
# ============================================================================
class RegisterRequest(BaseModel):
    """Тело запроса на регистрацию."""

    email: EmailStr
    username: str
    password: str
    full_name: str | None = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Минимальная сложность пароля: 8+ символов, цифра, заглавная."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        return v


class LoginRequest(BaseModel):
    """Тело запроса на вход."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Ответ с JWT-токеном и данными пользователя."""

    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str
    username: str


# ============================================================================
# POST /auth/register
# ============================================================================
@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_async_session),
) -> TokenResponse:
    """Регистрация: 201 — успех, 400 — email/username заняты, 422 — слабый пароль."""
    existing = await session.execute(select(User).where(User.email == request.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    existing = await session.execute(select(User).where(User.username == request.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        id=str(uuid.uuid4()),
        email=request.email,
        username=request.username,
        hashed_password=get_password_hash(request.password),
        full_name=request.full_name,
        role="user",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    access_token = create_access_token(data={"sub": user.id, "role": user.role})
    logger.info("User registered: %s", user.email)

    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        role=user.role,
        username=user.username,
    )


# ============================================================================
# POST /auth/login
# ============================================================================
@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_async_session),
) -> TokenResponse:
    """Вход: 200 — токен, 401 — неверные данные, 403 — аккаунт отключён."""
    result = await session.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    # Одинаковый текст ошибки для «нет пользователя» и «неверный пароль» —
    # чтобы нельзя было перебором проверить существование email.
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    access_token = create_access_token(data={"sub": user.id, "role": user.role})
    logger.info("User logged in: %s", user.email)

    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        role=user.role,
        username=user.username,
    )


# ============================================================================
# GET /auth/me
# ============================================================================
@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)) -> dict:
    """Данные текущего пользователя. Требует Authorization: Bearer <token>."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
    }
