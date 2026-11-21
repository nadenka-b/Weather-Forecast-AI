"""Хеширование паролей (bcrypt) и работа с JWT-токенами (PyJWT).

Пароли НИКОГДА не хранятся в открытом виде — в БД лежит bcrypt-хеш
(`$2b$12$...`, 60 символов). При логине введённый пароль хешируется
и сравнивается с хешем из БД.

Важно: passlib 1.7.4 несовместим с bcrypt>=5 (нет `__about__`),
поэтому в pyproject.toml зафиксировано bcrypt < 5.0.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt as pyjwt
from fastapi import HTTPException
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from passlib.context import CryptContext

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = settings.jwt_secret
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_expire_minutes


# ============================================================================
# Пароли
# ============================================================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет, что пароль соответствует хешу. Исключений не бросает."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Возвращает bcrypt-хеш (каждый вызов — разный, из-за случайной соли)."""
    return pwd_context.hash(password)


# ============================================================================
# JWT-токены
# ============================================================================
def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Создаёт JWT: header.payload.signature.

    :param data: полезная нагрузка, например {"sub": user_id, "role": "user"}
    :param expires_delta: время жизни; None -> JWT_EXPIRE_MINUTES из настроек
    """
    to_encode: dict[str, Any] = {}
    for key, value in data.items():
        # PyJWT сериализует только примитивы — остальное приводим к строке.
        # sub (id пользователя) всегда строка: User.id в БД — String.
        if key == "sub":
            to_encode[key] = str(value)
        elif isinstance(value, (str, int, float, bool)):
            to_encode[key] = value
        else:
            to_encode[key] = str(value)

    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire

    encoded = pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug("JWT created for sub=%s", to_encode.get("sub"))
    return encoded


def decode_token(token: str) -> dict[str, Any]:
    """Декодирует и проверяет JWT. Истёк/неверная подпись -> HTTPException(401)."""
    try:
        return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError:
        logger.warning("JWT token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError as exc:
        logger.warning("JWT decode error: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid token")
