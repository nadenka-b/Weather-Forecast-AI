"""Юнит-тесты backend/utils/security (bcrypt + PyJWT)."""
from datetime import timedelta

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from backend.utils.security import (
    ALGORITHM,
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)


# ============================================================================
# Хеширование паролей
# ============================================================================
class TestPasswordHashing:
    def test_hash_starts_with_bcrypt_prefix(self):
        h = get_password_hash("StrongPass1")
        assert h.startswith(("$2b$", "$2a$"))

    def test_same_password_different_hashes(self):
        """bcrypt добавляет случайную соль — хеши одного пароля различаются."""
        assert get_password_hash("StrongPass1") != get_password_hash("StrongPass1")

    def test_verify_correct_password(self):
        h = get_password_hash("StrongPass1")
        assert verify_password("StrongPass1", h) is True

    def test_verify_wrong_password(self):
        h = get_password_hash("StrongPass1")
        assert verify_password("WrongPass1", h) is False

    def test_verify_empty_password(self):
        h = get_password_hash("StrongPass1")
        assert verify_password("", h) is False

    def test_hash_is_60_chars(self):
        """Стандартный bcrypt-хеш — ровно 60 символов."""
        assert len(get_password_hash("StrongPass1")) == 60


# ============================================================================
# JWT-токены
# ============================================================================
class TestJWTCreation:
    def test_token_has_three_parts(self):
        token = create_access_token({"sub": "user-1", "role": "user"})
        assert len(token.split(".")) == 3

    def test_decode_returns_payload(self):
        token = create_access_token({"sub": "user-1", "role": "admin"})
        payload = decode_token(token)
        assert payload["sub"] == "user-1"
        assert payload["role"] == "admin"
        assert "exp" in payload

    def test_custom_expires_delta(self):
        token = create_access_token({"sub": "user-1"}, expires_delta=timedelta(minutes=5))
        assert decode_token(token)["sub"] == "user-1"

    def test_expired_token_raises_401(self):
        token = create_access_token({"sub": "user-1"}, expires_delta=timedelta(seconds=-1))
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Token expired"

    def test_invalid_signature_raises_401(self):
        """Токен, подписанный чужим ключом, не пройдёт проверку подписи."""
        wrong_token = pyjwt.encode(
            {"sub": "user-1", "exp": 9999999999}, "wrong-secret-key", algorithm=ALGORITHM
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_token(wrong_token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"

    def test_malformed_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            decode_token("not-a-jwt")
        assert exc_info.value.status_code == 401

    def test_non_string_sub_becomes_string(self):
        token = create_access_token({"sub": 12345})
        assert decode_token(token)["sub"] == "12345"
