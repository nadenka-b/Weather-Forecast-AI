"""Зависимости FastAPI: сессия БД, текущий пользователь, проверка ролей.

Использование в роутах:

    @router.get("/protected")
    async def protected(user: User = Depends(get_current_user)):
        return {"email": user.email}

    @router.post("/admin-only")
    async def admin_only(user: User = Depends(require_admin)):
        return {"status": "ok"}
"""
import logging

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import User
from backend.db.session import AsyncSessionLocal
from backend.utils.security import decode_token

logger = logging.getLogger(__name__)

# auto_error=False: без заголовка Authorization не бросаем 403 сразу,
# а возвращаем None — нужно и для обязательной, и для опциональной аутентификации
security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncSession:
    """Выдаёт сессию БД и закрывает её после ответа."""
    async with AsyncSessionLocal() as session:
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Текущий пользователь по JWT из заголовка Authorization.

    Бросает 401 (нет/невалиден токен) или 404 (пользователь удалён/деактивирован).
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
    )
    user = result.scalar_one_or_none()

    if not user:
        # токен валидный, но пользователя нет — это «не найден», а не «не аутентифицирован»
        raise HTTPException(status_code=404, detail="User not found")

    return user


async def optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Пользователь или None. Никогда не бросает исключений (публичные роуты)."""
    if not credentials:
        return None
    try:
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        result = await db.execute(
            select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()
    except Exception:  # noqa: BLE001 — публичный роут не должен падать из-за токена
        return None


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Только администраторы, иначе 403."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def require_editor(current_user: User = Depends(get_current_user)) -> User:
    """admin и editor (обработка HITL-очереди), иначе 403."""
    if current_user.role not in ("admin", "editor"):
        raise HTTPException(status_code=403, detail="Editor access required")
    return current_user
