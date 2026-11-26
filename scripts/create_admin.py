"""Создание администратора: python scripts/create_admin.py --email ... --username ... --password ...

Роль admin нужна для HITL-очереди и пополнения базы знаний (Этап 14).
Пароль хешируется bcrypt, дубликаты игнорируются.
"""
import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # прямой запуск: python scripts/...

from backend.db.models import User  # noqa: E402
from backend.db.session import AsyncSessionLocal  # noqa: E402
from backend.utils.security import get_password_hash  # noqa: E402


async def create_admin(email: str, username: str, password: str) -> None:
    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none():
            print(f"Пользователь {email} уже существует — пропускаем")
            return

        user = User(
            email=email,
            username=username,
            hashed_password=get_password_hash(password),
            role="admin",
        )
        session.add(user)
        await session.commit()
        print(f"Администратор создан: {email}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание администратора")
    parser.add_argument("--email", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()
    asyncio.run(create_admin(args.email, args.username, args.password))


if __name__ == "__main__":
    sys.exit(main())
