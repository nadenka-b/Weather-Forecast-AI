"""Асинхронный retry-декоратор (методичка Step_5, п. 5.1).

Повторяет только перечисленные типы исключений (сетевые сбои, таймауты).
Ошибки логики (ValueError, KeyError) не ретраятся — повтор не поможет.
"""
import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    backoff: float = 1.0,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Декоратор async-функции: до max_attempts попыток с паузой delay * backoff**n."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt >= max_attempts:
                        break
                    pause = delay * (backoff ** (attempt - 1))
                    logger.warning(
                        "%s: попытка %d/%d не удалась (%s: %s) — повтор через %.1f с",
                        func.__name__, attempt, max_attempts, type(exc).__name__, exc, pause,
                    )
                    await asyncio.sleep(pause)
            assert last_error is not None
            raise last_error

        return wrapper

    return decorator
