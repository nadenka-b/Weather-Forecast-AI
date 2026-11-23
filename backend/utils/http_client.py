"""Единый httpx.AsyncClient на каждый event loop (методичка Step_5, п. 5.2).

Клиент держит пул соединений: создавать его на каждый запрос — TCP handshake
каждый раз; шарить один между loop'ами (FastAPI и Celery) — RuntimeError:
Event loop is closed. Решение — один клиент на loop в слабых ссылках.

Для LLM-запросов таймаут большой: холодный старт модели + длинный промпт
могут занять минуты (совпадает с stop_grace_period в docker-compose).
"""
import asyncio
import logging
import threading
import weakref

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300.0

_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)
_lock = threading.Lock()


def get_http_client(timeout: float = _DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    """Клиент, привязанный к текущему event loop."""
    loop = asyncio.get_running_loop()
    with _lock:
        client = _clients.get(loop)
        if client is None:
            client = httpx.AsyncClient(timeout=timeout)
            _clients[loop] = client
            logger.debug("Created httpx.AsyncClient for loop id=%s", id(loop))
        return client


async def close_http_clients() -> None:
    """Закрывает всех клиентов (вызывается в lifespan при остановке)."""
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception as exc:  # noqa: BLE001 — повторный вызов должен быть no-op
            logger.warning("Error closing httpx.AsyncClient: %s", exc)
