"""BM25-индекс по текстовым описаниям кейсов (Этап 10).

Лексическая часть гибридного поиска: токенизация (буквенно-цифровые
токены, включая кириллицу), ранжирование Okapi BM25 (rank-bm25).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+")


def tokenize(text: str) -> list[str]:
    """Токены: слова в нижнем регистре (кириллица + латиница + цифры)."""
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


class BM25Index:
    """Индекс по descriptions кейсов; ищет топ-k по текстовому запросу."""

    def __init__(self, cases: list[dict[str, Any]]):
        self._case_ids = [case["series_id"] for case in cases]
        corpus = [tokenize(case.get("description") or "") for case in cases]
        # пустые документы ломают BM25 — заменяем на нейтральный токен
        corpus = [tokens if tokens else ["пусто"] for tokens in corpus]
        self._tokenized = corpus
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """[(series_id, bm25_score)] отсортированные по убыванию."""
        tokens = tokenize(query)
        if not tokens or not self._case_ids:
            return []
        scores = self._bm25.get_scores(tokens)
        if max(scores) <= 0:
            # На маленьком корпусе IDF зануляется (термин есть в половине документов),
            # и Okapi даёт всем 0. Запасной рейтинг — число совпадений термина.
            token_set = set(tokens)
            scores = [
                float(len(token_set & set(self._tokenized[i]))) for i in range(len(self._case_ids))
            ]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            (self._case_ids[i], float(scores[i]))
            for i in order[:top_k]
            if scores[i] > 0
        ]
