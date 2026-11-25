"""Гибридный поиск по базе знаний: BM25 + векторы + Reciprocal Rank Fusion (Этап 10).

Слияние: score(d) = Σ 1 / (k + rank_i(d)) по спискам выдач BM25 и векторного
поиска (k = RRF_K из настроек, стандартно 60). Запрос — статистические
признаки нового ряда (для векторов) + текстовое описание (для BM25).

Векторная часть (Этап 15):
- VECTOR_BACKEND=memory (по умолчанию) — flat cosine-индекс в процессе;
- VECTOR_BACKEND=qdrant — коллекция Qdrant (сервис qdrant в docker-compose);
  при недоступности Qdrant поиск откатывается на in-memory индекс с warning
  (как в методичке Step_7: «degrades to BM25/local if Qdrant unavailable»).

build_hybrid_retriever() собирает индексы из таблицы knowledge_base:
кейсы -> CaseVectorizer.fit -> векторы -> индекс.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from backend.core.config import get_settings
from backend.data.embeddings import CaseVectorizer, case_to_vector_features
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.vector_index import InMemoryVectorIndex, QdrantVectorIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievedCase:
    """Найденный похожий кейс с итоговым RRF-рейтингом."""

    case: dict[str, Any]
    rrf_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


class HybridRetriever:
    """BM25 + векторный поиск + RRF; фильтры применяются к векторной части."""

    def __init__(
        self,
        cases: list[dict[str, Any]],
        target_dim: int = 128,
        vector_index: Any | None = None,
    ):
        self.cases = cases
        self.vectorizer = CaseVectorizer(target_dim=target_dim).fit(
            [case_to_vector_features(case.get("features")) for case in cases]
        )
        self.bm25 = BM25Index(cases)
        vectors = self.vectorizer.transform(
            [case_to_vector_features(case.get("features")) for case in cases]
        )
        # in-memory индекс — всегда: локальный режим и запасной вариант для qdrant
        self.memory_index = InMemoryVectorIndex()
        self.memory_index.add(cases, vectors)
        self.index = vector_index  # QdrantVectorIndex в прод-режиме, иначе None
        self._by_id = {case["series_id"]: case for case in cases}

    @property
    def backend(self) -> str:
        return "qdrant" if self.index is not None else "memory"

    def _vector_search(
        self, query_vector: Any, top_k: int, filters: dict[str, Any] | None
    ) -> list[tuple[str, float]]:
        """Векторный поиск: Qdrant (если включён) с откатом на in-memory."""
        if self.index is not None:
            try:
                return self.index.search(query_vector, top_k=top_k, filters=filters)
            except Exception as exc:  # noqa: BLE001 — деградация вместо отказа
                logger.warning("Qdrant недоступен (%s) — векторный поиск переключён на in-memory", exc)
        return self.memory_index.search(query_vector, top_k=top_k, filters=filters)

    def retrieve(
        self,
        query_features: dict[str, Any] | None,
        query_text: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedCase]:
        """Топ-k похожих кейсов: fuses BM25-выдачу и косинусную близость."""
        settings = get_settings()
        k = settings.rrf_k

        bm25_hits = self.bm25.search(query_text, top_k=max(top_k * 3, top_k))
        bm25_ranks = {sid: rank for rank, (sid, _) in enumerate(bm25_hits, start=1)}

        vector_ranks: dict[str, int] = {}
        if query_features is not None:
            query_vector = self.vectorizer.transform_one(query_features)
            vector_hits = self._vector_search(query_vector, max(top_k * 3, top_k), filters)
            vector_ranks = {sid: rank for rank, (sid, _) in enumerate(vector_hits, start=1)}

        scores: dict[str, float] = {}
        for sid, rank in bm25_ranks.items():
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
        for sid, rank in vector_ranks.items():
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)

        top = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            RetrievedCase(
                case=self._by_id[sid],
                rrf_score=round(score, 6),
                bm25_rank=bm25_ranks.get(sid),
                vector_rank=vector_ranks.get(sid),
            )
            for sid, score in top
            if sid in self._by_id
        ]

    def few_shot_prompts(
        self,
        query_features: dict[str, Any] | None,
        query_text: str,
        top_k: int = 3,
    ) -> list[str]:
        """Готовые few-shot строки для промпта ModelSelector."""
        results = self.retrieve(query_features, query_text, top_k=top_k)
        return [
            "Кейс: {desc} Метрики: {metrics}.".format(
                desc=(result.case.get("description") or "").strip(),
                metrics=result.case.get("metrics") or {},
            )
            for result in results
        ]


async def load_cases_from_db(limit: int = 5000) -> list[dict[str, Any]]:
    """Кейсы из таблицы knowledge_base (для сборки индексов)."""
    from backend.db.models import KnowledgeCase
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        rows = await session.execute(select(KnowledgeCase).limit(limit))
        return [
            {
                "series_id": row.series_id,
                "task_type": row.task_type,
                "city": row.city,
                "freq": row.freq,
                "features": row.features,
                "best_model": row.best_model,
                "hyperparams": row.hyperparams,
                "metrics": row.metrics,
                "description": row.description,
            }
            for row in rows.scalars()
        ]


async def build_hybrid_retriever(target_dim: int = 128) -> HybridRetriever:
    """Собирает retriever из БД; бросает исключение, если база знаний пуста."""
    cases = await load_cases_from_db()
    if not cases:
        raise RuntimeError("База знаний пуста: запустите import_data.py")
    retriever = HybridRetriever(cases, target_dim=target_dim)

    if get_settings().vector_backend.lower() == "qdrant":
        _attach_qdrant(retriever, cases)

    logger.info("HybridRetriever: %d кейсов, vector backend=%s", len(cases), retriever.backend)
    return retriever


def _attach_qdrant(retriever: HybridRetriever, cases: list[dict[str, Any]]) -> None:
    """Подключает Qdrant-индекс; при недоступности остаёмся на in-memory."""
    settings = get_settings()
    try:
        index = QdrantVectorIndex(collection=settings.qdrant_collection, url=settings.qdrant_url)
        vectors = retriever.vectorizer.transform(
            [case_to_vector_features(case.get("features")) for case in cases]
        )
        # первичная загрузка, если коллекция пуста (далее — переиндексация после HITL)
        index.ensure_collection(vectors.shape[1])
        if index.count() == 0:
            index.add(cases, vectors)
        retriever.index = index
    except Exception as exc:  # noqa: BLE001 — Qdrant опционален
        logger.warning("Qdrant недоступен (%s) — RAG работает на in-memory индексе", exc)
