"""Гибридный поиск по базе знаний: BM25 + векторы + Reciprocal Rank Fusion (Этап 10).

Слияние: score(d) = Σ 1 / (k + rank_i(d)) по спискам выдач BM25 и векторного
поиска (k = RRF_K из настроек, стандартно 60). Запрос — статистические
признаки нового ряда (для векторов) + текстовое описание (для BM25).

build_hybrid_retriever() собирает индексы из таблицы knowledge_base:
кейсы -> CaseVectorizer.fit -> InMemoryVectorIndex (Qdrant — прод, Этап 15).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import select

from backend.core.config import get_settings
from backend.data.embeddings import CaseVectorizer, case_to_vector_features
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.vector_index import InMemoryVectorIndex

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

    def __init__(self, cases: list[dict[str, Any]], target_dim: int = 128):
        self.cases = cases
        self.vectorizer = CaseVectorizer(target_dim=target_dim).fit(
            [case_to_vector_features(case.get("features")) for case in cases]
        )
        self.bm25 = BM25Index(cases)
        vectors = self.vectorizer.transform(
            [case_to_vector_features(case.get("features")) for case in cases]
        )
        self.index = InMemoryVectorIndex()
        self.index.add(cases, vectors)
        self._by_id = {case["series_id"]: case for case in cases}

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
            vector_hits = self.index.search(
                query_vector, top_k=max(top_k * 3, top_k), filters=filters
            )
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
    logger.info("HybridRetriever: %d кейсов", len(cases))
    return HybridRetriever(cases, target_dim=target_dim)
