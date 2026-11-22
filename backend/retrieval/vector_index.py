"""Векторный индекс кейсов (Этап 10).

- InMemoryVectorIndex — локальная разработка: flat cosine на numpy
  (эквивалент FAISS FlatIP на наших объёмах ~1000 кейсов);
- QdrantVectorIndex — продакшен: обёртка над qdrant-client с фильтрами
  по метаданным (task_type, city, freq). Ленивый импорт — клиент
  подтягивается только в прод-режиме.

Оба поддерживают контракт: add(cases, vectors), search(vector, top_k, filters).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class InMemoryVectorIndex:
    """Flat cosine-индекс на numpy с фильтрами по метаданным."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._cases: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._cases)

    def add(self, cases: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Добавляет кейсы с их L2-нормированными векторами."""
        self._cases = list(cases)
        self._vectors = np.asarray(vectors, dtype="float64")

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """[(series_id, cosine)] по фильтрам (task_type/city/freq/variable)."""
        if self._vectors is None or not self._cases:
            return []

        mask = np.ones(len(self._cases), dtype=bool)
        for key, value in (filters or {}).items():
            if value is None:
                continue
            mask &= np.array(
                [str(case.get(key)) == str(value) for case in self._cases]
            )
        if not mask.any():
            return []

        candidates = np.where(mask)[0]
        query = np.asarray(vector, dtype="float64")
        query = query / (np.linalg.norm(query) or 1.0)
        sims = self._vectors[candidates] @ query
        order = np.argsort(-sims)[:top_k]
        return [
            (self._cases[candidates[i]]["series_id"], float(sims[i]))
            for i in order
        ]


class QdrantVectorIndex:
    """Продакшен-индекс на Qdrant (docker-compose, Этап 15).

    Пишет точки с payload-метаданными, ищет с фильтрами. До поднятия Qdrant
    гибридный retriever использует InMemoryVectorIndex.
    """

    def __init__(self, collection: str, url: str | None = None):
        self.collection = collection
        self._client = None
        self._url = url

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self._url, timeout=5)
        return self._client

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, cases: list[dict[str, Any]], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        client = self._get_client()
        self.ensure_collection(vectors.shape[1])
        points = [
            PointStruct(
                id=index,
                vector=vectors[index].tolist(),
                payload={
                    "series_id": case["series_id"],
                    "task_type": case.get("task_type", "timeseries"),
                    "city": case.get("city"),
                    "freq": case.get("freq"),
                    "variable": (case.get("features") or {}).get("variable"),
                },
            )
            for index, case in enumerate(cases)
        ]
        client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in (filters or {}).items()
            if value is not None
        ]
        result = self._get_client().query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype="float64").tolist(),
            limit=top_k,
            query_filter=Filter(must=conditions) if conditions else None,
        )
        return [(point.payload["series_id"], point.score) for point in result.points]
