"""Векторный индекс кейсов (Этап 10, прод-режим — Этап 15).

- InMemoryVectorIndex — локальная разработка: flat cosine на numpy
  (эквивалент FAISS FlatIP на наших объёмах ~1000 кейсов);
- QdrantVectorIndex — продакшен (VECTOR_BACKEND=qdrant): коллекция с
  payload-метаданными, поиск с фильтрами (task_type, city, freq, variable).
  ID точек — uuid5 по series_id: переиндексация идемпотентна (upsert),
  а не плодит дубликаты, как uuid4.

Оба поддерживают контракт: add(cases, vectors), search(vector, top_k, filters).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_QDRANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # RFC 4122 DNS


def point_id(series_id: str) -> str:
    """Детерминированный ID точки: uuid5 от series_id (идемпотентный upsert)."""
    return str(uuid.uuid5(_QDRANT_NAMESPACE, series_id))


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
            if key == "variable":
                mask &= np.array(
                    [(case.get("features") or {}).get("variable") == value for case in self._cases]
                )
            else:
                mask &= np.array([str(case.get(key)) == str(value) for case in self._cases])
        if not mask.any():
            return []

        candidates = np.where(mask)[0]
        query = np.asarray(vector, dtype="float64")
        query = query / (np.linalg.norm(query) or 1.0)
        sims = self._vectors[candidates] @ query
        order = np.argsort(-sims)[:top_k]
        return [(self._cases[candidates[i]]["series_id"], float(sims[i])) for i in order]


class QdrantVectorIndex:
    """Продакшен-индекс на Qdrant (docker-compose, Этап 15).

    client можно передать извне (тесты — QdrantClient(":memory:"));
    иначе клиент создаётся лениво по URL из настроек.
    """

    def __init__(self, collection: str, url: str | None = None, client: Any | None = None):
        self.collection = collection
        self._client = client
        self._url = url

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self._url, timeout=5)
        return self._client

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        exists = client.collection_exists(self.collection)
        if exists and recreate:
            client.delete_collection(collection_name=self.collection)
            exists = False
        if not exists:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def count(self) -> int:
        """Число точек в коллекции (точный подсчёт)."""
        client = self._get_client()
        if not client.collection_exists(self.collection):
            return 0
        try:  # qdrant-client >= 1.12
            from qdrant_client.models import CountRequest

            return int(client.count(collection_name=self.collection, request=CountRequest(exact=True)).count)
        except (TypeError, AssertionError):  # другие версии клиента: exact — аргумент
            return int(client.count(collection_name=self.collection, exact=True).count)

    def add(self, cases: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Идемпотентный upsert (uuid5 по series_id)."""
        from qdrant_client.models import PointStruct

        client = self._get_client()
        self.ensure_collection(vectors.shape[1])
        points = [
            PointStruct(
                id=point_id(case["series_id"]),
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
        return [(point.payload["series_id"], float(point.score)) for point in result.points]
