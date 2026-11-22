"""Эмбеддинги кейсов базы знаний (Этап 10).

Базовое требование задания MLF: статистические признаки ряда (тренд,
сезонность, автокорреляции, ADF) + снижение размерности (PCA) до 128 dims.

Схема:
1. числовые признаки кейса (features из knowledge_base) собираются
   в фиксированном порядке + one-hot по переменной (temperature, ...);
2. StandardScaler -> PCA (min(128, n_components_возможных)) -> L2-нормировка;
3. вектор кейса и вектор запроса (новый ряд) считаются одним трансформером,
   поэтому сравнимы по косинусной близости.

Классическая ловушка: PCA обучать на запросе нельзя — только на корпусе
кейсов, затем transform запроса. Fit один раз при сборке индекса
(синглтон в пределах процесса сборки retriever'а).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Числовые признаки кейса в фиксированном порядке (None -> 0.0)
NUMERIC_FEATURES: tuple[str, ...] = (
    "length",
    "mean",
    "std",
    "min",
    "max",
    "acf_lag1",
    "adf_pvalue",
    "stationary",
    "seasonal_period",
    "trend_strength",
    "seasonal_strength",
    "missing_rate",
)

KNOWN_VARIABLES: tuple[str, ...] = (
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "precipitation",
    "dew_point",
    "meantemp",
)


def case_to_vector_features(features: dict[str, Any] | None) -> list[float]:
    """Числовое представление кейса: стат. признаки + one-hot переменной."""
    features = features or {}
    row: list[float] = []
    for key in NUMERIC_FEATURES:
        value = features.get(key)
        row.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
    variable = str(features.get("variable", "")).lower()
    row.extend(1.0 if variable == known else 0.0 for known in KNOWN_VARIABLES)
    return row


class CaseVectorizer:
    """Фит на корпусе кейсов, transform для кейсов и запросов."""

    def __init__(self, target_dim: int = 128):
        self.target_dim = target_dim
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self._fitted = False

    @property
    def dim(self) -> int:
        return self.pca.n_components_ if self.pca is not None else len(NUMERIC_FEATURES) + len(
            KNOWN_VARIABLES
        )

    def fit(self, feature_rows: list[list[float]]) -> "CaseVectorizer":
        """Обучает scaler+PCA на корпусе; PCA до min(target_dim, признаков, кейсов)."""
        matrix = np.asarray(feature_rows, dtype="float64")
        scaled = self.scaler.fit_transform(matrix)
        n_components = min(self.target_dim, scaled.shape[1], max(1, scaled.shape[0]))
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(scaled)
        self._fitted = True
        logger.info(
            "CaseVectorizer: fit на %d кейсах, размерность %d -> %d",
            len(feature_rows), matrix.shape[1], n_components,
        )
        return self

    def transform(self, feature_rows: list[list[float]]) -> np.ndarray:
        """Проецирует строки в пространство PCA и L2-нормирует."""
        if not self._fitted:
            raise RuntimeError("CaseVectorizer не обучен: сначала fit(корпус)")
        matrix = np.asarray(feature_rows, dtype="float64")
        scaled = self.scaler.transform(matrix)
        embedded = self.pca.transform(scaled)
        norms = np.linalg.norm(embedded, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embedded / norms).astype("float64")

    def transform_one(self, features: dict[str, Any] | None) -> np.ndarray:
        """Вектор одного кейса/запроса (1, dim)."""
        vector = self.transform([case_to_vector_features(features)])
        return vector[0]
