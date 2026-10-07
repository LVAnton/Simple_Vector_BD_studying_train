"""Модели данных для векторной БД."""
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class VectorEntry:
    """Одна запись в векторной БД.

    Attributes:
        id: Уникальный идентификатор записи.
        vector: Вектор (массив чисел с плавающей точкой).
        data: Полезная нагрузка (произвольные байты).
        metadata: Метаданные для фильтрации (опционально).
    """
    id: str
    vector: np.ndarray
    data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Результат поиска ближайших соседей.

    Attributes:
        entry: Найденная запись.
        similarity: Косинусное сходство (от -1 до 1, чем больше — тем ближе).
    """
    entry: VectorEntry
    similarity: float