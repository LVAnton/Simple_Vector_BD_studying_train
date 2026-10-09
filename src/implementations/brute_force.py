"""Brute-force реализация INearestSearch.

Проходит по всем векторам, считает косинусное сходство,
сортирует и возвращает top-k.
"""
from typing import List, Optional, Dict, Any
import numpy as np

from ..interfaces import INearestSearch, IVectorStorage
from ..models import SearchResult, VectorEntry


class BruteForceSearch(INearestSearch):
    """Поиск ближайших соседей полным перебором.

    Работает за O(n) на запрос, где n — количество записей.
    Для больших БД медленно, но для учебного проекта — отлично.
    """

    def __init__(self, storage: IVectorStorage) -> None:
        """Принимает хранилище, по которому будем искать.

        Args:
            storage: Хранилище векторов.
        """
        self.storage = storage

    def get_nearest(
        self,
        query: np.ndarray,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Найти k ближайших векторов.

        Args:
            query: Вектор запроса.
            k: Сколько результатов вернуть.
            filter: Фильтр по метаданным (опционально).

        Returns:
            Список SearchResult, отсортированный по убыванию сходства.
        """
        arr = []
        for elem in self.storage.get_all():
            if self._matches_filter(elem, filter):
                cos_sim = self._cosine_similarity(elem.vector, query)
                arr.append([cos_sim, elem])
        arr.sort(key=lambda x: x[0], reverse=True)

        result = []
        for cos_sim, elem in arr[:k]:
            result.append(SearchResult(elem, cos_sim))
        return result

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Считает косинусное сходство между двумя векторами.

        Args:
            a: Первый вектор.
            b: Второй вектор.

        Returns:
            Сходство от -1 до 1.
        """
        x = np.sqrt(np.sum(a ** 2))
        y = np.sqrt(np.sum(b ** 2))
        if x == 0 or y == 0:
            return 0.0
        return float((a @ b) / (x * y))

    @staticmethod
    def _matches_filter(
        entry: VectorEntry,
        filter: Optional[Dict[str, Any]],
    ) -> bool:
        """Проверяет, подходит ли запись под фильтр.

        Если filter=None — подходит всегда.
        Если у записи нет ключа из filter — не подходит.

        Args:
            entry: Запись.
            filter: Фильтр по метаданным.

        Returns:
            True, если запись подходит.
        """
        if filter is None:
            return True

        for key in filter:
            if key not in entry.metadata:
                return False
            if entry.metadata[key] != filter[key]:
                return False
        return True