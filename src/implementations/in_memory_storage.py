"""In-memory реализация IVectorStorage."""
from typing import List, Optional, Dict, Any
import numpy as np

from ..interfaces import IVectorStorage
from ..models import VectorEntry


class InMemoryVectorStorage(IVectorStorage):
    """Хранилище векторов в оперативной памяти.

    Данные хранятся в обычном словаре dict[str, VectorEntry].
    Скорость доступа по ID — O(1).
    """

    def __init__(self) -> None:
        """Создаёт пустое хранилище."""
        self.storage = {}

    def put(
        self,
        id: str,
        vector: np.ndarray,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Добавить или обновить запись (upsert).

        Args:
            id: Уникальный идентификатор.
            vector: Вектор.
            data: Полезная нагрузка.
            metadata: Метаданные (опционально).
        """
        self.storage[id] = VectorEntry(id, vector, data, metadata or {})

    def get_by_id(self, id: str) -> Optional[VectorEntry]:
        """Получить запись по ID или None."""
        if id in self.storage:
            return self.storage[id]
        return None

    def delete(self, id: str) -> bool:
        """Удалить запись. True, если удалено, False — если не найдено."""
        if id in self.storage:
            del self.storage[id]
            return True
        return False

    def get_all(self) -> List[VectorEntry]:
        """Получить все записи."""
        return list(self.storage.values())

    def size(self) -> int:
        """Количество записей."""
        return len(self.storage)

    def clear(self) -> None:
        """Удалить все записи."""
        self.storage.clear()