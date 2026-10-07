"""Интерфейсы (абстрактные классы) для векторной БД.

Здесь описаны контракты — что должна уметь БД.
Реализация будет во 2 лабораторной.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any #Без этого было бы норм тоже, это чисто подписывает типы для красоты
import numpy as np

from .models import VectorEntry, SearchResult


class IVectorStorage(ABC):
    """Основное хранилище векторов.

    Отвечает за KV-часть: доступ по ID, вставка, удаление.
    Не отвечает за поиск ближайших соседей — это отдельный интерфейс.
    """

    @abstractmethod
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
            data: Полезная нагрузка. Типо объект, которому принадлежит эмбеддинг.
            metadata: Метаданные (опционально).
        """
        ...

    @abstractmethod
    def get_by_id(self, id: str) -> Optional[VectorEntry]:
        """Получить запись по ID.

        Args:
            id: Идентификатор записи.

        Returns:
            VectorEntry или None, если не найдено.
        """
        ...

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Удалить запись по ID.

        Args:
            id: Идентификатор записи.

        Returns:
            True, если запись была удалена, False — если не найдена.
        """
        ...

    @abstractmethod
    def get_all(self) -> List[VectorEntry]:
        """Получить все записи."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Количество записей в хранилище."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Удалить все записи из хранилища."""
        ...


class INearestSearch(ABC):
    """Поиск ближайших соседей по вектору.

    Отделён от IVectorStorage, потому что поиск может быть разным:
    brute force, IVF, remote (для шардирования).
    """

    @abstractmethod
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
        ...


class IPersistence(ABC):
    """Сохранение и загрузка БД на диск.

    Отдельный сервис — умеет сохранять любое хранилище.
    Это разделение ответственности (Single Responsibility Principle).
    """

    @abstractmethod
    def flush(self, storage: IVectorStorage, path: str) -> None:
        """Сохранить все данные хранилища в файл.

        Args:
            storage: Хранилище для сохранения.
            path: Путь к файлу.
        """
        ...

    @abstractmethod
    def load(self, storage: IVectorStorage, path: str) -> None:
        """Загрузить данные из файла в хранилище.

        Args:
            storage: Хранилище, куда загружать.
            path: Путь к файлу.
        """
        ...