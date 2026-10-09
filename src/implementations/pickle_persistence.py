"""Pickle-реализация IPersistence.

Сохраняет все записи хранилища в бинарный файл через pickle.
"""
import pickle

from ..interfaces import IPersistence, IVectorStorage


class PicklePersistence(IPersistence):
    """Сохранение и загрузка через pickle.

    Сохраняет все VectorEntry в один бинарный файл.
    При загрузке восстанавливает их в хранилище через put().
    """

    def flush(self, storage: IVectorStorage, path: str) -> None:
        """Сохранить все данные хранилища в файл.

        Args:
            storage: Хранилище для сохранения.
            path: Путь к файлу.
        """
        entries = storage.get_all()
        with open(path, "wb") as f:
            pickle.dump(entries, f)

    def load(self, storage: IVectorStorage, path: str) -> None:
        """Загрузить данные из файла в хранилище.

        Args:
            storage: Хранилище, куда загружать.
            path: Путь к файлу.
        """
        with open(path, "rb") as f:
            entries = pickle.load(f)
        for entry in entries:
            storage.put(
                entry.id,
                entry.vector,
                entry.data,
                entry.metadata,
            )