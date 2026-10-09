"""Тесты для реализаций: InMemoryVectorStorage, BruteForceSearch, PicklePersistence."""
import numpy as np

from src.implementations.in_memory_storage import InMemoryVectorStorage
from src.implementations.brute_force import BruteForceSearch
from src.implementations.pickle_persistence import PicklePersistence


def test_put_and_get_by_id():
    """Проверка, что вектор, который мы положили и извлекли по id совпадают по данным"""
    db = InMemoryVectorStorage()
    db.put("id1", np.array([1.0, 2.0]), b"cat", {"category": "cat"})

    entry = db.get_by_id("id1")
    assert entry is not None
    assert entry.id == "id1"
    assert np.array_equal(entry.vector, np.array([1.0, 2.0]))
    assert entry.data == b"cat"
    assert entry.metadata == {"category": "cat"}


def test_get_by_id_missing():
    """Проверка, что если вектор не существует возвращается None"""
    db = InMemoryVectorStorage()
    assert db.get_by_id("no_such_id") is None


def test_put_overwrites_existing():
    """Проверка put с существующим ID перезаписывает запись + не создает новую с таким же id"""
    db = InMemoryVectorStorage()
    db.put("id1", np.array([1.0, 2.0]), b"old")
    db.put("id1", np.array([9.0, 9.0]), b"new")

    assert db.size() == 1
    entry = db.get_by_id("id1")
    assert entry.data == b"new"


def test_delete_existing():
    """delete существующего ID возвращает True и удаляет запись."""
    db = InMemoryVectorStorage()
    db.put("id1", np.array([1.0, 2.0]), b"x")

    assert db.delete("id1") is True
    assert db.get_by_id("id1") is None
    assert db.size() == 0


def test_delete_missing():
    """delete несуществующего ID возвращает False."""
    db = InMemoryVectorStorage()
    assert db.delete("no_such_id") is False


def test_size_and_clear():
    """size считает записи, clear удаляет всё."""
    db = InMemoryVectorStorage()
    db.put("id1", np.array([1.0, 2.0]), b"a")
    db.put("id2", np.array([3.0, 4.0]), b"b")
    assert db.size() == 2

    db.clear()
    assert db.size() == 0
    assert db.get_all() == []


def test_get_all_returns_all_entries():
    """get_all возвращает все записи."""
    db = InMemoryVectorStorage()
    db.put("id1", np.array([1.0, 2.0]), b"a")
    db.put("id2", np.array([3.0, 4.0]), b"b")

    all_entries = db.get_all()
    assert len(all_entries) == 2
    ids = {e.id for e in all_entries}
    assert ids == {"id1", "id2"}


# ---------- INearestSearch: get_nearest ----------

def test_search_returns_nearest_first():
    """Поиск возвращает самого близкого первым."""
    db = InMemoryVectorStorage()
    db.put("1", np.array([3.0, 5.0]), b"a")
    db.put("2", np.array([3.0, 4.0]), b"b")
    db.put("3", np.array([7.0, 1.0]), b"c")

    search = BruteForceSearch(db)
    results = search.get_nearest(np.array([8.0, 1.0]), k=1)

    assert len(results) == 1
    assert results[0].entry.id == "3"


def test_search_top_k_order():
    """Поиск возвращает top-k в правильном порядке."""
    db = InMemoryVectorStorage()
    db.put("1", np.array([3.0, 5.0]), b"a")
    db.put("2", np.array([3.0, 4.0]), b"b")
    db.put("3", np.array([7.0, 1.0]), b"c")

    search = BruteForceSearch(db)
    results = search.get_nearest(np.array([3.0, 3.0]), k=2)

    assert len(results) == 2
    assert results[0].entry.id == "2"
    assert results[1].entry.id == "1"


def test_search_with_filter():
    """Поиск с фильтром учитывает метаданные."""
    db = InMemoryVectorStorage()
    db.put("1", np.array([3.0, 5.0]), b"a", {"category": "cat"})
    db.put("2", np.array([3.0, 4.0]), b"b", {"category": "dog"})
    db.put("3", np.array([7.0, 1.0]), b"c", {"category": "cat"})

    search = BruteForceSearch(db)
    results = search.get_nearest(np.array([3.0, 3.0]), k=5, filter={"category": "cat"})

    assert len(results) == 2
    ids = {r.entry.id for r in results}
    assert ids == {"1", "3"}


def test_search_empty_db():
    """Поиск на пустой БД возвращает пустой список."""
    db = InMemoryVectorStorage()
    search = BruteForceSearch(db)

    assert search.get_nearest(np.array([1.0, 1.0]), k=5) == []


def test_flush_and_load(tmp_path):
    """flush сохраняет, load восстанавливает данные."""
    db = InMemoryVectorStorage()
    db.put("1", np.array([3.0, 5.0]), b"a", {"category": "cat"})
    db.put("2", np.array([3.0, 4.0]), b"b", {"category": "dog"})

    path = tmp_path / "test.pkl"
    persistence = PicklePersistence()

    persistence.flush(db, str(path))
    assert path.exists()

    db.clear()
    assert db.size() == 0

    persistence.load(db, str(path))
    assert db.size() == 2
    entry = db.get_by_id("2")
    assert entry is not None
    assert entry.data == b"b"
    assert entry.metadata == {"category": "dog"}