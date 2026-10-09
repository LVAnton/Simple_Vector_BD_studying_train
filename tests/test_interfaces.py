"""Тесты для проверки, что интерфейсы абстрактны."""
import pytest

from src.interfaces import IVectorStorage, INearestSearch, IPersistence


def test_cannot_instantiate_storage():
    """IVectorStorage нельзя создать напрямую."""
    with pytest.raises(TypeError):
        IVectorStorage()


def test_cannot_instantiate_search():
    """INearestSearch нельзя создать напрямую."""
    with pytest.raises(TypeError):
        INearestSearch()


def test_cannot_instantiate_persistence():
    """IPersistence нельзя создать напрямую."""
    with pytest.raises(TypeError):
        IPersistence()