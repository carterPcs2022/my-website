import numpy as np
import pytest

from zane.memory.vector_index import FaissVectorIndex

faiss = pytest.importorskip("faiss")


@pytest.fixture
def index_path(tmp_path):
    return str(tmp_path / "zane_test.faiss")


def _unit_vector(*components) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_cold_start_search_on_empty_index(index_path):
    idx = FaissVectorIndex(index_path)
    assert idx.search(_unit_vector(1, 0, 0, 0), top_k=5) == []
    assert idx.size == 0


def test_add_and_search_returns_closest_match(index_path):
    idx = FaissVectorIndex(index_path)
    idx.add(1, _unit_vector(1, 0, 0, 0))
    idx.add(2, _unit_vector(0, 1, 0, 0))
    idx.add(3, _unit_vector(0.9, 0.1, 0, 0))

    hits = idx.search(_unit_vector(1, 0, 0, 0), top_k=2)
    hit_ids = [vec_id for vec_id, _score in hits]
    assert hit_ids[0] == 1  # exact match ranks first
    assert 3 in hit_ids  # near match should also be in the top 2
    assert 2 not in hit_ids  # orthogonal vector should rank last, outside top 2


def test_remove_deletes_vectors(index_path):
    idx = FaissVectorIndex(index_path)
    idx.add(1, _unit_vector(1, 0, 0, 0))
    idx.add(2, _unit_vector(0, 1, 0, 0))
    assert idx.size == 2

    idx.remove([1])
    assert idx.size == 1
    hits = idx.search(_unit_vector(1, 0, 0, 0), top_k=5)
    assert all(vec_id != 1 for vec_id, _ in hits)


def test_remove_empty_list_is_noop(index_path):
    idx = FaissVectorIndex(index_path)
    idx.add(1, _unit_vector(1, 0, 0, 0))
    idx.remove([])
    assert idx.size == 1


def test_persistence_across_reload(index_path):
    idx_a = FaissVectorIndex(index_path)
    idx_a.add(1, _unit_vector(1, 0, 0, 0))
    idx_a.add(2, _unit_vector(0, 1, 0, 0))

    idx_b = FaissVectorIndex(index_path)
    assert idx_b.size == 2
    hits = idx_b.search(_unit_vector(1, 0, 0, 0), top_k=1)
    assert hits[0][0] == 1


def test_top_k_larger_than_index_size_does_not_error(index_path):
    idx = FaissVectorIndex(index_path)
    idx.add(1, _unit_vector(1, 0, 0, 0))
    hits = idx.search(_unit_vector(1, 0, 0, 0), top_k=50)
    assert len(hits) == 1
