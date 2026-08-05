"""Tests for the real sentence-transformers-backed embedding model.

`EmbeddingBackend.embed()` needs to download model weights on first use in
environments without a local HuggingFace cache. That download requires
network access, which may not be available in a sandboxed CI/test
environment — so these tests attempt the real model and skip (rather than
fail) if it can't be loaded, while still asserting real behavior whenever
it can be.
"""
import numpy as np
import pytest

from zane.memory.embeddings import EmbeddingBackend, EmbeddingError


@pytest.fixture(scope="module")
def backend():
    b = EmbeddingBackend("all-MiniLM-L6-v2")
    try:
        b.embed("warm up the model")
    except EmbeddingError as exc:
        pytest.skip(f"Embedding model unavailable in this environment: {exc}")
    return b


def test_embed_returns_normalized_float32_vector(backend):
    vector = backend.embed("Zane is loyal to his family.")
    assert vector.dtype == np.float32
    assert vector.shape == (backend.dimension,)
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-3


def test_similar_sentences_score_higher_than_unrelated(backend):
    a = backend.embed("What is the weather like today?")
    b = backend.embed("Can you tell me today's forecast?")
    c = backend.embed("I would like to calculate a success probability.")

    sim_related = float(np.dot(a, b))
    sim_unrelated = float(np.dot(a, c))
    assert sim_related > sim_unrelated


def test_empty_text_raises_embedding_error(backend):
    with pytest.raises(EmbeddingError):
        backend.embed("")


def test_missing_dependency_raises_embedding_error(monkeypatch):
    b = EmbeddingBackend("all-MiniLM-L6-v2")

    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(EmbeddingError):
        b.embed("hello")
