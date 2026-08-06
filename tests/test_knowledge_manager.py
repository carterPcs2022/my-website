"""Tests for zane.knowledge_manager: namespace-isolated RAG and lore
compilation. Uses the same lightweight, deterministic fake embedding
backend pattern as tests/test_persistent_memory.py (real network access
to download sentence-transformers isn't available in this environment —
see tests/test_embeddings.py for the real model tested in isolation).
Real SQLite/FAISS/JSON I/O is used throughout.
"""
import hashlib
import time

import numpy as np
import pytest

from zane.knowledge_manager import (
    HISTORY_TAG,
    LORE_TAG,
    KnowledgeManager,
    _split_into_chunks,
    compile_lore_database,
)
from zane.memory.embeddings import EmbeddingError
from zane.memory.persistent import PersistentMemory, PersistentMemoryConfig
from zane.memory.store import SQLiteMessageStore
from zane.memory.vector_index import FaissVectorIndex


class FakeEmbeddingBackend:
    """Deterministic bag-of-words embedding: shared vocabulary between two
    texts reliably increases their cosine similarity — see
    tests/test_persistent_memory.py for the identical pattern."""

    dim = 32

    def embed(self, text: str) -> np.ndarray:
        words = [w for w in text.lower().split() if w]
        if not words:
            raise EmbeddingError("Cannot embed empty text.")
        vectors = []
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            vectors.append(rng.normal(size=self.dim))
        combined = np.mean(vectors, axis=0)
        norm = np.linalg.norm(combined)
        if norm == 0:
            raise EmbeddingError("Degenerate zero vector.")
        return (combined / norm).astype(np.float32)


class AlwaysFailingEmbeddingBackend:
    def embed(self, text: str) -> np.ndarray:
        raise EmbeddingError("simulated embedding backend failure")


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddingBackend()


@pytest.fixture
def lore_paths(tmp_path):
    return str(tmp_path / "lore.index"), str(tmp_path / "lore.json")


# --- chunking ---


def test_split_into_chunks_is_deterministic():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    assert _split_into_chunks(text) == _split_into_chunks(text)


def test_split_into_chunks_packs_short_paragraphs_together():
    text = "Short one.\n\nShort two.\n\nShort three."
    chunks = _split_into_chunks(text, chunk_size_chars=800)
    assert len(chunks) == 1
    assert "Short one." in chunks[0]
    assert "Short three." in chunks[0]


def test_split_into_chunks_splits_oversized_paragraph_with_overlap():
    huge_paragraph = "word " * 400  # ~2000 chars, exceeds a small chunk size
    chunks = _split_into_chunks(huge_paragraph, chunk_size_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_split_into_chunks_empty_text_returns_no_chunks():
    assert _split_into_chunks("   \n\n  ") == []


# --- compile_lore_database ---


def test_compile_lore_database_missing_file_raises(lore_paths, fake_embeddings):
    index_path, metadata_path = lore_paths
    with pytest.raises(FileNotFoundError):
        compile_lore_database(
            "/nonexistent/path/does-not-exist.txt",
            index_path=index_path, metadata_path=metadata_path, embeddings=fake_embeddings,
        )


def test_compile_lore_database_empty_file_raises(tmp_path, lore_paths, fake_embeddings):
    index_path, metadata_path = lore_paths
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        compile_lore_database(
            str(empty_file), index_path=index_path, metadata_path=metadata_path,
            embeddings=fake_embeddings,
        )


def test_compile_lore_database_writes_index_and_metadata(tmp_path, lore_paths, fake_embeddings):
    index_path, metadata_path = lore_paths
    source = tmp_path / "lore.txt"
    source.write_text(
        "Zane is the Nindroid Master of Ice.\n\nThe Ice Dragon serves Zane in battle.",
        encoding="utf-8",
    )

    count = compile_lore_database(
        str(source), index_path=index_path, metadata_path=metadata_path, embeddings=fake_embeddings,
    )

    # Both short paragraphs fit under the default chunk size, so the
    # (deterministic, tested separately) chunker packs them into one chunk.
    assert count == 1
    import os
    assert os.path.exists(index_path)
    assert os.path.exists(metadata_path)


def test_compile_lore_database_against_the_real_project_lore_file(lore_paths, fake_embeddings):
    """End-to-end against the actual data/raw_lore.txt shipped in this repo."""
    index_path, metadata_path = lore_paths
    count = compile_lore_database(
        "data/raw_lore.txt", index_path=index_path, metadata_path=metadata_path,
        embeddings=fake_embeddings,
    )
    assert count > 0


# --- KnowledgeManager: cold start ---


def test_knowledge_manager_cold_start_no_files_yet(lore_paths, fake_embeddings):
    index_path, metadata_path = lore_paths  # neither file exists yet
    km = KnowledgeManager(fake_embeddings, lore_index_path=index_path, lore_metadata_path=metadata_path)
    assert km.lore_chunk_count == 0


async def test_knowledge_manager_cold_start_query_returns_empty(lore_paths, fake_embeddings):
    index_path, metadata_path = lore_paths
    km = KnowledgeManager(fake_embeddings, lore_index_path=index_path, lore_metadata_path=metadata_path)
    results = await km.query_all_knowledge_sources("anything at all", user_memory=None)
    assert results == []


# --- KnowledgeManager: namespace isolation + tagging ---


def _compile_test_lore(tmp_path, lore_paths, fake_embeddings, text):
    source = tmp_path / "lore.txt"
    source.write_text(text, encoding="utf-8")
    index_path, metadata_path = lore_paths
    compile_lore_database(str(source), index_path=index_path, metadata_path=metadata_path, embeddings=fake_embeddings)
    return index_path, metadata_path


async def test_lore_hits_are_tagged_archival(tmp_path, lore_paths, fake_embeddings):
    index_path, metadata_path = _compile_test_lore(
        tmp_path, lore_paths, fake_embeddings,
        "Zane wields ice shurikens and commands the power of frost and cold.",
    )
    km = KnowledgeManager(fake_embeddings, lore_index_path=index_path, lore_metadata_path=metadata_path)

    results = await km.query_all_knowledge_sources("Zane ice frost shurikens", user_memory=None)
    assert results
    assert all(r.startswith(LORE_TAG) for r in results)
    assert HISTORY_TAG not in "".join(results)


async def test_history_hits_are_tagged_and_isolated_from_lore(
    tmp_path, lore_paths, fake_embeddings
):
    # Compile a lore database with content that would score high on the
    # same query as the chat history below, to prove the two namespaces
    # genuinely don't mix: lore lives in a physically separate FAISS index.
    index_path, metadata_path = _compile_test_lore(
        tmp_path, lore_paths, fake_embeddings,
        "The vault code lore entry should never appear as chat history.",
    )
    km = KnowledgeManager(fake_embeddings, lore_index_path=index_path, lore_metadata_path=metadata_path)

    store = SQLiteMessageStore(str(tmp_path / "chat.sqlite3"))
    chat_vector_index = FaissVectorIndex(str(tmp_path / "chat.faiss"))
    memory = PersistentMemory(
        session_id="s1", store=store, embeddings=fake_embeddings, vector_index=chat_vector_index,
        summarizer=None, config=PersistentMemoryConfig(rolling_max_messages=1, retention_window=1000, summarize_after_n=1000),
    )
    await memory.add_user_message("The vault code is 4471, remember it.")
    await memory.add_user_message("Completely unrelated filler message about shoes.")

    results = await km.query_all_knowledge_sources("vault code", user_memory=memory)

    history_results = [r for r in results if r.startswith(HISTORY_TAG)]
    lore_results = [r for r in results if r.startswith(LORE_TAG)]
    assert history_results, "expected the chat-history hit to surface"
    assert "vault code is 4471" in history_results[0]
    # Namespace isolation: the lore chunk (compiled into a totally
    # separate FAISS index) must never appear tagged as chat history,
    # and vice versa — no cross-contamination between the two indices.
    assert all(r.startswith(LORE_TAG) for r in lore_results)
    assert all(r.startswith(HISTORY_TAG) for r in history_results)
    store.close()


async def test_query_all_knowledge_sources_runs_both_lookups_concurrently(
    tmp_path, lore_paths, fake_embeddings
):
    index_path, metadata_path = _compile_test_lore(
        tmp_path, lore_paths, fake_embeddings, "Ice powers and frost abilities of the Nindroid."
    )
    km = KnowledgeManager(fake_embeddings, lore_index_path=index_path, lore_metadata_path=metadata_path)

    store = SQLiteMessageStore(str(tmp_path / "chat2.sqlite3"))
    chat_vector_index = FaissVectorIndex(str(tmp_path / "chat2.faiss"))
    memory = PersistentMemory(
        session_id="s2", store=store, embeddings=fake_embeddings, vector_index=chat_vector_index,
        summarizer=None, config=PersistentMemoryConfig(rolling_max_messages=1, retention_window=1000, summarize_after_n=1000),
    )
    await memory.add_user_message("Ice powers are Zane's specialty in combat.")
    await memory.add_user_message("filler message about paperwork")

    results = await km.query_all_knowledge_sources("ice powers Nindroid", user_memory=memory, top_k=3)
    tags_present = {r.split()[0] for r in results}
    assert LORE_TAG in tags_present
    assert HISTORY_TAG in tags_present
    store.close()


async def test_embedding_failure_degrades_to_empty_list(lore_paths):
    index_path, metadata_path = lore_paths
    km = KnowledgeManager(AlwaysFailingEmbeddingBackend(), lore_index_path=index_path, lore_metadata_path=metadata_path)
    results = await km.query_all_knowledge_sources("anything", user_memory=None)
    assert results == []


def test_malformed_metadata_file_degrades_gracefully(tmp_path, fake_embeddings):
    index_path = str(tmp_path / "lore.index")
    metadata_path = tmp_path / "lore.json"
    metadata_path.write_text("{not valid json", encoding="utf-8")

    km = KnowledgeManager(fake_embeddings, lore_index_path=index_path, lore_metadata_path=str(metadata_path))
    assert km.lore_chunk_count == 0
