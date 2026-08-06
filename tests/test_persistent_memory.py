"""Tests for zane.memory.persistent.PersistentMemory: durability, semantic
retrieval ranking, background summarization triggers, and pruning.

Embedding and summarization are swapped for lightweight, deterministic test
doubles (real network/model access isn't available in this environment —
see tests/test_embeddings.py for the real sentence-transformers backend
tested in isolation). SQLite storage and the FAISS vector index are the
real production implementations throughout.
"""
import hashlib
from typing import List, Sequence

import numpy as np
import pytest

from zane.memory.embeddings import EmbeddingError
from zane.memory.persistent import PersistentMemory, PersistentMemoryConfig
from zane.memory.store import MessageRow, SQLiteMessageStore
from zane.memory.summarizer import SummarizationError
from zane.memory.vector_index import FaissVectorIndex


class FakeEmbeddingBackend:
    """Deterministic bag-of-words embedding: each distinct word hashes to a
    fixed pseudo-random unit vector, and a text's embedding is the
    normalized average of its words' vectors. Shared vocabulary between two
    texts reliably increases their cosine similarity, which is all the
    ranking behavior under test actually needs."""

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


class FakeSummarizer:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: List[List[MessageRow]] = []

    async def summarize(self, messages: Sequence[MessageRow]) -> str:
        self.calls.append(list(messages))
        if self.fail:
            raise SummarizationError("simulated summarization failure")
        contents = ", ".join(m.content for m in messages)
        return f"Summary covering: {contents}"


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "mem.sqlite3")


@pytest.fixture
def index_path(tmp_path):
    return str(tmp_path / "mem.faiss")


def make_memory(
    db_path,
    index_path,
    session_id="s1",
    embeddings=None,
    summarizer=None,
    **config_overrides,
) -> PersistentMemory:
    store = SQLiteMessageStore(db_path)
    vector_index = FaissVectorIndex(index_path)
    return PersistentMemory(
        session_id=session_id,
        store=store,
        embeddings=embeddings or FakeEmbeddingBackend(),
        vector_index=vector_index,
        summarizer=summarizer or FakeSummarizer(),
        config=PersistentMemoryConfig(**config_overrides),
    )


async def test_write_durability_across_restart(db_path, index_path):
    memory_a = make_memory(db_path, index_path, session_id="durable-session")
    await memory_a.add_user_message("first message")
    await memory_a.add_assistant_message("first reply")

    # A brand new PersistentMemory pointed at the same underlying SQLite
    # file (simulating a process restart) must recover the conversation.
    memory_b = make_memory(db_path, index_path, session_id="durable-session")
    messages = memory_b.get_messages()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "first message"
    assert messages[1]["content"] == "first reply"


async def test_cold_start_empty_history(db_path, index_path):
    memory = make_memory(db_path, index_path, session_id="brand-new")
    assert memory.get_messages() == []
    assert memory.get_latest_summary() is None
    result = await memory.retrieve_relevant("anything at all")
    assert result == []


async def test_retrieval_relevance_ranking_excludes_recent_window(db_path, index_path):
    # rolling_max_messages=2 keeps only the two most recent turns in the
    # in-context window/recent-id set, forcing retrieval to reach further
    # back into genuinely "older" history for anything before that.
    memory = make_memory(
        db_path, index_path, session_id="ranking-test",
        rolling_max_messages=2, retrieval_top_k=2,
        retention_window=1000, summarize_after_n=1000,  # disable pruning for this test
    )

    await memory.add_user_message("ice frost winter snow cold")
    await memory.add_assistant_message("fire flame heat lava volcano")
    await memory.add_user_message("completely unrelated filler about shoes")
    await memory.add_assistant_message("more filler about spreadsheets")

    results = await memory.retrieve_relevant("ice frost cold")
    assert results, "expected at least one retrieved result"
    assert "ice frost winter snow cold" in results[0]
    # The fire-themed message shares no vocabulary with the query, so if it
    # appears at all it must rank behind the ice-themed one.
    if len(results) > 1:
        assert "ice frost winter snow cold" in results[0]
        assert "fire flame heat lava volcano" not in results[0]


async def test_retrieve_relevant_accepts_precomputed_query_vector(db_path, index_path, monkeypatch):
    memory = make_memory(
        db_path, index_path, session_id="precomputed-vector-test",
        rolling_max_messages=2, retrieval_top_k=2,
        retention_window=1000, summarize_after_n=1000,
    )
    await memory.add_user_message("ice frost winter snow cold")
    await memory.add_user_message("completely unrelated filler about shoes")
    await memory.add_user_message("more filler about spreadsheets")
    await memory.add_user_message("even more filler about paperwork")

    embed_calls = []
    original_embed = memory._embeddings.embed

    def _spying_embed(text):
        embed_calls.append(text)
        return original_embed(text)

    monkeypatch.setattr(memory._embeddings, "embed", _spying_embed)

    precomputed = original_embed("ice frost cold")
    results = await memory.retrieve_relevant("ice frost cold", query_vector=precomputed)

    assert results, "expected at least one retrieved result"
    assert "ice frost winter snow cold" in results[0]
    # The precomputed vector must be used directly — retrieve_relevant
    # itself must not call .embed() again for the query text.
    assert embed_calls == []


async def test_retrieve_relevant_never_returns_recent_window_messages(db_path, index_path):
    memory = make_memory(
        db_path, index_path, session_id="dedupe-test",
        rolling_max_messages=2, retrieval_top_k=5,
        retention_window=1000, summarize_after_n=1000,
    )
    await memory.add_user_message("alpha beta gamma")
    await memory.add_user_message("alpha beta gamma delta")  # still in rolling window

    results = await memory.retrieve_relevant("alpha beta gamma")
    # Both messages are near-identical in vocabulary, but the second one is
    # still within the rolling window (maxlen=2) alongside the first, so
    # neither should be surfaced as "relevant past context" yet.
    assert results == []


async def test_summarization_triggers_and_prunes_oldest_chunk(db_path, index_path):
    summarizer = FakeSummarizer(fail=False)
    memory = make_memory(
        db_path, index_path, session_id="prune-test",
        retention_window=2, summarize_after_n=2,
        rolling_max_messages=100,
    )
    memory._summarizer = summarizer  # swap in the spy after construction

    for i in range(4):
        await memory.add_user_message(f"msg-{i}")

    remaining = memory._store.get_all_messages_ordered("prune-test")
    assert [r.content for r in remaining] == ["msg-2", "msg-3"]

    summaries = memory._store.get_summaries("prune-test")
    assert len(summaries) == 1
    assert "msg-0" in summaries[0].summary_text
    assert "msg-1" in summaries[0].summary_text

    # Pruned messages' vectors must be removed from the index too.
    assert memory._vector_index.size == 2


async def test_failed_summarization_retains_raw_messages(db_path, index_path):
    memory = make_memory(
        db_path, index_path, session_id="fail-prune-test",
        retention_window=2, summarize_after_n=2,
        rolling_max_messages=100,
    )
    memory._summarizer = FakeSummarizer(fail=True)

    for i in range(4):
        await memory.add_user_message(f"msg-{i}")

    # Summarization failed, so nothing should have been pruned or summarized.
    remaining = memory._store.get_all_messages_ordered("fail-prune-test")
    assert len(remaining) == 4
    assert memory._store.get_summaries("fail-prune-test") == []


async def test_embedding_failure_degrades_gracefully(db_path, index_path):
    memory = make_memory(
        db_path, index_path, session_id="embed-fail-test",
        embeddings=AlwaysFailingEmbeddingBackend(),
    )

    # Must not raise even though every embed() call fails.
    await memory.add_user_message("this will not be embedded")

    stored = memory._store.get_all_messages_ordered("embed-fail-test")
    assert len(stored) == 1
    assert stored[0].content == "this will not be embedded"
    assert memory._vector_index.size == 0

    result = await memory.retrieve_relevant("this will not be embedded")
    assert result == []


async def test_reset_rolling_window_preserves_durable_history(db_path, index_path):
    memory = make_memory(db_path, index_path, session_id="reset-test")
    await memory.add_user_message("hello")
    await memory.add_assistant_message("hi there")

    memory.reset_rolling_window()
    assert memory.get_messages() == []

    # Durable history survives the reset even though the in-context window
    # was cleared.
    stored = memory._store.get_all_messages_ordered("reset-test")
    assert len(stored) == 2
