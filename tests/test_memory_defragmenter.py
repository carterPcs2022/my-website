import time

import numpy as np
import pytest

from zane.groq_client import GroqUnavailableError
from zane.memory.memory_defragmenter import CONSOLIDATED_ROLE, MemoryDefragmenter
from zane.memory.store import SQLiteMessageStore


class _FakeEmbeddings:
    """Same word -> same vector, so two "vault code" messages score as
    near-identical while unrelated text scores as orthogonal."""

    def embed(self, text):
        if "vault code" in text.lower():
            return np.array([1.0, 0.0], dtype="float32")
        return np.array([0.0, 1.0], dtype="float32")


class _FakeVectorIndex:
    def __init__(self):
        self.vectors = {}

    def add(self, vec_id, vector):
        self.vectors[vec_id] = vector

    def search(self, vector, top_k):
        scored = sorted(
            self.vectors.items(), key=lambda kv: -float(np.dot(kv[1], vector))
        )
        return [(vec_id, float(np.dot(v, vector))) for vec_id, v in scored[:top_k]]

    def remove(self, ids):
        for vec_id in ids:
            self.vectors.pop(vec_id, None)


class _FakeGroq:
    def __init__(self, response_text: str = None, fail: bool = False):
        self.response_text = response_text or (
            "CONFLICT: yes\nRESOLUTION: The vault code is 9981, per the most recent record."
        )
        self.fail = fail
        self.calls = 0

    async def chat_completion(self, messages, **kwargs):
        self.calls += 1
        if self.fail:
            raise GroqUnavailableError("simulated Groq outage")

        class _Msg:
            content = self.response_text

        class _Choice:
            message = _Msg()

        class _Completion:
            choices = [_Choice()]

        return _Completion()


@pytest.fixture
def store(tmp_path):
    s = SQLiteMessageStore(str(tmp_path / "defrag.sqlite3"))
    yield s
    s.close()


def _seed_conflicting_pair(store, embeddings, vector_index, older_text, newer_text):
    older_id = store.insert_message("s1", "user", older_text, timestamp=time.time() - 10000)
    vector_index.add(older_id, embeddings.embed(older_text))
    store.set_embedding_id(older_id, older_id)

    newer_id = store.insert_message("s1", "user", newer_text, timestamp=time.time())
    vector_index.add(newer_id, embeddings.embed(newer_text))
    store.set_embedding_id(newer_id, newer_id)

    return older_id, newer_id


async def test_confirmed_conflict_prunes_older_and_writes_consolidated_entry(store):
    embeddings = _FakeEmbeddings()
    vector_index = _FakeVectorIndex()
    groq = _FakeGroq()
    older_id, newer_id = _seed_conflicting_pair(
        store, embeddings, vector_index, "The vault code is 4471.", "The vault code is 9981."
    )

    defrag = MemoryDefragmenter(store, embeddings, vector_index, groq, similarity_threshold=0.5)
    newer_row = store.get_messages_by_ids([newer_id])[0]
    resolution = await defrag.check_for_conflicts(newer_row)

    assert resolution is not None
    assert resolution.pruned_message_id == older_id
    assert resolution.kept_message_id == newer_id

    remaining = store.get_all_messages_ordered("s1")
    remaining_ids = {r.id for r in remaining}
    assert older_id not in remaining_ids
    assert newer_id in remaining_ids
    assert any(r.role == CONSOLIDATED_ROLE for r in remaining)
    assert older_id not in vector_index.vectors


async def test_no_conflict_reported_leaves_both_entries_untouched(store):
    embeddings = _FakeEmbeddings()
    vector_index = _FakeVectorIndex()
    groq = _FakeGroq(response_text="CONFLICT: no\nRESOLUTION: n/a")
    older_id, newer_id = _seed_conflicting_pair(
        store, embeddings, vector_index,
        "The vault code was 4471 last week.", "The vault code is 9981 this week.",
    )

    defrag = MemoryDefragmenter(store, embeddings, vector_index, groq, similarity_threshold=0.5)
    newer_row = store.get_messages_by_ids([newer_id])[0]
    resolution = await defrag.check_for_conflicts(newer_row)

    assert resolution is None
    remaining_ids = {r.id for r in store.get_all_messages_ordered("s1")}
    assert older_id in remaining_ids
    assert newer_id in remaining_ids


async def test_unparsable_llm_response_never_deletes_anything(store):
    embeddings = _FakeEmbeddings()
    vector_index = _FakeVectorIndex()
    groq = _FakeGroq(response_text="I'm not sure, maybe?")
    older_id, newer_id = _seed_conflicting_pair(
        store, embeddings, vector_index, "The vault code is 4471.", "The vault code is 9981."
    )

    defrag = MemoryDefragmenter(store, embeddings, vector_index, groq, similarity_threshold=0.5)
    newer_row = store.get_messages_by_ids([newer_id])[0]
    resolution = await defrag.check_for_conflicts(newer_row)

    assert resolution is None
    remaining_ids = {r.id for r in store.get_all_messages_ordered("s1")}
    assert older_id in remaining_ids and newer_id in remaining_ids


async def test_groq_unavailable_degrades_to_no_action(store):
    embeddings = _FakeEmbeddings()
    vector_index = _FakeVectorIndex()
    groq = _FakeGroq(fail=True)
    older_id, newer_id = _seed_conflicting_pair(
        store, embeddings, vector_index, "The vault code is 4471.", "The vault code is 9981."
    )

    defrag = MemoryDefragmenter(store, embeddings, vector_index, groq, similarity_threshold=0.5)
    newer_row = store.get_messages_by_ids([newer_id])[0]
    resolution = await defrag.check_for_conflicts(newer_row)

    assert resolution is None
    remaining_ids = {r.id for r in store.get_all_messages_ordered("s1")}
    assert older_id in remaining_ids and newer_id in remaining_ids


async def test_unrelated_messages_below_similarity_threshold_are_never_compared(store):
    embeddings = _FakeEmbeddings()
    vector_index = _FakeVectorIndex()
    groq = _FakeGroq()
    _seed_conflicting_pair(
        store, embeddings, vector_index, "The vault code is 4471.", "I like ice cream."
    )

    defrag = MemoryDefragmenter(store, embeddings, vector_index, groq, similarity_threshold=0.9)
    newer_row = store.get_messages_by_ids(
        [r.id for r in store.get_all_messages_ordered("s1") if r.content == "I like ice cream."]
    )[0]
    resolution = await defrag.check_for_conflicts(newer_row)

    assert resolution is None
    assert groq.calls == 0  # never even asked the LLM: not similar enough to be candidates


def test_recency_weight_decays_with_age():
    embeddings = _FakeEmbeddings()
    vector_index = _FakeVectorIndex()
    groq = _FakeGroq()
    defrag = MemoryDefragmenter(store=None, embeddings=embeddings, vector_index=vector_index, groq_client=groq, half_life_s=3600.0)

    now = time.time()
    fresh_weight = defrag._recency_weight(now, now)
    half_life_weight = defrag._recency_weight(now - 3600.0, now)
    old_weight = defrag._recency_weight(now - 3600.0 * 10, now)

    assert fresh_weight == pytest.approx(1.0)
    assert half_life_weight == pytest.approx(0.5, abs=0.01)
    assert old_weight < half_life_weight


async def test_run_forever_sweeps_and_stops(store):
    embeddings = _FakeEmbeddings()
    vector_index = _FakeVectorIndex()
    groq = _FakeGroq()
    _seed_conflicting_pair(
        store, embeddings, vector_index, "The vault code is 4471.", "The vault code is 9981."
    )

    defrag = MemoryDefragmenter(
        store, embeddings, vector_index, groq, similarity_threshold=0.5, interval_s=0.01
    )

    import asyncio

    task = asyncio.create_task(defrag.run_forever())
    await asyncio.sleep(0.05)
    defrag.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    remaining = store.get_all_messages_ordered("s1")
    assert any(r.role == CONSOLIDATED_ROLE for r in remaining)
