from zane.pixal_memory import PixalMemoryStore


def test_memory_is_bounded_and_trims_content() -> None:
    store = PixalMemoryStore(max_entries=2, max_content_length=5)
    first = store.remember("abcdef", importance=0.1)
    store.remember("second", importance=0.9)
    store.remember("third", importance=0.9)

    assert len(store.snapshot()) == 2
    assert first.memory_id not in {item["memory_id"] for item in store.snapshot()}
    assert all(len(str(item["content"])) <= 5 for item in store.snapshot())


def test_relevant_prefers_keyword_matches() -> None:
    store = PixalMemoryStore()
    store.remember("battery warning", tags=["power"], importance=0.4)
    store.remember("weather report", tags=["environment"], importance=0.9)

    results = store.relevant("battery power")
    assert results[0].content == "battery warning"


def test_forget_removes_memory() -> None:
    store = PixalMemoryStore()
    entry = store.remember("remember this")
    assert store.forget(entry.memory_id) is True
    assert store.forget(entry.memory_id) is False
    assert store.snapshot() == []
