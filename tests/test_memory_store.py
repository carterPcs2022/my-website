import sqlite3

import pytest

from zane.memory.store import SQLiteMessageStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "zane_test.sqlite3")


def test_insert_and_get_recent_messages(db_path):
    store = SQLiteMessageStore(db_path)
    store.insert_message("session-a", "user", "Hello", timestamp=1.0)
    store.insert_message("session-a", "assistant", "Greetings.", timestamp=2.0)

    rows = store.get_recent_messages("session-a", limit=10)
    assert [r.role for r in rows] == ["user", "assistant"]
    assert [r.content for r in rows] == ["Hello", "Greetings."]
    store.close()


def test_write_durability_across_reopen(db_path):
    """A message written and the connection closed must still be readable
    from a brand new SQLiteMessageStore instance pointed at the same file —
    this is the "durability across restarts" requirement."""
    store_a = SQLiteMessageStore(db_path)
    msg_id = store_a.insert_message("session-b", "user", "Remember this.", timestamp=10.0)
    store_a.close()

    store_b = SQLiteMessageStore(db_path)
    rows = store_b.get_recent_messages("session-b", limit=10)
    assert len(rows) == 1
    assert rows[0].id == msg_id
    assert rows[0].content == "Remember this."
    store_b.close()


def test_get_messages_by_ids_and_empty_list(db_path):
    store = SQLiteMessageStore(db_path)
    id1 = store.insert_message("s", "user", "one", timestamp=1.0)
    id2 = store.insert_message("s", "assistant", "two", timestamp=2.0)

    rows = store.get_messages_by_ids([id1, id2])
    assert {r.content for r in rows} == {"one", "two"}

    assert store.get_messages_by_ids([]) == []
    store.close()


def test_prunable_messages_respects_retention_window(db_path):
    store = SQLiteMessageStore(db_path)
    ids = [store.insert_message("s", "user", f"msg-{i}", timestamp=float(i)) for i in range(10)]

    prunable = store.get_prunable_messages("s", retention_window=4)
    assert [r.content for r in prunable] == [f"msg-{i}" for i in range(6)]
    assert store.count_prunable_messages("s", retention_window=4) == 6

    # Fewer messages than the retention window -> nothing is prunable.
    assert store.get_prunable_messages("s", retention_window=100) == []
    assert store.count_prunable_messages("s", retention_window=100) == 0
    store.close()


def test_delete_messages_prunes_rows(db_path):
    store = SQLiteMessageStore(db_path)
    ids = [store.insert_message("s", "user", f"msg-{i}", timestamp=float(i)) for i in range(5)]

    store.delete_messages(ids[:2])
    remaining = store.get_all_messages_ordered("s")
    assert [r.content for r in remaining] == ["msg-2", "msg-3", "msg-4"]

    # Deleting an empty list is a safe no-op.
    store.delete_messages([])
    assert len(store.get_all_messages_ordered("s")) == 3
    store.close()


def test_summary_roundtrip(db_path):
    store = SQLiteMessageStore(db_path)
    assert store.get_latest_summary("s") is None

    store.insert_summary("s", time_range_start=1.0, time_range_end=5.0, summary_text="First chunk.")
    store.insert_summary("s", time_range_start=6.0, time_range_end=10.0, summary_text="Second chunk.")

    summaries = store.get_summaries("s")
    assert [s.summary_text for s in summaries] == ["First chunk.", "Second chunk."]

    latest = store.get_latest_summary("s")
    assert latest is not None
    assert latest.summary_text == "Second chunk."
    store.close()


def test_sessions_are_isolated(db_path):
    store = SQLiteMessageStore(db_path)
    store.insert_message("session-1", "user", "from session 1", timestamp=1.0)
    store.insert_message("session-2", "user", "from session 2", timestamp=1.0)

    rows_1 = store.get_all_messages_ordered("session-1")
    rows_2 = store.get_all_messages_ordered("session-2")
    assert [r.content for r in rows_1] == ["from session 1"]
    assert [r.content for r in rows_2] == ["from session 2"]
    store.close()


def test_cold_start_empty_history(db_path):
    store = SQLiteMessageStore(db_path)
    assert store.get_recent_messages("brand-new-session", limit=10) == []
    assert store.get_all_messages_ordered("brand-new-session") == []
    assert store.count_prunable_messages("brand-new-session", retention_window=5) == 0
    assert store.get_latest_summary("brand-new-session") is None
    store.close()


def test_schema_created_on_fresh_file(db_path):
    store = SQLiteMessageStore(db_path)
    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"messages", "summaries"} <= tables
    conn.close()
    store.close()
