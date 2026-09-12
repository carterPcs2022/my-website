from __future__ import annotations

import json

from zane.pixal_turso_memory import TursoPixalMemoryStore, build_pixal_memory_store


class FakeResult:
    def __init__(self, rows=()):
        self.rows = list(rows)


class FakeTursoClient:
    def __init__(self):
        self.rows = {}
        self.closed = False

    def execute(self, sql, args=()):
        sql_upper = " ".join(sql.upper().split())
        if sql_upper.startswith("CREATE TABLE") or sql_upper.startswith("CREATE INDEX"):
            return FakeResult()
        if sql_upper.startswith("INSERT INTO PIXAL_MEMORIES"):
            memory_id, content, role, importance, tags_json, created_at, last_accessed_at = args
            self.rows[memory_id] = (
                memory_id,
                content,
                role,
                importance,
                tags_json,
                created_at,
                last_accessed_at,
            )
            return FakeResult()
        if sql_upper.startswith("SELECT MEMORY_ID, CONTENT"):
            return FakeResult(sorted(self.rows.values(), key=lambda row: row[5]))
        if sql_upper.startswith("SELECT MEMORY_ID FROM PIXAL_MEMORIES"):
            return FakeResult([(memory_id,) for memory_id in self.rows])
        if sql_upper.startswith("DELETE FROM PIXAL_MEMORIES WHERE"):
            self.rows.pop(args[0], None)
            return FakeResult()
        if sql_upper.startswith("DELETE FROM PIXAL_MEMORIES"):
            self.rows.clear()
            return FakeResult()
        if sql_upper.startswith("SELECT 1"):
            return FakeResult([(1,)])
        raise AssertionError(f"Unhandled SQL: {sql}")

    def close(self):
        self.closed = True


def test_turso_memory_loads_and_persists_entries():
    client = FakeTursoClient()
    store = TursoPixalMemoryStore(client=client, max_entries=10)

    entry = store.remember(
        "Battery telemetry was nominal.",
        importance=0.8,
        tags=("telemetry", "nominal"),
    )

    restored = TursoPixalMemoryStore(client=client, max_entries=10)
    assert restored.snapshot()[0]["memory_id"] == entry.memory_id
    assert restored.relevant("battery telemetry")[0].content == "Battery telemetry was nominal."
    assert json.loads(client.rows[entry.memory_id][4]) == ["nominal", "telemetry"]


def test_turso_memory_deletes_forgotten_entry():
    client = FakeTursoClient()
    store = TursoPixalMemoryStore(client=client)
    entry = store.remember("Remember this event.")

    assert store.forget(entry.memory_id) is True
    assert entry.memory_id not in client.rows


def test_turso_memory_enforces_bound_and_deletes_evicted_rows():
    client = FakeTursoClient()
    store = TursoPixalMemoryStore(client=client, max_entries=2)
    first = store.remember("low importance", importance=0.1)
    store.remember("high importance", importance=0.9)
    store.remember("new event", importance=0.8)

    assert first.memory_id not in client.rows
    assert len(client.rows) == 2


def test_builder_uses_local_store_when_pixal_turso_is_not_configured(monkeypatch):
    monkeypatch.delenv("PIXAL_TDB_URL", raising=False)
    monkeypatch.delenv("PIXAL_TAT", raising=False)

    store = build_pixal_memory_store()
    assert type(store).__name__ == "PixalMemoryStore"


def test_builder_uses_turso_when_pixal_turso_is_configured(monkeypatch):
    monkeypatch.setenv("PIXAL_TDB_URL", "libsql://pixal.example")
    monkeypatch.setenv("PIXAL_TAT", "test-only-token")

    class FakeStore(TursoPixalMemoryStore):
        def __init__(self, **kwargs):
            super().__init__(client=FakeTursoClient(), **kwargs)

    monkeypatch.setattr("zane.pixal_turso_memory.TursoPixalMemoryStore", FakeStore)
    store = build_pixal_memory_store()
    assert isinstance(store, FakeStore)
