"""Turso-backed durable memory for P.I.X.A.L.

The in-process :class:`PixalMemoryStore` remains the public behavior model;
this adapter persists the same bounded entries in P.I.X.A.L.'s own Turso
database so Render restarts and redeploys do not erase companion memory.

Environment variables:
    PIXAL_TDB_URL: P.I.X.A.L.'s Turso/libSQL database URL.
    PIXAL_TAT: P.I.X.A.L.'s Turso auth token.

Secrets are read only from the process environment and are never logged.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Iterable, List, Optional

from zane.pixal_memory import PixalMemoryEntry, PixalMemoryStore

logger = logging.getLogger("zane.pixal_turso_memory")

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pixal_memories (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    role TEXT NOT NULL,
    importance REAL NOT NULL,
    tags_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL
)
"""


class PixalTursoConfigurationError(RuntimeError):
    """Raised when P.I.X.A.L.'s Turso configuration is incomplete."""


class TursoPixalMemoryStore(PixalMemoryStore):
    """Bounded P.I.X.A.L. memory synchronized with a dedicated Turso DB."""

    def __init__(
        self,
        *,
        max_entries: int = 100,
        max_content_length: int = 1000,
        client: Any | None = None,
    ) -> None:
        super().__init__(max_entries=max_entries, max_content_length=max_content_length)
        self._client = client or self._build_client()
        self._ensure_schema()
        self._load()
        self._enforce_capacity()

    @staticmethod
    def is_configured() -> bool:
        return bool(os.getenv("PIXAL_TDB_URL") and os.getenv("PIXAL_TAT"))

    @staticmethod
    def _build_client() -> Any:
        url = os.getenv("PIXAL_TDB_URL")
        token = os.getenv("PIXAL_TAT")
        if not url or not token:
            raise PixalTursoConfigurationError(
                "P.I.X.A.L. Turso requires PIXAL_TDB_URL and PIXAL_TAT."
            )
        try:
            from libsql_client import create_client_sync
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise PixalTursoConfigurationError(
                "P.I.X.A.L. Turso persistence requires the libsql-client package."
            ) from exc
        return create_client_sync(url, auth_token=token)

    def _ensure_schema(self) -> None:
        self._client.execute(_TABLE_SQL)

    def _load(self) -> None:
        result = self._client.execute(
            "SELECT memory_id, content, role, importance, tags_json, "
            "created_at, last_accessed_at FROM pixal_memories "
            "ORDER BY created_at ASC"
        )
        loaded: List[PixalMemoryEntry] = []
        for row in result.rows:
            loaded.append(
                PixalMemoryEntry(
                    content=str(row[1]),
                    role=str(row[2]),
                    importance=float(row[3]),
                    tags=tuple(json.loads(str(row[4]))),
                    created_at=float(row[5]),
                    last_accessed_at=float(row[6]),
                    memory_id=str(row[0]),
                )
            )
        self._entries = loaded

    def _persist(self, entry: PixalMemoryEntry) -> None:
        data = asdict(entry)
        self._client.execute(
            "INSERT INTO pixal_memories "
            "(memory_id, content, role, importance, tags_json, created_at, last_accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(memory_id) DO UPDATE SET "
            "content=excluded.content, role=excluded.role, importance=excluded.importance, "
            "tags_json=excluded.tags_json, created_at=excluded.created_at, "
            "last_accessed_at=excluded.last_accessed_at",
            (
                entry.memory_id,
                data["content"],
                data["role"],
                data["importance"],
                json.dumps(list(entry.tags)),
                data["created_at"],
                data["last_accessed_at"],
            ),
        )

    def _delete_from_db(self, memory_id: str) -> None:
        self._client.execute(
            "DELETE FROM pixal_memories WHERE memory_id = ?", (memory_id,)
        )

    def _enforce_capacity(self) -> None:
        while len(self._entries) > self.max_entries:
            victim = min(
                self._entries,
                key=lambda entry: (entry.importance, entry.last_accessed_at, entry.created_at),
            )
            self._entries.remove(victim)
            self._delete_from_db(victim.memory_id)

    def remember(
        self,
        content: str,
        *,
        role: str = "system",
        importance: float = 0.5,
        tags: Iterable[str] = (),
    ) -> PixalMemoryEntry:
        entry = super().remember(
            content, role=role, importance=importance, tags=tags
        )
        self._persist(entry)
        # The base class may have evicted an entry before we persisted the new
        # one; remove any rows that are no longer represented in memory.
        current_ids = {item.memory_id for item in self._entries}
        result = self._client.execute("SELECT memory_id FROM pixal_memories")
        for row in result.rows:
            if str(row[0]) not in current_ids:
                self._delete_from_db(str(row[0]))
        return entry

    def relevant(self, query: str, *, limit: int = 5) -> List[PixalMemoryEntry]:
        results = super().relevant(query, limit=limit)
        for entry in results:
            self._persist(entry)
        return results

    def forget(self, memory_id: str) -> bool:
        removed = super().forget(memory_id)
        if removed:
            self._delete_from_db(memory_id)
        return removed

    def clear(self) -> None:
        super().clear()
        self._client.execute("DELETE FROM pixal_memories")

    def ping(self) -> float:
        """Return a real Turso round-trip latency measurement in milliseconds."""
        import time

        start = time.monotonic()
        self._client.execute("SELECT 1")
        return (time.monotonic() - start) * 1000

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()


def build_pixal_memory_store(
    *, max_entries: int = 100, max_content_length: int = 1000
) -> PixalMemoryStore:
    """Build durable P.I.X.A.L. memory when configured, else local memory."""
    if TursoPixalMemoryStore.is_configured():
        try:
            return TursoPixalMemoryStore(
                max_entries=max_entries,
                max_content_length=max_content_length,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "P.I.X.A.L. Turso memory unavailable; falling back to in-process memory."
            )
    return PixalMemoryStore(
        max_entries=max_entries,
        max_content_length=max_content_length,
    )
