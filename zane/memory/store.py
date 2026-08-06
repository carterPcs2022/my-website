"""SQLite-backed durable storage for Zane's conversational memory.

Every message is written immediately on send/receive so conversation
history survives process restarts. This module is intentionally
synchronous (plain `sqlite3`); callers from async code (see
`zane/memory/persistent.py`) wrap calls in `asyncio.to_thread`.
"""
from __future__ import annotations

import logging
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger("zane.memory.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    embedding_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_session_ts ON messages(session_id, timestamp);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    time_range_start REAL NOT NULL,
    time_range_end REAL NOT NULL,
    summary_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, time_range_start);
"""


class MemoryStoreError(RuntimeError):
    """Raised when a durable-storage operation fails after all retries."""


@dataclass
class MessageRow:
    id: int
    session_id: str
    role: str
    content: str
    timestamp: float
    embedding_id: Optional[int]


@dataclass
class SummaryRow:
    id: int
    session_id: str
    time_range_start: float
    time_range_end: float
    summary_text: str


class SQLiteMessageStore:
    """Thread-safe wrapper around a single SQLite connection.

    A single connection is shared (via `check_same_thread=False`) and all
    access is serialized through `_lock`, since sqlite3 connections are not
    safe for truly concurrent use across threads even when that flag is
    set. WAL mode is enabled so readers don't block on a writer holding a
    short-lived lock; `_execute_with_retry` additionally retries on
    `database is locked` errors from contention with *other* processes
    /connections against the same file.
    """

    def __init__(self, db_path: str, max_retries: int = 5, base_retry_delay_s: float = 0.05) -> None:
        self._lock = threading.RLock()
        self._max_retries = max_retries
        self._base_retry_delay_s = base_retry_delay_s
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _execute_with_retry(self, fn, *args, **kwargs):
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                with self._lock:
                    return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if "locked" not in str(exc).lower() or attempt >= self._max_retries:
                    raise MemoryStoreError(f"SQLite operation failed: {exc}") from exc
                delay = self._base_retry_delay_s * (2 ** attempt) + random.uniform(0, 0.02)
                logger.warning(
                    "SQLite database locked, retrying in %.3fs [attempt %d/%d]",
                    delay, attempt + 1, self._max_retries,
                )
                time.sleep(delay)
            except sqlite3.Error as exc:
                raise MemoryStoreError(f"SQLite operation failed: {exc}") from exc

        raise MemoryStoreError(
            f"SQLite database still locked after {self._max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    # --- messages ---

    def insert_message(
        self, session_id: str, role: str, content: str, timestamp: Optional[float] = None
    ) -> int:
        ts = timestamp if timestamp is not None else time.time()

        def _do():
            cur = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, ts),
            )
            self._conn.commit()
            return cur.lastrowid

        return self._execute_with_retry(_do)

    def set_embedding_id(self, message_id: int, embedding_id: int) -> None:
        def _do():
            self._conn.execute(
                "UPDATE messages SET embedding_id = ? WHERE id = ?", (embedding_id, message_id)
            )
            self._conn.commit()

        self._execute_with_retry(_do)

    def get_recent_messages(self, session_id: str, limit: int) -> List[MessageRow]:
        def _do():
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [self._row_to_message(r) for r in reversed(rows)]

        return self._execute_with_retry(_do)

    def get_all_messages_ordered(self, session_id: str) -> List[MessageRow]:
        def _do():
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
            return [self._row_to_message(r) for r in rows]

        return self._execute_with_retry(_do)

    def get_messages_by_ids(self, ids: Sequence[int]) -> List[MessageRow]:
        if not ids:
            return []

        def _do():
            placeholders = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE id IN ({placeholders})", tuple(ids)
            ).fetchall()
            return [self._row_to_message(r) for r in rows]

        return self._execute_with_retry(_do)

    def get_prunable_messages(self, session_id: str, retention_window: int) -> List[MessageRow]:
        """Every message for this session older than the most recent
        `retention_window` rows, oldest first."""
        all_rows = self.get_all_messages_ordered(session_id)
        if len(all_rows) <= retention_window:
            return []
        return all_rows[: len(all_rows) - retention_window]

    def count_prunable_messages(self, session_id: str, retention_window: int) -> int:
        return len(self.get_prunable_messages(session_id, retention_window))

    def delete_messages(self, ids: Sequence[int]) -> None:
        if not ids:
            return

        def _do():
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", tuple(ids))
            self._conn.commit()

        self._execute_with_retry(_do)

    # --- summaries ---

    def insert_summary(
        self, session_id: str, time_range_start: float, time_range_end: float, summary_text: str
    ) -> int:
        def _do():
            cur = self._conn.execute(
                "INSERT INTO summaries (session_id, time_range_start, time_range_end, summary_text) "
                "VALUES (?, ?, ?, ?)",
                (session_id, time_range_start, time_range_end, summary_text),
            )
            self._conn.commit()
            return cur.lastrowid

        return self._execute_with_retry(_do)

    def get_summaries(self, session_id: str) -> List[SummaryRow]:
        def _do():
            rows = self._conn.execute(
                "SELECT * FROM summaries WHERE session_id = ? ORDER BY time_range_start ASC",
                (session_id,),
            ).fetchall()
            return [
                SummaryRow(
                    id=r["id"],
                    session_id=r["session_id"],
                    time_range_start=r["time_range_start"],
                    time_range_end=r["time_range_end"],
                    summary_text=r["summary_text"],
                )
                for r in rows
            ]

        return self._execute_with_retry(_do)

    def get_latest_summary(self, session_id: str) -> Optional[SummaryRow]:
        summaries = self.get_summaries(session_id)
        return summaries[-1] if summaries else None

    def get_recently_embedded_messages(self, limit: int) -> List[MessageRow]:
        """The most recently written messages that have a non-null
        embedding, newest first, across every session. Used by
        `zane.memory.memory_defragmenter` to sweep for conflicts without
        needing a per-write hook into `PersistentMemory`."""

        def _do():
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE embedding_id IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_message(r) for r in rows]

        return self._execute_with_retry(_do)

    def ping(self) -> float:
        """Times a trivial round-trip query against the database and
        returns the elapsed time in milliseconds. Used by
        `zane.falcon_worker` as a real (not simulated) DB responsiveness
        probe."""
        start = time.monotonic()

        def _do():
            self._conn.execute("SELECT 1").fetchone()

        self._execute_with_retry(_do)
        return (time.monotonic() - start) * 1000

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> MessageRow:
        return MessageRow(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
            embedding_id=row["embedding_id"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
