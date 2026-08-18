"""Persistent, retrieval-augmented memory for Zane's digital mind.

Combines four pieces:
  - `zane.memory.rolling.ConversationMemory` — the existing short-term,
    in-context rolling window (unchanged).
  - `zane.memory.store.SQLiteMessageStore` — durable storage; every
    message is written immediately on send/receive.
  - `zane.memory.embeddings.EmbeddingBackend` + `zane.memory.vector_index.
    FaissVectorIndex` — local semantic search over the full conversation
    history, so relevant older context can be retrieved even after it has
    scrolled out of the rolling window.
  - `zane.memory.summarizer.ConversationSummarizer` — periodically
    condenses aged-out history into a stored summary and prunes the raw
    rows, keeping the database lean.

This class is what `ZaneMind` talks to; none of the above pieces are used
directly by the orchestrator.
"""
from __future__ import annotations

import datetime
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import asyncio

from zane.memory.embeddings import EmbeddingBackend, EmbeddingError
from zane.memory.rolling import ConversationMemory
from zane.memory.store import MessageRow, SQLiteMessageStore
from zane.memory.summarizer import ConversationSummarizer, SummarizationError
from zane.memory.vector_index import FaissVectorIndex, VectorIndexError

logger = logging.getLogger("zane.memory.persistent")


@dataclass
class PersistentMemoryConfig:
    rolling_max_messages: int = 40
    rolling_max_chars: int = 24000
    retrieval_top_k: int = 5
    # Most recent raw messages always kept intact, never summarized/pruned.
    retention_window: int = 20
    # Once this many raw messages sit beyond the retention window, the
    # oldest chunk is summarized and pruned.
    summarize_after_n: int = 20


def _format_timestamp(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


class PersistentMemory:
    def __init__(
        self,
        session_id: str,
        store: SQLiteMessageStore,
        embeddings: EmbeddingBackend,
        vector_index: FaissVectorIndex,
        summarizer: ConversationSummarizer,
        config: Optional[PersistentMemoryConfig] = None,
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._embeddings = embeddings
        self._vector_index = vector_index
        self._summarizer = summarizer
        self.config = config or PersistentMemoryConfig()

        self.rolling = ConversationMemory(
            max_messages=self.config.rolling_max_messages,
            max_chars=self.config.rolling_max_chars,
        )
        # IDs of messages currently represented in the rolling window, used
        # to avoid surfacing the same content twice via semantic retrieval.
        self._recent_ids: Deque[int] = deque(maxlen=self.config.rolling_max_messages)

        self._bootstrap_from_store()

    def _bootstrap_from_store(self) -> None:
        """Restores the rolling window from SQLite on startup, so a process
        restart doesn't lose the recent conversation (cold start on a brand
        new session_id simply yields no rows, which is handled fine by the
        empty-list case below)."""
        try:
            rows = self._store.get_recent_messages(
                self.session_id, limit=self.config.rolling_max_messages
            )
        except Exception:
            logger.exception(
                "Failed to bootstrap rolling memory from store for session %s; "
                "starting with empty history.",
                self.session_id,
            )
            return

        for row in rows:
            if row.role == "user":
                self.rolling.add_user_message(row.content)
                self._recent_ids.append(row.id)
            elif row.role == "assistant":
                self.rolling.add_assistant_message(row.content)
                self._recent_ids.append(row.id)
            # Tool-call/tool-result rows are persisted for audit but are not
            # replayed into the rolling window on restart (they're only
            # meaningful alongside the specific tool_call_id round they
            # belonged to, which is not reconstructed here).

    # --- writes ---

    async def add_user_message(self, content: str) -> None:
        self.rolling.add_user_message(content)
        msg_id = await asyncio.to_thread(
            self._store.insert_message, self.session_id, "user", content, time.time()
        )
        self._recent_ids.append(msg_id)
        await self._embed_and_index(msg_id, content)
        await self._maybe_summarize_and_prune()

    async def add_assistant_message(self, content: str) -> None:
        self.rolling.add_assistant_message(content)
        msg_id = await asyncio.to_thread(
            self._store.insert_message, self.session_id, "assistant", content, time.time()
        )
        self._recent_ids.append(msg_id)
        await self._embed_and_index(msg_id, content)
        await self._maybe_summarize_and_prune()

    async def add_tool_exchange(self, tool_calls: List[dict], tool_results: List[Dict[str, str]]) -> None:
        self.rolling.add_tool_exchange(tool_calls, tool_results)

        ts = time.time()
        tool_call_summary = "; ".join(
            f"{tc.get('function', {}).get('name', 'unknown')}({tc.get('function', {}).get('arguments', '')})"
            for tc in tool_calls
        )
        await asyncio.to_thread(
            self._store.insert_message,
            self.session_id,
            "assistant_tool_call",
            tool_call_summary,
            ts,
        )
        for result in tool_results:
            await asyncio.to_thread(
                self._store.insert_message,
                self.session_id,
                "tool",
                result.get("content", ""),
                ts,
            )

    # --- reads ---

    def get_messages(self) -> List[dict]:
        """The current in-context rolling window, unchanged in shape from
        the original ConversationMemory-only design."""
        return self.rolling.get_messages()

    async def retrieve_relevant(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_vector: Optional["np.ndarray"] = None,
    ) -> List[str]:
        """Semantically retrieves up to `top_k` past messages relevant to
        `query`, excluding anything already present in the rolling window.
        Degrades to an empty list (never raises) on embedding or index
        failure, or on a cold-start empty history — the caller simply gets
        no "relevant past context" block that turn.

        `query_vector` lets a caller that's already embedded the query
        (e.g. `zane.knowledge_manager.KnowledgeManager`, which searches
        this same query against a second namespace concurrently) pass it
        straight through instead of paying to re-embed identical text."""
        k = top_k if top_k is not None else self.config.retrieval_top_k
        if k <= 0:
            return []

        if query_vector is None:
            try:
                query_vector = await asyncio.to_thread(self._embeddings.embed, query)
            except EmbeddingError as exc:
                logger.warning("Embedding failed for retrieval query; skipping RAG context: %s", exc)
                return []

        # Over-fetch so that filtering out rolling-window duplicates still
        # leaves up to `k` genuinely "older" results.
        overfetch = k + len(self._recent_ids)
        try:
            hits = await asyncio.to_thread(self._vector_index.search, query_vector, overfetch)
        except VectorIndexError as exc:
            logger.warning("Vector index search failed; skipping RAG context: %s", exc)
            return []

        if not hits:
            return []

        recent_ids = set(self._recent_ids)
        ranked_ids = [vec_id for vec_id, _score in hits if vec_id not in recent_ids][:k]
        if not ranked_ids:
            return []

        try:
            rows = await asyncio.to_thread(self._store.get_messages_by_ids, ranked_ids)
        except Exception:
            logger.exception("Failed to fetch retrieved messages from store; skipping RAG context.")
            return []

        rows_by_id = {row.id: row for row in rows}
        ordered_rows = [rows_by_id[i] for i in ranked_ids if i in rows_by_id]
        return [
            f"[{row.role} @ {_format_timestamp(row.timestamp)}] {row.content}"
            for row in ordered_rows
        ]

    def get_latest_summary(self) -> Optional[str]:
        try:
            summary = self._store.get_latest_summary(self.session_id)
        except Exception:
            logger.exception("Failed to fetch latest summary for session %s", self.session_id)
            return None
        return summary.summary_text if summary else None

    def reset_rolling_window(self) -> None:
        """Clears only the in-context rolling window (used by the `/reset`
        command); durable history in SQLite/the vector index is untouched
        so retrieval-augmented recall still works after a reset."""
        self.rolling.clear()
        self._recent_ids.clear()

    # --- background summarization/pruning ---

    async def _embed_and_index(self, msg_id: int, content: str) -> None:
        if not content or not content.strip():
            return
        try:
            vector = await asyncio.to_thread(self._embeddings.embed, content)
        except EmbeddingError as exc:
            logger.warning(
                "Embedding failed for message %d; message stored without embedding "
                "(will not be retrievable via semantic search): %s",
                msg_id, exc,
            )
            return

        try:
            await asyncio.to_thread(self._vector_index.add, msg_id, vector)
            await asyncio.to_thread(self._store.set_embedding_id, msg_id, msg_id)
        except VectorIndexError as exc:
            logger.warning("Vector index add failed for message %d: %s", msg_id, exc)

    async def _maybe_summarize_and_prune(self) -> None:
        try:
            prunable_count = await asyncio.to_thread(
                self._store.count_prunable_messages, self.session_id, self.config.retention_window
            )
        except Exception:
            logger.exception("Failed to check summarization threshold; skipping this cycle.")
            return

        if prunable_count < self.config.summarize_after_n:
            return

        try:
            chunk = await asyncio.to_thread(
                self._store.get_prunable_messages, self.session_id, self.config.retention_window
            )
        except Exception:
            logger.exception("Failed to fetch prunable messages; skipping this cycle.")
            return

        if not chunk:
            return

        try:
            summary_text = await self._summarizer.summarize(chunk)
        except SummarizationError as exc:
            # Do not prune on a failed summarization: the raw messages are
            # retained and this cycle is retried the next time a message is
            # added, so history is never silently destroyed.
            logger.warning(
                "Summarization failed for session %s (%d messages retained, will retry): %s",
                self.session_id, len(chunk), exc,
            )
            return

        start_ts = chunk[0].timestamp
        end_ts = chunk[-1].timestamp
        ids = [row.id for row in chunk]

        try:
            await asyncio.to_thread(
                self._store.insert_summary, self.session_id, start_ts, end_ts, summary_text
            )
            await asyncio.to_thread(self._vector_index.remove, ids)
            await asyncio.to_thread(self._store.delete_messages, ids)
        except Exception:
            logger.exception(
                "Failed to finalize summarization/pruning for session %s; raw messages "
                "may remain un-pruned and will be retried.",
                self.session_id,
            )
            return

        logger.info(
            "Summarized and pruned %d messages for session %s (%.0f -> %.0f).",
            len(ids), self.session_id, start_ts, end_ts,
        )
