"""RAG anomaly resolution: detects and resolves factual conflicts between
stored memory entries — e.g. two contradictory records of "the vault code"
or "the vehicle's last known coordinates" — so long-term semantic recall
stays consistent instead of surfacing whichever contradictory entry
happens to rank highest for a given query.

Two-stage design, and deliberately not a single embedding-similarity
threshold: cosine similarity measures topical relatedness, not
truth-value agreement, so it is used here only as a *candidate filter*
(entries about the same subject) — the actual contradiction judgment is
delegated to an LLM call, mirroring `zane.memory.summarizer`'s existing
pattern. **Nothing is ever pruned without an explicit LLM-confirmed
conflict**; an ambiguous or unparsable judgment, or an unavailable Groq
backend, always means "do nothing this cycle" — a missed defragmentation
is a far cheaper mistake than an incorrectly deleted memory.

Known limitation: each sweep re-checks every high-similarity pair in its
window, including pairs already judged "not a conflict" in a prior cycle,
until they age out of the window naturally. This trades a small amount of
redundant Groq calls for a much simpler, stateless implementation; add a
"checked, no conflict" cache if that cost becomes a real concern.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from zane.groq_client import AsyncGroqClient, GroqUnavailableError
from zane.memory.embeddings import EmbeddingBackend
from zane.memory.store import MessageRow, SQLiteMessageStore
from zane.memory.vector_index import FaissVectorIndex

if TYPE_CHECKING:
    from zane.core import SharedBackend

logger = logging.getLogger("zane.memory.memory_defragmenter")

CONSOLIDATED_ROLE = "memory_defrag_consolidated"

_CONFLICT_JUDGE_SYSTEM_PROMPT = """\
You are a precise data-consistency auditor for a memory system. You are \
given two stored entries that are semantically related. Determine \
whether they factually CONTRADICT each other — meaning they cannot both \
be true at the same time — as opposed to simply being related-but­\
compatible statements (e.g. one is a natural update/continuation of the \
other, or they describe different points in time).

Respond in exactly this format and nothing else:
CONFLICT: yes|no
RESOLUTION: <if yes, one consolidated sentence stating the resolved fact; if no, write "n/a">
"""

_RESPONSE_PATTERN = re.compile(
    r"CONFLICT:\s*(yes|no).*?RESOLUTION:\s*(.+)", re.IGNORECASE | re.DOTALL
)


@dataclass
class ConflictResolution:
    kept_message_id: int
    pruned_message_id: int
    consolidated_message_id: int
    consolidated_text: str
    reasoning: str
    validity_weight_kept: float
    validity_weight_pruned: float


class MemoryDefragmenter:
    def __init__(
        self,
        store: SQLiteMessageStore,
        embeddings: EmbeddingBackend,
        vector_index: FaissVectorIndex,
        groq_client: AsyncGroqClient,
        *,
        similarity_threshold: float = 0.90,
        half_life_s: float = 7 * 24 * 3600.0,
        sweep_limit: int = 20,
        search_k: int = 5,
        interval_s: float = 300.0,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.vector_index = vector_index
        self.groq_client = groq_client
        self.similarity_threshold = similarity_threshold
        self.half_life_s = half_life_s
        self.sweep_limit = sweep_limit
        self.search_k = search_k
        self.interval_s = interval_s
        self._running = False

    def _recency_weight(self, timestamp: float, now: Optional[float] = None) -> float:
        """Exponential half-life decay: a "validity weight" in (0, 1] that
        halves every `half_life_s` seconds of age. Used to break ties on
        which of two conflicting entries is authoritative and to attach a
        real numeric figure to that decision, not just "newer wins"."""
        now = now if now is not None else time.time()
        age_s = max(0.0, now - timestamp)
        return 2.0 ** (-age_s / self.half_life_s)

    async def _find_candidates(self, message: MessageRow) -> List[Tuple[MessageRow, float]]:
        try:
            vector = await asyncio.to_thread(self.embeddings.embed, message.content)
            hits = await asyncio.to_thread(self.vector_index.search, vector, self.search_k + 1)
        except Exception as exc:  # noqa: BLE001 - embedding/index failures degrade to "no candidates"
            logger.debug("Defragmenter candidate search failed for message %d: %s", message.id, exc)
            return []

        score_by_id = {vec_id: score for vec_id, score in hits}
        candidate_ids = [
            vec_id
            for vec_id, score in hits
            if vec_id != message.id and score >= self.similarity_threshold
        ]
        if not candidate_ids:
            return []

        rows = await asyncio.to_thread(self.store.get_messages_by_ids, candidate_ids)
        return [(row, score_by_id[row.id]) for row in rows if row.id in score_by_id]

    async def _judge_conflict(self, a: MessageRow, b: MessageRow) -> Optional[str]:
        """Returns the LLM's consolidated resolution text if it confirms a
        genuine conflict, else None (no conflict, or judgment unavailable/
        unparsable — always treated the same: do nothing)."""
        prompt = (
            f'Entry A (recorded {time.strftime("%Y-%m-%d", time.gmtime(a.timestamp))}): "{a.content}"\n'
            f'Entry B (recorded {time.strftime("%Y-%m-%d", time.gmtime(b.timestamp))}): "{b.content}"'
        )
        try:
            completion = await self.groq_client.chat_completion(
                messages=[
                    {"role": "system", "content": _CONFLICT_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                max_tokens=200,
                temperature=0.0,
            )
        except GroqUnavailableError as exc:
            logger.warning("Conflict judgment unavailable (Groq down); skipping this pair: %s", exc)
            return None

        content = completion.choices[0].message.content or ""
        match = _RESPONSE_PATTERN.search(content)
        if not match:
            logger.warning(
                "Conflict judgment response did not match the expected format; skipping "
                "rather than guessing. Raw response: %r", content,
            )
            return None

        is_conflict = match.group(1).strip().lower() == "yes"
        if not is_conflict:
            return None

        resolution_text = match.group(2).strip()
        if not resolution_text or resolution_text.lower() == "n/a":
            logger.warning(
                "LLM reported a conflict but gave no usable resolution text; skipping to "
                "avoid an ungrounded deletion."
            )
            return None
        return resolution_text

    async def check_for_conflicts(self, message: MessageRow) -> Optional[ConflictResolution]:
        """Checks one message against its semantic neighbors and, on the
        first LLM-confirmed conflict, commits a Memory Defragmentation
        Event: prunes the older entry and writes a consolidated entry
        capturing the resolved truth. Returns None if no conflict was
        found/confirmed."""
        if message.embedding_id is None:
            return None

        for candidate, _similarity in await self._find_candidates(message):
            resolution_text = await self._judge_conflict(message, candidate)
            if resolution_text is None:
                continue

            now = time.time()
            older, newer = (
                (candidate, message)
                if candidate.timestamp <= message.timestamp
                else (message, candidate)
            )
            weight_older = self._recency_weight(older.timestamp, now)
            weight_newer = self._recency_weight(newer.timestamp, now)

            consolidated_text = (
                f"[MEMORY DEFRAGMENTATION] {resolution_text} (superseded a conflicting "
                f"entry from {time.strftime('%Y-%m-%d', time.gmtime(older.timestamp))}; "
                f"validity weight kept={weight_newer:.3f} vs pruned={weight_older:.3f})"
            )

            try:
                consolidated_id = await asyncio.to_thread(
                    self.store.insert_message, newer.session_id, CONSOLIDATED_ROLE,
                    consolidated_text, now,
                )
                try:
                    consolidated_vector = await asyncio.to_thread(
                        self.embeddings.embed, consolidated_text
                    )
                    await asyncio.to_thread(
                        self.vector_index.add, consolidated_id, consolidated_vector
                    )
                    await asyncio.to_thread(
                        self.store.set_embedding_id, consolidated_id, consolidated_id
                    )
                except Exception as exc:  # noqa: BLE001 - matches PersistentMemory's best-effort embedding
                    logger.warning(
                        "Consolidated defrag entry %d stored without embedding: %s",
                        consolidated_id, exc,
                    )

                await asyncio.to_thread(self.vector_index.remove, [older.id])
                await asyncio.to_thread(self.store.delete_messages, [older.id])
            except Exception:
                logger.exception(
                    "Memory Defragmentation Event failed to commit for messages %d/%d; "
                    "original entries left untouched.", message.id, candidate.id,
                )
                return None

            logger.warning(
                "Memory Defragmentation Event: pruned message %d (weight=%.3f) in favor of "
                "consolidated entry %d (weight=%.3f). Reasoning: %s",
                older.id, weight_older, consolidated_id, weight_newer, resolution_text,
            )
            return ConflictResolution(
                kept_message_id=newer.id,
                pruned_message_id=older.id,
                consolidated_message_id=consolidated_id,
                consolidated_text=consolidated_text,
                reasoning=resolution_text,
                validity_weight_kept=weight_newer,
                validity_weight_pruned=weight_older,
            )

        return None

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            try:
                recent = await asyncio.to_thread(
                    self.store.get_recently_embedded_messages, self.sweep_limit
                )
                for message in recent:
                    await self.check_for_conflicts(message)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad sweep must not kill the daemon
                logger.exception("Memory defragmentation sweep failed; will retry next interval.")
            await asyncio.sleep(self.interval_s)

    def stop(self) -> None:
        self._running = False
