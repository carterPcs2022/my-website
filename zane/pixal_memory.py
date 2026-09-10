"""Bounded, deterministic memory for P.I.X.A.L.

This module stores small amounts of companion context without requiring a
network service or database. It is intentionally bounded and serializable so
it can later be backed by durable storage without changing the interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
import uuid
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class PixalMemoryEntry:
    """One bounded memory item."""

    content: str
    role: str = "system"
    importance: float = 0.5
    tags: Tuple[str, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        self.content = self.content.strip()
        self.importance = max(0.0, min(1.0, float(self.importance)))
        self.tags = tuple(sorted({str(tag).strip().lower() for tag in self.tags if str(tag).strip()}))


class PixalMemoryStore:
    """Small in-process memory with predictable capacity and relevance."""

    def __init__(self, *, max_entries: int = 100, max_content_length: int = 1000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if max_content_length < 1:
            raise ValueError("max_content_length must be at least 1")
        self.max_entries = max_entries
        self.max_content_length = max_content_length
        self._entries: List[PixalMemoryEntry] = []

    def remember(
        self,
        content: str,
        *,
        role: str = "system",
        importance: float = 0.5,
        tags: Iterable[str] = (),
    ) -> PixalMemoryEntry:
        """Add a memory, evicting the least useful old item if full."""
        text = str(content).strip()[: self.max_content_length]
        if not text:
            raise ValueError("memory content cannot be empty")
        entry = PixalMemoryEntry(text, role=role, importance=importance, tags=tuple(tags))
        self._entries.append(entry)
        self._evict_if_needed()
        return entry

    def recent(self, limit: int = 10) -> List[PixalMemoryEntry]:
        """Return the newest memories first."""
        return list(reversed(self._entries[-max(0, limit) :]))

    def relevant(self, query: str, *, limit: int = 5) -> List[PixalMemoryEntry]:
        """Rank memories by simple keyword overlap plus importance and recency."""
        terms = {word.lower() for word in query.split() if word.strip()}
        if not terms:
            return self.recent(limit)

        now = time.time()
        scored: List[Tuple[float, PixalMemoryEntry]] = []
        for entry in self._entries:
            haystack = set(entry.content.lower().split()) | set(entry.tags)
            overlap = len(terms & haystack)
            if overlap == 0:
                continue
            age_hours = max(0.0, (now - entry.created_at) / 3600.0)
            recency = 1.0 / (1.0 + age_hours / 24.0)
            score = overlap + entry.importance * 0.5 + recency * 0.25
            scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [entry for _, entry in scored[: max(0, limit)]]
        for entry in results:
            entry.last_accessed_at = now
        return results

    def forget(self, memory_id: str) -> bool:
        """Remove one memory by id."""
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if entry.memory_id != memory_id]
        return len(self._entries) != before

    def snapshot(self) -> List[Dict[str, object]]:
        """Return JSON-friendly memory data."""
        return [asdict(entry) for entry in self._entries]

    def clear(self) -> None:
        self._entries.clear()

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            victim = min(
                self._entries,
                key=lambda entry: (entry.importance, entry.last_accessed_at, entry.created_at),
            )
            self._entries.remove(victim)
