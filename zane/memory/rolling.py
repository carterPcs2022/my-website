"""Conversational memory for Zane's digital mind.

A rolling, token/char-budget-aware buffer of chat messages in the OpenAI/Groq
`{"role": ..., "content": ...}` message format. Deliberately simple and
dependency-free (no external vector store) — this is short-term working
memory for a single conversation session, not long-term recall.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Optional


class ConversationMemory:
    """Thread-safe rolling conversation buffer.

    Trims from the oldest end once either the message-count or total
    character budget is exceeded, always preserving the most recent turns
    intact (never truncates the middle of a message).
    """

    def __init__(self, max_messages: int = 40, max_chars: int = 24000) -> None:
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._messages: Deque[Dict[str, str]] = deque()

    def add_user_message(self, content: str) -> None:
        self._append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._append({"role": "assistant", "content": content})

    def add_tool_exchange(self, tool_calls: List[dict], tool_results: List[Dict[str, str]]) -> None:
        """Records an assistant tool-call turn plus the resulting tool
        outputs, exactly as Groq's chat-completions API expects them when
        replayed back in `messages` on the follow-up request."""
        with self._lock:
            self._messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            for result in tool_results:
                self._messages.append(result)
            self._trim_locked()

    def _append(self, message: Dict[str, str]) -> None:
        with self._lock:
            self._messages.append(message)
            self._trim_locked()

    def _trim_locked(self) -> None:
        while len(self._messages) > self.max_messages:
            self._messages.popleft()

        total_chars = sum(len(str(m.get("content") or "")) for m in self._messages)
        while total_chars > self.max_chars and len(self._messages) > 1:
            removed = self._messages.popleft()
            total_chars -= len(str(removed.get("content") or ""))

    def get_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)
