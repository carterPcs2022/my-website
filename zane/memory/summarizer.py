"""LLM-driven summarization for chunks of Zane's older conversation history.

Used by `zane/memory/persistent.py` to condense messages that have aged
out of the retention window before they're pruned from SQLite. Summarizing
is explicitly allowed to fail (Groq drop, rate limit, empty completion);
callers must not prune the underlying raw messages unless summarization
actually succeeded, or that history is lost forever.
"""
from __future__ import annotations

import logging
from typing import Sequence

from zane.groq_client import AsyncGroqClient, GroqUnavailableError
from zane.memory.store import MessageRow

logger = logging.getLogger("zane.memory.summarizer")

_SUMMARIZER_SYSTEM_PROMPT = (
    "You are a summarization assistant. Condense the following conversation "
    "excerpt into a concise, third-person summary (3-6 sentences) capturing "
    "key facts, decisions, and unresolved threads. Do not adopt any persona "
    "or add commentary; respond with plain factual prose only."
)


class SummarizationError(RuntimeError):
    """Raised when a conversation chunk could not be summarized. Callers
    must retain the raw messages and retry later rather than pruning."""


class ConversationSummarizer:
    def __init__(self, groq_client: AsyncGroqClient) -> None:
        self._groq = groq_client

    async def summarize(self, messages: Sequence[MessageRow]) -> str:
        if not messages:
            raise SummarizationError("Cannot summarize an empty message chunk.")

        transcript = "\n".join(
            f"{m.role}: {m.content}" for m in messages if m.content and m.content.strip()
        )
        if not transcript.strip():
            raise SummarizationError("Message chunk contained no summarizable content.")

        request_messages = [
            {"role": "system", "content": _SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]

        try:
            completion = await self._groq.chat_completion(
                request_messages, tools=None, max_tokens=300, temperature=0.2
            )
        except GroqUnavailableError as exc:
            raise SummarizationError(f"Summarization LLM call failed: {exc}") from exc

        content = completion.choices[0].message.content
        if not content or not content.strip():
            raise SummarizationError("Summarization LLM returned an empty response.")

        return content.strip()
