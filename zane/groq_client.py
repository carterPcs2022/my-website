"""Asynchronous Groq API client, tuned for low latency and resilient to
rate limits / transient API drops.

Wraps `groq.AsyncGroq` rather than replacing it: callers get the full
completion object back so tool-calls, usage stats, etc. remain accessible,
but every call is wrapped in an exponential-backoff-with-jitter retry loop
that specifically handles Groq's documented failure modes.

Model routing is intentionally deterministic and local: simple/short turns
use the fast model, while longer or explicitly complex turns use the deep
model. The model IDs are configured through GROQ_FAST_MODEL and
GROQ_DEEP_MODEL so Render/local deployments can change them without code
changes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger("zane.groq_client")

_RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

_DEPRECATED_GROQ_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
}

# These markers intentionally favor the fast path. Zane should feel
# immediate for ordinary conversation; only clearly demanding work should
# pay the latency cost of the larger model.
_DEEP_MARKERS = (
    "analyze",
    "analysis",
    "architect",
    "architecture",
    "debug",
    "debugging",
    "design",
    "diagnose",
    "evaluate",
    "explain in detail",
    "compare",
    "reason through",
    "step by step",
    "strategy",
    "research",
    "code",
    "coding",
    "program",
    "programming",
    "implement",
    "refactor",
    "algorithm",
    "complex",
)


class GroqUnavailableError(RuntimeError):
    """Raised when Groq's API is unreachable after all retries are exhausted."""


class AsyncGroqClient:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.4,
        max_tokens: int = 1024,
        timeout_s: float = 30.0,
        max_retries: int = 5,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 20.0,
    ) -> None:
        if not api_key:
            raise ValueError("A Groq API key is required to construct AsyncGroqClient.")

        self._client = AsyncGroq(api_key=api_key, timeout=timeout_s, max_retries=0)
        self.fast_model = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
        self.deep_model = os.getenv("GROQ_DEEP_MODEL", "openai/gpt-oss-120b")

        # Preserve compatibility with the existing Settings.groq_model field,
        # while preventing the retired Groq defaults from ever being selected.
        if model in _DEPRECATED_GROQ_MODELS:
            model = self.fast_model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s

    def _compute_backoff(self, attempt: int, exc: Exception) -> float:
        # Honor Groq's Retry-After header on 429s when present.
        retry_after = getattr(exc, "response", None)
        if retry_after is not None:
            header_value = retry_after.headers.get("retry-after") if hasattr(
                retry_after, "headers"
            ) else None
            if header_value:
                try:
                    return float(header_value)
                except ValueError:
                    pass

        backoff = min(self.max_backoff_s, self.base_backoff_s * (2 ** attempt))
        jitter = random.uniform(0, backoff * 0.25)
        return backoff + jitter

    def _select_model(self, messages: List[Dict[str, Any]]) -> tuple[str, str]:
        """Choose (model, reasoning_effort) without another LLM call.

        Fast is the default. Deep is reserved for clearly demanding turns,
        keeping ordinary Zane conversation on the ~1000 t/s model.
        """
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    user_text = content
                break

        normalized = user_text.strip().lower()
        is_long = len(normalized) >= 900
        is_complex = any(marker in normalized for marker in _DEEP_MARKERS)

        if is_long or is_complex:
            return self.deep_model, "medium"
        return self.fast_model, "low"

    async def _call_with_retries(self, coro_factory):
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self._compute_backoff(attempt, exc)
                logger.warning(
                    "Groq API call failed (%s: %s), retrying in %.2fs "
                    "[attempt %d/%d]",
                    type(exc).__name__,
                    exc,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                await asyncio.sleep(delay)
            except APIStatusError as exc:
                # Non-retryable status (e.g. 400 bad request, 401 auth) —
                # fail fast, these will not resolve by retrying.
                logger.error("Groq API returned a non-retryable error: %s", exc)
                raise

        raise GroqUnavailableError(
            f"Groq API unavailable after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        stream: bool = False,
        **overrides: Any,
    ) -> Any:
        """Runs a single (non-streaming) chat completion with full retry
        handling. Returns the raw Groq ChatCompletion object."""

        selected_model, selected_reasoning = self._select_model(messages)
        params: Dict[str, Any] = dict(
            model=overrides.pop("model", selected_model),
            messages=messages,
            temperature=overrides.pop("temperature", self.temperature),
            max_tokens=overrides.pop("max_tokens", self.max_tokens),
            stream=False,
        )
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        params.setdefault("reasoning_effort", overrides.pop("reasoning_effort", selected_reasoning))
        params.update(overrides)

        async def _factory():
            start = time.monotonic()
            result = await self._client.chat.completions.create(**params)
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.debug(
                "Groq completion returned in %.1fms using %s",
                elapsed_ms,
                params["model"],
            )
            return result

        return await self._call_with_retries(_factory)

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        **overrides: Any,
    ) -> AsyncIterator[str]:
        """Streams completion text chunks for low-latency, incremental
        rendering (e.g. token-by-token printing in the CLI). Retries only
        apply to establishing the stream — once tokens start flowing, a
        mid-stream drop propagates to the caller rather than silently
        restarting (which could duplicate partial output)."""

        selected_model, selected_reasoning = self._select_model(messages)
        params: Dict[str, Any] = dict(
            model=overrides.pop("model", selected_model),
            messages=messages,
            temperature=overrides.pop("temperature", self.temperature),
            max_tokens=overrides.pop("max_tokens", self.max_tokens),
            stream=True,
        )
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        params.setdefault("reasoning_effort", overrides.pop("reasoning_effort", selected_reasoning))
        params.update(overrides)

        async def _factory():
            return await self._client.chat.completions.create(**params)

        stream = await self._call_with_retries(_factory)
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except _RETRYABLE_EXCEPTIONS as exc:
            raise GroqUnavailableError(
                f"Groq stream dropped mid-response: {exc}"
            ) from exc

    async def close(self) -> None:
        await self._client.close()
