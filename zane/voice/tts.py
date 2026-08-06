"""Async ElevenLabs text-to-speech client.

Follows the same exponential-backoff-with-jitter retry pattern as
AsyncGroqClient (see zane/groq_client.py). Voice synthesis is an optional
post-processing step on top of Zane's text response — every failure mode
here degrades to "no audio" rather than raising into the conversation
turn; see `synthesize_with_fallback` below, which is what `ZaneMind.respond`
actually calls.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger("zane.voice.tts")

try:
    from elevenlabs.client import AsyncElevenLabs
except ImportError:  # pragma: no cover - elevenlabs is an optional dependency
    AsyncElevenLabs = None  # type: ignore

try:
    import httpx

    _NETWORK_EXCEPTIONS: Tuple[type, ...] = (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.HTTPError,
    )
except ImportError:  # pragma: no cover
    _NETWORK_EXCEPTIONS = ()


class TextToSpeechError(RuntimeError):
    """Raised when ElevenLabs synthesis fails after all retries are
    exhausted. Callers must treat this as a soft failure — voice
    synthesis must never break a conversational turn."""


class AsyncElevenLabsClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "",
        *,
        client: Optional[Any] = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
        max_retries: int = 4,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 15.0,
    ) -> None:
        if not voice_id:
            raise ValueError(
                "A voice_id is required to construct AsyncElevenLabsClient "
                "(set ZANE_VOICE_ID; do not hardcode it)."
            )

        if client is not None:
            # Dependency injection seam for tests — a fake exposing the
            # same `.text_to_speech.convert(...)` async-iterator interface
            # can be passed here without needing the real SDK or network.
            self._client = client
        else:
            if not api_key:
                raise ValueError(
                    "An ElevenLabs API key is required to construct AsyncElevenLabsClient."
                )
            if AsyncElevenLabs is None:
                raise TextToSpeechError(
                    "The `elevenlabs` package is not installed. Run `pip install "
                    "elevenlabs` to enable voice synthesis."
                )
            self._client = AsyncElevenLabs(api_key=api_key)

        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, _NETWORK_EXCEPTIONS):
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            return False
        return status_code == 429 or 500 <= status_code < 600

    def _compute_backoff(self, attempt: int, exc: Exception) -> float:
        headers = getattr(exc, "headers", None)
        if headers and hasattr(headers, "get"):
            retry_after = headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except (TypeError, ValueError):
                    pass

        backoff = min(self.max_backoff_s, self.base_backoff_s * (2 ** attempt))
        jitter = random.uniform(0, backoff * 0.25)
        return backoff + jitter

    async def synthesize(self, text: str, *, voice_id: Optional[str] = None) -> bytes:
        """Synthesizes `text` to audio bytes (encoded per `output_format`,
        MP3 by default). Raises TextToSpeechError if synthesis fails after
        all retries — callers must catch this and fall back to text-only."""
        if not text or not text.strip():
            raise TextToSpeechError("Cannot synthesize empty text.")

        target_voice = voice_id or self.voice_id
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                start = time.monotonic()
                audio_stream = self._client.text_to_speech.convert(
                    voice_id=target_voice,
                    text=text,
                    model_id=self.model_id,
                    output_format=self.output_format,
                )
                chunks = bytearray()
                async for chunk in audio_stream:
                    if chunk:
                        chunks.extend(chunk)

                if not chunks:
                    raise TextToSpeechError("ElevenLabs returned an empty audio stream.")

                elapsed_ms = (time.monotonic() - start) * 1000
                logger.debug(
                    "ElevenLabs synthesis returned %d bytes in %.1fms", len(chunks), elapsed_ms
                )
                return bytes(chunks)

            except TextToSpeechError:
                raise
            except Exception as exc:  # noqa: BLE001 - SDK exception taxonomy isn't fully known
                last_exc = exc
                if not self._is_retryable(exc) or attempt >= self.max_retries:
                    raise TextToSpeechError(f"ElevenLabs synthesis failed: {exc}") from exc
                delay = self._compute_backoff(attempt, exc)
                logger.warning(
                    "ElevenLabs synthesis failed (%s: %s), retrying in %.2fs [attempt %d/%d]",
                    type(exc).__name__, exc, delay, attempt + 1, self.max_retries,
                )
                await asyncio.sleep(delay)

        raise TextToSpeechError(f"ElevenLabs synthesis failed after retries: {last_exc}")

    async def close(self) -> None:
        closer = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if closer is None:
            return
        result = closer()
        if asyncio.iscoroutine(result):
            await result


async def synthesize_with_fallback(
    client: Optional[AsyncElevenLabsClient],
    enabled: bool,
    text: str,
    *,
    voice_id: Optional[str] = None,
) -> Tuple[Optional[bytes], Optional[str]]:
    """The single "optional post-processing step" every text response
    passes through. Never raises: a disabled toggle, a missing/unconfigured
    backend, or a synthesis failure all fall back to (None, None) — i.e.
    text-only output, exactly as if voice were off.

    `voice_id` is an optional per-call override of the client's own
    default voice — this is the whole dual-voice mechanism for P.I.X.A.L.
    (see zane/companion_bridge.py and zane/core.py): her notices reuse
    this exact same AsyncElevenLabsClient (and its retry/backoff/auth
    plumbing) rather than a second client instance, routed to
    `PIXAL_VOICE_ID` instead of Zane's `ZANE_VOICE_ID` purely via this
    parameter.
    """
    if not enabled or client is None or not text or not text.strip():
        return None, None
    try:
        audio = await client.synthesize(text, voice_id=voice_id)
        return audio, client.output_format
    except TextToSpeechError as exc:
        logger.warning("Voice synthesis failed, falling back to text-only: %s", exc)
        return None, None
