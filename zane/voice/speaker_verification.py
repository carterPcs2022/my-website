"""zane/voice/speaker_verification.py — voice-biometric speaker verification.

Optional feature, same degrade-gracefully shape as `zane/voice/tts.py`:
without the `resemblyzer` package (and its `torch` dependency) installed,
`SpeakerVerifier.available` is False and every real operation raises
`SpeakerVerificationUnavailableError`, which the API layer turns into a
503 rather than crashing the process.

Enrolled voiceprints are 256-dim embeddings from Resemblyzer's pretrained
speaker encoder, persisted as a flat JSON file — a handful of speakers
times a handful of samples each is small enough that a JSON file is
simpler than standing up a SQLite schema for it (unlike
`zane/memory/store.py`, which needs real query patterns over a growing
message log).

Audio format: send WAV or FLAC. Those decode via `soundfile` (bundles its
own `libsndfile`, no system package needed). Compressed formats like MP3
fall back to `audioread`, which needs `ffmpeg` on the host — not installed
in this project's Docker image, so uploads in those formats will fail to
decode unless you add it.
"""
from __future__ import annotations

import io
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("zane.voice.speaker_verification")

try:
    from resemblyzer import VoiceEncoder, preprocess_wav

    _RESEMBLYZER_AVAILABLE = True
except ImportError:  # pragma: no cover - resemblyzer/torch are optional
    VoiceEncoder = None  # type: ignore
    preprocess_wav = None  # type: ignore
    _RESEMBLYZER_AVAILABLE = False

# Resemblyzer's encoder expects >=~1s of speech at 16kHz to produce a
# stable embedding; shorter clips are rejected rather than silently
# producing a low-quality voiceprint that would just cause bad matches.
_MIN_AUDIO_SAMPLES = 16000


class SpeakerVerificationUnavailableError(RuntimeError):
    """resemblyzer/torch isn't installed — this is an optional deployment
    feature, not a bug; callers should degrade (e.g. HTTP 503), not crash."""


class SpeakerVerificationAudioError(RuntimeError):
    """The uploaded audio couldn't be decoded, or was too short to embed
    reliably."""


@dataclass
class SpeakerProfile:
    speaker_id: str
    embeddings: List[List[float]] = field(default_factory=list)

    def centroid(self) -> np.ndarray:
        """Average of all enrolled samples' embeddings, re-normalized to
        unit length so its dot product with another unit vector is a
        cosine similarity."""
        vectors = np.array(self.embeddings, dtype=np.float32)
        mean = vectors.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean


class SpeakerVerifier:
    """Enrolls and verifies speakers from short audio clips using
    Resemblyzer's pretrained speaker-encoder model.

    The encoder itself is loaded lazily on first real use (not at
    construction) so importing this module, or running the API/CLI
    without ever touching a /voice/* route, costs nothing extra and never
    fails just because torch isn't installed.
    """

    def __init__(self, profiles_path: str, threshold: float, max_samples_per_speaker: int = 10):
        self._profiles_path = Path(profiles_path)
        self.threshold = threshold
        self.max_samples_per_speaker = max_samples_per_speaker
        self._encoder = None
        # Guards both lazy encoder construction and profile file
        # read-modify-write — enroll()/delete_speaker() calls arriving
        # concurrently must not race on the same JSON file.
        self._lock = threading.Lock()
        self._profiles: Dict[str, SpeakerProfile] = self._load_profiles()

    @property
    def available(self) -> bool:
        return _RESEMBLYZER_AVAILABLE

    def _require_available(self) -> None:
        if not _RESEMBLYZER_AVAILABLE:
            raise SpeakerVerificationUnavailableError(
                "Speaker verification requires the optional 'resemblyzer' package "
                "(pulls in torch) — install requirements-voice-verify.txt."
            )

    def _get_encoder(self) -> "VoiceEncoder":
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    logger.info("Loading Resemblyzer speaker-encoder model (first use)...")
                    self._encoder = VoiceEncoder("cpu")
        return self._encoder

    def _load_profiles(self) -> Dict[str, SpeakerProfile]:
        if not self._profiles_path.exists():
            return {}
        try:
            raw = json.loads(self._profiles_path.read_text())
            return {
                sid: SpeakerProfile(speaker_id=sid, embeddings=data.get("embeddings", []))
                for sid, data in raw.items()
            }
        except Exception as exc:
            logger.warning("Failed to load speaker profiles, starting empty: %s", exc)
            return {}

    def _save_profiles(self) -> None:
        self._profiles_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {sid: {"embeddings": p.embeddings} for sid, p in self._profiles.items()}
        self._profiles_path.write_text(json.dumps(payload, indent=2))

    def embed_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decodes an uploaded audio clip into a 256-dim speaker
        embedding. Blocking/CPU-bound (model inference) — callers on an
        event loop should run this via `asyncio.to_thread`."""
        self._require_available()
        try:
            wav = preprocess_wav(io.BytesIO(audio_bytes))
        except Exception as exc:
            raise SpeakerVerificationAudioError(f"Could not decode audio: {exc}") from exc

        if wav.size < _MIN_AUDIO_SAMPLES:
            raise SpeakerVerificationAudioError(
                f"Audio clip too short to verify reliably "
                f"(need >= {_MIN_AUDIO_SAMPLES / 16000:.0f}s of speech)."
            )

        return self._get_encoder().embed_utterance(wav)

    def enroll(self, speaker_id: str, audio_bytes: bytes) -> dict:
        """Adds one more sample to `speaker_id`'s voiceprint. Multiple
        samples (capped at `max_samples_per_speaker`, oldest dropped
        first) are kept and averaged at verify time for a more robust
        reference than a single utterance."""
        embedding = self.embed_audio(audio_bytes)
        with self._lock:
            profile = self._profiles.setdefault(speaker_id, SpeakerProfile(speaker_id=speaker_id))
            profile.embeddings.append(embedding.tolist())
            profile.embeddings = profile.embeddings[-self.max_samples_per_speaker :]
            self._save_profiles()
            sample_count = len(profile.embeddings)
        return {"speaker_id": speaker_id, "samples": sample_count}

    def verify(self, audio_bytes: bytes, speaker_id: Optional[str] = None) -> dict:
        """Compares the given clip against one enrolled speaker (if
        `speaker_id` is given) or every enrolled speaker (best match
        wins), and reports whether the best score clears `threshold`."""
        embedding = self.embed_audio(audio_bytes)
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm if norm > 0 else embedding

        if speaker_id is not None and speaker_id not in self._profiles:
            return {
                "verified": False, "speaker_id": None, "similarity": 0.0,
                "threshold": self.threshold, "error": f"No enrolled profile for {speaker_id!r}",
            }

        candidates = {speaker_id: self._profiles[speaker_id]} if speaker_id else self._profiles
        if not candidates:
            return {
                "verified": False, "speaker_id": None, "similarity": 0.0,
                "threshold": self.threshold, "error": "No enrolled speakers.",
            }

        best_id, best_score = None, -1.0
        for sid, profile in candidates.items():
            if not profile.embeddings:
                continue
            score = float(np.dot(embedding, profile.centroid()))
            if score > best_score:
                best_id, best_score = sid, score

        verified = best_id is not None and best_score >= self.threshold
        return {
            "verified": verified,
            "speaker_id": best_id if verified else None,
            "similarity": round(best_score, 4) if best_id is not None else 0.0,
            "threshold": self.threshold,
        }

    def list_speakers(self) -> list:
        return [{"speaker_id": sid, "samples": len(p.embeddings)} for sid, p in self._profiles.items()]

    def delete_speaker(self, speaker_id: str) -> bool:
        with self._lock:
            existed = self._profiles.pop(speaker_id, None) is not None
            if existed:
                self._save_profiles()
        return existed


_verifier_lock = threading.Lock()
_verifier_singleton: Optional[SpeakerVerifier] = None


def get_verifier() -> SpeakerVerifier:
    """Process-wide singleton, built from `zane.config.settings` on first
    call. Speaker profiles are server-wide (not per-session, unlike
    `ZaneMind`), so unlike `SharedBackend` this doesn't need an explicit
    startup/shutdown lifecycle — there's no network client or thread pool
    to release."""
    global _verifier_singleton
    if _verifier_singleton is None:
        with _verifier_lock:
            if _verifier_singleton is None:
                from zane.config import settings

                _verifier_singleton = SpeakerVerifier(
                    profiles_path=settings.speaker_profiles_path,
                    threshold=settings.speaker_verify_threshold,
                    max_samples_per_speaker=settings.speaker_verify_max_samples,
                )
    return _verifier_singleton
