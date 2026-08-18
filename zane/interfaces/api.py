"""FastAPI deployment surface for Zane's digital mind.

Run with:  uvicorn zane.interfaces.api:app --host 0.0.0.0 --port 8000
(or `python -m zane --mode api`)

One `SharedBackend` (Groq client, C++ analytics engine, web search tool,
optional ElevenLabs TTS client) is built once at startup and shared across
every session; each `session_id` gets its own `ZaneMind` (i.e. its own
conversation memory, humor toggle, and voice toggle) lazily on first use.

Voice output design choice: when a session's voice toggle is on and
synthesis succeeds, `POST /chat` includes the audio as a base64 string
(`audio_base64` + `audio_format`) in the *same* JSON response as the text,
rather than switching the endpoint's response type or exposing a separate
streaming/binary route. One request/response pair per turn keeps the text
and its audio atomically tied together with no risk of a client fetching
audio for the wrong turn, and keeps a single, stable OpenAPI schema for
`/chat` regardless of whether voice is on. The tradeoff is the ~33%
size inflation base64 adds to the payload; for short spoken replies this
is an acceptable cost for the simplicity. A dedicated binary
`/chat/audio/{turn_id}` endpoint (returning `Response(media_type="audio/mpeg")`
directly, for e.g. an HTML `<audio>` tag `src`) would be the natural next
step if payload size or streaming playback becomes a real requirement, but
isn't needed yet and isn't implemented here.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from zane.config import settings
from zane.core import SharedBackend, ZaneMind
from zane.groq_client import GroqUnavailableError
from zane.interfaces.base import ZaneInterface
from zane.voice.speaker_verification import (
    SpeakerVerificationAudioError,
    SpeakerVerificationUnavailableError,
    get_verifier,
)

logger = logging.getLogger("zane.interfaces.api")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message to Zane.")
    session_id: str = Field(..., min_length=1, description="Stable ID for this conversation.")
    display_name: Optional[str] = Field(None, description="Name Zane should address the user by.")
    mission_context: Optional[str] = Field(None, description="Optional situational context.")


class ChatResponse(BaseModel):
    reply: str
    tool_calls_made: list[str]
    elapsed_ms: float
    humor_enabled: bool
    voice_enabled: bool
    # Populated only when the session's voice toggle is on and synthesis
    # succeeded; both are omitted/None otherwise (voice off, no TTS backend
    # configured, or synthesis failed) — see the module docstring for why
    # audio rides along in this same response instead of a separate route.
    audio_base64: Optional[str] = Field(
        None, description="Base64-encoded synthesized speech for `reply`, if voice is enabled."
    )
    audio_format: Optional[str] = Field(
        None, description="ElevenLabs output_format string for audio_base64, e.g. 'mp3_44100_128'."
    )


class CommandRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    command: str = Field(..., min_length=1, description="e.g. 'humor', 'reset', 'help'.")


class CommandResponse(BaseModel):
    result: str


class VoiceEnrollResponse(BaseModel):
    speaker_id: str
    samples: int = Field(..., description="Total samples now stored for this speaker.")


class VoiceVerifyResponse(BaseModel):
    verified: bool
    speaker_id: Optional[str] = Field(None, description="Best-matching enrolled speaker, if verified.")
    similarity: float = Field(..., description="Cosine similarity of the best match, in [-1, 1].")
    threshold: float


class SpeakerInfo(BaseModel):
    speaker_id: str
    samples: int


class SpeakerListResponse(BaseModel):
    speakers: List[SpeakerInfo]


class APIInterface(ZaneInterface):
    """Adapts ZaneInterface's shared command handling to a specific
    per-session ZaneMind, for use from FastAPI route handlers."""

    async def start(self) -> None:  # pragma: no cover - lifecycle managed by ASGI app
        return None

    async def stop(self) -> None:
        await self.mind.aclose()


class SessionManager:
    def __init__(self) -> None:
        self._backend: Optional[SharedBackend] = None
        self._sessions: Dict[str, ZaneMind] = {}

    async def startup(self) -> None:
        self._backend = SharedBackend.build(settings)

    async def shutdown(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()
        self._sessions.clear()

    def get_or_create(self, session_id: str) -> ZaneMind:
        if self._backend is None:
            raise RuntimeError("SessionManager used before startup().")
        if session_id not in self._sessions:
            self._sessions[session_id] = ZaneMind(shared=self._backend, session_id=session_id)
        return self._sessions[session_id]

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


sessions = SessionManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await sessions.startup()
    logger.info("Zane's digital mind is online (API interface).")
    yield
    await sessions.shutdown()
    logger.info("Zane's digital mind has powered down (API interface).")


app = FastAPI(
    title="Zane Digital Mind API",
    description="REST endpoints for Zane's hybrid C++/Python cognitive architecture.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_request_latency(request: Request, call_next):
    """Feeds Falcon Scout's `LatencyRecorder` (see zane/falcon_worker.py)
    real per-request timings; a no-op if the backend hasn't finished
    starting up yet or Falcon Scout is disabled (recorder always exists,
    just goes unread in that case)."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    if sessions._backend is not None:
        sessions._backend.latency_recorder.record(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
        )
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    mind = sessions.get_or_create(req.session_id)
    try:
        result = await mind.respond(
            req.message,
            addressed_by=req.display_name,
            mission_context=req.mission_context,
        )
    except GroqUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Zane's cognitive backend is unavailable: {exc}")

    audio_base64 = base64.b64encode(result.audio).decode("ascii") if result.audio else None

    return ChatResponse(
        reply=result.text,
        tool_calls_made=result.tool_calls_made,
        elapsed_ms=result.elapsed_ms,
        humor_enabled=mind.humor.enabled,
        voice_enabled=mind.voice.enabled,
        audio_base64=audio_base64,
        audio_format=result.audio_format if audio_base64 else None,
    )


@app.post("/command", response_model=CommandResponse)
async def command(req: CommandRequest) -> CommandResponse:
    mind = sessions.get_or_create(req.session_id)
    interface = APIInterface(mind)
    reply = await interface.handle_command(req.command)
    if reply is None:
        raise HTTPException(status_code=400, detail=f"Unrecognized command: {req.command!r}")
    return CommandResponse(result=reply)


@app.delete("/session/{session_id}")
async def end_session(session_id: str) -> Dict[str, str]:
    sessions.drop(session_id)
    return {"status": "session cleared", "session_id": session_id}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# --- Speaker verification (voice biometrics) ---
#
# Deliberately separate from /chat: this is an identity check on a raw
# audio clip, not part of the conversational turn shape. See
# zane/voice/speaker_verification.py for the enrollment/matching logic and
# requirements-voice-verify.txt for the (optional) dependency.


@app.post("/voice/enroll", response_model=VoiceEnrollResponse)
async def voice_enroll(
    speaker_id: str = Form(..., min_length=1),
    audio: UploadFile = File(...),
) -> VoiceEnrollResponse:
    verifier = get_verifier()
    audio_bytes = await audio.read()
    try:
        result = await asyncio.to_thread(verifier.enroll, speaker_id, audio_bytes)
    except SpeakerVerificationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except SpeakerVerificationAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return VoiceEnrollResponse(**result)


@app.post("/voice/verify", response_model=VoiceVerifyResponse)
async def voice_verify(
    audio: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
) -> VoiceVerifyResponse:
    verifier = get_verifier()
    audio_bytes = await audio.read()
    try:
        result = await asyncio.to_thread(verifier.verify, audio_bytes, speaker_id)
    except SpeakerVerificationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except SpeakerVerificationAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return VoiceVerifyResponse(**{k: v for k, v in result.items() if k != "error"})


@app.get("/voice/speakers", response_model=SpeakerListResponse)
async def voice_list_speakers() -> SpeakerListResponse:
    return SpeakerListResponse(speakers=get_verifier().list_speakers())


@app.delete("/voice/speakers/{speaker_id}")
async def voice_delete_speaker(speaker_id: str) -> Dict[str, object]:
    existed = await asyncio.to_thread(get_verifier().delete_speaker, speaker_id)
    if not existed:
        raise HTTPException(status_code=404, detail=f"No enrolled profile for {speaker_id!r}")
    return {"deleted": True, "speaker_id": speaker_id}
