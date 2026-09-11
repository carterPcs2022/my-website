"""FastAPI deployment surface for Zane's digital mind."""
from __future__ import annotations

import base64
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from zane.config import settings
from zane.core import SharedBackend, ZaneMind
from zane.email.oauth import GmailOAuthError, GmailOAuthManager
from zane.groq_client import GroqUnavailableError
from zane.interfaces.base import ZaneInterface

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
    audio_base64: Optional[str] = Field(None, description="Base64-encoded synthesized speech for `reply`, if voice is enabled.")
    audio_format: Optional[str] = Field(None, description="ElevenLabs output_format string for audio_base64, e.g. 'mp3_44100_128'.")


class CommandRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    command: str = Field(..., min_length=1, description="e.g. 'humor', 'reset', 'help'.")


class CommandResponse(BaseModel):
    result: str


class APIInterface(ZaneInterface):
    """Adapts ZaneInterface's shared command handling to a specific
    per-session ZaneMind, for use from FastAPI route handlers."""

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        await self.mind.aclose()


class SessionManager:
    def __init__(self) -> None:
        self._backend: Optional[SharedBackend] = None
        self._sessions: Dict[str, ZaneMind] = {}
        self._startup_error: Optional[str] = None

    async def startup(self) -> None:
        try:
            self._backend = SharedBackend.build(settings)
            self._startup_error = None
        except Exception as exc:
            self._backend = None
            self._startup_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Zane backend failed during startup")

    async def shutdown(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()
        self._sessions.clear()
        self._backend = None

    def get_or_create(self, session_id: str) -> ZaneMind:
        if self._backend is None:
            detail = self._startup_error or "Zane backend is not initialized."
            raise RuntimeError(detail)
        if session_id not in self._sessions:
            self._sessions[session_id] = ZaneMind(shared=self._backend, session_id=session_id)
        return self._sessions[session_id]

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @property
    def ready(self) -> bool:
        return self._backend is not None

    @property
    def startup_error(self) -> Optional[str]:
        return self._startup_error


sessions = SessionManager()
email_oauth = GmailOAuthManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await sessions.startup()
    logger.info("Zane's digital mind API process is online (backend_ready=%s).", sessions.ready)
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
    try:
        mind = sessions.get_or_create(req.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Zane backend is not ready: {exc}")
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
    try:
        mind = sessions.get_or_create(req.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Zane backend is not ready: {exc}")
    interface = APIInterface(mind)
    reply = await interface.handle_command(req.command)
    if reply is None:
        raise HTTPException(status_code=400, detail=f"Unrecognized command: {req.command!r}")
    return CommandResponse(result=reply)


@app.get("/email/auth/start")
async def email_auth_start() -> RedirectResponse:
    """Start the one-time Google OAuth flow for Zane's Gmail account."""
    state = secrets.token_urlsafe(32)
    try:
        authorization_url = email_oauth.authorization_url(state)
    except Exception as exc:
        logger.exception("Unable to start Gmail OAuth")
        raise HTTPException(status_code=500, detail="Gmail OAuth is not configured.") from exc

    response = RedirectResponse(authorization_url, status_code=302)
    response.set_cookie(
        "zane_gmail_oauth_state",
        state,
        max_age=600,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


@app.get("/email/auth/callback", response_class=HTMLResponse)
async def email_auth_callback(request: Request, code: str, state: str) -> HTMLResponse:
    """Validate OAuth state and exchange Google's authorization code."""
    expected_state = request.cookies.get("zane_gmail_oauth_state")
    if not expected_state or not secrets.compare_digest(expected_state, state):
        raise HTTPException(status_code=400, detail="Invalid or expired Gmail OAuth state.")

    try:
        email_oauth.exchange_code(code, state)
    except GmailOAuthError as exc:
        logger.exception("Gmail OAuth callback failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = HTMLResponse(
        "<!doctype html><html><head><title>Zane Gmail Authorization</title></head>"
        "<body><h1>Gmail authorization successful</h1>"
        "<p>Google approved Zane's Gmail send permission and issued a refresh token.</p>"
        "<p>The refresh token is intentionally not displayed here. It must be stored "
        "as <code>GMAIL_REFRESH_TOKEN</code> in Zane's server environment before sending email.</p>"
        "</body></html>"
    )
    response.delete_cookie("zane_gmail_oauth_state")
    logger.info("Gmail OAuth authorization completed; refresh token received.")
    return response


@app.delete("/session/{session_id}")
async def end_session(session_id: str) -> Dict[str, str]:
    sessions.drop(session_id)
    return {"status": "session cleared", "session_id": session_id}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Render liveness probe: intentionally lightweight and always 200."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> Dict[str, object]:
    """Readiness/diagnostic endpoint for humans and deployment checks."""
    if not sessions.ready:
        return {
            "status": "not_ready",
            "backend": "unavailable",
            "detail": sessions.startup_error or "backend not initialized",
        }
    return {
        "status": "ready",
        "backend": "ok",
    }
