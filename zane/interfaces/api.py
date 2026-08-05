"""FastAPI deployment surface for Zane's digital mind.

Run with:  uvicorn zane.interfaces.api:app --host 0.0.0.0 --port 8000
(or `python -m zane --mode api`)

One `SharedBackend` (Groq client, C++ analytics engine, web search tool) is
built once at startup and shared across every session; each `session_id`
gets its own `ZaneMind` (i.e. its own conversation memory + humor toggle)
lazily on first use.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from zane.config import settings
from zane.core import SharedBackend, ZaneMind
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


class CommandRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    command: str = Field(..., min_length=1, description="e.g. 'humor', 'reset', 'help'.")


class CommandResponse(BaseModel):
    result: str


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
            self._sessions[session_id] = ZaneMind(shared=self._backend)
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

    return ChatResponse(
        reply=result.text,
        tool_calls_made=result.tool_calls_made,
        elapsed_ms=result.elapsed_ms,
        humor_enabled=mind.humor.enabled,
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
