"""Request/result models for Zane's Gmail integration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmailDraft:
    """An email prepared for review before it is sent."""

    to: str
    subject: str
    body: str


@dataclass(frozen=True)
class EmailSendResult:
    """Safe summary returned after Gmail accepts an email."""

    message_id: str
    thread_id: str | None = None
