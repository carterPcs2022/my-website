"""Minimal Gmail API client for Zane's outbound email tool.

OAuth credentials are loaded from private environment variables plus the
secure token store. No Gmail password is used, and this module requests only
the gmail.send scope.
"""
from __future__ import annotations

import base64
import os
from email.message import EmailMessage
from typing import Any

from .models import EmailDraft, EmailSendResult
from .token_store import GmailTokenStore

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


class GmailConfigurationError(RuntimeError):
    """Raised when Gmail OAuth configuration is incomplete."""


class GmailEmailClient:
    """Send email through Gmail using a server-side OAuth refresh token."""

    def __init__(self, service: Any | None = None, token_store: GmailTokenStore | None = None) -> None:
        self._service = service
        self._token_store = token_store or GmailTokenStore()

    def _build_service(self) -> Any:
        if self._service is not None:
            return self._service

        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        refresh_token = os.getenv("GMAIL_REFRESH_TOKEN") or self._token_store.load()

        if not client_id or not client_secret or not refresh_token:
            raise GmailConfigurationError(
                "Gmail OAuth requires GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "
                "and a stored GMAIL_REFRESH_TOKEN."
            )

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise GmailConfigurationError(
                "Install google-api-python-client, google-auth, and google-auth-oauthlib."
            ) from exc

        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=[GMAIL_SEND_SCOPE],
        )
        credentials.refresh(Request())
        self._service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        return self._service

    @staticmethod
    def _message(draft: EmailDraft, sender: str) -> dict[str, str]:
        message = EmailMessage()
        message["To"] = draft.to
        message["From"] = sender
        message["Subject"] = draft.subject
        message.set_content(draft.body)
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        return {"raw": encoded}

    def send(self, draft: EmailDraft) -> EmailSendResult:
        sender = os.getenv("GMAIL_SENDER_ADDRESS")
        if not sender:
            raise GmailConfigurationError("GMAIL_SENDER_ADDRESS is not configured.")

        service = self._build_service()
        response = service.users().messages().send(
            userId="me",
            body=self._message(draft, sender),
        ).execute()
        return EmailSendResult(
            message_id=str(response.get("id", "")),
            thread_id=response.get("threadId"),
        )
