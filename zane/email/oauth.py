"""Google OAuth 2.0 flow for Zane's Gmail integration."""
from __future__ import annotations

import os
from typing import Any

from .errors import GmailConfigurationError
from .gmail import GMAIL_SEND_SCOPE
from .token_store import GmailTokenStore


class GmailOAuthError(RuntimeError):
    """Raised when the Gmail OAuth flow cannot be completed."""


class GmailOAuthManager:
    """Build Google's server-side OAuth flow for Zane's Gmail account."""

    def __init__(self, token_store: GmailTokenStore | None = None) -> None:
        self.token_store = token_store or GmailTokenStore()

    def _config(self) -> dict[str, Any]:
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
        if not client_id or not client_secret or not redirect_uri:
            raise GmailConfigurationError(
                "Gmail OAuth requires GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "
                "and GOOGLE_REDIRECT_URI."
            )
        return {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        }

    def _flow(self, state: str | None = None) -> Any:
        try:
            from google_auth_oauthlib.flow import Flow
        except ImportError as exc:
            raise GmailOAuthError(
                "Install google-api-python-client, google-auth, and google-auth-oauthlib."
            ) from exc

        redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]
        flow = Flow.from_client_config(
            self._config(),
            scopes=[GMAIL_SEND_SCOPE],
            state=state,
        )
        flow.redirect_uri = redirect_uri
        return flow

    def authorization_url(self, state: str) -> str:
        flow = self._flow(state)
        url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return url

    def exchange_code(self, code: str, state: str) -> str:
        flow = self._flow(state)
        try:
            flow.fetch_token(code=code)
        except Exception as exc:
            raise GmailOAuthError("Google authorization code exchange failed.") from exc

        refresh_token = flow.credentials.refresh_token
        if not refresh_token:
            raise GmailOAuthError(
                "Google did not return a refresh token. Re-authorize with consent."
            )
        return refresh_token

    def exchange_and_store(self, code: str, state: str) -> None:
        """Exchange the authorization code and persist only the refresh token."""
        refresh_token = self.exchange_code(code, state)
        try:
            self.token_store.save(refresh_token)
        except Exception as exc:
            raise GmailOAuthError("Gmail authorization succeeded, but token storage failed.") from exc
