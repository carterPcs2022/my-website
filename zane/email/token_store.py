"""Small secret store for Zane's Gmail OAuth refresh token.

The refresh token is never returned by an API response or written to logs.
On Render, the token file must live on persistent storage if this store is
used across restarts; otherwise set GMAIL_REFRESH_TOKEN directly as a secret.
"""
from __future__ import annotations

import os
from pathlib import Path

from .gmail import GmailConfigurationError


class GmailTokenStore:
    """Persist a Gmail refresh token outside source control."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or os.getenv("GMAIL_TOKEN_PATH", "/var/data/zane-gmail-refresh-token"))

    def save(self, refresh_token: str) -> None:
        token = refresh_token.strip()
        if not token:
            raise GmailConfigurationError("Cannot store an empty Gmail refresh token.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(token, encoding="utf-8")
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        temp.replace(self.path)

    def load(self) -> str | None:
        if not self.path.is_file():
            return None
        token = self.path.read_text(encoding="utf-8").strip()
        return token or None

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
