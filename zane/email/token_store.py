"""Persistent store for Zane's Gmail OAuth refresh token.

Production deployments use Turso so the credential survives Render restarts
and redeploys. A local file fallback remains available for development when
Turso is not configured. The refresh token is never returned by an API route
or written to application logs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .errors import GmailConfigurationError


class GmailTokenStore:
    """Persist a Gmail refresh token, preferring Turso in production."""

    def __init__(self, path: str | None = None, client: Any | None = None) -> None:
        self.path = Path(path or os.getenv("GMAIL_TOKEN_PATH", "/var/data/zane-gmail-refresh-token"))
        self._client = client
        self._initialized = False

    @property
    def uses_turso(self) -> bool:
        return bool(os.getenv("TURSO_DATABASE_URL") and os.getenv("TURSO_AUTH_TOKEN")) or self._client is not None

    def _turso(self) -> Any | None:
        if self._client is not None:
            self._ensure_schema(self._client)
            return self._client
        url = os.getenv("TURSO_DATABASE_URL")
        token = os.getenv("TURSO_AUTH_TOKEN")
        if not url or not token:
            return None
        try:
            from libsql_client import create_client_sync
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise GmailConfigurationError(
                "Turso persistence requires the libsql-client package."
            ) from exc
        self._client = create_client_sync(url, auth_token=token)
        self._ensure_schema(self._client)
        return self._client

    def _ensure_schema(self, client: Any) -> None:
        if self._initialized:
            return
        client.execute(
            "CREATE TABLE IF NOT EXISTS zane_secret_store "
            "(secret_name TEXT PRIMARY KEY, secret_value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        self._initialized = True

    def save(self, refresh_token: str) -> None:
        token = refresh_token.strip()
        if not token:
            raise GmailConfigurationError("Cannot store an empty Gmail refresh token.")

        client = self._turso()
        if client is not None:
            client.execute(
                "INSERT INTO zane_secret_store (secret_name, secret_value, updated_at) "
                "VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(secret_name) DO UPDATE SET "
                "secret_value=excluded.secret_value, updated_at=excluded.updated_at",
                ("gmail_refresh_token", token),
            )
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(token, encoding="utf-8")
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        temp.replace(self.path)

    def load(self) -> str | None:
        client = self._turso()
        if client is not None:
            result = client.execute(
                "SELECT secret_value FROM zane_secret_store WHERE secret_name = ?",
                ("gmail_refresh_token",),
            )
            if result.rows:
                return str(result.rows[0][0]) or None
            return None

        if not self.path.is_file():
            return None
        token = self.path.read_text(encoding="utf-8").strip()
        return token or None

    def delete(self) -> None:
        client = self._turso()
        if client is not None:
            client.execute(
                "DELETE FROM zane_secret_store WHERE secret_name = ?",
                ("gmail_refresh_token",),
            )
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
            self._client = None
            self._initialized = False
