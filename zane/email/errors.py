"""Errors shared by Zane's Gmail integration."""
from __future__ import annotations


class GmailConfigurationError(RuntimeError):
    """Raised when Gmail OAuth configuration is incomplete."""
