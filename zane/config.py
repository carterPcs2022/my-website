"""Central configuration for Zane's digital mind.

All values are sourced from environment variables (optionally loaded from a
.env file via python-dotenv) so the same codebase runs unmodified across the
CLI, Discord bot, and API deployments described in the task hooks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is an optional convenience
    pass


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass
class Settings:
    # --- Groq ---
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    groq_temperature: float = field(default_factory=lambda: _env_float("GROQ_TEMPERATURE", 0.4))
    groq_max_tokens: int = field(default_factory=lambda: _env_int("GROQ_MAX_TOKENS", 1024))
    groq_timeout_s: float = field(default_factory=lambda: _env_float("GROQ_TIMEOUT_S", 30.0))
    groq_max_retries: int = field(default_factory=lambda: _env_int("GROQ_MAX_RETRIES", 5))

    # --- Web search ---
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    search_max_results: int = field(default_factory=lambda: _env_int("SEARCH_MAX_RESULTS", 5))
    search_backend: str = field(
        default_factory=lambda: os.getenv("SEARCH_BACKEND", "auto")
    )  # "auto" | "tavily" | "duckduckgo"

    # --- Personality ---
    humor_enabled_default: bool = field(
        default_factory=lambda: _env_bool("ZANE_HUMOR_DEFAULT", False)
    )

    # --- Memory ---
    memory_max_messages: int = field(default_factory=lambda: _env_int("ZANE_MEMORY_MAX_MESSAGES", 40))
    memory_max_chars: int = field(default_factory=lambda: _env_int("ZANE_MEMORY_MAX_CHARS", 24000))

    # --- Analytics engine ---
    analytics_worker_threads: int = field(
        default_factory=lambda: _env_int("ZANE_ANALYTICS_WORKERS", 2)
    )
    analytics_seed: int = field(default_factory=lambda: _env_int("ZANE_ANALYTICS_SEED", 0))

    # --- Discord (optional deployment) ---
    discord_bot_token: Optional[str] = field(default_factory=lambda: os.getenv("DISCORD_BOT_TOKEN"))

    def validate_for_groq(self) -> None:
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it or add it to a .env file "
                "before starting Zane's digital mind."
            )


settings = Settings()
