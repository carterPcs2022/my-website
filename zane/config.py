"""Central configuration for Zane's digital mind.

All values are sourced from environment variables (optionally loaded from a
.env file via python-dotenv) so the same codebase runs unmodified across the
CLI and API deployments described in the task hooks.
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

    # --- Rolling in-context memory window ---
    memory_max_messages: int = field(default_factory=lambda: _env_int("ZANE_MEMORY_MAX_MESSAGES", 40))
    memory_max_chars: int = field(default_factory=lambda: _env_int("ZANE_MEMORY_MAX_CHARS", 24000))

    # --- Persistent, retrieval-augmented memory ---
    memory_db_path: str = field(
        default_factory=lambda: os.getenv("ZANE_MEMORY_DB_PATH", "zane_memory.sqlite3")
    )
    memory_vector_index_path: str = field(
        default_factory=lambda: os.getenv("ZANE_MEMORY_VECTOR_INDEX_PATH", "zane_memory.faiss")
    )
    memory_embedding_model: str = field(
        default_factory=lambda: os.getenv("ZANE_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    memory_retrieval_top_k: int = field(
        default_factory=lambda: _env_int("ZANE_MEMORY_RETRIEVAL_TOP_K", 5)
    )
    # How many of the most recent raw messages are always kept intact
    # (never summarized/pruned), independent of the in-context rolling window.
    memory_retention_window: int = field(
        default_factory=lambda: _env_int("ZANE_MEMORY_RETENTION_WINDOW", 20)
    )
    # Once more than this many raw messages sit beyond the retention window,
    # the oldest chunk is summarized via an LLM call and pruned.
    memory_summarize_after_n: int = field(
        default_factory=lambda: _env_int("ZANE_MEMORY_SUMMARIZE_AFTER_N", 20)
    )

    # --- Analytics engine ---
    analytics_worker_threads: int = field(
        default_factory=lambda: _env_int("ZANE_ANALYTICS_WORKERS", 2)
    )
    analytics_seed: int = field(default_factory=lambda: _env_int("ZANE_ANALYTICS_SEED", 0))

    # --- Voice (ElevenLabs text-to-speech) ---
    elevenlabs_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ELEVENLABS_API_KEY")
    )
    # Never hardcode a voice ID — always sourced from the environment.
    elevenlabs_voice_id: Optional[str] = field(default_factory=lambda: os.getenv("ZANE_VOICE_ID"))
    elevenlabs_model_id: str = field(
        default_factory=lambda: os.getenv("ZANE_VOICE_MODEL_ID", "eleven_multilingual_v2")
    )
    elevenlabs_output_format: str = field(
        default_factory=lambda: os.getenv("ZANE_VOICE_OUTPUT_FORMAT", "mp3_44100_128")
    )
    elevenlabs_max_retries: int = field(
        default_factory=lambda: _env_int("ZANE_VOICE_MAX_RETRIES", 4)
    )
    # Per-session voice toggle default. Deliberately OFF unless an operator
    # explicitly opts in — text-only behavior must be unaffected either way.
    voice_enabled_default: bool = field(
        default_factory=lambda: _env_bool("ZANE_VOICE_DEFAULT", False)
    )

    # --- Ice Protocol (thermal safety governor) ---
    # Defaults OFF: psutil sensor access is unavailable in most container/
    # cloud deployments (including this project's own Render/Docker
    # target — see zane/thermal_monitor.py's module docstring), so this is
    # meant to be opted into on bare-metal/local/robotics deployments.
    ice_protocol_enabled: bool = field(
        default_factory=lambda: _env_bool("ZANE_ICE_PROTOCOL_ENABLED", False)
    )
    ice_protocol_threshold_c: float = field(
        default_factory=lambda: _env_float("ZANE_ICE_PROTOCOL_THRESHOLD_C", 75.0)
    )
    ice_protocol_hysteresis_c: float = field(
        default_factory=lambda: _env_float("ZANE_ICE_PROTOCOL_HYSTERESIS_C", 5.0)
    )
    ice_protocol_poll_interval_s: float = field(
        default_factory=lambda: _env_float("ZANE_ICE_PROTOCOL_POLL_INTERVAL_S", 5.0)
    )

    # --- Falcon Scout (system log/health daemon) ---
    # Defaults ON: pure observability, no behavioral side effects on
    # responses (unlike the Ice Protocol or the memory defragmenter below).
    falcon_scout_enabled: bool = field(
        default_factory=lambda: _env_bool("ZANE_FALCON_SCOUT_ENABLED", True)
    )
    falcon_scout_interval_s: float = field(
        default_factory=lambda: _env_float("ZANE_FALCON_SCOUT_INTERVAL_S", 60.0)
    )
    falcon_scout_latency_threshold_ms: float = field(
        default_factory=lambda: _env_float("ZANE_FALCON_SCOUT_LATENCY_THRESHOLD_MS", 1500.0)
    )
    falcon_scout_db_latency_threshold_ms: float = field(
        default_factory=lambda: _env_float("ZANE_FALCON_SCOUT_DB_LATENCY_THRESHOLD_MS", 200.0)
    )

    # --- Memory Defragmenter (RAG conflict resolution) ---
    # Defaults OFF: unlike Falcon Scout, this autonomously deletes memory
    # rows (only ever on an explicit LLM-confirmed conflict — see
    # zane/memory/memory_defragmenter.py — but still a real, conservative
    # opt-in given the consequence of a mistake).
    memory_defrag_enabled: bool = field(
        default_factory=lambda: _env_bool("ZANE_MEMORY_DEFRAG_ENABLED", False)
    )
    memory_defrag_interval_s: float = field(
        default_factory=lambda: _env_float("ZANE_MEMORY_DEFRAG_INTERVAL_S", 300.0)
    )
    memory_defrag_similarity_threshold: float = field(
        default_factory=lambda: _env_float("ZANE_MEMORY_DEFRAG_SIMILARITY_THRESHOLD", 0.90)
    )
    memory_defrag_half_life_hours: float = field(
        default_factory=lambda: _env_float("ZANE_MEMORY_DEFRAG_HALF_LIFE_HOURS", 168.0)
    )

    def validate_for_groq(self) -> None:
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it or add it to a .env file "
                "before starting Zane's digital mind."
            )


settings = Settings()
