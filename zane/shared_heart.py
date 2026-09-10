"""Shared Heart: a bounded software bond core for Zane and P.I.X.A.L.

The Shared Heart is symbolic/coordination state, not a claim of consciousness.
It does not directly control power electronics or hardware actuators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict


@dataclass
class SharedHeartState:
    """Shared coordination state visible to both companions."""

    connection_strength: float = 1.0
    trust: float = 0.5
    synchronization: float = 1.0
    shared_context_version: int = 0
    status: str = "nominal"
    updated_at: float = field(default_factory=time.time)

    def clamp(self) -> None:
        self.connection_strength = max(0.0, min(1.0, self.connection_strength))
        self.trust = max(0.0, min(1.0, self.trust))
        self.synchronization = max(0.0, min(1.0, self.synchronization))


class SharedHeart:
    """Owns shared coordination state without merging identities."""

    def __init__(self, initial: SharedHeartState | None = None) -> None:
        self.state = initial or SharedHeartState()
        self.state.clamp()

    def record_shared_event(self, *, status: str = "nominal", sync_delta: float = 0.0) -> SharedHeartState:
        self.state.synchronization = max(0.0, min(1.0, self.state.synchronization + sync_delta))
        self.state.status = status
        self.state.shared_context_version += 1
        self.state.updated_at = time.time()
        return self.state

    def set_trust(self, value: float) -> SharedHeartState:
        self.state.trust = max(0.0, min(1.0, float(value)))
        self.state.updated_at = time.time()
        return self.state

    def snapshot(self) -> Dict[str, object]:
        return {
            "connection_strength": self.state.connection_strength,
            "trust": self.state.trust,
            "synchronization": self.state.synchronization,
            "shared_context_version": self.state.shared_context_version,
            "status": self.state.status,
            "updated_at": self.state.updated_at,
        }
