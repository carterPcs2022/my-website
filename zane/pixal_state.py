"""P.I.X.A.L. internal companion-state model.

This module represents software-level affect/state for P.I.X.A.L. without
claiming human feelings or consciousness. It is deliberately deterministic,
bounded, and side-effect free: state changes never directly actuate hardware.

The model is intended to become the foundation for companion behavior,
conversation style, prioritization, and Zane <-> P.I.X.A.L. coordination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Dict, Optional


class PixalStateEvent(str, Enum):
    """Small, explainable events that can influence companion state."""

    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SAFETY_RISK = "safety_risk"
    SYSTEM_NOMINAL = "system_nominal"
    NEW_INFORMATION = "new_information"
    ZANE_REQUEST = "zane_request"
    ZANE_CONFIRMATION = "zane_confirmation"


@dataclass
class PixalState:
    """Bounded software state used to shape P.I.X.A.L.'s behavior.

    Values are normalized to 0.0-1.0. They are behavioral control signals,
    not claims that P.I.X.A.L. experiences human emotions.
    """

    curiosity: float = 0.35
    confidence: float = 0.70
    concern: float = 0.05
    frustration: float = 0.00
    calmness: float = 0.80
    trust: float = 0.50
    updated_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        self._clamp_all()

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _clamp_all(self) -> None:
        self.curiosity = self._clamp(self.curiosity)
        self.confidence = self._clamp(self.confidence)
        self.concern = self._clamp(self.concern)
        self.frustration = self._clamp(self.frustration)
        self.calmness = self._clamp(self.calmness)
        self.trust = self._clamp(self.trust)
        self.updated_at = time()

    def as_dict(self) -> Dict[str, float]:
        """Return a serialization-friendly snapshot."""
        return {
            "curiosity": self.curiosity,
            "confidence": self.confidence,
            "concern": self.concern,
            "frustration": self.frustration,
            "calmness": self.calmness,
            "trust": self.trust,
            "updated_at": self.updated_at,
        }

    def apply_event(self, event: PixalStateEvent, *, intensity: float = 1.0) -> None:
        """Apply one bounded event using explainable deterministic rules."""
        intensity = self._clamp(intensity)

        if event is PixalStateEvent.TASK_COMPLETED:
            self.confidence += 0.08 * intensity
            self.frustration -= 0.10 * intensity
            self.calmness += 0.05 * intensity
        elif event is PixalStateEvent.TASK_FAILED:
            self.confidence -= 0.08 * intensity
            self.frustration += 0.12 * intensity
            self.curiosity += 0.04 * intensity
        elif event is PixalStateEvent.SAFETY_RISK:
            self.concern += 0.35 * intensity
            self.calmness -= 0.12 * intensity
        elif event is PixalStateEvent.SYSTEM_NOMINAL:
            self.concern -= 0.12 * intensity
            self.calmness += 0.10 * intensity
        elif event is PixalStateEvent.NEW_INFORMATION:
            self.curiosity += 0.12 * intensity
        elif event is PixalStateEvent.ZANE_REQUEST:
            self.trust += 0.02 * intensity
            self.curiosity += 0.03 * intensity
        elif event is PixalStateEvent.ZANE_CONFIRMATION:
            self.trust += 0.05 * intensity
            self.frustration -= 0.05 * intensity

        self._clamp_all()

    def dominant_state(self) -> str:
        """Return a concise label for downstream behavior selection."""
        scores = {
            "concerned": self.concern,
            "frustrated": self.frustration,
            "curious": self.curiosity,
            "confident": self.confidence,
            "calm": self.calmness,
        }
        return max(scores, key=scores.get)


class PixalStateEngine:
    """Owns P.I.X.A.L.'s current state and exposes explicit transitions."""

    def __init__(self, initial_state: Optional[PixalState] = None) -> None:
        self.state = initial_state or PixalState()

    def observe(self, event: PixalStateEvent, *, intensity: float = 1.0) -> PixalState:
        """Record an event and return the updated state."""
        self.state.apply_event(event, intensity=intensity)
        return self.state

    def snapshot(self) -> Dict[str, float]:
        return self.state.as_dict()
