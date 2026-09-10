"""P.I.X.A.L. behavior policy derived from bounded internal state.

This module translates software state into explainable communication guidance.
It does not call an LLM and never controls hardware directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from zane.pixal_state import PixalState


@dataclass(frozen=True)
class PixalBehaviorProfile:
    """Safe, inspectable communication profile for one moment."""

    mode: str
    urgency: float
    tone: str
    guidance: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "urgency": self.urgency,
            "tone": self.tone,
            "guidance": self.guidance,
        }


def derive_behavior(state: PixalState) -> PixalBehaviorProfile:
    """Map state to communication behavior using deterministic priorities."""
    if state.concern >= 0.65:
        return PixalBehaviorProfile(
            mode="safety",
            urgency=state.concern,
            tone="direct",
            guidance="Lead with the relevant risk, state uncertainty clearly, and recommend a safe next step.",
        )
    if state.frustration >= 0.65:
        return PixalBehaviorProfile(
            mode="recovery",
            urgency=state.frustration * 0.7,
            tone="patient",
            guidance="Acknowledge the failed attempt, avoid blame, and propose one clear recovery path.",
        )
    if state.curiosity >= 0.65:
        return PixalBehaviorProfile(
            mode="exploration",
            urgency=0.25,
            tone="inquisitive",
            guidance="Surface useful questions, observations, and relevant alternatives without inventing facts.",
        )
    if state.confidence >= 0.70:
        return PixalBehaviorProfile(
            mode="focused",
            urgency=0.15,
            tone="precise",
            guidance="Be concise, technically clear, and action-oriented.",
        )
    return PixalBehaviorProfile(
        mode="calm",
        urgency=0.10,
        tone="calm",
        guidance="Keep communication steady, concise, and transparent about uncertainty.",
    )


def build_behavior_context(state: PixalState) -> str:
    """Return a compact prompt-safe behavior block for cognition code."""
    profile = derive_behavior(state)
    return (
        "P.I.X.A.L. BEHAVIOR STATE:\n"
        f"- mode: {profile.mode}\n"
        f"- urgency: {profile.urgency:.2f}\n"
        f"- tone: {profile.tone}\n"
        f"- guidance: {profile.guidance}"
    )
