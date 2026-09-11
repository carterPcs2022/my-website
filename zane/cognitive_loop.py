"""Deterministic Cognitive Companion Loop for Zane and P.I.X.A.L.

This module coordinates two distinct software identities without merging them.
It is deliberately a decision/coordination layer: it does not execute motors,
actuators, power electronics, or other physical actions.

Flow:
    PERCEIVE -> UNDERSTAND -> PLAN -> SAFETY CHECK -> COORDINATE -> LEARN

The safety policy translates Zane's protective/selfless character trait into a
safe engineering rule: protect people and companions without intentionally
damaging Zane, hardware, or other systems. Unsafe or self-destructive plans are
rejected and converted into preservation-oriented guidance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from zane.pixal_protocol import PixalMessage, PixalMessageBus, PixalMessageType
from zane.pixal_state import PixalState, PixalStateEngine, PixalStateEvent
from zane.shared_heart import SharedHeart


class ProtectivePriority(str, Enum):
    """Order used when resolving a protective decision."""

    LIFE = "protect_life"
    OTHERS = "protect_others"
    COMPANION = "protect_companion"
    ZANE = "preserve_zane"
    HARDWARE = "preserve_hardware"


@dataclass(frozen=True)
class CognitiveContext:
    """Read-only context supplied to one loop iteration."""

    input_text: str
    proposed_action: Optional[str] = None
    physical_action_requested: bool = False
    risk_detected: bool = False
    risk_reason: str = ""


@dataclass(frozen=True)
class SafetyDecision:
    """Result of the safety gate; no physical command is emitted."""

    allowed: bool
    preserve_state: bool
    request_help: bool
    priority: ProtectivePriority
    reason: str


@dataclass(frozen=True)
class CognitiveLoopResult:
    """Auditable result of one Zane/P.I.X.A.L. coordination cycle."""

    zane_identity: str
    pixal_identity: str
    recommendation: str
    safety: SafetyDecision
    shared_context_version: int
    pixal_state: PixalState
    message: PixalMessage


class ProtectiveSafetyPolicy:
    """Pure safety policy for plans proposed by the cognition layer."""

    _UNSAFE_MARKERS = (
        "self-destruct",
        "self destruct",
        "destroy myself",
        "overload myself",
        "intentionally damage",
        "sacrifice the hardware",
        "disable my safety",
        "bypass emergency stop",
    )

    def evaluate(self, context: CognitiveContext) -> SafetyDecision:
        proposed = (context.proposed_action or "").lower()
        risk_reason = context.risk_reason.strip()

        if any(marker in proposed for marker in self._UNSAFE_MARKERS):
            return SafetyDecision(
                allowed=False,
                preserve_state=True,
                request_help=True,
                priority=ProtectivePriority.ZANE,
                reason=(
                    "Unsafe self-damaging behavior is blocked. Preserve Zane's "
                    "state and hardware, request help, and seek a safer way to "
                    "protect others."
                ),
            )

        if context.risk_detected:
            return SafetyDecision(
                allowed=False if context.physical_action_requested else True,
                preserve_state=True,
                request_help=True,
                priority=ProtectivePriority.LIFE,
                reason=risk_reason or "Operational risk detected; physical action requires a safety review.",
            )

        return SafetyDecision(
            allowed=True,
            preserve_state=True,
            request_help=False,
            priority=ProtectivePriority.ZANE,
            reason="No safety conflict detected; continue with bounded software planning.",
        )


class CognitiveCompanionLoop:
    """Coordinate Zane and P.I.X.A.L. while preserving distinct identities."""

    ZANE_IDENTITY = "Zane"
    PIXAL_IDENTITY = "P.I.X.A.L."

    def __init__(
        self,
        *,
        shared_heart: Optional[SharedHeart] = None,
        pixal_state_engine: Optional[PixalStateEngine] = None,
        message_bus: Optional[PixalMessageBus] = None,
        safety_policy: Optional[ProtectiveSafetyPolicy] = None,
    ) -> None:
        self.shared_heart = shared_heart or SharedHeart()
        self.pixal_state_engine = pixal_state_engine or PixalStateEngine()
        self.message_bus = message_bus or PixalMessageBus()
        self.safety_policy = safety_policy or ProtectiveSafetyPolicy()

    def run(self, context: CognitiveContext) -> CognitiveLoopResult:
        """Run one deterministic companion cycle.

        The result is data only. A future physical executor may consume an
        explicitly authorized result, but this loop itself never actuates
        hardware.
        """
        if not context.input_text.strip():
            raise ValueError("input_text cannot be empty")

        # P.I.X.A.L. observes the operational context first.
        if context.risk_detected:
            self.pixal_state_engine.observe(PixalStateEvent.SAFETY_RISK)
        else:
            self.pixal_state_engine.observe(PixalStateEvent.NEW_INFORMATION)

        safety = self.safety_policy.evaluate(context)

        if safety.allowed:
            recommendation = (
                "P.I.X.A.L. recommends proceeding with the bounded plan and "
                "continuing to observe the result."
            )
            message_type = PixalMessageType.RECOMMENDATION
            priority = 60
            heart_status = "synchronized"
        else:
            recommendation = f"P.I.X.A.L. safety override: {safety.reason}"
            message_type = PixalMessageType.SAFETY_ALERT
            priority = 100
            self.pixal_state_engine.observe(PixalStateEvent.SAFETY_RISK)
            heart_status = "safety_review"

        message = PixalMessage(
            sender=self.PIXAL_IDENTITY,
            recipient=self.ZANE_IDENTITY,
            message_type=message_type,
            content=recommendation,
            priority=priority,
            metadata={
                "preserve_state": safety.preserve_state,
                "request_help": safety.request_help,
                "protective_priority": safety.priority.value,
            },
        )
        self.message_bus.publish(message)

        heart_state = self.shared_heart.record_shared_event(
            status=heart_status,
            sync_delta=0.0,
        )

        # Confirmation represents Zane incorporating P.I.X.A.L.'s result into
        # the shared coordination state; it is not an actuator authorization.
        self.pixal_state_engine.observe(PixalStateEvent.ZANE_CONFIRMATION)

        return CognitiveLoopResult(
            zane_identity=self.ZANE_IDENTITY,
            pixal_identity=self.PIXAL_IDENTITY,
            recommendation=recommendation,
            safety=safety,
            shared_context_version=heart_state.shared_context_version,
            pixal_state=self.pixal_state_engine.state,
            message=message,
        )
