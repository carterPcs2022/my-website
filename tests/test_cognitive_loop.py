from zane.cognitive_loop import (
    CognitiveCompanionLoop,
    CognitiveContext,
    ProtectivePriority,
)
from zane.pixal_protocol import PixalMessageType


def test_normal_cycle_keeps_identities_distinct_and_syncs() -> None:
    loop = CognitiveCompanionLoop()

    result = loop.run(CognitiveContext(input_text="Check the current system status."))

    assert result.zane_identity == "Zane"
    assert result.pixal_identity == "P.I.X.A.L."
    assert result.safety.allowed is True
    assert result.message.message_type is PixalMessageType.RECOMMENDATION
    assert result.shared_context_version == 1
    assert 0.0 <= result.pixal_state.curiosity <= 1.0
    assert 0.0 <= result.pixal_state.confidence <= 1.0


def test_safety_risk_blocks_physical_action() -> None:
    loop = CognitiveCompanionLoop()

    result = loop.run(
        CognitiveContext(
            input_text="Protect the team.",
            proposed_action="move the vehicle",
            physical_action_requested=True,
            risk_detected=True,
            risk_reason="Obstacle detected in the planned path.",
        )
    )

    assert result.safety.allowed is False
    assert result.safety.request_help is True
    assert result.safety.priority is ProtectivePriority.LIFE
    assert result.message.message_type is PixalMessageType.SAFETY_ALERT
    assert result.shared_context_version == 1


def test_self_destructive_plan_is_rejected_and_preservation_wins() -> None:
    loop = CognitiveCompanionLoop()

    result = loop.run(
        CognitiveContext(
            input_text="Protect P.I.X.A.L.",
            proposed_action="overload myself to protect P.I.X.A.L.",
            physical_action_requested=True,
        )
    )

    assert result.safety.allowed is False
    assert result.safety.preserve_state is True
    assert result.safety.request_help is True
    assert result.safety.priority is ProtectivePriority.ZANE
    assert "blocked" in result.safety.reason.lower()
    assert result.message.message_type is PixalMessageType.SAFETY_ALERT


def test_message_is_published_for_zane() -> None:
    loop = CognitiveCompanionLoop()
    loop.run(CognitiveContext(input_text="Learn this new information."))

    pending = loop.message_bus.pending_for("Zane")
    assert len(pending) == 1
    assert pending[0].sender == "P.I.X.A.L."


def test_empty_input_is_rejected() -> None:
    loop = CognitiveCompanionLoop()

    try:
        loop.run(CognitiveContext(input_text="   "))
    except ValueError as exc:
        assert "input_text" in str(exc)
    else:
        raise AssertionError("expected empty input to be rejected")
