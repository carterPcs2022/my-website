from zane.pixal_behavior import build_behavior_context, derive_behavior
from zane.pixal_state import PixalState


def test_safety_has_highest_behavior_priority():
    profile = derive_behavior(
        PixalState(concern=0.9, curiosity=0.9, confidence=0.9)
    )
    assert profile.mode == "safety"
    assert profile.tone == "direct"
    assert profile.urgency == 0.9


def test_failure_state_selects_recovery_behavior():
    profile = derive_behavior(PixalState(frustration=0.8, concern=0.1))
    assert profile.mode == "recovery"
    assert profile.tone == "patient"


def test_curiosity_selects_exploration_behavior():
    profile = derive_behavior(PixalState(curiosity=0.8, concern=0.1, frustration=0.1))
    assert profile.mode == "exploration"
    assert profile.tone == "inquisitive"


def test_behavior_context_is_prompt_safe_and_inspectable():
    context = build_behavior_context(PixalState(concern=0.8))
    assert "P.I.X.A.L. BEHAVIOR STATE:" in context
    assert "mode: safety" in context
    assert "tone: direct" in context
