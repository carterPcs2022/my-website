from zane.pixal_state import PixalState, PixalStateEngine, PixalStateEvent


def test_default_state_is_bounded_and_serializable():
    state = PixalState()
    snapshot = state.as_dict()
    assert all(0.0 <= snapshot[name] <= 1.0 for name in (
        "curiosity", "confidence", "concern", "frustration", "calmness", "trust"
    ))


def test_state_clamps_out_of_range_values():
    state = PixalState(
        curiosity=2.0,
        confidence=-1.0,
        concern=5.0,
        frustration=-4.0,
        calmness=3.0,
        trust=-2.0,
    )
    assert state.curiosity == 1.0
    assert state.confidence == 0.0
    assert state.concern == 1.0
    assert state.frustration == 0.0
    assert state.calmness == 1.0
    assert state.trust == 0.0


def test_safety_risk_raises_concern_without_hardware_side_effects():
    engine = PixalStateEngine(PixalState(concern=0.1, calmness=0.8))
    engine.observe(PixalStateEvent.SAFETY_RISK)
    assert engine.state.concern > 0.1
    assert engine.state.calmness < 0.8


def test_nominal_system_reduces_concern_and_increases_calmness():
    engine = PixalStateEngine(PixalState(concern=0.7, calmness=0.3))
    engine.observe(PixalStateEvent.SYSTEM_NOMINAL)
    assert engine.state.concern < 0.7
    assert engine.state.calmness > 0.3


def test_task_failure_and_completion_change_state_predictably():
    engine = PixalStateEngine(PixalState(confidence=0.5, frustration=0.2))
    engine.observe(PixalStateEvent.TASK_FAILED)
    assert engine.state.confidence < 0.5
    assert engine.state.frustration > 0.2

    after_failure = engine.snapshot()
    engine.observe(PixalStateEvent.TASK_COMPLETED)
    assert engine.state.confidence > after_failure["confidence"]
    assert engine.state.frustration < after_failure["frustration"]


def test_dominant_state_is_explainable():
    state = PixalState(
        curiosity=0.2,
        confidence=0.3,
        concern=0.9,
        frustration=0.1,
        calmness=0.4,
    )
    assert state.dominant_state() == "concerned"


def test_engine_observe_returns_same_updated_state_object():
    engine = PixalStateEngine()
    updated = engine.observe(PixalStateEvent.NEW_INFORMATION, intensity=0.5)
    assert updated is engine.state
    assert updated.curiosity > 0.35
