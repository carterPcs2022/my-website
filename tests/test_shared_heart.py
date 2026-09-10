from zane.shared_heart import SharedHeart, SharedHeartState


def test_shared_heart_clamps_state() -> None:
    heart = SharedHeart(SharedHeartState(connection_strength=2.0, trust=-1.0, synchronization=3.0))
    assert heart.state.connection_strength == 1.0
    assert heart.state.trust == 0.0
    assert heart.state.synchronization == 1.0


def test_shared_event_advances_context() -> None:
    heart = SharedHeart()
    before = heart.state.shared_context_version
    heart.record_shared_event(status="synchronized", sync_delta=0.1)
    assert heart.state.status == "synchronized"
    assert heart.state.shared_context_version == before + 1
    assert heart.state.synchronization == 1.0


def test_trust_is_bounded() -> None:
    heart = SharedHeart()
    heart.set_trust(4.0)
    assert heart.state.trust == 1.0
    heart.set_trust(-2.0)
    assert heart.state.trust == 0.0
