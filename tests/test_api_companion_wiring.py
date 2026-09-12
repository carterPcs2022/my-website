from types import SimpleNamespace

from zane.cognitive_loop import CognitiveCompanionLoop
from zane.interfaces.api import SessionManager
from zane.pixal_protocol import PixalMessageBus
from zane.pixal_state import PixalStateEngine
from zane.shared_heart import SharedHeart


def test_session_manager_runs_persistent_companion_cycle():
    manager = SessionManager()
    state_engine = PixalStateEngine()
    manager._backend = SimpleNamespace(pixal=SimpleNamespace(state_engine=state_engine))
    manager._companion_loop = CognitiveCompanionLoop(
        pixal_state_engine=state_engine,
        shared_heart=SharedHeart(),
        message_bus=PixalMessageBus(),
    )

    first = manager.run_companion_cycle("Check the system status.")
    second = manager.run_companion_cycle("Continue monitoring the system.")

    assert first
    assert second
    assert manager._companion_loop.shared_heart.state.shared_context_version == 2
    assert state_engine.state.confidence >= 0.0
