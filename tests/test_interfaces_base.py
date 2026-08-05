import pytest

from zane.interfaces.base import ZaneInterface
from zane.personality import HumorSwitch
from zane.voice.switch import VoiceSwitch


class _FakeMind:
    """Duck-typed stand-in exposing just what ZaneInterface.handle_command
    touches, so these tests don't need a real Groq/memory-backed ZaneMind."""

    def __init__(self, tts=None):
        self.humor = HumorSwitch(enabled=False)
        self.voice = VoiceSwitch(enabled=False)
        self.tts = tts
        self.reset_called = False

    def reset_conversation(self):
        self.reset_called = True


class _ConcreteInterface(ZaneInterface):
    async def start(self):
        return None

    async def stop(self):
        return None


@pytest.fixture
def interface():
    return _ConcreteInterface(_FakeMind())


async def test_humor_command_toggles(interface):
    reply = await interface.handle_command("humor")
    assert "ENGAGED" in reply
    assert interface.mind.humor.enabled is True

    reply = await interface.handle_command("humor")
    assert "DISENGAGED" in reply
    assert interface.mind.humor.enabled is False


async def test_voice_command_toggles(interface):
    interface.mind.tts = object()  # pretend a TTS backend is configured
    reply = await interface.handle_command("voice")
    assert "ENGAGED" in reply
    assert interface.mind.voice.enabled is True

    reply = await interface.handle_command("voice")
    assert "DISENGAGED" in reply
    assert interface.mind.voice.enabled is False


async def test_voice_command_warns_when_no_backend_configured():
    interface = _ConcreteInterface(_FakeMind(tts=None))
    reply = await interface.handle_command("voice")
    assert interface.mind.voice.enabled is True
    assert "not currently configured" in reply


async def test_reset_command_clears_conversation(interface):
    reply = await interface.handle_command("reset")
    assert interface.mind.reset_called is True
    assert "cleared" in reply.lower()


async def test_help_command_lists_voice(interface):
    reply = await interface.handle_command("help")
    assert "voice" in reply


async def test_unrecognized_command_returns_none(interface):
    reply = await interface.handle_command("not_a_real_command")
    assert reply is None
