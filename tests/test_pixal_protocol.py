import pytest

from zane.pixal_protocol import PixalMessage, PixalMessageType, PixalMessageBus


def test_message_requires_content() -> None:
    with pytest.raises(ValueError):
        PixalMessage("zane", "pixal", PixalMessageType.STATUS, "")


def test_safety_messages_are_critical() -> None:
    message = PixalMessage("pixal", "zane", PixalMessageType.SAFETY_ALERT, "Battery low")
    assert message.is_safety_critical


def test_bus_prioritizes_and_drains_recipient() -> None:
    bus = PixalMessageBus(max_messages=10)
    bus.publish(PixalMessage("pixal", "zane", PixalMessageType.STATUS, "normal", priority=10))
    bus.publish(PixalMessage("pixal", "zane", PixalMessageType.SAFETY_ALERT, "urgent", priority=95))
    bus.publish(PixalMessage("pixal", "other", PixalMessageType.STATUS, "other"))

    pending = bus.pending_for("zane")
    assert pending[0].content == "urgent"
    drained = bus.drain_for("zane")
    assert len(drained) == 2
    assert len(bus.pending_for("zane")) == 0
    assert len(bus.pending_for("other")) == 1
