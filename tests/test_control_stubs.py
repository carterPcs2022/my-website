"""TASK 2 groundwork: these are typed interface stubs only, with no
implementation anywhere in the codebase. These tests just confirm the
seam exists and is properly abstract — not that it does anything yet."""
import pytest

from zane.control import ControlCommand, SensorReading, ZaneControlOutput, ZaneSensorInput


def test_sensor_and_control_classes_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ZaneSensorInput()
    with pytest.raises(TypeError):
        ZaneControlOutput()


def test_dataclasses_hold_generic_shape():
    reading = SensorReading(name="speed", value=42.0, unit="mph", timestamp_ns=123)
    assert reading.metadata == {}

    command = ControlCommand(channel="throttle", value=0.5, timestamp_ns=123)
    assert command.metadata == {}


async def test_a_minimal_concrete_subclass_satisfies_the_seam():
    class NoOpSensor(ZaneSensorInput):
        async def read(self) -> SensorReading:
            return SensorReading(name="noop", value=0.0, unit="", timestamp_ns=0)

        async def subscribe(self, callback) -> None:
            return None

    class NoOpControl(ZaneControlOutput):
        async def send(self, command: ControlCommand) -> None:
            return None

        async def emergency_stop(self) -> None:
            return None

    sensor = NoOpSensor()
    control = NoOpControl()
    reading = await sensor.read()
    assert reading.name == "noop"
    await control.send(ControlCommand(channel="brake", value=1.0, timestamp_ns=0))
    await control.emergency_stop()
