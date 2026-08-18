import asyncio

import pytest

from zane.falcon_worker import SystemFaultState
from zane.hardware.hal import MockHAL
from zane.hardware.hardware_state_controller import HardwareStateController
from zane.personality import HumorSwitch


async def test_gpio_switch_off_engages_override():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    controller = HardwareStateController(hal, humor, gpio_pin=4)
    controller.start()

    hal.simulate_pin_change(4, False)  # switch flipped OFF

    assert humor.locked is True
    assert humor.enabled is False
    await controller.stop()


async def test_gpio_switch_restored_clears_override():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    controller = HardwareStateController(hal, humor, gpio_pin=4)
    controller.start()

    hal.simulate_pin_change(4, False)
    assert humor.locked is True
    hal.simulate_pin_change(4, True)
    assert humor.locked is False

    await controller.stop()


async def test_locked_humor_switch_ignores_manual_on():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    controller = HardwareStateController(hal, humor, gpio_pin=4)
    controller.start()

    hal.simulate_pin_change(4, False)
    humor.on()
    assert humor.enabled is False  # refused while locked
    humor.toggle()
    assert humor.enabled is False  # also refused

    await controller.stop()


async def test_fault_state_engages_and_clears_override():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    fault_state = SystemFaultState()
    controller = HardwareStateController(hal, humor, fault_state=fault_state, fault_poll_interval_s=0.02)
    controller.start()

    fault_state.set(True, reason="db down")
    await asyncio.sleep(0.06)
    assert humor.locked is True

    fault_state.set(False)
    await asyncio.sleep(0.06)
    assert humor.locked is False

    await controller.stop()


async def test_both_triggers_must_clear_before_override_lifts():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    fault_state = SystemFaultState()
    controller = HardwareStateController(
        hal, humor, fault_state=fault_state, gpio_pin=4, fault_poll_interval_s=0.02
    )
    controller.start()

    hal.simulate_pin_change(4, False)
    fault_state.set(True, reason="db down")
    await asyncio.sleep(0.06)
    assert humor.locked is True

    # Restore the GPIO switch, but the fault is still active: must stay locked.
    hal.simulate_pin_change(4, True)
    await asyncio.sleep(0.02)
    assert humor.locked is True

    # Now the fault clears too: should unlock.
    fault_state.set(False)
    await asyncio.sleep(0.06)
    assert humor.locked is False

    await controller.stop()


async def test_socket_command_off_and_on():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    controller = HardwareStateController(hal, humor, socket_port=18765)
    controller.start()
    await asyncio.sleep(0.05)

    reader, writer = await asyncio.open_connection("127.0.0.1", 18765)
    writer.write(b"HUMOR_SWITCH:OFF\n")
    await writer.drain()
    response = await reader.readline()
    writer.close()
    await writer.wait_closed()

    assert response == b"OK\n"
    assert humor.locked is True

    reader2, writer2 = await asyncio.open_connection("127.0.0.1", 18765)
    writer2.write(b"HUMOR_SWITCH:ON\n")
    await writer2.drain()
    response2 = await reader2.readline()
    writer2.close()
    await writer2.wait_closed()

    assert response2 == b"OK\n"
    assert humor.locked is False

    await controller.stop()


async def test_socket_override_clear_command():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    controller = HardwareStateController(hal, humor, socket_port=18766)
    controller.start()
    await asyncio.sleep(0.05)

    reader, writer = await asyncio.open_connection("127.0.0.1", 18766)
    writer.write(b"HUMOR_SWITCH:OFF\n")
    await writer.drain()
    await reader.readline()
    writer.close()
    await writer.wait_closed()
    assert humor.locked is True

    reader2, writer2 = await asyncio.open_connection("127.0.0.1", 18766)
    writer2.write(b"OVERRIDE:CLEAR\n")
    await writer2.drain()
    response = await reader2.readline()
    writer2.close()
    await writer2.wait_closed()

    assert response == b"OK\n"
    assert humor.locked is False

    await controller.stop()


async def test_socket_unknown_command_returns_error():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    controller = HardwareStateController(hal, humor, socket_port=18767)
    controller.start()
    await asyncio.sleep(0.05)

    reader, writer = await asyncio.open_connection("127.0.0.1", 18767)
    writer.write(b"NOT_A_COMMAND\n")
    await writer.drain()
    response = await reader.readline()
    writer.close()
    await writer.wait_closed()

    assert response.startswith(b"ERR")
    await controller.stop()


async def test_stop_is_idempotent_and_safe_with_no_sources_configured():
    hal = MockHAL()
    humor = HumorSwitch(enabled=True)
    controller = HardwareStateController(hal, humor)  # no gpio, no socket, no fault_state
    controller.start()
    await controller.stop()
    await controller.stop()  # must not raise
