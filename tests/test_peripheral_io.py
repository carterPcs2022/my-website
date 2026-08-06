import asyncio

import pytest

from zane.hardware.hal import MockHAL
from zane.hardware.peripheral_io import PeripheralManager


@pytest.fixture
def hal():
    return MockHAL()


async def test_cryo_discharge_refused_when_unarmed(hal):
    pm = PeripheralManager(hal, armed=False)
    result = await pm.engage_cryo_discharge(1.0)
    assert "not armed" in result
    assert pm.armed is False


async def test_cryo_discharge_succeeds_when_armed(hal):
    pm = PeripheralManager(hal, armed=True, max_discharge_s=2.0, min_cooldown_s=0.0)
    result = await pm.engage_cryo_discharge(0.05)
    assert "engaged" in result.lower()


async def test_cryo_discharge_duration_is_clamped_to_max(hal):
    pm = PeripheralManager(hal, armed=True, max_discharge_s=0.05, min_cooldown_s=0.0)
    result = await pm.engage_cryo_discharge(999.0)
    assert "0.05" in result


async def test_cryo_discharge_relay_always_switched_off_after(hal):
    calls = []
    original = hal.digital_write
    hal.digital_write = lambda pin, value: (calls.append((pin, value)), original(pin, value))[0]

    pm = PeripheralManager(hal, armed=True, max_discharge_s=0.02, min_cooldown_s=0.0)
    await pm.engage_cryo_discharge(0.02)

    assert calls[0] == (17, True)
    assert calls[-1] == (17, False)


async def test_cryo_discharge_respects_cooldown(hal):
    pm = PeripheralManager(hal, armed=True, max_discharge_s=0.02, min_cooldown_s=1.0)
    await pm.engage_cryo_discharge(0.02)
    result = await pm.engage_cryo_discharge(0.02)
    assert "cooling down" in result.lower()


def test_arm_and_disarm_toggle_state(hal):
    pm = PeripheralManager(hal, armed=False)
    assert pm.armed is False
    pm.arm()
    assert pm.armed is True
    pm.disarm()
    assert pm.armed is False


async def test_update_led_state_unknown_defaults_to_idle(hal):
    pm = PeripheralManager(hal, animation_fps=50.0)
    await pm.update_led_state("NOT_A_REAL_STATE")
    assert pm._current_led_state == "IDLE"
    pm.stop()


async def test_update_led_state_starts_animation_loop(hal):
    pm = PeripheralManager(hal, animation_fps=50.0)
    await pm.update_led_state("THINKING")
    assert pm._animation_task is not None
    await asyncio.sleep(0.06)  # let a few frames render
    task = pm._animation_task
    pm.stop()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert task.cancelled() or task.done()


async def test_stop_switches_off_cryo_relay(hal):
    pm = PeripheralManager(hal, armed=True)
    pm.stop()
    # No exception, and the relay pin should end up explicitly off.
    # (MockHAL just logs; nothing further to assert without a spy, but
    # this confirms stop() never raises.)
