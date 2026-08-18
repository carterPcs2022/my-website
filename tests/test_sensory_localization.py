import asyncio

import pytest

from zane.hardware.hal import MockHAL
from zane.hardware.sensory_localization import (
    AcousticLocalizer,
    HeadTrackingState,
    _doa_to_servo_angle,
    _smooth_circular_angle,
)


def test_smooth_circular_angle_avoids_wraparound_bug():
    # The classic bug case: naive linear EMA of 359 and 1 drifts toward
    # 180; correct circular smoothing stays near the 0/360 boundary.
    result = _smooth_circular_angle(359.0, 1.0, alpha=0.5)
    assert result < 10 or result > 350


def test_smooth_circular_angle_normal_case():
    result = _smooth_circular_angle(90.0, 100.0, alpha=0.5)
    assert result == pytest.approx(95.0, abs=0.5)


def test_smooth_circular_angle_alpha_zero_keeps_previous():
    result = _smooth_circular_angle(45.0, 200.0, alpha=0.0)
    assert result == pytest.approx(45.0, abs=0.01)


def test_doa_to_servo_angle_within_range_passes_through():
    assert _doa_to_servo_angle(45.0, servo_range_deg=180.0) == 45.0
    assert _doa_to_servo_angle(180.0, servo_range_deg=180.0) == 180.0


def test_doa_to_servo_angle_behind_array_clamps_to_extreme():
    result = _doa_to_servo_angle(270.0, servo_range_deg=180.0)
    assert result in (0.0, 180.0)


def test_head_tracking_state_default_and_update():
    state = HeadTrackingState()
    assert state.target_neck_angle == 90.0
    state.update(target_neck_angle=45.0, doa_angle=45.0)
    snap = state.snapshot()
    assert snap.target_neck_angle == 45.0
    assert snap.last_doa_angle == 45.0
    assert snap.last_update_ts > 0


async def test_push_doa_frame_rejects_out_of_range_angle():
    localizer = AcousticLocalizer(MockHAL())
    with pytest.raises(ValueError):
        await localizer.push_doa_frame(400.0)


async def test_push_doa_frame_drops_oldest_when_queue_full():
    localizer = AcousticLocalizer(MockHAL(), queue_maxsize=2)
    await localizer.push_doa_frame(10.0)
    await localizer.push_doa_frame(20.0)
    await localizer.push_doa_frame(30.0)  # should drop 10.0, not block
    assert localizer._queue.qsize() == 2


async def test_run_forever_updates_state_from_pushed_frames():
    hal = MockHAL()
    localizer = AcousticLocalizer(hal, smoothing_alpha=1.0, queue_timeout_s=0.02)
    task = asyncio.create_task(localizer.run_forever())

    await localizer.push_doa_frame(45.0)
    await asyncio.sleep(0.1)

    snap = localizer.state.snapshot()
    assert snap.last_doa_angle == 45.0
    assert snap.target_neck_angle == pytest.approx(45.0, abs=0.5)

    localizer.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_run_forever_survives_bad_tick_and_keeps_running(monkeypatch):
    hal = MockHAL()
    localizer = AcousticLocalizer(hal, queue_timeout_s=0.02)

    async def _broken_tick(doa_angle):
        raise RuntimeError("simulated hardware fault")

    monkeypatch.setattr(localizer, "_tick", _broken_tick)
    task = asyncio.create_task(localizer.run_forever())

    await localizer.push_doa_frame(10.0)
    await asyncio.sleep(0.05)
    assert not task.done()  # the loop must still be alive after a bad tick

    localizer.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
