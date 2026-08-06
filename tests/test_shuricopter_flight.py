import dataclasses

import pytest

from zane.hardware.hal import MockHAL
from zane.hardware.peripheral_io import PeripheralManager
from zane.shuricopter_flight import (
    CRITICAL_VOLTAGE_THROTTLE,
    GROUND_EFFECT_LIFT_MULTIPLIER,
    PITCH_SAFETY_LIMIT_DEG,
    ROLL_SAFETY_LIMIT_DEG,
    InMemoryShuriCopterControlSink,
    InMemoryShuriCopterSensor,
    ShuriCopterAutopilot,
    ShuriCopterControlOutput,
    ShuriCopterTelemetry,
    _circular_delta_deg,
)


def _telemetry(**overrides) -> ShuriCopterTelemetry:
    base = dict(
        pitch_deg=0.0,
        roll_deg=0.0,
        yaw_heading_deg=180.0,
        climb_rate_ms=0.0,
        ground_clearance_m=15.0,
        rotor_rpm=2500.0,
    )
    base.update(overrides)
    return ShuriCopterTelemetry(**base)


def test_telemetry_is_immutable():
    telemetry = _telemetry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        telemetry.pitch_deg = 10.0  # type: ignore[misc]


def test_control_output_clamps_out_of_range_values():
    output = ShuriCopterControlOutput(
        main_throttle=2.0, cyclic_pitch=-3.0, cyclic_roll=5.0, tail_rotor_yaw=-9.0,
        ice_blaster_armed=False,
    )
    assert output.main_throttle == 1.0
    assert output.cyclic_pitch == -1.0
    assert output.cyclic_roll == 1.0
    assert output.tail_rotor_yaw == -1.0


# --- circular yaw wraparound -------------------------------------------------


def test_circular_delta_handles_0_360_wraparound():
    # Naive subtraction (1 - 359 = -358) gets this badly wrong; the true
    # angular distance from 359deg to 1deg is +2deg.
    assert _circular_delta_deg(1.0, 359.0) == pytest.approx(2.0)
    assert _circular_delta_deg(359.0, 1.0) == pytest.approx(-2.0)
    assert _circular_delta_deg(10.0, 10.0) == pytest.approx(0.0)


async def test_yaw_stabilization_stays_small_across_the_wrap_boundary():
    autopilot = ShuriCopterAutopilot(yaw_kp=0.02)
    await autopilot.process_flight_step(_telemetry(yaw_heading_deg=359.0))
    output = await autopilot.process_flight_step(_telemetry(yaw_heading_deg=1.0))
    # A naive linear EMA would see a ~358deg swing and slam this to the
    # -1.0 clamp; the circular-aware version sees the true +2deg delta.
    assert abs(output.tail_rotor_yaw) < 0.1


# --- altitude tracking + ground effect ---------------------------------------


async def test_altitude_tracking_increases_throttle_when_below_target():
    autopilot = ShuriCopterAutopilot(target_altitude_m=15.0)
    low = await autopilot.process_flight_step(_telemetry(ground_clearance_m=5.0))
    autopilot2 = ShuriCopterAutopilot(target_altitude_m=15.0)
    high = await autopilot2.process_flight_step(_telemetry(ground_clearance_m=14.0))
    assert low.main_throttle > high.main_throttle


async def test_ground_effect_multiplier_applies_below_clearance_threshold():
    near_ground = ShuriCopterAutopilot(altitude_kp=0.0)
    far_from_ground = ShuriCopterAutopilot(altitude_kp=0.0)

    near_output = await near_ground.process_flight_step(_telemetry(ground_clearance_m=1.0))
    far_output = await far_from_ground.process_flight_step(_telemetry(ground_clearance_m=5.0))

    assert near_output.main_throttle == pytest.approx(
        far_output.main_throttle * GROUND_EFFECT_LIFT_MULTIPLIER
    )


# --- tilt safeguard ------------------------------------------------------------


async def test_tilt_safeguard_clamps_cyclic_vectors_beyond_pitch_limit():
    autopilot = ShuriCopterAutopilot()
    output = await autopilot.process_flight_step(
        _telemetry(pitch_deg=PITCH_SAFETY_LIMIT_DEG + 5.0, roll_deg=0.0)
    )
    assert output.cyclic_pitch == -1.0
    assert output.main_throttle <= 0.6


async def test_tilt_safeguard_clamps_cyclic_vectors_beyond_roll_limit():
    autopilot = ShuriCopterAutopilot()
    output = await autopilot.process_flight_step(
        _telemetry(pitch_deg=0.0, roll_deg=-(ROLL_SAFETY_LIMIT_DEG + 5.0))
    )
    assert output.cyclic_roll == 1.0


async def test_within_safety_window_uses_proportional_correction_not_clamp():
    autopilot = ShuriCopterAutopilot(pitch_kp=0.04)
    output = await autopilot.process_flight_step(_telemetry(pitch_deg=5.0))
    assert -1.0 < output.cyclic_pitch < 0.0


# --- ice blaster / peripheral_io hookup ---------------------------------------


async def test_ice_blaster_armed_flag_reflects_target_acquisition():
    autopilot = ShuriCopterAutopilot()
    autopilot.set_target_acquired(True)
    output = await autopilot.process_flight_step(_telemetry())
    assert output.ice_blaster_armed is True


async def test_ice_blaster_relays_trigger_to_peripheral_manager():
    hal = MockHAL()
    pm = PeripheralManager(hal, armed=True, max_discharge_s=2.0, min_cooldown_s=0.0)
    autopilot = ShuriCopterAutopilot(peripheral_manager=pm)
    autopilot.set_target_acquired(True)

    output = await autopilot.process_flight_step(_telemetry())
    assert output.ice_blaster_armed is True
    assert autopilot._blaster_task is not None
    await autopilot._blaster_task  # let the fire-and-forget relay call finish


async def test_ice_blaster_only_fires_once_per_rising_edge(monkeypatch):
    autopilot = ShuriCopterAutopilot()
    fire_calls = []
    monkeypatch.setattr(autopilot, "_fire_ice_blaster", lambda: fire_calls.append(1))
    autopilot.set_target_acquired(True)

    await autopilot.process_flight_step(_telemetry())
    await autopilot.process_flight_step(_telemetry())

    assert len(fire_calls) == 1


async def test_ice_blaster_without_peripheral_manager_does_not_crash():
    autopilot = ShuriCopterAutopilot()
    autopilot.set_external_override(True)
    output = await autopilot.process_flight_step(_telemetry())
    assert output.ice_blaster_armed is True  # flag still reported, no relay to fire


# --- error handling -------------------------------------------------------------


async def test_process_flight_step_never_raises_on_internal_failure(monkeypatch):
    autopilot = ShuriCopterAutopilot()

    async def _boom(telemetry):
        raise RuntimeError("simulated IMU failure")

    monkeypatch.setattr(autopilot, "_process", _boom)
    output = await autopilot.process_flight_step(_telemetry())
    assert output.main_throttle == CRITICAL_VOLTAGE_THROTTLE
    assert "failed" in output.diagnostics.lower()


# --- Generic ABC reference implementations ---------------------------------------


async def test_in_memory_shuricopter_sensor_and_sink_round_trip():
    sensor = InMemoryShuriCopterSensor()
    sink = InMemoryShuriCopterControlSink()
    autopilot = ShuriCopterAutopilot()

    await sensor.push_frame(_telemetry(pitch_deg=2.0))
    frame = await sensor.read()
    output = await autopilot.process_flight_step(frame)
    await sink.send(output)

    assert len(sink.sent) == 1


async def test_in_memory_shuricopter_sink_emergency_stop():
    sink = InMemoryShuriCopterControlSink()
    await sink.emergency_stop()
    assert sink.emergency_stopped is True
    assert sink.sent[-1].main_throttle == CRITICAL_VOLTAGE_THROTTLE
