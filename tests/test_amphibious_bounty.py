import dataclasses

import pytest

from zane.amphibious_bounty import (
    AIR_DENSITY_KG_M3,
    BOUNTY_MAX_STRUCTURAL_DEPTH_M,
    FLIGHT_PITCH_LIMIT_DEG,
    WATER_DENSITY_KG_M3,
    AmphibiousAutopilot,
    AmphibiousControlOutput,
    AmphibiousTelemetry,
    BountyState,
    InMemoryAmphibiousControlSink,
    InMemoryAmphibiousSensor,
)
from zane.hardware.hal import MockHAL


def _telemetry(**overrides) -> AmphibiousTelemetry:
    base = dict(
        pitch_deg=0.0,
        roll_deg=0.0,
        yaw_heading_deg=90.0,
        altitude_or_depth_meters=10.0,
        forward_velocity_knot=5.0,
        external_fluid_pressure_psi=14.7,
    )
    base.update(overrides)
    return AmphibiousTelemetry(**base)


def test_bounty_state_has_all_three_nodes():
    assert {s.value for s in BountyState} == {"FLIGHT", "SURFACE", "SUBMERGED"}


def test_telemetry_is_immutable():
    telemetry = _telemetry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        telemetry.pitch_deg = 5.0  # type: ignore[misc]


def test_structural_constants_match_season_14_spec():
    assert BOUNTY_MAX_STRUCTURAL_DEPTH_M == 150.0
    assert AIR_DENSITY_KG_M3 == 1.225
    assert WATER_DENSITY_KG_M3 == 1025.0


# --- FLIGHT mode -----------------------------------------------------------


async def test_flight_mode_levels_toward_zero_on_small_tilt():
    autopilot = AmphibiousAutopilot(MockHAL())
    output = await autopilot.process_control_step(
        _telemetry(pitch_deg=10.0, roll_deg=-5.0), BountyState.FLIGHT
    )
    assert output.mode is BountyState.FLIGHT
    # Corrective output opposes the tilt direction.
    assert output.rotor_pitch_correction < 0
    assert output.rotor_roll_correction > 0
    assert output.ballast_pump_engagement is None


async def test_flight_mode_clamps_to_max_authority_beyond_pitch_limit():
    autopilot = AmphibiousAutopilot(MockHAL())
    output = await autopilot.process_control_step(
        _telemetry(pitch_deg=FLIGHT_PITCH_LIMIT_DEG + 10.0, roll_deg=0.0), BountyState.FLIGHT
    )
    assert output.rotor_pitch_correction == -1.0
    assert output.rotor_thrust_scale == 0.5


# --- SUBMERGED mode ----------------------------------------------------------


async def test_submerged_mode_disables_flight_fields():
    autopilot = AmphibiousAutopilot(MockHAL(), target_depth_m=20.0)
    output = await autopilot.process_control_step(
        _telemetry(altitude_or_depth_meters=20.0), BountyState.SUBMERGED
    )
    assert output.rotor_pitch_correction is None
    assert output.rotor_roll_correction is None
    assert output.rotor_thrust_scale is None
    assert output.ballast_pump_engagement == pytest.approx(0.0)


async def test_submerged_mode_blows_ballast_when_too_deep_relative_to_target():
    autopilot = AmphibiousAutopilot(MockHAL(), target_depth_m=10.0, depth_kp=0.1)
    output = await autopilot.process_control_step(
        _telemetry(altitude_or_depth_meters=15.0), BountyState.SUBMERGED
    )
    # Deeper than target -> must rise -> negative (blow) engagement.
    assert output.ballast_pump_engagement < 0


async def test_submerged_mode_floods_ballast_when_shallower_than_target():
    autopilot = AmphibiousAutopilot(MockHAL(), target_depth_m=10.0, depth_kp=0.1)
    output = await autopilot.process_control_step(
        _telemetry(altitude_or_depth_meters=5.0), BountyState.SUBMERGED
    )
    # Shallower than target -> must dive -> positive (flood) engagement.
    assert output.ballast_pump_engagement > 0


async def test_critical_depth_alarm_forces_emergency_surface_blow():
    hal = MockHAL()
    calls = []
    original = hal.digital_write
    hal.digital_write = lambda pin, value: (calls.append((pin, value)), original(pin, value))[0]

    autopilot = AmphibiousAutopilot(hal, max_structural_depth_m=150.0)
    output = await autopilot.process_control_step(
        _telemetry(altitude_or_depth_meters=200.0), BountyState.SUBMERGED
    )

    assert output.emergency_surface_override is True
    assert output.ballast_pump_engagement == -1.0
    # The relay was actually fired and always switched back off.
    assert calls[0] == (23, True)
    assert calls[-1] == (23, False)


async def test_depth_alarm_does_not_trigger_below_threshold():
    autopilot = AmphibiousAutopilot(MockHAL(), max_structural_depth_m=150.0)
    output = await autopilot.process_control_step(
        _telemetry(altitude_or_depth_meters=149.0), BountyState.SUBMERGED
    )
    assert output.emergency_surface_override is False


# --- SURFACE mode + error handling -------------------------------------------


async def test_surface_mode_returns_neutral_station_keeping():
    autopilot = AmphibiousAutopilot(MockHAL())
    output = await autopilot.process_control_step(_telemetry(), BountyState.SURFACE)
    assert output.mode is BountyState.SURFACE
    assert output.ballast_pump_engagement == 0.0


async def test_process_control_step_never_raises_on_internal_failure(monkeypatch):
    autopilot = AmphibiousAutopilot(MockHAL())

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated sensor fusion failure")

    monkeypatch.setattr(autopilot, "_process_flight", _boom)
    output = await autopilot.process_control_step(_telemetry(), BountyState.FLIGHT)
    assert "failed" in output.diagnostics.lower()
    assert output.rotor_pitch_correction is None


# --- Generic ABC reference implementations -----------------------------------


async def test_in_memory_amphibious_sensor_and_sink_round_trip():
    sensor = InMemoryAmphibiousSensor()
    sink = InMemoryAmphibiousControlSink()
    autopilot = AmphibiousAutopilot(MockHAL())

    await sensor.push_frame(_telemetry(pitch_deg=3.0))
    frame = await sensor.read()
    output = await autopilot.process_control_step(frame, BountyState.FLIGHT)
    await sink.send(output)

    assert len(sink.sent) == 1
    assert sink.sent[0].mode is BountyState.FLIGHT


async def test_in_memory_amphibious_sink_emergency_stop():
    sink = InMemoryAmphibiousControlSink()
    await sink.emergency_stop()
    assert sink.emergency_stopped is True
    assert sink.sent[-1].emergency_surface_override is True


def test_amphibious_control_output_defaults_are_none():
    output = AmphibiousControlOutput(mode=BountyState.FLIGHT)
    assert output.ballast_pump_engagement is None
    assert output.emergency_surface_override is False
