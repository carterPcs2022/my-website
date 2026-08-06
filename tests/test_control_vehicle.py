import pytest

from zane.analytics_bridge import AnalyticsEngine
from zane.control import (
    InMemoryVehicleControlSink,
    InMemoryVehicleSensor,
    TelemetryLoop,
    TelemetryParseError,
    build_probability_factors,
    compute_avoidance_vector,
    parse_telemetry_frame,
)


def _raw_frame(**overrides):
    base = {
        "vehicle_coordinates": [0.0, 0.0, 0.0],
        "velocity": [10.0, 0.0, 0.0],
        "obstacle_proximity_vectors": [[5.0, 2.0, 0.0]],
        "surface_friction_coefficient": 0.9,
    }
    base.update(overrides)
    return base


def test_parse_telemetry_frame_success():
    frame = parse_telemetry_frame(_raw_frame())
    assert frame.vehicle_coordinates == (0.0, 0.0, 0.0)
    assert frame.obstacle_proximity_vectors == [(5.0, 2.0, 0.0)]
    assert frame.timestamp_ns > 0


def test_parse_telemetry_frame_missing_key_raises():
    raw = _raw_frame()
    del raw["surface_friction_coefficient"]
    with pytest.raises(TelemetryParseError, match="missing required keys"):
        parse_telemetry_frame(raw)


def test_parse_telemetry_frame_wrong_dimension_raises():
    with pytest.raises(TelemetryParseError, match="3 components"):
        parse_telemetry_frame(_raw_frame(vehicle_coordinates=[0.0, 0.0]))


def test_parse_telemetry_frame_non_numeric_raises():
    with pytest.raises(TelemetryParseError, match="non-numeric"):
        parse_telemetry_frame(_raw_frame(velocity=["fast", 0.0, 0.0]))


def test_parse_telemetry_frame_friction_out_of_range_raises():
    with pytest.raises(TelemetryParseError, match="plausible range"):
        parse_telemetry_frame(_raw_frame(surface_friction_coefficient=5.0))


def test_compute_avoidance_vector_pushes_away_from_close_obstacle():
    close_ahead = parse_telemetry_frame(_raw_frame(obstacle_proximity_vectors=[[1.0, 0.0, 0.0]]))
    far_ahead = parse_telemetry_frame(_raw_frame(obstacle_proximity_vectors=[[9.0, 0.0, 0.0]]))
    close_fx, _, _ = compute_avoidance_vector(close_ahead)
    far_fx, _, _ = compute_avoidance_vector(far_ahead)
    # A closer obstacle directly ahead should push the forward component
    # down more than a distant one.
    assert close_fx < far_fx


def test_compute_avoidance_vector_handles_zero_distance_without_crashing():
    frame = parse_telemetry_frame(_raw_frame(obstacle_proximity_vectors=[[0.0, 0.0, 0.0]]))
    vec = compute_avoidance_vector(frame)
    assert vec[0] < 0  # maximal backward repulsion, no ZeroDivisionError


def test_build_probability_factors_more_obstacles_more_complexity():
    one_obstacle = parse_telemetry_frame(_raw_frame(obstacle_proximity_vectors=[[5.0, 0.0, 0.0]]))
    three_obstacles = parse_telemetry_frame(
        _raw_frame(obstacle_proximity_vectors=[[5.0, 0.0, 0.0], [4.0, 1.0, 0.0], [6.0, -1.0, 0.0]])
    )
    factors_one = build_probability_factors(one_obstacle)
    factors_three = build_probability_factors(three_obstacles)
    assert factors_three.complexity_index > factors_one.complexity_index


def test_build_probability_factors_uses_recent_outcomes():
    frame = parse_telemetry_frame(_raw_frame())
    no_history = build_probability_factors(frame, recent_outcomes=())
    good_history = build_probability_factors(frame, recent_outcomes=[True, True, True])
    bad_history = build_probability_factors(frame, recent_outcomes=[False, False, False])
    assert good_history.historical_success_rate == 100.0
    assert bad_history.historical_success_rate == 0.0
    assert no_history.historical_success_rate == 75.0


async def test_telemetry_loop_tick_normal_case_sends_no_emergency():
    sensor = InMemoryVehicleSensor()
    control = InMemoryVehicleControlSink()
    analytics = AnalyticsEngine(worker_threads=1, seed=7)
    loop = TelemetryLoop(sensor, control, analytics, min_confidence_percent=1.0)

    await sensor.push_raw(_raw_frame())
    output = await loop.tick()

    assert -1.0 <= output.steering <= 1.0
    assert 0.0 <= output.throttle <= 1.0
    assert 0.0 <= output.braking <= 1.0
    assert control.emergency_stopped is False
    analytics.shutdown()


async def test_telemetry_loop_tick_emergency_on_close_obstacle():
    sensor = InMemoryVehicleSensor()
    control = InMemoryVehicleControlSink()
    analytics = AnalyticsEngine(worker_threads=1, seed=7)
    loop = TelemetryLoop(sensor, control, analytics, emergency_proximity_m=0.75)

    await sensor.push_raw(_raw_frame(obstacle_proximity_vectors=[[0.2, 0.0, 0.0]]))
    output = await loop.tick()

    assert output.throttle == 0.0
    assert output.braking == 1.0
    assert control.emergency_stopped is True
    analytics.shutdown()


async def test_telemetry_loop_run_forever_stops_after_max_consecutive_errors():
    class _BrokenSensor(InMemoryVehicleSensor):
        async def read(self):
            raise RuntimeError("sensor offline")

    control = InMemoryVehicleControlSink()
    analytics = AnalyticsEngine(worker_threads=1, seed=1)
    loop = TelemetryLoop(
        _BrokenSensor(), control, analytics, tick_hz=1000.0, max_consecutive_errors=3
    )

    await loop.run_forever()  # should exit on its own after 3 failures

    assert control.emergency_stopped is True
    analytics.shutdown()


def test_vehicle_control_output_clamps_out_of_range_values():
    from zane.control import VehicleControlOutput

    output = VehicleControlOutput(steering=2.0, throttle=-1.0, braking=5.0, timestamp_ns=0)
    assert output.steering == 1.0
    assert output.throttle == 0.0
    assert output.braking == 1.0


def test_zane_sensor_input_and_control_output_are_still_properly_generic_abcs():
    from zane.control import ZaneControlOutput, ZaneSensorInput

    with pytest.raises(TypeError):
        ZaneSensorInput()
    with pytest.raises(TypeError):
        ZaneControlOutput()
