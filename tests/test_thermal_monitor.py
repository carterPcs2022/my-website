import pytest

import zane.thermal_monitor as thermal_monitor
from zane.analytics_bridge import AnalyticsEngine
from zane.thermal_monitor import (
    ICE_PROTOCOL_ALERT,
    IceProtocolState,
    ThermalMonitor,
    low_power_respond,
    maybe_handle_ice_protocol,
)


@pytest.fixture
def analytics():
    engine = AnalyticsEngine(worker_threads=1, seed=3)
    yield engine
    engine.shutdown()


def test_poll_once_activates_above_threshold(monkeypatch):
    monkeypatch.setattr(thermal_monitor, "_read_max_temperature_c", lambda: 80.0)
    state = IceProtocolState()
    monitor = ThermalMonitor(state, threshold_c=75.0, hysteresis_c=5.0)

    monitor._poll_once()

    assert state.active is True
    assert state.last_reading_c == 80.0


def test_poll_once_stays_active_within_hysteresis_band(monkeypatch):
    state = IceProtocolState()
    state._update(active=True, reading_c=80.0)
    monitor = ThermalMonitor(state, threshold_c=75.0, hysteresis_c=5.0)

    # 72 is below the 75 threshold but not below the 70 hysteresis floor.
    monkeypatch.setattr(thermal_monitor, "_read_max_temperature_c", lambda: 72.0)
    monitor._poll_once()

    assert state.active is True


def test_poll_once_deactivates_below_hysteresis_band(monkeypatch):
    state = IceProtocolState()
    state._update(active=True, reading_c=80.0)
    monitor = ThermalMonitor(state, threshold_c=75.0, hysteresis_c=5.0)

    monkeypatch.setattr(thermal_monitor, "_read_max_temperature_c", lambda: 65.0)
    monitor._poll_once()

    assert state.active is False


def test_poll_once_no_sensor_leaves_active_state_untouched(monkeypatch):
    monkeypatch.setattr(thermal_monitor, "_read_max_temperature_c", lambda: None)
    state = IceProtocolState()
    state._update(active=True, reading_c=80.0)
    monitor = ThermalMonitor(state, threshold_c=75.0)

    monitor._poll_once()

    # No sensor visible this cycle: state.active must not be silently
    # reset to False, and no fabricated reading is recorded.
    assert state.active is True
    assert state.last_reading_c is None


def test_low_power_respond_always_includes_the_mandated_alert(analytics):
    state = IceProtocolState()
    state._update(active=True, reading_c=90.0)
    text = low_power_respond("hello", state, analytics)
    assert ICE_PROTOCOL_ALERT in text


def test_low_power_respond_status_query_includes_reading(analytics):
    state = IceProtocolState()
    state._update(active=True, reading_c=88.5)
    text = low_power_respond("What is your status?", state, analytics, threshold_c=75.0)
    assert "88.5" in text


def test_low_power_respond_probability_query_uses_local_analytics(analytics):
    state = IceProtocolState()
    state._update(active=True, reading_c=88.5)
    text = low_power_respond("Calculate the probability of success.", state, analytics)
    assert "%" in text


async def test_maybe_handle_ice_protocol_returns_none_when_inactive(analytics):
    state = IceProtocolState()
    result = await maybe_handle_ice_protocol("hello", state, analytics)
    assert result is None


async def test_maybe_handle_ice_protocol_returns_text_when_active(analytics):
    state = IceProtocolState()
    state._update(active=True, reading_c=90.0)
    result = await maybe_handle_ice_protocol("hello", state, analytics)
    assert result is not None
    assert ICE_PROTOCOL_ALERT in result


def test_thermal_monitor_start_stop_lifecycle():
    state = IceProtocolState()
    monitor = ThermalMonitor(state, poll_interval_s=0.05)
    monitor.start()
    assert monitor._thread is not None
    assert monitor._thread.is_alive()
    monitor.stop()
    assert monitor._thread is None
