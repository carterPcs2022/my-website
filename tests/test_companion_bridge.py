import asyncio
import threading

import pytest

from zane.companion_bridge import (
    COMPANION_NOTICE_TAG,
    NeuralBridge,
    PixalSystemsCore,
    VehicleAnomaly,
    build_pixal_system_prompt,
)


class _FakeGroqClient:
    """PixalSystemsCore never actually calls the Groq client in
    analyze_vehicle_telemetry (detection is deterministic, see the module
    docstring) — a plain sentinel is enough to construct her."""


# --- build_pixal_system_prompt -----------------------------------------------


def test_build_pixal_system_prompt_contains_base_persona():
    prompt = build_pixal_system_prompt()
    assert "P.I.X.A.L." in prompt
    assert "Never break character" in prompt


def test_build_pixal_system_prompt_includes_context_notes():
    prompt = build_pixal_system_prompt(["Depth nominal.", "Battery at 12.1V."])
    assert "Depth nominal." in prompt
    assert "Battery at 12.1V." in prompt


# --- VehicleAnomaly -----------------------------------------------------------


def test_vehicle_anomaly_notice_text_uses_exact_required_tag():
    anomaly = VehicleAnomaly(
        source="falcon_worker", metric="api_latency_ms", value=2000.0, threshold=1500.0,
        severity="WARNING",
    )
    text = anomaly.as_notice_text()
    assert text.startswith(f"{COMPANION_NOTICE_TAG} P.I.X.A.L. reports")
    assert "WARNING" in text
    assert "falcon_worker.api_latency_ms" in text


# --- NeuralBridge: queueing --------------------------------------------------


async def test_flag_and_drain_round_trip():
    bridge = NeuralBridge()
    anomaly = VehicleAnomaly(source="s", metric="m", value=1.0, threshold=0.5, severity="WARNING")
    await bridge.flag_anomaly(anomaly)
    drained = await bridge.drain_pending()
    assert drained == [anomaly]
    # drain_pending empties the queue.
    assert await bridge.drain_pending() == []


async def test_drain_pending_on_empty_bridge_returns_empty_list():
    bridge = NeuralBridge()
    assert await bridge.drain_pending() == []


async def test_queue_full_drops_oldest_to_admit_newest():
    bridge = NeuralBridge(maxsize=2)
    a1 = VehicleAnomaly(source="s", metric="m1", value=1.0, threshold=0.5, severity="WARNING")
    a2 = VehicleAnomaly(source="s", metric="m2", value=1.0, threshold=0.5, severity="WARNING")
    a3 = VehicleAnomaly(source="s", metric="m3", value=1.0, threshold=0.5, severity="WARNING")

    await bridge.flag_anomaly(a1)
    await bridge.flag_anomaly(a2)
    await bridge.flag_anomaly(a3)  # queue was full at (a1, a2); a1 dropped

    drained = await bridge.drain_pending()
    assert [a.metric for a in drained] == ["m2", "m3"]


# --- NeuralBridge: prompt injection -------------------------------------------


async def test_inject_into_prompt_returns_unchanged_prompt_when_nothing_pending():
    bridge = NeuralBridge()
    prompt, notice_text = await bridge.inject_into_prompt("BASE PROMPT")
    assert prompt == "BASE PROMPT"
    assert notice_text is None


async def test_inject_into_prompt_appends_tagged_notice_block():
    bridge = NeuralBridge()
    anomaly = VehicleAnomaly(
        source="amphibious_bounty", metric="depth_m", value=200.0, threshold=150.0,
        severity="CRITICAL",
    )
    await bridge.flag_anomaly(anomaly)

    prompt, notice_text = await bridge.inject_into_prompt("BASE PROMPT")
    assert prompt.startswith("BASE PROMPT")
    assert COMPANION_NOTICE_TAG in prompt
    assert notice_text is not None
    assert COMPANION_NOTICE_TAG in notice_text
    # Prompt is consumed once — a second injection sees nothing pending.
    prompt2, notice2 = await bridge.inject_into_prompt("BASE PROMPT")
    assert prompt2 == "BASE PROMPT"
    assert notice2 is None


# --- NeuralBridge: thread-safety / deadlock resistance ------------------------


async def test_flag_anomaly_threadsafe_without_bound_loop_logs_and_drops():
    bridge = NeuralBridge()
    anomaly = VehicleAnomaly(source="s", metric="m", value=1.0, threshold=0.5, severity="WARNING")
    bridge.flag_anomaly_threadsafe(anomaly)  # no loop bound; must not raise
    assert await bridge.drain_pending() == []


async def test_flag_anomaly_threadsafe_from_a_real_background_thread():
    bridge = NeuralBridge()
    bridge.bind_loop(asyncio.get_running_loop())
    anomaly = VehicleAnomaly(source="s", metric="m", value=9.0, threshold=1.0, severity="CRITICAL")

    thread = threading.Thread(target=bridge.flag_anomaly_threadsafe, args=(anomaly,))
    thread.start()
    thread.join(timeout=2.0)

    # Give the scheduled coroutine a moment to actually run on this loop.
    for _ in range(20):
        drained = await bridge.drain_pending()
        if drained:
            assert drained[0].metric == "m"
            return
        await asyncio.sleep(0.01)
    pytest.fail("Anomaly flagged from a background thread never reached the queue.")


# --- PixalSystemsCore: analyze_vehicle_telemetry -------------------------------


async def test_analyze_reports_nominal_when_no_thresholds_breached():
    bridge = NeuralBridge()
    pixal = PixalSystemsCore(_FakeGroqClient(), bridge)
    summary = await pixal.analyze_vehicle_telemetry({"api_latency_ms": 100.0})
    assert "nominal" in summary.lower()
    assert await bridge.drain_pending() == []


async def test_analyze_flags_latency_breach_from_falcon_worker():
    bridge = NeuralBridge()
    pixal = PixalSystemsCore(_FakeGroqClient(), bridge, latency_threshold_ms=1500.0)
    summary = await pixal.analyze_vehicle_telemetry({"api_latency_ms": 3000.0})
    assert "latency" in summary.lower()
    drained = await bridge.drain_pending()
    assert len(drained) == 1
    assert drained[0].source == "falcon_worker"
    assert drained[0].severity == "WARNING"


async def test_analyze_flags_low_battery_as_critical():
    bridge = NeuralBridge()
    pixal = PixalSystemsCore(_FakeGroqClient(), bridge, battery_voltage_threshold_v=10.5)
    summary = await pixal.analyze_vehicle_telemetry({"battery_voltage_v": 9.0})
    assert "battery" in summary.lower()
    drained = await bridge.drain_pending()
    assert drained[0].severity == "CRITICAL"
    assert drained[0].source == "amphibious_bounty"


async def test_analyze_flags_critical_depth_breach():
    bridge = NeuralBridge()
    pixal = PixalSystemsCore(_FakeGroqClient(), bridge, critical_depth_threshold_m=150.0)
    summary = await pixal.analyze_vehicle_telemetry({"depth_m": 175.0})
    assert "depth" in summary.lower()
    drained = await bridge.drain_pending()
    assert drained[0].metric == "depth_m"
    assert drained[0].severity == "CRITICAL"


async def test_analyze_flags_multiple_simultaneous_breaches():
    bridge = NeuralBridge()
    pixal = PixalSystemsCore(_FakeGroqClient(), bridge)
    await pixal.analyze_vehicle_telemetry(
        {"api_latency_ms": 5000.0, "battery_voltage_v": 8.0, "depth_m": 160.0}
    )
    drained = await bridge.drain_pending()
    assert len(drained) == 3


async def test_analyze_ignores_missing_and_non_numeric_metrics():
    bridge = NeuralBridge()
    pixal = PixalSystemsCore(_FakeGroqClient(), bridge)
    summary = await pixal.analyze_vehicle_telemetry({"battery_voltage_v": "not a number"})
    assert "nominal" in summary.lower()


async def test_analyze_never_raises_reports_degraded_status_instead():
    bridge = NeuralBridge()
    pixal = PixalSystemsCore(_FakeGroqClient(), bridge)
    summary = await pixal.analyze_vehicle_telemetry(["not", "a", "dict"])  # type: ignore[arg-type]
    assert "internal error" in summary.lower() or "could not be confirmed" in summary.lower()
