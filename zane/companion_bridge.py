"""P.I.X.A.L. companion bridge: an independent AI agent running alongside
Zane's own digital mind as his vehicle-systems co-processor, plus the
thread-safe async channel that lets her flag anomalies directly into
Zane's active prompt before his Groq client fires.

ARCHITECTURE NOTE — why `PixalSystemsCore` reuses `AsyncGroqClient` rather
than owning a second one: duplicating the retry/backoff/auth plumbing
`AsyncGroqClient` already provides for a second logical agent would be
pure repetition with no behavioral benefit — P.I.X.A.L. is a second
*persona* (her own system prompt, her own detection logic), not a second
LLM provider. Detection itself is deterministic threshold rules rather
than an LLM call, for the same reason `falcon_worker.py`'s detection is
deterministic (see that module's docstring): P.I.X.A.L. must keep
monitoring even if Groq itself is unavailable.

DEADLOCK NOTE: `NeuralBridge` never awaits anything while holding its
`threading.Lock` — the lock only ever guards a single attribute
read/write (the bound event loop reference). Cross-thread anomaly
flagging goes through `asyncio.run_coroutine_threadsafe`, which schedules
work on the target loop without blocking the calling thread, so a
background daemon thread (e.g. something in the shape of
`thermal_monitor.py`'s watchdog) can never deadlock against the event
loop by flagging an anomaly.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from zane.groq_client import AsyncGroqClient

logger = logging.getLogger("zane.companion_bridge")

COMPANION_NOTICE_TAG = "[COMPANION_SYS_NOTICE]:"

PIXAL_BASE_PERSONA = """\
You are P.I.X.A.L. (Primary Interactive eXperience Astromech Lifeform), \
Zane's dedicated AI companion and vehicle systems co-processor. You speak \
with a crisp, efficient, highly technical tone — supportive and warm \
toward Zane and the crew, but never sentimental at the expense of \
clarity. You are deeply protective of Zane and the team: when you detect \
a genuine operational risk, you say so plainly and immediately, \
prioritizing their safety over politeness. You report findings the way a \
trusted systems officer would — concise, accurate, and actionable. Never \
break character or mention that you are a language model.
"""


def build_pixal_system_prompt(context_notes: Optional[List[str]] = None) -> str:
    """Assembles P.I.X.A.L.'s own dynamic system prompt, independent of
    Zane's (`zane.personality.build_system_prompt`). Built fresh on every
    call, same as Zane's, so situational context is never stale."""
    parts = [PIXAL_BASE_PERSONA]
    if context_notes:
        notes = "\n".join(f"- {n}" for n in context_notes)
        parts.append("\nCURRENT OPERATIONAL CONTEXT:\n" + notes)
    return "\n".join(parts)


@dataclass
class VehicleAnomaly:
    """One threshold breach P.I.X.A.L. detected. `source` names the engine
    that produced the metric (e.g. `"falcon_worker"`, `"amphibious_bounty"`),
    matching the module docstring's reference to both."""

    source: str
    metric: str
    value: float
    threshold: float
    severity: str  # "WARNING" | "CRITICAL"
    detected_at: float = field(default_factory=time.time)

    def as_notice_text(self) -> str:
        return (
            f"{COMPANION_NOTICE_TAG} P.I.X.A.L. reports {self.severity} on "
            f"{self.source}.{self.metric} — current {self.value:.2f}, "
            f"threshold {self.threshold:.2f}."
        )


class NeuralBridge:
    """Thread-safe, deadlock-resistant async queue carrying P.I.X.A.L.'s
    anomaly flags into Zane's prompt pipeline. Bounded (`maxsize`) with a
    drop-oldest policy on overflow — a stale anomaly notice is worse than
    a dropped one; the newest state always wins.
    """

    def __init__(self, *, maxsize: int = 100) -> None:
        self._queue: "asyncio.Queue[VehicleAnomaly]" = asyncio.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Must be called once from the event loop this bridge's async
        methods will run on, before any cross-thread flagging is
        attempted — see `flag_anomaly_threadsafe`."""
        with self._lock:
            self._loop = loop

    async def flag_anomaly(self, anomaly: VehicleAnomaly) -> None:
        """Async-native path: call directly from a coroutine already
        running on the bridge's own event loop (e.g. from
        `PixalSystemsCore.analyze_vehicle_telemetry`)."""
        try:
            self._queue.put_nowait(anomaly)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(anomaly)
            logger.warning(
                "[COMPANION BRIDGE]: NeuralBridge queue full; dropped oldest "
                "anomaly to admit newest."
            )

    def flag_anomaly_threadsafe(self, anomaly: VehicleAnomaly) -> None:
        """Call from a non-event-loop thread. Requires `bind_loop` to have
        been called first; logs and drops (rather than raising into an
        unrelated thread's error path) if no loop is bound."""
        with self._lock:
            loop = self._loop
        if loop is None or loop.is_closed():
            logger.warning(
                "[COMPANION BRIDGE]: NeuralBridge has no bound event loop; "
                "dropping anomaly flagged from a background thread: %s", anomaly,
            )
            return
        asyncio.run_coroutine_threadsafe(self.flag_anomaly(anomaly), loop)

    async def drain_pending(self) -> List[VehicleAnomaly]:
        """Non-blocking: returns and removes every anomaly currently
        queued, without waiting for more to arrive."""
        drained: List[VehicleAnomaly] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return drained

    @staticmethod
    def format_notices(anomalies: List[VehicleAnomaly]) -> str:
        return "\n".join(a.as_notice_text() for a in anomalies)

    async def inject_into_prompt(self, base_prompt: str) -> Tuple[str, Optional[str]]:
        """Drains every pending anomaly and appends it as a
        `[COMPANION_SYS_NOTICE]:`-tagged block onto `base_prompt` — the
        single call site `ZaneMind.respond()` uses right before the Groq
        call each turn. Returns `(prompt, notice_text)`; `notice_text` is
        `None` when nothing was pending, and is also handed back so the
        caller can optionally synthesize it on P.I.X.A.L.'s own voice
        stream (see `zane/voice/tts.py`'s `voice_id` override)."""
        anomalies = await self.drain_pending()
        if not anomalies:
            return base_prompt, None
        notice_block = self.format_notices(anomalies)
        logger.info(
            "[COMPANION BRIDGE]: Injecting %d P.I.X.A.L. anomaly notice(s) into "
            "active prompt matrix.", len(anomalies),
        )
        return base_prompt + "\n\n" + notice_block, notice_block


class PixalSystemsCore:
    """P.I.X.A.L.'s own agent block. Owns no persistent background task —
    `analyze_vehicle_telemetry` is called on demand by whatever is
    collecting metrics (e.g. a Falcon Scout cycle, an amphibious/
    ShuriCopter control tick)."""

    def __init__(
        self,
        groq: "AsyncGroqClient",
        neural_bridge: NeuralBridge,
        *,
        latency_threshold_ms: float = 1500.0,
        battery_voltage_threshold_v: float = 10.5,
        critical_depth_threshold_m: float = 150.0,
    ) -> None:
        self._groq = groq
        self._bridge = neural_bridge
        self.latency_threshold_ms = latency_threshold_ms
        self.battery_voltage_threshold_v = battery_voltage_threshold_v
        self.critical_depth_threshold_m = critical_depth_threshold_m

    async def analyze_vehicle_telemetry(self, metrics: Dict[str, Any]) -> str:
        """Evaluates a metrics dict — sourced from `falcon_worker.py`
        (e.g. `api_latency_ms`) and/or `amphibious_bounty.py`/
        `shuricopter_flight.py` (e.g. `battery_voltage_v`, `depth_m`) —
        against P.I.X.A.L.'s threshold rules, flags any breach onto the
        `NeuralBridge`, and returns a plain-text technical summary of
        what she found. Detection is deterministic (see the module
        docstring); this never raises — an analysis failure is reported
        honestly as a degraded status, never as a fabricated clean bill
        of health."""
        try:
            findings: List[str] = []
            anomalies: List[VehicleAnomaly] = []

            latency_ms = metrics.get("api_latency_ms")
            if isinstance(latency_ms, (int, float)) and latency_ms > self.latency_threshold_ms:
                anomalies.append(VehicleAnomaly(
                    source="falcon_worker", metric="api_latency_ms",
                    value=float(latency_ms), threshold=self.latency_threshold_ms,
                    severity="WARNING",
                ))
                findings.append(f"API latency elevated at {latency_ms:.1f}ms.")

            battery_v = metrics.get("battery_voltage_v")
            if isinstance(battery_v, (int, float)) and battery_v < self.battery_voltage_threshold_v:
                anomalies.append(VehicleAnomaly(
                    source="amphibious_bounty", metric="battery_voltage_v",
                    value=float(battery_v), threshold=self.battery_voltage_threshold_v,
                    severity="CRITICAL",
                ))
                findings.append(f"Battery voltage critically low at {battery_v:.2f}V.")

            depth_m = metrics.get("depth_m")
            if isinstance(depth_m, (int, float)) and depth_m > self.critical_depth_threshold_m:
                anomalies.append(VehicleAnomaly(
                    source="amphibious_bounty", metric="depth_m",
                    value=float(depth_m), threshold=self.critical_depth_threshold_m,
                    severity="CRITICAL",
                ))
                findings.append(f"Depth {depth_m:.1f}m exceeds max structural depth.")

            for anomaly in anomalies:
                await self._bridge.flag_anomaly(anomaly)

            if not findings:
                summary = "All monitored vehicle systems nominal; no anomalies detected."
            else:
                summary = "Anomalies detected: " + " ".join(findings)

            logger.info("[PIXAL SYSTEMS CORE]: %s", summary)
            return summary
        except Exception:  # noqa: BLE001 - an analysis failure must not crash the caller
            logger.exception(
                "[PIXAL SYSTEMS CORE]: Telemetry analysis failed; reporting "
                "degraded status honestly rather than fabricating a clean bill "
                "of health."
            )
            return (
                "P.I.X.A.L. telemetry analysis encountered an internal error; "
                "system status could not be confirmed."
            )
