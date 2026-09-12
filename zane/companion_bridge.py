"""P.I.X.A.L. companion bridge and bounded companion-state model wiring.

P.I.X.A.L. runs alongside Zane as a logical companion/co-processor. This
module keeps telemetry detection deterministic and adds an explicit,
bounded software state model that can influence future companion behavior.
State changes never directly actuate hardware.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from zane.pixal_state import PixalState, PixalStateEngine, PixalStateEvent
from zane.pixal_turso_memory import build_pixal_memory_store

if TYPE_CHECKING:
    from zane.groq_client import AsyncGroqClient
    from zane.pixal_memory import PixalMemoryStore

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
trusted systems officer would — concise, accurate, and actionable. Do not \
pretend to have human experiences or consciousness; describe internal \
state as software state when relevant.
"""


def build_pixal_system_prompt(context_notes: Optional[List[str]] = None) -> str:
    """Build P.I.X.A.L.'s dynamic system prompt with optional context."""
    parts = [PIXAL_BASE_PERSONA]
    if context_notes:
        notes = "\n".join(f"- {n}" for n in context_notes)
        parts.append("\nCURRENT OPERATIONAL CONTEXT:\n" + notes)
    return "\n".join(parts)


@dataclass
class VehicleAnomaly:
    """One threshold breach detected by P.I.X.A.L."""

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
    """Thread-safe bounded async queue for P.I.X.A.L. notices."""

    def __init__(self, *, maxsize: int = 100) -> None:
        self._queue: "asyncio.Queue[VehicleAnomaly]" = asyncio.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    async def flag_anomaly(self, anomaly: VehicleAnomaly) -> None:
        try:
            self._queue.put_nowait(anomaly)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(anomaly)
            logger.warning(
                "[COMPANION BRIDGE]: NeuralBridge queue full; dropped oldest anomaly."
            )

    def flag_anomaly_threadsafe(self, anomaly: VehicleAnomaly) -> None:
        with self._lock:
            loop = self._loop
        if loop is None or loop.is_closed():
            logger.warning(
                "[COMPANION BRIDGE]: no bound event loop; dropping background anomaly: %s",
                anomaly,
            )
            return
        asyncio.run_coroutine_threadsafe(self.flag_anomaly(anomaly), loop)

    async def drain_pending(self) -> List[VehicleAnomaly]:
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
        anomalies = await self.drain_pending()
        if not anomalies:
            return base_prompt, None
        notice_block = self.format_notices(anomalies)
        logger.info(
            "[COMPANION BRIDGE]: Injecting %d P.I.X.A.L. anomaly notice(s).",
            len(anomalies),
        )
        return base_prompt + "\n\n" + notice_block, notice_block


class PixalSystemsCore:
    """P.I.X.A.L.'s systems, companion-state, and durable-memory core.

    Telemetry analysis remains deterministic and on-demand. The state engine
    is deterministic and bounded, while companion memory persists important
    operational events in P.I.X.A.L.'s dedicated Turso database when the
    PIXAL_TDB_URL and PIXAL_TAT environment variables are configured.
    """

    def __init__(
        self,
        groq: "AsyncGroqClient",
        neural_bridge: NeuralBridge,
        *,
        latency_threshold_ms: float = 1500.0,
        battery_voltage_threshold_v: float = 10.5,
        critical_depth_threshold_m: float = 150.0,
        initial_state: Optional[PixalState] = None,
        memory_store: Optional["PixalMemoryStore"] = None,
    ) -> None:
        self._groq = groq
        self._bridge = neural_bridge
        self.latency_threshold_ms = latency_threshold_ms
        self.battery_voltage_threshold_v = battery_voltage_threshold_v
        self.critical_depth_threshold_m = critical_depth_threshold_m
        self.state_engine = PixalStateEngine(initial_state)
        self.memory = memory_store or build_pixal_memory_store()

    @property
    def state(self) -> PixalState:
        """Current bounded software state."""
        return self.state_engine.state

    def observe_state_event(
        self, event: PixalStateEvent, *, intensity: float = 1.0
    ) -> PixalState:
        """Apply a non-hardware state transition and return the new state."""
        return self.state_engine.observe(event, intensity=intensity)

    def state_snapshot(self) -> Dict[str, float]:
        """Return a serialization-friendly state snapshot."""
        return self.state_engine.snapshot()

    def remember(self, content: str, *, importance: float = 0.5, tags: Tuple[str, ...] = ()) -> str:
        """Persist a bounded companion memory and return its stable id."""
        entry = self.memory.remember(
            content,
            role="pixal",
            importance=importance,
            tags=tags,
        )
        return entry.memory_id

    def recall(self, query: str, *, limit: int = 5) -> List[str]:
        """Recall relevant P.I.X.A.L. memories without exposing storage details."""
        return [entry.content for entry in self.memory.relevant(query, limit=limit)]

    async def analyze_vehicle_telemetry(self, metrics: Dict[str, Any]) -> str:
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
                self.remember(
                    anomaly.as_notice_text(),
                    importance=0.9 if anomaly.severity == "CRITICAL" else 0.7,
                    tags=("telemetry", anomaly.severity.lower(), anomaly.metric),
                )

            if anomalies:
                self.observe_state_event(
                    PixalStateEvent.SAFETY_RISK,
                    intensity=max(0.0, min(1.0, len(anomalies) / 3.0)),
                )
            else:
                self.observe_state_event(PixalStateEvent.SYSTEM_NOMINAL)

            if not findings:
                summary = "All monitored vehicle systems nominal; no anomalies detected."
            else:
                summary = "Anomalies detected: " + " ".join(findings)

            logger.info("[PIXAL SYSTEMS CORE]: %s", summary)
            return summary
        except Exception:  # noqa: BLE001
            logger.exception("[PIXAL SYSTEMS CORE]: Telemetry analysis failed.")
            self.observe_state_event(PixalStateEvent.TASK_FAILED)
            return (
                "P.I.X.A.L. telemetry analysis encountered an internal error; "
                "system status could not be confirmed."
            )
