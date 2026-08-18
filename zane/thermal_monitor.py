"""The 'Ice Protocol': a real host thermal safety governor, modeled on
Zane's in-universe ice-elemental cooling systems, that actually throttles
compute when the host is running hot.

DEPLOYMENT CAVEAT — read before enabling this in production:
`psutil.sensors_temperatures()` only returns real data on Linux hosts with
exposed `hwmon` sensors, never reports GPU temperature on its own, and
returns an empty dict with no error on macOS, Windows, and — critically
for this project's actual Render/Docker deployment target — inside
virtually all containers and VMs, since guest processes have no access to
host thermal sensors there. This module never fabricates a reading to
compensate: when no sensors are visible, `ice_protocol_active` simply
never trips, which is the correct and safe behavior. It's meant for
bare-metal or VM deployments where psutil genuinely has sensor access
(e.g. Zane running locally, or on real robotics hardware) — in the
Render/Docker deployment described in README.md, this feature is inert by
construction, which is why it defaults to disabled (`ZANE_ICE_PROTOCOL_ENABLED`).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from zane.analytics_bridge import AnalyticsEngine, ProbabilityFactors

logger = logging.getLogger("zane.thermal_monitor")

ICE_PROTOCOL_ALERT = (
    "Internal temperature critical. Engaging ice-cooling protocols. "
    "High-compute systems throttled."
)

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is required only if enabled
    psutil = None  # type: ignore


@dataclass
class IceProtocolState:
    """Thread-safe holder for the governor's current state. `ThermalMonitor`
    writes it from its background thread; `maybe_handle_ice_protocol` reads
    it from the (async) request-handling path — hence the lock."""

    active: bool = False
    last_reading_c: Optional[float] = None
    last_check_ts: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def snapshot(self) -> "IceProtocolState":
        with self._lock:
            return IceProtocolState(
                active=self.active, last_reading_c=self.last_reading_c,
                last_check_ts=self.last_check_ts,
            )

    def _update(self, active: bool, reading_c: Optional[float]) -> None:
        with self._lock:
            self.active = active
            self.last_reading_c = reading_c
            self.last_check_ts = time.time()


def _read_max_temperature_c() -> Optional[float]:
    """Returns the hottest currently-reported sensor temperature in
    Celsius, or None if no sensors are visible on this platform/host —
    never raises, never fabricates a value."""
    if psutil is None:
        return None
    reader = getattr(psutil, "sensors_temperatures", None)
    if reader is None:
        # Not implemented on this platform at all (e.g. macOS, Windows).
        return None
    try:
        sensors = reader()
    except Exception as exc:  # noqa: BLE001 - platform sensor backends vary widely
        logger.debug("psutil.sensors_temperatures() failed: %s", exc)
        return None

    if not sensors:
        return None

    readings = [
        entry.current
        for entries in sensors.values()
        for entry in entries
        if entry.current is not None
    ]
    return max(readings) if readings else None


class ThermalMonitor:
    """Background-thread host thermal watchdog. Uses a plain
    `threading.Thread` rather than an asyncio task deliberately: psutil's
    sensor calls are blocking, and this needs to keep sampling even if the
    event loop is busy handling a slow request."""

    def __init__(
        self,
        state: Optional[IceProtocolState] = None,
        *,
        threshold_c: float = 75.0,
        hysteresis_c: float = 5.0,
        poll_interval_s: float = 5.0,
    ) -> None:
        self.state = state or IceProtocolState()
        self.threshold_c = threshold_c
        self.hysteresis_c = hysteresis_c
        self.poll_interval_s = poll_interval_s
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if psutil is None:
            logger.warning(
                "ThermalMonitor enabled but `psutil` is not installed; the Ice "
                "Protocol will never trip. Run `pip install psutil` to enable "
                "real thermal monitoring."
            )
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="zane-ice-protocol", daemon=True)
        self._thread.start()
        logger.info(
            "Ice Protocol thermal monitor started (threshold=%.1f°C, poll=%.1fs).",
            self.threshold_c, self.poll_interval_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_s * 2)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001 - the watchdog must never die silently
                logger.exception("ThermalMonitor poll cycle failed; will retry next interval.")
            self._stop_event.wait(self.poll_interval_s)

    def _poll_once(self) -> None:
        reading = _read_max_temperature_c()
        if reading is None:
            # No sensors visible on this host — leave the current state
            # (almost always inactive) untouched rather than guessing.
            self.state._update(active=self.state.active, reading_c=None)
            return

        was_active = self.state.active
        if not was_active and reading >= self.threshold_c:
            logger.warning(
                "Ice Protocol ENGAGED: host temperature %.1f°C >= threshold %.1f°C.",
                reading, self.threshold_c,
            )
            self.state._update(active=True, reading_c=reading)
        elif was_active and reading < (self.threshold_c - self.hysteresis_c):
            # Hysteresis band avoids rapidly flapping in/out right at the
            # threshold; only deactivates once meaningfully below it.
            logger.info(
                "Ice Protocol DISENGAGED: host temperature %.1f°C dropped below "
                "%.1f°C.", reading, self.threshold_c - self.hysteresis_c,
            )
            self.state._update(active=False, reading_c=reading)
        else:
            self.state._update(active=was_active, reading_c=reading)


# --------------------------------------------------------------------------
# Low-power fallback: a rules-based responder that needs neither Groq nor
# the torch-backed memory stack.
# --------------------------------------------------------------------------

_STATUS_PATTERN = re.compile(r"\b(status|temperature|temp|how (are|is) (you|zane))\b", re.IGNORECASE)
_PROBABILITY_PATTERN = re.compile(r"\b(probability|chance|odds|calculate)\b", re.IGNORECASE)


def low_power_respond(
    user_input: str,
    state: IceProtocolState,
    analytics: AnalyticsEngine,
    threshold_c: float = 75.0,
) -> str:
    """A minimal, local, rules-based responder used only while the Ice
    Protocol is active. Deliberately simple: no LLM call, no embeddings —
    this exists specifically to keep functioning when the heavy stack is
    being conserved."""
    lines = [ICE_PROTOCOL_ALERT]

    if state.last_reading_c is not None:
        lines.append(
            f"Current internal reading: {state.last_reading_c:.1f}°C "
            f"(threshold {threshold_c:.1f}°C)."
        )

    if _STATUS_PATTERN.search(user_input):
        lines.append(
            "All core subsystems remain online in reduced-power mode; only the "
            "high-compute language and memory pipelines are paused."
        )
    elif _PROBABILITY_PATTERN.search(user_input):
        # Local-only estimate: no situational parsing is available without
        # the LLM, so this uses neutral, honestly-labeled default factors
        # rather than pretending to have assessed the actual situation.
        factors = ProbabilityFactors(
            danger_level=50.0, team_synergy=50.0, resource_availability=50.0,
            historical_success_rate=50.0, complexity_index=50.0,
        )
        result = analytics.calculate_success_probability(factors)
        lines.append(
            f"Local analytical core (no situational context available in this "
            f"mode) estimates a baseline {result.success_probability:.2f}% — "
            f"treat this as a placeholder figure, not a real assessment."
        )
    else:
        lines.append("I will address your request in full once thermal levels normalize.")

    return " ".join(lines)


async def maybe_handle_ice_protocol(
    user_input: str,
    state: IceProtocolState,
    analytics: AnalyticsEngine,
    threshold_c: float = 75.0,
) -> Optional[str]:
    """The single interception point `ZaneMind.respond()` checks. Returns
    the low-power response text if the Ice Protocol is active (meaning:
    skip Groq and the RAG memory stack entirely for this turn), or None to
    signal "proceed normally"."""
    snapshot = state.snapshot()
    if not snapshot.active:
        return None
    return low_power_respond(user_input, snapshot, analytics, threshold_c)
