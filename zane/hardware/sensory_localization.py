"""Acoustic sound-to-servo mapping: consumes Direction-of-Arrival (DoA)
frames from a microphone array and drives a physical neck-tracking servo
toward the speaker.

Deliberately does **not** reuse `control.py`'s `VehicleControlOutput`/
`ZaneControlOutput[VehicleControlOutput]` for the tracked angle, even
though an earlier spec for this module named it
`ZaneControlOutput.target_neck_angle`: that dataclass models vehicle
steering/throttle/braking, a wholly different actuator domain, and
bolting a neck-servo field onto it would be exactly the kind of
separation-of-concerns mistake this codebase has otherwise been careful
to avoid (a "throttle" has no meaning for a servo tracking a speaker's
voice). `HeadTrackingState` below is a small, purpose-built holder
instead, following the same thread-safe snapshot pattern as
`zane.thermal_monitor.IceProtocolState`.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from zane.hardware.hal import HardwareAbstractionLayer

logger = logging.getLogger("zane.hardware.sensory_localization")

_DEFAULT_SERVO_RANGE_DEG = 180.0
_CENTERED_ANGLE_DEG = 90.0


@dataclass
class HeadTrackingState:
    """Thread-safe holder for the neck servo's current target angle."""

    target_neck_angle: float = _CENTERED_ANGLE_DEG
    last_doa_angle: Optional[float] = None
    last_update_ts: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def snapshot(self) -> "HeadTrackingState":
        with self._lock:
            return HeadTrackingState(
                target_neck_angle=self.target_neck_angle,
                last_doa_angle=self.last_doa_angle,
                last_update_ts=self.last_update_ts,
            )

    def update(self, target_neck_angle: float, doa_angle: float) -> None:
        with self._lock:
            self.target_neck_angle = target_neck_angle
            self.last_doa_angle = doa_angle
            self.last_update_ts = time.time()


def _smooth_circular_angle(previous_deg: float, new_deg: float, alpha: float) -> float:
    """Exponential-moving-average smoothing that stays correct across the
    0°/360° wrap boundary. A naive linear EMA (`prev + alpha*(new-prev)`)
    produces a jump artifact right at the wrap — e.g. blending 359° and 1°
    naively drifts toward 180° instead of the correct ~0°. Averaging unit
    vectors and recovering the angle via `atan2` avoids that; this is the
    standard technique for smoothing circular/angular quantities."""
    prev_rad = math.radians(previous_deg)
    new_rad = math.radians(new_deg)
    x = (1 - alpha) * math.cos(prev_rad) + alpha * math.cos(new_rad)
    y = (1 - alpha) * math.sin(prev_rad) + alpha * math.sin(new_rad)
    return math.degrees(math.atan2(y, x)) % 360.0


def _doa_to_servo_angle(doa_deg: float, servo_range_deg: float = _DEFAULT_SERVO_RANGE_DEG) -> float:
    """Maps a 360°-azimuth DoA reading onto the servo's physical sweep
    (default 0-180°, front-facing). A DoA reading behind the array's
    forward hemisphere is outside the servo's physical reach — Zane's neck
    cannot spin past its mechanical stop — so it clamps to whichever
    physical extreme is angularly closer, rather than silently wrapping."""
    normalized = doa_deg % 360.0
    if normalized <= servo_range_deg:
        return normalized
    distance_to_zero = normalized - servo_range_deg
    distance_to_far_extreme = 360.0 - normalized
    return servo_range_deg if distance_to_far_extreme < distance_to_zero else 0.0


class AcousticLocalizer:
    """Async background subscriber: `push_doa_frame` is the ingestion
    point a real microphone-array driver calls with each new DoA reading;
    `run_forever` consumes the queue, smooths it, and drives the servo.
    Runs independently and continuously, so the servo is generally already
    tracking the speaker by the time any given response begins speaking —
    intentionally not tightly coupled to the voice/TTS pipeline, which
    would add significant synchronization complexity for uncertain
    benefit over just always tracking."""

    def __init__(
        self,
        hal: HardwareAbstractionLayer,
        state: Optional[HeadTrackingState] = None,
        *,
        servo_pin: int = 22,
        smoothing_alpha: float = 0.3,
        servo_range_deg: float = _DEFAULT_SERVO_RANGE_DEG,
        queue_timeout_s: float = 0.1,
        queue_maxsize: int = 64,
    ) -> None:
        self._hal = hal
        self.state = state or HeadTrackingState()
        self.servo_pin = servo_pin
        self.smoothing_alpha = smoothing_alpha
        self.servo_range_deg = servo_range_deg
        self.queue_timeout_s = queue_timeout_s
        self._queue: "asyncio.Queue[float]" = asyncio.Queue(maxsize=queue_maxsize)
        self._running = False

    async def push_doa_frame(self, angle_deg: float) -> None:
        """Ingests one DoA reading. Raises ValueError on an out-of-range
        angle rather than silently clamping — a caller feeding garbage
        data should find out immediately, not have it quietly absorbed."""
        if not (0.0 <= angle_deg <= 360.0):
            raise ValueError(f"DoA angle out of range [0, 360]: {angle_deg}")
        if self._queue.full():
            # Drop the oldest reading rather than blocking the producer —
            # servo tracking cares about the latest direction, not a
            # backlog of stale ones.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            logger.debug("DoA queue full; dropped oldest frame to make room.")
        await self._queue.put(angle_deg)

    async def _tick(self, doa_angle: float) -> None:
        previous = self.state.snapshot().target_neck_angle
        servo_target = _doa_to_servo_angle(doa_angle, self.servo_range_deg)
        smoothed = _smooth_circular_angle(previous, servo_target, self.smoothing_alpha)

        self.state.update(smoothed, doa_angle)
        self._hal.set_servo_angle(self.servo_pin, smoothed)
        logger.debug(
            "[LOCALIZATION] DoA=%.1f° -> servo_target=%.1f° (smoothed from %.1f°)",
            doa_angle, smoothed, previous,
        )

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            try:
                doa_angle = await asyncio.wait_for(self._queue.get(), timeout=self.queue_timeout_s)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            try:
                await self._tick(doa_angle)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad frame must not kill tracking entirely
                logger.exception(
                    "Acoustic localization tick failed; retaining previous neck angle."
                )

    def stop(self) -> None:
        self._running = False
