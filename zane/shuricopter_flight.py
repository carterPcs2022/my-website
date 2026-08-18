"""ShuriCopter automated flight controller — a 12x real-world scaling of
the official LEGO 70673 technical assembly.

Same standalone "vehicle telemetry" seam as `zane/control.py` and
`zane/amphibious_bounty.py`: `ZaneSensorInput`/`ZaneControlOutput` are
`Generic` so this module can register its own telemetry/command shapes
into the existing pipeline without changing the abstract contract. Not
wired into `SharedBackend`/`ZaneMind.respond()` for the same reason
`control.py`'s own `TelemetryLoop` isn't — there is no live flight
simulator feeding this project's actual chat deployment.

DESIGN NOTE — the "circular-mean" heading requirement: the spec asks for
`telemetry.yaw_heading_deg % 360.0` to "handle the 0°/360° heading wrap
discontinuity smoothly." A naive `current - previous` on raw headings
breaks exactly at that boundary (359° -> 1° reads as a -358° swing
instead of the true +2°) — `zane/hardware/sensory_localization.py`
already solved this exact problem for acoustic direction-of-arrival
smoothing, and `_circular_delta_deg` below applies the same technique
here: it is the yaw-rate-damping feedback term for heading-hold
stabilization, not a literal control target (the telemetry schema has no
explicit target-heading field to track against).

DESIGN NOTE — `ice_blaster_armed`: the spec's `process_flight_step`
signature takes only `telemetry`, so "target acquisition" and "external
override" cannot be call-time arguments; they are external state set via
`set_target_acquired`/`set_external_override`, read at tick time — the
same shape as `PeripheralManager.arm()`/`disarm()` being external state
read by `engage_cryo_discharge`. The actual relay fire still goes through
`PeripheralManager.engage_cryo_discharge`, which keeps its own hard
arm/duration/cooldown gate (see `peripheral_io.py`'s safety note) as the
real safety boundary — this controller's judgment about *when* to ask for
a discharge is never trusted as that boundary, exactly as an LLM's isn't.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from zane.control import ZaneControlOutput, ZaneSensorInput
from zane.hardware.peripheral_io import PeripheralManager

logger = logging.getLogger("zane.shuricopter_flight")

# --------------------------------------------------------------------------
# Physical structural constants (12x-scale manual figures).
# --------------------------------------------------------------------------

SHURICOPTER_MASS_KG = 680.0
SHURICOPTER_MAIN_ROTOR_SPAN_FT = 12.5
# Minimum safe throttle floor to keep rotor RPM above a voltage-sag stall
# risk — used as the safe-hold throttle on an internal failure, and as a
# floor during a tilt-recovery maneuver.
CRITICAL_VOLTAGE_THROTTLE = 0.25

PITCH_SAFETY_LIMIT_DEG = 25.0
ROLL_SAFETY_LIMIT_DEG = 20.0
GROUND_EFFECT_CLEARANCE_M = 2.0
GROUND_EFFECT_LIFT_MULTIPLIER = 1.15

_BASE_HOVER_THROTTLE = 0.55
_ICE_BLASTER_REQUESTED_DURATION_S = 1.5


@dataclass(frozen=True)
class ShuriCopterTelemetry:
    """One physics-vector sample. Immutable — a sensor reading is a
    historical fact once captured. Ranges below are the sensor's
    documented physical envelope; unlike the control output, telemetry
    input is trusted rather than clamped (unusual readings are exactly
    what a controller needs to see, not silently reshape)."""

    pitch_deg: float  # -90.0 .. 90.0
    roll_deg: float  # -90.0 .. 90.0
    yaw_heading_deg: float  # 0.0 .. 360.0
    climb_rate_ms: float
    ground_clearance_m: float
    rotor_rpm: float


@dataclass(frozen=True)
class ShuriCopterControlOutput:
    """Normalized actuator signals mapped directly to brushless ESCs.
    Out-of-range values are clamped (and logged) here — an ESC given
    main_throttle=1.4 is a bug worth catching at the boundary, not
    passing through to the hardware."""

    main_throttle: float  # 0.0 .. 1.0
    cyclic_pitch: float  # -1.0 .. 1.0
    cyclic_roll: float  # -1.0 .. 1.0
    tail_rotor_yaw: float  # -1.0 .. 1.0
    ice_blaster_armed: bool
    diagnostics: str = ""

    def __post_init__(self) -> None:
        clamped_throttle = max(0.0, min(1.0, self.main_throttle))
        clamped_pitch = max(-1.0, min(1.0, self.cyclic_pitch))
        clamped_roll = max(-1.0, min(1.0, self.cyclic_roll))
        clamped_yaw = max(-1.0, min(1.0, self.tail_rotor_yaw))
        if (clamped_throttle, clamped_pitch, clamped_roll, clamped_yaw) != (
            self.main_throttle, self.cyclic_pitch, self.cyclic_roll, self.tail_rotor_yaw,
        ):
            logger.warning(
                "[NINDROID FLIGHT ENGINE]: ShuriCopterControlOutput received "
                "out-of-range actuator values; clamped to safe ESC bounds."
            )
        # frozen=True: __setattr__ is disabled outside __init__/__post_init__
        # machinery, so clamping requires object.__setattr__ here.
        object.__setattr__(self, "main_throttle", clamped_throttle)
        object.__setattr__(self, "cyclic_pitch", clamped_pitch)
        object.__setattr__(self, "cyclic_roll", clamped_roll)
        object.__setattr__(self, "tail_rotor_yaw", clamped_yaw)


def _circular_delta_deg(current_deg: float, reference_deg: float) -> float:
    """Smallest signed angular delta `current - reference`, wrapped into
    [-180, 180]. Naive subtraction gets the 359°->1° transition wrong
    (reads as -358° instead of +2°); this doesn't."""
    return (current_deg - reference_deg + 180.0) % 360.0 - 180.0


class ShuriCopterAutopilot:
    def __init__(
        self,
        *,
        target_altitude_m: float = 15.0,
        altitude_kp: float = 0.05,
        pitch_kp: float = 0.04,
        roll_kp: float = 0.04,
        yaw_kp: float = 0.02,
        peripheral_manager: Optional[PeripheralManager] = None,
    ) -> None:
        self.target_altitude_m = target_altitude_m
        self.altitude_kp = altitude_kp
        self.pitch_kp = pitch_kp
        self.roll_kp = roll_kp
        self.yaw_kp = yaw_kp
        self._peripheral_manager = peripheral_manager

        self._last_yaw_heading_deg: Optional[float] = None
        self._target_acquired = False
        self._external_override_active = False
        self._blaster_previously_armed = False
        self._blaster_task: "Optional[asyncio.Task]" = None

    # --- external state (read at tick time; see the module docstring) ------

    def set_target_acquired(self, acquired: bool) -> None:
        self._target_acquired = acquired

    def set_external_override(self, active: bool) -> None:
        self._external_override_active = active

    # --- flight control ------------------------------------------------------

    async def process_flight_step(self, telemetry: ShuriCopterTelemetry) -> ShuriCopterControlOutput:
        """Runs one flight-control tick. Never raises: an internal failure
        is logged and answered with a minimal safe-hold throttle rather
        than propagating — a bad tick during a high-frequency simulation
        run must not crash the loop or the airframe."""
        try:
            return await self._process(telemetry)
        except Exception:  # noqa: BLE001 - a bad tick must not crash the loop
            logger.exception(
                "[NINDROID FLIGHT ENGINE]: Flight step failed; issuing minimal "
                "safe-hold throttle."
            )
            return ShuriCopterControlOutput(
                main_throttle=CRITICAL_VOLTAGE_THROTTLE,
                cyclic_pitch=0.0,
                cyclic_roll=0.0,
                tail_rotor_yaw=0.0,
                ice_blaster_armed=False,
                diagnostics="Flight step failed; holding minimal safe throttle.",
            )

    async def _process(self, telemetry: ShuriCopterTelemetry) -> ShuriCopterControlOutput:
        # Linear altitude tracking against ground_clearance_m (the only
        # altitude-like signal this telemetry schema provides — see the
        # module docstring for the yaw-wrap design note above; the same
        # "map the spec onto the actual available fields" reasoning
        # applies here).
        altitude_error_m = self.target_altitude_m - telemetry.ground_clearance_m
        throttle = _BASE_HOVER_THROTTLE + self.altitude_kp * altitude_error_m

        if telemetry.ground_clearance_m < GROUND_EFFECT_CLEARANCE_M:
            throttle *= GROUND_EFFECT_LIFT_MULTIPLIER
            logger.info(
                "[NINDROID FLIGHT ENGINE]: Ground-effect zone entered (clearance=%.2fm); "
                "applying %.0f%% lift multiplier.",
                telemetry.ground_clearance_m, (GROUND_EFFECT_LIFT_MULTIPLIER - 1.0) * 100.0,
            )

        # Circular-mean yaw-rate damping (see module docstring).
        current_heading = telemetry.yaw_heading_deg % 360.0
        if self._last_yaw_heading_deg is None:
            yaw_rate_estimate = 0.0
        else:
            yaw_rate_estimate = _circular_delta_deg(current_heading, self._last_yaw_heading_deg)
        self._last_yaw_heading_deg = current_heading
        tail_rotor_yaw = max(-1.0, min(1.0, -self.yaw_kp * yaw_rate_estimate))

        pitch_correction = max(-1.0, min(1.0, -self.pitch_kp * telemetry.pitch_deg))
        roll_correction = max(-1.0, min(1.0, -self.roll_kp * telemetry.roll_deg))

        tilt_exceeded = (
            abs(telemetry.pitch_deg) > PITCH_SAFETY_LIMIT_DEG
            or abs(telemetry.roll_deg) > ROLL_SAFETY_LIMIT_DEG
        )
        if tilt_exceeded:
            logger.critical(
                "[NINDROID FLIGHT ENGINE]: Structural tilt safety window exceeded "
                "(pitch=%.1f° roll=%.1f°); forcefully clamping cyclic vectors to "
                "maximum corrective authority to prevent frame flip.",
                telemetry.pitch_deg, telemetry.roll_deg,
            )
            pitch_correction = -1.0 if telemetry.pitch_deg > 0 else 1.0
            roll_correction = -1.0 if telemetry.roll_deg > 0 else 1.0
            throttle = max(CRITICAL_VOLTAGE_THROTTLE, min(throttle, 0.6))

        should_arm_blaster = self._target_acquired or self._external_override_active
        if should_arm_blaster and not self._blaster_previously_armed:
            logger.info(
                "[NINDROID FLIGHT ENGINE]: Ice blaster arm criteria satisfied "
                "(target_acquired=%s, external_override=%s); relaying trigger to "
                "cryo-discharge peripheral.",
                self._target_acquired, self._external_override_active,
            )
            self._fire_ice_blaster()
        self._blaster_previously_armed = should_arm_blaster

        diagnostics = (
            f"Telemetry delta resolved. Adjusting cyclic vector constants... "
            f"(altitude_error={altitude_error_m:.2f}m throttle={throttle:.2f} "
            f"pitch={pitch_correction:.2f} roll={roll_correction:.2f} yaw={tail_rotor_yaw:.2f})"
        )
        logger.info("[NINDROID FLIGHT ENGINE]: %s", diagnostics)

        return ShuriCopterControlOutput(
            main_throttle=throttle,
            cyclic_pitch=pitch_correction,
            cyclic_roll=roll_correction,
            tail_rotor_yaw=tail_rotor_yaw,
            ice_blaster_armed=should_arm_blaster,
            diagnostics=diagnostics,
        )

    # --- ice blaster relay hookup --------------------------------------------

    def _fire_ice_blaster(self) -> None:
        if self._peripheral_manager is None:
            logger.warning(
                "[NINDROID FLIGHT ENGINE]: Ice blaster trigger requested but no "
                "PeripheralManager is bound; no physical relay to fire."
            )
            return
        if self._blaster_task is not None and not self._blaster_task.done():
            logger.debug(
                "[NINDROID FLIGHT ENGINE]: Ice blaster relay call already in "
                "flight; skipping duplicate trigger."
            )
            return
        logger.info(
            "[NINDROID FLIGHT ENGINE]: Relaying ice blaster discharge request to "
            "cryo-discharge peripheral (final arm/duration/cooldown gating enforced there)."
        )
        self._blaster_task = asyncio.create_task(
            self._fire_and_log(_ICE_BLASTER_REQUESTED_DURATION_S)
        )

    async def _fire_and_log(self, duration_s: float) -> None:
        try:
            result = await self._peripheral_manager.engage_cryo_discharge(duration_s)
            logger.info("[NINDROID FLIGHT ENGINE]: Ice blaster peripheral response: %s", result)
        except Exception:  # noqa: BLE001 - a fire-and-forget task must not go unhandled
            logger.exception("[NINDROID FLIGHT ENGINE]: Ice blaster discharge relay call failed.")


# --------------------------------------------------------------------------
# Reference ZaneSensorInput/ZaneControlOutput implementations — same
# pattern as control.py's InMemoryVehicleSensor/InMemoryVehicleControlSink.
# --------------------------------------------------------------------------


class InMemoryShuriCopterSensor(ZaneSensorInput[ShuriCopterTelemetry]):
    def __init__(self, maxsize: int = 32) -> None:
        self._queue: "asyncio.Queue[ShuriCopterTelemetry]" = asyncio.Queue(maxsize=maxsize)
        self._subscribers: List[Callable[[ShuriCopterTelemetry], Awaitable[None]]] = []

    async def push_frame(self, frame: ShuriCopterTelemetry) -> None:
        await self._queue.put(frame)
        for callback in list(self._subscribers):
            await callback(frame)

    async def read(self) -> ShuriCopterTelemetry:
        return await self._queue.get()

    async def subscribe(
        self, callback: Callable[[ShuriCopterTelemetry], Awaitable[None]]
    ) -> None:
        self._subscribers.append(callback)


class InMemoryShuriCopterControlSink(ZaneControlOutput[ShuriCopterControlOutput]):
    def __init__(self) -> None:
        self.sent: List[ShuriCopterControlOutput] = []
        self.emergency_stopped = False

    async def send(self, command: ShuriCopterControlOutput) -> None:
        self.sent.append(command)

    async def emergency_stop(self) -> None:
        self.emergency_stopped = True
        self.sent.append(
            ShuriCopterControlOutput(
                main_throttle=CRITICAL_VOLTAGE_THROTTLE,
                cyclic_pitch=0.0,
                cyclic_roll=0.0,
                tail_rotor_yaw=0.0,
                ice_blaster_armed=False,
                diagnostics="EMERGENCY STOP",
            )
        )
