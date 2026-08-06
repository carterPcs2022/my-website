"""Season 14 Amphibious Destiny's Bounty ("Land Bounty" successor) dual-mode
vehicle automation: heavy-lift multi-rotor flight drone and deep-sea
submersible in one airframe/hull.

This slots into the same standalone "vehicle telemetry" seam as
`zane/control.py` — `ZaneSensorInput`/`ZaneControlOutput` are `Generic`
specifically so a concrete integration like this one can carry its own
telemetry/command shapes without changing the abstract contract (see
`control.py`'s module docstring). Like `control.py`'s own
`InMemoryVehicleSensor`/`InMemoryVehicleControlSink`, this module is
exercised standalone by its own tests and is not wired into
`SharedBackend`/`ZaneMind.respond()` — there is no live amphibious vehicle
simulator feeding this project's actual chat deployment, exactly as there
is no live driving simulator feeding `control.py`'s `TelemetryLoop` today.

DESIGN DEVIATION FROM THE LITERAL SPEC — documented per this project's
established practice of never silently guessing: the spec's
`process_control_step` signature was written as returning `Any`. This
module returns the concrete `AmphibiousControlOutput` dataclass instead —
`Any` was almost certainly a placeholder for "whatever output type makes
sense," and the project's standing instruction to strictly type-hint
everything takes priority over copying a deliberately vague placeholder
type verbatim.

HAL SAFETY NOTE: the critical-depth emergency surface blow drives a real
compressed-air ballast-blow relay through the same `HardwareAbstractionLayer`
used everywhere else in `zane/hardware/` (see `hal.py`) — mirroring
`peripheral_io.py`'s cryo-discharge relay: the relay is switched off in a
`finally` block so a mid-blow exception or task cancellation can never
leave it energized. If no HAL is supplied, a `MockHAL` is used, so this
module logs safely in this project's cloud/container deployment (and in
tests) without ever touching real hardware.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, List, Optional

from zane.control import ZaneControlOutput, ZaneSensorInput
from zane.hardware.hal import HardwareAbstractionLayer, MockHAL

logger = logging.getLogger("zane.amphibious_bounty")

# --------------------------------------------------------------------------
# Season 14 technical specifications (hardcoded structural constants).
# --------------------------------------------------------------------------

BOUNTY_VEHICLE_MASS_KG = 5400.0
BOUNTY_MAX_STRUCTURAL_DEPTH_M = 150.0
AIR_DENSITY_KG_M3 = 1.225
WATER_DENSITY_KG_M3 = 1025.0

FLIGHT_PITCH_LIMIT_DEG = 25.0
FLIGHT_ROLL_LIMIT_DEG = 25.0

_EMERGENCY_BLOW_RELAY_PIN_DEFAULT = 23
_EMERGENCY_BLOW_DURATION_S = 3.0


class BountyState(Enum):
    FLIGHT = "FLIGHT"
    SURFACE = "SURFACE"
    SUBMERGED = "SUBMERGED"


@dataclass(frozen=True)
class AmphibiousTelemetry:
    """One telemetry sample from the Bounty's flight/dive sensor suite.
    Immutable — a telemetry reading is a historical fact once captured,
    never mutated in place."""

    pitch_deg: float
    roll_deg: float
    yaw_heading_deg: float
    # Positive while FLIGHT means altitude AGL in meters; positive while
    # SUBMERGED means depth below the surface in meters. SURFACE readings
    # are nominally ~0. Single field per the spec rather than two, since
    # the vehicle is never in both regimes at once.
    altitude_or_depth_meters: float
    forward_velocity_knot: float
    external_fluid_pressure_psi: float


@dataclass(frozen=True)
class AmphibiousControlOutput:
    """One control tick's output. Only the fields relevant to `mode` are
    populated (flight rotor corrections vs. submerged ballast engagement);
    the rest stay at their None default — this is the concrete
    documentation of "flight multi-rotor algorithms are disabled entirely"
    while submerged, and vice versa, rather than a comment claiming it."""

    mode: BountyState
    rotor_pitch_correction: Optional[float] = None  # -1.0..1.0, FLIGHT only
    rotor_roll_correction: Optional[float] = None  # -1.0..1.0, FLIGHT only
    rotor_thrust_scale: Optional[float] = None  # 0.0..1.0, FLIGHT only
    # -1.0 = blow ballast tanks to surface, 1.0 = flood tanks to dive,
    # SUBMERGED (and SURFACE, trivially near 0.0) only.
    ballast_pump_engagement: Optional[float] = None
    emergency_surface_override: bool = False
    diagnostics: str = ""


class AmphibiousAutopilot:
    """The transformation engine: routes telemetry to the correct control
    regime for `target_mode`, with a hard depth alarm that overrides
    whatever ballast command the normal control loop would have produced.
    """

    def __init__(
        self,
        hal: Optional[HardwareAbstractionLayer] = None,
        *,
        target_depth_m: float = 20.0,
        depth_kp: float = 0.05,
        flight_pitch_kp: float = 0.04,
        flight_roll_kp: float = 0.04,
        max_structural_depth_m: float = BOUNTY_MAX_STRUCTURAL_DEPTH_M,
        emergency_blow_relay_pin: int = _EMERGENCY_BLOW_RELAY_PIN_DEFAULT,
    ) -> None:
        # Defaults to MockHAL — see the module docstring's HAL safety note.
        self._hal = hal or MockHAL()
        self.target_depth_m = target_depth_m
        self.depth_kp = depth_kp
        self.flight_pitch_kp = flight_pitch_kp
        self.flight_roll_kp = flight_roll_kp
        self.max_structural_depth_m = max_structural_depth_m
        self._emergency_blow_relay_pin = emergency_blow_relay_pin
        self._last_fluid_density_kg_m3 = AIR_DENSITY_KG_M3

    async def process_control_step(
        self, telemetry: AmphibiousTelemetry, target_mode: BountyState
    ) -> AmphibiousControlOutput:
        """Runs one control tick for `target_mode`. Never raises: any
        unexpected failure is logged and answered with a safe-hold output
        (zero rotor authority, zero ballast movement) rather than
        propagating into whatever loop is driving this — a bad tick on a
        vehicle that can fly or dive must not crash the controller."""
        try:
            self._log_fluid_density_transition(target_mode)
            if target_mode is BountyState.FLIGHT:
                return self._process_flight(telemetry)
            if target_mode is BountyState.SUBMERGED:
                return await self._process_submerged(telemetry)
            if target_mode is BountyState.SURFACE:
                return self._process_surface(telemetry)
            raise ValueError(f"Unrecognized BountyState: {target_mode!r}")
        except Exception:  # noqa: BLE001 - a bad tick must not crash the controller
            logger.exception(
                "[AMPHIBIOUS CONTROL CORE]: Control step failed for mode %s; "
                "issuing safe-hold output.", target_mode,
            )
            return AmphibiousControlOutput(
                mode=target_mode,
                diagnostics="Control step failed; holding safe defaults pending recovery.",
            )

    def _log_fluid_density_transition(self, target_mode: BountyState) -> None:
        new_density = (
            WATER_DENSITY_KG_M3
            if target_mode in (BountyState.SUBMERGED, BountyState.SURFACE)
            else AIR_DENSITY_KG_M3
        )
        if new_density != self._last_fluid_density_kg_m3:
            logger.info(
                "[AMPHIBIOUS CONTROL CORE]: Liquid density change confirmed. "
                "Recalibrating control surface drag matrices for %.1f kg/m^3 medium.",
                new_density,
            )
            self._last_fluid_density_kg_m3 = new_density

    # --- FLIGHT: multi-rotor vertical leveling -----------------------------

    def _process_flight(self, telemetry: AmphibiousTelemetry) -> AmphibiousControlOutput:
        pitch = telemetry.pitch_deg
        roll = telemetry.roll_deg

        tilt_exceeded = abs(pitch) > FLIGHT_PITCH_LIMIT_DEG or abs(roll) > FLIGHT_ROLL_LIMIT_DEG
        if tilt_exceeded:
            logger.warning(
                "[AMPHIBIOUS CONTROL CORE]: Angular safeguard triggered (pitch=%.1f° "
                "roll=%.1f°, limit=%.1f°); commanding maximum corrective authority.",
                pitch, roll, FLIGHT_PITCH_LIMIT_DEG,
            )
            pitch_correction = -1.0 if pitch > 0 else 1.0
            roll_correction = -1.0 if roll > 0 else 1.0
            thrust_scale = 0.5  # reduced, steady thrust while recovering attitude
        else:
            pitch_correction = max(-1.0, min(1.0, -self.flight_pitch_kp * pitch))
            roll_correction = max(-1.0, min(1.0, -self.flight_roll_kp * roll))
            tilt_severity = max(abs(pitch), abs(roll)) / FLIGHT_PITCH_LIMIT_DEG
            thrust_scale = max(0.4, 1.0 - 0.3 * min(1.0, tilt_severity))

        diagnostics = (
            f"Multi-rotor vertical leveling: pitch={pitch:.1f}° roll={roll:.1f}° -> "
            f"corrections (pitch={pitch_correction:.2f}, roll={roll_correction:.2f}), "
            f"thrust_scale={thrust_scale:.2f}."
        )
        logger.info("[AMPHIBIOUS CONTROL CORE]: %s", diagnostics)
        return AmphibiousControlOutput(
            mode=BountyState.FLIGHT,
            rotor_pitch_correction=pitch_correction,
            rotor_roll_correction=roll_correction,
            rotor_thrust_scale=thrust_scale,
            diagnostics=diagnostics,
        )

    # --- SUBMERGED: ballast control + critical depth alarm ------------------

    async def _process_submerged(self, telemetry: AmphibiousTelemetry) -> AmphibiousControlOutput:
        # Flight multi-rotor algorithms are entirely bypassed here: this
        # branch never touches rotor_pitch_correction/rotor_roll_correction/
        # rotor_thrust_scale, which stay at AmphibiousControlOutput's None
        # default.
        depth = telemetry.altitude_or_depth_meters

        if depth > self.max_structural_depth_m:
            logger.critical(
                "[AMPHIBIOUS CONTROL CORE]: CRITICAL DEPTH ALARM. Current depth "
                "%.1fm exceeds maximum structural depth %.1fm. Overriding manual "
                "ballast control — commencing emergency compressed-air blow to surface.",
                depth, self.max_structural_depth_m,
            )
            await self._execute_emergency_surface_blow()
            return AmphibiousControlOutput(
                mode=BountyState.SUBMERGED,
                ballast_pump_engagement=-1.0,
                emergency_surface_override=True,
                diagnostics=(
                    f"CRITICAL DEPTH ALARM at {depth:.1f}m (limit "
                    f"{self.max_structural_depth_m:.1f}m); forced emergency surface."
                ),
            )

        depth_error = depth - self.target_depth_m
        # Sign convention: depth_error > 0 means currently deeper than
        # target, which must produce a NEGATIVE engagement (blow ballast,
        # rise) — engagement is the negative of the depth error, not the
        # error itself.
        ballast_pump_engagement = max(-1.0, min(1.0, -self.depth_kp * depth_error))
        diagnostics = (
            f"Ballast control: depth={depth:.1f}m target={self.target_depth_m:.1f}m "
            f"error={depth_error:.1f}m -> engagement={ballast_pump_engagement:.2f}."
        )
        logger.info("[AMPHIBIOUS CONTROL CORE]: %s", diagnostics)
        return AmphibiousControlOutput(
            mode=BountyState.SUBMERGED,
            ballast_pump_engagement=ballast_pump_engagement,
            diagnostics=diagnostics,
        )

    async def _execute_emergency_surface_blow(self) -> None:
        """Fires the physical compressed-air ballast-blow relay. Mirrors
        `peripheral_io.py`'s cryo-discharge safety pattern: the relay is
        always switched off in a `finally` block, so a mid-blow exception
        or cancellation can never leave it energized."""
        logger.critical(
            "[AMPHIBIOUS CONTROL CORE]: Firing emergency ballast-blow relay "
            "(pin=%d) for %.1fs.", self._emergency_blow_relay_pin, _EMERGENCY_BLOW_DURATION_S,
        )
        try:
            self._hal.digital_write(self._emergency_blow_relay_pin, True)
            await asyncio.sleep(_EMERGENCY_BLOW_DURATION_S)
        finally:
            try:
                self._hal.digital_write(self._emergency_blow_relay_pin, False)
            except Exception:  # noqa: BLE001 - shutoff must not raise past this point
                logger.exception(
                    "[AMPHIBIOUS CONTROL CORE]: Failed to command ballast-blow relay "
                    "off; treat the ballast system as unsafe."
                )

    # --- SURFACE: station-keeping -------------------------------------------

    def _process_surface(self, telemetry: AmphibiousTelemetry) -> AmphibiousControlOutput:
        diagnostics = (
            f"Surface station-keeping: pressure={telemetry.external_fluid_pressure_psi:.2f}psi, "
            f"velocity={telemetry.forward_velocity_knot:.2f}kn."
        )
        logger.info("[AMPHIBIOUS CONTROL CORE]: %s", diagnostics)
        return AmphibiousControlOutput(
            mode=BountyState.SURFACE,
            ballast_pump_engagement=0.0,
            diagnostics=diagnostics,
        )


# --------------------------------------------------------------------------
# Reference ZaneSensorInput/ZaneControlOutput implementations — the same
# pattern as control.py's InMemoryVehicleSensor/InMemoryVehicleControlSink,
# registering this vehicle's telemetry shape into the existing generic seam.
# --------------------------------------------------------------------------


class InMemoryAmphibiousSensor(ZaneSensorInput[AmphibiousTelemetry]):
    def __init__(self, maxsize: int = 32) -> None:
        self._queue: "asyncio.Queue[AmphibiousTelemetry]" = asyncio.Queue(maxsize=maxsize)
        self._subscribers: List[Callable[[AmphibiousTelemetry], Awaitable[None]]] = []

    async def push_frame(self, frame: AmphibiousTelemetry) -> None:
        await self._queue.put(frame)
        for callback in list(self._subscribers):
            await callback(frame)

    async def read(self) -> AmphibiousTelemetry:
        return await self._queue.get()

    async def subscribe(
        self, callback: Callable[[AmphibiousTelemetry], Awaitable[None]]
    ) -> None:
        self._subscribers.append(callback)


class InMemoryAmphibiousControlSink(ZaneControlOutput[AmphibiousControlOutput]):
    def __init__(self) -> None:
        self.sent: List[AmphibiousControlOutput] = []
        self.emergency_stopped = False

    async def send(self, command: AmphibiousControlOutput) -> None:
        self.sent.append(command)

    async def emergency_stop(self) -> None:
        self.emergency_stopped = True
        self.sent.append(
            AmphibiousControlOutput(
                mode=BountyState.SURFACE,
                ballast_pump_engagement=-1.0,
                emergency_surface_override=True,
                diagnostics="EMERGENCY STOP",
            )
        )
