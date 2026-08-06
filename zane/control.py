"""Driving-simulation / vehicle control integration for Zane's digital mind.

`ZaneSensorInput`/`ZaneControlOutput` started as unimplemented, generic
scaffolding (see git history) — this module is their first concrete
consumer: a Gymnasium/OpenAI-Gym-style telemetry bridge. The abstract seam
itself is now `Generic` so it can type either the original simple
`SensorReading`/`ControlCommand` pair or the richer `VehicleTelemetryFrame`/
`VehicleControlOutput` pair below, without breaking the original contract
or requiring `ZaneMind` itself to change.

Design note on the C++ engine: `ZaneAnalytics` has no pathfinding methods,
and none were invented for this. The deterministic obstacle-avoidance
vector math (`compute_avoidance_vector`) is honest, dependency-free pure
Python (an artificial-potential-field, a standard robotics technique).
What genuinely does route through the bridged `zane_cpp` engine (or its
pure-Python fallback) is the maneuver's confidence/risk score, via
`AnalyticsEngine.calculate_success_probability` — the same real
computation Zane already uses for mission-style risk assessment, reused
here with a documented field mapping (see `build_probability_factors`).
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from zane.analytics_bridge import AnalyticsEngine, ProbabilityFactors

logger = logging.getLogger("zane.control")


# --------------------------------------------------------------------------
# Original generic seam (unchanged in shape, now parameterized so concrete
# integrations — like the vehicle telemetry one below — can plug in their
# own richer reading/command types).
# --------------------------------------------------------------------------


@dataclass
class SensorReading:
    """A single timestamped reading from a generic sensor source (e.g.
    speed, heading, proximity, lane position)."""

    name: str
    value: float
    unit: str
    timestamp_ns: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlCommand:
    """A single generic control command (e.g. throttle, steering angle,
    brake)."""

    channel: str
    value: float
    timestamp_ns: int
    metadata: Dict[str, Any] = field(default_factory=dict)


SensorCallback = Callable[[SensorReading], Awaitable[None]]

TReading = TypeVar("TReading")
TCommand = TypeVar("TCommand")


class ZaneSensorInput(ABC, Generic[TReading]):
    """Abstract seam for feeding real-time sensor data into Zane's digital
    mind. Generic over the reading type so a concrete integration (like
    `InMemoryVehicleSensor` below) can carry richer, domain-specific data
    than the generic `SensorReading` while still satisfying this contract.
    """

    @abstractmethod
    async def read(self) -> TReading:
        """Returns the next available sensor reading. Implementations
        decide their own polling/streaming strategy."""

    @abstractmethod
    async def subscribe(self, callback: Callable[[TReading], Awaitable[None]]) -> None:
        """Registers a callback to be invoked with each new reading as it
        arrives, for push-based sensor sources."""


class ZaneControlOutput(ABC, Generic[TCommand]):
    """Abstract seam for Zane's digital mind to issue control commands to
    an actuated system. Generic over the command type for the same reason
    as `ZaneSensorInput`."""

    @abstractmethod
    async def send(self, command: TCommand) -> None:
        """Dispatches a single control command to the underlying system."""

    @abstractmethod
    async def emergency_stop(self) -> None:
        """Immediately halts all actuation. Required on every
        implementation so ZaneMind always has a safe fallback available,
        regardless of which concrete driving/control backend is plugged
        in."""


# --------------------------------------------------------------------------
# Vehicle telemetry: the concrete driving-sim data shapes.
# --------------------------------------------------------------------------


@dataclass
class VehicleTelemetryFrame:
    """One telemetry sample from a driving simulator, in a plain
    Gymnasium/OpenAI-Gym-style observation shape (dict of arrays) once
    parsed. Axis convention: X=forward, Y=left(+)/right(-), Z=up — applies
    to `vehicle_coordinates`, `velocity`, and every vector in
    `obstacle_proximity_vectors` (each of which is the displacement *from
    the vehicle to that obstacle*, in the vehicle's local frame)."""

    vehicle_coordinates: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    obstacle_proximity_vectors: List[Tuple[float, float, float]]
    surface_friction_coefficient: float
    timestamp_ns: int


@dataclass
class VehicleControlOutput:
    """A single tick's control output. Values are clamped to their valid
    ranges on construction — a physically invalid command (e.g. steering
    of 1.7) is a bug worth clamping-and-logging, not passing through to
    whatever is driving the vehicle."""

    steering: float  # -1.0 (full left) .. 1.0 (full right)
    throttle: float  # 0.0 .. 1.0
    braking: float  # 0.0 .. 1.0
    timestamp_ns: int
    confidence_percent: float = 0.0
    diagnostics: str = ""

    def __post_init__(self) -> None:
        clamped_steering = max(-1.0, min(1.0, self.steering))
        clamped_throttle = max(0.0, min(1.0, self.throttle))
        clamped_braking = max(0.0, min(1.0, self.braking))
        if (clamped_steering, clamped_throttle, clamped_braking) != (
            self.steering,
            self.throttle,
            self.braking,
        ):
            logger.warning(
                "VehicleControlOutput received out-of-range values "
                "(steering=%.3f throttle=%.3f braking=%.3f); clamped to "
                "(%.3f, %.3f, %.3f).",
                self.steering, self.throttle, self.braking,
                clamped_steering, clamped_throttle, clamped_braking,
            )
        self.steering = clamped_steering
        self.throttle = clamped_throttle
        self.braking = clamped_braking


class TelemetryParseError(ValueError):
    """Raised when a raw telemetry dict is missing keys, has the wrong
    shape, or contains non-numeric/out-of-range data."""


_REQUIRED_TELEMETRY_KEYS = (
    "vehicle_coordinates",
    "velocity",
    "obstacle_proximity_vectors",
    "surface_friction_coefficient",
)


def parse_telemetry_frame(
    raw: Dict[str, Any], timestamp_ns: Optional[int] = None
) -> VehicleTelemetryFrame:
    """Parses a raw Gymnasium-style observation dict into a
    `VehicleTelemetryFrame`, raising `TelemetryParseError` with a specific
    reason on any malformed input rather than letting a `KeyError`/
    `TypeError` propagate from deep inside the avoidance math later."""
    missing = [key for key in _REQUIRED_TELEMETRY_KEYS if key not in raw]
    if missing:
        raise TelemetryParseError(f"Telemetry frame missing required keys: {missing}")

    try:
        coordinates = tuple(float(v) for v in raw["vehicle_coordinates"])
        velocity = tuple(float(v) for v in raw["velocity"])
        obstacles = [tuple(float(v) for v in vec) for vec in raw["obstacle_proximity_vectors"]]
        friction = float(raw["surface_friction_coefficient"])
    except (TypeError, ValueError) as exc:
        raise TelemetryParseError(f"Telemetry frame contains non-numeric data: {exc}") from exc

    if len(coordinates) != 3:
        raise TelemetryParseError(
            f"vehicle_coordinates must have exactly 3 components, got {len(coordinates)}"
        )
    if len(velocity) != 3:
        raise TelemetryParseError(f"velocity must have exactly 3 components, got {len(velocity)}")
    if any(len(vec) != 3 for vec in obstacles):
        raise TelemetryParseError("every obstacle_proximity_vectors entry must have 3 components")
    if not (0.0 <= friction <= 2.0):
        raise TelemetryParseError(
            f"surface_friction_coefficient out of plausible range [0, 2]: {friction}"
        )

    return VehicleTelemetryFrame(
        vehicle_coordinates=(coordinates[0], coordinates[1], coordinates[2]),
        velocity=(velocity[0], velocity[1], velocity[2]),
        obstacle_proximity_vectors=[(v[0], v[1], v[2]) for v in obstacles],
        surface_friction_coefficient=friction,
        timestamp_ns=timestamp_ns if timestamp_ns is not None else time.time_ns(),
    )


# --------------------------------------------------------------------------
# Deterministic pathfinding / obstacle-avoidance math (pure Python).
# --------------------------------------------------------------------------

_REPULSION_GAIN = 8.0
_REPULSION_EPSILON = 0.25  # keeps repulsion finite as distance -> 0
_FORWARD_BIAS = 1.0  # constant "keep going straight" attractive component
_MAX_SAFE_SPEED_MPS = 40.0  # ~144 km/h; arbitrary but documented ceiling
_STEERING_NORMALIZER = 6.0
_BASE_THROTTLE = 0.6


def _vector_length(vec: Tuple[float, float, float]) -> float:
    x, y, z = vec
    return math.sqrt(x * x + y * y + z * z)


def _closest_obstacle_distance(frame: VehicleTelemetryFrame) -> Optional[float]:
    if not frame.obstacle_proximity_vectors:
        return None
    return min(_vector_length(vec) for vec in frame.obstacle_proximity_vectors)


def compute_avoidance_vector(frame: VehicleTelemetryFrame) -> Tuple[float, float, float]:
    """Artificial-potential-field obstacle avoidance: sums a repulsive
    force from every reported obstacle (inversely proportional to squared
    distance, pointing away from it) with a constant forward attractive
    bias. This telemetry schema has no destination coordinate, so "goal"
    defaults to continuing straight ahead unless deflected — a standard
    simplification of the technique when no explicit waypoint is given.
    """
    fx, fy, fz = _FORWARD_BIAS, 0.0, 0.0
    for ox, oy, oz in frame.obstacle_proximity_vectors:
        distance = _vector_length((ox, oy, oz))
        if distance < 1e-6:
            # Reported at zero distance: treat as maximal repulsion
            # straight backward rather than dividing by zero.
            fx -= _REPULSION_GAIN
            continue
        magnitude = _REPULSION_GAIN / (distance * distance + _REPULSION_EPSILON)
        fx += -(ox / distance) * magnitude
        fy += -(oy / distance) * magnitude
        fz += -(oz / distance) * magnitude
    return (fx, fy, fz)


def build_probability_factors(
    frame: VehicleTelemetryFrame, recent_outcomes: Sequence[bool] = ()
) -> ProbabilityFactors:
    """Maps a telemetry frame onto `ProbabilityFactors` so the C++-bridged
    `ZaneAnalytics` engine can score this maneuver with the same weighted-
    heuristic math it uses for mission scenarios. `ProbabilityFactors`'
    field names were designed for Ninja missions, not vehicle dynamics, so
    this is a deliberate, documented reuse of the math rather than a
    literal semantic match:

    - danger_level        <- closest obstacle distance (closer = higher)
    - team_synergy         <- "sensor confidence": more tracked obstacle
                               vectors implies a richer picture of the
                               surroundings
    - resource_availability <- current speed headroom below
                               `_MAX_SAFE_SPEED_MPS`
    - historical_success_rate <- rolling record of recent ticks that
                               stayed above the confidence floor, or a
                               conservative 75% prior with no history yet
    - complexity_index     <- how many obstacles are being reasoned about
                               simultaneously
    """
    closest = _closest_obstacle_distance(frame)
    if closest is None:
        danger_level = 5.0
    else:
        danger_level = max(0.0, min(100.0, 100.0 * (1.0 - min(closest, 10.0) / 10.0)))

    team_synergy = max(0.0, min(100.0, 20.0 + 15.0 * len(frame.obstacle_proximity_vectors)))

    speed = _vector_length(frame.velocity)
    resource_availability = max(
        0.0, min(100.0, 100.0 * (1.0 - min(speed, _MAX_SAFE_SPEED_MPS) / _MAX_SAFE_SPEED_MPS))
    )

    if recent_outcomes:
        historical_success_rate = 100.0 * (
            sum(1 for ok in recent_outcomes if ok) / len(recent_outcomes)
        )
    else:
        historical_success_rate = 75.0

    complexity_index = max(0.0, min(100.0, 12.0 * len(frame.obstacle_proximity_vectors)))

    return ProbabilityFactors(
        danger_level=danger_level,
        team_synergy=team_synergy,
        resource_availability=resource_availability,
        historical_success_rate=historical_success_rate,
        complexity_index=complexity_index,
    )


# --------------------------------------------------------------------------
# Reference implementations (also what the test suite exercises end-to-end
# without needing an actual simulator running).
# --------------------------------------------------------------------------


class InMemoryVehicleSensor(ZaneSensorInput[VehicleTelemetryFrame]):
    """`asyncio.Queue`-backed `ZaneSensorInput`. A real Gymnasium/OpenAI-
    Gym sim integration would call `push_raw`/`push_frame` from its own
    `step()` loop; this class is the reference implementation the rest of
    this module (and its tests) run against."""

    def __init__(self, maxsize: int = 32) -> None:
        self._queue: "asyncio.Queue[VehicleTelemetryFrame]" = asyncio.Queue(maxsize=maxsize)
        self._subscribers: List[Callable[[VehicleTelemetryFrame], Awaitable[None]]] = []

    async def push_raw(self, raw: Dict[str, Any]) -> None:
        await self.push_frame(parse_telemetry_frame(raw))

    async def push_frame(self, frame: VehicleTelemetryFrame) -> None:
        await self._queue.put(frame)
        for callback in list(self._subscribers):
            await callback(frame)

    async def read(self) -> VehicleTelemetryFrame:
        return await self._queue.get()

    async def subscribe(
        self, callback: Callable[[VehicleTelemetryFrame], Awaitable[None]]
    ) -> None:
        self._subscribers.append(callback)


class InMemoryVehicleControlSink(ZaneControlOutput[VehicleControlOutput]):
    """Records every command sent, for tests/inspection. A real sim
    integration would translate `VehicleControlOutput` into whatever the
    sim's own control API expects."""

    def __init__(self) -> None:
        self.sent: List[VehicleControlOutput] = []
        self.emergency_stopped = False

    async def send(self, command: VehicleControlOutput) -> None:
        self.sent.append(command)

    async def emergency_stop(self) -> None:
        self.emergency_stopped = True
        self.sent.append(
            VehicleControlOutput(
                steering=0.0,
                throttle=0.0,
                braking=1.0,
                timestamp_ns=time.time_ns(),
                confidence_percent=0.0,
                diagnostics="EMERGENCY STOP",
            )
        )


# --------------------------------------------------------------------------
# TelemetryLoop: ties sensor -> avoidance math + ZaneAnalytics -> control.
# --------------------------------------------------------------------------


class TelemetryLoop:
    """Runs the sense -> decide -> act cycle at a configurable tick rate.
    Any obstacle inside `emergency_proximity_m` triggers an immediate
    `emergency_stop()` bypassing the normal decision math entirely. Repeated
    tick failures (`max_consecutive_errors`) also trigger an emergency stop
    and end the loop, rather than spinning forever on a broken sensor."""

    def __init__(
        self,
        sensor: ZaneSensorInput[VehicleTelemetryFrame],
        control: ZaneControlOutput[VehicleControlOutput],
        analytics: AnalyticsEngine,
        *,
        tick_hz: float = 20.0,
        min_confidence_percent: float = 25.0,
        emergency_proximity_m: float = 0.75,
        max_consecutive_errors: int = 5,
        history_len: int = 50,
    ) -> None:
        self.sensor = sensor
        self.control = control
        self.analytics = analytics
        self.tick_interval_s = 1.0 / tick_hz
        self.min_confidence_percent = min_confidence_percent
        self.emergency_proximity_m = emergency_proximity_m
        self.max_consecutive_errors = max_consecutive_errors
        self._recent_outcomes: Deque[bool] = deque(maxlen=history_len)
        self._consecutive_errors = 0
        self._running = False

    async def tick(self) -> VehicleControlOutput:
        """Runs exactly one sense -> decide cycle and returns the control
        output, without sending it. `run_forever` calls this in a loop and
        sends the result; exposed separately so callers/tests can drive
        single ticks deterministically."""
        frame = await self.sensor.read()

        closest = _closest_obstacle_distance(frame)
        if closest is not None and closest <= self.emergency_proximity_m:
            await self.control.emergency_stop()
            self._recent_outcomes.append(False)
            return VehicleControlOutput(
                steering=0.0,
                throttle=0.0,
                braking=1.0,
                timestamp_ns=frame.timestamp_ns,
                confidence_percent=0.0,
                diagnostics=(
                    f"Emergency stop: obstacle at {closest:.2f}m is within the "
                    f"{self.emergency_proximity_m:.2f}m safety envelope."
                ),
            )

        avoidance = compute_avoidance_vector(frame)
        factors = build_probability_factors(frame, self._recent_outcomes)
        result = self.analytics.calculate_success_probability(factors)

        steering = max(-1.0, min(1.0, avoidance[1] / _STEERING_NORMALIZER))
        confidence_scale = max(0.0, min(1.0, result.success_probability / 100.0))
        friction_scale = max(0.0, min(1.0, frame.surface_friction_coefficient))

        throttle = _BASE_THROTTLE * confidence_scale * friction_scale
        braking = 0.0
        if result.success_probability < self.min_confidence_percent:
            braking = 1.0 - confidence_scale
            throttle = 0.0

        outcome_ok = result.success_probability >= self.min_confidence_percent
        self._recent_outcomes.append(outcome_ok)

        return VehicleControlOutput(
            steering=steering,
            throttle=throttle,
            braking=braking,
            timestamp_ns=frame.timestamp_ns,
            confidence_percent=result.success_probability,
            diagnostics=(
                f"Maneuver confidence {result.success_probability:.2f}% "
                f"(risk index {result.risk_index:.2f}%); {result.analytical_summary}"
            ),
        )

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            try:
                output = await self.tick()
                await self.control.send(output)
                self._consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad tick must not kill the loop silently
                self._consecutive_errors += 1
                logger.error(
                    "TelemetryLoop tick failed (%d/%d consecutive): %s",
                    self._consecutive_errors, self.max_consecutive_errors, exc,
                )
                if self._consecutive_errors >= self.max_consecutive_errors:
                    logger.critical(
                        "TelemetryLoop exceeded max consecutive errors; issuing emergency stop."
                    )
                    await self.control.emergency_stop()
                    self._running = False
                    break
            await asyncio.sleep(self.tick_interval_s)

    def stop(self) -> None:
        self._running = False
