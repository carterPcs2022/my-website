"""Groundwork for a future driving-simulation / control integration.

These are intentionally unimplemented, typed interfaces — a clean seam so
a future driving-sim or other actuated-system module can plug into
`ZaneMind` without requiring any changes to the core orchestrator later.
No sensor or control logic exists yet; nothing here is wired into
`zane/core.py`. See the "Future integration seam" section of README.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict


@dataclass
class SensorReading:
    """A single timestamped reading from a future sensor source (e.g.
    speed, heading, proximity, lane position). Deliberately generic —
    shape follows whatever a concrete driving-sim module defines."""

    name: str
    value: float
    unit: str
    timestamp_ns: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlCommand:
    """A single control command a future driving/control module would
    emit (e.g. throttle, steering angle, brake). Deliberately generic."""

    channel: str
    value: float
    timestamp_ns: int
    metadata: Dict[str, Any] = field(default_factory=dict)


SensorCallback = Callable[[SensorReading], Awaitable[None]]


class ZaneSensorInput(ABC):
    """Abstract seam for feeding real-time sensor data into Zane's digital
    mind. A future driving-simulation integration implements this to
    stream readings (e.g. from a sim's telemetry bus) into ZaneMind's
    analytical core (see `zane/analytics_bridge.py`'s `ZaneAnalytics`,
    which already accepts arbitrary named state via
    `update_internal_state`) without touching `ZaneMind` itself.

    Not implemented anywhere yet — no driving-sim exists in this codebase.
    """

    @abstractmethod
    async def read(self) -> SensorReading:
        """Returns the next available sensor reading. Implementations
        decide their own polling/streaming strategy."""

    @abstractmethod
    async def subscribe(self, callback: SensorCallback) -> None:
        """Registers a callback to be invoked with each new SensorReading
        as it arrives, for push-based sensor sources."""


class ZaneControlOutput(ABC):
    """Abstract seam for Zane's digital mind to issue control commands to
    a future driving-simulation (or any other actuated system). A future
    module implements this to translate a `ControlCommand` into whatever
    the sim's own control API expects.

    Not implemented anywhere yet — no driving-sim exists in this codebase.
    """

    @abstractmethod
    async def send(self, command: ControlCommand) -> None:
        """Dispatches a single control command to the underlying system."""

    @abstractmethod
    async def emergency_stop(self) -> None:
        """Immediately halts all actuation. Required on every
        implementation so ZaneMind always has a safe fallback available,
        regardless of which concrete driving/control backend is plugged
        in later."""
