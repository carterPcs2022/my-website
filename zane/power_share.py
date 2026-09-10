"""Protected software coordinator for future Zane/P.I.X.A.L. power sharing.

This is intentionally simulation/control-policy only. It never switches a
real power rail, charger, MOSFET, relay, or actuator. Physical power transfer
must be implemented behind dedicated battery-management hardware with
hardware-enforced limits and a manual disconnect.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PowerShareDecision(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    STOP = "stop"


@dataclass(frozen=True)
class PowerTelemetry:
    donor_percent: float
    recipient_percent: float
    donor_temperature_c: float
    recipient_temperature_c: float
    donor_healthy: bool = True
    recipient_healthy: bool = True


@dataclass(frozen=True)
class PowerSharePolicy:
    donor_min_percent: float = 30.0
    recipient_request_percent: float = 15.0
    max_temperature_c: float = 45.0
    target_recipient_percent: float = 35.0


@dataclass(frozen=True)
class PowerShareResult:
    decision: PowerShareDecision
    reason: str


class ProtectedPowerShareCoordinator:
    """Decide whether a future hardware layer may consider power sharing."""

    def __init__(self, policy: PowerSharePolicy | None = None) -> None:
        self.policy = policy or PowerSharePolicy()

    def evaluate_request(self, telemetry: PowerTelemetry) -> PowerShareResult:
        p = self.policy
        if not telemetry.donor_healthy or not telemetry.recipient_healthy:
            return PowerShareResult(PowerShareDecision.REJECT, "A battery reports unhealthy status.")
        if telemetry.donor_temperature_c > p.max_temperature_c or telemetry.recipient_temperature_c > p.max_temperature_c:
            return PowerShareResult(PowerShareDecision.REJECT, "Battery temperature is outside the sharing envelope.")
        if telemetry.donor_percent < p.donor_min_percent:
            return PowerShareResult(PowerShareDecision.REJECT, "Donor reserve is below the protected minimum.")
        if telemetry.recipient_percent >= p.target_recipient_percent:
            return PowerShareResult(PowerShareDecision.REJECT, "Recipient does not currently need assistance.")
        return PowerShareResult(PowerShareDecision.ALLOW, "Protected power-sharing conditions are satisfied.")

    def should_stop(self, telemetry: PowerTelemetry) -> PowerShareResult:
        p = self.policy
        if not telemetry.donor_healthy or not telemetry.recipient_healthy:
            return PowerShareResult(PowerShareDecision.STOP, "Battery health changed; stop sharing.")
        if telemetry.donor_temperature_c > p.max_temperature_c or telemetry.recipient_temperature_c > p.max_temperature_c:
            return PowerShareResult(PowerShareDecision.STOP, "Temperature limit reached; stop sharing.")
        if telemetry.donor_percent <= p.donor_min_percent or telemetry.recipient_percent >= p.target_recipient_percent:
            return PowerShareResult(PowerShareDecision.STOP, "A protected power-sharing boundary was reached.")
        return PowerShareResult(PowerShareDecision.ALLOW, "Continue monitoring.")
