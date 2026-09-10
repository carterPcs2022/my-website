from zane.power_share import (
    PowerShareDecision,
    PowerSharePolicy,
    PowerTelemetry,
    ProtectedPowerShareCoordinator,
)


def telemetry(donor=80.0, recipient=10.0, donor_temp=30.0, recipient_temp=30.0, healthy=True):
    return PowerTelemetry(donor, recipient, donor_temp, recipient_temp, healthy, healthy)


def test_allows_healthy_low_recipient() -> None:
    coordinator = ProtectedPowerShareCoordinator()
    result = coordinator.evaluate_request(telemetry())
    assert result.decision is PowerShareDecision.ALLOW


def test_rejects_low_donor_reserve() -> None:
    coordinator = ProtectedPowerShareCoordinator(PowerSharePolicy(donor_min_percent=30.0))
    result = coordinator.evaluate_request(telemetry(donor=20.0))
    assert result.decision is PowerShareDecision.REJECT


def test_rejects_high_temperature() -> None:
    result = ProtectedPowerShareCoordinator().evaluate_request(telemetry(donor_temp=50.0))
    assert result.decision is PowerShareDecision.REJECT


def test_stops_when_boundary_is_reached() -> None:
    coordinator = ProtectedPowerShareCoordinator()
    result = coordinator.should_stop(telemetry(donor=30.0))
    assert result.decision is PowerShareDecision.STOP
