import time

from zane.analytics_bridge import AnalyticsEngine, DataStreamFrame, ProbabilityFactors


def test_calculate_success_probability_bounds():
    engine = AnalyticsEngine(worker_threads=1, seed=42)
    factors = ProbabilityFactors(
        danger_level=80,
        team_synergy=90,
        resource_availability=70,
        historical_success_rate=85,
        complexity_index=60,
    )
    result = engine.calculate_success_probability(factors)

    assert 0.0 <= result.success_probability <= 100.0
    assert 0.0 <= result.risk_index <= 100.0
    assert result.confidence_interval_low <= result.confidence_interval_high
    assert "nindroid heuristic model" in result.analytical_summary
    engine.shutdown()


def test_deterministic_with_fixed_seed():
    engine_a = AnalyticsEngine(worker_threads=1, seed=1234)
    engine_b = AnalyticsEngine(worker_threads=1, seed=1234)
    factors = ProbabilityFactors(
        danger_level=50,
        team_synergy=50,
        resource_availability=50,
        historical_success_rate=50,
        complexity_index=50,
    )
    result_a = engine_a.calculate_success_probability(factors)
    result_b = engine_b.calculate_success_probability(factors)

    if not engine_a.is_native:
        # The pure-Python fallback uses Python's random.Random directly, so a
        # fixed seed reproduces bit-identical output.
        assert result_a.success_probability == result_b.success_probability

    engine_a.shutdown()
    engine_b.shutdown()


def test_submit_and_get_result_blocking():
    engine = AnalyticsEngine(worker_threads=2, seed=7)
    factors = ProbabilityFactors(
        danger_level=30,
        team_synergy=95,
        resource_availability=90,
        historical_success_rate=88,
        complexity_index=20,
    )
    computation_id = engine.submit_calculation(factors)
    result = engine.get_result_blocking(computation_id, timeout_ms=2000)
    assert result.computation_id == computation_id
    assert 0.0 <= result.success_probability <= 100.0
    engine.shutdown()


def test_format_data_stream():
    engine = AnalyticsEngine(worker_threads=1, seed=1)
    frame = DataStreamFrame(label="TEST_METRIC", values=[1.0, 2.0, 3.0, 4.0], timestamp_ns=123)
    formatted = engine.format_data_stream(frame)
    assert "TEST_METRIC" in formatted
    assert "n=4" in formatted
    engine.shutdown()


def test_internal_state_roundtrip():
    engine = AnalyticsEngine(worker_threads=1, seed=1)
    engine.update_internal_state("alert_level", 42.0)
    snapshot = engine.get_internal_state_snapshot()
    assert snapshot["alert_level"] == 42.0
    engine.shutdown()
