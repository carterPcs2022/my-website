"""Bridge to the C++ ZaneAnalytics engine, with a pure-Python fallback.

The real engine lives in cpp/ and is compiled into the `zane_cpp` extension
module (see setup.py / CMakeLists.txt). Requiring every consumer of this
package to have a working C++ toolchain just to run the CLI would be poor
engineering, so this module transparently falls back to a pure-Python
implementation with an identical interface when `zane_cpp` isn't importable.
Both implementations are thread-safe and expose the same method signatures,
so callers (see zane/core.py) never need to know which one is active.
"""
from __future__ import annotations

import logging
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("zane.analytics_bridge")

try:
    import zane_cpp  # type: ignore

    _NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without the build
    zane_cpp = None  # type: ignore
    _NATIVE_AVAILABLE = False


@dataclass
class ProbabilityFactors:
    danger_level: float = 0.0
    team_synergy: float = 0.0
    resource_availability: float = 0.0
    historical_success_rate: float = 0.0
    complexity_index: float = 0.0


@dataclass
class ProbabilityResult:
    computation_id: int
    success_probability: float
    risk_index: float
    confidence_interval_low: float
    confidence_interval_high: float
    analytical_summary: str


@dataclass
class DataStreamFrame:
    label: str
    values: List[float] = field(default_factory=list)
    timestamp_ns: int = 0


class _PurePythonZaneAnalytics:
    """Drop-in, thread-safe reimplementation of zane::ZaneAnalytics.

    Mirrors the weighting/jitter model in cpp/src/analytics.cpp exactly so
    Zane's quoted percentages behave the same whether or not the native
    extension was built for this platform.
    """

    def __init__(self, worker_threads: int = 2, seed: int = 0) -> None:
        self._state_lock = threading.Lock()
        self._internal_state: Dict[str, float] = {
            "alert_level": 0.0,
            "mission_clock_s": 0.0,
            "nindroid_core_temp_c": 36.6,
        }
        self._rng_lock = threading.Lock()
        self._rng = random.Random(seed if seed != 0 else None)
        self._results_lock = threading.Lock()
        self._results: Dict[int, ProbabilityResult] = {}
        self._next_id_lock = threading.Lock()
        self._next_id = 1
        self._worker_threads = max(1, worker_threads)

    def _alloc_id(self) -> int:
        with self._next_id_lock:
            new_id = self._next_id
            self._next_id += 1
            return new_id

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(100.0, v))

    def _compute(self, factors: ProbabilityFactors, computation_id: int) -> ProbabilityResult:
        danger = self._clamp(factors.danger_level)
        synergy = self._clamp(factors.team_synergy)
        resources = self._clamp(factors.resource_availability)
        history = self._clamp(factors.historical_success_rate)
        complexity = self._clamp(factors.complexity_index)

        base = (
            synergy * 0.28
            + resources * 0.22
            + history * 0.30
            + (100.0 - danger) * 0.12
            + (100.0 - complexity) * 0.08
        )

        sigma = 1.5 + (complexity / 100.0) * 4.0
        with self._rng_lock:
            jitter = self._rng.gauss(0.0, sigma)

        success = self._clamp(base + jitter)
        risk = self._clamp(100.0 - success + (danger * 0.05))
        half_width = 1.645 * sigma
        ci_low = self._clamp(success - half_width)
        ci_high = self._clamp(success + half_width)

        summary = (
            f"Weighted nindroid heuristic model (team_synergy={synergy:.2f}, "
            f"resources={resources:.2f}, history={history:.2f}, danger={danger:.2f}, "
            f"complexity={complexity:.2f}) with stochastic variance correction "
            f"(sigma={sigma:.2f})."
        )

        return ProbabilityResult(
            computation_id=computation_id,
            success_probability=success,
            risk_index=risk,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            analytical_summary=summary,
        )

    def calculate_success_probability(self, factors: ProbabilityFactors) -> ProbabilityResult:
        return self._compute(factors, self._alloc_id())

    def submit_calculation(self, factors: ProbabilityFactors) -> int:
        computation_id = self._alloc_id()

        def _run() -> None:
            result = self._compute(factors, computation_id)
            with self._results_lock:
                self._results[computation_id] = result

        threading.Thread(target=_run, daemon=True, name=f"zane-analytics-{computation_id}").start()
        return computation_id

    def try_get_result(self, computation_id: int) -> Optional[ProbabilityResult]:
        with self._results_lock:
            return self._results.pop(computation_id, None)

    def get_result_blocking(self, computation_id: int, timeout_ms: int = 5000) -> ProbabilityResult:
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            result = self.try_get_result(computation_id)
            if result is not None:
                return result
            time.sleep(0.01)
        raise TimeoutError(f"ZaneAnalytics: computation {computation_id} timed out")

    def format_data_stream(self, frame: DataStreamFrame) -> str:
        values = frame.values
        n = len(values)
        mean = statistics.fmean(values) if n else 0.0
        min_v = min(values) if n else 0.0
        max_v = max(values) if n else 0.0
        stddev = statistics.pstdev(values) if n > 0 else 0.0
        ts = frame.timestamp_ns or time.time_ns()
        return (
            f"[ZANE::DATASTREAM] label={frame.label} n={n} mean={mean:.3f} "
            f"min={min_v:.3f} max={max_v:.3f} stddev={stddev:.3f} ts_ns={ts}"
        )

    def get_internal_state_snapshot(self) -> Dict[str, float]:
        with self._state_lock:
            return dict(self._internal_state)

    def update_internal_state(self, key: str, value: float) -> None:
        with self._state_lock:
            self._internal_state[key] = value

    def shutdown(self) -> None:
        # Daemon threads for submit_calculation exit with the process; no
        # persistent worker pool to join in the pure-Python fallback.
        return None


class AnalyticsEngine:
    """Thin, uniform facade over the native or pure-Python analytics engine."""

    def __init__(self, worker_threads: int = 2, seed: int = 0) -> None:
        self.is_native = _NATIVE_AVAILABLE
        if self.is_native:
            logger.info("ZaneAnalytics: using compiled zane_cpp extension.")
            self._engine = zane_cpp.ZaneAnalytics(worker_threads, seed)  # type: ignore[union-attr]
        else:
            logger.warning(
                "ZaneAnalytics: zane_cpp extension not found, falling back to the "
                "pure-Python engine. Build it with `pip install -e .` for the "
                "native multi-threaded implementation."
            )
            self._engine = _PurePythonZaneAnalytics(worker_threads, seed)

    def _to_native_factors(self, factors: ProbabilityFactors):
        if not self.is_native:
            return factors
        return zane_cpp.ProbabilityFactors(  # type: ignore[union-attr]
            danger_level=factors.danger_level,
            team_synergy=factors.team_synergy,
            resource_availability=factors.resource_availability,
            historical_success_rate=factors.historical_success_rate,
            complexity_index=factors.complexity_index,
        )

    @staticmethod
    def _from_native_result(result) -> ProbabilityResult:
        return ProbabilityResult(
            computation_id=result.computation_id,
            success_probability=result.success_probability,
            risk_index=result.risk_index,
            confidence_interval_low=result.confidence_interval_low,
            confidence_interval_high=result.confidence_interval_high,
            analytical_summary=result.analytical_summary,
        )

    def calculate_success_probability(self, factors: ProbabilityFactors) -> ProbabilityResult:
        native_factors = self._to_native_factors(factors)
        result = self._engine.calculate_success_probability(native_factors)
        return result if not self.is_native else self._from_native_result(result)

    def submit_calculation(self, factors: ProbabilityFactors) -> int:
        return self._engine.submit_calculation(self._to_native_factors(factors))

    def try_get_result(self, computation_id: int) -> Optional[ProbabilityResult]:
        result = self._engine.try_get_result(computation_id)
        if result is None:
            return None
        return result if not self.is_native else self._from_native_result(result)

    def get_result_blocking(self, computation_id: int, timeout_ms: int = 5000) -> ProbabilityResult:
        result = self._engine.get_result_blocking(computation_id, timeout_ms)
        return result if not self.is_native else self._from_native_result(result)

    def format_data_stream(self, frame: DataStreamFrame) -> str:
        if self.is_native:
            native_frame = zane_cpp.DataStreamFrame(  # type: ignore[union-attr]
                label=frame.label, values=frame.values, timestamp_ns=frame.timestamp_ns
            )
            return self._engine.format_data_stream(native_frame)
        return self._engine.format_data_stream(frame)

    def get_internal_state_snapshot(self) -> Dict[str, float]:
        return dict(self._engine.get_internal_state_snapshot())

    def update_internal_state(self, key: str, value: float) -> None:
        self._engine.update_internal_state(key, value)

    def shutdown(self) -> None:
        self._engine.shutdown()
