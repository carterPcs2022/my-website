"""Falcon Scout: an async background system-health daemon, named after
Zane's Falcon.

Every `interval_s` (default 60s) it profiles the running process — API
request latency, SQLite responsiveness, and any exceptions logged since
the last cycle — and, only when something is actually wrong, writes a
technical summary directly into the shared RAG memory pipeline (SQLite +
FAISS) tagged `role="falcon_scout_telemetry"`, so Zane can recall his own
operational history the same way he recalls anything else a user told
him. (See the "Memory" section of README.md: semantic retrieval already
searches across every session, not just the current one, so no special
retrieval-path change was needed for these entries to surface.)

Deliberately does not call the LLM to produce the summary: Falcon Scout
needs to keep working even if Groq itself is the thing having problems,
so the summary is built with plain deterministic string formatting.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from zane.memory.embeddings import EmbeddingBackend
from zane.memory.store import SQLiteMessageStore
from zane.memory.vector_index import FaissVectorIndex

logger = logging.getLogger("zane.falcon_worker")

FALCON_SCOUT_SOURCE_ROLE = "falcon_scout_telemetry"
FALCON_SCOUT_SESSION_ID = "zane-system-telemetry"


@dataclass
class RequestLatencySample:
    path: str
    method: str
    status_code: int
    latency_ms: float
    timestamp: float


class LatencyRecorder:
    """Thread/coroutine-safe bounded ring buffer of recent request timings.
    Written to by FastAPI middleware (see `zane/interfaces/api.py`), read
    and drained by `FalconWorker` each cycle."""

    def __init__(self, maxlen: int = 2000) -> None:
        self._lock = threading.Lock()
        self._samples: Deque[RequestLatencySample] = deque(maxlen=maxlen)

    def record(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._samples.append(
                RequestLatencySample(
                    path=path, method=method, status_code=status_code,
                    latency_ms=latency_ms, timestamp=time.time(),
                )
            )

    def snapshot_and_clear(self) -> List[RequestLatencySample]:
        with self._lock:
            samples = list(self._samples)
            self._samples.clear()
            return samples


class InMemoryLogCapture(logging.Handler):
    """A `logging.Handler` that captures ERROR-and-above records from
    Zane's own logger hierarchy into a bounded buffer, so Falcon Scout can
    report "internal exceptions caught" without needing an external log
    aggregation service."""

    def __init__(self, maxlen: int = 500) -> None:
        super().__init__(level=logging.ERROR)
        self._lock = threading.Lock()
        self._records: Deque[str] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted = self.format(record)
        except Exception:  # noqa: BLE001 - a formatting bug must not crash the logger
            formatted = f"{record.levelname} {record.name}: {record.getMessage()}"
        with self._lock:
            self._records.append(formatted)

    def drain(self) -> List[str]:
        with self._lock:
            records = list(self._records)
            self._records.clear()
            return records

    def install(self, logger_name: str = "zane") -> None:
        logging.getLogger(logger_name).addHandler(self)


@dataclass
class SystemFaultState:
    """Thread-safe holder for "is something critically wrong right now,"
    set by `FalconWorker` and read by
    `zane.hardware.hardware_state_controller.HardwareStateController`,
    which engages its priority override (discarding the humor protocol)
    while a critical fault is active. Same snapshot/update pattern as
    `zane.thermal_monitor.IceProtocolState`."""

    active: bool = False
    reason: str = ""
    since_ts: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def snapshot(self) -> "SystemFaultState":
        with self._lock:
            return SystemFaultState(active=self.active, reason=self.reason, since_ts=self.since_ts)

    def set(self, active: bool, reason: str = "") -> None:
        with self._lock:
            if active and not self.active:
                self.since_ts = time.time()
            self.active = active
            self.reason = reason if active else ""


@dataclass
class FalconWorkerConfig:
    interval_s: float = 60.0
    latency_threshold_ms: float = 1500.0
    db_latency_threshold_ms: float = 200.0
    session_id: str = FALCON_SCOUT_SESSION_ID
    # A cycle reporting at least this many caught exceptions, or a DB probe
    # that fails entirely (not just slow), is classified "critical" and
    # sets SystemFaultState.active — see FalconWorker._classify_severity.
    critical_exception_count: int = 5


class FalconWorker:
    def __init__(
        self,
        memory_store: SQLiteMessageStore,
        embeddings: EmbeddingBackend,
        vector_index: FaissVectorIndex,
        latency_recorder: LatencyRecorder,
        log_capture: InMemoryLogCapture,
        config: Optional[FalconWorkerConfig] = None,
        fault_state: Optional[SystemFaultState] = None,
    ) -> None:
        self._memory_store = memory_store
        self._embeddings = embeddings
        self._vector_index = vector_index
        self._latency_recorder = latency_recorder
        self._log_capture = log_capture
        self.config = config or FalconWorkerConfig()
        # Optional: when provided, critical cycles update this so
        # zane.hardware.hardware_state_controller can react in real time,
        # independent of (and faster than) anything reading it back out of
        # the RAG memory store.
        self._fault_state = fault_state
        self._running = False

    def _summarize_latency(self, samples: List[RequestLatencySample]) -> Optional[str]:
        if not samples:
            return None
        spikes = [s for s in samples if s.latency_ms >= self.config.latency_threshold_ms]
        if not spikes:
            return None
        worst = max(spikes, key=lambda s: s.latency_ms)
        return (
            f"{len(spikes)}/{len(samples)} requests exceeded the "
            f"{self.config.latency_threshold_ms:.0f}ms latency threshold "
            f"(worst: {worst.method} {worst.path} at {worst.latency_ms:.0f}ms, "
            f"status {worst.status_code})."
        )

    def _probe_db(self) -> Tuple[Optional[str], bool]:
        """Returns (issue message or None, is_critical). A DB that fails
        entirely is critical; one that's merely slow is a lesser finding."""
        try:
            elapsed_ms = self._memory_store.ping()
        except Exception as exc:  # noqa: BLE001
            return f"SQLite ping failed entirely: {exc}", True
        if elapsed_ms >= self.config.db_latency_threshold_ms:
            return (
                f"SQLite ping took {elapsed_ms:.1f}ms, exceeding the "
                f"{self.config.db_latency_threshold_ms:.0f}ms threshold."
            ), False
        return None, False

    async def _run_cycle(self) -> None:
        latency_samples = self._latency_recorder.snapshot_and_clear()
        latency_issue = self._summarize_latency(latency_samples)
        db_issue, db_critical = await asyncio.to_thread(self._probe_db)
        caught_exceptions = self._log_capture.drain()

        findings: List[str] = []
        if latency_issue:
            findings.append(f"LATENCY: {latency_issue}")
        if db_issue:
            findings.append(f"DATABASE: {db_issue}")
        if caught_exceptions:
            findings.append(
                f"EXCEPTIONS: {len(caught_exceptions)} error(s) logged since last scout cycle. "
                f"Most recent: {caught_exceptions[-1]}"
            )

        is_critical = db_critical or len(caught_exceptions) >= self.config.critical_exception_count

        if not findings:
            logger.debug("Falcon Scout cycle: nothing to report (%d requests observed).", len(latency_samples))
            if self._fault_state is not None:
                self._fault_state.set(False)
            return

        summary = (
            f"[FALCON SCOUT REPORT @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] "
            + " | ".join(findings)
        )

        if self._fault_state is not None:
            self._fault_state.set(is_critical, reason=summary if is_critical else "")
            if is_critical:
                logger.critical("[FALCON SCOUT] Critical system fault flagged: %s", summary)

        await self._write_telemetry(summary)

    async def _write_telemetry(self, summary: str) -> None:
        try:
            msg_id = await asyncio.to_thread(
                self._memory_store.insert_message,
                self.config.session_id,
                FALCON_SCOUT_SOURCE_ROLE,
                summary,
                time.time(),
            )
        except Exception:
            logger.exception("Falcon Scout failed to write telemetry to SQLite; dropping this report.")
            return

        try:
            vector = await asyncio.to_thread(self._embeddings.embed, summary)
            await asyncio.to_thread(self._vector_index.add, msg_id, vector)
            await asyncio.to_thread(self._memory_store.set_embedding_id, msg_id, msg_id)
        except Exception as exc:  # noqa: BLE001 - embedding is best-effort, matching PersistentMemory
            logger.warning("Falcon Scout telemetry stored without embedding: %s", exc)

        logger.info("Falcon Scout logged a telemetry report to long-term memory: %s", summary)

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad cycle must not kill the daemon
                logger.exception("Falcon Scout cycle raised unexpectedly; continuing.")
            await asyncio.sleep(self.config.interval_s)

    def stop(self) -> None:
        self._running = False
