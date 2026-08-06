import logging

import numpy as np
import pytest

from zane.falcon_worker import (
    FALCON_SCOUT_SOURCE_ROLE,
    FalconWorker,
    FalconWorkerConfig,
    InMemoryLogCapture,
    LatencyRecorder,
)
from zane.memory.store import SQLiteMessageStore


class _FakeEmbeddings:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed(self, text):
        if self.fail:
            raise RuntimeError("embedding backend unavailable")
        return np.array([1.0, 0.0], dtype="float32")


class _FakeVectorIndex:
    def __init__(self):
        self.added = []

    def add(self, vec_id, vector):
        self.added.append(vec_id)


@pytest.fixture
def store(tmp_path):
    s = SQLiteMessageStore(str(tmp_path / "falcon.sqlite3"))
    yield s
    s.close()


def test_latency_recorder_record_and_snapshot_clears():
    recorder = LatencyRecorder()
    recorder.record("/chat", "POST", 200, 42.0)
    recorder.record("/chat", "POST", 500, 1234.0)

    samples = recorder.snapshot_and_clear()
    assert len(samples) == 2
    assert samples[1].status_code == 500

    assert recorder.snapshot_and_clear() == []


def test_latency_recorder_is_bounded():
    recorder = LatencyRecorder(maxlen=3)
    for i in range(10):
        recorder.record("/x", "GET", 200, float(i))
    samples = recorder.snapshot_and_clear()
    assert len(samples) == 3
    assert samples[-1].latency_ms == 9.0  # only the most recent 3 survive


def test_in_memory_log_capture_only_captures_error_and_above():
    capture = InMemoryLogCapture()
    logger = logging.getLogger("zane.test.falcon")
    logger.addHandler(capture)
    logger.setLevel(logging.DEBUG)

    logger.info("this should not be captured")
    logger.error("this should be captured")

    records = capture.drain()
    assert len(records) == 1
    assert "this should be captured" in records[0]
    assert capture.drain() == []  # drained


async def test_run_cycle_writes_nothing_when_all_clear(store):
    worker = FalconWorker(
        store, _FakeEmbeddings(), _FakeVectorIndex(), LatencyRecorder(), InMemoryLogCapture()
    )
    await worker._run_cycle()
    assert store.get_all_messages_ordered("zane-system-telemetry") == []


async def test_run_cycle_reports_latency_spike(store):
    latency = LatencyRecorder()
    latency.record("/chat", "POST", 200, 5000.0)
    worker = FalconWorker(
        store, _FakeEmbeddings(), _FakeVectorIndex(), latency, InMemoryLogCapture(),
        FalconWorkerConfig(latency_threshold_ms=1000.0),
    )
    await worker._run_cycle()

    rows = store.get_all_messages_ordered("zane-system-telemetry")
    assert len(rows) == 1
    assert rows[0].role == FALCON_SCOUT_SOURCE_ROLE
    assert "LATENCY" in rows[0].content


async def test_run_cycle_reports_captured_exceptions(store):
    log_capture = InMemoryLogCapture()
    logging.getLogger("zane.test.falcon2").addHandler(log_capture)
    logging.getLogger("zane.test.falcon2").error("a real internal exception")

    worker = FalconWorker(store, _FakeEmbeddings(), _FakeVectorIndex(), LatencyRecorder(), log_capture)
    await worker._run_cycle()

    rows = store.get_all_messages_ordered("zane-system-telemetry")
    assert len(rows) == 1
    assert "EXCEPTIONS" in rows[0].content
    assert "a real internal exception" in rows[0].content


async def test_run_cycle_degrades_gracefully_when_embedding_fails(store):
    latency = LatencyRecorder()
    latency.record("/chat", "POST", 200, 5000.0)
    worker = FalconWorker(
        store, _FakeEmbeddings(fail=True), _FakeVectorIndex(), latency, InMemoryLogCapture(),
        FalconWorkerConfig(latency_threshold_ms=1000.0),
    )
    await worker._run_cycle()  # must not raise

    rows = store.get_all_messages_ordered("zane-system-telemetry")
    assert len(rows) == 1  # still durably written despite the embedding failure


async def test_run_cycle_reports_db_probe_failure(store, monkeypatch):
    def _broken_ping():
        raise RuntimeError("db is on fire")

    monkeypatch.setattr(store, "ping", _broken_ping)
    worker = FalconWorker(store, _FakeEmbeddings(), _FakeVectorIndex(), LatencyRecorder(), InMemoryLogCapture())
    await worker._run_cycle()

    rows = store.get_all_messages_ordered("zane-system-telemetry")
    assert len(rows) == 1
    assert "DATABASE" in rows[0].content
