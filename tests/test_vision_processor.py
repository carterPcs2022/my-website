import pytest

from zane.hardware.hal import MockHAL
from zane.hardware.vision_processor import (
    FaceDetection,
    VisionFrameResult,
    VisionPipeline,
    VisionPipelineError,
)


@pytest.fixture
def pipeline():
    return VisionPipeline(MockHAL(), known_face_width_m=0.16, focal_length_px=700.0)


def test_estimate_distance_m_uses_pinhole_geometry(pipeline):
    # distance = (known_width * focal_length) / bbox_width_px
    distance = pipeline._estimate_distance_m(350.0)
    assert distance == pytest.approx((0.16 * 700.0) / 350.0)


def test_estimate_distance_m_closer_face_larger_bbox(pipeline):
    far = pipeline._estimate_distance_m(100.0)
    near = pipeline._estimate_distance_m(400.0)
    assert near < far


def test_estimate_distance_m_zero_width_is_infinite(pipeline):
    assert pipeline._estimate_distance_m(0.0) == float("inf")


def test_format_hud_block_no_faces():
    result = VisionFrameResult(detected_faces_count=0, faces=[], timestamp_ns=123)
    block = VisionPipeline.format_hud_block(result)
    assert block.startswith("[SENSORY_HUD_INPUT]:")
    assert "0 detected" in block
    assert "no targets in frame" in block


def test_format_hud_block_with_faces():
    faces = [
        FaceDetection(bounding_box=(10, 20, 100, 100), estimated_distance_m=1.5),
        FaceDetection(bounding_box=(200, 20, 80, 80), estimated_distance_m=2.75),
    ]
    result = VisionFrameResult(detected_faces_count=2, faces=faces, timestamp_ns=123)
    block = VisionPipeline.format_hud_block(result)
    assert "2 detected" in block
    assert "1.50m" in block
    assert "2.75m" in block
    assert "(10, 20, 100, 100)" in block


async def test_tick_degrades_gracefully_without_cv2_installed(pipeline):
    # cv2 is not a project dependency and is not installed in this
    # environment by design (see requirements-hardware.txt) — this
    # exercises the real "opencv not installed" degradation path.
    await pipeline._tick()
    assert pipeline.get_latest_hud() is None
    assert pipeline._permanently_disabled is True


async def test_run_forever_stops_after_permanent_disable(pipeline):
    pipeline.detection_interval_s = 0.01
    await pipeline.run_forever()  # must return promptly, not loop forever
    assert pipeline._permanently_disabled is True


def test_stop_is_safe_when_never_started(pipeline):
    pipeline.stop()  # must not raise even though _capture was never opened
    assert pipeline.get_latest_hud() is None
