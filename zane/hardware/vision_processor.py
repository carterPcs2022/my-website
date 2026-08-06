"""Optical scanning pipeline: camera capture + local face detection,
formatted into a "[SENSORY_HUD_INPUT]" block injected into Zane's system
prompt so he can reason about who/what is physically in front of him.

Uses a real (not fabricated) `cv2.CascadeClassifier` Haar-cascade face
detector — the classifier XML ships inside the `opencv-python`/
`opencv-python-headless` package itself (`cv2.data.haarcascades`), so no
model download or network access is required. `spatial_distance_meters`
is a genuine monocular distance estimate via the pinhole-camera similar-
triangles approximation (known object width / apparent pixel width x
focal length) — a standard, honestly-approximate technique, not a
LIDAR-grade measurement; see `_estimate_distance_m`'s docstring.

Privacy-by-design: HUD data is deliberately ephemeral. It lives only in
`VisionPipeline`'s in-memory state and the live system-prompt context for
the current turn — it is never written to the persistent RAG memory
store, so this does not create a standing log of who has appeared in
front of the camera.

`cv2` is not a project dependency by default (see requirements-hardware.txt)
and there is no camera in this project's Render/Docker deployment; both
paths degrade to "no HUD data" rather than raising.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from zane.hardware.hal import HardwareAbstractionLayer

logger = logging.getLogger("zane.hardware.vision_processor")

BoundingBox = Tuple[int, int, int, int]  # x, y, width, height in pixels

# Rough default calibration: an average adult face is ~16cm wide, and this
# focal length (in pixels) is a common ballpark for a 720p webcam at
# default FOV — replace both with a real calibration for your specific
# camera if accuracy matters.
_DEFAULT_KNOWN_FACE_WIDTH_M = 0.16
_DEFAULT_FOCAL_LENGTH_PX = 700.0


class VisionPipelineError(RuntimeError):
    """Raised internally when the camera or cascade classifier can't be
    initialized; always caught by `VisionPipeline.run_forever`, never
    propagates."""


@dataclass
class FaceDetection:
    bounding_box: BoundingBox
    estimated_distance_m: float


@dataclass
class VisionFrameResult:
    detected_faces_count: int
    faces: List[FaceDetection]
    timestamp_ns: int


class VisionPipeline:
    def __init__(
        self,
        hal: HardwareAbstractionLayer,
        *,
        camera_index: int = 0,
        detection_interval_s: float = 1.0,
        known_face_width_m: float = _DEFAULT_KNOWN_FACE_WIDTH_M,
        focal_length_px: float = _DEFAULT_FOCAL_LENGTH_PX,
    ) -> None:
        self._hal = hal
        self.camera_index = camera_index
        self.detection_interval_s = detection_interval_s
        self.known_face_width_m = known_face_width_m
        self.focal_length_px = focal_length_px

        self._lock = threading.Lock()
        self._latest_hud: Optional[str] = None
        self._cascade = None  # lazily loaded cv2.CascadeClassifier
        self._capture = None  # lazily opened cv2.VideoCapture
        self._permanently_disabled = False  # True if cv2 itself isn't installed
        self._running = False

    # --- lazy hardware/model initialization ---

    def _ensure_ready(self) -> None:
        if self._cascade is not None and self._capture is not None:
            return
        try:
            import cv2
        except ImportError as exc:
            raise VisionPipelineError(
                "opencv-python(-headless) is not installed. See "
                "requirements-hardware.txt (only needed for real vision hardware)."
            ) from exc

        if self._cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise VisionPipelineError(f"Failed to load bundled Haar cascade at {cascade_path}")
            self._cascade = cascade

        if self._capture is None:
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                capture.release()
                raise VisionPipelineError(f"Camera index {self.camera_index} could not be opened.")
            self._capture = capture

    def _estimate_distance_m(self, bbox_width_px: float) -> float:
        """Monocular distance estimate via similar triangles:
        distance = (known_real_width x focal_length_px) / apparent_width_px.
        This is a rough, single-axis approximation — accurate only insofar
        as `known_face_width_m`/`focal_length_px` are calibrated for the
        actual camera and actual person, not a precise measurement."""
        if bbox_width_px <= 0:
            return float("inf")
        return (self.known_face_width_m * self.focal_length_px) / bbox_width_px

    def _detect_faces(self, frame) -> VisionFrameResult:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        faces = [
            FaceDetection(
                bounding_box=(int(x), int(y), int(w), int(h)),
                estimated_distance_m=self._estimate_distance_m(float(w)),
            )
            for (x, y, w, h) in boxes
        ]
        return VisionFrameResult(
            detected_faces_count=len(faces), faces=faces, timestamp_ns=time.time_ns()
        )

    def _capture_and_detect_blocking(self) -> VisionFrameResult:
        self._ensure_ready()
        ok, frame = self._capture.read()
        if not ok:
            raise VisionPipelineError("Camera read() failed (device disconnected?).")
        return self._detect_faces(frame)

    @staticmethod
    def format_hud_block(result: VisionFrameResult) -> str:
        """Formats a detection result as the "[SENSORY_HUD_INPUT]" block
        injected into the system prompt (see zane/personality.py's
        `PersonaContext.sensory_hud`)."""
        if result.detected_faces_count == 0:
            body = "no targets in frame"
        else:
            parts = [
                f"target_{i + 1}(distance={face.estimated_distance_m:.2f}m, "
                f"bbox={face.bounding_box})"
                for i, face in enumerate(result.faces)
            ]
            body = "; ".join(parts)
        return (
            f"[SENSORY_HUD_INPUT]: Target array updated "
            f"({result.detected_faces_count} detected) — {body}"
        )

    async def _tick(self) -> None:
        try:
            result = await asyncio.to_thread(self._capture_and_detect_blocking)
        except VisionPipelineError as exc:
            if "opencv-python" in str(exc):
                # Missing dependency entirely: this will never succeed
                # without an install + restart, so stop retrying.
                self._permanently_disabled = True
                logger.warning("Vision pipeline permanently disabled: %s", exc)
            else:
                logger.debug("Vision pipeline tick degraded (will retry): %s", exc)
            with self._lock:
                self._latest_hud = None
            return

        hud = self.format_hud_block(result)
        with self._lock:
            self._latest_hud = hud
        logger.debug(hud)

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            if self._permanently_disabled:
                return
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bad tick must not kill the loop
                logger.exception("Vision pipeline tick failed unexpectedly.")
                with self._lock:
                    self._latest_hud = None
            await asyncio.sleep(self.detection_interval_s)

    def get_latest_hud(self) -> Optional[str]:
        with self._lock:
            return self._latest_hud

    def stop(self) -> None:
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
