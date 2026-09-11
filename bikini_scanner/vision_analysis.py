"""Face detection for the cascade's region planning and age gate.

OpenCV 5 removed `CascadeClassifier` and no longer ships the Haar XML files, so the
old cascade-based detector silently found nothing on this build. What OpenCV 5 does
have is YuNet (`cv2.FaceDetectorYN`), a small, far more accurate DNN detector - but it
needs a ~230 KB model file that opencv-python does not bundle.

So face detection is optional here. With the model present, regions are anchored to
real faces and the age gate reads actual face crops. Without it, `detect_face_boxes`
returns an empty list, region planning falls back to fixed body bands, and the age gate
judges the whole frame. Nothing breaks either way - but callers must treat "no faces"
as "unknown", never as "no people".
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .user_prefs import prefs_path

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2: Any | None = None  # type: ignore[no-redef]

if TYPE_CHECKING:
    from PIL import Image

LOGGER = logging.getLogger(__name__)

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
# Official OpenCV Zoo release asset. Only ever fetched on an explicit user action.
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_APPROX_BYTES = 232589

DETECT_MAX_SIDE = 640
# YuNet confidence needed to accept a face. Deliberately low. A missed face is far more
# costly than a spurious one now that the age gate reasons per subject: an undetected
# person's body becomes unattributed evidence, and a photo whose only *detected* faces
# are children is excluded outright - which is exactly what happened to adults looking
# down or away at 0.7. A spurious box, by contrast, yields a face crop with no age
# signal either way, which the gate simply does not act on.
_SCORE_THRESHOLD = 0.3
_MIN_FACE_PX = 16

_LOCK = threading.Lock()
_DETECTOR: Any | None = None
_DETECTOR_PATH: Path | None = None
# resolve_model() stats two candidate paths, and it used to run two to four times per
# image: face_detection_available() called it, then detect_face_boxes() -> _detector()
# called it again. On a 20 000-image scan with face counting on that is ~80 000 syscalls
# for an answer that only changes when the model is installed or removed.
_RESOLVED: Path | None = None
_RESOLVED_KNOWN = False


@dataclass(frozen=True, slots=True)
class FaceBox:
    """A detected face in full-resolution image coordinates."""

    x: int
    y: int
    width: int
    height: int
    score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.width) * int(self.height)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def model_path() -> Path:
    """Where the face model lives once installed."""
    return prefs_path().parent / "models" / MODEL_FILENAME


def bundled_model_path() -> Path:
    """Location a packaged build may ship the model in."""
    return Path(__file__).resolve().parent.parent / "assets" / MODEL_FILENAME


def _locate_model() -> Path | None:
    for candidate in (model_path(), bundled_model_path()):
        try:
            if candidate.is_file() and candidate.stat().st_size > 1024:
                return candidate
        except OSError:
            continue
    return None


def resolve_model() -> Path | None:
    """Where the face model is, memoised for the life of the process.

    `forget_model_location()` clears it, and installing one calls that, so the app picks
    up a model it downloaded itself without a restart. A file dropped in by hand while
    the app is running is the one case that still needs one.
    """
    global _RESOLVED, _RESOLVED_KNOWN
    if _RESOLVED_KNOWN:
        return _RESOLVED
    with _LOCK:
        if not _RESOLVED_KNOWN:
            _RESOLVED = _locate_model()
            _RESOLVED_KNOWN = True
    return _RESOLVED


def forget_model_location() -> None:
    """Re-check where the model is on the next call."""
    global _RESOLVED, _RESOLVED_KNOWN
    with _LOCK:
        _RESOLVED = None
        _RESOLVED_KNOWN = False


def face_detection_available() -> bool:
    return cv2 is not None and hasattr(cv2, "FaceDetectorYN") and resolve_model() is not None


def _detector() -> Any | None:
    global _DETECTOR, _DETECTOR_PATH
    if cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
        return None
    path = resolve_model()
    if path is None:
        return None
    with _LOCK:
        if _DETECTOR is not None and path == _DETECTOR_PATH:
            return _DETECTOR
        try:
            detector = cv2.FaceDetectorYN.create(
                str(path),
                "",
                (320, 320),
                _SCORE_THRESHOLD,
                0.3,
                50,
            )
        except Exception:
            LOGGER.exception("Face model %s could not be loaded", path)
            return None
        _DETECTOR = detector
        _DETECTOR_PATH = path
        LOGGER.info("Face detection enabled using %s", path.name)
        return detector


def detect_face_boxes(image: Image.Image) -> list[FaceBox]:
    """Faces in full-resolution coordinates, largest first.

    An empty list means "no faces found *or* no detector installed" - it is not
    evidence that the image contains no people.
    """
    detector = _detector()
    if detector is None or cv2 is None:
        return []
    try:
        rgb = np.asarray(image.convert("RGB"))
    except Exception:  # noqa: BLE001
        return []
    if rgb.size == 0:
        return []
    height, width = rgb.shape[:2]
    scale = 1.0
    if max(height, width) > DETECT_MAX_SIDE:
        scale = DETECT_MAX_SIDE / float(max(height, width))
        rgb = cv2.resize(
            rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, detections = detector.detect(bgr)
    except Exception:
        LOGGER.exception("Face detection failed")
        return []
    if detections is None:
        return []
    faces: list[FaceBox] = []
    for row in detections:
        x, y, w, h = (float(value) for value in row[:4])
        # Measured against the original frame, not the downscaled copy the detector saw.
        # Comparing in detector coordinates made the threshold scale with the image: on a
        # 4000px photo reduced to 640px, a box of 16px there is a 100px face here, and
        # perfectly usable faces on large photos were being dropped while small photos
        # kept much smaller ones.
        full_width = w / scale
        full_height = h / scale
        if full_width < _MIN_FACE_PX or full_height < _MIN_FACE_PX:
            continue
        faces.append(
            FaceBox(
                x=max(0, int(round(x / scale))),
                y=max(0, int(round(y / scale))),
                width=int(round(full_width)),
                height=int(round(full_height)),
                score=float(row[-1]) if len(row) >= 15 else 1.0,
            )
        )
    faces.sort(key=lambda face: face.area, reverse=True)
    return faces


def detect_face_count(image: Image.Image) -> int | None:
    """Face count, or None when no detector is installed (unknown, not zero)."""
    if not face_detection_available():
        return None
    return len(detect_face_boxes(image))


def install_model_from_bytes(payload: bytes) -> Path:
    """Write a downloaded face model into place after verifying its checksum."""
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"unexpected checksum {digest}")
    target = model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    temporary.write_bytes(payload)
    temporary.replace(target)
    global _DETECTOR, _DETECTOR_PATH, _RESOLVED, _RESOLVED_KNOWN
    # One acquisition for all four: forget_model_location() takes the same lock, so
    # calling it from in here would deadlock.
    with _LOCK:
        _DETECTOR = None
        _DETECTOR_PATH = None
        _RESOLVED = None
        _RESOLVED_KNOWN = False
    return target
