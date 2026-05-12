from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FaceObservation:
    nose_xy: tuple[float, float]              # in original image pixel coords
    bbox_xy: tuple[float, float, float, float]  # x1,y1,x2,y2
    blendshapes: dict[str, float]
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float


def _rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Return yaw (Y), pitch (X), roll (Z) in degrees from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-R[1, 2], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class FaceDetector:
    """Wraps MediaPipe Face Landmarker (Tasks API, VIDEO mode).

    Returns one FaceObservation per detected face, with blendshapes (52 AU-like
    outputs in 0..1) and head pose Euler angles in degrees.
    """

    def __init__(self, model_path: str, max_faces: int) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                "Run: uv run python scripts/download_face_model.py"
            )

        # Imports are local so a missing mediapipe install fails late, not at
        # process startup if face_enabled=False later.
        import mediapipe as mp  # noqa: WPS433
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=max_faces,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        log.info("Face landmarker loaded: %s (max_faces=%d)", model_path, max_faces)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[FaceObservation]:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # MediaPipe expects RGB
        rgb = frame_bgr[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            log.exception("Face landmarker failed")
            return []

        H, W = frame_bgr.shape[:2]
        observations: list[FaceObservation] = []

        n_faces = len(result.face_landmarks)
        for i in range(n_faces):
            lms = result.face_landmarks[i]
            blendshapes_raw = (
                result.face_blendshapes[i] if i < len(result.face_blendshapes) else []
            )
            matrices = (
                result.facial_transformation_matrixes
                if hasattr(result, "facial_transformation_matrixes")
                else []
            )
            matrix = matrices[i] if i < len(matrices) else None

            # Pixel-space landmarks
            xs = np.fromiter((lm.x * W for lm in lms), dtype=np.float32)
            ys = np.fromiter((lm.y * H for lm in lms), dtype=np.float32)
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
            nose_idx = 1  # MediaPipe nose tip
            nose = (float(xs[nose_idx]), float(ys[nose_idx]))

            blend = {b.category_name: float(b.score) for b in blendshapes_raw}

            if matrix is not None:
                R = np.asarray(matrix)[:3, :3]
                yaw, pitch, roll = _rotation_to_euler(R)
            else:
                yaw = pitch = roll = 0.0

            observations.append(
                FaceObservation(
                    nose_xy=nose,
                    bbox_xy=(x1, y1, x2, y2),
                    blendshapes=blend,
                    head_yaw_deg=yaw,
                    head_pitch_deg=pitch,
                    head_roll_deg=roll,
                )
            )
        return observations

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
