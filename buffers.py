from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


# COCO-17 keypoint groups
KEYPOINTS_HEAD = [0, 1, 2, 3, 4]           # nose, eyes, ears
KEYPOINTS_SHOULDERS = [5, 6]
KEYPOINTS_ARMS_L = [7, 9]                   # elbow, wrist
KEYPOINTS_ARMS_R = [8, 10]
KEYPOINTS_ARMS = KEYPOINTS_ARMS_L + KEYPOINTS_ARMS_R
KEYPOINTS_HIPS = [11, 12]
KEYPOINTS_LEGS_L = [13, 15]                 # knee, ankle
KEYPOINTS_LEGS_R = [14, 16]
KEYPOINTS_LEGS = KEYPOINTS_LEGS_L + KEYPOINTS_LEGS_R

HIP_L, HIP_R = 11, 12
SHOULDER_L, SHOULDER_R = 5, 6
NOSE = 0


@dataclass
class BufferSample:
    t: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox: tuple[float, float, float, float]


class KeypointBuffer:
    def __init__(self, window_seconds: float, target_fps: int) -> None:
        self.window_seconds = window_seconds
        self.maxlen = math.ceil(window_seconds * target_fps) + 4
        self.samples: deque[BufferSample] = deque(maxlen=self.maxlen)

    def append(
        self,
        t: float,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> None:
        self.samples.append(BufferSample(t, keypoints_xy.copy(), keypoints_conf.copy(), bbox))
        cutoff = t - self.window_seconds
        while self.samples and self.samples[0].t < cutoff:
            self.samples.popleft()

    def __len__(self) -> int:
        return len(self.samples)

    def timespan(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].t - self.samples[0].t

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.samples)
        times = np.empty(n, dtype=np.float64)
        kxy = np.empty((n, 17, 2), dtype=np.float32)
        kconf = np.empty((n, 17), dtype=np.float32)
        bbox = np.empty((n, 4), dtype=np.float32)
        for i, s in enumerate(self.samples):
            times[i] = s.t
            kxy[i] = s.keypoints_xy
            kconf[i] = s.keypoints_conf
            bbox[i] = s.bbox
        return times, kxy, kconf, bbox


def normalize_keypoints(kxy: np.ndarray, kconf: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Translate to hip-midpoint and scale by torso length (fallback to bbox diag)."""
    out = np.empty_like(kxy, dtype=np.float32)
    for t in range(kxy.shape[0]):
        hips = kxy[t, [HIP_L, HIP_R]]
        hip_c = (kconf[t, HIP_L] + kconf[t, HIP_R]) / 2.0
        shoulders = kxy[t, [SHOULDER_L, SHOULDER_R]]
        sh_c = (kconf[t, SHOULDER_L] + kconf[t, SHOULDER_R]) / 2.0

        if hip_c > 0.3:
            origin = hips.mean(axis=0)
        else:
            origin = np.array(
                [(bbox[t, 0] + bbox[t, 2]) / 2, (bbox[t, 1] + bbox[t, 3]) / 2],
                dtype=np.float32,
            )

        if hip_c > 0.3 and sh_c > 0.3:
            torso = float(np.linalg.norm(shoulders.mean(axis=0) - hips.mean(axis=0)))
        else:
            torso = 0.0
        if torso < 1e-3:
            torso = float(math.hypot(bbox[t, 2] - bbox[t, 0], bbox[t, 3] - bbox[t, 1]))
        if torso < 1e-3:
            torso = 1.0
        out[t] = (kxy[t] - origin) / torso
    return out


# ---------------------------------------------------------------------------
# Face landmark buffer (driven by MediaPipe blendshapes + head pose)
# ---------------------------------------------------------------------------

@dataclass
class FaceSample:
    t: float
    blendshapes: dict[str, float]
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float


class FaceBuffer:
    """Per-track rolling buffer of face observations (blendshapes + head pose)."""

    def __init__(self, window_seconds: float, target_fps: int) -> None:
        self.window_seconds = window_seconds
        self.maxlen = math.ceil(window_seconds * target_fps) + 4
        self.samples: deque[FaceSample] = deque(maxlen=self.maxlen)

    def append(
        self,
        t: float,
        blendshapes: dict[str, float],
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
    ) -> None:
        self.samples.append(FaceSample(t, dict(blendshapes), yaw_deg, pitch_deg, roll_deg))
        cutoff = t - self.window_seconds
        while self.samples and self.samples[0].t < cutoff:
            self.samples.popleft()

    def __len__(self) -> int:
        return len(self.samples)

    def timespan(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].t - self.samples[0].t

    def series(self, blendshape_name: str) -> tuple[np.ndarray, np.ndarray]:
        t = np.fromiter((s.t for s in self.samples), dtype=np.float64)
        v = np.fromiter(
            (s.blendshapes.get(blendshape_name, 0.0) for s in self.samples),
            dtype=np.float32,
        )
        return t, v

    def head_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.samples)
        t = np.empty(n, dtype=np.float64)
        yaw = np.empty(n, dtype=np.float32)
        pitch = np.empty(n, dtype=np.float32)
        roll = np.empty(n, dtype=np.float32)
        for i, s in enumerate(self.samples):
            t[i] = s.t
            yaw[i] = s.head_yaw_deg
            pitch[i] = s.head_pitch_deg
            roll[i] = s.head_roll_deg
        return t, yaw, pitch, roll
