from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    track_id: int
    bbox: tuple[float, float, float, float]
    conf: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray


class Detector:
    def __init__(
        self,
        model_path: str,
        tracker: str,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
    ) -> None:
        from ultralytics import YOLO

        if not Path(model_path).exists() and model_path.endswith(".pt"):
            log.warning("Model %s missing — Ultralytics will download yolo11n-pose.pt", model_path)
        self.model = YOLO(model_path, task="pose")
        self.tracker = tracker
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._last_results = None

    def step(self, frame: np.ndarray) -> tuple[list[Detection], object]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return [], None
        result = results[0]
        self._last_results = result

        detections: list[Detection] = []
        boxes = result.boxes
        kpts = result.keypoints
        if boxes is None or kpts is None or boxes.id is None:
            return [], result

        ids = boxes.id.int().cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        kp_xy = kpts.xy.cpu().numpy()
        kp_conf = (
            kpts.conf.cpu().numpy()
            if kpts.conf is not None
            else np.ones(kp_xy.shape[:2], dtype=np.float32)
        )
        for i, tid in enumerate(ids):
            detections.append(
                Detection(
                    track_id=int(tid),
                    bbox=tuple(float(v) for v in xyxy[i]),
                    conf=float(confs[i]),
                    keypoints_xy=kp_xy[i],
                    keypoints_conf=kp_conf[i],
                )
            )
        return detections, result

    def plot(self, result) -> np.ndarray | None:
        if result is None:
            return None
        return result.plot()
