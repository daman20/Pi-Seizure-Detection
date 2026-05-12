from __future__ import annotations

import logging
import threading
import time

import cv2

from buffers import FaceBuffer, KeypointBuffer
from camera import Camera
from config import Settings
from detector import Detector
from face_detector import FaceDetector, FaceObservation
from face_scorer import FaceScorer
from scorer import BodyScorer, TypeScores
from state import PersonStateStore

log = logging.getLogger(__name__)


class Worker(threading.Thread):
    def __init__(
        self,
        settings: Settings,
        camera: Camera,
        detector: Detector,
        body_scorer: BodyScorer,
        face_scorer: FaceScorer | None,
        face_detector: FaceDetector | None,
        store: PersonStateStore,
    ) -> None:
        super().__init__(daemon=True, name="seizure-worker")
        self.settings = settings
        self.camera = camera
        self.detector = detector
        self.body_scorer = body_scorer
        self.face_scorer = face_scorer
        self.face_detector = face_detector
        self.store = store
        self._stop = threading.Event()
        self._kp_buffers: dict[int, KeypointBuffer] = {}
        self._face_buffers: dict[int, FaceBuffer] = {}
        self._last_face_seen_ts: dict[int, float] = {}

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def run(self) -> None:
        cv2.setNumThreads(2)
        frame_interval = 1.0 / max(self.settings.target_fps, 1)
        last_loop = 0.0

        while not self._stop.is_set():
            now = time.time()
            wait = frame_interval - (now - last_loop)
            if wait > 0:
                time.sleep(wait)
            last_loop = time.time()

            ok, frame = self.camera.read()
            if not ok or frame is None:
                continue

            try:
                detections, raw_result = self.detector.step(frame)
            except Exception:
                log.exception("Detector step failed")
                continue

            ts = time.time()

            # Optional face pass on the whole frame, then match each face to a track_id.
            faces_by_track: dict[int, FaceObservation] = {}
            if self.face_detector is not None and detections:
                try:
                    faces = self.face_detector.detect(frame, timestamp_ms=int(ts * 1000))
                except Exception:
                    log.exception("Face detector failed")
                    faces = []
                faces_by_track = self._match_faces(faces, detections)
                for tid in faces_by_track:
                    self._last_face_seen_ts[tid] = ts

            per_track_probs: dict[int, dict[str, float]] = {}
            per_track_features: dict[int, dict[str, float]] = {}
            per_track_window: dict[int, float] = {}

            for det in detections:
                # Body buffer + score
                kp_buf = self._kp_buffers.get(det.track_id)
                if kp_buf is None:
                    kp_buf = KeypointBuffer(
                        window_seconds=self.settings.window_seconds,
                        target_fps=self.settings.target_fps,
                    )
                    self._kp_buffers[det.track_id] = kp_buf
                kp_buf.append(ts, det.keypoints_xy, det.keypoints_conf, det.bbox)
                body_score: TypeScores | None = self.body_scorer.score(det.track_id, kp_buf)

                # Face buffer + score
                face_score: TypeScores | None = None
                obs = faces_by_track.get(det.track_id)
                if self.face_scorer is not None and obs is not None:
                    fb = self._face_buffers.get(det.track_id)
                    if fb is None:
                        fb = FaceBuffer(
                            window_seconds=self.settings.window_seconds,
                            target_fps=self.settings.target_fps,
                        )
                        self._face_buffers[det.track_id] = fb
                    fb.append(
                        ts,
                        obs.blendshapes,
                        obs.head_yaw_deg,
                        obs.head_pitch_deg,
                        obs.head_roll_deg,
                    )
                    body_stillness = body_score.features.get("stillness", 0.0) if body_score else 0.0
                    body_any = body_score.features.get("any_motion", 0.0) if body_score else 0.0
                    face_score = self.face_scorer.score(det.track_id, fb, body_stillness, body_any)

                merged_probs: dict[str, float] = {}
                merged_features: dict[str, float] = {}
                window_sec = 0.0
                if body_score is not None:
                    merged_probs.update(body_score.probabilities)
                    merged_features.update(body_score.features)
                    window_sec = max(window_sec, body_score.window_seconds)
                if face_score is not None:
                    merged_probs.update(face_score.probabilities)
                    merged_features.update(face_score.features)
                    window_sec = max(window_sec, face_score.window_seconds)

                # Phantom-track guard: if no face has been matched to this
                # track within face_match_ttl_seconds, attenuate every emitted
                # probability so the chart caps at no_face_attenuation.
                last_face_ts = self._last_face_seen_ts.get(det.track_id, float("-inf"))
                if (
                    merged_probs
                    and ts - last_face_ts > self.settings.face_match_ttl_seconds
                ):
                    atten = self.settings.no_face_attenuation
                    merged_probs = {k: v * atten for k, v in merged_probs.items()}

                per_track_probs[det.track_id] = merged_probs
                per_track_features[det.track_id] = merged_features
                per_track_window[det.track_id] = window_sec

                self.store.update_person(
                    det.track_id,
                    det.bbox,
                    ts,
                    merged_probs or None,
                    merged_features or None,
                    window_sec,
                )

            evicted = self.store.evict_stale(ts, self.settings.track_ttl_seconds)
            for tid in evicted:
                self._kp_buffers.pop(tid, None)
                self._face_buffers.pop(tid, None)
                self._last_face_seen_ts.pop(tid, None)
                self.body_scorer.forget(tid)
                if self.face_scorer is not None:
                    self.face_scorer.forget(tid)

            annotated = self._annotate(frame, raw_result, detections, per_track_probs)
            ok2, jpeg = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
            )
            if ok2:
                self.store.set_jpeg(jpeg.tobytes())
            self.store.tick_fps(ts)

        self.camera.close()
        if self.face_detector is not None:
            self.face_detector.close()
        log.info("Worker stopped")

    # ------------------------------------------------------------------
    @staticmethod
    def _match_faces(faces: list[FaceObservation], detections) -> dict[int, FaceObservation]:
        out: dict[int, FaceObservation] = {}
        if not faces:
            return out
        used: set[int] = set()
        # Pass 1: nose strictly inside bbox (smallest enclosing bbox wins).
        for f in faces:
            nx, ny = f.nose_xy
            best_id: int | None = None
            best_area = float("inf")
            for det in detections:
                if det.track_id in used:
                    continue
                x1, y1, x2, y2 = det.bbox
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if area < best_area:
                        best_area = area
                        best_id = det.track_id
            if best_id is not None:
                out[best_id] = f
                used.add(best_id)
        # Pass 2: faces without a strict bbox match (e.g. nose just above the
        # bbox top because the person is close to the camera). Assign to the
        # nearest unused track whose horizontal center is within bbox width.
        for f in faces:
            if any(f is v for v in out.values()):
                continue
            nx, ny = f.nose_xy
            best_id = None
            best_dist = float("inf")
            for det in detections:
                if det.track_id in used:
                    continue
                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if x1 <= nx <= x2:
                    d = abs(ny - cy)
                    if d < best_dist:
                        best_dist = d
                        best_id = det.track_id
            if best_id is not None:
                out[best_id] = f
                used.add(best_id)
        return out

    def _annotate(self, frame, raw_result, detections, per_track_probs):
        try:
            base = self.detector.plot(raw_result) if raw_result is not None else frame.copy()
            if base is None:
                base = frame.copy()
        except Exception:
            base = frame.copy()

        threshold = self.settings.alert_threshold
        for det in detections:
            probs = per_track_probs.get(det.track_id, {})
            if probs:
                dom_type = max(probs, key=lambda k: probs[k])
                dom_p = probs[dom_type]
            else:
                dom_type = ""
                dom_p = None
            label = f"id={det.track_id}"
            if dom_p is not None:
                short = dom_type.replace("_", " ")[:14]
                label += f" {short}={dom_p:.2f}"
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            color = (0, 0, 255) if (dom_p is not None and dom_p > threshold) else (0, 200, 0)
            cv2.rectangle(base, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                base,
                label,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return base
