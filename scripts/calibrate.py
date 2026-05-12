"""Run the full pipeline against a non-seizure baseline source (webcam or
video file) and report per-type probability statistics + recommended bias
adjustments so that the 95th-percentile baseline lands near 0.20.

Usage:
    # Live webcam, 60 seconds of "normal" behavior
    uv run python scripts/calibrate.py --seconds 60

    # Pre-recorded baseline video file(s)
    uv run python scripts/calibrate.py --source path/to/clip.mp4
    uv run python scripts/calibrate.py --source path/to/dir   # all .mp4 in dir

The script prints a config-edit snippet you can paste into config.py (or set
via SD_* env vars). It does NOT modify config.py itself.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from buffers import FaceBuffer, KeypointBuffer  # noqa: E402
from config import settings  # noqa: E402
from detector import Detector  # noqa: E402
from face_detector import FaceDetector  # noqa: E402
from face_scorer import FaceScorer  # noqa: E402
from scorer import BodyScorer  # noqa: E402


# Map probability type -> bias attribute name in Settings
BIAS_FOR_TYPE: dict[str, str] = {
    "tonic_clonic": "w_tc_bias",
    "clonic": "w_cl_bias",
    "myoclonic": "w_my_bias",
    "atonic": "w_at_bias",
    "focal_motor": "w_fm_bias",
    "versive": "w_ve_bias",
    "eyelid_myoclonia": "w_em_bias",
    "oral_automatism": "w_oa_bias",
    "hemifacial_clonic": "w_hf_bias",
}

TARGET_P95 = 0.20  # baseline p95 should sit at or below this


def _iter_sources(source: str) -> list:
    if source.isdigit():
        return [int(source)]
    p = Path(source)
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.mp4")) + sorted(
            str(x) for x in p.glob("*.mov")
        )
    if p.is_file():
        return [str(p)]
    raise SystemExit(f"Source not found: {source}")


def _match_faces(faces, detections):
    out = {}
    used = set()
    for f in faces:
        nx, ny = f.nose_xy
        best_id, best_area = None, float("inf")
        for det in detections:
            if det.track_id in used:
                continue
            x1, y1, x2, y2 = det.bbox
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area, best_id = area, det.track_id
        if best_id is not None:
            out[best_id] = f
            used.add(best_id)
    for f in faces:
        if any(f is v for v in out.values()):
            continue
        nx, ny = f.nose_xy
        best_id, best_dist = None, float("inf")
        for det in detections:
            if det.track_id in used:
                continue
            x1, y1, x2, y2 = det.bbox
            cy = (y1 + y2) / 2
            if x1 <= nx <= x2:
                d = abs(ny - cy)
                if d < best_dist:
                    best_dist, best_id = d, det.track_id
        if best_id is not None:
            out[best_id] = f
            used.add(best_id)
    return out


def _logit(p: float) -> float:
    p = max(1e-4, min(1 - 1e-4, p))
    return math.log(p / (1 - p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default=str(settings.camera_source),
        help="Webcam index (e.g. 0), single video file, or directory of videos",
    )
    ap.add_argument(
        "--seconds", type=float, default=60.0, help="Max seconds for webcam capture"
    )
    ap.add_argument(
        "--warmup-seconds",
        type=float,
        default=settings.window_seconds + 0.5,
        help="Skip this many seconds at the start of each source (window fill)",
    )
    args = ap.parse_args()

    sources = _iter_sources(args.source)
    print(f"Calibrating against {len(sources)} source(s): {sources}")

    detector = Detector(
        model_path=settings.resolve_model_path(),
        tracker=settings.tracker,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        imgsz=settings.imgsz,
    )
    body_scorer = BodyScorer(settings)

    face_detector = None
    face_scorer = None
    if Path(settings.face_model_path).exists():
        face_detector = FaceDetector(settings.face_model_path, settings.face_max_num)
        face_scorer = FaceScorer(settings)
    else:
        print(
            f"WARNING: {settings.face_model_path} missing — face calibration skipped"
        )

    samples: dict[str, list[float]] = defaultdict(list)
    frame_interval = 1.0 / max(settings.target_fps, 1)

    for src in sources:
        is_webcam = isinstance(src, int)
        cap = cv2.VideoCapture(
            src, cv2.CAP_DSHOW if is_webcam else cv2.CAP_ANY
        )
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
            cap.set(cv2.CAP_PROP_FPS, settings.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"  could not open {src}, skipping")
            continue
        print(f"\n>>> source: {src}")

        kp_bufs: dict[int, KeypointBuffer] = {}
        face_bufs: dict[int, FaceBuffer] = {}

        t_start = time.time()
        t_source_start = t_start
        last_loop = 0.0
        n_frames = 0
        n_faces_detected = 0
        n_faces_matched = 0
        n_detections = 0
        while True:
            if is_webcam:
                now = time.time()
                w = frame_interval - (now - last_loop)
                if w > 0:
                    time.sleep(w)
                last_loop = time.time()
                if time.time() - t_start > args.seconds:
                    break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            n_frames += 1
            ts = time.time()

            detections, _ = detector.step(frame)
            n_detections += len(detections)
            faces_by_id = {}
            if face_detector is not None and detections:
                try:
                    faces = face_detector.detect(frame, int(ts * 1000))
                except Exception as exc:
                    print("  face detect error:", exc)
                    faces = []
                n_faces_detected += len(faces)
                faces_by_id = _match_faces(faces, detections)
                n_faces_matched += len(faces_by_id)

            for det in detections:
                kp = kp_bufs.get(det.track_id)
                if kp is None:
                    kp = KeypointBuffer(settings.window_seconds, settings.target_fps)
                    kp_bufs[det.track_id] = kp
                kp.append(ts, det.keypoints_xy, det.keypoints_conf, det.bbox)
                bs = body_scorer.score(det.track_id, kp)
                after_warmup = (ts - t_source_start) > args.warmup_seconds
                if bs is not None and after_warmup:
                    for k, v in bs.probabilities.items():
                        samples[k].append(float(v))

                obs = faces_by_id.get(det.track_id)
                if obs is not None and face_scorer is not None:
                    fb = face_bufs.get(det.track_id)
                    if fb is None:
                        fb = FaceBuffer(settings.window_seconds, settings.target_fps)
                        face_bufs[det.track_id] = fb
                    fb.append(
                        ts,
                        obs.blendshapes,
                        obs.head_yaw_deg,
                        obs.head_pitch_deg,
                        obs.head_roll_deg,
                    )
                    body_st = bs.features.get("stillness", 0.0) if bs else 0.0
                    body_mo = bs.features.get("any_motion", 0.0) if bs else 0.0
                    fs = face_scorer.score(det.track_id, fb, body_st, body_mo)
                    if fs is not None and after_warmup:
                        for k, v in fs.probabilities.items():
                            samples[k].append(float(v))

        cap.release()
        face_rate = (n_faces_detected / max(n_frames, 1))
        match_rate = (n_faces_matched / max(n_faces_detected, 1)) if n_faces_detected else 0.0
        print(
            f"  processed {n_frames} frames; "
            f"avg detections/frame={n_detections / max(n_frames, 1):.2f}; "
            f"faces/frame={face_rate:.2f}; matched={match_rate:.2%}"
        )

    if face_detector is not None:
        face_detector.close()

    if not samples:
        print("No samples collected. Increase --seconds or check the source.")
        return 1

    # --- Report ---------------------------------------------------------
    print("\n=== Baseline per-type probability distribution ===")
    print(f"{'type':22s} {'p50':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s} {'n':>7s}")
    stats: dict[str, dict[str, float]] = {}
    for t, vals in samples.items():
        if not vals:
            continue
        arr = np.array(vals)
        st = {
            "p50": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "n": len(arr),
        }
        stats[t] = st
        print(
            f"{t:22s} {st['p50']:7.3f} {st['p90']:7.3f} {st['p95']:7.3f} "
            f"{st['p99']:7.3f} {st['max']:7.3f} {st['n']:7d}"
        )

    print("\n=== Recommended bias adjustments (push p95 toward 0.20) ===")
    print(f"Target p95={TARGET_P95}.  delta = logit(target) - logit(p95)")
    print()
    target_logit = _logit(TARGET_P95)
    bias_lines: list[str] = []
    for t, st in stats.items():
        p95 = st["p95"]
        if p95 < TARGET_P95:
            continue
        delta = target_logit - _logit(p95)        # NEGATIVE — subtract from bias
        bias_key = BIAS_FOR_TYPE.get(t)
        if bias_key is None:
            continue
        old = getattr(settings, bias_key)
        new = old + delta
        bias_lines.append(
            f"    {bias_key}: float = {new:.3f}   # was {old:.3f}, p95 {p95:.3f} -> ~{TARGET_P95}"
        )
    if not bias_lines:
        print("  All types already under target — no changes recommended.")
    else:
        print("Paste into config.py inside class Settings:")
        print()
        for line in bias_lines:
            print(line)
        print()
        print("Or export at runtime:")
        for line in bias_lines:
            key = line.split(":")[0].strip()
            val = line.split("=")[1].split("#")[0].strip()
            print(f"  $env:SD_{key.upper()} = '{val}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
