"""Two-phase calibration. Accepts either:

  * a pre-recorded video file with known phase boundaries
        uv run python scripts/self_tune.py --source eyelid-seizure.MOV \
            --baseline 30 --positive 30 --type eyelid_myoclonia

  * the live webcam, with on-screen countdowns
        uv run python scripts/self_tune.py --type eyelid_myoclonia

The script fits the type's primary weight + bias so that:
    positive-phase p50 -> ~target_pos (default 0.75)
    baseline      p95 -> ~target_neg (default 0.20)

It prints a config snippet — it writes nothing. Stop the main service before
running (it owns the webcam).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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


TUNE_RECIPE: dict[str, tuple[str, str, str]] = {
    "eyelid_myoclonia": ("w_em_rhythmic", "w_em_bias", "eyelid_rhythmic"),
    "oral_automatism": ("w_oa_rhythmic", "w_oa_bias", "jaw_rhythmic"),
    "hemifacial_clonic": ("w_hf_asym", "w_hf_bias", "asym_rhythmic"),
    "tonic_clonic": ("w_tc_rhythmic", "w_tc_bias", "rhythmic_weighted"),
    "clonic": ("w_cl_rhythmic", "w_cl_bias", "rhythmic_weighted"),
    "myoclonic": ("w_my_jerk_rate", "w_my_bias", "jerk_rate"),
    "atonic": ("w_at_drop", "w_at_bias", "drop"),
    "versive": ("w_ve_yaw", "w_ve_bias", "yaw_mean_abs"),
}


@dataclass
class PhaseSamples:
    probabilities: list[float]
    features: list[float]
    body_stillness: list[float]


def _match_faces(faces, detections):
    out, used = {}, set()
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


def _countdown(label: str, seconds: int) -> None:
    print(f"\n>>> {label}")
    for s in range(seconds, 0, -1):
        print(f"    starts in {s}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 50, end="\r")


def _process_frame(
    frame,
    ts: float,
    detector: Detector,
    face_detector: FaceDetector,
    face_scorer: FaceScorer,
    body_scorer: BodyScorer,
    kp_bufs: dict[int, KeypointBuffer],
    face_bufs: dict[int, FaceBuffer],
    target_type: str,
    feat_key: str,
):
    """Run one frame through the pipeline. Returns (prob, feat, stillness) or None."""
    detections, _ = detector.step(frame)
    faces_by_id = {}
    if detections:
        try:
            faces = face_detector.detect(frame, int(ts * 1000))
        except Exception as exc:
            print("face detect error:", exc)
            faces = []
        faces_by_id = _match_faces(faces, detections)

    for det in detections:
        kp = kp_bufs.get(det.track_id)
        if kp is None:
            kp = KeypointBuffer(settings.window_seconds, settings.target_fps)
            kp_bufs[det.track_id] = kp
        kp.append(ts, det.keypoints_xy, det.keypoints_conf, det.bbox)
        bs = body_scorer.score(det.track_id, kp)

        obs = faces_by_id.get(det.track_id)
        fs = None
        if obs is not None:
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

        src_score = None
        if bs is not None and target_type in bs.probabilities:
            src_score = bs
        elif fs is not None and target_type in fs.probabilities:
            src_score = fs
        if src_score is None:
            continue
        prob = float(src_score.probabilities[target_type])
        feat = float(src_score.features.get(feat_key, 0.0))
        still = float(bs.features.get("stillness", 0.0)) if bs else 0.0
        return prob, feat, still
    return None


def run_video_file(
    path: Path,
    baseline_seconds: float,
    positive_seconds: float,
    args,
    detector: Detector,
    face_detector: FaceDetector,
    face_scorer: FaceScorer,
    body_scorer: BodyScorer,
    target_type: str,
    feat_key: str,
):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(src_fps / settings.target_fps))
    duration = src_frames / src_fps
    print(
        f"Video: {path}  fps={src_fps:.1f}  frames={src_frames}  "
        f"duration={duration:.1f}s  stride={stride} (effective {src_fps / stride:.1f} fps)"
    )

    phases: dict[str, PhaseSamples] = {
        "baseline": PhaseSamples([], [], []),
        "positive": PhaseSamples([], [], []),
    }
    kp_bufs: dict[int, KeypointBuffer] = {}
    face_bufs: dict[int, FaceBuffer] = {}

    pos_start = baseline_seconds
    pos_end = baseline_seconds + positive_seconds

    frame_idx = 0
    last_log = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        ts = frame_idx / src_fps  # video time in seconds

        if ts < pos_end + 0.001:
            result = _process_frame(
                frame,
                ts,
                detector,
                face_detector,
                face_scorer,
                body_scorer,
                kp_bufs,
                face_bufs,
                target_type,
                feat_key,
            )
            if result is not None:
                prob, feat, still = result
                t_in_phase = (
                    ts - 0.0
                    if ts < pos_start
                    else ts - pos_start
                    if ts < pos_end
                    else None
                )
                phase_name = None
                if ts < pos_start:
                    phase_name = "baseline"
                elif ts < pos_end:
                    phase_name = "positive"
                if phase_name and t_in_phase is not None and t_in_phase >= args.per_phase_warmup:
                    phases[phase_name].probabilities.append(prob)
                    phases[phase_name].features.append(feat)
                    phases[phase_name].body_stillness.append(still)

        # progress log every 5 seconds of video time
        sec = int(ts)
        if sec > last_log and sec % 5 == 0:
            phase_lbl = "baseline" if ts < pos_start else ("positive" if ts < pos_end else "done")
            print(f"  t={sec:3d}s  phase={phase_lbl}  collected b={len(phases['baseline'].probabilities)} p={len(phases['positive'].probabilities)}")
            last_log = sec
        if ts >= pos_end:
            break
        frame_idx += 1
    cap.release()
    return phases


def run_webcam(
    args,
    detector: Detector,
    face_detector: FaceDetector,
    face_scorer: FaceScorer,
    body_scorer: BodyScorer,
    target_type: str,
    feat_key: str,
):
    src = settings.resolve_camera()
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src, int) else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
    cap.set(cv2.CAP_PROP_FPS, settings.target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    phases: dict[str, PhaseSamples] = {
        "baseline": PhaseSamples([], [], []),
        "positive": PhaseSamples([], [], []),
    }
    kp_bufs: dict[int, KeypointBuffer] = {}
    face_bufs: dict[int, FaceBuffer] = {}

    _countdown("BASELINE — sit naturally", args.gap)
    phase = "baseline"
    phase_t0 = time.time()
    end_baseline = phase_t0 + args.baseline
    gap_until: float | None = None
    end_positive: float | None = None
    interval = 1.0 / max(settings.target_fps, 1)
    last_loop = 0.0

    while True:
        now = time.time()
        if phase == "baseline" and now >= end_baseline:
            phase = "gap"
            gap_until = now + args.gap
            print(f"\n>>> BASELINE done. SIMULATION starts in {args.gap}s — get ready.")
        if phase == "gap" and gap_until is not None and now >= gap_until:
            phase = "positive"
            end_positive = now + args.positive
            print(f"\n>>> SIMULATION — go!  ({int(args.positive)}s)")
            phase_t0_pos = now
        if phase == "positive" and end_positive is not None and now >= end_positive:
            break

        w = interval - (now - last_loop)
        if w > 0:
            time.sleep(w)
        last_loop = time.time()

        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ts = time.time()
        result = _process_frame(
            frame, ts, detector, face_detector, face_scorer,
            body_scorer, kp_bufs, face_bufs, target_type, feat_key,
        )
        if result is None:
            continue
        prob, feat, still = result
        if phase in phases:
            t_in_phase = (
                ts - phase_t0
                if phase == "baseline"
                else (ts - (end_positive - args.positive)) if phase == "positive"
                else None
            )
            if t_in_phase is not None and t_in_phase >= args.per_phase_warmup:
                phases[phase].probabilities.append(prob)
                phases[phase].features.append(feat)
                phases[phase].body_stillness.append(still)

    cap.release()
    return phases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--type", default="eyelid_myoclonia", choices=sorted(TUNE_RECIPE.keys())
    )
    ap.add_argument("--source", default=None, help="optional video file path")
    ap.add_argument("--baseline", type=float, default=30.0)
    ap.add_argument("--positive", type=float, default=15.0)
    ap.add_argument("--gap", type=int, default=5, help="countdown between phases (webcam only)")
    ap.add_argument("--target-pos", type=float, default=0.75)
    ap.add_argument("--target-neg", type=float, default=0.20)
    ap.add_argument(
        "--per-phase-warmup",
        type=float,
        default=settings.window_seconds + 0.5,
        help="seconds to skip at the start of each phase so the analysis window fills with phase-specific data",
    )
    args = ap.parse_args()
    w_attr, b_attr, feat_key = TUNE_RECIPE[args.type]

    print("=" * 60)
    print(f"CALIBRATION FOR  {args.type}")
    print("=" * 60)

    detector = Detector(
        model_path=settings.resolve_model_path(),
        tracker=settings.tracker,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        imgsz=settings.imgsz,
    )
    body_scorer = BodyScorer(settings)
    face_detector = FaceDetector(settings.face_model_path, settings.face_max_num)
    face_scorer = FaceScorer(settings)

    if args.source:
        phases = run_video_file(
            Path(args.source),
            args.baseline,
            args.positive,
            args,
            detector,
            face_detector,
            face_scorer,
            body_scorer,
            args.type,
            feat_key,
        )
    else:
        print(f"Phases: BASELINE {int(args.baseline)}s  -> SIMULATION {int(args.positive)}s")
        if args.type == "eyelid_myoclonia":
            print("Simulation: rapid partial eyelid closures, ~3 per second.")
        phases = run_webcam(
            args,
            detector,
            face_detector,
            face_scorer,
            body_scorer,
            args.type,
            feat_key,
        )

    face_detector.close()

    neg = phases["baseline"]
    pos = phases["positive"]
    if not neg.probabilities or not pos.probabilities:
        print("Insufficient samples — make sure the face/body was in frame.")
        return 1

    neg_p = np.array(neg.probabilities)
    pos_p = np.array(pos.probabilities)
    neg_s = np.array(neg.features)
    pos_s = np.array(pos.features)
    neg_still = float(np.mean(neg.body_stillness) or 0.5)
    pos_still = float(np.mean(pos.body_stillness) or 0.5)

    print(f"\n--- BASELINE  (n={len(neg_p)}) ---")
    print(f"  {args.type}: p50={np.median(neg_p):.3f}  p95={np.percentile(neg_p,95):.3f}")
    print(f"  feature {feat_key}: p50={np.median(neg_s):.3f}  p95={np.percentile(neg_s,95):.3f}")
    print(f"  body_stillness avg = {neg_still:.3f}")

    print(f"\n--- POSITIVE  (n={len(pos_p)}) ---")
    print(f"  {args.type}: p50={np.median(pos_p):.3f}  p95={np.percentile(pos_p,95):.3f}")
    print(f"  feature {feat_key}: p50={np.median(pos_s):.3f}  p95={np.percentile(pos_s,95):.3f}")
    print(f"  body_stillness avg = {pos_still:.3f}")

    s_pos = float(np.percentile(pos_s, 50))
    s_neg = float(np.percentile(neg_s, 95))
    target_pos_l = _logit(args.target_pos)
    target_neg_l = _logit(args.target_neg)
    denom = s_pos - s_neg
    if denom < 1e-3:
        print(
            f"\nWARNING: signal during positive phase ({s_pos:.3f}) is no higher "
            f"than baseline p95 ({s_neg:.3f}). The heuristic feature isn't capturing "
            f"the target motion — either the band/gate is wrong or the positive "
            f"phase doesn't contain the expected signature."
        )
        return 1
    w_new = (target_pos_l - target_neg_l) / denom
    b_no_extras = target_pos_l - w_new * s_pos
    if args.type == "eyelid_myoclonia":
        b_new = b_no_extras - settings.w_em_stillness * ((neg_still + pos_still) / 2)
    else:
        b_new = b_no_extras

    cur_w = getattr(settings, w_attr)
    cur_b = getattr(settings, b_attr)
    print("\n=== Recommended config edits ===")
    print(f"  {w_attr}: float = {w_new:.3f}   # was {cur_w}")
    print(f"  {b_attr}: float = {b_new:.3f}   # was {cur_b}")
    print(
        f"\nPredicted: baseline p95 ≈ {args.target_neg:.2f}, "
        f"positive p50 ≈ {args.target_pos:.2f}"
    )
    print(f"Feature separation:  positive p50={s_pos:.3f}  vs  baseline p95={s_neg:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
