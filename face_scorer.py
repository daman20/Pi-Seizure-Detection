from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from buffers import FaceBuffer
from scorer import TypeScores, _sigmoid


FACE_TYPES = (
    "versive",
    "eyelid_myoclonia",
    "oral_automatism",
    "hemifacial_clonic",
)


# Blendshapes that come in symmetric L/R pairs (used for hemifacial asymmetry).
SYMMETRIC_PAIRS: list[tuple[str, str]] = [
    ("mouthSmileLeft", "mouthSmileRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthPressLeft", "mouthPressRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"),
    ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("cheekSquintLeft", "cheekSquintRight"),
    ("browDownLeft", "browDownRight"),
    ("browOuterUpLeft", "browOuterUpRight"),
    ("noseSneerLeft", "noseSneerRight"),
    ("eyeBlinkLeft", "eyeBlinkRight"),
]

# All blendshapes for "flat affect" (low overall variability).
EXPRESSION_BLENDSHAPES: list[str] = [
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthOpen", "jawOpen",
    "browInnerUp", "browDownLeft", "browDownRight",
    "noseSneerLeft", "noseSneerRight",
    "cheekPuff",
    "eyeWideLeft", "eyeWideRight",
    "eyeSquintLeft", "eyeSquintRight",
]


@dataclass
class FaceScoreFeatures:
    yaw_mean_abs: float
    yaw_variance: float
    eyelid_rhythmic: float
    jaw_rhythmic: float
    asym_rhythmic: float
    face_flat: float
    blink_rate_hz: float
    low_blink: float


class FaceScorer:
    def __init__(self, settings) -> None:
        self.s = settings
        self.fps = settings.target_fps
        self.min_samples = max(4, int(settings.min_window_seconds * self.fps))
        self._prev_probs: dict[int, dict[str, float]] = {}

    # ------ helpers ------------------------------------------------------
    def _resample(self, t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t0, t1 = t[0], t[-1]
        if t1 - t0 < 1e-3:
            return t, v
        n = max(int(round((t1 - t0) * self.fps)) + 1, len(t))
        grid = np.linspace(t0, t1, n)
        return grid, np.interp(grid, t, v)

    def _band_ratio(self, signal: np.ndarray, band: tuple[float, float]) -> float:
        n = signal.shape[0]
        if n < 8:
            return 0.0
        nperseg = min(n, max(16, (n // 2) * 2))
        try:
            freqs, psd = welch(
                signal, fs=self.fps, nperseg=nperseg, detrend="linear", scaling="spectrum"
            )
        except ValueError:
            return 0.0
        bmask = (freqs >= band[0]) & (freqs <= band[1])
        amask = (freqs >= self.s.analysis_band_low) & (freqs <= self.s.analysis_band_high)
        if not bmask.any() or not amask.any():
            return 0.0
        total = psd[amask].sum()
        if total < 1e-9:
            return 0.0
        return float(psd[bmask].sum() / total)

    def _blink_rate(self, t: np.ndarray, v: np.ndarray) -> float:
        """Count blink crossings (rising edges over 0.5 threshold) per second."""
        if t.size < 4:
            return 0.0
        crossings = int(np.sum((v[:-1] < 0.5) & (v[1:] >= 0.5)))
        span = float(t[-1] - t[0])
        return crossings / span if span > 0 else 0.0

    # ------ feature extraction ------------------------------------------
    def _features(self, buf: FaceBuffer) -> FaceScoreFeatures:
        s = self.s
        t_pose, yaw, pitch, roll = buf.head_pose()
        yaw_mean_abs = float(np.mean(np.abs(yaw))) if yaw.size else 0.0
        yaw_variance = float(np.std(yaw)) if yaw.size else 0.0

        # Eyelid rhythmic — mean of L and R eyeBlink in the 3-6 Hz band, gated
        # by amplitude. Band-power *ratio* alone is noise-dominated when the
        # underlying signal is flat, so multiply by a saturating gate on the
        # signal's std so quiet faces score near zero.
        t_l, blink_l = buf.series("eyeBlinkLeft")
        _, blink_r = buf.series("eyeBlinkRight")
        if blink_l.size >= 4:
            _, rl = self._resample(t_l, blink_l)
            _, rr = self._resample(t_l, blink_r)
            band = (s.eyelid_band_low, s.eyelid_band_high)
            ratio = (self._band_ratio(rl, band) + self._band_ratio(rr, band)) / 2.0
            amp = float(max(blink_l.std(), blink_r.std()))
            gate = min(1.0, amp / s.eyelid_amp_gate)
            eyelid_rhythmic = ratio * gate
            blink_rate = self._blink_rate(t_l, (blink_l + blink_r) / 2.0)
        else:
            eyelid_rhythmic = 0.0
            blink_rate = 0.0

        # Jaw rhythmic in 1-3 Hz band, amplitude-gated the same way.
        t_j, jaw = buf.series("jawOpen")
        if jaw.size >= 4:
            _, rj = self._resample(t_j, jaw)
            ratio = self._band_ratio(rj, (s.oral_band_low, s.oral_band_high))
            amp = float(jaw.std())
            gate = min(1.0, amp / s.jaw_amp_gate)
            jaw_rhythmic = ratio * gate
        else:
            jaw_rhythmic = 0.0

        # Hemifacial asymmetry — for each L/R pair build a diff signal and
        # take the strongest rhythmic component in 1-5 Hz.
        asym_rhythmic = 0.0
        for left, right in SYMMETRIC_PAIRS:
            tl, vl = buf.series(left)
            _, vr = buf.series(right)
            if vl.size < 4:
                continue
            diff = vl - vr
            _, rd = self._resample(tl, diff)
            r = self._band_ratio(rd, (1.0, 5.0))
            # weight by total energy to avoid noise-driven ratios at near-zero amplitude
            r *= float(np.std(diff))
            if r > asym_rhythmic:
                asym_rhythmic = r
        # squash to 0..1
        asym_rhythmic = float(1.0 - math.exp(-5.0 * asym_rhythmic))

        # Flat affect — low std across expression blendshapes
        flats: list[float] = []
        for name in EXPRESSION_BLENDSHAPES:
            _, v = buf.series(name)
            if v.size >= 2:
                flats.append(float(v.std()))
        face_var = float(np.mean(flats)) if flats else 0.0
        face_flat = 1.0 / (1.0 + 20.0 * face_var)  # 1.0 = perfectly still face

        # Low-blink-rate score (kept around for any future re-add of an arrest
        # detector). Threshold at 0.05 Hz so only true blink suppression scores
        # 1.0; normal blinking 0.25-0.33 Hz scores 0.
        low_blink = float(max(0.0, min(1.0, (0.05 - blink_rate) / 0.05)))

        return FaceScoreFeatures(
            yaw_mean_abs=yaw_mean_abs,
            yaw_variance=yaw_variance,
            eyelid_rhythmic=eyelid_rhythmic,
            jaw_rhythmic=jaw_rhythmic,
            asym_rhythmic=asym_rhythmic,
            face_flat=face_flat,
            blink_rate_hz=blink_rate,
            low_blink=low_blink,
        )

    # ------ scoring ------------------------------------------------------
    def score(
        self,
        track_id: int,
        face_buf: FaceBuffer,
        body_stillness: float = 0.0,
        body_any_motion: float = 0.0,
    ) -> TypeScores | None:
        if len(face_buf) < self.min_samples or face_buf.timespan() < 0.5:
            self._prev_probs.pop(track_id, None)
            return None
        s = self.s
        f = self._features(face_buf)

        # Normalize yaw inputs into 0..1 ranges for the linear logit
        yaw_signal = max(0.0, min(1.0, (f.yaw_mean_abs - 15.0) / 45.0))     # 1.0 at ~60°
        yaw_var_norm = min(1.0, f.yaw_variance / 30.0)                      # 1.0 at high osc.

        raw = {
            "versive": _sigmoid(
                s.w_ve_yaw * yaw_signal
                + s.w_ve_variance * yaw_var_norm
                + s.w_ve_bias
            ),
            "eyelid_myoclonia": _sigmoid(
                s.w_em_rhythmic * f.eyelid_rhythmic
                + s.w_em_stillness * body_stillness
                + s.w_em_bias
            ),
            "oral_automatism": _sigmoid(
                s.w_oa_rhythmic * f.jaw_rhythmic
                + s.w_oa_motion_var * float(min(1.0, f.jaw_rhythmic * 2))
                + s.w_oa_whole_body * body_any_motion
                + s.w_oa_bias
            ),
            "hemifacial_clonic": _sigmoid(
                s.w_hf_asym * f.asym_rhythmic
                + s.w_hf_bias
            ),
        }

        prev = self._prev_probs.get(track_id, {})
        smoothed = {
            k: v if prev.get(k) is None else s.ema_alpha * v + (1 - s.ema_alpha) * prev[k]
            for k, v in raw.items()
        }
        self._prev_probs[track_id] = smoothed

        return TypeScores(
            probabilities={k: float(v) for k, v in smoothed.items()},
            features={
                "yaw_mean_abs": f.yaw_mean_abs,
                "yaw_variance": f.yaw_variance,
                "eyelid_rhythmic": f.eyelid_rhythmic,
                "jaw_rhythmic": f.jaw_rhythmic,
                "asym_rhythmic": f.asym_rhythmic,
                "face_flat": f.face_flat,
                "blink_rate_hz": f.blink_rate_hz,
                "low_blink": f.low_blink,
            },
            window_seconds=float(face_buf.timespan()),
        )

    def forget(self, track_id: int) -> None:
        self._prev_probs.pop(track_id, None)
