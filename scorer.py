from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from buffers import (
    KEYPOINTS_ARMS,
    KEYPOINTS_ARMS_L,
    KEYPOINTS_ARMS_R,
    KEYPOINTS_HEAD,
    KEYPOINTS_LEGS,
    KEYPOINTS_LEGS_L,
    KEYPOINTS_LEGS_R,
    KEYPOINTS_SHOULDERS,
    NOSE,
    KeypointBuffer,
    normalize_keypoints,
)


JOINT_CONF_MIN = 0.3

BODY_TYPES = (
    "tonic_clonic",
    "clonic",
    "myoclonic",
    "atonic",
    "focal_motor",
)


@dataclass
class TypeScores:
    probabilities: dict[str, float]
    features: dict[str, float]
    window_seconds: float

    @property
    def max_probability(self) -> float:
        return max(self.probabilities.values()) if self.probabilities else 0.0

    @property
    def dominant_type(self) -> str | None:
        if not self.probabilities:
            return None
        return max(self.probabilities, key=lambda k: self.probabilities[k])


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class BodyScorer:
    def __init__(self, settings) -> None:
        self.s = settings
        self.fps = settings.target_fps
        self.min_samples = max(4, int(settings.min_window_seconds * self.fps))
        self._prev_probs: dict[int, dict[str, float]] = {}

    # ------ helpers ------------------------------------------------------
    def _resample(self, times: np.ndarray, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t0, t1 = times[0], times[-1]
        if t1 - t0 < 1e-3:
            return times, series
        n = max(int(round((t1 - t0) * self.fps)) + 1, len(times))
        grid = np.linspace(t0, t1, n)
        if series.ndim == 1:
            return grid, np.interp(grid, times, series)
        return grid, np.stack(
            [np.interp(grid, times, series[:, k]) for k in range(series.shape[1])], axis=1
        )

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

    def _group_rhythmic(
        self, times: np.ndarray, norm: np.ndarray, conf: np.ndarray, joints: list[int]
    ) -> float:
        band = (self.s.seizure_band_low, self.s.seizure_band_high)
        ratios: list[float] = []
        for j in joints:
            if float(conf[:, j].mean()) < JOINT_CONF_MIN:
                continue
            series = norm[:, j, :]
            _, resampled = self._resample(times, series)
            for ax in range(2):
                ratios.append(self._band_ratio(resampled[:, ax], band))
        return float(np.mean(ratios)) if ratios else 0.0

    def _group_motion(self, norm: np.ndarray, joints: list[int]) -> float:
        if not joints:
            return 0.0
        return float(norm[:, joints, :].std(axis=0).mean())

    @staticmethod
    def _cos_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        return float(np.dot(ba, bc) / denom)

    # ------ feature extraction ------------------------------------------
    def _features(self, buf: KeypointBuffer) -> dict[str, float]:
        times, kxy, kconf, bbox = buf.as_arrays()
        norm = normalize_keypoints(kxy, kconf, bbox)
        s = self.s

        # Per-group rhythmic 3-5 Hz power AND amplitude. Band-power-ratio alone
        # is dominated by noise when the signal is quiet, so each group's
        # rhythmic contribution is gated by its motion amplitude (std).
        head_r = self._group_rhythmic(times, norm, kconf, KEYPOINTS_HEAD)
        shoulders_r = self._group_rhythmic(times, norm, kconf, KEYPOINTS_SHOULDERS)
        arms_r = self._group_rhythmic(times, norm, kconf, KEYPOINTS_ARMS)
        legs_r = self._group_rhythmic(times, norm, kconf, KEYPOINTS_LEGS)
        arm_l = self._group_rhythmic(times, norm, kconf, KEYPOINTS_ARMS_L)
        arm_r = self._group_rhythmic(times, norm, kconf, KEYPOINTS_ARMS_R)
        leg_l = self._group_rhythmic(times, norm, kconf, KEYPOINTS_LEGS_L)
        leg_r = self._group_rhythmic(times, norm, kconf, KEYPOINTS_LEGS_R)

        head_amp = self._group_motion(norm, KEYPOINTS_HEAD)
        shoulders_amp = self._group_motion(norm, KEYPOINTS_SHOULDERS)
        arms_amp = self._group_motion(norm, KEYPOINTS_ARMS)
        legs_amp = self._group_motion(norm, KEYPOINTS_LEGS)
        head_r *= min(1.0, head_amp / s.head_amp_gate)
        shoulders_r *= min(1.0, shoulders_amp / s.shoulders_amp_gate)
        arms_r *= min(1.0, arms_amp / s.arms_amp_gate)
        legs_r *= min(1.0, legs_amp / s.legs_amp_gate)
        arm_l *= min(1.0, self._group_motion(norm, KEYPOINTS_ARMS_L) / s.arms_amp_gate)
        arm_r *= min(1.0, self._group_motion(norm, KEYPOINTS_ARMS_R) / s.arms_amp_gate)
        leg_l *= min(1.0, self._group_motion(norm, KEYPOINTS_LEGS_L) / s.legs_amp_gate)
        leg_r *= min(1.0, self._group_motion(norm, KEYPOINTS_LEGS_R) / s.legs_amp_gate)

        # Weighted rhythmic — head and shoulders get sensitivity boosts.
        # MAX of weighted group contributions so any one rhythmic axis can
        # drive the score (shoulder shrug alone counts as clonic motion).
        rhythmic_weighted = max(
            s.head_sensitivity * head_r,
            s.shoulders_sensitivity * shoulders_r,
            s.arms_sensitivity * arms_r,
            s.legs_sensitivity * legs_r,
        )

        # Bilateral symmetry (tonic-clonic likes L ≈ R), gated by actual power
        def sym(l: float, r: float) -> float:
            return 1.0 - abs(l - r) / (l + r + 1e-6)

        def asym(l: float, r: float) -> float:
            return abs(l - r) / (l + r + 1e-6)

        bilateral = (sym(arm_l, arm_r) + sym(leg_l, leg_r)) / 2.0
        bilateral *= max(arms_r, legs_r)

        asym_score = max(
            asym(arm_l, arm_r) * max(arm_l, arm_r),
            asym(leg_l, leg_r) * max(leg_l, leg_r),
        )

        # Per-group motion variance (weighted)
        head_m = self._group_motion(norm, KEYPOINTS_HEAD)
        shoulders_m = self._group_motion(norm, KEYPOINTS_SHOULDERS)
        arms_m = self._group_motion(norm, KEYPOINTS_ARMS)
        legs_m = self._group_motion(norm, KEYPOINTS_LEGS)
        w_sum = s.head_sensitivity + s.shoulders_sensitivity + s.arms_sensitivity + s.legs_sensitivity
        weighted_motion = (
            s.head_sensitivity * head_m
            + s.shoulders_sensitivity * shoulders_m
            + s.arms_sensitivity * arms_m
            + s.legs_sensitivity * legs_m
        ) / w_sum
        any_motion = max(head_m, shoulders_m, arms_m, legs_m)
        stillness = 1.0 / (1.0 + 10.0 * any_motion)

        # Joint extension (average over recent frames). cos≈-1 means fully extended.
        recent = max(1, min(len(buf), int(self.fps)))  # last ~1 sec
        ext_vals: list[float] = []
        for i in range(len(buf) - recent, len(buf)):
            frame = kxy[i]
            conf = kconf[i]
            for a, b, c in [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]:
                if min(conf[a], conf[b], conf[c]) < JOINT_CONF_MIN:
                    continue
                ca = self._cos_angle(frame[a], frame[b], frame[c])
                ext_vals.append(max(0.0, -ca))  # 0=bent, 1=fully extended
        extension = float(np.mean(ext_vals)) if ext_vals else 0.0

        # Locomotion
        cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
        cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        bbox_h = bbox[:, 3] - bbox[:, 1]
        bbox_w = bbox[:, 2] - bbox[:, 0]
        bbox_diag = float(np.median(np.hypot(bbox_w, bbox_h)))
        path = float(np.sum(np.hypot(np.diff(cx), np.diff(cy))))
        window = max(buf.timespan(), 1e-3)
        locomotion = path / (max(bbox_diag, 1.0) * window)

        # Drop velocity: peak downward velocity of head over a 3-frame window,
        # normalized by torso height. Smoothing keeps single-frame jitter from
        # firing the atonic detector at rest.
        head_y = kxy[:, NOSE, 1]
        ref_h = float(np.median(bbox_h))
        if len(times) >= 4 and ref_h > 1.0:
            window_n = min(3, len(head_y) - 1)
            dy = (head_y[window_n:] - head_y[:-window_n]) / ref_h
            dt = times[window_n:] - times[:-window_n]
            v = dy / np.maximum(dt, 1e-3)
            drop_raw = float(np.max(v)) if v.size else 0.0
        else:
            drop_raw = 0.0
        # Sigmoid centered so a 2x-torso-height/sec downward burst → ~0.7
        drop = 1.0 / (1.0 + math.exp(-(drop_raw - 2.0) * 2.0))

        # Bbox aspect — wide+short = lying
        aspect = bbox_w / np.maximum(bbox_h, 1.0)
        low_torso = float(np.mean(aspect > 1.2))
        upright = float(np.mean(aspect < 0.9))

        # Jerk burst rate. We need (a) a high z-score peak AND (b) the absolute
        # speed to be a meaningful fraction of torso — z-score alone fires on
        # quiet noise. Weighted by group so head jerks count more than wrist.
        jerks_weighted = 0.0
        for group, weight in (
            (KEYPOINTS_HEAD, s.head_sensitivity),
            (KEYPOINTS_SHOULDERS, s.shoulders_sensitivity),
            (KEYPOINTS_ARMS, s.arms_sensitivity),
            (KEYPOINTS_LEGS, s.legs_sensitivity),
        ):
            count = 0
            checked = 0
            for j in group:
                if float(kconf[:, j].mean()) < JOINT_CONF_MIN:
                    continue
                series = norm[:, j, :]
                sp = np.linalg.norm(np.diff(series, axis=0), axis=1)
                if sp.size < 4:
                    continue
                mu = sp.mean()
                sd = sp.std() + 1e-6
                z = (sp - mu) / sd
                # Require both z>3.5 AND magnitude > 0.05 (5% of torso per frame)
                peaks = ((z > 3.5) & (sp > 0.05)).sum()
                count += int(peaks)
                checked += 1
            if checked:
                jerks_weighted += weight * (count / checked) / window
        jerk_norm = 1.0 / (1.0 + math.exp(-(jerks_weighted - 1.0) * 3.0))

        return {
            "head_rhythmic": head_r,
            "shoulders_rhythmic": shoulders_r,
            "arms_rhythmic": arms_r,
            "legs_rhythmic": legs_r,
            "rhythmic_weighted": rhythmic_weighted,
            "bilateral": bilateral,
            "asym": asym_score,
            "head_motion": head_m,
            "shoulders_motion": shoulders_m,
            "arms_motion": arms_m,
            "legs_motion": legs_m,
            "weighted_motion": weighted_motion,
            "any_motion": any_motion,
            "stillness": stillness,
            "extension": extension,
            "locomotion": locomotion,
            "drop": drop,
            "low_torso": low_torso,
            "upright": upright,
            "jerk_rate": jerk_norm,
        }

    # ------ scoring ------------------------------------------------------
    def score(self, track_id: int, buf: KeypointBuffer) -> TypeScores | None:
        if len(buf) < self.min_samples or buf.timespan() < 0.5:
            self._prev_probs.pop(track_id, None)
            return None
        s = self.s
        f = self._features(buf)

        raw = {
            "tonic_clonic": _sigmoid(
                s.w_tc_rhythmic * f["rhythmic_weighted"]
                + s.w_tc_motion * f["weighted_motion"]
                + s.w_tc_bilateral * f["bilateral"]
                + s.w_tc_loco * f["locomotion"]
                + s.w_tc_bias
            ),
            "clonic": _sigmoid(
                s.w_cl_rhythmic * f["rhythmic_weighted"]
                + s.w_cl_motion * f["weighted_motion"]
                + s.w_cl_loco * f["locomotion"]
                + s.w_cl_bias
            ),
            "myoclonic": _sigmoid(
                s.w_my_jerk_rate * f["jerk_rate"]
                + s.w_my_sustained * f["weighted_motion"]
                + s.w_my_bias
            ),
            "atonic": _sigmoid(
                s.w_at_drop * f["drop"]
                + s.w_at_low_torso * f["low_torso"]
                + s.w_at_upright * f["upright"]
                + s.w_at_bias
            ),
            "focal_motor": _sigmoid(
                s.w_fm_asym * f["asym"]
                + s.w_fm_motion * f["weighted_motion"]
                + s.w_fm_loco * f["locomotion"]
                + s.w_fm_bias
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
            features={k: float(v) for k, v in f.items()},
            window_seconds=float(buf.timespan()),
        )

    def forget(self, track_id: int) -> None:
        self._prev_probs.pop(track_id, None)
