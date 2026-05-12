from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SD_", env_file=".env", extra="ignore")

    # --- camera -----------------------------------------------------------
    camera_source: str = "0"
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 10

    # --- pose model -------------------------------------------------------
    model_path: str = "models/yolo11n-pose_ncnn_model"
    fallback_model_path: str = "models/yolo11n-pose.pt"
    tracker: str = "bytetrack.yaml"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    imgsz: int = 480

    # --- face model -------------------------------------------------------
    face_enabled: bool = True
    face_model_path: str = "models/face_landmarker.task"
    face_max_num: int = 4

    # --- analysis window --------------------------------------------------
    window_seconds: float = 4.0
    min_window_seconds: float = 1.5

    # rhythmic-band of interest for clonic / tonic-clonic.
    # Widened from 3-5 to 2.5-5.5 Hz: real ictal rhythm sits at 3-5, but
    # voluntary shakes during testing tend slightly slower, and clonic phase
    # can slow toward seizure end.
    seizure_band_low: float = 2.5
    seizure_band_high: float = 5.5
    analysis_band_low: float = 0.5
    analysis_band_high: float = 8.0

    # face-specific bands. Eyelid kept at 3-6 Hz so harmonics of natural
    # blinks (0.25 Hz fundamental) don't contaminate the band — rhythmic
    # ictal flutter has a sharp 3 Hz peak that this captures cleanly.
    eyelid_band_low: float = 3.0
    eyelid_band_high: float = 6.0
    oral_band_low: float = 1.0
    oral_band_high: float = 3.0

    # Face-feature amplitude gates (signal std required for full rhythmic
    # contribution). Tuned so isolated blinks don't pass at full weight.
    eyelid_amp_gate: float = 0.15
    jaw_amp_gate: float = 0.10

    # --- weighted-group sensitivity --------------------------------------
    # Head movement is more clinically meaningful than limb movement: the
    # same rhythmic amplitude in the head produces a much stronger score
    # than in the wrists/ankles. Tune via SD_HEAD_SENSITIVITY.
    head_sensitivity: float = 3.0
    shoulders_sensitivity: float = 2.5      # shoulder shrug is a hallmark clonic motion
    arms_sensitivity: float = 1.0
    legs_sensitivity: float = 1.0
    torso_sensitivity: float = 0.5

    # Per-group amplitude gates (torso-units of motion std required for the
    # rhythmic-ratio to count at full weight). Head and shoulders are gated
    # more loosely than limbs because their absolute amplitudes are smaller
    # but more clinically meaningful.
    head_amp_gate: float = 0.02
    shoulders_amp_gate: float = 0.025
    arms_amp_gate: float = 0.05
    legs_amp_gate: float = 0.05

    # --- score smoothing --------------------------------------------------
    # Convention: smoothed = α·new + (1−α)·prev. Higher α = more reactive.
    # 0.95 → score reaches ~99% of true value within one frame.
    ema_alpha: float = 0.95
    track_ttl_seconds: float = 2.0

    # --- per-type logit weights (uncalibrated heuristic priors) ----------
    # tonic_clonic: rhythmic 3-5 Hz + bilateral motion
    w_tc_rhythmic: float = 5.0
    w_tc_motion: float = 2.0
    w_tc_bilateral: float = 1.5
    w_tc_loco: float = -3.0
    w_tc_bias: float = -2.8

    # clonic: rhythmic without bilateral requirement
    w_cl_rhythmic: float = 4.5
    w_cl_motion: float = 1.5
    w_cl_loco: float = -3.0
    w_cl_bias: float = -2.6

    # myoclonic: brief jerk bursts (high bias = false-positive resistant)
    w_my_jerk_rate: float = 5.0
    w_my_sustained: float = -3.0
    w_my_bias: float = -4.5

    # atonic: rapid drop + sustained low torso (gated by actual drop + lying posture)
    w_at_drop: float = 5.0
    w_at_low_torso: float = 3.0
    w_at_upright: float = -2.5
    w_at_bias: float = -4.700   # calibrated 2026-05-12, p95 baseline 0.234 -> 0.20

    # focal_motor: asymmetric rhythmic
    w_fm_asym: float = 4.0
    w_fm_motion: float = 1.5
    w_fm_loco: float = -2.5
    w_fm_bias: float = -2.5

    # versive: sustained head yaw
    w_ve_yaw: float = 4.5
    w_ve_variance: float = -2.0
    w_ve_bias: float = -2.4

    # eyelid_myoclonia: 3-6 Hz eyelid jerking.
    # Calibrated on a real eyelid-myoclonia recording (eyelid-seizure.MOV):
    # baseline p95 → 0.20, ictal p50 → 0.75.
    w_em_rhythmic: float = 19.928
    w_em_stillness: float = 1.0
    w_em_bias: float = -6.547

    # oral_automatism: 1-3 Hz jaw motion
    w_oa_rhythmic: float = 4.5
    w_oa_motion_var: float = 1.0
    w_oa_whole_body: float = -2.0
    w_oa_bias: float = -3.283   # calibrated 2026-05-12, p95 baseline 0.331 -> 0.20

    # hemifacial_clonic: L/R facial AU asymmetry rhythmic
    w_hf_asym: float = 5.0
    w_hf_bias: float = -3.822   # calibrated 2026-05-12, p95 baseline 0.459 -> 0.20

    # --- server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 75
    log_level: str = "info"

    # --- alerting ---------------------------------------------------------
    alert_threshold: float = 0.6

    def resolve_camera(self) -> int | str:
        if self.camera_source.isdigit():
            return int(self.camera_source)
        return self.camera_source

    def resolve_model_path(self) -> str:
        ncnn = Path(self.model_path)
        if ncnn.exists():
            return str(ncnn)
        pt = Path(self.fallback_model_path)
        if pt.exists():
            return str(pt)
        return "yolo11n-pose.pt"


settings = Settings()
