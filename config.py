from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Streamer-side settings. Inference, scoring, and dashboard weights all
    live in the companion Seizure-Processor project. All settings can be
    overridden by SD_* environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="SD_", env_file=".env", extra="ignore")

    # --- camera -----------------------------------------------------------
    camera_source: str = "0"             # "0" = /dev/video0; URL also accepted
    frame_width: int = 640
    frame_height: int = 480
    # Streaming FPS — pushed at full webcam rate so consumers can subsample.
    target_fps: int = 30

    # --- streamer HTTP ---------------------------------------------------
    host: str = "0.0.0.0"
    streamer_port: int = 8080
    streamer_jpeg_quality: int = 75       # 0-100; 75 is a sane LAN default

    log_level: str = "info"

    def resolve_camera(self) -> int | str:
        if self.camera_source.isdigit():
            return int(self.camera_source)
        return self.camera_source


settings = Settings()
