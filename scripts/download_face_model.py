"""Download the MediaPipe Face Landmarker model bundle.

Usage:
    uv run python scripts/download_face_model.py

Writes:
    models/face_landmarker.task
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def main() -> int:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    out = models_dir / "face_landmarker.task"
    if out.exists():
        print(f"Already present: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0
    print(f"Downloading {MODEL_URL} -> {out}")
    urllib.request.urlretrieve(MODEL_URL, out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
