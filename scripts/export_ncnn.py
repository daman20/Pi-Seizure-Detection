"""One-shot: download yolo11n-pose weights and export to NCNN for RPi inference.

Usage:
    uv run python scripts/export_ncnn.py

Writes:
    models/yolo11n-pose.pt
    models/yolo11n-pose_ncnn_model/   (directory with .param / .bin / metadata)
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    weights = models_dir / "yolo11n-pose.pt"

    if not weights.exists():
        print(f"Downloading yolo11n-pose.pt to {weights} ...")
        downloaded = YOLO("yolo11n-pose.pt")
        ckpt = Path(downloaded.ckpt_path) if hasattr(downloaded, "ckpt_path") else None
        if ckpt and ckpt.exists() and ckpt.resolve() != weights.resolve():
            weights.write_bytes(ckpt.read_bytes())
    else:
        print(f"Found existing {weights}")

    model = YOLO(str(weights))
    print("Exporting to NCNN ...")
    exported = model.export(format="ncnn", imgsz=480)
    print(f"NCNN export written to: {exported}")


if __name__ == "__main__":
    main()
