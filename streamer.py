"""Minimal MJPEG webcam streamer for the Pi side of the seizure-detection
system. All inference (YOLO + MediaPipe + scoring + dashboard) lives in the
companion Seizure-Processor service on a separate x86_64 Linux box. This
module's only job: webcam → JPEG → /stream.mjpg.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from camera import Camera
from config import settings

log = logging.getLogger(__name__)
MJPEG_BOUNDARY = "frame"


class FrameBus:
    """Single-slot latest-frame fanout. Producer is the capture thread;
    consumers are any number of in-flight /stream.mjpg clients."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._latest: bytes | None = None
        self._version: int = 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._latest = jpeg
            self._version += 1
            self._cond.notify_all()

    def wait_for_new(self, last_version: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._cond:
            self._cond.wait_for(lambda: self._version != last_version, timeout=timeout)
            return self._latest, self._version

    @property
    def version(self) -> int:
        with self._cond:
            return self._version


class CaptureWorker(threading.Thread):
    def __init__(self, camera: Camera, bus: FrameBus, quality: int) -> None:
        super().__init__(daemon=True, name="capture")
        self.camera = camera
        self.bus = bus
        self.quality = quality
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cv2.setNumThreads(2)
        while not self._stop.is_set():
            ok, frame = self.camera.read()
            if not ok or frame is None:
                continue
            ok2, jpeg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if ok2:
                self.bus.publish(jpeg.tobytes())
        self.camera.close()
        log.info("Capture worker stopped")


bus = FrameBus()
_worker: CaptureWorker | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _worker
    camera = Camera(
        source=settings.resolve_camera(),
        width=settings.frame_width,
        height=settings.frame_height,
        fps=settings.target_fps,
    )
    _worker = CaptureWorker(camera, bus, settings.streamer_jpeg_quality)
    _worker.start()
    log.info(
        "Streamer ready: source=%s %dx%d @ %dfps  jpeg_q=%d",
        settings.camera_source,
        settings.frame_width,
        settings.frame_height,
        settings.target_fps,
        settings.streamer_jpeg_quality,
    )
    try:
        yield
    finally:
        if _worker is not None:
            _worker.stop()
            _worker.join(timeout=2.0)


app = FastAPI(title="Pi Seizure Detection — webcam streamer", version="0.2.0", lifespan=lifespan)


@app.get("/stream.mjpg")
def stream() -> StreamingResponse:
    async def gen():
        last = -1
        loop = asyncio.get_event_loop()
        while True:
            jpeg, last = await loop.run_in_executor(None, bus.wait_for_new, last, 1.0)
            if jpeg is None:
                continue
            yield (
                b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "frames_served": bus.version}


@app.get("/", include_in_schema=False)
def root() -> Response:
    html = (
        "<!doctype html><html><body style='background:#0e0e10;color:#e6e6e6;"
        "font-family:-apple-system,system-ui,sans-serif;margin:0;padding:16px;'>"
        "<h2 style='margin:0 0 8px;'>Pi-Seizure-Detection — webcam streamer</h2>"
        f"<p style='color:#888;font-family:ui-monospace,monospace;font-size:12px;'>"
        f"source={settings.camera_source}  "
        f"{settings.frame_width}x{settings.frame_height}@{settings.target_fps}fps  "
        f"jpeg_q={settings.streamer_jpeg_quality}</p>"
        "<ul style='font-family:ui-monospace,monospace;font-size:13px;'>"
        "<li><a style='color:#8cf' href='/stream.mjpg'>/stream.mjpg</a></li>"
        "<li><a style='color:#8cf' href='/healthz'>/healthz</a></li>"
        "</ul>"
        "<img src='/stream.mjpg' style='max-width:100%;border-radius:6px;background:#000;'/>"
        "</body></html>"
    )
    return Response(html, media_type="text/html")
