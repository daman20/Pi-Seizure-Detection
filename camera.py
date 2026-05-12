from __future__ import annotations

import logging
import platform
import time

import cv2

log = logging.getLogger(__name__)


class Camera:
    def __init__(
        self,
        source: int | str,
        width: int,
        height: int,
        fps: int,
        max_consecutive_failures: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.max_consecutive_failures = max_consecutive_failures
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0

    def _backend(self) -> int:
        if not isinstance(self.source, int):
            return cv2.CAP_ANY
        system = platform.system()
        if system == "Linux":
            return cv2.CAP_V4L2
        if system == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def open(self) -> None:
        if self._cap is not None:
            self._cap.release()
        cap = cv2.VideoCapture(self.source, self._backend())
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source {self.source!r}")
        self._cap = cap
        self._failures = 0
        log.info("Camera opened (source=%s, %dx%d @ %d fps)", self.source, self.width, self.height, self.fps)

    def read(self):
        if self._cap is None:
            self.open()
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= self.max_consecutive_failures:
                log.warning("Camera read failed %d times, reopening", self._failures)
                time.sleep(0.5)
                try:
                    self.open()
                except RuntimeError as exc:
                    log.error("Camera reopen failed: %s", exc)
                    time.sleep(1.0)
            return False, None
        self._failures = 0
        return True, frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
