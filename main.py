from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import uvicorn

from api import create_app
from camera import Camera
from config import settings
from detector import Detector
from face_detector import FaceDetector
from face_scorer import FaceScorer
from scorer import BodyScorer
from state import PersonStateStore
from worker import Worker


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def main() -> int:
    configure_logging(settings.log_level)
    log = logging.getLogger("main")

    model_path = settings.resolve_model_path()
    log.info("Using pose model: %s", model_path)

    camera = Camera(
        source=settings.resolve_camera(),
        width=settings.frame_width,
        height=settings.frame_height,
        fps=settings.target_fps,
    )
    detector = Detector(
        model_path=model_path,
        tracker=settings.tracker,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        imgsz=settings.imgsz,
    )
    body_scorer = BodyScorer(settings)

    face_detector: FaceDetector | None = None
    face_scorer: FaceScorer | None = None
    if settings.face_enabled:
        if Path(settings.face_model_path).exists():
            try:
                face_detector = FaceDetector(
                    model_path=settings.face_model_path,
                    max_faces=settings.face_max_num,
                )
                face_scorer = FaceScorer(settings)
                log.info("Face stage enabled")
            except Exception:
                log.exception("Face stage failed to initialize — continuing body-only")
                face_detector = None
                face_scorer = None
        else:
            log.warning(
                "Face model %s missing — run scripts/download_face_model.py. Continuing body-only.",
                settings.face_model_path,
            )

    store = PersonStateStore(model_name=model_path)
    worker = Worker(
        settings=settings,
        camera=camera,
        detector=detector,
        body_scorer=body_scorer,
        face_scorer=face_scorer,
        face_detector=face_detector,
        store=store,
    )

    def _stop(signum, _frame):
        log.info("Signal %s received, shutting down", signum)
        worker.stop()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    worker.start()
    app = create_app(store, settings)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )

    worker.stop()
    worker.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
