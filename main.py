"""Entry point — runs the MJPEG streamer FastAPI app."""
from __future__ import annotations

import logging
import sys

import uvicorn

from config import settings
from streamer import app


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.streamer_port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
