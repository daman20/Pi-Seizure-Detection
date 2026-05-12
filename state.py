from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from scorer import BODY_TYPES
from face_scorer import FACE_TYPES


HISTORY_SECONDS = 120.0
ALL_TYPES: tuple[str, ...] = tuple(BODY_TYPES) + tuple(FACE_TYPES)


@dataclass
class PersonState:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_seen_ts: float
    probabilities: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    window_seconds: float = 0.0


@dataclass
class HealthSnapshot:
    ok: bool
    fps: float
    model: str
    uptime_seconds: float
    tracked_people: int = 0


class PersonStateStore:
    def __init__(
        self,
        model_name: str,
        fps_ema_alpha: float = 0.2,
        history_seconds: float = HISTORY_SECONDS,
    ) -> None:
        self._lock = threading.RLock()
        self._frame_cond = threading.Condition(self._lock)
        self._people: dict[int, PersonState] = {}
        self._history: dict[int, deque[tuple[float, dict[str, float]]]] = {}
        self._history_seconds = history_seconds
        self._latest_jpeg: bytes | None = None
        self._jpeg_version: int = 0
        self._fps: float = 0.0
        self._fps_alpha = fps_ema_alpha
        self._last_frame_ts: float | None = None
        self._model_name = model_name
        self._start_ts = time.time()

    # writes ---------------------------------------------------------------
    def update_person(
        self,
        track_id: int,
        bbox,
        ts: float,
        probabilities: dict[str, float] | None,
        features: dict[str, float] | None = None,
        window_seconds: float = 0.0,
    ) -> None:
        with self._lock:
            existing = self._people.get(track_id)
            if existing is None:
                self._people[track_id] = PersonState(
                    track_id=track_id,
                    bbox=tuple(bbox),
                    last_seen_ts=ts,
                    probabilities=dict(probabilities or {}),
                    features=dict(features or {}),
                    window_seconds=window_seconds,
                )
            else:
                existing.bbox = tuple(bbox)
                existing.last_seen_ts = ts
                if probabilities:
                    existing.probabilities = dict(probabilities)
                if features:
                    existing.features = dict(features)
                if window_seconds:
                    existing.window_seconds = window_seconds
            if probabilities:
                hist = self._history.setdefault(track_id, deque())
                hist.append((ts, {k: float(v) for k, v in probabilities.items()}))
                cutoff = ts - self._history_seconds
                while hist and hist[0][0] < cutoff:
                    hist.popleft()

    def evict_stale(self, now: float, ttl_seconds: float) -> list[int]:
        with self._lock:
            stale = [
                tid for tid, p in self._people.items() if now - p.last_seen_ts > ttl_seconds
            ]
            for tid in stale:
                del self._people[tid]
                self._history.pop(tid, None)
            return stale

    def set_jpeg(self, jpeg: bytes) -> None:
        with self._frame_cond:
            self._latest_jpeg = jpeg
            self._jpeg_version += 1
            self._frame_cond.notify_all()

    def tick_fps(self, ts: float) -> None:
        with self._lock:
            if self._last_frame_ts is None:
                self._last_frame_ts = ts
                return
            dt = ts - self._last_frame_ts
            self._last_frame_ts = ts
            if dt <= 0:
                return
            instant = 1.0 / dt
            self._fps = (
                instant
                if self._fps == 0.0
                else self._fps_alpha * instant + (1 - self._fps_alpha) * self._fps
            )

    # reads ----------------------------------------------------------------
    def snapshot(self) -> tuple[list[PersonState], float]:
        with self._lock:
            return list(self._people.values()), self._fps

    def history_snapshot(self) -> dict[int, list[tuple[float, dict[str, float]]]]:
        with self._lock:
            return {tid: list(d) for tid, d in self._history.items()}

    @property
    def history_seconds(self) -> float:
        return self._history_seconds

    def health(self) -> HealthSnapshot:
        with self._lock:
            return HealthSnapshot(
                ok=True,
                fps=self._fps,
                model=self._model_name,
                uptime_seconds=time.time() - self._start_ts,
                tracked_people=len(self._people),
            )

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def wait_for_new_jpeg(self, last_version: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._frame_cond:
            self._frame_cond.wait_for(lambda: self._jpeg_version != last_version, timeout=timeout)
            return self._latest_jpeg, self._jpeg_version

    @property
    def jpeg_version(self) -> int:
        with self._lock:
            return self._jpeg_version
