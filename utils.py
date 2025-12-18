from __future__ import annotations

import os
import time


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


class FPSLimiter:
    """Best-effort limiter to cap processing FPS."""
    def __init__(self, max_fps: float | None):
        self.max_fps = max_fps
        self._last = 0.0

    def sleep_if_needed(self) -> None:
        if not self.max_fps or self.max_fps <= 0:
            return
        now = time.time()
        if self._last == 0.0:
            self._last = now
            return
        min_dt = 1.0 / self.max_fps
        dt = now - self._last
        if dt < min_dt:
            time.sleep(min_dt - dt)
        self._last = time.time()
