"""Bounded, gentle P0 head-roll motion; never addresses neck or jaw servos."""
from __future__ import annotations

import math
import threading
import time

from .servo_controller import ConfigurationError, ServoController, validate_config


class HeadRollAnimator:
    """Alternate only between saved P0 limits using cosine ease-in/ease-out."""
    def __init__(self, controller: ServoController, *, transition_s: float = 2.0,
                 update_s: float = .1, speed_dps: float = 6.0,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        if transition_s <= 0 or update_s <= 0 or speed_dps <= 0:
            raise ValueError("head-roll timing and speed must be positive")
        self.controller, self.transition_s = controller, transition_s
        self.update_s, self.speed_dps, self.clock, self.sleep = update_s, speed_dps, clock, sleep
        self.cancelled = threading.Event(); self.thread: threading.Thread | None = None

    def _limits(self) -> tuple[float, float]:
        cfg = self.controller.configs.get("head_roll")
        if cfg is None:
            raise ConfigurationError("head_roll is not configured")
        validate_config("head_roll", cfg, require_calibrated=True)
        assert cfg.min_angle is not None and cfg.max_angle is not None
        return cfg.min_angle, cfg.max_angle

    def _transition(self, target: float) -> None:
        start = self.controller.position("head_roll")
        if start is None:
            raise ConfigurationError("head_roll requires a trusted powered reference")
        began = self.clock(); steps = max(2, round(self.transition_s / self.update_s))
        for step in range(1, steps + 1):
            if self.cancelled.is_set(): return
            progress = step / steps
            eased = .5 - .5 * math.cos(math.pi * progress)
            position = start + (target - start) * eased
            # ServoController enforces saved min/max on every interpolated move.
            self.controller.move_to("head_roll", position, speed_dps=self.speed_dps, step=.25)
            remaining = began + step * self.transition_s / steps - self.clock()
            if remaining > 0: self.sleep(remaining)

    def run(self) -> None:
        low, high = self._limits()
        target = high if self.controller.position("head_roll") != high else low
        while not self.cancelled.is_set():
            self._transition(target)
            target = low if target == high else high

    def start(self) -> None:
        self._limits()
        if self.controller.position("head_roll") is None:
            raise ConfigurationError("head_roll requires a trusted powered reference")
        self.thread = threading.Thread(target=self.run, name="head-roll", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.cancelled.set()
        if self.thread and self.thread is not threading.current_thread(): self.thread.join(timeout=2)
