"""Reusable, endpoint-bounded controls for the two jaw servos."""
from __future__ import annotations

import math
import time
import logging
from typing import Callable

from .servo_controller import ConfigurationError, ServoController

JAW_NAMES = ("upper_jaw", "lower_jaw")


class MouthController:
    """Move calibrated jaws together; it never commands neck/head servos."""

    def __init__(self, controller: ServoController, sleep: Callable[[float], None] = time.sleep) -> None:
        self.controller = controller
        self._sleep = sleep
        self.command_log: list[dict[str, float]] = []
        self._logger = logging.getLogger(__name__)

    def _jaw(self, name: str):
        if name not in JAW_NAMES:
            raise ConfigurationError(f"Not a mouth servo: {name}")
        cfg = self.controller.configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None or cfg.opening_direction is None:
            raise ConfigurationError(f"{name} opening endpoint is not calibrated")
        return cfg

    def target_angle(self, name: str, amount: float) -> float:
        if not 0.0 <= amount <= 1.0:
            raise ValueError("openness must be between 0.0 and 1.0")
        cfg = self._jaw(name)
        assert cfg.neutral is not None and cfg.opening_endpoint is not None
        return cfg.neutral + (cfg.opening_endpoint - cfg.neutral) * amount

    def set_openness(self, amount: float, speed_dps: float = 10.0) -> None:
        """Set both jaws to 0.0 closed through 1.0 fully open, smoothly."""
        if speed_dps <= 0:
            raise ValueError("speed must be positive")
        targets = {name: self.target_angle(name, amount) for name in JAW_NAMES}
        currents = {name: self.controller.position(name) for name in JAW_NAMES}
        if any(value is None for value in currents.values()):
            raise ConfigurationError("Both jaw starting positions must be explicitly established")
        max_delta = max(abs(targets[name] - currents[name]) for name in JAW_NAMES)  # type: ignore[operator]
        if max_delta == 0:
            return
        steps = max(1, math.ceil(max_delta))
        for index in range(1, steps + 1):
            fraction = index / steps
            for name in JAW_NAMES:
                current = currents[name]
                assert current is not None
                target = current + (targets[name] - current) * fraction
                self.controller.move_confirmed_position(name, target)
            self._sleep(1.0 / speed_dps)

    def set_openness_step(self, amount: float) -> None:
        """Write one already-rate-limited synchronized mouth target without sleeping.

        AdaptiveLipSynchronizer supplies the acceleration and speed profile;
        this method only maps the safe 0..1 openness fraction to calibrated
        jaw endpoints.
        """
        targets = {name: self.target_angle(name, amount) for name in JAW_NAMES}
        if any(self.controller.position(name) is None for name in JAW_NAMES):
            raise ConfigurationError("Both jaw starting positions must be explicitly established")
        started = time.monotonic()
        upper_started = time.monotonic()
        self.controller.move_confirmed_position("upper_jaw", targets["upper_jaw"])
        upper_completed = time.monotonic()
        lower_started = time.monotonic()
        self.controller.move_confirmed_position("lower_jaw", targets["lower_jaw"])
        lower_completed = time.monotonic()
        completed = time.monotonic()
        record = {"timestamp": completed, "command_started_at": started, "command_completed_at": completed,
                  "command_duration_ms": (completed - started) * 1000, "openness": amount,
                  "upper_write_ms": (upper_completed - upper_started) * 1000,
                  "lower_write_ms": (lower_completed - lower_started) * 1000,
                  "upper_jaw_angle": targets["upper_jaw"], "lower_jaw_angle": targets["lower_jaw"]}
        self.command_log.append(record)
        self._logger.debug("jaw command timestamp=%.6f openness=%.4f upper=%.3f lower=%.3f (commanded, not measured)",
                           record["timestamp"], amount, record["upper_jaw_angle"], record["lower_jaw_angle"])

    def open(self, amount: float = 0.5, speed_dps: float = 10.0) -> None:
        self.set_openness(amount, speed_dps)

    def close(self, speed_dps: float = 10.0) -> None:
        self.set_openness(0.0, speed_dps)
