"""Servo configuration and movement primitives. No import-time movement."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

LIBRARY_MIN = -90.0
LIBRARY_MAX = 90.0


class ServoError(Exception):
    """Base error for safe servo operations."""


class ConfigurationError(ServoError):
    pass


class NotCalibratedError(ServoError):
    pass


class LimitError(ServoError):
    pass


class PositionUnknownError(ServoError):
    pass


@dataclass
class ServoConfig:
    port: str
    neutral: float | None
    direction: int
    min_angle: float | None
    max_angle: float | None
    opening_endpoint: float | None = None
    opening_direction: int | None = None

    @property
    def calibrated(self) -> bool:
        return None not in (self.neutral, self.min_angle, self.max_angle)


class Backend(Protocol):
    def write(self, port: str, angle: float) -> None: ...


class RobotHatBackend:
    """Lazy Robot HAT adapter; construction and import issue no angle command."""

    def __init__(self) -> None:
        self._servos: dict[str, object] = {}

    def write(self, port: str, angle: float) -> None:
        if port not in self._servos:
            try:
                from robot_hat import Servo
            except ImportError as exc:
                raise ServoError("robot_hat is unavailable; use --mock off-Pi") from exc
            self._servos[port] = Servo(port)
        self._servos[port].angle(angle)  # type: ignore[attr-defined]


class MockBackend:
    def __init__(self) -> None:
        self.commands: list[tuple[str, float]] = []

    def write(self, port: str, angle: float) -> None:
        self.commands.append((port, angle))
        print(f"[MOCK] {port} <- {angle:.1f} degrees (Robot HAT angle)")


def validate_config(name: str, cfg: ServoConfig, require_calibrated: bool = True) -> None:
    if cfg.direction not in (-1, 1):
        raise ConfigurationError(f"{name}: direction must be 1 or -1")
    if not isinstance(cfg.port, str) or not cfg.port.startswith("P"):
        raise ConfigurationError(f"{name}: invalid port")
    values = (cfg.neutral, cfg.min_angle, cfg.max_angle)
    if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in values):
        raise ConfigurationError(f"{name}: angles must be finite numbers or null")
    if cfg.opening_endpoint is not None and (
        not isinstance(cfg.opening_endpoint, (int, float))
        or not math.isfinite(cfg.opening_endpoint)
        or not LIBRARY_MIN <= cfg.opening_endpoint <= LIBRARY_MAX
    ):
        raise ConfigurationError(f"{name}: opening endpoint must be a finite angle from -90 to 90 or null")
    if cfg.opening_direction not in (None, -1, 1):
        raise ConfigurationError(f"{name}: opening direction must be 1, -1, or null")
    if (cfg.opening_endpoint is None) != (cfg.opening_direction is None):
        raise ConfigurationError(f"{name}: opening endpoint and direction must be saved together")
    if cfg.opening_endpoint is not None:
        if cfg.neutral is None:
            raise ConfigurationError(f"{name}: opening endpoint requires a neutral angle")
        expected_direction = 1 if cfg.opening_endpoint > cfg.neutral else -1
        if cfg.opening_endpoint == cfg.neutral or cfg.opening_direction != expected_direction:
            raise ConfigurationError(f"{name}: opening endpoint must differ from neutral and match opening direction")
    if require_calibrated and not cfg.calibrated:
        raise NotCalibratedError(f"{name} is not fully calibrated")
    if cfg.calibrated:
        assert cfg.min_angle is not None and cfg.neutral is not None and cfg.max_angle is not None
        if not (LIBRARY_MIN <= cfg.min_angle <= cfg.neutral <= cfg.max_angle <= LIBRARY_MAX):
            raise ConfigurationError(f"{name}: require -90 <= min <= neutral <= max <= 90")


def load_config(path: Path) -> dict[str, ServoConfig]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["servos"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigurationError(f"Cannot read servo configuration: {exc}") from exc
    result = {}
    for name, item in raw.items():
        try:
            cfg = ServoConfig(**item)
        except (TypeError, KeyError) as exc:
            raise ConfigurationError(f"Invalid configuration for {name}: {exc}") from exc
        validate_config(name, cfg, require_calibrated=False)
        result[name] = cfg
    return result


def save_config(path: Path, configs: dict[str, ServoConfig]) -> None:
    for name, cfg in configs.items():
        validate_config(name, cfg, require_calibrated=False)
    data = {"servos": {name: vars(cfg) for name, cfg in configs.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class ServoController:
    def __init__(self, configs: dict[str, ServoConfig], backend: Backend,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.configs = configs
        self.backend = backend
        self._sleep = sleep
        self._positions: dict[str, float] = {}

    def _get(self, name: str) -> ServoConfig:
        try:
            return self.configs[name]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown servo: {name}") from exc

    @staticmethod
    def _library_angle(cfg: ServoConfig, logical_angle: float) -> float:
        physical = logical_angle * cfg.direction
        if not LIBRARY_MIN <= physical <= LIBRARY_MAX:
            raise LimitError("Mapped Robot HAT angle is outside -90..90")
        return physical

    def establish_position(self, name: str, angle: float) -> None:
        """Issue the explicitly confirmed first absolute command."""
        cfg = self._get(name)
        if not LIBRARY_MIN <= angle <= LIBRARY_MAX:
            raise LimitError("Initial angle must be between -90 and 90")
        self.backend.write(cfg.port, self._library_angle(cfg, angle))
        self._positions[name] = float(angle)

    def adopt_confirmed_position(self, name: str, angle: float) -> None:
        """Record a user-confirmed already-commanded position without writing PWM.

        This is intentionally only for a servo that the user confirms remained
        powered and untouched since that exact command. It cannot measure or
        verify shaft position.
        """
        self._get(name)
        if not LIBRARY_MIN <= angle <= LIBRARY_MAX:
            raise LimitError("Confirmed angle must be between -90 and 90")
        self._positions[name] = float(angle)

    def move_to(self, name: str, target: float, speed_dps: float = 30.0, step: float = 1.0) -> None:
        cfg = self._get(name)
        validate_config(name, cfg)
        if name not in self._positions:
            raise PositionUnknownError("Starting position is unknown; establish it explicitly first")
        assert cfg.min_angle is not None and cfg.max_angle is not None
        if not cfg.min_angle <= target <= cfg.max_angle:
            raise LimitError(f"Target {target:g} is outside {cfg.min_angle:g}..{cfg.max_angle:g}")
        if speed_dps <= 0 or step <= 0:
            raise ValueError("Speed and step must be positive")
        current = self._positions[name]
        delta = target - current
        if delta == 0:
            return
        increment = math.copysign(min(step, abs(delta)), delta)
        delay = abs(increment) / speed_dps
        while abs(target - current) > abs(increment):
            current += increment
            self.backend.write(cfg.port, self._library_angle(cfg, current))
            self._positions[name] = current
            self._sleep(delay)
        self.backend.write(cfg.port, self._library_angle(cfg, target))
        self._positions[name] = float(target)

    def calibration_step(self, name: str, delta: int, speed_dps: float = 15.0) -> float:
        if delta not in (-2, -1, 1, 2):
            raise ValueError("Calibration steps must be 1 or 2 degrees")
        if name not in self._positions:
            raise PositionUnknownError("Establish a starting position first")
        target = self._positions[name] + delta
        if not LIBRARY_MIN <= target <= LIBRARY_MAX:
            raise LimitError("Step would exceed Robot HAT's -90..90 range")
        cfg = self._get(name)
        self.backend.write(cfg.port, self._library_angle(cfg, target))
        self._positions[name] = target
        self._sleep(abs(delta) / speed_dps)
        return target

    def move_confirmed_position(self, name: str, angle: float) -> None:
        """Move from a previously established position; caller supplies its own safe range."""
        if name not in self._positions:
            raise PositionUnknownError("Starting position is unknown; establish or adopt it first")
        cfg = self._get(name)
        if not LIBRARY_MIN <= angle <= LIBRARY_MAX:
            raise LimitError("Target must be between -90 and 90")
        self.backend.write(cfg.port, self._library_angle(cfg, angle))
        self._positions[name] = float(angle)

    def position(self, name: str) -> float | None:
        return self._positions.get(name)
