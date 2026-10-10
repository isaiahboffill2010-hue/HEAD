"""Playback-clock-driven, adaptive lip synchronization for calibrated jaws."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from .mouth import JAW_NAMES
from .servo_controller import ConfigurationError

LOGGER = logging.getLogger(__name__)
DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / "config" / "mouth_motion.json"


class SteppedMouth(Protocol):
    def set_openness_step(self, amount: float) -> None: ...


@dataclass(frozen=True)
class MotionSettings:
    conservative_speed_dps: float = 8.0
    hard_software_speed_cap_dps: float = 60.0
    acceleration_dps2: float = 240.0
    update_interval_s: float = 0.02
    verified_practical_speed_dps: float | None = None

    def speed(self, requested: float | None = None) -> float:
        # Once hardware has been supervised and accepted, normal speech should
        # actually use that result.  Before calibration, retain the cautious
        # fallback rather than guessing what a particular linkage can handle.
        value = (self.verified_practical_speed_dps or self.conservative_speed_dps) if requested is None else requested
        if not 0 < value <= self.hard_software_speed_cap_dps:
            raise ValueError(f"speed must be > 0 and <= {self.hard_software_speed_cap_dps:g} deg/s")
        if self.verified_practical_speed_dps is not None and value > self.verified_practical_speed_dps:
            raise ValueError("speed exceeds manually verified practical speed")
        return value


def load_motion_settings(path: Path = DEFAULT_SETTINGS) -> MotionSettings:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = MotionSettings(**data)
        settings.speed()
        if settings.acceleration_dps2 <= 0 or settings.update_interval_s <= 0:
            raise ValueError("acceleration and update interval must be positive")
        return settings
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid mouth motion settings: {exc}") from exc


def save_motion_settings(settings: MotionSettings, path: Path = DEFAULT_SETTINGS) -> None:
    """Atomically save validated motion settings."""
    settings.speed()
    if settings.acceleration_dps2 <= 0 or settings.update_interval_s <= 0:
        raise ValueError("acceleration and update interval must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(settings), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class TimingRecord:
    scheduled_time: float
    command_time: float
    drift_s: float
    openness: float
    skipped: bool = False
    amplitude_reduced: bool = False


class AdaptiveLipSynchronizer:
    """Follow the *actual playback clock*, never a pre-queued servo loop.

    Call ``run(events, playback_clock)`` when audio starts. ``playback_clock``
    must return elapsed audio-playback seconds from that same start instant.
    """

    def __init__(self, mouth: SteppedMouth, max_jaw_travel_deg: float,
                 settings: MotionSettings | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 logger: logging.Logger = LOGGER) -> None:
        if max_jaw_travel_deg <= 0:
            raise ValueError("max_jaw_travel_deg must be positive")
        self.mouth = mouth
        self.max_jaw_travel_deg = max_jaw_travel_deg
        self.settings = settings or load_motion_settings()
        self._sleep = sleep
        self._logger = logger
        self.records: list[TimingRecord] = []

    @staticmethod
    def validate(events: Sequence[Mapping[str, float]]) -> None:
        previous = -1.0
        for event in events:
            if set(event) != {"time", "openness"}:
                raise ValueError("each timing event must contain only time and openness")
            event_time, openness = event["time"], event["openness"]
            if event_time < 0 or event_time < previous:
                raise ValueError("timing events must be ordered and non-negative")
            if not 0.0 <= openness <= 1.0:
                raise ValueError("openness must be between 0.0 and 1.0")
            previous = event_time

    def adapt(self, events: Sequence[Mapping[str, float]], speed_dps: float) -> list[dict[str, float]]:
        """Reduce changes that cannot fit their speech interval at safe speed."""
        self.validate(events)
        speed_fraction = speed_dps / self.max_jaw_travel_deg
        prior_time, prior_open = 0.0, 0.0
        adapted: list[dict[str, float]] = []
        for event in events:
            allowed = speed_fraction * (event["time"] - prior_time)
            change = event["openness"] - prior_open
            if abs(change) > allowed:
                openness = prior_open + (allowed if change > 0 else -allowed)
            else:
                openness = event["openness"]
            openness = min(1.0, max(0.0, openness))
            adapted.append({"time": event["time"], "openness": openness})
            prior_time, prior_open = event["time"], openness
        return adapted

    def run(self, events: Sequence[Mapping[str, float]], playback_clock: Callable[[], float],
            speed_dps: float | None = None, start_openness: float = 0.0) -> list[TimingRecord]:
        """Synchronize to playback, skipping stale targets instead of chasing them late."""
        if not 0.0 <= start_openness <= 1.0:
            raise ValueError("start_openness must be between 0.0 and 1.0")
        speed = self.settings.speed(speed_dps)
        # Do not pre-shrink targets here. The real-time rate/acceleration
        # controller below already limits every command. Pre-adaptation made
        # short phonemes tiny and then acceleration limited them a second time.
        self.validate(events)
        planned = [dict(event) for event in events]
        if not planned:
            return []
        self.records = []
        event_index = 0
        target = start_openness
        position = start_openness
        velocity = 0.0  # openness fraction per second
        last_clock = playback_clock()
        final_time = planned[-1]["time"]
        # Permit the final target (normally closed) to settle after audio ends.
        # This is bounded so a stalled/non-advancing clock cannot run forever.
        settle_deadline = final_time + (self.max_jaw_travel_deg / speed) + (
            speed / self.settings.acceleration_dps2) + self.settings.update_interval_s
        next_tick = last_clock
        while True:
            now = playback_clock()
            elapsed = now
            due_start = event_index
            while event_index < len(planned) and planned[event_index]["time"] <= elapsed:
                target = planned[event_index]["openness"]
                event_index += 1
            if event_index > due_start:
                # Earlier simultaneous/expired targets are deliberately skipped.
                for index in range(due_start, event_index):
                    event = planned[index]
                    skipped = index < event_index - 1
                    # Real-time telemetry: this event demanded more travel than
                    # had actually been achieved when it became due.
                    reduced = abs(event["openness"] - position) >= 0.0001
                    record = TimingRecord(event["time"], elapsed, elapsed - event["time"],
                                          event["openness"], skipped, reduced)
                    self.records.append(record)
                    self._logger.debug("mouth timing scheduled=%.3f command=%.3f drift=%+.3f skipped=%s reduced=%s",
                                       record.scheduled_time, record.command_time, record.drift_s,
                                       record.skipped, record.amplitude_reduced)
            dt = max(0.001, now - last_clock)
            last_clock = now
            speed_fraction = speed / self.max_jaw_travel_deg
            accel_fraction = self.settings.acceleration_dps2 / self.max_jaw_travel_deg
            error = target - position
            desired_velocity = max(-speed_fraction, min(speed_fraction, error / dt))
            max_velocity_change = accel_fraction * dt
            velocity += max(-max_velocity_change, min(max_velocity_change, desired_velocity - velocity))
            next_position = position + velocity * dt
            if (error >= 0 and next_position > target) or (error <= 0 and next_position < target):
                next_position, velocity = target, 0.0
            if abs(next_position - position) >= 0.0001:
                self.mouth.set_openness_step(min(1.0, max(0.0, next_position)))
                position = next_position
            if elapsed >= final_time and (abs(target - position) < 0.0001 or elapsed >= settle_deadline):
                break
            # Keep command times on playback deadlines. Servo/I2C write time is
            # part of the interval, not extra delay added after every update.
            next_tick += self.settings.update_interval_s
            remaining = next_tick - playback_clock()
            if remaining > 0:
                self._sleep(remaining)
        return list(self.records)


def calibrated_max_jaw_travel(configs: Mapping[str, object]) -> float:
    travels = []
    for name in JAW_NAMES:
        cfg = configs[name]
        neutral = getattr(cfg, "neutral")
        endpoint = getattr(cfg, "opening_endpoint")
        if neutral is None or endpoint is None:
            raise ConfigurationError(f"{name} opening endpoint is not calibrated")
        travels.append(abs(endpoint - neutral))
    return max(travels)
