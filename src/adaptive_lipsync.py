"""Playback-clock-driven, adaptive lip synchronization for calibrated jaws."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
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
    hard_software_speed_cap_dps: float = 20.0
    acceleration_dps2: float = 24.0
    update_interval_s: float = 0.05
    verified_practical_speed_dps: float | None = None

    def speed(self, requested: float | None = None) -> float:
        value = self.conservative_speed_dps if requested is None else requested
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
        planned = self.adapt(events, speed)
        if not planned:
            return []
        self.records = []
        event_index = 0
        target = start_openness
        position = start_openness
        velocity = 0.0  # openness fraction per second
        last_clock = playback_clock()
        final_time = planned[-1]["time"]
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
                    reduced = event["openness"] != events[index]["openness"]
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
            # At the final playback timestamp, do not chase a stale final target.
            if elapsed >= final_time:
                break
            self._sleep(self.settings.update_interval_s)
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
