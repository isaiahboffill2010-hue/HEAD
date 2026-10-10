"""Timed, speech-like mouth animation independent of hardware setup."""
from __future__ import annotations

import time
from typing import Callable, Mapping, Protocol, Sequence


class MouthMotion(Protocol):
    def set_openness(self, amount: float, speed_dps: float = 10.0) -> None: ...


class MouthAnimationEngine:
    """Play externally supplied timing events through a bounded mouth controller."""

    def __init__(self, mouth: MouthMotion, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.mouth = mouth
        self._sleep = sleep
        self._clock = clock

    @staticmethod
    def validate(events: Sequence[Mapping[str, float]]) -> None:
        previous = -1.0
        for event in events:
            if set(event) != {"time", "openness"}:
                raise ValueError("each event must contain only time and openness")
            event_time, openness = event["time"], event["openness"]
            if event_time < 0 or event_time < previous:
                raise ValueError("event times must be non-negative and ordered")
            if not 0.0 <= openness <= 1.0:
                raise ValueError("event openness must be between 0.0 and 1.0")
            previous = event_time
        if not events or events[-1]["openness"] != 0.0:
            raise ValueError("animation must end with mouth closed (openness 0.0)")

    def animate(self, events: Sequence[Mapping[str, float]], speed_dps: float = 8.0) -> None:
        self.validate(events)
        if speed_dps <= 0:
            raise ValueError("speed must be positive")
        started = self._clock()
        for event in events:
            remaining = event["time"] - (self._clock() - started)
            if remaining > 0:
                self._sleep(remaining)
            # MouthController enforces each jaw's calibrated endpoint on every call.
            self.mouth.set_openness(event["openness"], speed_dps)
