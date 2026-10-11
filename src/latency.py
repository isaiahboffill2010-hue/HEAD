"""Per-turn monotonic latency measurements."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LatencyTrace:
    clock: Callable[[], float] = time.monotonic
    started: float = field(init=False)
    marks: dict[str, float] = field(default_factory=dict)
    values_ms: dict[str, float] = field(default_factory=dict)
    estimated: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.started = self.clock()
        self.marks["enter"] = self.started

    def mark(self, name: str) -> float:
        value = self.clock()
        self.marks.setdefault(name, value)
        return value

    def span(self, name: str, start: str, end: str) -> None:
        if start in self.marks and end in self.marks:
            self.values_ms[name] = (self.marks[end] - self.marks[start]) * 1000

    def add_ms(self, name: str, value: float) -> None:
        self.values_ms[name] = self.values_ms.get(name, 0.0) + value

    def finish(self, bluetooth_latency_ms: float) -> None:
        self.span("Gemini first token", "text_request", "text_first_token")
        self.span("Gemini complete text", "text_request", "text_complete")
        self.span("Gemini after first token", "text_first_token", "text_complete")
        self.span("Despina first audio chunk", "tts_request", "tts_first_chunk")
        self.span("Playback initialization", "playback_init_start", "playback_init_end")
        self.span("Enter to first audio submitted", "enter", "first_audio_submitted")
        if "first_audio_submitted" in self.marks:
            self.values_ms["Enter to first audible response"] = (
                (self.marks["first_audio_submitted"] - self.started) * 1000 + bluetooth_latency_ms
            )
            self.estimated.add("Enter to first audible response")

    def format(self) -> str:
        lines = ["Latency summary:"]
        for name, value in self.values_ms.items():
            suffix = " (estimated)" if name in self.estimated else ""
            lines.append(f"  {name}: {value:.1f} ms{suffix}")
        cumulative = {"Gemini complete text", "Enter to first audio submitted"}
        measured = {key: value for key, value in self.values_ms.items()
                    if key not in self.estimated and key not in cumulative}
        if measured:
            name = max(measured, key=measured.get)
            lines.append(f"  Largest measured stage: {name} ({measured[name]:.1f} ms)")
        return "\n".join(lines)
