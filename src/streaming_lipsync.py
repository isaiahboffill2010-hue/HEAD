"""Non-blocking incremental PCM-energy mouth targets."""
from __future__ import annotations

import audioop
import os
import queue
import threading
import time

from .adaptive_lipsync import MotionSettings


class StreamingLipSynchronizer:
    """Consume submitted PCM on a separate bounded worker; never delays audio."""
    def __init__(self, mouth, max_jaw_travel_deg: float, settings: MotionSettings,
                 sample_rate: int = 24000, clock=time.monotonic,
                 latency_ms: float | None = None) -> None:
        self.mouth, self.settings = mouth, settings
        self.sample_rate, self.clock = sample_rate, clock
        self.max_jaw_travel_deg = max_jaw_travel_deg
        self.queue: queue.Queue = queue.Queue(maxsize=128)
        self.cancelled = threading.Event()
        self.thread = None
        self.analysis_ms = 0.0
        self._position = 0.0
        self._last = None
        self._started = None
        self.dropped_chunks = 0
        value = float(os.getenv("HEAD_AUDIO_LATENCY_MS", "0")) if latency_ms is None else latency_ms
        self.latency_s = max(0.0, value / 1000)

    def start(self) -> None:
        self._started = self._last = self.clock()
        self.thread = threading.Thread(target=self._run, name="mouth-pcm", daemon=True)
        self.thread.start()

    def submit(self, pcm: bytes, audio_time: float) -> None:
        # API chunk sizes vary with the network. Normalize to 20 ms audio
        # frames so mouth timing is based on samples, not packet boundaries.
        frame_bytes = self.sample_rate * 2 // 50
        for offset in range(0, len(pcm), frame_bytes):
            frame = pcm[offset:offset + frame_bytes]
            try:
                self.queue.put_nowait((frame, audio_time + offset / (self.sample_rate * 2)))
            except queue.Full:
                self.dropped_chunks += 1

    def _run(self) -> None:
        peak = 1.0
        speed_fraction = self.settings.speed() / self.max_jaw_travel_deg
        while not self.cancelled.is_set():
            try:
                pcm, audio_time = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            due = (self._started or self.clock()) + self.latency_s + audio_time
            wait = due - self.clock()
            if wait > 0 and self.cancelled.wait(wait):
                break
            began = self.clock()
            rms = audioop.rms(pcm, 2) if pcm else 0
            peak = max(float(rms), peak * 0.995)
            target = 0.0 if rms < peak * 0.07 else min(0.72, (rms / peak) ** 0.5 * 0.72)
            now = self.clock()
            dt = max(0.001, now - (self._last or now))
            self._last = now
            allowed = speed_fraction * dt
            self._position += max(-allowed, min(allowed, target - self._position))
            self.mouth.set_openness_step(max(0.0, min(1.0, self._position)))
            self.analysis_ms += (self.clock() - began) * 1000

    def stop(self, close: bool = True) -> None:
        self.cancelled.set()
        if self.thread:
            self.thread.join(timeout=1)
        if close and self._position:
            self.mouth.set_openness_step(0.0)
