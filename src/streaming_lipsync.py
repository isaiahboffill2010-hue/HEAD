"""Non-blocking incremental PCM-energy mouth targets."""
from __future__ import annotations

import audioop
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

from .adaptive_lipsync import MotionSettings


@dataclass
class StreamingLipSyncDiagnostics:
    """Times are monotonic seconds; audio times are seconds from PCM start."""
    final_tts_pcm_received_at: float | None = None
    final_pcm_audio_end_s: float | None = None
    last_speech_frame_audio_s: float | None = None
    last_scheduled_mouth_at: float | None = None
    playback_completed_at: float | None = None
    last_mouth_frame_processed_at: float | None = None
    dropped_frames: int = 0
    shutdown_reason: str | None = None


class StreamingLipSynchronizer:
    """Consume submitted PCM without delaying playback, then drain it at EOS.

    ``finish`` is deliberately separate from ``stop``. Network/TTS completion
    merely means no more PCM will arrive; it must not cancel frames already
    queued for their (possibly Bluetooth-delayed) playback time.
    """
    def __init__(self, mouth, max_jaw_travel_deg: float, settings: MotionSettings,
                 sample_rate: int = 24000, clock=time.monotonic,
                 latency_ms: float | None = None) -> None:
        self.mouth, self.settings = mouth, settings
        self.sample_rate, self.clock = sample_rate, clock
        self.max_jaw_travel_deg = max_jaw_travel_deg
        self.queue: queue.Queue = queue.Queue(maxsize=128)
        self.cancelled = threading.Event()
        self._eos = object()
        self.thread = None
        self.analysis_ms = 0.0
        self._position = 0.0
        self._last = None
        self._started = None
        self.dropped_chunks = 0
        self.diagnostics = StreamingLipSyncDiagnostics()
        self._logger = logging.getLogger(__name__)
        value = float(os.getenv("HEAD_AUDIO_LATENCY_MS", "0")) if latency_ms is None else latency_ms
        self.latency_s = max(0.0, value / 1000)

    def start(self) -> None:
        self._started = self._last = self.clock()
        self.thread = threading.Thread(target=self._run, name="mouth-pcm", daemon=True)
        self.thread.start()

    def note_tts_pcm_received(self) -> None:
        self.diagnostics.final_tts_pcm_received_at = self.clock()

    def submit(self, pcm: bytes, audio_time: float) -> None:
        # API chunk sizes vary with the network. Normalize to 20 ms audio
        # frames so mouth timing is based on samples, not packet boundaries.
        frame_bytes = self.sample_rate * 2 // 50
        self.diagnostics.final_pcm_audio_end_s = audio_time + len(pcm) / (self.sample_rate * 2)
        for offset in range(0, len(pcm), frame_bytes):
            frame = pcm[offset:offset + frame_bytes]
            try:
                self.queue.put_nowait((frame, audio_time + offset / (self.sample_rate * 2)))
            except queue.Full:
                self.dropped_chunks += 1
                self.diagnostics.dropped_frames = self.dropped_chunks

    def _run(self) -> None:
        peak = 1.0
        speed_fraction = self.settings.speed() / self.max_jaw_travel_deg
        while not self.cancelled.is_set():
            item = self.queue.get()
            if item is self._eos:
                self.diagnostics.shutdown_reason = "cancelled" if self.cancelled.is_set() else "drained end-of-stream"
                break
            pcm, audio_time = item
            due = (self._started or self.clock()) + self.latency_s + audio_time
            wait = due - self.clock()
            if wait > 0 and self.cancelled.wait(wait):
                break
            began = self.clock()
            rms = audioop.rms(pcm, 2) if pcm else 0
            peak = max(float(rms), peak * 0.995)
            target = 0.0 if rms < peak * 0.07 else min(0.72, (rms / peak) ** 0.5 * 0.72)
            if target > 0:
                self.diagnostics.last_speech_frame_audio_s = audio_time
            now = self.clock()
            dt = max(0.001, now - (self._last or now))
            self._last = now
            allowed = speed_fraction * dt
            self._position += max(-allowed, min(allowed, target - self._position))
            self.mouth.set_openness_step(max(0.0, min(1.0, self._position)))
            self.diagnostics.last_scheduled_mouth_at = due
            self.diagnostics.last_mouth_frame_processed_at = self.clock()
            self.analysis_ms += (self.clock() - began) * 1000
        if self.cancelled.is_set() and self.diagnostics.shutdown_reason is None:
            self.diagnostics.shutdown_reason = "cancelled"

    def finish(self, playback_completed_at: float | None = None, close: bool = True) -> None:
        """Drain all queued frames before closing, including delayed Bluetooth frames."""
        self.diagnostics.playback_completed_at = playback_completed_at or self.clock()
        if self.thread and self.thread.is_alive() and not self.cancelled.is_set():
            self.queue.put(self._eos)
            self.thread.join()
        if close and not self.cancelled.is_set():
            self.mouth.set_openness_step(0.0)
        self._logger.debug(
            "mouth EOS final_tts_pcm=%s last_speech_audio=%s last_scheduled=%s playback_complete=%s "
            "last_processed=%s dropped=%s shutdown=%s",
            self.diagnostics.final_tts_pcm_received_at, self.diagnostics.last_speech_frame_audio_s,
            self.diagnostics.last_scheduled_mouth_at, self.diagnostics.playback_completed_at,
            self.diagnostics.last_mouth_frame_processed_at, self.diagnostics.dropped_frames,
            self.diagnostics.shutdown_reason)

    def stop(self, close: bool = True) -> None:
        """Cancel immediately (interrupt/error path); no queued command is retained."""
        self.cancelled.set()
        if self.thread:
            try:
                self.queue.put_nowait(self._eos)
            except queue.Full:
                pass
            self.thread.join(timeout=1)
        if close:
            self.mouth.set_openness_step(0.0)
