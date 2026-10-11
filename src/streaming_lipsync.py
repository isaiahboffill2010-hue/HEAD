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
    last_speech_audio_end_s: float | None = None
    last_scheduled_mouth_at: float | None = None
    last_speech_mouth_command_at: float | None = None
    final_audible_audio_estimate_at: float | None = None
    playback_completed_at: float | None = None
    last_mouth_frame_processed_at: float | None = None
    dropped_frames: int = 0
    shutdown_reason: str | None = None
    speech_frames: int = 0
    quiet_frames: int = 0
    commanded_frames: int = 0
    stationary_commands_suppressed: int = 0
    stale_frames_coalesced: int = 0
    late_hardware_commands: int = 0
    max_command_lateness_ms: float = 0.0
    max_command_duration_ms: float = 0.0
    last_command_completed_at: float | None = None


@dataclass(frozen=True)
class MouthTuning:
    """Software-only energy/motion controls; never alter calibrated endpoints."""
    sensitivity: float = 1.0
    max_openness: float = 0.72
    smoothing: float = 0.0
    command_deadband_deg: float = 0.5

    @classmethod
    def from_environment(cls) -> "MouthTuning":
        value = cls(float(os.getenv("HEAD_MOUTH_SENSITIVITY", "1.0")),
                    float(os.getenv("HEAD_MOUTH_MAX_OPENNESS", "0.72")),
                    float(os.getenv("HEAD_MOUTH_SMOOTHING", "0")),
                    float(os.getenv("HEAD_MOUTH_COMMAND_DEADBAND_DEG", "0.5")))
        if not .25 <= value.sensitivity <= 4:
            raise ValueError("HEAD_MOUTH_SENSITIVITY must be between 0.25 and 4")
        if not .05 <= value.max_openness <= 1:
            raise ValueError("HEAD_MOUTH_MAX_OPENNESS must be between 0.05 and 1")
        if not 0 <= value.smoothing <= .95:
            raise ValueError("HEAD_MOUTH_SMOOTHING must be between 0 and 0.95")
        if not .1 <= value.command_deadband_deg <= 2:
            raise ValueError("HEAD_MOUTH_COMMAND_DEADBAND_DEG must be between 0.1 and 2")
        return value


class StreamingLipSynchronizer:
    """Consume submitted PCM without delaying playback, then drain it at EOS.

    ``finish`` is deliberately separate from ``stop``. Network/TTS completion
    merely means no more PCM will arrive; it must not cancel frames already
    queued for their (possibly Bluetooth-delayed) playback time.
    """
    def __init__(self, mouth, max_jaw_travel_deg: float, settings: MotionSettings,
                 sample_rate: int = 24000, clock=time.monotonic,
                 latency_ms: float | None = None, mouth_offset_ms: float | None = None,
                 tuning: MouthTuning | None = None) -> None:
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
        self._smoothed_target = 0.0
        self._last_commanded_position = 0.0
        self.motion_records: list[dict[str, float | bool]] = []
        self.dropped_chunks = 0
        self.diagnostics = StreamingLipSyncDiagnostics()
        self._logger = logging.getLogger(__name__)
        value = float(os.getenv("HEAD_AUDIO_LATENCY_MS", "0")) if latency_ms is None else latency_ms
        self.latency_s = max(0.0, value / 1000)
        offset = float(os.getenv("HEAD_MOUTH_OFFSET_MS", "0")) if mouth_offset_ms is None else mouth_offset_ms
        if offset < 0:
            raise ValueError("HEAD_MOUTH_OFFSET_MS cannot be negative")
        self.mouth_offset_s = offset / 1000
        self.effective_delay_s = self.latency_s + self.mouth_offset_s
        self.tuning = tuning or MouthTuning.from_environment()

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
        pending = None
        end_of_stream = False
        while not self.cancelled.is_set():
            item = pending if pending is not None else self.queue.get()
            pending = None
            if item is self._eos:
                self.diagnostics.shutdown_reason = "cancelled" if self.cancelled.is_set() else "drained end-of-stream"
                break
            pcm, audio_time = item
            due = (self._started or self.clock()) + self.effective_delay_s + audio_time
            wait = due - self.clock()
            if wait > 0 and self.cancelled.wait(wait):
                break
            # If a hardware write or the Pi scheduler made us late, analyze all
            # expired PCM frames but retain only the newest target. Replaying old
            # movement commands would make the jaw chase audio that already passed.
            batch = [(pcm, audio_time, due)]
            while True:
                try:
                    candidate = self.queue.get_nowait()
                except queue.Empty:
                    break
                if candidate is self._eos:
                    end_of_stream = True; break
                candidate_due = (self._started or self.clock()) + self.effective_delay_s + candidate[1]
                if candidate_due <= self.clock():
                    batch.append((candidate[0], candidate[1], candidate_due))
                else:
                    pending = candidate; break
            if len(batch) > 1:
                self.diagnostics.stale_frames_coalesced += len(batch) - 1
            latest = None
            for frame, frame_time, frame_due in batch:
                began = self.clock(); rms = audioop.rms(frame, 2) if frame else 0
                peak = max(float(rms), peak * 0.995); threshold = peak * (0.07 / self.tuning.sensitivity)
                raw_target = 0.0 if rms < threshold else min(
                    self.tuning.max_openness, (rms / peak) ** 0.5 * self.tuning.max_openness)
                target = self._smoothed_target * self.tuning.smoothing + raw_target * (1 - self.tuning.smoothing)
                self._smoothed_target = target
                if target > 0:
                    self.diagnostics.speech_frames += 1
                    self.diagnostics.last_speech_frame_audio_s = frame_time
                    self.diagnostics.last_speech_audio_end_s = frame_time + len(frame) / (self.sample_rate * 2)
                else: self.diagnostics.quiet_frames += 1
                latest = (target, frame_time, frame_due, float(rms), threshold, began)
            assert latest is not None
            target, frame_time, frame_due, rms, threshold, began = latest
            now = self.clock(); dt = max(0.001, now - (self._last or now)); self._last = now
            allowed = speed_fraction * dt; self._position += max(-allowed, min(allowed, target - self._position))
            position = max(0.0, min(1.0, self._position))
            command = abs(position - self._last_commanded_position) >= (self.tuning.command_deadband_deg / self.max_jaw_travel_deg)
            command_started = command_completed = None
            if command:
                command_started = self.clock(); lateness_ms = max(0.0, (command_started - frame_due) * 1000)
                self.mouth.set_openness_step(position)
                command_completed = self.clock(); duration_ms = (command_completed - command_started) * 1000
                self._last_commanded_position = position; self.diagnostics.commanded_frames += 1
                self.diagnostics.max_command_lateness_ms = max(self.diagnostics.max_command_lateness_ms, lateness_ms)
                self.diagnostics.max_command_duration_ms = max(self.diagnostics.max_command_duration_ms, duration_ms)
                self.diagnostics.last_command_completed_at = command_completed
                if lateness_ms > self.settings.update_interval_s * 1000: self.diagnostics.late_hardware_commands += 1
                if target > 0: self.diagnostics.last_speech_mouth_command_at = command_completed
            else: self.diagnostics.stationary_commands_suppressed += 1
            self.motion_records.append({"audio_time": frame_time, "rms": rms, "threshold": threshold,
                                        "target": target, "openness": position, "commanded": command,
                                        "scheduled_at": frame_due, "command_started_at": command_started or 0.0,
                                        "command_completed_at": command_completed or 0.0,
                                        "command_lateness_ms": (command_started - frame_due) * 1000 if command_started else 0.0,
                                        "command_duration_ms": (command_completed - command_started) * 1000 if command else 0.0,
                                        "coalesced_frames": len(batch) - 1})
            self.diagnostics.last_scheduled_mouth_at = frame_due
            self.diagnostics.last_mouth_frame_processed_at = self.clock()
            self.analysis_ms += (self.clock() - began) * 1000
            if end_of_stream:
                self.diagnostics.shutdown_reason = "drained end-of-stream"; break
        if self.cancelled.is_set() and self.diagnostics.shutdown_reason is None:
            self.diagnostics.shutdown_reason = "cancelled"

    def finish(self, playback_completed_at: float | None = None, close: bool = True,
               playback_started_at: float | None = None) -> None:
        """Drain all queued frames before closing, including delayed Bluetooth frames."""
        self.diagnostics.playback_completed_at = playback_completed_at or self.clock()
        if self.thread and self.thread.is_alive() and not self.cancelled.is_set():
            self.queue.put(self._eos)
            self.thread.join()
        if playback_started_at is not None and self.diagnostics.last_speech_audio_end_s is not None:
            self.diagnostics.final_audible_audio_estimate_at = (
                playback_started_at + self.latency_s + self.diagnostics.last_speech_audio_end_s)
        if close and not self.cancelled.is_set():
            self.mouth.set_openness_step(0.0)
            self.diagnostics.commanded_frames += 1
        self._logger.debug(
            "mouth EOS final_tts_pcm=%s last_speech_audio=%s last_scheduled=%s playback_complete=%s "
            "last_processed=%s dropped=%s shutdown=%s audio_delay=%.3f mouth_offset=%.3f speech=%s quiet=%s commands=%s suppressed=%s",
            self.diagnostics.final_tts_pcm_received_at, self.diagnostics.last_speech_frame_audio_s,
            self.diagnostics.last_scheduled_mouth_at, self.diagnostics.playback_completed_at,
            self.diagnostics.last_mouth_frame_processed_at, self.diagnostics.dropped_frames,
            self.diagnostics.shutdown_reason, self.latency_s, self.mouth_offset_s,
            self.diagnostics.speech_frames, self.diagnostics.quiet_frames,
            self.diagnostics.commanded_frames, self.diagnostics.stationary_commands_suppressed)

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
