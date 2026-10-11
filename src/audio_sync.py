"""WAV energy fallback and aplay playback clock integration."""
from __future__ import annotations
import audioop
import os
import shutil
import subprocess
import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

def wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as source:
        return source.getnframes() / source.getframerate()


def energy_events(wav_path: Path, window_s: float = 0.04) -> list[dict[str, float]]:
    """Audio-based fallback: speech energy opens, quiet windows close the mouth."""
    with wave.open(str(wav_path), "rb") as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        frames = max(1, int(rate * window_s)); chunks = []
        while data := source.readframes(frames): chunks.append(data)
    if not chunks: return [{"time": 0.0, "openness": 0.0}]
    rms = [audioop.rms(chunk, width) for chunk in chunks]; peak = max(rms) or 1
    events = []
    for index, value in enumerate(rms):
        # retain pauses as 0; compress energy for natural partial opening.
        openness = 0.0 if value < peak * 0.08 else min(0.7, (value / peak) ** 0.5 * 0.7)
        events.append({"time": index * window_s, "openness": openness})
    duration = len(chunks) * window_s
    # Begin closing during the final audio window, not after playback ends.
    events.append({"time": max(0.0, duration - window_s), "openness": 0.0})
    events.append({"time": duration, "openness": 0.0})
    return events

def word_events(words: list[dict[str, float]], duration: float | None = None,
                speed_fraction: float | None = None) -> list[dict[str, float]]:
    """Create syllable-like pulses bounded by each aligned word interval."""
    events = [{"time": 0.0, "openness": 0.0}]
    for word in words:
        start, end = word["start"], word["end"]
        if end <= start or (duration is not None and end > duration + 0.05):
            continue
        text = str(word.get("word", ""))
        syllables = max(1, min(4, len(__import__("re").findall(r"[aeiouy]+", text.lower()))))
        slot = (end - start) / syllables
        for index in range(syllables):
            onset = start + slot * index
            close = min(end, onset + slot * 0.78)
            amplitude = 0.62
            if speed_fraction is not None:
                # A complete open/close must fit in this syllable. Short words
                # get smaller motion instead of leaving the jaw chasing late.
                amplitude = min(amplitude, max(0.02, speed_fraction * slot * 0.45))
            events.extend(({"time": onset, "openness": amplitude},
                           {"time": close, "openness": 0.12}))
        events.append({"time": end, "openness": 0.0})
    if duration is not None:
        events.append({"time": duration, "openness": 0.0})
    return sorted(events, key=lambda event: event["time"])


def validate_word_timings(words: list[dict[str, float]], duration: float) -> bool:
    previous = 0.0
    if not words:
        return False
    for word in words:
        start, end = word["start"], word["end"]
        if start < previous or end < start or end > duration + 0.05:
            return False
        previous = start
    return True


def limit_event_amplitudes(events: list[dict[str, float]], speed_fraction: float) -> list[dict[str, float]]:
    """Clamp targets in both directions so openings and closings fit in time."""
    limited = [dict(event) for event in events]
    position = 0.0; previous = 0.0
    for event in limited:
        allowed = speed_fraction * max(0.0, event["time"] - previous)
        event["openness"] = max(position - allowed, min(position + allowed, event["openness"]))
        position, previous = event["openness"], event["time"]
    position = limited[-1]["openness"] if limited else 0.0
    next_time = limited[-1]["time"] if limited else 0.0
    for event in reversed(limited[:-1]):
        allowed = speed_fraction * max(0.0, next_time - event["time"])
        event["openness"] = max(position - allowed, min(position + allowed, event["openness"]))
        position, next_time = event["openness"], event["time"]
    return limited


@dataclass
class PlaybackDiagnostics:
    chunks: int = 0
    bytes_submitted: int = 0
    samples_submitted: int = 0
    duration_submitted_s: float = 0.0
    samples_played: int = 0
    completed_normally: bool = False
    return_code: int | None = None
    producer_error: str | None = None
    dropped_chunks: int = 0
    last_pcm_submitted_at: float | None = None
    playback_started_at: float | None = None
    playback_completed_at: float | None = None

def select_audio_backend(requested: str | None = None, which: Callable[[str], str | None] = shutil.which) -> str:
    """Prefer the desktop user's PipeWire/PulseAudio default sink."""
    choice = (requested or os.getenv("HEAD_AUDIO_BACKEND", "auto")).lower()
    if choice not in {"auto", "pulse", "alsa"}:
        raise RuntimeError("HEAD_AUDIO_BACKEND must be auto, pulse, or alsa")
    if choice in {"auto", "pulse"} and which("paplay"):
        return "pulse"
    if choice == "pulse":
        raise RuntimeError("paplay is unavailable; install pulseaudio-utils or PipeWire PulseAudio compatibility")
    if which("aplay"):
        return "alsa"
    raise RuntimeError("No audio player found; install pulseaudio-utils (paplay) or alsa-utils (aplay)")

class AudioPlayer:
    """User-session audio player with a playback-clock estimate for lip sync.

    paplay targets the desktop session's default PulseAudio/PipeWire sink, which
    is normally the selected Bluetooth speaker. Bluetooth acoustic latency is
    not exposed by paplay; HEAD_AUDIO_LATENCY_MS can compensate conservatively.
    """
    def __init__(self, wav_path: Path, backend: str | None = None, latency_ms: float | None = None,
                 popen=subprocess.Popen, clock: Callable[[], float] = time.monotonic,
                 which: Callable[[str], str | None] = shutil.which) -> None:
        self.wav_path, self.backend, self._popen, self._clock = wav_path, select_audio_backend(backend, which), popen, clock
        self.latency_s = (float(os.getenv("HEAD_AUDIO_LATENCY_MS", "0")) if latency_ms is None else latency_ms) / 1000
        if self.latency_s < 0: raise RuntimeError("HEAD_AUDIO_LATENCY_MS cannot be negative")
        self.process = self.started = None
        self.diagnostics = PlaybackDiagnostics()
    def start(self) -> Callable[[], float]:
        command = ["paplay", str(self.wav_path)] if self.backend == "pulse" else ["aplay", str(self.wav_path)]
        self.process = self._popen(command)
        self.started = self._clock()
        return lambda: max(0.0, self._clock() - self.started - self.latency_s)
    def wait(self) -> int:
        code = self.process.wait() if self.process else 0
        self.diagnostics.return_code = code
        self.diagnostics.completed_normally = code == 0
        if self.wav_path.exists():
            with wave.open(str(self.wav_path), "rb") as source:
                samples = source.getnframes()
            self.diagnostics.chunks = 1
            self.diagnostics.samples_submitted = samples
            self.diagnostics.samples_played = samples if code == 0 else 0
            self.diagnostics.duration_submitted_s = wav_duration(self.wav_path)
        return code
    def stop(self) -> None:
        if self.process and self.process.poll() is None: self.process.terminate()

# Backward-compatible name used by older callers.
Aplay = AudioPlayer


class RawPCMPlayer:
    """Bounded, backpressured raw-PCM playback for streamed TTS."""
    def __init__(self, backend: str | None = None, sample_rate: int = 24000,
                 latency_ms: float | None = None, popen=subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic,
                 which: Callable[[str], str | None] = shutil.which,
                 queue_chunks: int = 6) -> None:
        self.backend = select_audio_backend(backend, which)
        self.sample_rate, self._popen, self._clock = sample_rate, popen, clock
        self.latency_ms = float(os.getenv("HEAD_AUDIO_LATENCY_MS", "0")) if latency_ms is None else latency_ms
        if self.latency_ms < 0:
            raise RuntimeError("HEAD_AUDIO_LATENCY_MS cannot be negative")
        self.queue_chunks = max(2, queue_chunks)
        self.process = None
        self.cancelled = threading.Event()
        self.diagnostics = PlaybackDiagnostics()

    def _command(self) -> list[str]:
        if self.backend == "pulse":
            return ["paplay", "--raw", f"--rate={self.sample_rate}", "--channels=1", "--format=s16le"]
        return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1"]

    def play(self, chunks, *, on_ready=None, on_first_submit=None, on_chunk=None) -> int:
        pcm_queue: queue.Queue = queue.Queue(maxsize=self.queue_chunks)
        sentinel = object()
        errors: list[BaseException] = []

        def enqueue(item) -> bool:
            while not self.cancelled.is_set():
                try:
                    pcm_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for item in chunks:
                    if self.cancelled.is_set():
                        break
                    if not enqueue(item):
                        break
            except BaseException as exc:
                errors.append(exc)
                self.diagnostics.producer_error = repr(exc)
            finally:
                enqueue(sentinel)

        self.process = self._popen(self._command(), stdin=subprocess.PIPE, bufsize=0)
        self.diagnostics.playback_started_at = self._clock()
        if on_ready:
            on_ready()
        producer = threading.Thread(target=produce, name="despina-stream", daemon=True)
        producer.start()
        submitted = 0
        consumer_error = None
        try:
            while not self.cancelled.is_set():
                item = pcm_queue.get()
                if item is sentinel:
                    break
                data = item[0] if isinstance(item, tuple) else item
                if not data:
                    continue
                self.process.stdin.write(data)
                self.diagnostics.last_pcm_submitted_at = self._clock()
                if submitted == 0 and on_first_submit:
                    on_first_submit()
                if on_chunk:
                    on_chunk(data, submitted / (self.sample_rate * 2))
                submitted += len(data)
                self.diagnostics.chunks += 1
        except BaseException as exc:
            consumer_error = exc
            self.cancelled.set()
            if self.process.poll() is None:
                self.process.terminate()
        finally:
            if self.process.stdin:
                self.process.stdin.close()
            producer.join(timeout=2)
            code = self.process.wait()
            self.diagnostics.playback_completed_at = self._clock()
        self.diagnostics.bytes_submitted = submitted
        self.diagnostics.samples_submitted = submitted // 2
        self.diagnostics.duration_submitted_s = submitted / (self.sample_rate * 2)
        self.diagnostics.return_code = code
        self.diagnostics.completed_normally = code == 0 and not errors and not self.cancelled.is_set()
        self.diagnostics.samples_played = self.diagnostics.samples_submitted if self.diagnostics.completed_normally else 0
        if consumer_error:
            raise consumer_error
        if errors:
            raise errors[0]
        return code

    def stop(self) -> None:
        self.cancelled.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
