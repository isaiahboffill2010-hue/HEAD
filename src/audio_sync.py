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
from pathlib import Path
from typing import Callable

def energy_events(wav_path: Path, window_s: float = 0.08) -> list[dict[str, float]]:
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
    events.append({"time": duration, "openness": 0.0})
    return events

def word_events(words: list[dict[str, float]]) -> list[dict[str, float]]:
    events = [{"time": 0.0, "openness": 0.0}]
    for word in words:
        start, end = word["start"], word["end"]
        events.extend(({"time": start, "openness": 0.55}, {"time": end, "openness": 0.0}))
    return sorted(events, key=lambda event: event["time"])

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
    def start(self) -> Callable[[], float]:
        command = ["paplay", str(self.wav_path)] if self.backend == "pulse" else ["aplay", str(self.wav_path)]
        self.process = self._popen(command)
        self.started = self._clock()
        return lambda: max(0.0, self._clock() - self.started - self.latency_s)
    def wait(self) -> int:
        return self.process.wait() if self.process else 0
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

    def _command(self) -> list[str]:
        if self.backend == "pulse":
            return ["paplay", "--raw", f"--rate={self.sample_rate}", "--channels=1", "--format=s16le"]
        return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1"]

    def play(self, chunks, *, on_ready=None, on_first_submit=None, on_chunk=None) -> int:
        pcm_queue: queue.Queue = queue.Queue(maxsize=self.queue_chunks)
        sentinel = object()
        errors: list[BaseException] = []

        def produce() -> None:
            try:
                for item in chunks:
                    if self.cancelled.is_set():
                        break
                    pcm_queue.put(item)
            except BaseException as exc:
                errors.append(exc)
            finally:
                pcm_queue.put(sentinel)

        self.process = self._popen(self._command(), stdin=subprocess.PIPE, bufsize=0)
        if on_ready:
            on_ready()
        producer = threading.Thread(target=produce, name="despina-stream", daemon=True)
        producer.start()
        submitted = 0
        while not self.cancelled.is_set():
            item = pcm_queue.get()
            if item is sentinel:
                break
            data = item[0] if isinstance(item, tuple) else item
            if not data:
                continue
            self.process.stdin.write(data)
            if submitted == 0 and on_first_submit:
                on_first_submit()
            if on_chunk:
                on_chunk(data, submitted / (self.sample_rate * 2))
            submitted += len(data)
        if self.process.stdin:
            self.process.stdin.close()
        producer.join(timeout=2)
        code = self.process.wait()
        if errors:
            raise errors[0]
        return code

    def stop(self) -> None:
        self.cancelled.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
