"""WAV energy fallback and aplay playback clock integration."""
from __future__ import annotations
import audioop
import shutil
import subprocess
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

class Aplay:
    def __init__(self, wav_path: Path, popen=subprocess.Popen, clock: Callable[[], float] = time.monotonic) -> None:
        self.wav_path, self._popen, self._clock, self.process, self.started = wav_path, popen, clock, None, None
    def start(self) -> Callable[[], float]:
        if not shutil.which("aplay"): raise RuntimeError("aplay is unavailable; install alsa-utils and check speaker setup")
        self.process = self._popen(["aplay", str(self.wav_path)])
        self.started = self._clock()
        return lambda: self._clock() - self.started
    def wait(self) -> int:
        return self.process.wait() if self.process else 0
    def stop(self) -> None:
        if self.process and self.process.poll() is None: self.process.terminate()
