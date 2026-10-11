"""Persistent lightweight ALSA microphone capture and energy VAD for Pi-class hosts."""
from __future__ import annotations
import audioop, os, shutil, subprocess, time, wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Utterance:
    pcm: bytes
    duration_s: float


class Microphone:
    """One persistent arecord process; no local ML model or per-turn device init."""
    def __init__(self, device: str | None = None, rate: int = 16000, popen=subprocess.Popen,
                 clock=time.monotonic) -> None:
        self.device = device or os.getenv("HEAD_MIC_DEVICE", "default")
        self.rate, self._popen, self.clock = rate, popen, clock
        self.process = None
        self.frame_bytes = rate * 2 // 50  # 20 ms, mono S16_LE
        self.threshold = max(80, int(os.getenv("HEAD_MIC_VAD_RMS", "450")))
        self.silence_s = min(.6, max(.45, float(os.getenv("HEAD_MIC_SILENCE_MS", "550")) / 1000))

    @staticmethod
    def devices() -> str:
        if not shutil.which("arecord"): return "arecord is unavailable; install alsa-utils"
        return subprocess.run(["arecord", "-l"], text=True, capture_output=True, check=False).stdout.strip()

    def start(self) -> None:
        if self.process is None:
            self.process = self._popen(["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
                                        "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def listen(self, max_s: float = 15.0) -> Utterance | None:
        self.start(); assert self.process and self.process.stdout
        pre = deque(maxlen=15); audio = []; detected = False; silence = 0.0; began = self.clock()
        while self.clock() - began < max_s:
            frame = self.process.stdout.read(self.frame_bytes)
            if len(frame) != self.frame_bytes: raise RuntimeError("microphone capture ended unexpectedly")
            rms = audioop.rms(frame, 2); pre.append(frame)
            if not detected:
                if rms < self.threshold: continue
                detected = True; audio.extend(pre); silence = 0.0; print("Speech detected.")
            else:
                audio.append(frame)
                silence = silence + .02 if rms < self.threshold else 0.0
                if silence >= self.silence_s: break
        if not detected: return None
        pcm = b"".join(audio); return Utterance(pcm, len(pcm) / (self.rate * 2))

    def write_wav(self, utterance: Utterance, path: Path) -> Path:
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(self.rate); out.writeframes(utterance.pcm)
        return path

    def close(self) -> None:
        if self.process and self.process.poll() is None: self.process.terminate()
        self.process = None
