"""Persistent lightweight ALSA microphone capture and energy VAD for Pi-class hosts."""
from __future__ import annotations
import audioop, os, shutil, subprocess, threading, time, wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Utterance:
    pcm: bytes
    duration_s: float


class _CaptureEnded(RuntimeError):
    """The persistent ALSA process closed its PCM stream."""


class Microphone:
    """One persistent arecord process; no local ML model or per-turn device init."""
    def __init__(self, device: str | None = None, rate: int = 16000, popen=subprocess.Popen,
                 clock=time.monotonic) -> None:
        self.device = device or os.getenv("HEAD_MIC_DEVICE", "default")
        self.rate, self._popen, self.clock = rate, popen, clock
        self.process = None
        self._stderr: deque[str] = deque(maxlen=64)
        self._stderr_thread: threading.Thread | None = None
        self.frame_bytes = rate * 2 // 50  # 20 ms, mono S16_LE
        self.threshold = max(80, int(os.getenv("HEAD_MIC_VAD_RMS", "450")))
        self.silence_s = min(.6, max(.45, float(os.getenv("HEAD_MIC_SILENCE_MS", "550")) / 1000))
        self.recovery_attempts = min(3, max(0, int(os.getenv("HEAD_MIC_RECOVERY_ATTEMPTS", "2"))))

    @staticmethod
    def devices() -> str:
        if not shutil.which("arecord"): return "arecord is unavailable; install alsa-utils"
        return subprocess.run(["arecord", "-l"], text=True, capture_output=True, check=False).stdout.strip()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self._stderr.clear()
        # Raw PCM and no duration (-d): this one process remains open across turns.
        self.process = self._popen(["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
                                    "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        stderr = getattr(self.process, "stderr", None)
        if stderr is not None:
            self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(stderr,), daemon=True,
                                                   name="head-mic-stderr")
            self._stderr_thread.start()

    def _drain_stderr(self, stream) -> None:
        """Prevent a noisy ALSA stderr pipe from blocking arecord; retain diagnostics."""
        while True:
            try:
                data = stream.read(256)
            except (OSError, ValueError, AttributeError):
                return
            if not data:
                return
            self._stderr.append(data.decode("utf-8", errors="replace"))

    def _capture_error(self) -> str:
        # Give the drain thread a brief chance to collect the final ALSA line.
        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=0.05)
        code = self.process.poll() if self.process is not None else None
        stderr = "".join(self._stderr).strip() or "no ALSA stderr captured"
        state = str(code) if code is not None else "still running"
        return f"microphone capture ended unexpectedly (arecord exit code: {state}; ALSA stderr: {stderr})"

    def _read_frame(self) -> bytes:
        """Read precisely one 20 ms frame; pipe reads may legitimately be short."""
        assert self.process and self.process.stdout
        remaining = self.frame_bytes
        chunks: list[bytes] = []
        while remaining:
            try:
                chunk = self.process.stdout.read(remaining)
            except (OSError, ValueError) as exc:
                raise _CaptureEnded(f"{self._capture_error()}; read error: {exc}") from exc
            if not chunk:
                raise _CaptureEnded(self._capture_error())
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _restart_after_interruption(self) -> None:
        self.close()
        self.start()

    def listen(self, max_s: float = 15.0) -> Utterance | None:
        self.start(); assert self.process and self.process.stdout
        pre = deque(maxlen=15); audio = []; detected = False; silence = 0.0; began = self.clock()
        recoveries = 0
        while self.clock() - began < max_s:
            try:
                frame = self._read_frame()
            except _CaptureEnded as exc:
                if recoveries >= self.recovery_attempts:
                    raise RuntimeError(f"{exc}; recovery attempts exhausted") from exc
                recoveries += 1
                print(f"Microphone capture interrupted; restarting ({recoveries}/{self.recovery_attempts}).")
                self._restart_after_interruption()
                continue
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
        process = self.process
        self.process = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except (subprocess.TimeoutExpired, AttributeError):
                pass
        for stream_name in ("stdout", "stderr"):
            stream = getattr(process, stream_name, None) if process else None
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError, AttributeError):
                pass
