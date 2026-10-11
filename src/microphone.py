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
        # Ziggy's hands-free turns need a little more room between final words
        # than the original push-to-talk path.  Pre-roll remains 300 ms.
        self.silence_s = min(1.2, max(.45, float(os.getenv("HEAD_MIC_SILENCE_MS", "1000")) / 1000))
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

    def _read_pcm(self, byte_count: int) -> bytes:
        """Read exactly byte_count bytes; pipe reads may legitimately be short."""
        assert self.process and self.process.stdout
        remaining = byte_count
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

    def _read_frame(self) -> bytes:
        """Read one 20 ms mono S16_LE analysis frame."""
        return self._read_pcm(self.frame_bytes)

    def read_pcm(self, byte_count: int) -> bytes:
        """Read an exact raw PCM block, recovering a genuinely interrupted capture."""
        if byte_count <= 0 or byte_count % 2:
            raise ValueError("PCM reads require a positive whole number of S16_LE samples")
        self.start()
        for recovery in range(self.recovery_attempts + 1):
            try:
                return self._read_pcm(byte_count)
            except _CaptureEnded as exc:
                if recovery >= self.recovery_attempts:
                    raise RuntimeError(f"{exc}; recovery attempts exhausted") from exc
                print(f"Microphone capture interrupted; restarting ({recovery + 1}/{self.recovery_attempts}).")
                self._restart_after_interruption()
        raise AssertionError("unreachable")

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

    def record_after_wake(self, prefix_pcm: bytes = b"", max_s: float = 15.0) -> Utterance:
        """Capture one complete post-wake request, retaining pre-roll safely.

        The wake detector has already heard speech, so the silence timer begins
        after activation and resets for every non-silent 20 ms frame.
        """
        if len(prefix_pcm) % 2:
            raise ValueError("prefix PCM must be S16_LE samples")
        self.start()
        audio = [prefix_pcm] if prefix_pcm else []
        silence = 0.0; began = self.clock(); recoveries = 0
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
            audio.append(frame)
            if audioop.rms(frame, 2) < self.threshold:
                silence += .02
                if silence >= self.silence_s:
                    break
            else:
                silence = 0.0
        pcm = b"".join(audio)
        return Utterance(pcm, len(pcm) / (self.rate * 2))

    def record_question_after_wake(self, prefix_pcm: bytes = b"", wait_s: float = 5.0,
                                   max_s: float = 15.0) -> Utterance | None:
        """Wait for new speech, then record it; wake-only activation returns None."""
        if len(prefix_pcm) % 2:
            raise ValueError("prefix PCM must be S16_LE samples")
        self.start(); audio: list[bytes] = []; detected = False; silence = 0.0
        began = self.clock()
        while self.clock() - began < max_s:
            frame = self._read_frame()
            rms = audioop.rms(frame, 2)
            if not detected:
                if rms < self.threshold:
                    if self.clock() - began >= wait_s:
                        return None
                    continue
                detected = True
                # Preserve the immediate post-wake phrase only after genuine
                # question speech has started; wake-only audio reaches no API.
                if prefix_pcm: audio.append(prefix_pcm)
            audio.append(frame)
            silence = silence + .02 if rms < self.threshold else 0.0
            if silence >= self.silence_s:
                break
        if not detected:
            return None
        pcm = b"".join(audio)
        return Utterance(pcm, len(pcm) / (self.rate * 2))

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
