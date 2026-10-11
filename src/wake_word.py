"""Small, offline custom-keyword detector for Ziggy.

The detector deliberately has no cloud fallback: background audio must never
be sent to Gemini merely to discover a wake phrase.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path


class WakeWordConfigurationError(RuntimeError):
    pass


class PorcupineWakeWord:
    """Adapter for one Picovoice Porcupine custom .ppn keyword model."""
    def __init__(self, engine) -> None:
        self._engine = engine
        self.frame_length = int(engine.frame_length)
        self.sample_rate = int(engine.sample_rate)

    @classmethod
    def from_environment(cls):
        access_key = os.getenv("PICOVOICE_ACCESS_KEY", "").strip()
        model_path = Path(os.getenv("HEAD_WAKE_WORD_MODEL", "")).expanduser()
        if not access_key or not str(model_path) or not model_path.is_file():
            raise WakeWordConfigurationError(
                "Ziggy wake word requires PICOVOICE_ACCESS_KEY and HEAD_WAKE_WORD_MODEL "
                "pointing to a Raspberry-Pi Ziggy .ppn model; no cloud wake-word fallback is used")
        try:
            import pvporcupine
        except ImportError as exc:
            raise WakeWordConfigurationError("pvporcupine is unavailable; install requirements.txt") from exc
        sensitivity = min(1.0, max(0.0, float(os.getenv("HEAD_WAKE_WORD_SENSITIVITY", "0.5"))))
        engine = pvporcupine.create(access_key=access_key, keyword_paths=[str(model_path)],
                                    sensitivities=[sensitivity])
        return cls(engine)

    def detected(self, pcm: bytes) -> bool:
        expected = self.frame_length * 2
        if len(pcm) != expected:
            raise ValueError(f"wake detector requires {expected} PCM bytes, received {len(pcm)}")
        samples = struct.unpack("<" + "h" * self.frame_length, pcm)
        return self._engine.process(samples) >= 0

    def close(self) -> None:
        self._engine.delete()
