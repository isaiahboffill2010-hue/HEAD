"""Free, local Vosk wake recognition for Ziggy; never sends microphone audio online."""
from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path

DEFAULT_MODEL_NAME = "vosk-model-small-en-us-0.15"
DEFAULT_MODEL_URL = f"https://alphacephei.com/vosk/models/{DEFAULT_MODEL_NAME}.zip"


class VoskWakeWordError(RuntimeError):
    pass


def model_path() -> Path:
    configured = os.getenv("HEAD_VOSK_MODEL_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "models" / DEFAULT_MODEL_NAME


def wake_variants() -> set[str]:
    values = os.getenv("HEAD_WAKE_WORD_VARIANTS", "ziggy,ziggie").split(",")
    result = {value.strip().lower() for value in values if value.strip()}
    return result or {"ziggy"}


def _tokens(payload: str) -> set[str]:
    try:
        text = str(json.loads(payload).get("text") or json.loads(payload).get("partial") or "")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return set()
    return set(re.findall(r"[a-z]+", text.lower()))


class VoskWakeWord:
    """Constrained-vocabulary local recognizer for the wake phrase only."""
    def __init__(self, recognizer, *, sample_rate: int, variants: set[str]) -> None:
        self._recognizer, self.sample_rate, self.variants = recognizer, sample_rate, variants
        # 20 ms matches the existing persistent ALSA capture cadence.
        self.frame_length = sample_rate // 50

    @classmethod
    def from_environment(cls):
        path = model_path()
        if not (path / "am" / "final.mdl").is_file():
            raise VoskWakeWordError(
                f"Vosk wake model is missing at {path}. Run `python3 -m src.vosk_setup` once; "
                "ordinary startup never downloads models")
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as exc:
            machine = platform.machine() or "unknown architecture"
            raise VoskWakeWordError(
                f"Vosk is unavailable for Python {platform.python_version()} on {machine}. "
                "Run the setup command; it will report installation compatibility.") from exc
        SetLogLevel(-1)
        variants = wake_variants()
        # Vosk's small model supports a dynamic grammar. [unk] lets decoding
        # continue through unrelated speech without treating it as a wake hit.
        grammar = json.dumps(sorted(variants) + ["[unk]"])
        try:
            recognizer = KaldiRecognizer(Model(str(path)), 16000, grammar)
        except Exception as exc:
            raise VoskWakeWordError(f"Cannot initialize Vosk wake model at {path}: {exc}") from exc
        return cls(recognizer, sample_rate=16000, variants=variants)

    def detected(self, pcm: bytes) -> bool:
        if len(pcm) % 2:
            raise ValueError("Vosk wake input must be S16_LE PCM")
        accepted = self._recognizer.AcceptWaveform(pcm)
        payload = self._recognizer.Result() if accepted else self._recognizer.PartialResult()
        return bool(_tokens(payload) & self.variants)

    def close(self) -> None:
        self._recognizer = None
