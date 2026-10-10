"""Google GenAI conversation, Despina TTS, and optional word-timing adapter."""
from __future__ import annotations
import base64
import json
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GeminiError(RuntimeError): pass

def load_dotenv(path: Path = Path(".env.local")) -> None:
    """Minimal local env loader; existing environment values always win."""
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    chat_model: str
    tts_model: str
    voice: str
    align_model: str

    @classmethod
    def from_environment(cls) -> "GeminiConfig":
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY", "")
        if not key: raise GeminiError("GEMINI_API_KEY is missing; create .env.local from .env.example")
        return cls(key, os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
                   os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"),
                   os.getenv("GEMINI_TTS_VOICE", "Despina"),
                   os.getenv("GEMINI_ALIGN_MODEL", "gemini-3.5-transcribe"))

def _audio_bytes(value: Any) -> bytes:
    if isinstance(value, bytes): return value
    if isinstance(value, str): return base64.b64decode(value)
    raise GeminiError("TTS response did not contain audio bytes")

def write_wav(path: Path, audio: bytes, mime_type: str | None = None, sample_rate: int = 24000) -> Path:
    """Preserve unary WAV or wrap known raw 16-bit PCM returned by an API."""
    if audio[:4] == b"RIFF":
        path.write_bytes(audio); return path
    if mime_type and ("l16" in mime_type.lower() or "pcm" in mime_type.lower()):
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(sample_rate); out.writeframes(audio)
        return path
    raise GeminiError(f"Unsupported non-WAV TTS audio format: {mime_type or 'unknown'}")

def parse_word_timings(text: str) -> list[dict[str, float]]:
    """Parse [{word,start,end}] JSON produced by a supported aligner."""
    match = re.search(r"\[[\s\S]*\]", text)
    if not match: return []
    try: raw = json.loads(match.group(0))
    except json.JSONDecodeError: return []
    result = []
    for item in raw if isinstance(raw, list) else []:
        try:
            start, end = float(item["start"]), float(item["end"])
            if 0 <= start <= end: result.append({"start": start, "end": end})
        except (KeyError, TypeError, ValueError): pass
    return result

class GeminiService:
    def __init__(self, config: GeminiConfig, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            try:
                from google import genai
                client = genai.Client(api_key=config.api_key)
            except ImportError as exc: raise GeminiError("Install dependencies: pip install -r requirements.txt") from exc
        self.client = client
        self.history: list[dict[str, str]] = []

    def reply(self, user_text: str) -> str:
        prompt = "You are HEAD, a friendly cardboard animatronic robot. Reply naturally and briefly.\n"
        for item in self.history[-8:]: prompt += f"{item['role']}: {item['text']}\n"
        prompt += f"user: {user_text}\nassistant:"
        try:
            response = self.client.models.generate_content(model=self.config.chat_model, contents=prompt)
            text = getattr(response, "text", "").strip()
        except Exception as exc: raise GeminiError(f"Gemini conversation failed: {exc}") from exc
        if not text: raise GeminiError("Gemini returned an empty response")
        self.history.extend(({"role": "user", "text": user_text}, {"role": "assistant", "text": text}))
        return text

    def synthesize(self, text: str, destination: Path) -> Path:
        """Use official Google GenAI TTS interactions API and retain returned WAV."""
        try:
            interaction = self.client.interactions.create(
                model=self.config.tts_model,
                input=[{"type": "user_input", "content": [{"type": "text", "text": text,
                        "annotations": [{"type": "speech_metadata", "style": "friendly conversational robot"}]}]}],
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": self.config.voice}]},
            )
            output = interaction.output_audio
            return write_wav(destination, _audio_bytes(output.data), getattr(output, "mime_type", "audio/wav"),
                             getattr(output, "sample_rate", 24000))
        except GeminiError: raise
        except Exception as exc: raise GeminiError(f"Gemini TTS failed: {exc}") from exc

    def word_timings(self, wav_path: Path) -> list[dict[str, float]]:
        """Attempt supported transcription timing; caller must use energy fallback on failure."""
        try:
            uploaded = self.client.files.upload(file=str(wav_path))
            prompt = ("Return ONLY JSON list of words with seconds: "
                      '[{"word":"...","start":0.0,"end":0.2}]. Include word-level timings.')
            response = self.client.interactions.create(model=self.config.align_model, input=[
                {"type": "audio", "uri": uploaded.uri, "mime_type": getattr(uploaded, "mime_type", "audio/wav")},
                {"type": "text", "text": prompt},
            ])
            return parse_word_timings(getattr(response, "output_text", ""))
        except Exception:
            return []
