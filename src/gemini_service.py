"""Google GenAI conversation, Despina TTS, and optional word-timing adapter."""
from __future__ import annotations
import base64
import json
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


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
    thinking: str = "0"
    max_history_turns: int = 4

    @classmethod
    def from_environment(cls) -> "GeminiConfig":
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY", "")
        if not key: raise GeminiError("GEMINI_API_KEY is missing; create .env.local from .env.example")
        return cls(key, os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite"),
                   os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts"),
                   os.getenv("GEMINI_TTS_VOICE", "Despina"),
                   os.getenv("GEMINI_ALIGN_MODEL", "gemini-3.5-transcribe"),
                   os.getenv("GEMINI_THINKING", "0"),
                   max(1, int(os.getenv("GEMINI_MAX_HISTORY_TURNS", "4"))))

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

    def _prompt(self, user_text: str) -> str:
        prompt = "You are HEAD, a friendly animatronic robot. Answer in one or two short, natural sentences.\n"
        for item in self.history[-self.config.max_history_turns * 2:]:
            prompt += f"{item['role']}: {item['text']}\n"
        return prompt + f"user: {user_text}\nassistant:"

    def _generation_config(self) -> dict[str, Any]:
        value = self.config.thinking.strip().lower()
        config: dict[str, Any] = {"max_output_tokens": 120}
        if value in {"minimal", "low", "medium", "high"}:
            config["thinking_config"] = {"thinking_level": value}
        else:
            config["thinking_config"] = {"thinking_budget": int(value)}
        return config

    def verify_models(self) -> dict[str, str]:
        """Verify configured model IDs are visible to this API key."""
        result = {}
        for role, model in (("conversation", self.config.chat_model), ("TTS", self.config.tts_model)):
            try:
                found = self.client.models.get(model=model)
                result[role] = getattr(found, "name", model)
            except Exception as exc:
                raise GeminiError(f"Configured {role} model {model!r} is unavailable: {exc}") from exc
        return result

    def reply_stream(self, user_text: str, on_first_token=None) -> Iterator[str]:
        """Yield text as Gemini returns it, then update bounded local history."""
        pieces: list[str] = []
        try:
            stream = self.client.models.generate_content_stream(
                model=self.config.chat_model, contents=self._prompt(user_text),
                config=self._generation_config(),
            )
            first = True
            for response in stream:
                text = getattr(response, "text", "") or ""
                if not text:
                    continue
                if first and on_first_token:
                    on_first_token()
                first = False
                pieces.append(text)
                yield text
        except Exception as exc:
            raise GeminiError(f"Gemini conversation failed: {exc}") from exc
        answer = "".join(pieces).strip()
        if not answer:
            raise GeminiError("Gemini returned an empty response")
        self.history.extend(({"role": "user", "text": user_text}, {"role": "assistant", "text": answer}))
        self.history = self.history[-self.config.max_history_turns * 2:]

    def reply(self, user_text: str) -> str:
        return "".join(self.reply_stream(user_text)).strip()

    def reply_complete(self, user_text: str) -> str:
        """Original non-streaming text path, retained for fallback/benchmarks."""
        try:
            response = self.client.models.generate_content(
                model=self.config.chat_model, contents=self._prompt(user_text),
                config=self._generation_config(),
            )
            text = (getattr(response, "text", "") or "").strip()
        except Exception as exc:
            raise GeminiError(f"Gemini conversation failed: {exc}") from exc
        if not text:
            raise GeminiError("Gemini returned an empty response")
        self.history.extend(({"role": "user", "text": user_text}, {"role": "assistant", "text": text}))
        self.history = self.history[-self.config.max_history_turns * 2:]
        return text

    def stream_speech(self, text: str) -> Iterator[tuple[bytes, float]]:
        """Yield raw 24 kHz mono signed-16-bit PCM and decode time in ms."""
        try:
            stream = self.client.interactions.create(
                model=self.config.tts_model,
                input=[{"type": "user_input", "content": [{"type": "text", "text": text,
                        "annotations": [{"type": "speech_metadata", "style": "friendly conversational robot"}]}]}],
                response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
                generation_config={"speech_config": [{"voice": self.config.voice}]},
                stream=True,
            )
            import time
            for event in stream:
                if getattr(event, "event_type", "") != "step.delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", "") != "audio":
                    continue
                started = time.monotonic()
                data = _audio_bytes(getattr(delta, "data", b""))
                decode_ms = (time.monotonic() - started) * 1000
                if data:
                    yield data, decode_ms
        except GeminiError:
            raise
        except Exception as exc:
            raise GeminiError(f"Streaming Gemini TTS failed: {exc}") from exc

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
