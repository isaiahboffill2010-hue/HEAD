"""Google GenAI conversation, Despina TTS, and word-timing adapter."""
from __future__ import annotations
import base64, json, os, re, wave
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class GeminiError(RuntimeError): pass


def load_dotenv(path: Path = Path(".env.local")) -> None:
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str; chat_model: str; tts_model: str; voice: str; align_model: str
    thinking: str = "low"; max_history_turns: int = 4
    max_output_tokens: int = 1024; retry_max_output_tokens: int = 2048
    transcription_model: str = "gemini-3.5-transcribe"; transcription_timeout_s: float = 20.0

    @classmethod
    def from_environment(cls) -> "GeminiConfig":
        load_dotenv(); key = os.getenv("GEMINI_API_KEY", "")
        if not key: raise GeminiError("GEMINI_API_KEY is missing; create .env.local from .env.example")
        return cls(key, os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"),
            os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"),
            os.getenv("GEMINI_TTS_VOICE", "Despina"), os.getenv("GEMINI_ALIGN_MODEL", "gemini-3.5-transcribe"),
            os.getenv("GEMINI_THINKING", "low"), max(1, int(os.getenv("GEMINI_MAX_HISTORY_TURNS", "4"))),
            max(1, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))),
            max(1, int(os.getenv("GEMINI_RETRY_MAX_OUTPUT_TOKENS", "2048"))),
            os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.5-transcribe"),
            max(1.0, float(os.getenv("GEMINI_TRANSCRIPTION_TIMEOUT_S", "20"))))


def _audio_bytes(value: Any) -> bytes:
    if isinstance(value, bytes): return value
    if isinstance(value, str): return base64.b64decode(value)
    raise GeminiError("TTS response did not contain audio bytes")


def write_wav(path: Path, audio: bytes, mime_type: str | None = None, sample_rate: int = 24000) -> Path:
    if audio[:4] == b"RIFF": path.write_bytes(audio); return path
    if mime_type and ("l16" in mime_type.lower() or "pcm" in mime_type.lower()):
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(sample_rate); out.writeframes(audio)
        return path
    raise GeminiError(f"Unsupported non-WAV TTS audio format: {mime_type or 'unknown'}")


def parse_word_timings(text: str) -> list[dict[str, float]]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match: return []
    try: raw = json.loads(match.group(0))
    except json.JSONDecodeError: return []
    result = []
    for item in raw if isinstance(raw, list) else []:
        try:
            start, end = float(item["start"]), float(item["end"])
            if 0 <= start <= end: result.append({"word": str(item.get("word", "")), "start": start, "end": end})
        except (KeyError, TypeError, ValueError): pass
    return result


class GeminiService:
    def __init__(self, config: GeminiConfig, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            try:
                from google import genai; client = genai.Client(api_key=config.api_key)
            except ImportError as exc: raise GeminiError("Install dependencies: pip install -r requirements.txt") from exc
        self.client, self.history = client, []
        self.last_finish_reason = "unknown"; self.last_alignment_error = None
        self.last_generation_usage: dict[str, int | None] = {}; self.last_generation_attempts = 0

    def _prompt(self, user_text: str) -> str:
        prompt = ("You are HEAD, a friendly conversational robot. Reply directly in one to three short, natural "
                  "spoken sentences. Use plain prose only. Never use asterisks, stage directions, roleplay, "
                  "fictional sound effects, or descriptions of blinking, smiling, head movement, or other physical "
                  "behavior unless the user asks about implemented behavior.\n")
        for item in self.history[-self.config.max_history_turns * 2:]: prompt += f"{item['role']}: {item['text']}\n"
        return prompt + f"user: {user_text}\nassistant:"

    def _generation_config(self, budget: int | None = None) -> dict[str, Any]:
        value = self.config.thinking.strip().lower()
        config: dict[str, Any] = {"max_output_tokens": budget or self.config.max_output_tokens}
        if self.config.chat_model == "gemini-3.8-flash":
            if value in {"", "0", "minimal"}: value = "low"
            if value not in {"low", "medium", "high"}:
                raise GeminiError("gemini-3.8-flash supports GEMINI_THINKING=low, medium, or high")
            config["thinking_config"] = {"thinking_level": value}
        elif value in {"minimal", "low", "medium", "high"}:
            config["thinking_config"] = {"thinking_level": value}
        else:
            config["thinking_config"] = {"thinking_budget": int(value)}
        return config

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        candidates = getattr(response, "candidates", None) or []
        if not candidates: return None
        value = getattr(candidates[0], "finish_reason", None)
        return None if value is None else str(getattr(value, "name", value))

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None: return
        self.last_generation_usage = {
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }

    @staticmethod
    def _clean_spoken_text(text: str) -> str:
        text = re.sub(r"\*[^*]*\*", "", text); text = re.sub(r"\[[^\]]*\]", "", text)
        physical = re.compile(r"\b(blink(?:s|ed|ing)?|smil(?:e|es|ed|ing)|whirr(?:s|ed|ing)?|"
                              r"nod(?:s|ded|ding)?|move(?:s|d|ing)? (?:my |the )?head)\b", re.I)
        text = " ".join(part for part in re.split(r"(?<=[.!?])\s+", text) if not physical.search(part))
        return re.sub(r"\s+", " ", text).strip()

    def _collect_stream(self, user_text: str, budget: int, on_first_token=None) -> tuple[list[str], str]:
        pieces, reason, first = [], "unknown", True
        stream = self.client.models.generate_content_stream(model=self.config.chat_model,
            contents=self._prompt(user_text), config=self._generation_config(budget))
        for response in stream:
            reason = self._finish_reason(response) or reason; self._record_usage(response)
            text = getattr(response, "text", "") or ""
            if text:
                if first and on_first_token: on_first_token()
                first = False; pieces.append(text)
        return pieces, reason

    def _save_answer(self, user_text: str, answer: str) -> None:
        self.history.extend(({"role": "user", "text": user_text}, {"role": "assistant", "text": answer}))
        self.history = self.history[-self.config.max_history_turns * 2:]

    def reply_stream(self, user_text: str, on_first_token=None) -> Iterator[str]:
        """Consume all chunks; retry once before exposing any speechable text."""
        self.last_generation_attempts = 1; self.last_generation_usage = {}
        try:
            pieces, reason = self._collect_stream(user_text, self.config.max_output_tokens, on_first_token)
            if "MAX_TOKENS" in reason.upper():
                self.last_generation_attempts = 2
                pieces, reason = self._collect_stream(user_text,
                    max(self.config.retry_max_output_tokens, self.config.max_output_tokens + 1))
        except Exception as exc: raise GeminiError(f"Gemini conversation failed: {exc}") from exc
        self.last_finish_reason = reason; answer = self._clean_spoken_text("".join(pieces))
        if not answer: raise GeminiError("Gemini returned an empty response")
        if "MAX_TOKENS" in reason.upper():
            raise GeminiError("Gemini stopped at the output-token limit after one retry; refusing to speak truncated text")
        self._save_answer(user_text, answer); yield answer

    def reply(self, user_text: str) -> str: return "".join(self.reply_stream(user_text)).strip()

    def reply_complete(self, user_text: str) -> str:
        self.last_generation_attempts = 1; self.last_generation_usage = {}
        try:
            response = self.client.models.generate_content(model=self.config.chat_model, contents=self._prompt(user_text),
                config=self._generation_config())
            reason = self._finish_reason(response) or "unknown"; self._record_usage(response)
            if "MAX_TOKENS" in reason.upper():
                self.last_generation_attempts = 2
                response = self.client.models.generate_content(model=self.config.chat_model, contents=self._prompt(user_text),
                    config=self._generation_config(max(self.config.retry_max_output_tokens, self.config.max_output_tokens + 1)))
                reason = self._finish_reason(response) or "unknown"; self._record_usage(response)
            text = self._clean_spoken_text(getattr(response, "text", "") or "")
        except Exception as exc: raise GeminiError(f"Gemini conversation failed: {exc}") from exc
        self.last_finish_reason = reason
        if not text: raise GeminiError("Gemini returned an empty response")
        if "MAX_TOKENS" in reason.upper():
            raise GeminiError("Gemini stopped at the output-token limit after one retry; refusing to speak truncated text")
        self._save_answer(user_text, text); return text

    def verify_models(self) -> dict[str, str]:
        result = {}
        for role, model in (("conversation", self.config.chat_model), ("TTS", self.config.tts_model),
                            ("transcription", self.config.transcription_model)):
            try: result[role] = getattr(self.client.models.get(model=model), "name", model)
            except Exception as exc: raise GeminiError(f"Configured {role} model {model!r} is unavailable: {exc}") from exc
        return result

    def transcribe_wav(self, path: Path) -> str:
        """Short VAD-bounded WAV transcription; client/model remain reused per session."""
        def request() -> str:
            uploaded = self.client.files.upload(file=str(path))
            response = self.client.interactions.create(model=self.config.transcription_model, input=[
                {"type": "audio", "uri": uploaded.uri, "mime_type": "audio/wav"}],
                generation_config={"transcription_config": {"mode": "smart"}})
            return (getattr(response, "output_text", "") or "").strip()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(request).result(timeout=self.config.transcription_timeout_s)
        except FuturesTimeout as exc:
            raise GeminiError(f"transcription timed out after {self.config.transcription_timeout_s:g} seconds") from exc
        except Exception as exc:
            raise GeminiError(f"Gemini transcription failed: {exc}") from exc

    def stream_speech(self, text: str) -> Iterator[tuple[bytes, float]]:
        try:
            stream = self.client.interactions.create(model=self.config.tts_model,
                input=[{"type": "user_input", "content": [{"type": "text", "text": text,
                "annotations": [{"type": "speech_metadata", "style": "friendly conversational robot"}]}]}],
                response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
                generation_config={"speech_config": [{"voice": self.config.voice}]}, stream=True)
            import time
            for event in stream:
                if getattr(event, "event_type", "") != "step.delta" or getattr(getattr(event, "delta", None), "type", "") != "audio": continue
                began = time.monotonic(); data = _audio_bytes(getattr(event.delta, "data", b""))
                if data: yield data, (time.monotonic() - began) * 1000
        except GeminiError: raise
        except Exception as exc: raise GeminiError(f"Streaming Gemini TTS failed: {exc}") from exc

    def synthesize(self, text: str, destination: Path) -> Path:
        try:
            interaction = self.client.interactions.create(model=self.config.tts_model,
                input=[{"type": "user_input", "content": [{"type": "text", "text": text,
                "annotations": [{"type": "speech_metadata", "style": "friendly conversational robot"}]}]}],
                response_format={"type": "audio"}, generation_config={"speech_config": [{"voice": self.config.voice}]})
            output = interaction.output_audio
            return write_wav(destination, _audio_bytes(output.data), getattr(output, "mime_type", "audio/wav"),
                             getattr(output, "sample_rate", 24000))
        except GeminiError: raise
        except Exception as exc: raise GeminiError(f"Gemini TTS failed: {exc}") from exc

    def word_timings(self, wav_path: Path) -> list[dict[str, float]]:
        try:
            self.last_alignment_error = None; uploaded = self.client.files.upload(file=str(wav_path))
            response = self.client.interactions.create(model=self.config.align_model, input=[
                {"type": "audio", "uri": uploaded.uri, "mime_type": getattr(uploaded, "mime_type", "audio/wav")},
                {"type": "text", "text": 'Return ONLY JSON list of words with seconds: [{"word":"...","start":0.0,"end":0.2}].'}])
            return parse_word_timings(getattr(response, "output_text", ""))
        except Exception as exc: self.last_alignment_error = repr(exc); return []
