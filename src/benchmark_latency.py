"""Real-API comparison of complete-response and streaming availability latency."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from .gemini_service import GeminiConfig, GeminiService


DEFAULT_PROMPTS = ("What color is the sky?", "Tell me one short robot joke.")


def benchmark(service: GeminiService, prompts=DEFAULT_PROMPTS) -> list[dict]:
    results = []
    for prompt in prompts:
        started = time.monotonic()
        text = service.reply_complete(prompt)
        text_done = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="head-benchmark-") as directory:
            service.synthesize(text, Path(directory) / "complete.wav")
        unary_audio = time.monotonic()

        stream_started = time.monotonic()
        first_token = None
        parts = []
        for part in service.reply_stream(prompt):
            first_token = first_token or time.monotonic()
            parts.append(part)
        stream_text_done = time.monotonic()
        first_audio = None
        for _, _decode_ms in service.stream_speech("".join(parts)):
            first_audio = time.monotonic()
            break
        if first_audio is None:
            raise RuntimeError("Streaming TTS returned no audio")
        results.append({
            "prompt": prompt,
            "complete_text_ms": (text_done - started) * 1000,
            "complete_first_audio_available_ms": (unary_audio - started) * 1000,
            "stream_first_token_ms": (first_token - stream_started) * 1000,
            "stream_complete_text_ms": (stream_text_done - stream_started) * 1000,
            "stream_first_audio_available_ms": (first_audio - stream_started) * 1000,
        })
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark real Gemini complete vs streaming latency")
    parser.add_argument("prompts", nargs="*", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    service = GeminiService(GeminiConfig.from_environment())
    service.verify_models()
    data = benchmark(service, args.prompts)
    rendered = json.dumps(data, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
