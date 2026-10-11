"""Offline energy versus commanded-mouth diagnostic; it never opens hardware."""
from __future__ import annotations

import argparse
import audioop
import csv
import html
import wave
from pathlib import Path

from .adaptive_lipsync import load_motion_settings
from .streaming_lipsync import MouthTuning


def analyze_wav(path: Path, *, frame_s: float = .02, max_travel_deg: float = 20.0,
                tuning: MouthTuning | None = None) -> list[dict[str, float | bool]]:
    """Recreate streaming energy/rate logic from a WAV without servo commands."""
    tuning = tuning or MouthTuning.from_environment()
    settings = load_motion_settings(); speed_fraction = settings.speed() / max_travel_deg
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError("diagnostic WAV must be mono 16-bit PCM")
        rate = source.getframerate(); frame_bytes = max(2, int(rate * frame_s) * 2)
        frames = iter(lambda: source.readframes(frame_bytes // 2), b"")
        peak = 1.0; smooth = position = commanded = 0.0; records = []
        for index, pcm in enumerate(frames):
            rms = audioop.rms(pcm, 2); peak = max(float(rms), peak * .995)
            threshold = peak * (.07 / tuning.sensitivity)
            raw = 0.0 if rms < threshold else min(tuning.max_openness, (rms / peak) ** .5 * tuning.max_openness)
            target = smooth * tuning.smoothing + raw * (1 - tuning.smoothing); smooth = target
            position += max(-speed_fraction * frame_s, min(speed_fraction * frame_s, target - position))
            emitted = abs(position - commanded) >= tuning.command_deadband_deg / max_travel_deg
            if emitted: commanded = position
            records.append({"audio_time": index * frame_s, "rms": float(rms), "threshold": threshold,
                            "target": target, "openness": position, "commanded": emitted})
    return records


def write_svg(records: list[dict[str, float | bool]], path: Path) -> None:
    """Small dependency-free graph: gray=energy, blue=generated openness, dots=commands."""
    width, height, pad = 1200, 360, 35
    duration = max(.02, float(records[-1]["audio_time"]) + .02) if records else .02
    peak = max([float(row["rms"]) for row in records] or [1])
    points = lambda key, scale: " ".join(f"{pad + float(r['audio_time']) / duration * (width - 2 * pad):.1f},{height - pad - float(r[key]) / scale * (height - 2 * pad):.1f}" for r in records)
    energy = points("rms", peak); openness = points("openness", 1)
    dots = "".join(f'<circle cx="{pad + float(r["audio_time"]) / duration * (width - 2 * pad):.1f}" cy="{height - pad - float(r["openness"]) * (height - 2 * pad):.1f}" r="2" fill="#06c"/>' for r in records if r["commanded"])
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{pad}" y="20">Audio energy (gray), generated jaw openness (blue), emitted commands (dots)</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>
<polyline fill="none" stroke="#888" points="{energy}"/><polyline fill="none" stroke="#06c" points="{openness}"/>{dots}</svg>''', encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze recorded Despina WAV without servo hardware")
    parser.add_argument("wav", type=Path); parser.add_argument("--csv", type=Path)
    parser.add_argument("--svg", type=Path); args = parser.parse_args(argv)
    records = analyze_wav(args.wav)
    csv_path = args.csv or args.wav.with_suffix(".offline-mouth.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=("audio_time", "rms", "threshold", "target", "openness", "commanded"))
        writer.writeheader(); writer.writerows(records)
    svg_path = args.svg or args.wav.with_suffix(".offline-mouth.svg"); write_svg(records, svg_path)
    print(f"Wrote {csv_path} and {svg_path}; no hardware was initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
