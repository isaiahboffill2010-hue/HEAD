"""Offline energy versus commanded-mouth diagnostic; it never opens hardware."""
from __future__ import annotations

import argparse
import audioop
import csv
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
        peak = noise_floor = 1.0; recent_speech_frames = 0; smooth = position = commanded = 0.0; records = []
        for index, pcm in enumerate(frames):
            rms = audioop.rms(pcm, 2); peak = max(float(rms), peak * .995)
            threshold = peak * (.07 / tuning.sensitivity); base_speech = rms >= threshold
            if not base_speech: noise_floor = noise_floor * .98 + float(rms) * .02
            rescue = max(noise_floor * 3, peak * tuning.final_speech_floor_ratio)
            speech = base_speech or (recent_speech_frames > 0 and rms >= rescue)
            raw = 0.0 if not speech else min(tuning.max_openness, (rms / peak) ** .5 * tuning.max_openness)
            recent_speech_frames = 10 if speech else max(0, recent_speech_frames - 1)
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


def compare_recordings(mouth_csv: Path, jaw_csv: Path) -> dict[str, float | int]:
    """Summarize recorded scheduling and command-call timing; no position inference."""
    with mouth_csv.open(newline="", encoding="utf-8") as source: motion = list(csv.DictReader(source))
    with jaw_csv.open(newline="", encoding="utf-8") as source: jaws = list(csv.DictReader(source))
    numbers = lambda rows, name: [float(row[name]) for row in rows if row.get(name) not in (None, "")]
    return {"analysis_frames": len(motion), "jaw_commands": len(jaws),
            "max_scheduled_lateness_ms": max(numbers(motion, "command_lateness_ms") or [0]),
            "max_hardware_lateness_ms": max(numbers(motion, "hardware_lateness_ms") or [0]),
            "max_command_duration_ms": max(numbers(jaws, "command_duration_ms") or [0]),
            "max_p8_write_ms": max(numbers(jaws, "upper_write_ms") or [0]),
            "max_p10_write_ms": max(numbers(jaws, "lower_write_ms") or [0]),
            "coalesced_frames": int(sum(numbers(motion, "coalesced_frames") or [0]))}


def final_speech_report(wav_path: Path, mouth_csv: Path, jaw_csv: Path, output: Path) -> dict[str, int | float]:
    """Write the final two seconds: PCM energy, targets, and commanded angles."""
    with wave.open(str(wav_path), "rb") as source:
        duration = source.getnframes() / source.getframerate()
    with mouth_csv.open(newline="", encoding="utf-8") as source: motion = list(csv.DictReader(source))
    with jaw_csv.open(newline="", encoding="utf-8") as source: jaws = list(csv.DictReader(source))
    final = [row for row in motion if float(row.get("audio_time") or 0) >= max(0, duration - 2)]
    jaw_times = [(float(row.get("timestamp") or 0), row) for row in jaws]
    fields = ("audio_time", "rms", "threshold", "target", "openness", "commanded", "scheduled_at",
              "hardware_started_at", "hardware_completed_at", "hardware_lateness_ms", "hardware_duration_ms",
              "jaw_command_timestamp", "upper_jaw_angle", "lower_jaw_angle", "upper_write_ms", "lower_write_ms")
    with output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for row in final:
            moment = float(row.get("hardware_completed_at") or 0)
            nearest = min(jaw_times, key=lambda pair: abs(pair[0] - moment))[1] if moment and jaw_times else {}
            writer.writerow({**row, "jaw_command_timestamp": nearest.get("timestamp", ""),
                             "upper_jaw_angle": nearest.get("upper_jaw_angle", ""),
                             "lower_jaw_angle": nearest.get("lower_jaw_angle", ""),
                             "upper_write_ms": nearest.get("upper_write_ms", ""),
                             "lower_write_ms": nearest.get("lower_write_ms", "")})
    return {"duration_s": duration, "final_frames": len(final),
            "energy_frames": sum(float(row.get("rms") or 0) > 0 for row in final),
            "nonzero_targets": sum(float(row.get("target") or 0) > 0 for row in final),
            "hardware_commands": sum(float(row.get("hardware_completed_at") or 0) > 0 for row in final)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze recorded Despina WAV without servo hardware")
    parser.add_argument("wav", type=Path); parser.add_argument("--csv", type=Path)
    parser.add_argument("--svg", type=Path); parser.add_argument("--recorded-mouth-csv", type=Path)
    parser.add_argument("--jaw-csv", type=Path); args = parser.parse_args(argv)
    records = analyze_wav(args.wav)
    csv_path = args.csv or args.wav.with_suffix(".offline-mouth.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=("audio_time", "rms", "threshold", "target", "openness", "commanded"))
        writer.writeheader(); writer.writerows(records)
    svg_path = args.svg or args.wav.with_suffix(".offline-mouth.svg"); write_svg(records, svg_path)
    print(f"Wrote {csv_path} and {svg_path}; no hardware was initialized.")
    if args.recorded_mouth_csv or args.jaw_csv:
        if not (args.recorded_mouth_csv and args.jaw_csv):
            parser.error("--recorded-mouth-csv and --jaw-csv must be supplied together")
        summary = compare_recordings(args.recorded_mouth_csv, args.jaw_csv)
        print("Recorded command timing: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
        final_path = args.wav.with_suffix(".final-speech.csv")
        final = final_speech_report(args.wav, args.recorded_mouth_csv, args.jaw_csv, final_path)
        print(f"Final-speech report: {final_path}; " + ", ".join(f"{key}={value}" for key, value in final.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
