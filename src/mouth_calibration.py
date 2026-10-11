"""Safety-first individual jaw calibration and one-shot combined mouth test."""
from __future__ import annotations

import argparse
import sys
import statistics
import time
from pathlib import Path

from .calibration import DEFAULT_CONFIG, SAFETY
from .mouth import JAW_NAMES, MouthController
from dataclasses import replace

from .adaptive_lipsync import DEFAULT_SETTINGS, load_motion_settings, save_motion_settings
from .servo_controller import (ConfigurationError, MockBackend, RobotHatBackend,
    ServoController, ServoError, load_config, save_config)

JAW_BY_COMMAND = {"upper": "upper_jaw", "lower": "lower_jaw"}


def _known_reference(name, cfg, controller, input_func=input) -> bool:
    print(f"{name} saved closed neutral is {cfg.neutral:g} deg. No command has been sent.")
    print("This tool cannot read a positional servo's shaft angle. Continue only if you know")
    print("its current commanded angle, power remained continuous, and no linkage was repositioned.")
    try:
        current = float(input_func("Enter that current commanded logical angle (-90..90): "))
    except ValueError:
        print("Invalid angle; nothing moved.")
        return False
    if not -90 <= current <= 90:
        print("Angle must be between -90 and 90; nothing moved.")
        return False
    print(SAFETY)
    if input_func("Type POWERED to confirm continuous power and no repositioning: ").strip() != "POWERED":
        print("Reference not confirmed; nothing moved. Re-establish a safe reference first.")
        return False
    controller.adopt_confirmed_position(name, current)
    print(f"Reference recorded at {current:g} deg without a servo command.")
    return True


def calibrate_jaw(name, configs, controller, path: Path, input_func=input) -> int:
    cfg = configs[name]
    if cfg.neutral is None:
        raise ConfigurationError(f"{name} has no saved closed neutral")
    print(f"Calibrating only {name} on {cfg.port}; saved neutral will not be changed.")
    if not _known_reference(name, cfg, controller, input_func):
        return 1
    lower, upper = max(-90.0, cfg.neutral - 5.0), min(90.0, cfg.neutral + 5.0)
    print(f"Initial provisional window is {lower:g}..{upper:g} deg around closed neutral.")
    print("Use + or - for one degree. Watch which direction opens the jaw. Commands: +, -, extend +, extend -, SAVE, STOP, quit")
    while True:
        command = input_func(f"{name} [{controller.position(name):g} deg]> ").strip()
        if command in {"+", "+1"}:
            delta = 1
        elif command in {"-", "-1"}:
            delta = -1
        elif command.upper() == "STOP":
            print("Stopped: no future commands will be sent. Disconnect servo power physically if unsafe.")
            return 1
        elif command.upper() == "SAVE":
            endpoint = controller.position(name)
            assert endpoint is not None
            if endpoint == cfg.neutral:
                print("Cannot save: move at least 1 degree in the observed opening direction first.")
                continue
            cfg.opening_endpoint = endpoint
            cfg.opening_direction = 1 if endpoint > cfg.neutral else -1
            save_config(path, configs)
            print(f"Saved {name} opening endpoint {endpoint:g} deg, direction {cfg.opening_direction}. Neutral remains {cfg.neutral:g} deg.")
            return 0
        elif command.lower() in {"quit", "q"}:
            print("Left jaw calibration without saving changes.")
            return 0
        elif command.lower() in {"extend +", "extend -"}:
            direction = 1 if command.endswith("+") else -1
            current = controller.position(name)
            boundary = upper if direction > 0 else lower
            if current != boundary:
                print(f"Rejected: extension is allowed only at the {'upper' if direction > 0 else 'lower'} boundary ({boundary:g} deg).")
                continue
            proposed = min(90.0, upper + 5.0) if direction > 0 else max(-90.0, lower - 5.0)
            if proposed == boundary:
                print("Rejected: reached the Robot HAT hardware angle limit.")
                continue
            print(SAFETY)
            print("Check clearance and stop for any resistance, binding, or strain.")
            if input_func("Type CLEAR to add 5 degrees on this side (no movement occurs): ").strip() != "CLEAR":
                print("Extension not confirmed; no movement sent.")
                continue
            if direction > 0:
                upper = proposed
            else:
                lower = proposed
            print(f"Provisional window is now {lower:g}..{upper:g} deg. No movement was sent.")
            continue
        else:
            print("Unknown command; use +, -, extend +, extend -, SAVE, STOP, or quit.")
            continue
        target = controller.position(name) + delta
        if not lower <= target <= upper:
            needed = "extend +" if target > upper else "extend -"
            print(f"Stopped at provisional boundary {lower:g}..{upper:g} deg. Inspect clearance before {needed}.")
            continue
        controller.calibration_step(name, delta)
        print(f"Now {controller.position(name):g} deg. Stop immediately if the jaw strains or binds.")


def combined_test(configs, controller, speed: float, input_func=input) -> int:
    load_motion_settings().speed(speed)
    for name in JAW_NAMES:
        cfg = configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None or cfg.opening_direction is None:
            raise ConfigurationError(f"{name} opening range is not calibrated")
    print("Combined mouth test: no command has been sent.")
    for name in JAW_NAMES:
        cfg = configs[name]
        print(f"  {name}: closed {cfg.neutral:g} deg -> open {cfg.opening_endpoint:g} deg (direction {cfg.opening_direction})")
        if not _known_reference(name, cfg, controller, input_func):
            return 1
    print("Planned one cycle: both jaws open slowly to their saved endpoints, then close to saved neutral.")
    print(SAFETY)
    if input_func("Type CYCLE to run this one slow open-and-close cycle: ").strip() != "CYCLE":
        print("Cycle not confirmed; nothing moved.")
        return 1
    mouth = MouthController(controller)
    mouth.open(1.0, speed)
    mouth.close(speed)
    print("One mouth cycle completed; both jaws were commanded back to their saved closed neutral positions.")
    return 0


def speed_calibration(configs, controller, path: Path = DEFAULT_SETTINGS, input_func=input,
                      target_speed: float | None = None) -> int:
    """Supervised, progressively faster one-cycle jaw test.

    Every stage requires an immediate RUN confirmation. A speed is persisted
    only after the operator explicitly reports a clean, unstrained cycle.
    """
    settings = load_motion_settings(path)
    maximum = settings.maximum_speed()
    if target_speed is not None:
        if maximum is None:
            raise ValueError("cannot select a target speed before a maximum speed has been physically verified")
        settings.speed(target_speed)
        if target_speed > maximum:
            raise ValueError("target speed exceeds the existing verified maximum")
    for name in JAW_NAMES:
        cfg = configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None or cfg.opening_direction is None:
            raise ConfigurationError(f"{name} opening range is not calibrated")
    print("Mouth speed calibration: no command has been sent.")
    print("Keep a hand at servo power. Stop for binding, chatter, heat, impact, or strain.")
    for name in JAW_NAMES:
        cfg = configs[name]
        print(f"  {name}: closed {cfg.neutral:g} deg -> open {cfg.opening_endpoint:g} deg")
        if not _known_reference(name, cfg, controller, input_func):
            return 1
        if controller.position(name) != cfg.neutral:
            print("Speed calibration starts only with both jaws at saved closed neutral.")
            return 1
    if target_speed is not None:
        assert maximum is not None
        print(f"\nTarget operating speed: {target_speed:g} deg/s (verified maximum remains {maximum:g} deg/s).")
        command = input_func(f"Type RUN {target_speed:g} to test this operating speed, or STOP: ").strip()
        if command != f"RUN {target_speed:g}":
            print("Target-speed test not confirmed; no movement was sent."); return 1
        mouth = MouthController(controller); mouth.open(1.0, target_speed); mouth.close(target_speed)
        if input_func("Type PASS only if motion was clean, or STOP: ").strip().upper() != "PASS":
            print("Target speed was not saved. Disconnect power if anything looked unsafe."); return 1
        save_motion_settings(replace(settings, selected_operating_speed_dps=float(target_speed)), path)
        print(f"Saved operating speed {target_speed:g} deg/s; verified maximum remains {maximum:g} deg/s.")
        return 0
    stages = [speed for speed in (8, 12, 16, 20, 25, 30, 40, 50, 60)
              if speed <= settings.hard_software_speed_cap_dps]
    mouth = MouthController(controller)
    accepted = maximum
    for speed in stages:
        if accepted is not None and speed <= accepted:
            continue
        print(f"\nNext stage: one full open/close cycle at {speed:g} deg/s.")
        command = input_func(f"Type RUN {speed:g} to execute, or STOP to finish: ").strip()
        if command.upper() == "STOP":
            break
        if command != f"RUN {speed:g}":
            print("Exact confirmation not entered; calibration stopped without movement at this stage.")
            break
        mouth.open(1.0, speed)
        mouth.close(speed)
        result = input_func("Type PASS only if motion was clean, or STOP: ").strip().upper()
        if result != "PASS":
            print("Stage not accepted; it was not saved. Disconnect power if anything looked unsafe.")
            break
        prior_maximum = accepted; accepted = float(speed)
        selected = settings.selected_operating_speed_dps
        # Preserve an explicitly chosen lower operating speed; legacy behavior
        # otherwise selects the newly verified maximum for normal speech.
        if selected is None or selected == prior_maximum:
            selected = accepted
        settings = replace(settings, verified_practical_speed_dps=accepted,
                           maximum_verified_speed_dps=accepted, selected_operating_speed_dps=selected)
        save_motion_settings(settings, path)
        print(f"Accepted and saved {speed:g} deg/s as the verified practical speed.")
    if accepted is None:
        print("No speed was accepted; conservative speech speed remains in use.")
    else:
        print(f"Calibration finished. Verified maximum is {accepted:g} deg/s; normal lip sync uses {settings.speed():g} deg/s.")
    return 0


def latency_benchmark(configs, controller, samples: int = 20, input_func=input) -> int:
    """Supervised P8/P10 software-call benchmark; no Gemini, audio, or config writes."""
    settings = load_motion_settings(); speed = settings.speed()
    if samples < 4 or samples > 100:
        raise ValueError("samples must be between 4 and 100")
    for name in JAW_NAMES:
        cfg = configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None or cfg.opening_direction is None:
            raise ConfigurationError(f"{name} opening range is not calibrated")
    print("Jaw command-latency benchmark: no command has been sent.")
    print("This measures software call duration only, not actual shaft motion or I2C electrical timing.")
    print("It alternates only 0% and 10% openness within your saved jaw endpoints at the existing safe speed.")
    for name in JAW_NAMES:
        if not _known_reference(name, configs[name], controller, input_func): return 1
    if any(controller.position(name) != configs[name].neutral for name in JAW_NAMES):
        print("Benchmark starts only with both jaws at saved closed neutral."); return 1
    print(SAFETY)
    if input_func(f"Type BENCHMARK {samples} to run, or STOP: ").strip() != f"BENCHMARK {samples}":
        print("Benchmark not confirmed; nothing moved."); return 1
    mouth = MouthController(controller); openness = 0.0
    for _ in range(samples):
        openness = .1 if openness == 0 else 0.0
        mouth.set_openness_step(openness)
        # The next target is no further than 2 degrees (upper jaw), so this
        # delay preserves the current verified/conservative speed limit.
        time.sleep((.1 * max(abs(configs[n].opening_endpoint - configs[n].neutral) for n in JAW_NAMES)) / speed)
    mouth.set_openness_step(0.0)
    def report(label, values):
        ordered = sorted(values)
        percentile = lambda pct: ordered[min(len(ordered) - 1, round((len(ordered) - 1) * pct))]
        print(f"{label}: median={statistics.median(values):.2f} ms p95={percentile(.95):.2f} ms p99={percentile(.99):.2f} ms max={max(values):.2f} ms")
    report("both-jaw command call", [row["command_duration_ms"] for row in mouth.command_log])
    report("P8 write", [row["upper_write_ms"] for row in mouth.command_log])
    report("P10 write", [row["lower_write_ms"] for row in mouth.command_log])
    print("Both jaws were commanded closed. Compare these software timings with streaming debug CSV timings.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Safe individual jaw calibration and one-shot mouth test")
    parser.add_argument("--mock", action="store_true", help="print commands instead of using Robot HAT hardware")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upper", help="calibrate upper jaw only")
    sub.add_parser("lower", help="calibrate lower jaw only")
    test = sub.add_parser("test", help="run one confirmed open-and-close mouth cycle")
    test.add_argument("--speed", type=float, default=8.0,
                      help="degrees per second; capped by config/mouth_motion.json")
    speed = sub.add_parser("speed", help="interactively test and save progressively faster mouth speeds")
    speed.add_argument("--motion-config", type=Path, default=DEFAULT_SETTINGS)
    speed.add_argument("--target-speed", type=float,
                       help="supervised lower operating speed; must not exceed existing verified maximum")
    bench = sub.add_parser("latency", help="supervised P8/P10 software-call latency benchmark")
    bench.add_argument("--samples", type=int, default=20)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configs = load_config(args.config)
        controller = ServoController(configs, MockBackend() if args.mock else RobotHatBackend())
        if args.command in JAW_BY_COMMAND:
            return calibrate_jaw(JAW_BY_COMMAND[args.command], configs, controller, args.config)
        if args.command == "test":
            return combined_test(configs, controller, args.speed)
        if args.command == "speed":
            return speed_calibration(configs, controller, args.motion_config, target_speed=args.target_speed)
        return latency_benchmark(configs, controller, args.samples)
    except KeyboardInterrupt:
        print("\nSTOPPED: no future commands will be sent. Disconnect servo power physically if unsafe.", file=sys.stderr)
        return 130
    except (ServoError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
