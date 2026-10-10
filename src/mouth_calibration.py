"""Safety-first individual jaw calibration and one-shot combined mouth test."""
from __future__ import annotations

import argparse
import sys
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


def speed_calibration(configs, controller, path: Path = DEFAULT_SETTINGS, input_func=input) -> int:
    """Supervised, progressively faster one-cycle jaw test.

    Every stage requires an immediate RUN confirmation. A speed is persisted
    only after the operator explicitly reports a clean, unstrained cycle.
    """
    settings = load_motion_settings(path)
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
    stages = [speed for speed in (8, 12, 16, 20, 25, 30, 40, 50, 60)
              if speed <= settings.hard_software_speed_cap_dps]
    mouth = MouthController(controller)
    accepted = settings.verified_practical_speed_dps
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
        accepted = float(speed)
        settings = replace(settings, verified_practical_speed_dps=accepted)
        save_motion_settings(settings, path)
        print(f"Accepted and saved {speed:g} deg/s as the verified practical speed.")
    if accepted is None:
        print("No speed was accepted; conservative speech speed remains in use.")
    else:
        print(f"Calibration finished. Normal lip sync will use {accepted:g} deg/s.")
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
        return speed_calibration(configs, controller, args.motion_config)
    except KeyboardInterrupt:
        print("\nSTOPPED: no future commands will be sent. Disconnect servo power physically if unsafe.", file=sys.stderr)
        return 130
    except (ServoError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
