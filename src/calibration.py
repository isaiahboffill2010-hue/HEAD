"""Interactive, one-servo-at-a-time calibration CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .servo_controller import (ConfigurationError, LimitError, MockBackend,
    RobotHatBackend, ServoController, ServoError, load_config, save_config,
    validate_config)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "servos.json"
SAFETY = """SAFETY CHECK
- Support the head so the selected joint cannot drop or pull the mechanism.
- Clear hands, wires, cardboard, brackets, and tools from the full linkage path.
- Verify servo power and keep its physical disconnect within immediate reach.
- Ctrl+C stops future software commands; it cannot guarantee instant torque removal.
  In an emergency, physically disconnect servo power."""


def confirmed_first_move() -> bool:
    print(SAFETY)
    print("The first absolute command may produce a LARGE movement. No center is assumed.")
    return input("After checking that the proposed angle is mechanically safe, type MOVE: ").strip() == "MOVE"


def read_angle(prompt: str) -> float:
    value = float(input(prompt))
    if not -90 <= value <= 90:
        raise ValueError("angle must be between -90 and 90")
    return value


def calibrate(name, configs, controller: ServoController, path: Path) -> int:
    cfg = configs[name]
    print(f"Calibrating only {name} on {cfg.port}. No command has been sent.")
    direction_text = input(f"Direction is {cfg.direction}. Enter 1 or -1 (Enter keeps it): ").strip()
    if direction_text:
        if direction_text not in ("1", "-1"):
            print("Direction must be 1 or -1; nothing moved.", file=sys.stderr)
            return 2
        cfg.direction = int(direction_text)
    try:
        initial = read_angle("Enter a mechanically safe initial logical angle (-90..90): ")
    except ValueError as exc:
        print(f"Invalid angle: {exc}; nothing moved.", file=sys.stderr)
        return 2
    print(f"Proposed: {name} logical {initial:g}°, Robot HAT command {initial * cfg.direction:g}°.")
    if not confirmed_first_move():
        print("Confirmation not received; nothing moved.")
        return 1
    controller.establish_position(name, initial)
    print(f"Known starting position established at {initial:g}°.")
    print("Commands: +1, +2, -1, -2, min, max, neutral, show, save, quit")
    while True:
        command = input(f"{name} [{controller.position(name):g}°]> ").strip().lower()
        if command in {"+1", "+2", "-1", "-2"}:
            try:
                position = controller.calibration_step(name, int(command))
                print(f"Now {position:g}° (Robot HAT {position * cfg.direction:g}°).")
            except ServoError as exc:
                print(f"Rejected: {exc}")
        elif command in {"min", "max", "neutral"}:
            position = controller.position(name)
            attr = "min_angle" if command == "min" else "max_angle" if command == "max" else "neutral"
            setattr(cfg, attr, position)
            print(f"Recorded {command} = {position:g}° in memory (not saved yet).")
        elif command == "show":
            print(f"port={cfg.port}, direction={cfg.direction}, min={cfg.min_angle}, neutral={cfg.neutral}, max={cfg.max_angle}")
        elif command == "save":
            try:
                validate_config(name, cfg, require_calibrated=True)
                save_config(path, configs)
            except ServoError as exc:
                print(f"Cannot save: {exc}")
            else:
                print(f"Saved {name} calibration to {path}.")
                return 0
        elif command in {"quit", "q"}:
            print("Left calibration without saving changes.")
            return 0
        else:
            print("Unknown command; use only the commands shown above.")


def test_servo(name, configs, controller: ServoController, speed: float) -> int:
    cfg = configs[name]
    validate_config(name, cfg, require_calibrated=True)
    print(f"Testing only {name} on {cfg.port}; calibrated range {cfg.min_angle}..{cfg.max_angle}°. No command has been sent.")
    start = read_angle("Enter the servo's currently safe logical angle to establish position: ")
    assert cfg.min_angle is not None and cfg.max_angle is not None
    if not cfg.min_angle <= start <= cfg.max_angle:
        raise LimitError("Starting angle is outside calibrated limits")
    if not confirmed_first_move():
        print("Confirmation not received; nothing moved.")
        return 1
    controller.establish_position(name, start)
    print("Enter targets within calibrated limits, or q to quit.")
    while True:
        text = input(f"{name} [{controller.position(name):g}°] target> ").strip().lower()
        if text in {"q", "quit"}:
            return 0
        try:
            target = float(text)
            controller.move_to(name, target, speed_dps=speed)
            print(f"Reached {target:g}°.")
        except (ValueError, ServoError) as exc:
            print(f"Rejected: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe Robot HAT servo calibration")
    parser.add_argument("--mock", action="store_true", help="print commands instead of using hardware")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list configured servos and ports")
    cal = sub.add_parser("calibrate", help="calibrate exactly one servo")
    cal.add_argument("servo")
    test = sub.add_parser("test", help="manually test one calibrated servo")
    test.add_argument("servo")
    test.add_argument("--speed", type=float, default=30.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configs = load_config(args.config)
        if args.command == "list":
            for name, cfg in configs.items():
                state = "calibrated" if cfg.calibrated else "UNCALIBRATED"
                print(f"{name:<10} {cfg.port:<3} {state}")
            return 0
        if args.servo not in configs:
            raise ConfigurationError(f"Unknown servo {args.servo!r}; use the list command")
        controller = ServoController(configs, MockBackend() if args.mock else RobotHatBackend())
        if args.command == "calibrate":
            return calibrate(args.servo, configs, controller, args.config)
        return test_servo(args.servo, configs, controller, args.speed)
    except KeyboardInterrupt:
        print("\nSTOPPED: no future commands will be sent. Disconnect servo power physically if unsafe.", file=sys.stderr)
        return 130
    except (ServoError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
