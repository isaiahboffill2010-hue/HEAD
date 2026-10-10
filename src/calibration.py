"""Interactive, one-servo-at-a-time calibration CLI."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .servo_controller import (ConfigurationError, LimitError, MockBackend, RobotHatBackend, ServoController, ServoError, load_config, save_config, validate_config)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "servos.json"
SAFETY = """SAFETY CHECK
- Support the head. Keep fingers, wires, cardboard, brackets, and tools clear.
- Stop immediately if a servo strains, chatters, binds, or the mechanism shifts.
- Keep the physical servo-power disconnect within immediate reach.
- Ctrl+C stops future software commands only; it cannot guarantee torque removal.
  In an emergency, physically disconnect servo power."""

def preparation_text(name: str, port: str) -> str:
    return f"""PREPARATION ONLY: {name} on {port}
This command sends NO servo commands.

Standard positional servos cannot report their current shaft angle through the
Robot HAT. Software cannot discover or automatically preserve the head's
current physical position. Do not guess an angle while the linkage is attached.

Mechanical-centering procedure (one servo at a time):
1. Support the head and power OFF the servo system.
2. Temporarily disconnect only this servo's linkage or horn. Keep clear.
3. With the linkage safely disengaged, run `center {name}` and type CENTER
   immediately before it commands 0 degrees.
4. Power OFF again. Set the head to the desired neutral pose by hand and
   reattach the linkage/horn without forcing it.
5. Restore servo power, run `calibrate {name}`, confirm the centered baseline,
   and use only 1- or 2-degree steps to find conservative limits.

Never continue if the servo strains or binds."""

def confirmed_center_move(input_func=input) -> bool:
    print(SAFETY)
    print("The next command is the known Robot HAT center: logical 0 degrees.")
    print("It is safe only with this servo's linkage/horn disengaged or proven clear.")
    return input_func("Type CENTER to command 0 degrees now: ").strip() == "CENTER"

def center(name, configs, controller: ServoController, input_func=input) -> int:
    cfg = configs[name]
    print(preparation_text(name, cfg.port))
    if not confirmed_center_move(input_func):
        print("Confirmation not received; nothing moved.")
        return 1
    controller.establish_position(name, 0)
    print("Center command sent. Power OFF servo power before reattaching the linkage/horn.")
    return 0

def calibrated_baseline(name, configs, controller: ServoController, input_func=input) -> bool:
    print("This program cannot verify shaft position. Continue only after completing")
    print("mechanical centering and reattaching the linkage at the desired neutral pose.")
    print(SAFETY)
    if input_func("Type CENTERED to confirm that procedure is complete: ").strip() != "CENTERED":
        print("Mechanical centering was not confirmed; nothing moved.")
        return False
    if not confirmed_center_move(input_func):
        print("Confirmation not received; nothing moved.")
        return False
    controller.establish_position(name, 0)
    print("Centered baseline established at logical 0 degrees.")
    return True

def align(name, configs, controller: ServoController, path: Path,
          provisional_limit: float = 15.0, input_func=input) -> int:
    """Find neutral from a continuously powered, user-confirmed reference."""
    if not 1 <= provisional_limit <= 20:
        raise ValueError("provisional limit must be between 1 and 20 degrees")
    cfg = configs[name]
    print(f"Aligning only {name} on {cfg.port}. No command has been sent.")
    print("This does NOT read servo position. Use it only when you know the servo's")
    print("previously commanded angle and it has remained powered and unrepositioned.")
    try:
        start = float(input_func("Enter that previously commanded logical angle (-90..90): "))
    except ValueError:
        print("Invalid angle; nothing moved.", file=sys.stderr)
        return 2
    if not -90 <= start <= 90:
        print("Angle must be between -90 and 90; nothing moved.", file=sys.stderr)
        return 2
    print(SAFETY)
    confirmation = input_func(
        "Type POWERED to confirm it remained powered and unrepositioned since that command: "
    ).strip()
    if confirmation != "POWERED":
        print("Reference not confirmed; nothing moved. Establish a safe reference with prepare/center first.")
        return 1
    controller.adopt_confirmed_position(name, start)
    lower, upper = max(-90.0, start - provisional_limit), min(90.0, start + provisional_limit)
    print(f"Trusted reference recorded without a servo command: {start:g} deg.")
    print(f"Provisional software window: {lower:g}..{upper:g} deg. No automatic sweep will occur.")
    print("Commands: +, -, SAVE, quit")
    while True:
        command = input_func(f"{name} [{controller.position(name):g} deg]> ").strip()
        if command in {"+", "+1"}:
            delta = 1
        elif command in {"-", "-1"}:
            delta = -1
        elif command.upper() == "SAVE":
            cfg.neutral = controller.position(name)
            # Alignment establishes a neutral only; all travel limits need a
            # separate cautious calibration around that neutral.
            cfg.min_angle = None
            cfg.max_angle = None
            save_config(path, configs)
            print(f"Saved neutral {cfg.neutral:g} deg only. Movement limits remain UNCALIBRATED.")
            return 0
        elif command.lower() in {"quit", "q"}:
            print("Left alignment without saving changes.")
            return 0
        else:
            print("Unknown command; use +, -, SAVE, or quit.")
            continue
        target = controller.position(name) + delta
        if not lower <= target <= upper:
            print(f"Rejected: {target:g} deg is outside provisional {lower:g}..{upper:g} deg window.")
            continue
        try:
            position = controller.calibration_step(name, delta)
            print(f"Now {position:g} deg (Robot HAT {position * cfg.direction:g} deg).")
        except ServoError as exc:
            print(f"Rejected: {exc}")

def calibrate(name, configs, controller: ServoController, path: Path) -> int:
    cfg = configs[name]
    print(f"Calibrating only {name} on {cfg.port}. No command has been sent.")
    direction_text = input(f"Direction is {cfg.direction}. Enter 1 or -1 (Enter keeps it): ").strip()
    if direction_text:
        if direction_text not in ("1", "-1"):
            print("Direction must be 1 or -1; nothing moved.", file=sys.stderr)
            return 2
        cfg.direction = int(direction_text)
    if not calibrated_baseline(name, configs, controller):
        return 1
    print("Commands: +1, +2, -1, -2, min, max, neutral, show, save, quit")
    while True:
        command = input(f"{name} [{controller.position(name):g} deg]> ").strip().lower()
        if command in {"+1", "+2", "-1", "-2"}:
            try:
                position = controller.calibration_step(name, int(command))
                print(f"Now {position:g} deg (Robot HAT {position * cfg.direction:g} deg).")
            except ServoError as exc:
                print(f"Rejected: {exc}")
        elif command in {"min", "max", "neutral"}:
            position = controller.position(name)
            setattr(cfg, "min_angle" if command == "min" else "max_angle" if command == "max" else "neutral", position)
            print(f"Recorded {command} = {position:g} deg in memory (not saved yet).")
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
    print(f"Testing only {name} on {cfg.port}; calibrated range {cfg.min_angle}..{cfg.max_angle} deg.")
    if not calibrated_baseline(name, configs, controller):
        return 1
    print("Enter targets within calibrated limits, or q to quit.")
    while True:
        text = input(f"{name} [{controller.position(name):g} deg] target> ").strip().lower()
        if text in {"q", "quit"}:
            return 0
        try:
            target = float(text)
            controller.move_to(name, target, speed_dps=speed)
            print(f"Reached {target:g} deg.")
        except (ValueError, ServoError) as exc:
            print(f"Rejected: {exc}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe Robot HAT servo calibration")
    parser.add_argument("--mock", action="store_true", help="print commands instead of using hardware")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list configured servos and ports")
    prep = sub.add_parser("prepare", help="show no-movement mechanical-centering instructions")
    prep.add_argument("servo")
    centered = sub.add_parser("center", help="explicitly command 0 deg with linkage disengaged")
    centered.add_argument("servo")
    aligned = sub.add_parser("align", help="find neutral from a confirmed powered reference")
    aligned.add_argument("servo")
    aligned.add_argument("--provisional-limit", type=float, default=15.0,
                         help="temporary +/- degree window from reference (1..20; default 15)")
    cal = sub.add_parser("calibrate", help="calibrate one mechanically centered servo")
    cal.add_argument("servo")
    test = sub.add_parser("test", help="manually test one calibrated, centered servo")
    test.add_argument("servo")
    test.add_argument("--speed", type=float, default=30.0)
    return parser

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configs = load_config(args.config)
        if args.command == "list":
            for name, cfg in configs.items():
                print(f"{name:<10} {cfg.port:<3} {'calibrated' if cfg.calibrated else 'UNCALIBRATED'}")
            return 0
        if args.servo not in configs:
            raise ConfigurationError(f"Unknown servo {args.servo!r}; use the list command")
        if args.command == "prepare":
            print(preparation_text(args.servo, configs[args.servo].port))
            return 0
        controller = ServoController(configs, MockBackend() if args.mock else RobotHatBackend())
        if args.command == "center":
            return center(args.servo, configs, controller)
        if args.command == "align":
            return align(args.servo, configs, controller, args.config, args.provisional_limit)
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
