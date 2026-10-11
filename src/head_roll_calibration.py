"""Supervised, narrow-range calibration and testing for P0 head_roll only."""
from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import DEFAULT_CONFIG, SAFETY
from .head_roll_motion import HeadRollAnimator
from .servo_controller import (ConfigurationError, MockBackend, RobotHatBackend, ServoController,
                               ServoError, load_config, save_config, validate_config)

NAME = "head_roll"


def adopt_head_roll_reference(configs, controller, input_func=input) -> bool:
    cfg = configs[NAME]
    print(f"Head roll only: P0 saved neutral is {cfg.neutral:g} deg. No command has been sent.")
    print("P0 has no position feedback. Enter its actual previously commanded angle only if it remained powered and untouched.")
    try:
        current = float(input_func("Enter current commanded P0 logical angle (-90..90): "))
    except ValueError:
        print("Invalid angle; no command was sent."); return False
    if not -90 <= current <= 90:
        print("Angle must be between -90 and 90; no command was sent."); return False
    print(SAFETY)
    if input_func("Type POWERED to confirm continuous P0 power and no repositioning: ").strip() != "POWERED":
        print("P0 reference not confirmed; no command was sent."); return False
    controller.adopt_confirmed_position(NAME, current)
    print(f"Trusted P0 reference recorded at {current:g} deg without movement.")
    return True


def _step_to(controller, target: float, input_func=input) -> bool:
    current = controller.position(NAME)
    assert current is not None
    if abs(target - current) != 1:
        raise ValueError("quick calibration permits only one-degree moves")
    print(SAFETY)
    if input_func(f"Type TEST {target:g} to move P0 one degree to {target:g}, or STOP: ").strip() != f"TEST {target:g}":
        print("Stopped; no further command was sent."); return False
    controller.calibration_step(NAME, int(target - current), speed_dps=5)
    if input_func("Type SAFE only if this exact movement was safe, or STOP: ").strip() != "SAFE":
        print("Stopped after this movement. Do not continue if there was any strain or unexpected motion."); return False
    return True


def quick(configs, controller, path: Path, input_func=input) -> int:
    """Verify only neutral-2..neutral+2, saving nothing unless all checks pass."""
    cfg = configs[NAME]
    if cfg.neutral is None:
        raise ConfigurationError("head_roll has no saved neutral")
    center = float(cfg.neutral)
    print("Quick P0 head-roll calibration. It never moves P2, P6, P8, or P10.")
    if not adopt_head_roll_reference(configs, controller, input_func): return 1
    if controller.position(NAME) != center:
        print(f"Quick calibration requires a confirmed current P0 position of {center:g} deg; nothing moved.")
        return 1
    # The order deliberately verifies one degree at a time and requires a fresh
    # explicit confirmation before returning toward center.
    for target in (center - 1, center - 2, center - 1, center):
        if not _step_to(controller, target, input_func): return 1
    for target in (center + 1, center + 2):
        if not _step_to(controller, target, input_func): return 1
    print(f"Both P0 sides were physically verified: {center - 2:g}..{center + 2:g} deg.")
    if input_func("Type SAVE LIMITS to save only these initial P0 limits, or STOP: ").strip() != "SAVE LIMITS":
        print("Limits were not saved; no further command was sent."); return 1
    cfg.min_angle, cfg.max_angle = center - 2, center + 2
    save_config(path, configs)
    print(f"Saved P0 head-roll limits {cfg.min_angle:g}..{cfg.max_angle:g} deg. No wider range is automatic.")
    return 0


def extend(configs, controller, path: Path, input_func=input) -> int:
    """Extend exactly one already-verified P0 endpoint, one confirmed degree at a time."""
    cfg = configs[NAME]
    validate_config(NAME, cfg, require_calibrated=True)
    assert cfg.min_angle is not None and cfg.max_angle is not None
    print(f"P0 extension starts only from a saved endpoint: {cfg.min_angle:g}..{cfg.max_angle:g} deg.")
    print("No mechanical limit beyond the currently saved verified range is assumed.")
    if not adopt_head_roll_reference(configs, controller, input_func): return 1
    current = controller.position(NAME)
    assert current is not None
    if current == cfg.max_angle:
        direction, target = 1, current + 1
    elif current == cfg.min_angle:
        direction, target = -1, current - 1
    else:
        print("P0 must be at one saved endpoint before extension; no movement was sent.")
        return 1
    while True:
        if not -90 <= target <= 90:
            print("Stopped at the Robot HAT electrical -90..90 degree limit; no movement was sent.")
            return 0
        print(SAFETY)
        print("Inspect the bracket and full linkage clearance before every new degree.")
        command = input_func(f"Type TEST {target:g} to test this next unverified P0 angle, or STOP: ").strip()
        if command == "STOP":
            print("Extension stopped. Previously saved verified limits are unchanged.")
            return 0
        if command != f"TEST {target:g}":
            print("No movement was sent. Type the displayed TEST command or STOP.")
            continue
        controller.calibration_step(NAME, direction, speed_dps=5)
        if input_func("Type SAFE only if this exact movement was safe, or STOP: ").strip() != "SAFE":
            print("This new angle was NOT saved. Stop and inspect; no further command was sent.")
            return 1
        if direction > 0:
            cfg.max_angle = target
        else:
            cfg.min_angle = target
        save_config(path, configs)
        print(f"Saved verified P0 {'maximum' if direction > 0 else 'minimum'} {target:g} deg.")
        target += direction


def roll_test(configs, controller, input_func=input) -> int:
    cfg = configs[NAME]; validate_config(NAME, cfg, require_calibrated=True)
    print(f"P0 roll-only test: saved verified range {cfg.min_angle:g}..{cfg.max_angle:g} deg.")
    if not adopt_head_roll_reference(configs, controller, input_func): return 1
    position = controller.position(NAME)
    assert position is not None
    if not cfg.min_angle <= position <= cfg.max_angle:
        print("Confirmed P0 position is outside saved verified range; no movement was sent."); return 1
    print(SAFETY)
    if input_func("Type ROLL to run one gentle P0-only left/right cycle, or STOP: ").strip() != "ROLL": return 1
    animator = HeadRollAnimator(controller)
    try:
        animator._transition(cfg.max_angle)
        animator._transition(cfg.min_angle)
    except KeyboardInterrupt:
        animator.stop(); raise
    print("P0 roll-only test completed. No automatic return-to-center command was sent.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Safe P0 head-roll calibration")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("quick", "extend", "test"))
    args = parser.parse_args(argv)
    try:
        configs = load_config(args.config); controller = ServoController(configs, MockBackend() if args.mock else RobotHatBackend())
        if args.command == "quick": return quick(configs, controller, args.config)
        if args.command == "extend": return extend(configs, controller, args.config)
        return roll_test(configs, controller)
    except KeyboardInterrupt:
        print("Stopped: no P0 return-to-center command was sent. Disconnect servo power physically if unsafe."); return 130
    except (ConfigurationError, ServoError, ValueError) as exc:
        print(f"Setup error: {exc}"); return 2


if __name__ == "__main__": raise SystemExit(main())
