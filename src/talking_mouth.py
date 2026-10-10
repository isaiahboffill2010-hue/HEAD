"""Manually triggered, one-shot speech-like mouth animation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibration import DEFAULT_CONFIG, SAFETY
from .mouth import JAW_NAMES, MouthController
from .mouth_animation import MouthAnimationEngine
from .adaptive_lipsync import load_motion_settings
from .mouth_calibration import _known_reference
from .servo_controller import (ConfigurationError, MockBackend, RobotHatBackend,
    ServoController, ServoError, load_config)

# A visual rhythm only; it is not speech synthesis or audio synchronization.
HELLO_SEQUENCE = (
    {"time": 0.00, "openness": 0.0},
    {"time": 0.25, "openness": 0.25},
    {"time": 1.05, "openness": 0.42},
    {"time": 1.75, "openness": 0.05},
    {"time": 2.45, "openness": 0.0},
    {"time": 2.85, "openness": 0.22},
    {"time": 3.55, "openness": 0.36},
    {"time": 4.25, "openness": 0.0},
    {"time": 4.70, "openness": 0.18},
    {"time": 5.45, "openness": 0.50},
    {"time": 6.25, "openness": 0.12},
    {"time": 7.00, "openness": 0.0},
    {"time": 7.50, "openness": 0.30},
    {"time": 8.25, "openness": 0.18},
    {"time": 9.00, "openness": 0.0},
)


def ensure_jaws_calibrated(configs) -> None:
    for name in JAW_NAMES:
        cfg = configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None or cfg.opening_direction is None:
            raise ConfigurationError(f"{name} opening endpoint is not calibrated")


def show_plan(configs) -> None:
    print('Planned one-shot visual phrase: "Hello, how are you today?"')
    for name in JAW_NAMES:
        cfg = configs[name]
        print(f"  {name}: closed {cfg.neutral:g} deg -> open {cfg.opening_endpoint:g} deg")
    print("  timed openness events:", ", ".join(f"{event['time']:.2f}s:{event['openness']:.2f}" for event in HELLO_SEQUENCE))


def talking_test(configs, controller, speed: float, input_func=input,
                 mouth_factory=MouthController, animation_factory=MouthAnimationEngine) -> int:
    ensure_jaws_calibrated(configs)
    load_motion_settings().speed(speed)
    print("Talking-mouth test: no command has been sent.")
    show_plan(configs)
    for name in JAW_NAMES:
        if not _known_reference(name, configs[name], controller, input_func):
            return 1
    print(SAFETY)
    print("This runs once only. Ctrl+C stops future commands; do not assume positions after interruption.")
    if input_func("Type TALK to run the displayed sequence: ").strip() != "TALK":
        print("Talking animation not confirmed; nothing moved.")
        return 1
    mouth = mouth_factory(controller)
    animation_factory(mouth).animate(HELLO_SEQUENCE, speed)
    # The validated sequence's final event is closed. Do not issue a blind
    # recovery close on KeyboardInterrupt: the exception reaches main instead.
    print("Talking animation completed once; both jaws are at their saved closed positions.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Safe one-shot talking-mouth animation")
    parser.add_argument("--mock", action="store_true", help="print commands instead of using Robot HAT hardware")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("test", help="run one confirmed speech-like mouth sequence")
    test.add_argument("--speed", type=float, default=8.0, help="degrees per second; capped by config/mouth_motion.json")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configs = load_config(args.config)
        controller = ServoController(configs, MockBackend() if args.mock else RobotHatBackend())
        return talking_test(configs, controller, args.speed)
    except KeyboardInterrupt:
        print("\nSTOPPED: no future commands will be sent. Do not blindly close the jaws; disconnect servo power physically if unsafe.", file=sys.stderr)
        return 130
    except (ServoError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
