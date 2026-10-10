import json
import tempfile
import unittest
from pathlib import Path

from src.servo_controller import (ConfigurationError, LimitError, MockBackend,
    NotCalibratedError, PositionUnknownError, ServoConfig, ServoController,
    load_config, save_config)


class QuietMock(MockBackend):
    def write(self, port, angle):
        self.commands.append((port, angle))


class ServoControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ServoConfig("P6", 0, 1, -10, 10)
        self.backend = QuietMock()
        self.controller = ServoController({"neck_pan": self.cfg}, self.backend, sleep=lambda _: None)

    def test_construction_does_not_move(self):
        self.assertEqual(self.backend.commands, [])

    def test_unknown_position_blocks_smooth_move(self):
        with self.assertRaises(PositionUnknownError):
            self.controller.move_to("neck_pan", 2)
        self.assertEqual(self.backend.commands, [])

    def test_uncalibrated_servo_blocks_normal_move(self):
        ctl = ServoController({"neck_pan": ServoConfig("P6", None, 1, None, None)}, self.backend, sleep=lambda _: None)
        ctl.establish_position("neck_pan", 0)
        with self.assertRaises(NotCalibratedError):
            ctl.move_to("neck_pan", 1)

    def test_limits_are_enforced_without_command(self):
        self.controller.establish_position("neck_pan", 0)
        count = len(self.backend.commands)
        with self.assertRaises(LimitError):
            self.controller.move_to("neck_pan", 11)
        self.assertEqual(len(self.backend.commands), count)

    def test_smooth_move_and_direction_mapping(self):
        self.cfg.direction = -1
        self.controller.establish_position("neck_pan", 0)
        self.controller.move_to("neck_pan", 3, speed_dps=10, step=1)
        self.assertEqual(self.backend.commands, [("P6", 0), ("P6", -1), ("P6", -2), ("P6", -3)])

    def test_calibration_only_allows_small_steps(self):
        self.controller.establish_position("neck_pan", 0)
        with self.assertRaises(ValueError):
            self.controller.calibration_step("neck_pan", 3)

    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"
            save_config(path, {"neck_pan": self.cfg})
            self.assertEqual(load_config(path)["neck_pan"], self.cfg)

    def test_bad_direction_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"
            path.write_text(json.dumps({"servos": {"x": {"port": "P0", "neutral": None, "direction": 0, "min_angle": None, "max_angle": None}}}))
            with self.assertRaises(ConfigurationError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
