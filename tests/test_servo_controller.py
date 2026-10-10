import json
import tempfile
import unittest
from pathlib import Path

from src.servo_controller import (ConfigurationError, LimitError, MockBackend,
    NotCalibratedError, PositionUnknownError, ServoConfig, ServoController,
    load_config, save_config)
from src.calibration import align, center, preparation_text
from src.mouth import MouthController
from src.mouth_calibration import calibrate_jaw, combined_test


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

    def test_preparation_includes_no_command_and_position_warning(self):
        text = preparation_text("neck_pan", "P6")
        self.assertIn("NO servo commands", text)
        self.assertIn("cannot report their current shaft angle", text)
        self.assertIn("Software cannot discover", text)

    def test_center_refusal_sends_no_command(self):
        result = center("neck_pan", {"neck_pan": self.cfg}, self.controller,
                        input_func=lambda _: "no")
        self.assertEqual(result, 1)
        self.assertEqual(self.backend.commands, [])

    def test_center_requires_confirmation_then_commands_zero(self):
        result = center("neck_pan", {"neck_pan": self.cfg}, self.controller,
                        input_func=lambda _: "CENTER")
        self.assertEqual(result, 0)
        self.assertEqual(self.backend.commands, [("P6", 0)])

    def test_adopt_confirmed_position_sends_no_command(self):
        self.controller.adopt_confirmed_position("neck_pan", 0)
        self.assertEqual(self.controller.position("neck_pan"), 0)
        self.assertEqual(self.backend.commands, [])

    def test_align_rejects_unconfirmed_reference_without_command(self):
        answers = iter(["0", "no"])
        result = align("neck_pan", {"neck_pan": self.cfg}, self.controller,
                       Path("unused.json"), input_func=lambda _: next(answers))
        self.assertEqual(result, 1)
        self.assertEqual(self.backend.commands, [])

    def test_align_saves_neutral_only_with_one_degree_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"
            answers = iter(["0", "POWERED", "+", "SAVE"])
            result = align("neck_pan", {"neck_pan": self.cfg}, self.controller, path,
                           provisional_limit=2, input_func=lambda _: next(answers))
            self.assertEqual(result, 0)
            self.assertEqual(self.backend.commands, [("P6", 1)])
            saved = load_config(path)["neck_pan"]
            self.assertEqual(saved.neutral, 1)
            self.assertIsNone(saved.min_angle)
            self.assertIsNone(saved.max_angle)

    def test_align_rejects_step_outside_provisional_window(self):
        answers = iter(["0", "POWERED", "+", "+", "quit"])
        result = align("neck_pan", {"neck_pan": self.cfg}, self.controller,
                       Path("unused.json"), provisional_limit=1,
                       input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        self.assertEqual(self.backend.commands, [("P6", 1)])

    def test_align_expands_only_after_boundary_and_fresh_clearance_confirmation(self):
        answers = iter(["0", "POWERED", "-", "-", "extend -", "CLEAR", "-", "quit"])
        result = align("neck_pan", {"neck_pan": self.cfg}, self.controller,
                       Path("unused.json"), provisional_limit=1,
                       input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        # The second '-' is blocked; extend itself sends no command. Only the
        # initial step and the later explicitly requested step are written.
        self.assertEqual(self.backend.commands, [("P6", -1), ("P6", -2)])

    def test_align_rejected_extension_keeps_boundary_and_sends_nothing(self):
        answers = iter(["0", "POWERED", "-", "extend -", "no", "-", "quit"])
        result = align("neck_pan", {"neck_pan": self.cfg}, self.controller,
                       Path("unused.json"), provisional_limit=1,
                       input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        self.assertEqual(self.backend.commands, [("P6", -1)])

    def test_align_never_extends_beyond_absolute_hardware_limit(self):
        answers = iter(["-90", "POWERED", "-", "extend -", "quit"])
        self.controller.configs["head_roll"] = self.cfg
        result = align("head_roll", {"head_roll": self.cfg}, self.controller,
                       Path("unused.json"), provisional_limit=1,
                       input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        self.assertEqual(self.backend.commands, [])

    def test_neck_pan_extends_negative_side_in_confirmed_five_degree_sections(self):
        # Reach the initial -15 degree boundary with explicit 1-degree moves,
        # then extend to -20. The extension itself must not write hardware.
        answers = iter(["0", "POWERED"] + ["-"] * 15 + ["extend -", "CLEAR", "quit"])
        result = align("neck_pan", {"neck_pan": self.cfg}, self.controller,
                       Path("unused.json"), provisional_limit=15,
                       input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        self.assertEqual(self.backend.commands, [("P6", angle) for angle in range(-1, -16, -1)])

    def test_neck_pan_positive_alignment_boundary_stays_at_fifteen(self):
        answers = iter(["0", "POWERED"] + ["+"] * 15 + ["extend +", "quit"])
        result = align("neck_pan", {"neck_pan": self.cfg}, self.controller,
                       Path("unused.json"), provisional_limit=15,
                       input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        self.assertEqual(self.backend.commands, [("P6", angle) for angle in range(1, 16)])

    def test_neck_pan_negative_alignment_boundary_never_exceeds_minus_eighty(self):
        # A confirmed reference at -80 has no further negative software travel.
        answers = iter(["-80", "POWERED", "extend -", "quit"])
        result = align("neck_pan", {"neck_pan": self.cfg}, self.controller,
                       Path("unused.json"), provisional_limit=15,
                       input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        self.assertEqual(self.backend.commands, [])

    def test_jaw_calibration_saves_endpoint_and_preserves_neutral(self):
        jaw = ServoConfig("P8", 1, 1, None, None)
        controller = ServoController({"upper_jaw": jaw}, self.backend, sleep=lambda _: None)
        answers = iter(["1", "POWERED", "+", "SAVE"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"
            result = calibrate_jaw("upper_jaw", {"upper_jaw": jaw}, controller, path,
                                   input_func=lambda _: next(answers))
            self.assertEqual(result, 0)
            saved = load_config(path)["upper_jaw"]
            self.assertEqual(saved.neutral, 1)
            self.assertEqual(saved.opening_endpoint, 2)
            self.assertEqual(saved.opening_direction, 1)
        self.assertEqual(self.backend.commands, [("P8", 2)])

    def test_jaw_calibration_unconfirmed_reference_sends_no_command(self):
        jaw = ServoConfig("P10", 15, 1, None, None)
        controller = ServoController({"lower_jaw": jaw}, self.backend, sleep=lambda _: None)
        answers = iter(["15", "no"])
        result = calibrate_jaw("lower_jaw", {"lower_jaw": jaw}, controller,
                               Path("unused.json"), input_func=lambda _: next(answers))
        self.assertEqual(result, 1)
        self.assertEqual(self.backend.commands, [])

    def test_mouth_controller_moves_only_jaws_and_returns_closed(self):
        upper = ServoConfig("P8", 1, 1, None, None, 3, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 12, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        controller.adopt_confirmed_position("upper_jaw", 1)
        controller.adopt_confirmed_position("lower_jaw", 15)
        mouth = MouthController(controller, sleep=lambda _: None)
        mouth.open(1.0, speed_dps=10)
        mouth.close(speed_dps=10)
        self.assertEqual(controller.position("upper_jaw"), 1)
        self.assertEqual(controller.position("lower_jaw"), 15)
        self.assertTrue(all(port in {"P8", "P10"} for port, _ in self.backend.commands))

    def test_combined_test_requires_both_calibrated_jaws(self):
        upper = ServoConfig("P8", 1, 1, None, None, 3, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 12, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        answers = iter(["1", "POWERED", "15", "POWERED", "CYCLE"])
        result = combined_test({"upper_jaw": upper, "lower_jaw": lower}, controller,
                               speed=10, input_func=lambda _: next(answers))
        self.assertEqual(result, 0)
        self.assertEqual(controller.position("upper_jaw"), 1)
        self.assertEqual(controller.position("lower_jaw"), 15)
        self.assertTrue(all(port in {"P8", "P10"} for port, _ in self.backend.commands))


if __name__ == "__main__":
    unittest.main()
