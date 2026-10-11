import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from src.servo_controller import (ConfigurationError, LimitError, MockBackend,
    NotCalibratedError, PositionUnknownError, ServoConfig, ServoController,
    load_config, save_config)
from src.calibration import align, center, preparation_text
from src.mouth import MouthController
from src.mouth_calibration import calibrate_jaw, combined_test, speed_calibration
from src.mouth_animation import MouthAnimationEngine
from src.talking_mouth import HELLO_SEQUENCE, talking_test
from src.adaptive_lipsync import (AdaptiveLipSynchronizer, MotionSettings,
    calibrated_max_jaw_travel, load_motion_settings)
from src.audio_sync import (AudioPlayer, PlaybackDiagnostics, RawPCMPlayer, energy_events,
    select_audio_backend, validate_word_timings, word_events)
from src.gemini_service import GeminiConfig, GeminiError, GeminiService, parse_word_timings, write_wav
from src.streaming_lipsync import MouthTuning, StreamingLipSynchronizer
from src.mouth_diagnostics import analyze_wav, compare_recordings, final_speech_report, write_svg
from src.robot_chat import (establish_mouth, generate_complete_text, speak_streaming,
    ziggy_silence_seconds)
import src.robot_chat as robot_chat
from src.latency import LatencyTrace
from src.microphone import Microphone, Utterance
from src.vosk_wake import VoskWakeWord, VoskWakeWordError
from src.vosk_setup import ensure_model
from src.head_roll_calibration import (extend as extend_head_roll, quick as quick_head_roll,
    wide as wide_head_roll)
from src.head_roll_motion import HeadRollAnimator


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

    def test_talking_animation_is_synchronized_bounded_and_ends_closed(self):
        upper = ServoConfig("P8", 1, 1, None, None, 21, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 0, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        answers = iter(["1", "POWERED", "15", "POWERED", "TALK"])
        mouth_factory = lambda ctl: MouthController(ctl, sleep=lambda _: None)
        animation_factory = lambda mouth: MouthAnimationEngine(mouth, sleep=lambda _: None, clock=lambda: 0)
        result = talking_test({"upper_jaw": upper, "lower_jaw": lower}, controller,
                              speed=8, input_func=lambda _: next(answers),
                              mouth_factory=mouth_factory, animation_factory=animation_factory)
        self.assertEqual(result, 0)
        self.assertEqual(controller.position("upper_jaw"), 1)
        self.assertEqual(controller.position("lower_jaw"), 15)
        upper_commands = [angle for port, angle in self.backend.commands if port == "P8"]
        lower_commands = [angle for port, angle in self.backend.commands if port == "P10"]
        self.assertTrue(all(1 <= angle <= 21 for angle in upper_commands))
        self.assertTrue(all(0 <= angle <= 15 for angle in lower_commands))
        self.assertTrue(all(port in {"P8", "P10"} for port, _ in self.backend.commands))

    def test_talking_test_rejects_unsafe_speed_before_movement(self):
        upper = ServoConfig("P8", 1, 1, None, None, 21, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 0, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        with self.assertRaises(ValueError):
            talking_test({"upper_jaw": upper, "lower_jaw": lower}, controller, speed=61,
                         input_func=lambda _: "unused")
        self.assertEqual(self.backend.commands, [])

    def test_speed_calibration_progressively_saves_only_passed_stage(self):
        upper = ServoConfig("P8", 1, 1, None, None, 3, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 12, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        answers = iter(["1", "POWERED", "15", "POWERED", "RUN 8", "PASS", "RUN 12", "STOP"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            path.write_text(json.dumps({
                "conservative_speed_dps": 8.0,
                "hard_software_speed_cap_dps": 60.0,
                "acceleration_dps2": 240.0,
                "update_interval_s": 0.02,
                "verified_practical_speed_dps": None,
            }))
            result = speed_calibration(
                {"upper_jaw": upper, "lower_jaw": lower}, controller, path,
                input_func=lambda _: next(answers),
            )
            self.assertEqual(result, 0)
            self.assertEqual(load_motion_settings(path).verified_practical_speed_dps, 8.0)
            self.assertEqual(load_motion_settings(path).selected_operating_speed_dps, 8.0)
        self.assertEqual(controller.position("upper_jaw"), 1)
        self.assertEqual(controller.position("lower_jaw"), 15)

    def test_target_speed_preserves_verified_maximum_and_selects_lower_operating_speed(self):
        upper = ServoConfig("P8", 1, 1, None, None, 21, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 0, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        answers = iter(["1", "POWERED", "15", "POWERED", "RUN 15", "PASS"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            path.write_text(json.dumps({"conservative_speed_dps": 8, "hard_software_speed_cap_dps": 60,
                                        "acceleration_dps2": 240, "update_interval_s": .02,
                                        "verified_practical_speed_dps": 25}))
            self.assertEqual(speed_calibration({"upper_jaw": upper, "lower_jaw": lower}, controller, path,
                                               input_func=lambda _: next(answers), target_speed=15), 0)
            settings = load_motion_settings(path)
            self.assertEqual(settings.maximum_speed(), 25)
            self.assertEqual(settings.selected_operating_speed_dps, 15)
            self.assertEqual(settings.speed(), 15)

    def test_target_speed_rejects_above_verified_maximum_without_motion(self):
        upper = ServoConfig("P8", 1, 1, None, None, 21, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 0, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            path.write_text(json.dumps({"verified_practical_speed_dps": 25}))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                speed_calibration({"upper_jaw": upper, "lower_jaw": lower}, controller, path, target_speed=30)
        self.assertEqual(self.backend.commands, [])

    def test_animation_cancellation_does_not_issue_a_recovery_close(self):
        class InterruptingMouth:
            def __init__(self):
                self.calls = []
            def set_openness(self, amount, speed_dps=10):
                self.calls.append((amount, speed_dps))
                raise KeyboardInterrupt
        mouth = InterruptingMouth()
        engine = MouthAnimationEngine(mouth, sleep=lambda _: None, clock=lambda: 0)
        with self.assertRaises(KeyboardInterrupt):
            engine.animate(HELLO_SEQUENCE, speed_dps=8)
        self.assertEqual(len(mouth.calls), 1)

    def test_animation_events_and_speed_are_validated(self):
        class RecordingMouth:
            def __init__(self):
                self.calls = []
            def set_openness(self, amount, speed_dps=10):
                self.calls.append((amount, speed_dps))
        mouth = RecordingMouth()
        engine = MouthAnimationEngine(mouth, sleep=lambda _: None, clock=lambda: 0)
        with self.assertRaises(ValueError):
            engine.animate(({"time": 0.0, "openness": 1.1}, {"time": 1.0, "openness": 0.0}))
        with self.assertRaises(ValueError):
            engine.animate(HELLO_SEQUENCE, speed_dps=0)

    def test_adaptive_lipsync_skips_expired_fast_targets_and_reduces_amplitude(self):
        class FakeClock:
            now = 0.0
            def clock(self):
                return self.now
            def sleep(self, duration):
                self.now += duration
        class RecordingMouth:
            def __init__(self):
                self.targets = []
            def set_openness_step(self, amount):
                self.targets.append(amount)
        fake = FakeClock()
        mouth = RecordingMouth()
        settings = MotionSettings(8, 20, 24, 0.05, None)
        sync = AdaptiveLipSynchronizer(mouth, 20, settings, sleep=fake.sleep)
        records = sync.run((
            {"time": 0.0, "openness": 0.0},
            {"time": 0.01, "openness": 1.0},
            {"time": 0.02, "openness": 0.0},
            {"time": 0.30, "openness": 0.0},
        ), fake.clock)
        self.assertTrue(any(record.skipped for record in records))
        self.assertTrue(any(record.amplitude_reduced for record in records))
        self.assertTrue(all(0 <= target <= 1 for target in mouth.targets))

    def test_adaptive_lipsync_synchronizes_jaws_and_closes_in_long_pause(self):
        class FakeClock:
            now = 0.0
            def clock(self):
                return self.now
            def sleep(self, duration):
                self.now += duration
        upper = ServoConfig("P8", 1, 1, None, None, 21, 1)
        lower = ServoConfig("P10", 15, 1, None, None, 0, -1)
        controller = ServoController({"upper_jaw": upper, "lower_jaw": lower}, self.backend, sleep=lambda _: None)
        controller.adopt_confirmed_position("upper_jaw", 1)
        controller.adopt_confirmed_position("lower_jaw", 15)
        fake = FakeClock()
        mouth = MouthController(controller, sleep=lambda _: None)
        sync = AdaptiveLipSynchronizer(
            mouth, calibrated_max_jaw_travel({"upper_jaw": upper, "lower_jaw": lower}),
            MotionSettings(8, 20, 40, 0.05, None), sleep=fake.sleep,
        )
        sync.run((
            {"time": 0.0, "openness": 0.0},
            {"time": 0.20, "openness": 0.4},
            {"time": 2.50, "openness": 0.0},
            {"time": 4.00, "openness": 0.0},
        ), fake.clock)
        self.assertAlmostEqual(controller.position("upper_jaw"), 1, places=3)
        self.assertAlmostEqual(controller.position("lower_jaw"), 15, places=3)
        self.assertTrue(all(port in {"P8", "P10"} for port, _ in self.backend.commands))
        self.assertEqual(len(self.backend.commands) % 2, 0)
        self.assertTrue(all(self.backend.commands[i][0:1] == ("P8",) and self.backend.commands[i + 1][0:1] == ("P10",)
                            for i in range(0, len(self.backend.commands), 2)))

    def test_adaptive_lipsync_cancellation_sends_no_recovery_close(self):
        class InterruptingMouth:
            calls = 0
            def set_openness_step(self, amount):
                self.calls += 1
                raise KeyboardInterrupt
        mouth = InterruptingMouth()
        sync = AdaptiveLipSynchronizer(mouth, 20, MotionSettings(8, 20, 40, 0.05, None), sleep=lambda _: None)
        times = iter([0.0, 0.1])
        with self.assertRaises(KeyboardInterrupt):
            sync.run(({"time": 0.0, "openness": 0.0}, {"time": 0.1, "openness": 1.0},
                      {"time": 1.0, "openness": 0.0}), lambda: next(times))
        self.assertEqual(mouth.calls, 1)

    def test_gemini_response_history_and_despina_tts_wav_are_mocked(self):
        class Models:
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"text": "Hello from HEAD."})()
            def generate_content_stream(self, **kwargs):
                self.kwargs = kwargs
                return iter((type("Response", (), {"text": "Hello "})(),
                             type("Response", (), {"text": "from HEAD."})()))
        class Interactions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Interaction", (), {"output_audio": type("Audio", (), {
                    "data": b"RIFFtest", "mime_type": "audio/wav"})()})()
        client = type("Client", (), {"models": Models(), "interactions": Interactions()})()
        service = GeminiService(GeminiConfig("key", "chat", "tts", "Despina", "align"), client)
        self.assertEqual(service.reply("Hi"), "Hello from HEAD.")
        self.assertEqual(len(service.history), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = service.synthesize("Hello", Path(directory) / "voice.wav")
            self.assertEqual(path.read_bytes(), b"RIFFtest")
        self.assertEqual(client.interactions.kwargs["generation_config"]["speech_config"][0]["voice"], "Despina")

    def test_streaming_gemini_reports_chunks_and_bounds_history(self):
        class Models:
            def generate_content_stream(self, **kwargs):
                return iter((type("R", (), {"text": "Short "})(), type("R", (), {"text": "answer."})()))
        client = type("Client", (), {"models": Models()})()
        service = GeminiService(GeminiConfig("key", "chat", "tts", "Despina", "align", "0", 1), client)
        first = []
        self.assertEqual("".join(service.reply_stream("Hi", lambda: first.append(True))), "Short answer.")
        self.assertEqual(first, [True])
        self.assertEqual(len(service.history), 2)

    def test_short_greeting_uses_low_thinking_and_complete_response(self):
        stop = type("Finish", (), {"name": "STOP"})()
        response = type("R", (), {"text": "Hello! How can I help?",
                                     "candidates": [type("C", (), {"finish_reason": stop})()]})()
        class Models:
            def generate_content_stream(self, **kwargs):
                self.kwargs = kwargs
                return iter((response,))
        models = Models()
        config = GeminiConfig("key", "gemini-3.8-flash", "tts", "Despina", "align", "low", 4, 1024, 2048)
        service = GeminiService(config, type("Client", (), {"models": models})())
        self.assertEqual(service.reply("HI"), "Hello! How can I help?")
        self.assertEqual(models.kwargs["config"]["max_output_tokens"], 1024)
        self.assertEqual(models.kwargs["config"]["thinking_config"], {"thinking_level": "low"})
        self.assertEqual(service.last_finish_reason, "STOP")

    def test_max_token_finish_is_not_spoken_as_complete_text(self):
        finish = type("Finish", (), {"name": "MAX_TOKENS"})()
        candidate = type("Candidate", (), {"finish_reason": finish})()
        response = type("R", (), {"text": "My day has", "candidates": [candidate]})()
        class Models:
            def generate_content_stream(self, **kwargs): return iter((response,))
        service = GeminiService(GeminiConfig("key", "chat", "tts", "Despina", "align"),
                                type("Client", (), {"models": Models()})())
        with self.assertRaisesRegex(GeminiError, "token limit"):
            list(service.reply_stream("How was your day?"))
        self.assertEqual(service.history, [])

    def test_token_exhaustion_retries_once_with_larger_budget(self):
        finish = type("Finish", (), {"name": "MAX_TOKENS"})()
        capped = type("R", (), {"text": "Incomplete", "candidates": [type("C", (), {"finish_reason": finish})()]})()
        stop = type("Finish", (), {"name": "STOP"})()
        usage = type("Usage", (), {"candidates_token_count": 7, "thoughts_token_count": 3,
                                     "prompt_token_count": 10, "total_token_count": 20})()
        complete = type("R", (), {"text": "Hello there.", "candidates": [type("C", (), {"finish_reason": stop})()],
                                  "usage_metadata": usage})()
        class Models:
            def __init__(self): self.calls = []
            def generate_content_stream(self, **kwargs):
                self.calls.append(kwargs); return iter((capped,) if len(self.calls) == 1 else (complete,))
        models = Models()
        config = GeminiConfig("key", "gemini-3.8-flash", "tts", "Despina", "align", "low", 4, 1024, 2048)
        service = GeminiService(config, type("Client", (), {"models": models})())
        self.assertEqual("".join(service.reply_stream("HI")), "Hello there.")
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(models.calls[0]["config"]["max_output_tokens"], 1024)
        self.assertEqual(models.calls[1]["config"]["max_output_tokens"], 2048)
        self.assertEqual(models.calls[0]["config"]["thinking_config"], {"thinking_level": "low"})
        self.assertEqual(service.last_generation_usage["output_tokens"], 7)

    def test_roleplay_markup_and_physical_description_are_suppressed(self):
        stop = type("Finish", (), {"name": "STOP"})()
        response = type("R", (), {"text": "*whirrs softly and blinks* I am ready. I smile at you.",
                                    "candidates": [type("C", (), {"finish_reason": stop})()]})()
        class Models:
            def generate_content_stream(self, **kwargs): return iter((response,))
        service = GeminiService(GeminiConfig("key", "chat", "tts", "Despina", "align"),
                                type("Client", (), {"models": Models()})())
        answer = service.reply("HI")
        self.assertEqual(answer, "I am ready.")
        self.assertNotIn("*", answer)

    def test_streaming_despina_decodes_raw_pcm_chunks(self):
        import base64
        audio = base64.b64encode(b"\x01\x00" * 20).decode()
        delta = type("Delta", (), {"type": "audio", "data": audio})()
        event = type("Event", (), {"event_type": "step.delta", "delta": delta})()
        class Interactions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return iter((event,))
        client = type("Client", (), {"interactions": Interactions()})()
        service = GeminiService(GeminiConfig("key", "chat", "tts", "Despina", "align"), client)
        chunks = list(service.stream_speech("hello"))
        self.assertEqual(chunks[0][0], b"\x01\x00" * 20)
        self.assertTrue(client.interactions.kwargs["stream"])
        self.assertEqual(client.interactions.kwargs["response_format"]["sample_rate"], 24000)

    def test_raw_pcm_player_buffers_submits_and_waits(self):
        class Input:
            def __init__(self): self.data = bytearray(); self.closed = False
            def write(self, data): self.data.extend(data)
            def close(self): self.closed = True
        class Process:
            def __init__(self): self.stdin = Input(); self.waited = False
            def wait(self): self.waited = True; return 0
            def poll(self): return 0
        process = Process(); calls = []
        player = RawPCMPlayer(backend="pulse", popen=lambda command, **kwargs: calls.append(command) or process,
                              which=lambda _: "/usr/bin/paplay", queue_chunks=2)
        submitted = []
        self.assertEqual(player.play(iter((b"aa", b"bb")), on_first_submit=lambda: submitted.append(True)), 0)
        self.assertEqual(process.stdin.data, b"aabb")
        self.assertEqual(submitted, [True])
        self.assertTrue(process.stdin.closed and process.waited)

    def test_raw_pcm_player_propagates_slow_stream_failure_after_cleanup(self):
        class Input:
            def __init__(self): self.closed = False
            def write(self, data): pass
            def close(self): self.closed = True
        class Process:
            def __init__(self): self.stdin = Input(); self.waited = False
            def wait(self): self.waited = True; return 0
            def poll(self): return 0
        process = Process()
        def failing_stream():
            yield b"first"
            raise RuntimeError("network stalled")
        player = RawPCMPlayer(backend="pulse", popen=lambda *args, **kwargs: process,
                              which=lambda _: "/usr/bin/paplay", queue_chunks=2)
        with self.assertRaisesRegex(RuntimeError, "network stalled"):
            player.play(failing_stream())
        self.assertTrue(process.stdin.closed and process.waited)

    def test_raw_pcm_player_waits_for_buffered_output_process_completion(self):
        import time
        class Input:
            def write(self, data): pass
            def close(self): pass
        class Process:
            stdin = Input()
            def poll(self): return 0
            def wait(self): time.sleep(.05); return 0  # output sink still buffered after submission
        player = RawPCMPlayer(backend="pulse", popen=lambda *args, **kwargs: Process(),
                              which=lambda _: "/usr/bin/paplay")
        began = time.monotonic(); player.play(iter((b"\x01\x00" * 20,)))
        self.assertGreaterEqual(time.monotonic() - began, .045)
        self.assertIsNotNone(player.diagnostics.playback_completed_at)

    def test_raw_pcm_player_bounds_fast_pcm_lead_without_delaying_first_frame(self):
        import time
        class Input:
            def __init__(self): self.writes = 0
            def write(self, data): self.writes += 1
            def close(self): pass
        class Process:
            def __init__(self): self.stdin = Input()
            def poll(self): return 0
            def wait(self): return 0
        process = Process(); first = []
        player = RawPCMPlayer(backend="pulse", popen=lambda *args, **kwargs: process,
                              which=lambda _: "/usr/bin/paplay", max_ahead_ms=20)
        began = time.monotonic()
        player.play(iter((b"\x01\x00" * (240 * 10),)), on_first_submit=lambda: first.append(time.monotonic()))
        self.assertLess(first[0] - began, .03)
        self.assertGreater(player.diagnostics.pacing_wait_ms, 50)
        self.assertGreater(process.stdin.writes, 1)

    def test_streaming_lipsync_is_bounded_and_closes(self):
        import time
        class Mouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value)
        mouth = Mouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(8, 60, 240, .02, 25))
        sync.start()
        for _ in range(20):
            sync.submit(b"\xff\x7f" * 240, 0)
        time.sleep(.08)
        sync.stop()
        self.assertTrue(mouth.targets)
        self.assertTrue(all(0 <= value <= 1 for value in mouth.targets))
        self.assertEqual(mouth.targets[-1], 0.0)

    def test_streaming_lipsync_drains_last_words_before_closing(self):
        import time
        class Mouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value)
        mouth = Mouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25), latency_ms=40)
        started = time.monotonic(); sync.start()
        sync.note_tts_pcm_received()
        sync.submit(b"\xff\x7f" * 480, 0.0)  # final, speech-bearing 20 ms PCM frame
        sync.finish()
        self.assertGreaterEqual(time.monotonic() - started, .025)
        self.assertTrue(any(value > 0 for value in mouth.targets))
        self.assertEqual(mouth.targets[-1], 0.0)
        self.assertEqual(sync.diagnostics.shutdown_reason, "drained end-of-stream")
        self.assertIsNotNone(sync.diagnostics.last_speech_frame_audio_s)
        self.assertIsNotNone(sync.diagnostics.last_mouth_frame_processed_at)

    def test_mouth_only_offset_is_added_once_after_audio_latency(self):
        class Mouth:
            def set_openness_step(self, value): pass
        sync = StreamingLipSynchronizer(Mouth(), 20, MotionSettings(25, 60, 240, .02, 25),
                                        latency_ms=180, mouth_offset_ms=100)
        self.assertAlmostEqual(sync.latency_s, .18)
        self.assertAlmostEqual(sync.mouth_offset_s, .1)
        self.assertAlmostEqual(sync.effective_delay_s, .28)

    def test_final_mouth_diagnostic_compares_audible_estimate(self):
        import time
        class Mouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value)
        mouth = Mouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25),
                                        latency_ms=10, mouth_offset_ms=40)
        started = time.monotonic(); sync.start()
        sync.submit(b"\xff\x7f" * 480, 0.0)
        sync.finish(playback_started_at=started)
        self.assertIsNotNone(sync.diagnostics.final_audible_audio_estimate_at)
        self.assertIsNotNone(sync.diagnostics.last_speech_mouth_command_at)
        self.assertGreater(sync.diagnostics.last_speech_mouth_command_at -
                           sync.diagnostics.final_audible_audio_estimate_at, .015)

    def test_delayed_final_tts_chunk_is_processed_before_worker_shutdown(self):
        import time
        class Mouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value)
        mouth = Mouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25))
        sync.start()
        time.sleep(.02)  # Worker must remain ready while the TTS stream is briefly quiet.
        sync.note_tts_pcm_received()
        sync.submit(b"\xff\x7f" * 480, 0.04)
        sync.finish()
        self.assertTrue(any(value > 0 for value in mouth.targets))
        self.assertEqual(sync.diagnostics.shutdown_reason, "drained end-of-stream")

    def test_trailing_silence_closes_without_fake_talking(self):
        class Mouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value)
        mouth = Mouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25))
        sync.start()
        sync.submit(b"\xff\x7f" * 480, 0.0)
        sync.submit(b"\x00\x00" * 480, .02)
        sync.finish()
        self.assertIsNotNone(sync.diagnostics.last_speech_frame_audio_s)
        self.assertEqual(sync.diagnostics.final_pcm_audio_end_s, .04)
        self.assertEqual(mouth.targets[-1], 0.0)

    def test_streaming_lipsync_cancellation_stops_without_draining(self):
        class Mouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value)
        mouth = Mouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25), latency_ms=500)
        sync.start(); sync.submit(b"\xff\x7f" * 480, 0.0); sync.stop()
        self.assertEqual(sync.diagnostics.shutdown_reason, "cancelled")
        self.assertEqual(mouth.targets[-1], 0.0)

    def test_streaming_lipsync_reports_stationary_commands_and_quiet_frames(self):
        class Mouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value)
        mouth = Mouth()
        tuning = MouthTuning(sensitivity=1, max_openness=.72, smoothing=0, command_deadband_deg=.5)
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25), tuning=tuning)
        sync.start(); sync.submit(b"\xff\x7f" * 480, 0.0); sync.submit(b"\x00\x00" * 480, .02); sync.finish()
        self.assertGreaterEqual(len(sync.motion_records), 2)  # Includes final safe-close record.
        self.assertEqual(sync.diagnostics.quiet_frames, 1)
        self.assertGreaterEqual(sync.diagnostics.commanded_frames, 1)  # final safe close
        self.assertGreaterEqual(sync.diagnostics.stationary_commands_suppressed, 0)

    def test_offline_mouth_diagnostic_writes_energy_and_motion_graph_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "despina.wav"; svg_path = Path(directory) / "mouth.svg"
            with wave.open(str(wav_path), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
                out.writeframes(b"\x00\x00" * 480 + b"\xff\x7f" * 480)
            records = analyze_wav(wav_path, tuning=MouthTuning(command_deadband_deg=.1))
            write_svg(records, svg_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["target"], 0.0)
            self.assertGreater(records[1]["target"], 0.0)
            self.assertIn("generated jaw openness", svg_path.read_text(encoding="utf-8"))

    def test_slow_hardware_write_coalesces_stale_targets_instead_of_replaying_them(self):
        import time
        class SlowMouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value): self.targets.append(value); time.sleep(.06)
        mouth = SlowMouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25),
                                        tuning=MouthTuning(command_deadband_deg=.1))
        sync.start()
        for index in range(10):
            sync.submit(b"\xff\x7f" * 480, index * .02)
        sync.finish()
        self.assertGreater(sync.diagnostics.stale_frames_coalesced, 0)
        self.assertGreater(sync.diagnostics.late_hardware_commands, 0)
        self.assertGreater(sync.diagnostics.max_command_duration_ms, 40)
        self.assertLess(len(mouth.targets), 10)  # No historical command backlog.
        self.assertEqual(mouth.targets[-1], 0.0)

    def test_one_second_hardware_call_keeps_only_latest_pending_target(self):
        import time
        class BlockingMouth:
            def __init__(self): self.targets = []
            def set_openness_step(self, value):
                self.targets.append(value)
                if len(self.targets) == 1: time.sleep(1)
        mouth = BlockingMouth()
        sync = StreamingLipSynchronizer(mouth, 20, MotionSettings(25, 60, 240, .02, 25),
                                        tuning=MouthTuning(command_deadband_deg=.1))
        sync.start()
        for index in range(20): sync.submit(b"\xff\x7f" * 480, index * .02)
        sync.finish()
        self.assertGreaterEqual(sync.diagnostics.stale_frames_coalesced, 1)
        self.assertGreaterEqual(sync.diagnostics.max_command_duration_ms, 900)
        self.assertLess(len(mouth.targets), 6)  # one active, newest replacement, safe close
        self.assertEqual(mouth.targets[-1], 0.0)

    def test_offline_recording_comparison_reports_command_timing_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mouth_csv = root / "mouth.csv"; jaw_csv = root / "jaw.csv"
            mouth_csv.write_text("command_lateness_ms,coalesced_frames\n25,3\n", encoding="utf-8")
            jaw_csv.write_text("command_duration_ms\n4\n", encoding="utf-8")
            summary = compare_recordings(mouth_csv, jaw_csv)
            self.assertEqual(summary["analysis_frames"], 1)
            self.assertEqual(summary["jaw_commands"], 1)
            self.assertEqual(summary["max_scheduled_lateness_ms"], 25)
            self.assertEqual(summary["coalesced_frames"], 3)

    def test_quiet_final_words_keep_energy_targets_but_trailing_silence_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "quiet-final.wav"
            with wave.open(str(wav_path), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
                # Loud speech, then three quieter speech frames, then real silence.
                out.writeframes(b"\xff\x7f" * 480 + b"\xd6\x03" * (480 * 3) + b"\x00\x00" * (480 * 12))
            rows = analyze_wav(wav_path, tuning=MouthTuning(command_deadband_deg=.1))
            self.assertTrue(all(row["target"] > 0 for row in rows[1:4]))
            self.assertEqual(rows[-1]["target"], 0.0)

    def test_final_speech_report_includes_energy_targets_and_commanded_angles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); wav_path = root / "voice.wav"; mouth = root / "voice.mouth.csv"; jaw = root / "voice.jaw.csv"; report = root / "final.csv"
            with wave.open(str(wav_path), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(1000); out.writeframes(b"\x01\x00" * 3000)
            mouth.write_text("audio_time,rms,threshold,target,openness,commanded,hardware_completed_at\n1.5,10,5,.2,.1,True,50\n2.5,9,5,.2,.1,True,51\n", encoding="utf-8")
            jaw.write_text("timestamp,upper_jaw_angle,lower_jaw_angle,upper_write_ms,lower_write_ms\n51,4,12,1,1\n", encoding="utf-8")
            summary = final_speech_report(wav_path, mouth, jaw, report)
            self.assertEqual(summary["final_frames"], 2)
            self.assertEqual(summary["nonzero_targets"], 2)
            self.assertIn("upper_jaw_angle", report.read_text(encoding="utf-8"))

    def test_multisentence_tiny_text_chunks_are_fully_consumed(self):
        class Service:
            last_finish_reason = "STOP"
            def reply_stream(self, user, on_first_token=None):
                for index, character in enumerate("First sentence. Second sentence."):
                    if index == 0: on_first_token()
                    yield character
        answer = generate_complete_text("hello", Service(), LatencyTrace())
        self.assertEqual(answer, "First sentence. Second sentence.")

    def test_streaming_submits_entire_response_as_one_tts_segment(self):
        class Service:
            def __init__(self): self.texts = []
            def stream_speech(self, text):
                self.texts.append(text)
                yield b"\x01\x00" * 240, 0.01
                yield b"\x02\x00" * 240, 0.01
        class Player:
            def __init__(self): self.diagnostics = PlaybackDiagnostics()
            def play(self, chunks, **callbacks):
                callbacks["on_ready"](); total = 0
                for chunk in chunks:
                    if total == 0 and callbacks.get("on_first_submit"): callbacks["on_first_submit"]()
                    callbacks.get("on_chunk") and callbacks["on_chunk"](chunk, total / 48000)
                    total += len(chunk); self.diagnostics.chunks += 1
                self.diagnostics.samples_submitted = total // 2
                self.diagnostics.samples_played = total // 2
                self.diagnostics.duration_submitted_s = total / 48000
                self.diagnostics.completed_normally = True; self.diagnostics.return_code = 0
                return 0
            def stop(self): pass
        service = Service()
        text = "First sentence. Second sentence remains complete."
        speak_streaming(text, service, player_factory=Player)
        self.assertEqual(service.texts, [text])

    def test_word_timing_validation_and_fast_word_amplitude(self):
        words = [{"word": "hi", "start": 0.1, "end": 0.18},
                 {"word": "robot", "start": 0.4, "end": 0.9}]
        self.assertTrue(validate_word_timings(words, 1.0))
        events = word_events(words, duration=1.0, speed_fraction=0.4)
        fast_openings = [event["openness"] for event in events
                         if 0.1 <= event["time"] < 0.18 and event["openness"] not in (0.0, 0.12)]
        self.assertTrue(fast_openings)
        self.assertLess(max(fast_openings), 0.1)
        self.assertEqual(events[-1], {"time": 1.0, "openness": 0.0})

    def test_invalid_or_out_of_duration_word_timing_is_rejected(self):
        self.assertFalse(validate_word_timings(
            [{"word": "late", "start": 0.9, "end": 1.2}], 1.0))

    def test_word_timing_parser_and_wav_energy_fallback(self):
        self.assertEqual(parse_word_timings('[{"word":"hi","start":0,"end":0.2}]'),
                         [{"word": "hi", "start": 0.0, "end": 0.2}])
        word_motion = word_events([{"start": 0.1, "end": 0.3}])
        self.assertEqual(word_motion[0], {"time": 0.0, "openness": 0.0})
        self.assertEqual(word_motion[-1], {"time": 0.3, "openness": 0.0})
        self.assertTrue(any(event["openness"] > 0 for event in word_motion))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(8000)
                out.writeframes(b"\x00\x00" * 800 + b"\xff\x7f" * 800)
            events = energy_events(path, window_s=0.1)
            self.assertEqual(events[0]["openness"], 0.0)
            self.assertEqual(events[-1]["openness"], 0.0)

    def test_desktop_audio_backend_prefers_paplay_then_falls_back_to_aplay(self):
        both = lambda command: f"/usr/bin/{command}"
        alsa_only = lambda command: "/usr/bin/aplay" if command == "aplay" else None
        self.assertEqual(select_audio_backend("auto", both), "pulse")
        self.assertEqual(select_audio_backend("auto", alsa_only), "alsa")
        self.assertEqual(select_audio_backend("alsa", both), "alsa")
        with self.assertRaises(RuntimeError):
            select_audio_backend("pulse", alsa_only)

    def test_paplay_clock_applies_configured_bluetooth_latency_offset(self):
        calls = []
        class Process:
            def wait(self): return 0
            def poll(self): return 0
        ticks = iter([10.0, 10.25])
        player = AudioPlayer(Path("test.wav"), backend="pulse", latency_ms=100,
                             popen=lambda command: calls.append(command) or Process(),
                             clock=lambda: next(ticks), which=lambda _: "/usr/bin/paplay")
        clock = player.start()
        self.assertEqual(calls, [["paplay", "test.wav"]])
        self.assertAlmostEqual(clock(), 0.15)

    def test_microphone_vad_keeps_preroll_and_ends_after_silence(self):
        class Output:
            def __init__(self, frames): self.frames = iter(frames)
            def read(self, size): return next(self.frames, b"")
        class Process:
            def __init__(self, frames): self.stdout = Output(frames)
            def poll(self): return 0
            def terminate(self): pass
        ticks = iter(index * .02 for index in range(200))
        quiet = b"\x00\x00" * 320; speech = b"\xff\x7f" * 320
        frames = [quiet] * 3 + [speech] * 3 + [quiet] * 28
        mic = Microphone(rate=16000, popen=lambda *args, **kwargs: Process(frames), clock=lambda: next(ticks))
        mic.threshold = 100; mic.silence_s = .5
        utterance = mic.listen(max_s=3)
        self.assertIsNotNone(utterance)
        self.assertGreater(utterance.duration_s, .5)

    def test_offline_vosk_wake_detector_uses_constrained_local_result(self):
        class Recognizer:
            def AcceptWaveform(self, pcm): return False
            def PartialResult(self): return '{"partial": "ziggy"}'
        detector = VoskWakeWord(Recognizer(), sample_rate=16000, variants={"ziggy"})
        self.assertTrue(detector.detected(b"\x01\x00\xff\xff"))

    def test_vosk_false_trigger_text_is_rejected(self):
        class Recognizer:
            def AcceptWaveform(self, pcm): return True
            def Result(self): return '{"text": "background conversation"}'
        detector = VoskWakeWord(Recognizer(), sample_rate=16000, variants={"ziggy", "ziggie"})
        self.assertFalse(detector.detected(b"\x01\x00\xff\xff"))

    def test_vosk_model_must_be_cached_and_is_never_downloaded_at_startup(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"HEAD_VOSK_MODEL_PATH": str(Path(directory) / "missing")}, clear=False):
            with self.assertRaises(VoskWakeWordError):
                VoskWakeWord.from_environment()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model"; (target / "am").mkdir(parents=True)
            (target / "am" / "final.mdl").write_bytes(b"cached")
            self.assertEqual(ensure_model(target=target), target)

    def test_background_audio_is_not_transcribed_until_local_wake_detects(self):
        class MicrophoneDouble:
            device = "plughw:2,0"; rate = 16000; silence_s = 1.0
            def __init__(self): self.blocks = iter([b"background", b"ziggy", KeyboardInterrupt()]); self.prefixes = []
            def read_pcm(self, size):
                item = next(self.blocks)
                if isinstance(item, BaseException): raise item
                return item
            def record_after_wake(self, prefix):
                self.prefixes.append(prefix); return Utterance(b"question", .5)
            def write_wav(self, utterance, path): path.write_bytes(utterance.pcm); return path
            def close(self): pass
        class Wake:
            frame_length = 2; sample_rate = 16000
            def detected(self, pcm): return pcm == b"ziggy"
            def close(self): pass
        class WakeFactory:
            @classmethod
            def from_environment(cls): return Wake()
        mic = MicrophoneDouble(); service = type("Service", (), {"calls": 0})()
        def transcribe(path): service.calls += 1; return "Ziggy, what is your name?"
        service.transcribe_wav = transcribe
        with patch.object(robot_chat, "generate_complete_text", return_value="I am Ziggy."), \
             patch.object(robot_chat, "speak_streaming") as speak:
            with self.assertRaises(KeyboardInterrupt):
                robot_chat.run_listen_session(service, mode="streaming", microphone_factory=lambda: mic,
                                               wake_factory=WakeFactory)
        self.assertEqual(service.calls, 1)
        self.assertEqual(mic.prefixes, [b"backgroundziggy"])
        speak.assert_called_once()

    def test_wake_state_machine_handles_two_complete_turns_without_duplicates(self):
        class MicrophoneDouble:
            device = "plughw:2,0"; rate = 16000; silence_s = 1.0
            def __init__(self): self.blocks = iter([b"ziggy", b"ziggy", KeyboardInterrupt()]); self.recorded = 0
            def read_pcm(self, size):
                item = next(self.blocks)
                if isinstance(item, BaseException): raise item
                return item
            def record_after_wake(self, prefix):
                self.recorded += 1; return Utterance(b"question", .5)
            def write_wav(self, utterance, path): path.write_bytes(utterance.pcm); return path
            def close(self): pass
        class Wake:
            frame_length = 2; sample_rate = 16000
            def detected(self, pcm): return True
            def close(self): pass
        class WakeFactory:
            @classmethod
            def from_environment(cls): return Wake()
        mic = MicrophoneDouble(); service = type("Service", (), {"calls": 0})()
        def transcribe(path): service.calls += 1; return f"question {service.calls}"
        service.transcribe_wav = transcribe
        with patch.object(robot_chat, "generate_complete_text", return_value="answer"), \
             patch.object(robot_chat, "speak_streaming") as speak:
            with self.assertRaises(KeyboardInterrupt):
                robot_chat.run_listen_session(service, mode="streaming", microphone_factory=lambda: mic,
                                               wake_factory=WakeFactory)
        self.assertEqual((mic.recorded, service.calls, speak.call_count), (2, 2, 2))

    def test_ziggy_uses_one_second_silence_even_with_legacy_mic_setting(self):
        with patch.dict("os.environ", {"HEAD_MIC_SILENCE_MS": "550"}, clear=False):
            self.assertEqual(ziggy_silence_seconds(), 1.0)

    def test_post_wake_recording_keeps_short_pause_and_ends_after_one_second(self):
        quiet = b"\x00\x00" * 320; speech = b"\xff\x7f" * 320
        frames = [speech] * 3 + [quiet] * 20 + [speech] * 3 + [quiet] * 50
        ticks = iter(index * .02 for index in range(200))
        mic = Microphone(clock=lambda: next(ticks))
        mic.start = lambda: None
        mic._read_frame = lambda: frames.pop(0)
        mic.threshold = 100; mic.silence_s = 1.0
        utterance = mic.record_after_wake()
        self.assertGreaterEqual(utterance.duration_s, 1.5)
        self.assertEqual(len(frames), 0)

    def test_hands_free_startup_never_constructs_or_prompts_for_jaws(self):
        service = object()
        with patch.object(robot_chat.GeminiConfig, "from_environment", return_value=object()), \
             patch.object(robot_chat, "GeminiService", return_value=service), \
             patch.object(robot_chat, "ServoController") as controller, \
             patch.object(robot_chat, "run_listen_session", return_value=0) as listening:
            self.assertEqual(robot_chat.main(["--listen", "--streaming", "--skip-model-check"]), 0)
        controller.assert_not_called()
        self.assertIsNone(listening.call_args.kwargs["mouth"])

    def test_supervised_jaw_reference_adoption_sends_no_servo_command(self):
        configs = {
            "upper_jaw": ServoConfig("P8", 1, 1, -5, 21, 21, 1),
            "lower_jaw": ServoConfig("P10", 15, 1, 0, 20, 0, -1),
        }
        backend = QuietMock(); controller = ServoController(configs, backend, sleep=lambda _: None)
        replies = iter(["1", "POWERED", "15", "POWERED"])
        self.assertTrue(establish_mouth(configs, controller, input_func=lambda _: next(replies)))
        self.assertEqual(backend.commands, [])
        self.assertEqual(controller.position("upper_jaw"), 1)
        self.assertEqual(controller.position("lower_jaw"), 15)

    def test_quick_head_roll_calibration_saves_only_verified_plus_minus_two_limits(self):
        configs = {"head_roll": ServoConfig("P0", 70, 1, None, None)}
        backend = QuietMock(); controller = ServoController(configs, backend, sleep=lambda _: None)
        replies = iter(["70", "POWERED", "TEST 69", "SAFE", "TEST 68", "SAFE", "TEST 69", "SAFE",
                        "TEST 70", "SAFE", "TEST 71", "SAFE", "TEST 72", "SAFE", "SAVE LIMITS"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"
            self.assertEqual(quick_head_roll(configs, controller, path, input_func=lambda _: next(replies)), 0)
            saved = load_config(path)["head_roll"]
        self.assertEqual((saved.min_angle, saved.neutral, saved.max_angle), (68, 70, 72))
        self.assertTrue(all(port == "P0" for port, _ in backend.commands))

    def test_head_roll_animation_uses_saved_60_to_80_verified_limits(self):
        configs = {"head_roll": ServoConfig("P0", 70, 1, 60, 80)}
        backend = QuietMock(); controller = ServoController(configs, backend, sleep=lambda _: None)
        controller.adopt_confirmed_position("head_roll", 70)
        animator = HeadRollAnimator(controller, clock=lambda: 0.0, sleep=lambda _: None, update_s=.5)
        animator._transition(80)
        self.assertEqual(controller.position("head_roll"), 80)
        self.assertTrue(all(port == "P0" and 60 <= angle <= 80 for port, angle in backend.commands))

    def test_head_roll_extension_saves_each_new_safe_endpoint_only(self):
        configs = {"head_roll": ServoConfig("P0", 70, 1, 68, 72)}
        backend = QuietMock(); controller = ServoController(configs, backend, sleep=lambda _: None)
        replies = iter(["72", "POWERED", "TEST 73", "SAFE", "STOP"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"
            self.assertEqual(extend_head_roll(configs, controller, path, input_func=lambda _: next(replies)), 0)
            saved = load_config(path)["head_roll"]
        self.assertEqual((saved.min_angle, saved.max_angle), (68, 73))
        self.assertTrue(all(port == "P0" and angle == 73 for port, angle in backend.commands))

    def test_head_roll_extension_refuses_non_endpoint_without_movement(self):
        configs = {"head_roll": ServoConfig("P0", 70, 1, 68, 72)}
        backend = QuietMock(); controller = ServoController(configs, backend, sleep=lambda _: None)
        replies = iter(["70", "POWERED"])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(extend_head_roll(configs, controller, Path(directory) / "servos.json",
                                              input_func=lambda _: next(replies)), 1)
        self.assertEqual(backend.commands, [])

    def test_wide_head_roll_setup_saves_60_to_80_only_after_both_sides_safe(self):
        configs = {"head_roll": ServoConfig("P0", 70, 1, None, None)}
        backend = QuietMock(); controller = ServoController(configs, backend, sleep=lambda _: None)
        route = list(range(71, 81)) + list(range(79, 69, -1)) + list(range(69, 59, -1))
        replies = iter(["70", "POWERED"] + [item for angle in route
                                              for item in (f"TEST {angle}", "SAFE")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"
            self.assertEqual(wide_head_roll(configs, controller, path, input_func=lambda _: next(replies)), 0)
            saved = load_config(path)["head_roll"]
        self.assertEqual((saved.min_angle, saved.neutral, saved.max_angle), (60, 70, 80))
        self.assertTrue(all(port == "P0" and 60 <= angle <= 80 for port, angle in backend.commands))

    def test_wide_head_roll_interruption_preserves_saved_range(self):
        configs = {"head_roll": ServoConfig("P0", 70, 1, None, None)}
        backend = QuietMock(); controller = ServoController(configs, backend, sleep=lambda _: None)
        replies = iter(["70", "POWERED", "TEST 71", "STOP"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servos.json"; save_config(path, configs)
            self.assertEqual(wide_head_roll(configs, controller, path, input_func=lambda _: next(replies)), 1)
            saved = load_config(path)["head_roll"]
        self.assertEqual((saved.min_angle, saved.max_angle), (None, None))

    def test_microphone_uses_one_continuous_raw_process_for_multiple_turns(self):
        calls = []
        quiet = b"\x00\x00" * 320; speech = b"\xff\x7f" * 320
        # Each utterance has 550 ms of trailing quiet frames; both come from one pipe.
        frames = [speech] * 2 + [quiet] * 28 + [speech] * 2 + [quiet] * 28
        class Output:
            def read(self, size): return frames.pop(0) if frames else b""
            def close(self): pass
        class Process:
            stdout = Output()
            def poll(self): return None
            def terminate(self): pass
            def wait(self, timeout=None): return 0
        ticks = iter(index * .02 for index in range(300))
        mic = Microphone(device="plughw:2,0", popen=lambda *args, **kwargs: calls.append(args[0]) or Process(),
                         clock=lambda: next(ticks))
        mic.threshold = 100; mic.silence_s = .5
        self.assertIsNotNone(mic.listen())
        self.assertIsNotNone(mic.listen())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["arecord", "-q", "-D", "plughw:2,0", "-f", "S16_LE", "-r", "16000",
                                    "-c", "1", "-t", "raw"])

    def test_microphone_assembles_partial_pcm_reads(self):
        quiet = b"\x00\x00" * 320; speech = b"\xff\x7f" * 320
        pieces = [speech[:103], speech[103:], *([quiet] * 28)]
        class Output:
            def read(self, size): return pieces.pop(0) if pieces else b""
            def close(self): pass
        class Process:
            stdout = Output()
            def poll(self): return None
            def terminate(self): pass
            def wait(self, timeout=None): return 0
        ticks = iter(index * .02 for index in range(100))
        mic = Microphone(popen=lambda *args, **kwargs: Process(), clock=lambda: next(ticks))
        mic.threshold = 100; mic.silence_s = .5
        utterance = mic.listen()
        self.assertIsNotNone(utterance)
        self.assertGreaterEqual(utterance.duration_s, .5)

    def test_microphone_reports_exit_code_and_alsa_error_after_bounded_recovery(self):
        class Output:
            def read(self, size): return b""
            def close(self): pass
        class Error:
            def __init__(self): self.read_once = False
            def read(self, size):
                if self.read_once: return b""
                self.read_once = True
                return b"arecord: audio open error: Device or resource busy\n"
        class Process:
            stdout = Output(); stderr = Error()
            def poll(self): return 1
            def terminate(self): pass
            def wait(self, timeout=None): return 1
        ticks = iter(index * .02 for index in range(100))
        mic = Microphone(popen=lambda *args, **kwargs: Process(), clock=lambda: next(ticks))
        mic.recovery_attempts = 0
        with self.assertRaisesRegex(RuntimeError, r"exit code: 1.*audio open error"):
            mic.listen()

    def test_microphone_recovers_once_then_records_without_reinitializing_per_turn(self):
        quiet = b"\x00\x00" * 320; speech = b"\xff\x7f" * 320
        class Empty:
            def read(self, size): return b""
            def close(self): pass
        class Frames:
            def __init__(self): self.frames = [speech] * 2 + [quiet] * 28
            def read(self, size): return self.frames.pop(0) if self.frames else b""
            def close(self): pass
        class Failed:
            stdout = Empty()
            def poll(self): return 1
            def terminate(self): pass
            def wait(self, timeout=None): return 1
        class Working:
            stdout = Frames()
            def poll(self): return None
            def terminate(self): pass
            def wait(self, timeout=None): return 0
        processes = iter([Failed(), Working()]); ticks = iter(index * .02 for index in range(100))
        mic = Microphone(popen=lambda *args, **kwargs: next(processes), clock=lambda: next(ticks))
        mic.threshold = 100; mic.silence_s = .5; mic.recovery_attempts = 1
        self.assertIsNotNone(mic.listen())

    def test_microphone_cleanup_terminates_persistent_process(self):
        class Output:
            def close(self): pass
        class Process:
            stdout = Output(); stderr = Output(); terminated = False
            def poll(self): return None
            def terminate(self): self.terminated = True
            def wait(self, timeout=None): return 0
        process = Process()
        mic = Microphone(popen=lambda *args, **kwargs: process)
        mic.start(); mic.close()
        self.assertTrue(process.terminated)
        self.assertIsNone(mic.process)

    def test_transcription_model_is_verified_and_short_audio_is_sent(self):
        class Models:
            def get(self, *, model): return type("M", (), {"name": model})()
        class Files:
            def upload(self, *, file): self.file = file; return type("F", (), {"uri": "test://audio"})()
        class Interactions:
            def create(self, **kwargs): self.kwargs = kwargs; return type("R", (), {"output_text": " hello head "})()
        client = type("Client", (), {"models": Models(), "files": Files(), "interactions": Interactions()})()
        service = GeminiService(GeminiConfig("key", "chat", "tts", "Despina", "align", transcription_model="transcribe"), client)
        self.assertEqual(service.verify_models()["transcription"], "transcribe")
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "voice.wav"; wav.write_bytes(b"RIFFtest")
            self.assertEqual(service.transcribe_wav(wav), "hello head")
        self.assertEqual(client.interactions.kwargs["model"], "transcribe")


if __name__ == "__main__":
    unittest.main()
