"""Terminal Gemini chat with Despina audio and optional playback-clock mouth sync."""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path

from .adaptive_lipsync import AdaptiveLipSynchronizer, calibrated_max_jaw_travel, load_motion_settings
from .audio_sync import Aplay, energy_events, word_events
from .gemini_service import GeminiConfig, GeminiError, GeminiService
from .mouth import JAW_NAMES, MouthController
from .mouth_calibration import _known_reference
from .servo_controller import ConfigurationError, RobotHatBackend, ServoController, load_config

def establish_mouth(configs, controller, input_func=input) -> bool:
    for name in JAW_NAMES:
        cfg = configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None: raise ConfigurationError(f"{name} is not calibrated")
        if not _known_reference(name, cfg, controller, input_func): return False
        if controller.position(name) != cfg.neutral:
            print("Mouth mode requires both jaws to be confirmed at their saved closed neutral. Use voice-only otherwise.")
            return False
    return True

def speak(text, service: GeminiService, *, mouth=None, configs=None, player_factory=Aplay) -> None:
    with tempfile.TemporaryDirectory(prefix="head-tts-") as directory:
        wav = service.synthesize(text, Path(directory) / "despina.wav")
        words = service.word_timings(wav)
        events = word_events(words) if words else energy_events(wav)
        print("Mouth timing:", "word timing" if words else "audio-energy fallback")
        player = player_factory(wav)
        clock = player.start()
        try:
            if mouth is not None:
                sync = AdaptiveLipSynchronizer(mouth, calibrated_max_jaw_travel(configs), load_motion_settings())
                sync.run(events, clock)
                print(f"Mouth timing records: {len(sync.records)}")
            player.wait()
        except KeyboardInterrupt:
            player.stop()
            raise

def run_session(service, *, voice_only: bool, mouth=None, configs=None, input_func=input) -> int:
    print("HEAD ROBOT - AI CHAT\n")
    while True:
        try: user = input_func("You: ").strip()
        except EOFError: return 0
        if user.lower() in {"exit", "quit"}: return 0
        if not user: continue
        print("Thinking...")
        try:
            answer = service.reply(user)
            print(f"\nRobot: {answer}\n\nGenerating Despina voice...\nSpeaking...")
            speak(answer, service, mouth=None if voice_only else mouth, configs=configs)
        except GeminiError as exc: print(f"Gemini error: {exc}")
        except RuntimeError as exc: print(f"Audio error: {exc}")

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gemini + Despina HEAD robot terminal chat")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--voice-only", action="store_true")
    mode.add_argument("--mouth-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        service = GeminiService(GeminiConfig.from_environment())
        if args.voice_only: return run_session(service, voice_only=True)
        configs = load_config(Path(__file__).resolve().parents[1] / "config" / "servos.json")
        controller = ServoController(configs, RobotHatBackend())
        if not establish_mouth(configs, controller):
            print("Mouth movement disabled. Re-run with --voice-only to speak without servos.")
            return 1
        mouth = MouthController(controller)
        if args.mouth_test:
            speak("Hello, this is a short mouth synchronization test.", service, mouth=mouth, configs=configs)
            return 0
        return run_session(service, voice_only=False, mouth=mouth, configs=configs)
    except KeyboardInterrupt:
        print("\nSTOPPED: no future mouth commands will be sent. Physically disconnect servo power if unsafe.")
        return 130
    except (GeminiError, ConfigurationError, RuntimeError) as exc:
        print(f"Setup error: {exc}")
        return 2

if __name__ == "__main__": raise SystemExit(main())
