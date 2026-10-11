"""Terminal Gemini chat with Despina audio and optional playback-clock mouth sync."""
from __future__ import annotations
import argparse
import tempfile
import threading
from pathlib import Path

from .adaptive_lipsync import AdaptiveLipSynchronizer, calibrated_max_jaw_travel, load_motion_settings
from .audio_sync import AudioPlayer, RawPCMPlayer, energy_events, word_events
from .gemini_service import GeminiConfig, GeminiError, GeminiService
from .latency import LatencyTrace
from .mouth import JAW_NAMES, MouthController
from .mouth_calibration import _known_reference
from .servo_controller import ConfigurationError, RobotHatBackend, ServoController, load_config
from .streaming_lipsync import StreamingLipSynchronizer

def establish_mouth(configs, controller, input_func=input) -> bool:
    for name in JAW_NAMES:
        cfg = configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None: raise ConfigurationError(f"{name} is not calibrated")
        if not _known_reference(name, cfg, controller, input_func): return False
        if controller.position(name) != cfg.neutral:
            print("Mouth mode requires both jaws to be confirmed at their saved closed neutral. Use voice-only otherwise.")
            return False
    return True

def speak(text, service: GeminiService, *, mouth=None, configs=None, player_factory=AudioPlayer,
          trace: LatencyTrace | None = None) -> None:
    """Complete-WAV compatibility path."""
    with tempfile.TemporaryDirectory(prefix="head-tts-") as directory:
        if trace: trace.mark("tts_request")
        wav = service.synthesize(text, Path(directory) / "despina.wav")
        if trace:
            trace.mark("tts_first_chunk")
            trace.mark("audio_prepare_start")
        words = service.word_timings(wav)
        events = word_events(words) if words else energy_events(wav)
        if trace:
            trace.mark("audio_prepare_end")
            trace.span("Audio decoding/preparation", "audio_prepare_start", "audio_prepare_end")
            trace.add_ms("Speech alignment", trace.values_ms["Audio decoding/preparation"])
        print("Mouth timing:", "word timing" if words else "audio-energy fallback")
        player = player_factory(wav)
        if trace: trace.mark("playback_init_start")
        clock = player.start()
        if trace:
            trace.mark("playback_init_end")
            trace.mark("first_audio_submitted")
        try:
            if mouth is not None:
                sync = AdaptiveLipSynchronizer(mouth, calibrated_max_jaw_travel(configs), load_motion_settings())
                sync.run(events, clock)
                print(f"Mouth timing records: {len(sync.records)}")
            player.wait()
        except KeyboardInterrupt:
            player.stop()
            raise


def speak_streaming(text, service: GeminiService, *, mouth=None, configs=None,
                    trace: LatencyTrace | None = None, player_factory=RawPCMPlayer) -> None:
    """Stream raw PCM to playback; incremental lip analysis runs off-thread."""
    player = player_factory()
    lip = None
    if mouth is not None:
        lip = StreamingLipSynchronizer(
            mouth, calibrated_max_jaw_travel(configs), load_motion_settings(), sample_rate=24000,
        )

    def chunks():
        if trace: trace.mark("tts_request")
        first = True
        for data, decode_ms in service.stream_speech(text):
            if trace:
                trace.add_ms("Audio decoding/preparation", decode_ms)
                if first: trace.mark("tts_first_chunk")
            first = False
            yield data

    try:
        if trace: trace.mark("playback_init_start")
        def ready():
            if trace: trace.mark("playback_init_end")
            if lip: lip.start()
        player.play(
            chunks(),
            on_ready=ready,
            on_first_submit=(lambda: trace.mark("first_audio_submitted")) if trace else None,
            on_chunk=(lambda data, offset: lip.submit(data, offset)) if lip else None,
        )
        if lip and trace:
            trace.add_ms("Speech alignment", lip.analysis_ms)
    except KeyboardInterrupt:
        player.stop()
        if lip: lip.stop(close=False)
        raise
    finally:
        if lip and not lip.cancelled.is_set():
            lip.stop(close=True)

def run_session(service, *, voice_only: bool, mouth=None, configs=None, input_func=input,
                streaming: bool = True) -> int:
    print("HEAD ROBOT - AI CHAT\n")
    while True:
        try: user = input_func("You: ").strip()
        except EOFError: return 0
        if user.lower() in {"exit", "quit"}: return 0
        if not user: continue
        trace = LatencyTrace()
        print("Thinking...")
        try:
            if not streaming:
                trace.mark("text_request")
                answer = service.reply_complete(user)
                trace.mark("text_first_token"); trace.mark("text_complete")
                print(f"\nRobot: {answer}\n\nGenerating Despina voice...\nSpeaking...")
                speak(answer, service, mouth=None if voice_only else mouth, configs=configs, trace=trace)
            else:
                trace.mark("text_request")
                print("\nRobot: ", end="", flush=True)
                pieces = []
                spoken_prefix = ""
                first_phrase_spoken = False
                drain_thread = None
                drain_errors = []
                text_stream = service.reply_stream(user, on_first_token=lambda: trace.mark("text_first_token"))

                for piece in text_stream:
                    pieces.append(piece)
                    print(piece, end="", flush=True)
                    accumulated = "".join(pieces)
                    # Begin TTS at the first complete phrase while Gemini may
                    # still be producing a second sentence.
                    if not first_phrase_spoken:
                        match = __import__("re").search(r"[.!?](?:\s|$)", accumulated)
                        if match:
                            spoken_prefix = accumulated[:match.end()].strip()
                            def drain_text():
                                try:
                                    for remainder_piece in text_stream:
                                        pieces.append(remainder_piece)
                                        print(remainder_piece, end="", flush=True)
                                    trace.mark("text_complete")
                                except BaseException as exc:
                                    drain_errors.append(exc)
                            drain_thread = threading.Thread(target=drain_text, name="gemini-text-drain", daemon=True)
                            drain_thread.start()
                            try:
                                speak_streaming(spoken_prefix, service, mouth=None if voice_only else mouth,
                                                configs=configs, trace=trace)
                            except GeminiError as exc:
                                print(f"\nStreaming unavailable ({exc}); using complete-WAV fallback.")
                                speak(spoken_prefix, service, mouth=None if voice_only else mouth,
                                      configs=configs, trace=trace)
                            first_phrase_spoken = True
                            break
                if drain_thread:
                    drain_thread.join()
                if drain_errors:
                    raise drain_errors[0]
                answer = "".join(pieces).strip()
                trace.mark("text_complete")
                print("\n\nStreaming Despina voice...\n")
                try:
                    if not first_phrase_spoken:
                        spoken_prefix = answer
                        speak_streaming(answer, service, mouth=None if voice_only else mouth,
                                        configs=configs, trace=trace)
                    remainder = answer[len(spoken_prefix):].strip()
                    if remainder:
                        speak_streaming(remainder, service, mouth=None if voice_only else mouth,
                                        configs=configs)
                except GeminiError as exc:
                    print(f"Streaming unavailable ({exc}); using complete-WAV fallback.")
                    speak(answer, service, mouth=None if voice_only else mouth, configs=configs, trace=trace)
            latency = float(__import__("os").getenv("HEAD_AUDIO_LATENCY_MS", "0"))
            trace.finish(latency)
            print("\n" + trace.format() + "\n")
        except GeminiError as exc: print(f"Gemini error: {exc}")
        except RuntimeError as exc: print(f"Audio error: {exc}")

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gemini + Despina HEAD robot terminal chat")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--voice-only", action="store_true")
    mode.add_argument("--mouth-test", action="store_true")
    mode.add_argument("--voice-test", action="store_true", help="play one Despina test phrase without servos")
    parser.add_argument("--complete-audio", action="store_true", help="disable streaming and use the complete-WAV fallback")
    parser.add_argument("--skip-model-check", action="store_true", help="skip startup model access verification")
    args = parser.parse_args(argv)
    try:
        service = GeminiService(GeminiConfig.from_environment())
        if not args.skip_model_check:
            models = service.verify_models()
            print("Verified models:", ", ".join(f"{key}={value}" for key, value in models.items()))
        if args.voice_test:
            speak("Hello. This is a Despina Bluetooth speaker test.", service)
            return 0
        if args.voice_only: return run_session(service, voice_only=True, streaming=not args.complete_audio)
        configs = load_config(Path(__file__).resolve().parents[1] / "config" / "servos.json")
        controller = ServoController(configs, RobotHatBackend())
        if not establish_mouth(configs, controller):
            print("Mouth movement disabled. Re-run with --voice-only to speak without servos.")
            return 1
        mouth = MouthController(controller)
        if args.mouth_test:
            speak("Hello, this is a short mouth synchronization test.", service, mouth=mouth, configs=configs)
            return 0
        return run_session(service, voice_only=False, mouth=mouth, configs=configs,
                           streaming=not args.complete_audio)
    except KeyboardInterrupt:
        print("\nSTOPPED: no future mouth commands will be sent. Physically disconnect servo power if unsafe.")
        return 130
    except (GeminiError, ConfigurationError, RuntimeError) as exc:
        print(f"Setup error: {exc}")
        return 2

if __name__ == "__main__": raise SystemExit(main())
