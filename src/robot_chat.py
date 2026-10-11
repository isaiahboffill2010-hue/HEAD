"""Reliable and streaming Gemini + Despina terminal conversation."""
from __future__ import annotations
import argparse, csv, os, tempfile, time, wave
from collections import deque
from enum import Enum
from pathlib import Path

from .adaptive_lipsync import AdaptiveLipSynchronizer, calibrated_max_jaw_travel, load_motion_settings
from .audio_sync import (AudioPlayer, RawPCMPlayer, energy_events, limit_event_amplitudes,
                         validate_word_timings, wav_duration, word_events)
from .gemini_service import GeminiConfig, GeminiError, GeminiService
from .latency import LatencyTrace
from .mouth import JAW_NAMES, MouthController
from .mouth_calibration import _known_reference
from .servo_controller import ConfigurationError, RobotHatBackend, ServoController, load_config
from .streaming_lipsync import StreamingLipSynchronizer
from .microphone import Microphone
from .wake_word import PorcupineWakeWord, WakeWordConfigurationError


class ConversationState(str, Enum):
    WAITING_FOR_WAKE_WORD = "WAITING_FOR_WAKE_WORD"
    RECORDING_QUESTION = "RECORDING_QUESTION"
    TRANSCRIBING = "TRANSCRIBING"
    RESPONDING = "RESPONDING"


def ziggy_silence_seconds() -> float:
    """Separate wake-word turn timing from legacy push-to-talk VAD tuning."""
    try:
        milliseconds = float(os.getenv("HEAD_ZIGGY_SILENCE_MS", "1000"))
    except ValueError:
        milliseconds = 1000.0
    return min(1.2, max(.45, milliseconds / 1000))


def establish_mouth(configs, controller, input_func=input) -> bool:
    for name in JAW_NAMES:
        cfg = configs[name]
        if cfg.neutral is None or cfg.opening_endpoint is None:
            raise ConfigurationError(f"{name} is not calibrated")
        if not _known_reference(name, cfg, controller, input_func): return False
        if controller.position(name) != cfg.neutral:
            print("Mouth mode requires both jaws at saved closed neutral. Use --voice-only otherwise.")
            return False
    return True


def _debug_path(debug: bool, suffix: str):
    if debug:
        directory = Path("debug_audio"); directory.mkdir(exist_ok=True)
        unique = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
        return directory / f"head-{unique}.{suffix}", None
    temporary = tempfile.TemporaryDirectory(prefix="head-tts-")
    return Path(temporary.name) / f"despina.{suffix}", temporary


def _print_diagnostics(text, segments, diagnostics, *, mode, audio_path=None, alignment=None,
                       dropped_segments=0, lip=None):
    print("\nSpeech diagnostics:")
    print(f"  mode: {mode}\n  full response: {text!r}\n  TTS segments ({len(segments)}): {segments!r}")
    print(f"  audio chunks: {diagnostics.chunks}\n  submitted samples: {diagnostics.samples_submitted}")
    print(f"  played samples: {diagnostics.samples_played}\n  submitted duration: {diagnostics.duration_submitted_s:.3f} s")
    print(f"  maximum PCM output lead: {diagnostics.max_pcm_ahead_ms:.1f} ms\n  PCM pacing wait: {diagnostics.pacing_wait_ms:.1f} ms")
    print(f"  playback completed normally: {diagnostics.completed_normally}\n  playback return code: {diagnostics.return_code}")
    print(f"  failed producer: {diagnostics.producer_error or 'none'}\n  dropped/failed TTS segments: {dropped_segments}")
    print(f"  dropped mouth-analysis frames: {diagnostics.dropped_chunks}")
    if lip:
        detail = lip.diagnostics
        print("  final TTS PCM frame timestamp: " + str(detail.final_tts_pcm_received_at))
        print("  last speech-bearing frame timestamp: " + str(detail.last_speech_frame_audio_s))
        print("  last scheduled mouth movement timestamp: " + str(detail.last_scheduled_mouth_at))
        print("  estimated final audible-audio timestamp: " + str(detail.final_audible_audio_estimate_at))
        print("  last speech-bearing mouth command timestamp: " + str(detail.last_speech_mouth_command_at))
        if detail.final_audible_audio_estimate_at and detail.last_speech_mouth_command_at:
            delta = (detail.last_speech_mouth_command_at - detail.final_audible_audio_estimate_at) * 1000
            print(f"  final mouth-vs-audible estimate: {delta:+.1f} ms")
        print("  audio playback completion timestamp: " + str(detail.playback_completed_at))
        print("  last mouth-frame processing timestamp: " + str(detail.last_mouth_frame_processed_at))
        print(f"  mouth worker shutdown reason: {detail.shutdown_reason or 'not finished'}")
        print(f"  speech/quiet analysis frames: {detail.speech_frames}/{detail.quiet_frames}")
        print(f"  commanded/suppressed stationary mouth frames: {detail.commanded_frames}/{detail.stationary_commands_suppressed}")
        print(f"  coalesced stale frames: {detail.stale_frames_coalesced}")
        print(f"  late hardware commands: {detail.late_hardware_commands}; max lateness: {detail.max_command_lateness_ms:.1f} ms")
        print(f"  max jaw-command execution duration: {detail.max_command_duration_ms:.1f} ms")
        print("  final-word diagnosis: " + lip.final_speech_summary())
    if alignment: print(f"  mouth alignment: {alignment}")
    if audio_path: print(f"  preserved audio: {audio_path}")


def generate_complete_text(user: str, service: GeminiService, trace: LatencyTrace) -> str:
    """Fully consume Gemini's iterator into a buffer independent of TTS."""
    trace.mark("text_request"); pieces = []
    print("\nRobot: ", end="", flush=True)
    for piece in service.reply_stream(user, on_first_token=lambda: trace.mark("text_first_token")):
        pieces.append(piece); print(piece, end="", flush=True)
    trace.mark("text_complete"); answer = "".join(pieces).strip()
    print(f"\nGemini finish reason: {service.last_finish_reason}")
    usage = getattr(service, "last_generation_usage", {})
    print("Gemini usage: " + ", ".join(
        f"{name}={usage.get(name, 'unavailable')}" for name in
        ("output_tokens", "thinking_tokens", "input_tokens", "total_tokens")))
    print(f"Gemini generation attempts: {getattr(service, 'last_generation_attempts', 1)}")
    return answer


def speak_reliable(text, service: GeminiService, *, mouth=None, configs=None, trace=None,
                   debug_audio=False, player_factory=AudioPlayer):
    """Complete audio, validate alignment, then share one playback clock."""
    wav, temporary = _debug_path(debug_audio, "wav")
    try:
        if trace: trace.mark("tts_request")
        service.synthesize(text, wav)
        if trace: trace.mark("tts_first_chunk"); trace.mark("audio_prepare_start")
        duration = wav_duration(wav); events = None; alignment = "disabled (voice-only)"
        if mouth is not None:
            began = time.monotonic(); words = service.word_timings(wav)
            if trace: trace.add_ms("Speech alignment", (time.monotonic() - began) * 1000)
            settings = load_motion_settings(); travel = calibrated_max_jaw_travel(configs)
            if validate_word_timings(words, duration):
                events = word_events(words, duration, settings.speed() / travel)
                alignment = f"validated word timing ({len(words)} words)"
            else:
                events = energy_events(wav)
                detail = f"; aligner error={service.last_alignment_error}" if service.last_alignment_error else ""
                alignment = f"40 ms actual-audio energy fallback (not exact word sync){detail}"
            events = limit_event_amplitudes(events, settings.speed() / travel)
        if trace:
            trace.mark("audio_prepare_end"); trace.span("Audio decoding/preparation", "audio_prepare_start", "audio_prepare_end")
            trace.mark("playback_init_start")
        player = player_factory(wav); clock = player.start()
        if trace: trace.mark("playback_init_end"); trace.mark("first_audio_submitted")
        try:
            if mouth is not None:
                AdaptiveLipSynchronizer(mouth, calibrated_max_jaw_travel(configs), load_motion_settings()).run(
                    events, clock, settle_after_final=False)
            code = player.wait()
            if code != 0: raise RuntimeError(f"audio player exited with status {code}")
        except KeyboardInterrupt:
            player.stop(); raise
        _print_diagnostics(text, [text], player.diagnostics, mode="reliable",
                           audio_path=wav if debug_audio else None, alignment=alignment)
    finally:
        if temporary: temporary.cleanup()


def speak_streaming(text, service: GeminiService, *, mouth=None, configs=None, trace=None,
                    debug_audio=False, player_factory=RawPCMPlayer, mouth_offset_ms=None):
    """One ordered TTS stream for the entire completed response."""
    player = player_factory(); lip = None; capture = bytearray(); saved = None; lip_finished = False
    if mouth is not None and debug_audio and hasattr(mouth, "command_log"):
        mouth.command_log.clear()
    if mouth is not None:
        lip = StreamingLipSynchronizer(mouth, calibrated_max_jaw_travel(configs), load_motion_settings(),
                                       mouth_offset_ms=mouth_offset_ms)
    def chunks():
        if trace: trace.mark("tts_request")
        first = True
        for data, decode_ms in service.stream_speech(text):
            if trace:
                trace.add_ms("Audio decoding/preparation", decode_ms)
                if first: trace.mark("tts_first_chunk")
            first = False
            if lip: lip.note_tts_pcm_received()
            if debug_audio: capture.extend(data)
            yield data
        tail = b"\x00\x00" * 2880
        if debug_audio: capture.extend(tail)
        yield tail
    try:
        if trace: trace.mark("playback_init_start")
        def ready():
            if trace: trace.mark("playback_init_end")
            if lip: lip.start()
        code = player.play(chunks(), on_ready=ready,
            on_first_submit=(lambda: trace.mark("first_audio_submitted")) if trace else None,
            on_chunk=(lambda data, offset: lip.submit(data, offset)) if lip else None)
        if code != 0: raise RuntimeError(f"audio player exited with status {code}")
        if lip:
            # RawPCMPlayer has drained paplay/aplay before returning.  Drain the
            # mouth queue as well; do not treat TTS end-of-stream as mouth EOS.
            lip.finish(player.diagnostics.playback_completed_at,
                       playback_started_at=player.diagnostics.playback_started_at)
            lip_finished = True
        if lip and trace: trace.add_ms("Speech alignment", lip.analysis_ms)
        if lip: player.diagnostics.dropped_chunks = lip.dropped_chunks
        if debug_audio:
            saved, _ = _debug_path(True, "wav")
            with wave.open(str(saved), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000); out.writeframes(capture)
            if lip:
                timeline = saved.with_suffix(".mouth.csv")
                with timeline.open("w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, fieldnames=("audio_time", "rms", "threshold", "target", "openness", "commanded",
                                                             "scheduled_at", "command_started_at", "command_completed_at",
                                                             "command_lateness_ms", "command_duration_ms", "hardware_started_at",
                                                             "hardware_completed_at", "hardware_lateness_ms", "hardware_duration_ms",
                                                             "coalesced_frames"))
                    writer.writeheader(); writer.writerows(lip.motion_records)
                print(f"  preserved mouth analysis: {timeline}")
            if mouth is not None and getattr(mouth, "command_log", None):
                commands = saved.with_suffix(".jaw-commands.csv")
                with commands.open("w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, fieldnames=("timestamp", "command_started_at", "command_completed_at",
                                                             "command_duration_ms", "upper_write_ms", "lower_write_ms", "openness",
                                                             "upper_jaw_angle", "lower_jaw_angle"))
                    writer.writeheader(); writer.writerows(mouth.command_log)
                print(f"  preserved commanded jaw log (not position feedback): {commands}")
        _print_diagnostics(text, [text], player.diagnostics, mode="streaming", audio_path=saved,
            alignment="incremental 20 ms audio energy (lower accuracy)" if lip else "disabled (voice-only)", lip=lip)
    except KeyboardInterrupt:
        player.stop()
        if lip: lip.stop(close=True)
        raise
    except BaseException:
        if lip: lip.stop(close=True)
        _print_diagnostics(text, [text], player.diagnostics, mode="streaming-failed",
            alignment="incremental energy; stream did not complete", dropped_segments=1, lip=lip)
        raise
    finally:
        if lip and not lip_finished and not lip.cancelled.is_set(): lip.stop(close=True)


speak = speak_reliable


def calibrate_mouth_offset(service, mouth, configs, input_func=input) -> int:
    """Supervised physical timing comparison; it never changes calibration or speed."""
    print("Mouth-offset calibration changes no servo endpoint, speed, or config file.")
    print("Keep clear of the mechanism and stop if either jaw strains, chatters, or binds.")
    phrase = "Please watch the final words of this sentence."
    for offset in (0, 100, 200, 300):
        command = input_func(f"Type TEST {offset} to speak once at a {offset} ms mouth-only delay, or STOP: ").strip()
        if command == "STOP":
            break
        if command != f"TEST {offset}":
            print("Skipped; no servo or audio command was sent.")
            continue
        speak_streaming(phrase, service, mouth=mouth, configs=configs, mouth_offset_ms=offset)
        print("Observe the final words. Choose the smallest offset that prevents an early mouth stop.")
    print("Set only the selected value in .env.local as HEAD_MOUTH_OFFSET_MS; no setting was saved.")
    return 0


def run_session(service, *, voice_only, mode="reliable", mouth=None, configs=None,
                input_func=input, debug_audio=False):
    print(f"HEAD ROBOT - AI CHAT ({mode} mode)\n")
    while True:
        try: user = input_func("You: ").strip()
        except EOFError: return 0
        if user.lower() in {"exit", "quit"}: return 0
        if not user: continue
        trace = LatencyTrace(); print("Thinking...")
        try:
            answer = generate_complete_text(user, service, trace)
            helper = speak_streaming if mode == "streaming" else speak_reliable
            helper(answer, service, mouth=None if voice_only else mouth, configs=configs,
                   trace=trace, debug_audio=debug_audio)
            trace.finish(float(__import__("os").getenv("HEAD_AUDIO_LATENCY_MS", "0")))
            print("\n" + trace.format() + "\n")
        except GeminiError as exc: print(f"Gemini error: {exc}")
        except RuntimeError as exc: print(f"Audio error: {exc}")


def run_listen_session(service, *, mode, mouth=None, configs=None, debug_audio=False,
                       microphone_factory=Microphone, wake_factory=PorcupineWakeWord) -> int:
    """Hands-free turns. Capture remains open; it is idle while HEAD speaks."""
    microphone = microphone_factory()
    microphone.silence_s = ziggy_silence_seconds()
    wake = wake_factory.from_environment()
    if wake.sample_rate != microphone.rate:
        wake.close()
        raise WakeWordConfigurationError(
            f"Ziggy wake model requires {wake.sample_rate} Hz but microphone capture is {microphone.rate} Hz")
    pre_roll = deque(maxlen=max(1, round(.3 * wake.sample_rate / wake.frame_length)))
    state = ConversationState.WAITING_FOR_WAKE_WORD
    print(f"Ziggy: Listening for wake word (mic: {microphone.device})")
    print("Mouth: Ready" if mouth is not None else "Mouth: Disabled — reference required")
    try:
        while True:
            if state is ConversationState.WAITING_FOR_WAKE_WORD:
                print("Listening for wake word...")
                while True:
                    pcm = microphone.read_pcm(wake.frame_length * 2)
                    pre_roll.append(pcm)
                    if wake.detected(pcm):
                        print("Ziggy: Wake word detected.")
                        state = ConversationState.RECORDING_QUESTION
                        break
            # Include the detector's 300 ms rolling audio history. This keeps
            # a question that immediately follows “Ziggy” intact.
            utterance = microphone.record_after_wake(b"".join(pre_roll))
            pre_roll.clear()
            print(f"Recording completed ({utterance.duration_s:.2f} s). Transcribing...")
            state = ConversationState.TRANSCRIBING
            trace = LatencyTrace(); trace.mark("stt_request")
            with tempfile.TemporaryDirectory(prefix="head-mic-") as directory:
                wav = microphone.write_wav(utterance, Path(directory) / "utterance.wav")
                text = service.transcribe_wav(wav)
            trace.mark("stt_complete"); trace.span("STT transcription", "stt_request", "stt_complete")
            if not text:
                print("No usable speech recognized; listening again.")
                state = ConversationState.WAITING_FOR_WAKE_WORD
                continue
            print(f"You said: {text}")
            state = ConversationState.RESPONDING
            answer = generate_complete_text(text, service, trace)
            helper = speak_streaming if mode == "streaming" else speak_reliable
            helper(answer, service, mouth=mouth, configs=configs, trace=trace, debug_audio=debug_audio)
            trace.finish(float(__import__("os").getenv("HEAD_AUDIO_LATENCY_MS", "0")))
            trace.span("End of speech to first audio", "stt_complete", "first_audio_submitted")
            print("\n" + trace.format())
            if trace.values_ms:
                slowest = max(trace.values_ms.items(), key=lambda item: item[1])
                print(f"Slowest measured stage: {slowest[0]} ({slowest[1]:.0f} ms)\n")
            state = ConversationState.WAITING_FOR_WAKE_WORD
    finally:
        microphone.close()
        wake.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gemini + Despina HEAD robot terminal chat")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--reliable", action="store_true", help="complete audio + alignment (default)")
    modes.add_argument("--streaming", action="store_true", help="lower-latency incremental energy mode")
    parser.add_argument("--voice-only", action="store_true")
    parser.add_argument("--mouth-test", action="store_true")
    parser.add_argument("--mouth-offset-calibration", action="store_true",
                        help="supervised 0/100/200/300 ms jaw-only timing comparison")
    parser.add_argument("--voice-test", action="store_true")
    parser.add_argument("--listen", action="store_true", help="persistent microphone VAD -> Gemini STT -> reply loop")
    parser.add_argument("--initialize-jaws", action="store_true",
                        help="supervised, session-only powered-reference confirmation for P8/P10")
    parser.add_argument("--list-mics", action="store_true", help="list ALSA capture devices and exit")
    parser.add_argument("--debug-audio", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args(argv); mode = "streaming" if args.streaming else "reliable"
    try:
        if args.list_mics:
            print(Microphone.devices()); return 0
        service = GeminiService(GeminiConfig.from_environment())
        # Do not delay wake-word capture with network model probing. Requests still
        # fail visibly if a configured model is unavailable.
        if not args.skip_model_check and not args.listen:
            found = service.verify_models(); print("Verified models:", ", ".join(f"{k}={v}" for k, v in found.items()))
        if args.voice_test:
            speak_reliable("Hello. This is a Despina Bluetooth speaker test.", service, debug_audio=args.debug_audio); return 0
        if args.voice_only: return run_session(service, voice_only=True, mode=mode, debug_audio=args.debug_audio)
        configs = load_config(Path(__file__).resolve().parents[1] / "config" / "servos.json")
        mouth = None
        if args.initialize_jaws:
            controller = ServoController(configs, RobotHatBackend())
            print("Jaw initialization is session-only: it sends no command until speech begins.")
            if not establish_mouth(configs, controller): return 1
            mouth = MouthController(controller)
            print("Mouth: Ready for this continuously powered session.")
        elif not args.listen:
            # Interactive and test modes retain their explicit supervised setup.
            controller = ServoController(configs, RobotHatBackend())
            if not establish_mouth(configs, controller): return 1
            mouth = MouthController(controller)
        if args.mouth_offset_calibration:
            if mouth is None: raise ConfigurationError("--mouth-offset-calibration requires --initialize-jaws")
            return calibrate_mouth_offset(service, mouth, configs)
        if args.mouth_test:
            if mouth is None: raise ConfigurationError("--mouth-test requires --initialize-jaws")
            (speak_streaming if mode == "streaming" else speak_reliable)(
                "Hello, this is a short mouth synchronization test.", service,
                mouth=mouth, configs=configs, debug_audio=args.debug_audio); return 0
        if args.listen:
            return run_listen_session(service, mode=mode, mouth=mouth, configs=configs, debug_audio=args.debug_audio)
        return run_session(service, voice_only=False, mode=mode, mouth=mouth,
                           configs=configs, debug_audio=args.debug_audio)
    except KeyboardInterrupt:
        print("\nSTOPPED: no future mouth commands will be sent. Physically disconnect servo power if unsafe."); return 130
    except (GeminiError, ConfigurationError, WakeWordConfigurationError, RuntimeError) as exc:
        print(f"Setup error: {exc}"); return 2


if __name__ == "__main__": raise SystemExit(main())
