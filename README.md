# HEAD - Phase 1 servo control

Importing or starting this safety-first application never commands a servo.

## Exact Raspberry Pi commands

## Hands-free voice conversation

Install/update in one command (this preserves `.env.local` and all calibration files):

```bash
sudo apt-get update && sudo apt-get install -y alsa-utils && cd ~/HEAD && git pull origin main && python3 -m pip install -r requirements.txt
```

List USB/ALSA capture devices with `python3 -m src.robot_chat --list-mics`; set the desired ALSA device in `.env.local` as `HEAD_MIC_DEVICE=default` or an explicit `plughw:CARD=...,DEV=...`. Launch hands-free streaming conversation with:

```bash
python3 -m src.robot_chat --listen --streaming
```

Ziggy prints `Listening for wake word` and only sends a transcript to Gemini when it contains `Ziggy`, such as “Ziggy, what time is it?”. It starts voice-only if the jaws have not been safely referenced; this is intentional and never commands P8/P10. The capture process remains open between turns, uses 16 kHz mono S16_LE and lightweight energy VAD, retains 300 ms of pre-roll, stops after 1,000 ms silence by default, and limits utterances to 15 seconds.

Wake detection is local and does not upload background speech. It uses the free Vosk small English model, cached under `models/` after one setup download. There is no Picovoice account, key, `.ppn` model, or subscription. Before the first launch, run `python3 -m src.vosk_setup`; ordinary Ziggy startup never downloads a model. Vosk uses a constrained grammar for `ziggy`/`ziggie`; unlike a dedicated wake engine, it has no equivalent confidence slider.

To enable mouth motion for one continuously powered run, establish trusted P8/P10 references in that same process. Initialization sends no jaw command; enter the actual already-commanded angles and `POWERED` only if power remained continuous and neither linkage was moved:

```bash
python3 -m src.robot_chat --listen --streaming --initialize-jaws
```

That reference deliberately disappears after exit, reboot, power interruption, or physical repositioning. Use the normal command for voice-only operation until it is re-established. Saved endpoints and the selected 15°/s operating speed are loaded automatically, but are never treated as position feedback.

## P0 head-roll (optional)

P0 (`head_roll`) is separate from the jaws and never commands P2, P6, P8, or P10. With the power disconnect accessible, establish an initial narrow P0-only range through the supervised one-degree procedure:

```bash
python3 -m src.head_roll_calibration quick
```

It saves 68°–72° only after every displayed movement has been accepted as safe. Before normal use, run one supervised P0-only cycle:

```bash
python3 -m src.head_roll_calibration test
```

To extend one existing P0 endpoint beyond the saved range, begin with P0 already at that verified endpoint and run:

```bash
python3 -m src.head_roll_calibration extend
```

It permits only `TEST` of the next adjacent one-degree angle and saves each endpoint only after `SAFE`; `STOP` preserves all previously verified limits.

For the visible 60°–80° P0 range, use the all-or-nothing supervised setup. It begins from a freshly confirmed saved 70° neutral even if P0 travel limits are incomplete, tests every move in one-degree `TEST`/`SAFE` steps, and saves 60°–80° only once both endpoints are accepted:

```bash
python3 -m src.head_roll_calibration wide
```

Then start Ziggy with optional gentle two-second P0 side-to-side motion. It asks once for a fresh P0 powered reference for this process and never restores that reference after restart:

```bash
python3 -m src.robot_chat --listen --streaming --head-roll
```

The capture process remains open between turns, uses 16 kHz mono S16_LE and lightweight energy VAD, retains 300 ms of pre-roll, stops after 550 ms silence (configurable 450–600 ms), and limits utterances to 15 seconds. It does not read microphone frames while HEAD is speaking. Gemini verifies the configured transcription model at startup and transcribes each short WAV with `gemini-3.5-transcribe` by default.

The Ziggy default above (1,000 ms silence) supersedes the legacy 550 ms timing note.

From the repository directory, run the first no-movement command:

```bash
cd ~/HEAD
python3 -m src.calibration prepare neck_pan
```

`prepare` only prints instructions. Standard positional servos cannot report their current shaft angle through the Robot HAT; software cannot discover or preserve the head's assembled physical position. Never guess an angle while a linkage is attached.

For each servo (`neck_pan=P6`, `neck_tilt=P2`, `head_roll=P0`, `upper_jaw=P8`, `lower_jaw=P10`), do this one at a time:

```bash
# With servo power off: support head, disconnect only this linkage/horn.
python3 -m src.calibration center neck_pan
# Type CENTER only after the linkage is safely disengaged.

# Power off again; put head in desired neutral pose; reattach without forcing.
# Restore power, then re-establish the known centered baseline and calibrate.
python3 -m src.calibration calibrate neck_pan
```

`center` never runs automatically and commands 0 degrees only after the immediate `CENTER` confirmation. `calibrate` requires you to type `CENTERED`, then requires `CENTER` immediately before it recommands that known baseline. Afterward only `+1`, `+2`, `-1`, and `-2` moves are permitted; set `min`, `max`, and `neutral`, then `save`.

## Software alignment from a trusted powered reference

If a servo was already commanded to a known angle and has remained powered and physically untouched since that command, `align` can find the desired forward-facing neutral without moving the horn. For example, if `neck_pan` was commanded to 0 degrees and the head is currently facing right:

```bash
python3 -m src.calibration align neck_pan
```

Enter `0` as the previously commanded angle, then type `POWERED` only if the servo has remained powered and has not been repositioned. This initial reference is recorded without sending a servo command. Then use only `+` and `-` for explicit one-degree moves; no sweep occurs. The default temporary software window is plus/minus 15 degrees from the reference. Set a smaller or larger initial conservative window (maximum 20 degrees) with:

```bash
python3 -m src.calibration align neck_pan --provisional-limit 10
```

At a provisional boundary, the tool stops and identifies the limit. If more travel is necessary, inspect the full mechanism first, then enter `extend +` or `extend -` for the required direction. It permits only one additional 5-degree section per `CLEAR` confirmation, sends no movement while extending the window, and never goes beyond the Robot HAT's -90 to +90 degree range. Every extra section requires a fresh clearance check and `CLEAR` confirmation.

For `neck_pan` on P6, the positive alignment boundary remains +15 degrees. Its negative provisional boundary can be extended from -15 degrees toward -80 degrees only with repeated `extend -` and fresh `CLEAR` confirmations. The -80-degree value is a software stop, **not** a verified mechanically safe position or calibrated movement limit. Stop immediately if the head binds, strains, chatters, or reaches its physical limit; do not continue merely because software allows another step.

When the head faces forward, type `SAVE`. This saves `neutral` only and deliberately clears movement limits: neutral alignment does **not** calibrate left/right travel. Later use the cautious `calibrate neck_pan` workflow to establish limits. If power was removed, the servo moved, or the horn/linkage was repositioned, do not use `align`; establish a safe reference with `prepare` and `center` instead. In particular, after the program exits or restarts, it does not assume the last servo position: explicitly verify the still-powered, untouched reference and enter it again before any movement.

Keep fingers, wiring, cardboard, brackets, and tools clear. Stop if a servo strains, chatters, binds, or shifts the mechanism. Ctrl+C stops future software commands but cannot guarantee torque removal; physically disconnect servo power in an emergency.

Run checks and mock-mode development with:

```bash
robot_hat version
robot_hat doctor
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
python -m src.calibration --mock prepare neck_pan
```

Use `python3 -m src.calibration test neck_pan --speed 15` only after the same mechanical-centering confirmation. Settings are saved atomically to `config/servos.json`.

## Phase 1B: mouth movement calibration

The saved closed-mouth neutrals are `upper_jaw` P8 at +1 degrees and `lower_jaw` P10 at +15 degrees. These values are not changed by mouth calibration. Calibrate one jaw at a time; the tool never moves a servo on startup and it does not command neutral after a restart:

```bash
python3 -m src.mouth_calibration upper
python3 -m src.mouth_calibration lower
```

For each jaw, enter its known current commanded angle and type `POWERED` only if power has remained continuously applied and the linkage was not repositioned. Then use `+` and `-` for explicit 1-degree movements to observe which direction opens it. The initial window is 5 degrees either side of closed neutral. At a boundary, inspect clearance and use `extend +` or `extend -`; each additional 5 degrees requires a fresh `CLEAR` confirmation and does not move the servo by itself. Type `STOP` for resistance, binding, or strain; disconnect servo power physically if unsafe.

At a verified safe opening endpoint, type `SAVE`. It saves `opening_endpoint` and `opening_direction` while preserving the saved closed neutral. It does not calibrate general servo movement limits.

After both jaws have endpoints, inspect the displayed plan and run exactly one slow, explicitly confirmed cycle:

```bash
python3 -m src.mouth_calibration test --speed 8
```

Enter the known current angle and `POWERED` confirmation for each jaw. The command shows both closed/open angles, requires `CYCLE`, smoothly opens both jaws once, and commands both back to their saved closed positions. Ctrl+C stops future commands; physically disconnect servo power in an emergency. The mouth controller is reusable through `MouthController.open(amount)`, `close()`, and `set_openness(amount)` for a future speech layer, but no speech or Gemini integration is included.

## Phase 1C: talking-mouth animation

After the jaw endpoints are calibrated, run a single manually confirmed, speech-like visual sequence:

```bash
python3 -m src.talking_mouth test --speed 8
```

The command reads the saved jaw endpoints from `config/servos.json`; it does not hardcode mouth angles or alter calibration. It displays the timed partial-opening plan, asks for each jaw's known current commanded angle and `POWERED` confirmation, then requires `TALK`. The sequence uses varying openness amounts and pauses, runs once, and ends closed. Speed must be positive and no higher than the configured software cap (or an already verified practical speed); 8 is the conservative default.

No servo moves on startup or import. Ctrl+C cancels future movement. On interruption the program does **not** blindly command a close because jaw positions may be uncertain. If binding, strain, or resistance occurs, physically disconnect servo power immediately. The timing engine (`MouthAnimationEngine.animate(events)`) accepts future external `{ "time", "openness" }` events, while `MouthController` remains the hardware-bounded jaw layer. No audio, speech synthesis, or Gemini integration is included.

## Adaptive lip synchronization (Phase 2 preparation)

`AdaptiveLipSynchronizer` is the playback-clock-driven layer intended for future Despina audio. Start it from the same monotonic reference used for real audio playback and pass external `{ "time", "openness" }` events. It uses the latest due target, skips stale targets rather than replaying them late, reduces rapid opening changes to achievable partial movement, and logs target-command timing drift at Python debug level. Long pause events with `openness: 0.0` close the mouth during speech pauses.

Motion settings are in `config/mouth_motion.json`. Until hardware calibration, speech retains the conservative 8 degrees/second fallback. The 60 degrees/second **software** cap is only the upper boundary of the supervised test and is not a claim about the servos' physical maximum. The synchronizer uses a 20 ms playback-deadline loop and a 240 degrees/second-squared acceleration limit; servo write time is included in that deadline instead of being added as extra delay. It keeps both jaws synchronized through their individual calibrated endpoints and never controls head/neck ports.

Run the dedicated supervised calibration command. It starts at 8 degrees/second and offers 12, 16, 20, 25, 30, 40, 50, and 60 in order. Every cycle requires an exact `RUN <speed>` confirmation and is saved only after `PASS`. Stop at the first sign of strain, chatter, binding, heat, impact, or unreliable motion:

```bash
python3 -m src.mouth_calibration speed
```

Once a stage passes, its value is atomically saved as the verified maximum and normal Gemini/Despina lip sync uses it automatically unless a lower operating speed is selected. To safely select a lower operating speed without implying the previous maximum was unsafe, run a supervised one-cycle test that requires the same powered references and `PASS` before it saves:

```bash
python3 -m src.mouth_calibration speed --target-speed 15
```

This stores `selected_operating_speed_dps: 15` while retaining the verified maximum (for example, 25) in `config/mouth_motion.json`. A target above the verified maximum is rejected before any motion. Do not accept a stage from a mock test. Stop and physically disconnect servo power if either jaw strains, binds, or chatters.

## Phase 2: Gemini + Despina terminal chat

On the Pi, install the cloud SDK and ALSA player without changing the Robot HAT installation:

```bash
cd ~/HEAD
python3 -m pip install --user -r requirements.txt
sudo apt-get install -y pulseaudio-utils alsa-utils
cp .env.example .env.local
nano .env.local
```

Put your Gemini API key in `.env.local`. It is ignored by Git. First verify the speaker and Gemini/Despina pipeline without servo motion:

```bash
robot_hat doctor
python3 -m src.robot_chat --voice-only
```

Type a short message, then `quit`. Full mouth mode (`python3 -m src.robot_chat`) requires confirming that both jaws are continuously powered, unrepositioned, and currently at their saved closed neutral positions. `--mouth-test` speaks one test phrase after the same confirmations. Reliable mode obtains and validates word timing before playback; if unavailable, the terminal reports its actual-audio energy fallback. Playback and the adaptive mouth synchronizer then use one shared clock—not a fixed animation. Ctrl+C stops future motion and audio; do not assume that it removes torque, so physically disconnect servo power if binding occurs.

### Bluetooth / desktop audio output

Audio playback runs as the current desktop user, never as root. With `HEAD_AUDIO_BACKEND=auto` (the default), HEAD prefers `paplay`, which sends audio to the current PipeWire/PulseAudio default sink—the Bluetooth speaker selected in Raspberry Pi Desktop. Choose explicitly with `HEAD_AUDIO_BACKEND=pulse` or `HEAD_AUDIO_BACKEND=alsa`; ALSA (`aplay`) is the fallback for direct devices.

After connecting and selecting the Bluetooth speaker in the desktop audio menu, test Despina without moving servos:

```bash
cd ~/HEAD
source .venv/bin/activate
HEAD_AUDIO_BACKEND=pulse python3 -m src.robot_chat --voice-test
```

`paplay` does not expose the Bluetooth speaker's actual acoustic latency. Mouth synchronization uses the audio-process playback clock and can apply an explicit estimate with `HEAD_AUDIO_LATENCY_MS` (for example, `HEAD_AUDIO_LATENCY_MS=180`). A positive value delays mouth motion to match sound that emerges later from the speaker. If the final words still outlast mouth motion, add `HEAD_MOUTH_OFFSET_MS` (initially `0`) as a separate, mouth-only delay. It is added once after the audio-latency estimate; do not add it to `HEAD_AUDIO_LATENCY_MS` as well. The terminal prints an estimated final audible timestamp and the final speech-bearing mouth command; Bluetooth acoustic timing remains an estimate.

To compare safe timing offsets without changing servo speed, endpoints, or calibration files, use the supervised command below. It speaks one short phrase only after each exact confirmation at 0, 100, 200, and 300 ms; choose the smallest offset that keeps movement present for the final words, then manually set that value in `.env.local`.

```bash
python3 -m src.robot_chat --streaming --mouth-offset-calibration
```

Streaming lip-sync uses 20 ms PCM-energy frames, then rate-limits generated openness to the verified (or conservative) jaw speed. Very small per-frame target changes can be physically ineffective even though the Robot HAT receives a command. To avoid repeatedly sending such stationary commands, HEAD coalesces changes smaller than `HEAD_MOUTH_COMMAND_DEADBAND_DEG` (default `0.5`; range `0.1..2`). This does not increase speed or change endpoints. You may tune `HEAD_MOUTH_SENSITIVITY` (`0.25..4`), `HEAD_MOUTH_MAX_OPENNESS` (`0.05..1`), and `HEAD_MOUTH_SMOOTHING` (`0..0.95`) in `.env.local`; start with the defaults and change one value at a time. Sensitivity can help quiet final words; it never opens the mouth for PCM detected as genuine silence.

Run a response with `--debug-audio` to preserve its WAV, energy/openess timeline CSV, and commanded-jaw-angle CSV. Jaw angles are commands only, not shaft-position feedback. The offline tool reads a preserved WAV and produces a CSV plus dependency-free SVG graph without initializing any hardware:

```bash
python3 -m src.mouth_diagnostics debug_audio/head-YYYYMMDD-HHMMSS.wav
```

The streaming CSV also records each target's playback deadline, command start/completion timestamps, software-call duration, lateness, and count of expired frames coalesced into the newest target. Compare an existing capture without hardware access:

```bash
python3 -m src.mouth_diagnostics debug_audio/head-YYYYMMDD-HHMMSS.wav \
  --recorded-mouth-csv debug_audio/head-YYYYMMDD-HHMMSS.mouth.csv \
  --jaw-csv debug_audio/head-YYYYMMDD-HHMMSS.jaw-commands.csv
```

That command also writes `*.final-speech.csv` for the final two seconds: PCM RMS/threshold, generated target and openness, scheduling/hardware timestamps, and nearest commanded P8/P10 angles. The angles are issued commands, never shaft-position feedback. Use its `energy_frames`, `nonzero_targets`, and `hardware_commands` counts to distinguish quiet-word detection from command dispatch. A nonzero command record does not prove Bluetooth acoustic timing; `paplay` does not expose physical speaker playback progress.

If hardware writes are slow, HEAD analyzes all expired PCM frames but emits only the newest valid rate-limited jaw target. It never replays a stale servo-command backlog after the related audio has passed.

To isolate Robot HAT application-call timing from Gemini, TTS, PulseAudio, and Bluetooth, run the supervised jaw-only benchmark. It requires the same known-powered references and exact confirmation, alternates only 0%/10% openness inside saved endpoints at the existing verified/conservative speed, leaves both jaws closed, and does not write configuration files. It reports median, p95, p99, and maximum total/P8/P10 software-call durations; those are not shaft-position measurements.

```bash
python3 -m src.mouth_calibration latency --samples 20
```

## Reliable and streaming conversation modes

Reliable mode is the temporary default. It fully consumes Gemini text, submits the complete response to Despina once, validates word timings against the generated WAV duration, precomputes speed-reachable syllable pulses, and starts WAV playback plus `AdaptiveLipSynchronizer` from one latency-adjusted clock. If alignment is unavailable or invalid, it uses 40 ms energy windows from the actual audio and labels that fallback as approximate.

```bash
python3 -m src.robot_chat --reliable
```

Streaming mode also fully consumes Gemini text and makes one ordered TTS request for the complete response. It streams headerless 24 kHz mono signed-16-bit PCM into `paplay` or `aplay`, normalizes variable network chunks into 20 ms analysis frames, and appends a short silent tail so the mouth closes during playback. This energy mode has lower mouth accuracy than reliable word alignment.

```bash
python3 -m src.robot_chat --streaming
```

Add `--voice-only` to either mode to avoid servo initialization. Add `--debug-audio` to retain generated audio under `debug_audio/` and print the full response, exact TTS segments, sample/chunk counts, playback completion, producer failures, and dropped analysis frames.

Every response prints millisecond timings for Gemini's first token and completion, Despina's first audio chunk, PCM preparation, incremental alignment, player startup, first submitted audio, and estimated first audible sound. The audible value adds `HEAD_AUDIO_LATENCY_MS`; it is an estimate because PulseAudio/PipeWire does not expose the Bluetooth acoustic delay.

Recommended Raspberry Pi deployment:

```bash
cd ~/HEAD
git pull --ff-only origin main
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
sudo apt-get install -y pulseaudio-utils alsa-utils
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

Keep the existing `.env.local`, `config/servos.json`, and verified value in `config/mouth_motion.json`. For a low-latency configuration, set these in `.env.local` only if the startup model check confirms that your API key can access them:

```dotenv
GEMINI_CHAT_MODEL=gemini-3.8-flash
GEMINI_TTS_MODEL=gemini-3.8-flash-tts
GEMINI_TTS_VOICE=Despina
GEMINI_THINKING=low
GEMINI_MAX_HISTORY_TURNS=4
GEMINI_MAX_OUTPUT_TOKENS=1024
GEMINI_RETRY_MAX_OUTPUT_TOKENS=2048
HEAD_AUDIO_BACKEND=pulse
HEAD_AUDIO_LATENCY_MS=180
```

For `gemini-3.8-flash`, `low` is the lowest supported thinking level; do not set `minimal` or a numeric thinking budget for that model. A response that ends with `MAX_TOKENS` is never sent to Despina. HEAD consumes the first attempt completely, retries once at `GEMINI_RETRY_MAX_OUTPUT_TOKENS`, and only speaks the successful complete result. The terminal prints finish reason, available token counts, and the number of attempts.

Test reliable voice-only first, compare streaming, then test the servos:

```bash
python3 -m src.robot_chat --reliable --voice-only --debug-audio
python3 -m src.robot_chat --streaming --voice-only --debug-audio
python3 -m src.robot_chat --reliable --mouth-test
python3 -m src.robot_chat --reliable
```

To compare real API availability latency for identical prompts, run:

```bash
python3 -m src.benchmark_latency --output latency-benchmark.json
```

This benchmark makes billable real requests. It reports complete-response and streaming measurements without claiming Bluetooth acoustic timing.
