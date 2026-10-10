# HEAD - Phase 1 servo control

Importing or starting this safety-first application never commands a servo.

## Exact Raspberry Pi commands

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

The command reads the saved jaw endpoints from `config/servos.json`; it does not hardcode mouth angles or alter calibration. It displays the timed partial-opening plan, asks for each jaw's known current commanded angle and `POWERED` confirmation, then requires `TALK`. The sequence uses varying openness amounts and pauses, runs once, and ends closed. Speed must be between 1 and 12 degrees/second; 8 is the conservative default.

No servo moves on startup or import. Ctrl+C cancels future movement. On interruption the program does **not** blindly command a close because jaw positions may be uncertain. If binding, strain, or resistance occurs, physically disconnect servo power immediately. The timing engine (`MouthAnimationEngine.animate(events)`) accepts future external `{ "time", "openness" }` events, while `MouthController` remains the hardware-bounded jaw layer. No audio, speech synthesis, or Gemini integration is included.

## Adaptive lip synchronization (Phase 2 preparation)

`AdaptiveLipSynchronizer` is the playback-clock-driven layer intended for future Despina audio. Start it from the same monotonic reference used for real audio playback and pass external `{ "time", "openness" }` events. It uses the latest due target, skips stale targets rather than replaying them late, reduces rapid opening changes to achievable partial movement, and logs target-command timing drift at Python debug level. Long pause events with `openness: 0.0` close the mouth during speech pauses.

Motion settings are in `config/mouth_motion.json`: the default is a conservative 8 degrees/second, with a 20 degrees/second **software** cap. Neither is a claim about the servos' physical maximum. Determine a practical maximum only through separate, supervised hardware testing; record it as `verified_practical_speed_dps` only after that test. The synchronizer rejects speeds above the cap or above a populated verified value. It keeps both jaws synchronized through their individual calibrated endpoints and never controls head/neck ports.

Use the already-confirmed, one-cycle mouth test for supervised speed testing, beginning at 8 and increasing only after a successful physical observation:

```bash
python3 -m src.mouth_calibration test --speed 8
```

Do not set `verified_practical_speed_dps` merely from a mock test. Stop and physically disconnect servo power if either jaw strains, binds, or chatters.

## Phase 2: Gemini + Despina terminal chat

On the Pi, install the cloud SDK and ALSA player without changing the Robot HAT installation:

```bash
cd ~/HEAD
python3 -m pip install --user -r requirements.txt
sudo apt-get install -y alsa-utils
cp .env.example .env.local
nano .env.local
```

Put your Gemini API key in `.env.local`. It is ignored by Git. First verify the speaker and Gemini/Despina pipeline without servo motion:

```bash
robot_hat doctor
python3 -m src.robot_chat --voice-only
```

Type a short message, then `quit`. Full mouth mode (`python3 -m src.robot_chat`) requires confirming that both jaws are continuously powered, unrepositioned, and currently at their saved closed neutral positions. `--mouth-test` speaks one test phrase after the same confirmations. The pipeline first tries word timing from the configured alignment model; if unavailable, the terminal reports its audio-energy fallback. Audio starts first, and the adaptive mouth synchronizer uses that playback clock—not a fixed animation. Ctrl+C stops future motion and audio; do not assume that it removes torque, so physically disconnect servo power if binding occurs.
