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
