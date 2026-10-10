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
