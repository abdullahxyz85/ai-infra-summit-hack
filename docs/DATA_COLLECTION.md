# Collecting training data: exact workflow

## Choose ACT first

Use **LeRobot ACT** for the first learned controllers. It is a small
imitation-learning model designed for precise bimanual manipulation and takes
multiple RGB views plus joint state. It is the right first experiment because
we can validate each atomic skill before tackling language-conditioned
multi-task control.

ACT does not make the robot understand arbitrary English by itself. For the
hackathon pipeline, keep the language parser/state machine: it maps an English
command to an atomic skill and calls the corresponding ACT policy. After the
atomic controllers work, combine the data and evaluate SmolVLA for an actual
language-conditioned learned policy.

## Data specification

Every accepted episode must contain:

- `overhead` and `front` RGB images at 25 FPS (one frame every 20 physics steps, as in `stage3_policy/learned/schema.py`);
- 12 joint positions: 6 for Arm A then 6 for Arm B, including each gripper;
- 12 matching position-target actions;
- task string, skill name, random seed, timestamps, contact-audit result;
- a visual replay GIF and every individual PNG frame.

The new `record_skill_demos.py` writes this exact raw bundle. A saved episode
is eligible only when `manifest.json` says `accepted_for_training: true`.
`--save-rejected` is for diagnosis only; never train on those episodes.

## Physics Validation Status

The MuJoCo scene, collision geometries, and bimanual primitives have passed validation with a **100% success rate (10/10 seeds passing)** on the evaluation harness:

```powershell
python scripts/run_pipeline.py --seed 0
python scripts/evaluate.py --seeds 10
```

Also view each motion in the interactive MuJoCo viewer:

```powershell
python scripts/visualize_run.py --seed 0
```

The GUI is the source of truth for what the robot does. The terminal audit is
the source of truth for whether that visible motion is physically acceptable.

## Record one task at a time

After the physics checks pass, run 50 initial examples per atomic skill. Seeds
0–9 are the evaluation seeds and the recorder refuses them; every skill gets its
own seed block so no two skills share a layout:

```powershell
python scripts/record_skill_demos.py --task open_drawer --episodes 50 --seed 100
python scripts/record_skill_demos.py --task pick_plate  --episodes 50 --seed 200
python scripts/record_skill_demos.py --task place_plate --episodes 50 --seed 300
python scripts/record_skill_demos.py --task pick_mug    --episodes 50 --seed 400
python scripts/record_skill_demos.py --task pick_bottle --episodes 50 --seed 500
python scripts/record_skill_demos.py --task pour_water  --episodes 50 --seed 600
```

`--jitter 1.0` (the default) adds seed-deterministic variation to the scripted
expert's standoffs, lift heights, tilt angle, hold time and timing on top of the
scene randomization in `configs/default.yaml`; `--jitter 0` replays the nominal
script. Each `manifest.json` carries the primitive's own success metrics
(e.g. mouth offset from the mug axis, vessel tilts after release).

## Trajectory diversity between episodes

Scripted experts give a learned policy almost no between-episode variation
unless the scene and the script vary. Measured on seeds 100–109 as the mean
per-joint standard deviation across time-normalized episodes (rad), against
the ALOHA scripted transfer-cube dataset that ACT is known to learn from
(50 episodes, 0.032 rad, 0.037 over its moving arm):

| Setting | pour_water | open_drawer | pick_mug | verdict |
|---|---|---|---|---|
| ±2 cm placement, no jitter (before 2026-09-15) | 0.019 (old pour) / 0.012 (new pour) | 0.001 | 0.015 | too narrow: ~1/3 of the ALOHA reference, drawer skill identical every episode |
| ±2 cm placement + jitter 1.0 | 0.040 | – | – | comparable |
| shipped ranges (±3/±4 cm, mug yaw ±30°, drawer ±2 cm) + jitter 1.0 | 0.071 | 0.021 | 0.043 | ~1.3–2× the reference, all skills 10/10 |
| stress (±6 cm, yaw ±45°, jitter 1.5) | 0.102 | – | – | 10/10 for the pour; not validated for the other skills |

Keep the shipped ranges for the first dataset. Widen only after a policy trained
on it is evaluated on the held-out seeds 0–9.

The recorder renders two camera views and writes, for example,
`data/raw_skill_demos/pick_mug/episode_00000_seed_0300/`.

To inspect the generated camera evidence immediately, open its
`front_replay.gif`. To replay the recorded actions in a live MuJoCo window:

```powershell
python scripts/replay_skill_demo.py data/raw_skill_demos/pick_mug/episode_00000_seed_0300
```

For an unsafe diagnostic recording only, add `--save-rejected`. Its manifest
will explicitly say rejected; the repository currently contains one such
example under `data/raw_skill_demos_test/` so you can inspect the format.

## What you need to do

You do **not** need physical SO-101 hardware for this challenge. You need to:

1. Visually inspect the MuJoCo sequence and help choose sensible object
   spacing/poses after each physics fix.
2. Approve only demonstrations whose GIF/replay looks like the intended human
   movement and whose manifest is accepted.
3. Keep the cameras fixed while learning each atomic skill. Add variation
   gradually: first placement, then lighting, then friction/mass/background.
4. Keep 10 held-out seeds per skill; never train on them. Use them for the
   final reported success rate.
5. Install the official `lerobot` package in the project virtual environment
   when network/package access is available. Then convert these accepted raw
   bundles with the installed `LeRobotDataset` API and train ACT. This avoids
   hard-coding an untested, version-specific exporter.

The linked tutorial follows the same useful pattern—keyboard teleoperation,
camera images, state/action recording, and replay—but we are using a safer
two-arm, contact-gated workflow and current official LeRobot APIs.
