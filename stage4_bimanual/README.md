# stage4_bimanual — Owner: Azeem

Executes planned `Action` sequences with dual SO-101 robotic arms in MuJoCo physics
simulation: drawer opening, plate transfer, mug hold, bottle body grasp and a
coordinated bimanual pour.

- **Input**: `list[Action]` (+ `MuJoCoSim` handle from `reset_scene`).
- **Output**: `ExecutionResult` with per-action results, a `ContactAudit` verdict and the final `SceneState`.
- **Contract** (`CONTRACTS.md`): `execute(actions, sim)`, `reset_scene(seed)`, `get_camera_frame(sim)`.
  `reset_scene` also accepts an optional `trajectory_jitter=` keyword (see below).

## Standalone test

```bash
python -c "from common.types import Action, ActionType; from stage4_bimanual import execute, reset_scene; sim = reset_scene(0); print(execute([Action(step_id=1, action=ActionType.OPEN_DRAWER, arm='A', object='top_drawer')], sim))"
VOICE_STUB=1 python scripts/evaluate.py --seeds 10      # whole pipeline, offline parser
python -m pytest -q stage4_bimanual/tests               # IK, jitter, collision groups, one full pour
```

## Primitives (`primitives.py`)

| Action | Primitive | Arm | What it does |
|---|---|---|---|
| `open_drawer` | `OpenDrawerPrimitive` | A | descends onto the D-handle, pinches, pulls 7.5 cm, loosens the jaws and lifts off with the gripper pitch held |
| `pick plate` | `PickPlatePrimitive` | A | pinches the plate rim (jaws close until both pads touch), lifts |
| `place plate` | `PlacePlatePrimitive` | A | transits, levels the plate by pointing the fingers down, aims the plate centre at (0.06, 0.00) from the live pinch offset, lowers it onto the table, opens and lifts straight up |
| `pick mug` | `PickMugPrimitive` | B | grasps the handle with the fingers pitched 65° down, lifts and holds the mug upright at the pour station |
| `pick water_bottle` | `PickBottlePrimitive` | A | side grasp of the bottle **body** between the fingertips, jaws close until both touch, lifts 9 cm |
| `pour` | `PourWaterPrimitive` | A + B | see below; then returns the bottle, sets the mug down, parks both arms |

Anything else returns `False` and stops the sequence: an executor that cannot do
an action must never report success (CONTRACT_PROPOSAL P2).

### Pour kinematics

All targets are live measurements of the simulator, never assumed poses.

1. Arm A carries the upright bottle to a point beside the mug (13 cm from the mug axis,
   on the far side of arm B's reach so the mug can tip toward it).
2. The pour is a **wrist roll** about the fingers' axis. Every substep the primitive
   measures the bottle mouth, predicts where the next roll increment moves it, and
   re-solves the arm so the mouth follows a path from "beside the mug" to
   `POUR_MOUTH_CLEARANCE` (3.5 cm) above the rim, 1.2 cm toward the bottle. The tilt
   reaches `POUR_TILT_DEG` (105°; ~99° measured because the roll axis is pitched 20°).
3. In the same substeps arm B tips the mug `MUG_TILT_DEG` (12°) toward the bottle while
   keeping the rim centre where it was.
4. Timing: 1.5 s tilt, 1.0 s hold, 1.2 s untilt, minimum-jerk profile; every segment is
   rate-limited to 2 rad/s (3 rad/s gripper) by `BaseManipulationPrimitive.move`.
5. Success requires the mouth over the mug at full tilt (< 3 cm from the axis, 0–9 cm above
   the rim, tilt > 85°) and both vessels standing upright on the table afterwards.

Measured on seeds 100–109 (nominal script): mouth 1.2 cm from the mug axis, 3.4 cm above the
rim, bottle 99°, mug 12.5°, both vessels < 0.1° from upright at the end, mug set down within
1 mm of its target, 630 frames at 25 Hz.

### Kinematics (`kinematics.py`)

`DLSInverseKinematics.solve(arm, target, wrist_roll, *, pitch=None, site=None, q_init=None)`:
damped-least-squares position IK, optionally with the **gripper pitch** (elevation of the
fingers' pointing axis) as a fourth constraint. The SO-101 can only choose position, pitch
and roll, so this is the whole controllable orientation. Holding the pitch is what keeps a
welded object level while the arm moves.

### Grasp model

A grasp is a MuJoCo weld attached **only after real contact** (`attach_weld`, optionally
requiring both jaws), released explicitly. Jaws close incrementally until contact
(`close_until_contact`) instead of to a fixed angle: a blind close drives the pads several
millimetres into the object with the full 35 N·m servo torque, bends the weld and rotates
the object. The mug and bottle welds use `solref="0.006 1"`.

Collision approximation: the finger-base pads (group 1) touch everything; the new
fingertip pads (`*_tip_pad`, group 2) touch only the bottle, whose body needs a 4.8 cm
opening the base pads cannot provide. Fingertips still pass through the table, plate,
drawer and mug exactly as in the original model, so those grasps are unchanged.

## Domain randomization (`sim.py`, `configs/default.yaml -> randomization`)

| Key | Effect |
|---|---|
| `object_placement_cm` | default ±xy jitter for mug, water_bottle, spoon, fork |
| `placement_cm_by_object` | per-object override; `drawer_unit` moves the cabinet and the plate inside it |
| `object_yaw_deg` | yaw about +z (mug handle direction) |
| `lighting_intensity`, `friction` | as before |
| `mass_scale` | tableware only (robot links stay nominal) |
| `trajectory_jitter` | scale of the scripted expert's seed-deterministic waypoint/timing variation (`primitives.TrajectoryJitter`); 0 for evaluation, 1.0 for recording |

`scripts/record_skill_demos.py` records 25 Hz raw bundles (`--task pick_bottle` and
`--task pour_water` included), refuses evaluation seeds 0–9, defaults to `--jitter 1.0`,
and writes the primitive's metrics into `manifest.json`.

## Validation (2026-09-15)

- `VOICE_STUB=1 python scripts/evaluate.py --seeds 10`: **10/10** with the wider randomization.
- Per skill on training seeds 100–109 with jitter 1.0: open_drawer 10/10, pick_plate 10/10,
  place_plate 10/10, pick_mug 10/10, pick_bottle 10/10, pour_water 10/10, contact audit clean.
- Between-episode joint-trajectory variation of the pour (mean per-joint std across
  time-normalized episodes): 0.012 rad with ±2 cm placement and no jitter, 0.071 rad with
  the shipped ranges + jitter; the ALOHA scripted transfer-cube reference ACT trains on is 0.032 rad.

## Known limitations

- No fluid: the pour is a kinematic proxy; stage6 checks poses only.
- Fingertips have no collision with the table/plate/drawer/mug (original approximation).
- The drawer, plate and mug primitives still ignore `Action.target_pose`; they read poses from the simulator.
- `scripts/collect_lerobot_data.py` references an undefined `save_video` (pre-existing).
