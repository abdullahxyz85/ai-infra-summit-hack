#!/usr/bin/env python3
"""Record visually inspectable MuJoCo demonstrations for one atomic skill.

This intentionally writes an open raw bundle (PNG frames + compressed NumPy
arrays + JSON manifest) rather than pretending to be a LeRobot dataset when
the LeRobot package is not installed.  Only episodes that pass the contact
audit AND the primitive's own postcondition checks are marked
``accepted_for_training``.  The bundle is designed for a small conversion step
using the installed LeRobotDataset API.

Frames are sampled at 25 Hz (stage3_policy/learned/schema.py: one frame every
20 physics steps). Seeds 0-9 are the evaluation seeds and are refused unless
--allow-eval-seeds is given. --jitter scales the scripted expert's
seed-deterministic waypoint/timing variation (see primitives.TrajectoryJitter);
1.0 is the recommended value for training data, 0 replays the nominal script.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image

from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.primitives import (
    OpenDrawerPrimitive,
    PickBottlePrimitive,
    PickMugPrimitive,
    PickPlatePrimitive,
    PlacePlatePrimitive,
    PourWaterPrimitive,
)
from stage4_bimanual.sim import MuJoCoSim
from stage4_bimanual.trajectory import TrajectoryExecutor

EVAL_SEEDS = set(range(10))  # configs/default.yaml seeds; never used for demonstrations
DEFAULT_FPS = 25             # schema.FPS: 1 / (0.002 s * 20 physics steps)

SKILLS = {
    "open_drawer": (OpenDrawerPrimitive, "Open the top drawer with arm A."),
    "pick_plate": (PickPlatePrimitive, "Pick up the plate from the open drawer with arm A."),
    "place_plate": (PlacePlatePrimitive, "Place the plate at the center of the table with arm A."),
    "pick_mug": (PickMugPrimitive, "Pick up and hold the mug with arm B."),
    "pick_bottle": (PickBottlePrimitive, "Pick up the water bottle by its body with arm A."),
    "pour_water": (PourWaterPrimitive, "Hold the mug with arm B and pour water from the bottle with arm A."),
}

PREREQUISITES = {
    "open_drawer": [],
    "pick_plate": [OpenDrawerPrimitive],
    "place_plate": [OpenDrawerPrimitive, PickPlatePrimitive],
    "pick_mug": [],
    "pick_bottle": [PickMugPrimitive],
    "pour_water": [PickMugPrimitive, PickBottlePrimitive],
}


class DemonstrationExecutor(TrajectoryExecutor):
    """Record actions and two synchronized rendered views during physics steps."""

    def __init__(self, model, data, *, fps: int, width: int, height: int, contact_audit):
        super().__init__(model, data, contact_audit=contact_audit)
        self.sample_interval = max(1, round(1 / (fps * model.opt.timestep)))
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.front_camera = mujoco.MjvCamera()
        self.front_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.front_camera.lookat[:] = [-0.04, 0.00, 0.74]
        self.front_camera.distance = 1.52
        self.front_camera.elevation = -58.0
        self.front_camera.azimuth = 145.0
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.overhead_frames: list[np.ndarray] = []
        self.front_frames: list[np.ndarray] = []
        self._substeps = 0

    def capture(self) -> None:
        self.states.append(np.asarray(self.data.qpos[36:48], dtype=np.float32).copy())
        self.actions.append(np.asarray(self.data.ctrl[:12], dtype=np.float32).copy())
        self.timestamps.append(float(self.data.time))
        self.renderer.update_scene(self.data, camera="overhead_cam")
        self.overhead_frames.append(self.renderer.render().copy())
        self.renderer.update_scene(self.data, camera=self.front_camera)
        self.front_frames.append(self.renderer.render().copy())

    def interpolate(self, target_ctrl, steps: int = 60) -> None:
        start = np.copy(self.data.ctrl)
        target = np.asarray(target_ctrl, dtype=np.float64)
        for step in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step + 1) / steps))
            self.data.ctrl[:] = start + alpha * (target - start)
            mujoco.mj_step(self.model, self.data)
            self.contact_audit.sample(self.model, self.data)
            self._substeps += 1
            if self._substeps % self.sample_interval == 0:
                self.capture()
        self.data.ctrl[:] = target
        for _ in range(min(25, max(10, steps // 3))):
            mujoco.mj_step(self.model, self.data)
            self.contact_audit.sample(self.model, self.data)
            self._substeps += 1
            if self._substeps % self.sample_interval == 0:
                self.capture()


def prepare_for_skill(sim: MuJoCoSim, skill: str, executor: DemonstrationExecutor) -> None:
    """Execute prerequisite skills in the same real scene; do not record them."""
    for primitive_class in PREREQUISITES[skill]:
        if not primitive_class(executor, sim).execute():
            raise RuntimeError(f"prerequisite {primitive_class.__name__} failed")
    # Start the recorded portion cleanly: prerequisite contacts are useful for
    # debugging but must not invalidate the requested atomic demonstration.
    executor.contact_audit.violations.clear()
    executor.states.clear(); executor.actions.clear(); executor.timestamps.clear()
    executor.overhead_frames.clear(); executor.front_frames.clear()
    executor.capture()


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_episode(
    output_root: Path, skill: str, index: int, seed: int, executor: DemonstrationExecutor,
    success: bool, *, jitter: float, metrics: dict, fps: int,
) -> Path:
    episode_dir = output_root / skill / f"episode_{index:05d}_seed_{seed:04d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    overhead_dir = episode_dir / "images" / "overhead"
    front_dir = episode_dir / "images" / "front"
    overhead_dir.mkdir(parents=True)
    front_dir.mkdir(parents=True)
    for frame_index, (overhead, front) in enumerate(zip(executor.overhead_frames, executor.front_frames)):
        Image.fromarray(overhead).save(overhead_dir / f"{frame_index:06d}.png")
        Image.fromarray(front).save(front_dir / f"{frame_index:06d}.png")
    np.savez_compressed(
        episode_dir / "trajectory.npz",
        observation_state=np.asarray(executor.states, dtype=np.float32),
        action=np.asarray(executor.actions, dtype=np.float32),
        timestamp=np.asarray(executor.timestamps, dtype=np.float64),
    )
    accepted = bool(success and executor.contact_audit.ok)
    manifest = {
        "format": "bimanual_raw_demo_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": SKILLS[skill][1],
        "skill": skill,
        "seed": seed,
        "split": "train",
        "physics_only": True,
        "trajectory_jitter": jitter,
        "fps_requested": fps,
        "state": {"name": "observation.state", "shape": [12], "meaning": "A then B: 5 joints + gripper"},
        "action": {"name": "action", "shape": [12], "meaning": "position actuator targets, A then B"},
        "cameras": ["overhead", "front"],
        "frames": len(executor.states),
        "primitive_returned_success": success,
        "primitive_metrics": _jsonable(metrics),
        "contact_violations": [v.__dict__ for v in executor.contact_audit.violations],
        "accepted_for_training": accepted,
        "rejection_reason": "" if accepted else (metrics.get("failure") or executor.contact_audit.summary()),
    }
    (episode_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # A quick visual proof without requiring a GUI or video codec.
    if executor.front_frames:
        frames = [Image.fromarray(frame) for frame in executor.front_frames]
        frames[0].save(episode_dir / "front_replay.gif", save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
    return episode_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one visually inspectable atomic MuJoCo skill demonstration.")
    parser.add_argument("--task", required=True, choices=SKILLS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100, help="first seed; seeds 0-9 are evaluation seeds")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--jitter", type=float, default=1.0, help="scripted-expert variation scale (0 = nominal script)")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw_skill_demos")
    parser.add_argument("--allow-eval-seeds", action="store_true", help="permit seeds 0-9 (diagnosis only)")
    parser.add_argument("--save-rejected", action="store_true", help="Save unsafe episodes for debugging; they remain rejected for training.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    seeds = [args.seed + offset for offset in range(args.episodes)]
    if not args.allow_eval_seeds and any(seed in EVAL_SEEDS for seed in seeds):
        parser.error("seeds 0-9 are the evaluation seeds and must never be used for demonstrations (--allow-eval-seeds to override)")

    primitive_class, _ = SKILLS[args.task]
    accepted = 0
    skill_root = args.output / args.task
    start_index = len(list(skill_root.glob("episode_*"))) if skill_root.exists() else 0
    for offset, seed in enumerate(seeds):
        index = start_index + offset
        sim = reset_scene(seed, trajectory_jitter=args.jitter)
        if not isinstance(sim, MuJoCoSim):
            raise RuntimeError("MuJoCo is required for demonstration collection")
        executor = DemonstrationExecutor(sim.model, sim.data, fps=args.fps, width=640, height=480, contact_audit=sim.contact_audit)
        metrics: dict = {}
        try:
            prepare_for_skill(sim, args.task, executor)
            primitive = primitive_class(executor, sim)
            success = primitive.execute()
            metrics = dict(getattr(primitive, "metrics", {}))
        except Exception as exc:
            success = False
            metrics = {"failure": f"primitive error: {exc}"}
            print(f"seed {seed}: primitive error: {exc}")
        safe = success and sim.contact_audit.ok
        if safe or args.save_rejected:
            path = save_episode(args.output, args.task, index, seed, executor, success,
                                jitter=args.jitter, metrics=metrics, fps=args.fps)
            print(f"seed {seed}: {'ACCEPTED' if safe else 'REJECTED'} ({len(executor.states)} frames) -> {path}")
        else:
            print(f"seed {seed}: REJECTED (not saved): {metrics.get('failure') or sim.contact_audit.summary()}")
        accepted += int(safe)
    print(f"Accepted training demonstrations: {accepted}/{args.episodes}")
    return 0 if accepted == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
