"""Collect bimanual demonstration episodes in Hugging Face LeRobot v2.0 dataset format.

Produces:
  meta/info.json                          - LeRobot v2.0 dataset metadata & feature schemas
  meta/tasks.jsonl                        - Language task prompt description
  meta/episodes.jsonl                     - Per-episode length, index, and task mapping
  meta/stats.json                         - Dataset normalization stats (mean/std/min/max)
  data/chunk-000/episode_{id:06d}.parquet - 12-DoF state & action trajectory vectors
  videos/chunk-000/observation.images.overhead/episode_{id:06d}.mp4 (or frames)

Features:
  observation.state: float32[12] (Arm A: 6 joints + Arm B: 6 joints)
  action:            float32[12] (Arm A: 6 control targets + Arm B: 6 control targets)
  timestamp:         float32[1]
  frame_index:       int64[1]
  episode_index:     int64[1]
  index:             int64[1]
  task_index:        int64[1]
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import cv2
    import mujoco
    import numpy as np
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as err:
    sys.exit(f"Missing dependency: {err}. Please run: pip install pyarrow Pillow mujoco numpy opencv-python")

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

JOINT_NAMES = [
    "a_shoulder_pan",
    "a_shoulder_lift",
    "a_elbow_flex",
    "a_wrist_flex",
    "a_wrist_roll",
    "a_gripper",
    "b_shoulder_pan",
    "b_shoulder_lift",
    "b_elbow_flex",
    "b_wrist_flex",
    "b_wrist_roll",
    "b_gripper",
]

TASK_DESCRIPTION = (
    "Set the table bimanually by opening the drawer, picking the ceramic plate, "
    "placing it at the table center, holding the mug with Arm B, and pouring water with Arm A."
)


class RecordingTrajectoryExecutor(TrajectoryExecutor):
    """Trajectory executor that logs high-frequency 12-DoF state, action, and camera frames."""

    def __init__(
        self,
        model: Any,
        data: Any,
        renderer: Any = None,
        camera: Any = None,
        sample_every: int = 4,  # MuJoCo timestep is 0.002s; 4 steps = 0.008s (~120Hz downsampled to ~30Hz)
    ):
        super().__init__(model, data)
        self.renderer = renderer
        self.camera = camera
        self.sample_every = sample_every

        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.images: list[np.ndarray] = []
        self.substep_count = 0

    def _sample(self) -> None:
        """Record current 12-joint state and control target."""
        # Arm A qpos: indices 36..41; Arm B qpos: indices 42..47
        qpos_12 = np.concatenate([self.data.qpos[36:42], self.data.qpos[42:48]]).astype(np.float32)
        # Actuator controls: 0..11
        ctrl_12 = np.copy(self.data.ctrl[0:12]).astype(np.float32)

        self.states.append(qpos_12)
        self.actions.append(ctrl_12)

        if self.renderer is not None and self.camera is not None:
            self.renderer.update_scene(self.data, camera=self.camera)
            img = self.renderer.render()
            self.images.append(img)

    def interpolate(self, target_ctrl: np.ndarray | list[float], steps: int = 60) -> None:
        if self.model is None or self.data is None:
            return

        start_ctrl = np.copy(self.data.ctrl)
        target_ctrl_arr = np.asarray(target_ctrl, dtype=np.float64)

        for s in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
            self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl_arr - start_ctrl)
            mujoco.mj_step(self.model, self.data)
            self.substep_count += 1
            if self.substep_count % self.sample_every == 0:
                self._sample()

        # Settling phase
        self.data.ctrl[:] = target_ctrl_arr
        settle_steps = min(25, max(10, steps // 3))
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            self.substep_count += 1
            if self.substep_count % self.sample_every == 0:
                self._sample()


def _save_video(images: list[np.ndarray], output_file: Path, fps: int = 30) -> None:
    """Save RGB image sequence as an MP4 video."""
    if not images:
        return
    h, w, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, float(fps), (w, h))
    for frame in images:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _save_gif(images: list[np.ndarray], output_file: Path, fps: int = 15, stride: int = 2) -> None:
    """Save RGB image sequence as an animated GIF."""
    if not images:
        return
    pil_images = [Image.fromarray(img) for img in images[::stride]]
    if pil_images:
        pil_images[0].save(
            output_file,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )


def collect_dataset(
    num_episodes: int = 50,
    output_dir: Path | str = "data/lerobot_bimanual_v2",
    fps: int = 30,
    render_images: bool = True,
    save_gif: bool = True,
) -> None:
    output_path = Path(output_dir)
    data_dir = output_path / "data" / "chunk-000"
    meta_dir = output_path / "meta"
    videos_dir = output_path / "videos" / "chunk-000" / "observation.images.overhead"

    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if render_images and save_gif:
        videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"Hugging Face LeRobot v2.0 Dataset Collection Pipeline")
    print(f"Target Episodes: {num_episodes}")
    print(f"Output Directory: {output_path.resolve()}")
    print(f"Render Visual Frames: {render_images}")
    print(f"Save Animated GIF:    {save_gif and render_images}")
    print(f"============================================================\n")

    episodes_meta = []
    global_frame_idx = 0
    all_states = []
    all_actions = []

    successful_episodes = 0
    seed = 0

    while successful_episodes < num_episodes:
        print(f"--- Recording Episode {successful_episodes + 1}/{num_episodes} (Seed {seed}) ---")
        sim = reset_scene(seed=seed)
        if not isinstance(sim, MuJoCoSim):
            print(f"Skipping seed {seed}: MuJoCo not available.")
            seed += 1
            continue

        m, d = sim.model, sim.data
        renderer = mujoco.Renderer(m, height=480, width=640) if render_images else None
        cam = mujoco.MjvCamera() if render_images else None
        if cam is not None:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat = [0.08, -0.05, 0.75]
            cam.distance = 1.05
            cam.elevation = -28
            cam.azimuth = 135

        executor = RecordingTrajectoryExecutor(
            m, d, renderer=renderer, camera=cam, sample_every=4
        )
        executor._sample()

        steps = [
            ("Step 1: Open Drawer", OpenDrawerPrimitive),
            ("Step 2: Pick Plate", PickPlatePrimitive),
            ("Step 3: Place Plate", PlacePlatePrimitive),
            ("Step 4: Pick Mug", PickMugPrimitive),
            ("Step 5: Pick Bottle", PickBottlePrimitive),
            ("Step 6: Pour Water", PourWaterPrimitive),
        ]

        ep_success = True
        for desc, PrimitiveClass in steps:
            p = PrimitiveClass(executor, sim)
            ok = p.execute()
            if not ok:
                ep_success = False
                print(f"  {desc}: FAILED")
                break

        if not ep_success or len(executor.states) < 10:
            print(f"  Episode failed. Discarding seed {seed}.\n")
            seed += 1
            continue

        ep_len = len(executor.states)
        ep_idx = successful_episodes
        ep_indices = np.full((ep_len,), ep_idx, dtype=np.int64)
        frame_indices = np.arange(ep_len, dtype=np.int64)
        global_indices = np.arange(global_frame_idx, global_frame_idx + ep_len, dtype=np.int64)
        task_indices = np.zeros((ep_len,), dtype=np.int64)
        timestamps = (frame_indices / float(fps)).astype(np.float32)

        states_arr = np.array(executor.states, dtype=np.float32)
        actions_arr = np.array(executor.actions, dtype=np.float32)

        all_states.append(states_arr)
        all_actions.append(actions_arr)

        # Build Arrow Table for this episode
        table = pa.table({
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": ep_indices,
            "index": global_indices,
            "task_index": task_indices,
            "observation.state": [row.tolist() for row in states_arr],
            "action": [row.tolist() for row in actions_arr],
        })

        parquet_file = data_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, parquet_file, compression="zstd")

        # Save training example GIF preview for instant verification
        if render_images and executor.images and save_gif:
            gif_file = videos_dir / f"episode_{ep_idx:06d}.gif"
            _save_gif(executor.images, gif_file, fps=15, stride=2)

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [TASK_DESCRIPTION],
            "length": int(ep_len),
        })

        global_frame_idx += ep_len
        successful_episodes += 1
        gif_note = " + gif" if (render_images and save_gif) else ""
        print(f"  -> Saved {parquet_file.name} ({ep_len} timesteps, {ep_len/fps:.1f}s){gif_note}")
        seed += 1

    # Write meta/tasks.jsonl
    tasks_file = meta_dir / "tasks.jsonl"
    with open(tasks_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")

    # Write meta/episodes.jsonl
    episodes_file = meta_dir / "episodes.jsonl"
    with open(episodes_file, "w", encoding="utf-8") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # Compute normalization stats
    all_states_cat = np.concatenate(all_states, axis=0)
    all_actions_cat = np.concatenate(all_actions, axis=0)

    stats = {
        "observation.state": {
            "mean": all_states_cat.mean(axis=0).tolist(),
            "std": all_states_cat.std(axis=0).tolist(),
            "min": all_states_cat.min(axis=0).tolist(),
            "max": all_states_cat.max(axis=0).tolist(),
        },
        "action": {
            "mean": all_actions_cat.mean(axis=0).tolist(),
            "std": all_actions_cat.std(axis=0).tolist(),
            "min": all_actions_cat.min(axis=0).tolist(),
            "max": all_actions_cat.max(axis=0).tolist(),
        },
    }
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Write meta/info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "dual_so101",
        "total_episodes": successful_episodes,
        "total_frames": global_frame_idx,
        "total_tasks": 1,
        "fps": fps,
        "splits": {
            "train": f"0:{successful_episodes}",
        },
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [12],
                "names": JOINT_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [12],
                "names": JOINT_NAMES,
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    if render_images and save_video:
        info["features"]["observation.images.overhead"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.codec": "mp4v",
            },
        }

    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\n============================================================")
    print(f"Dataset collection complete!")
    print(f"Total Successful Episodes: {successful_episodes}")
    print(f"Total Frames Logged:      {global_frame_idx}")
    print(f"Saved to:                 {output_path.resolve()}")
    print(f"============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect LeRobot v2.0 demonstration dataset.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to record (default: 5)")
    parser.add_argument("--output-dir", type=str, default="data/lerobot_bimanual_v2", help="Dataset directory")
    parser.add_argument("--fps", type=int, default=30, help="Framerate (default: 30)")
    parser.add_argument("--no-images", action="store_true", help="Skip rendering visual camera frames")
    parser.add_argument("--no-gif", action="store_true", help="Skip saving animated GIF previews")
    args = parser.parse_args()

    collect_dataset(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        fps=args.fps,
        render_images=not args.no_images,
        save_gif=not args.no_gif,
    )
