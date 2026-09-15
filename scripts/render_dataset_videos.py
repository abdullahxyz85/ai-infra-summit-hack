"""Render MP4 videos and animated GIFs for LeRobot dataset episodes.

Allows quick visual verification of collected training demonstrations.
You can render any episode seed or multiple episodes at once.
"""

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import cv2
    import mujoco
    import numpy as np
    from PIL import Image
except ImportError as err:
    sys.exit(f"Missing dependency: {err}. Please run: pip install opencv-python mujoco numpy Pillow")

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


class VideoCaptureTrajectoryExecutor(TrajectoryExecutor):
    """Logs rendered frames every sample_every steps for high quality playback."""

    def __init__(self, model, data, renderer, camera, sample_every: int = 4):
        super().__init__(model, data)
        self.renderer = renderer
        self.camera = camera
        self.sample_every = sample_every
        self.frames: list[np.ndarray] = []
        self.substep_count = 0

    def _sample(self) -> None:
        self.renderer.update_scene(self.data, camera=self.camera)
        self.frames.append(self.renderer.render())

    def interpolate(self, target_ctrl, steps: int = 60) -> None:
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

        self.data.ctrl[:] = target_ctrl_arr
        settle_steps = min(25, max(10, steps // 3))
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            self.substep_count += 1
            if self.substep_count % self.sample_every == 0:
                self._sample()


def render_episode(
    seed: int,
    out_dir: Path,
    ep_idx: int = 0,
    fps: int = 30,
    save_mp4: bool = True,
    save_gif: bool = True,
) -> bool:
    """Render a full manipulation episode to MP4 and/or GIF."""
    sim = reset_scene(seed=seed)
    if not isinstance(sim, MuJoCoSim):
        print(f"Seed {seed}: MuJoCo not available.")
        return False

    m, d = sim.model, sim.data
    renderer = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.08, -0.05, 0.75]
    cam.distance = 1.05
    cam.elevation = -28
    cam.azimuth = 135

    executor = VideoCaptureTrajectoryExecutor(m, d, renderer=renderer, camera=cam, sample_every=4)
    executor._sample()

    steps = [
        ("Step 1: Open Drawer", OpenDrawerPrimitive),
        ("Step 2: Pick Plate", PickPlatePrimitive),
        ("Step 3: Place Plate", PlacePlatePrimitive),
        ("Step 4: Pick Mug", PickMugPrimitive),
        ("Step 5: Pick Bottle", PickBottlePrimitive),
        ("Step 6: Pour Water", PourWaterPrimitive),
    ]

    for desc, PrimitiveClass in steps:
        p = PrimitiveClass(executor, sim)
        ok = p.execute()
        if not ok:
            print(f"  Seed {seed} - {desc} failed.")
            return False

    out_dir.mkdir(parents=True, exist_ok=True)

    if save_mp4 and executor.frames:
        mp4_path = out_dir / f"episode_{ep_idx:06d}.mp4"
        h, w, _ = executor.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (w, h))
        for frame in executor.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  -> Saved MP4: {mp4_path.resolve()} ({len(executor.frames)} frames)")

    if save_gif and executor.frames:
        gif_path = out_dir / f"episode_{ep_idx:06d}.gif"
        pil_frames = [Image.fromarray(f) for f in executor.frames[::2]]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / (fps / 2)),
            loop=0,
            optimize=True,
        )
        print(f"  -> Saved GIF: {gif_path.resolve()} ({len(pil_frames)} frames)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Render LeRobot dataset episode videos/GIFs.")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2], help="Episode indices/seeds to render")
    parser.add_argument("--output-dir", type=str, default="data/lerobot_bimanual_v2/videos/chunk-000/observation.images.overhead", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Framerate")
    parser.add_argument("--gif-only", action="store_true", help="Only generate GIFs")
    parser.add_argument("--mp4-only", action="store_true", help="Only generate MP4s")
    args = parser.parse_args()

    save_mp4 = not args.gif_only
    save_gif = not args.mp4_only
    out_dir = Path(args.output_dir)

    print(f"=== Rendering {len(args.episodes)} Episode Videos/GIFs ===")
    print(f"Target Output: {out_dir.resolve()}")
    print(f"Formats: MP4={save_mp4}, GIF={save_gif}\n")

    for ep in args.episodes:
        print(f"Rendering Episode {ep:06d} (Seed {ep})...")
        success = render_episode(
            seed=ep,
            out_dir=out_dir,
            ep_idx=ep,
            fps=args.fps,
            save_mp4=save_mp4,
            save_gif=save_gif,
        )
        if success:
            print(f"  -> Episode {ep:06d}: SUCCESS\n")
        else:
            print(f"  -> Episode {ep:06d}: FAILED\n")

    print("Rendering complete!")


if __name__ == "__main__":
    main()
