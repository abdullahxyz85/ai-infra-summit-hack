"""Interactive 3D real-time visualizer for the bimanual table-setting simulation.

Uses the REAL manipulation primitives (same as the pipeline) so you can
visually verify the robot physically moves every object — no teleportation.

Run with:
    python scripts/visualize_run.py
    python scripts/visualize_run.py --seed 3
"""

import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco
import mujoco.viewer
import numpy as np

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


class ViewerSyncTrajectoryExecutor(TrajectoryExecutor):
    """Extends TrajectoryExecutor to sync with the MuJoCo interactive viewer
    after each physics step, so the user sees every movement in real time.
    """

    def __init__(self, model, data, viewer, contact_audit, step_delay: float = 0.012):
        super().__init__(model, data, contact_audit=contact_audit)
        self._viewer = viewer
        self._step_delay = step_delay

    def interpolate(self, target_ctrl, steps: int = 60) -> None:
        """Smooth cosine interpolation with viewer sync after each substep."""
        if self.model is None or self.data is None:
            return

        start_ctrl = np.copy(self.data.ctrl)
        target_ctrl_arr = np.asarray(target_ctrl, dtype=np.float64)

        for s in range(steps):
            if not self._viewer.is_running():
                return

            alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
            self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl_arr - start_ctrl)
            mujoco.mj_step(self.model, self.data)
            if self.contact_audit is not None:
                self.contact_audit.sample(self.model, self.data)
            self._viewer.sync()
            time.sleep(self._step_delay)

        # Hold target and settle physical joints so robot arrives before next step
        self.data.ctrl[:] = target_ctrl_arr
        settle_steps = min(25, max(10, steps // 3))
        for _ in range(settle_steps):
            if not self._viewer.is_running():
                return
            mujoco.mj_step(self.model, self.data)
            if self.contact_audit is not None:
                self.contact_audit.sample(self.model, self.data)
            self._viewer.sync()
            time.sleep(self._step_delay)


def run_interactive(seed: int = 0) -> None:
    print(f"Initializing MuJoCo bimanual scene (seed={seed})...")
    sim = reset_scene(seed=seed)
    if not isinstance(sim, MuJoCoSim):
        print("Error: MuJoCo is required for the visualizer.")
        return

    m, d = sim.model, sim.data

    print("\n" + "=" * 60)
    print("Launching MuJoCo 3D Viewer — PHYSICS-VALIDATED MODE")
    print("The sequence stops at the first invalid grasp or unsafe contact.")
    print("Controls:")
    print("  - Left click + drag:  Orbit camera")
    print("  - Right click + drag: Zoom camera")
    print("  - Middle click:       Pan camera")
    print("  - ESC:                Close viewer")
    print("=" * 60 + "\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Configure optimal isometric 3/4 perspective (centering both arms, tabletop, and all tableware)
        # Wide elevated view avoids the drawer occluding the arms and table.
        viewer.cam.lookat[:] = [-0.04, 0.00, 0.74]
        viewer.cam.distance = 1.52
        viewer.cam.elevation = -58.0
        viewer.cam.azimuth = 145.0

        time.sleep(0.5)

        if not viewer.is_running():
            return

        # Create the viewer-syncing trajectory executor
        executor = ViewerSyncTrajectoryExecutor(m, d, viewer, sim.contact_audit, step_delay=0.012)

        # Execute each REAL primitive
        steps = [
            ("Step 1/5: Arm A opening drawer...", OpenDrawerPrimitive),
            ("Step 2/5: Arm A picking plate from drawer...", PickPlatePrimitive),
            ("Step 3/5: Arm A placing plate on table...", PlacePlatePrimitive),
            ("Step 4/6: Arm B picking and holding mug...", PickMugPrimitive),
            ("Step 5/6: Arm A grasping the bottle by its body...", PickBottlePrimitive),
            ("Step 6/6: Arm A pouring water into Arm B's mug...", PourWaterPrimitive),
        ]

        for desc, PrimitiveClass in steps:
            if not viewer.is_running():
                return

            print(desc)
            primitive = PrimitiveClass(executor, sim)
            success = primitive.execute()
            status = "OK" if success else "FAILED"
            print(f"  -> {status}")

            if not success or not sim.contact_audit.ok:
                print(f"  -> Stopping. This is not a valid demonstration: {sim.contact_audit.summary()}")
                break

            if not viewer.is_running():
                return

        # Print final scene state
        positions = sim.get_object_positions()
        drawer_state = sim.get_drawer_state()
        print(f"\n=== Final Scene State ===")
        print(f"  Drawer: {drawer_state}")
        for name, pos in positions.items():
            print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        print(f"\nContact audit: {sim.contact_audit.summary()}")
        print("Inspect the table in 3D. Press ESC to exit.")

        # Keep the viewer open for inspection
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive 3D bimanual visualizer.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()
    run_interactive(seed=args.seed)
