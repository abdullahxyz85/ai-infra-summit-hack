"""Stage 4: Bimanual execution of Actions with dual SO-101 arms in MuJoCo.

Provides simulation environment initialization, domain randomization,
overhead camera rendering, and bimanual trajectory actuation conforming to
the integration contracts.
"""

from pathlib import Path
from typing import Any

from common.types import Action, ExecutionResult, SceneState
from stage4_bimanual.constants import (
    ARM_A_STANDBY,
    ARM_B_STANDBY,
    GRIPPER_OPEN,
)
from stage4_bimanual.primitives import (
    OpenDrawerPrimitive,
    PickBottlePrimitive,
    PickMugPrimitive,
    PickPlatePrimitive,
    PlacePlatePrimitive,
    PourWaterPrimitive,
)
from stage4_bimanual.sim import (
    DomainRandomizer,
    FakeSim,
    HAS_MUJOCO,
    MuJoCoSim,
)
from stage4_bimanual.trajectory import TrajectoryExecutor

try:
    import mujoco
except ImportError:
    pass

_ROOT = Path(__file__).resolve().parents[1]
_SCENE_XML = _ROOT / "assets" / "bimanual_scene.xml"
_CONFIG_PATH = _ROOT / "configs" / "default.yaml"


def reset_scene(seed: int = 0, *, trajectory_jitter: float | None = None) -> Any:
    """Initialize and randomize the dual SO-101 MuJoCo scene for a given seed.

    Args:
        seed: Random seed for domain randomization (placement, yaw, lighting,
            friction, mass) read from configs/default.yaml -> randomization.
        trajectory_jitter: scale of the scripted expert's seed-deterministic
            waypoint/timing variation; None reads randomization.trajectory_jitter
            from the config (0 by default = nominal script).

    Returns:
        MuJoCoSim handle if MuJoCo is available, else FakeSim fallback.
    """
    if trajectory_jitter is None:
        trajectory_jitter = float(DomainRandomizer.load_config(_CONFIG_PATH).get("trajectory_jitter", 0.0) or 0.0)
    if not HAS_MUJOCO or not _SCENE_XML.exists():
        return FakeSim(seed=seed, trajectory_jitter=trajectory_jitter)

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)

    # Initialize arms directly in tucked standby configurations (zero sweep at t=0)
    data.qpos[36:41] = ARM_A_STANDBY
    data.qpos[41] = GRIPPER_OPEN
    data.qpos[42:47] = ARM_B_STANDBY
    data.qpos[47] = GRIPPER_OPEN
    data.ctrl[0:5] = ARM_A_STANDBY
    data.ctrl[5] = GRIPPER_OPEN
    data.ctrl[6:11] = ARM_B_STANDBY
    data.ctrl[11] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    # Apply domain randomization per seed
    DomainRandomizer.randomize(model, data, seed=seed, config_path=_CONFIG_PATH)

    # Maintain standby control holding torque after gravity settling
    data.ctrl[0:5] = ARM_A_STANDBY
    data.ctrl[5] = GRIPPER_OPEN
    data.ctrl[6:11] = ARM_B_STANDBY
    data.ctrl[11] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    return MuJoCoSim(model=model, data=data, seed=seed, trajectory_jitter=trajectory_jitter)


def get_camera_frame(sim: Any) -> Any | None:
    """Capture the tabletop camera frame as an RGB numpy array (480, 640, 3).

    Args:
        sim: Simulation handle returned by reset_scene.

    Returns:
        RGB numpy array or None if rendering fails.
    """
    if isinstance(sim, MuJoCoSim):
        return sim.get_camera_frame("overhead_cam")
    return None


def execute(actions: list[Action], sim: Any | None = None) -> ExecutionResult:
    """Execute planned Actions on the dual SO-101 MuJoCo simulation.

    Dispatches high-level actions to specialized manipulation primitives
    and verifies the resulting physical state.

    Args:
        actions: Ordered list of planned Action models to execute.
        sim: MuJoCo simulation instance.

    Returns:
        ExecutionResult containing per-action success and final SceneState.
    """
    if not isinstance(sim, MuJoCoSim) or not HAS_MUJOCO:
        final_scene = SceneState(
            objects={
                "plate": (0.05, 0.0, 0.715),
                "mug": (0.06, 0.18, 0.748),
                "water_bottle": (0.12, -0.04, 0.78),
                "spoon": (0.18, 0.08, 0.705),
                "fork": (0.18, 0.02, 0.705),
            },
            drawers={"top_drawer": "open"},
        )
        return ExecutionResult(
            action_results={a.step_id: False for a in actions},
            success=False,
            final_scene=final_scene,
            error="MuJoCo is unavailable; no manipulation was executed.",
        )

    executor = TrajectoryExecutor(sim.model, sim.data, contact_audit=sim.contact_audit)
    action_results: dict[int, bool] = {}

    for action in actions:
        act_name = str(
            action.action.value if hasattr(action.action, "value") else action.action
        )
        obj = action.object

        try:
            if act_name in ("open_drawer", "ActionType.OPEN_DRAWER"):
                primitive = OpenDrawerPrimitive(executor, sim)
                success = primitive.execute()

            elif act_name in ("pick", "ActionType.PICK") and obj == "plate":
                primitive = PickPlatePrimitive(executor, sim)
                success = primitive.execute()

            elif act_name in ("place", "ActionType.PLACE") and obj == "plate":
                primitive = PlacePlatePrimitive(executor, sim)
                success = primitive.execute()

            elif act_name in ("pick", "ActionType.PICK") and obj == "mug":
                primitive = PickMugPrimitive(executor, sim)
                success = primitive.execute()

            elif act_name in ("pick", "ActionType.PICK") and obj == "water_bottle":
                primitive = PickBottlePrimitive(executor, sim)
                success = primitive.execute()

            elif act_name in ("pour", "ActionType.POUR"):
                primitive = PourWaterPrimitive(executor, sim)
                success = primitive.execute()

            else:
                # An action this executor cannot perform is a failure, never a
                # silent success (CONTRACT_PROPOSAL.md P2).
                print(f"[stage4_bimanual] Action {action.step_id}: no primitive for {act_name} {obj!r}")
                success = False

            action_results[action.step_id] = success
            if not success:
                break  # later actions depend on this one; do not pretend to run them

        except Exception as err:
            print(f"[stage4_bimanual] Action {action.step_id} execution error: {err}")
            action_results[action.step_id] = False

    # Compute final scene state from the physical simulation
    sim_objects = sim.get_object_positions()
    drawer_state = sim.get_drawer_state()

    final_scene = SceneState(
        objects={
            "plate": sim_objects.get("plate", (0.05, 0.0, 0.715)),
            "mug": sim_objects.get("mug", (0.06, 0.18, 0.748)),
            "water_bottle": sim_objects.get("water_bottle", (-0.02, -0.08, 0.78)),
            "spoon": sim_objects.get("spoon", (0.18, 0.08, 0.705)),
            "fork": sim_objects.get("fork", (0.18, 0.02, 0.705)),
        },
        drawers={"top_drawer": drawer_state},
    )

    failed = [step for step, ok in action_results.items() if not ok]
    skipped = [a.step_id for a in actions if a.step_id not in action_results]
    for step in skipped:
        action_results[step] = False
    errors = []
    if failed:
        errors.append(f"action(s) {failed} failed" + (f"; {skipped} not attempted" if skipped else ""))
    if not sim.contact_audit.ok:
        errors.append(f"collision audit failed: {sim.contact_audit.summary()}")
    return ExecutionResult(
        action_results=action_results,
        success=all(action_results.values()) and sim.contact_audit.ok,
        final_scene=final_scene,
        error="; ".join(errors) or None,
    )
