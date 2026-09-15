"""Stage 4 pour kinematics: IK pitch constraint, trajectory jitter, and the bimanual pour itself.

Every test that needs the simulator is skipped without mujoco. The integration
test runs the real primitives on seed 100 (never an evaluation seed) and takes
about half a minute.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
needs_mujoco = pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco is not installed")


def test_trajectory_jitter_is_seed_deterministic_and_off_by_default():
    from stage4_bimanual.primitives import TrajectoryJitter

    assert TrajectoryJitter.sample(100) == TrajectoryJitter.sample(100)
    assert TrajectoryJitter.sample(100) != TrajectoryJitter.sample(101)
    nominal = TrajectoryJitter()
    assert nominal.tilt_deg == 0.0 and nominal.time_scale == 1.0

    class NoJitter:
        seed = 100
        trajectory_jitter = 0.0

    assert TrajectoryJitter.for_sim(NoJitter()) == nominal


def test_default_config_randomization_keys_are_well_formed():
    from pathlib import Path

    from stage4_bimanual.sim import DomainRandomizer

    cfg = DomainRandomizer.load_config(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    lo, hi = cfg["object_placement_cm"]
    assert lo < 0 < hi
    for name, (a, b) in (cfg.get("placement_cm_by_object") or {}).items():
        assert a < b, name
    assert float(cfg.get("trajectory_jitter", 0.0)) == 0.0, "evaluation runs must use the nominal script"


@needs_mujoco
def test_pitch_constrained_ik_points_the_gripper_where_asked():
    from stage4_bimanual.bimanual import reset_scene
    from stage4_bimanual.kinematics import DLSInverseKinematics

    sim = reset_scene(100)
    ik = DLSInverseKinematics(sim.model, sim.data)
    for arm, target, pitch, roll, site in (
        ("A", [0.08, -0.08, 0.77], -0.35, -np.pi / 2, "a_side_grasp_site"),
        ("B", [0.02, 0.10, 0.84], -1.134, 0.0, "b_pinch_site"),
        ("B", [0.02, 0.10, 0.84], -1.134 - 0.21, 0.0, "b_pinch_site"),
    ):
        _, q, residual = ik.solve(arm, target, roll, pitch=pitch, site=site)
        assert residual < 0.004, (arm, target, pitch, residual)
        pointing = ik.gripper_pointing_axis(arm, ik.ik_data)
        assert abs(np.arcsin(pointing[2]) - pitch) < 0.05
        assert q[4] == pytest.approx(roll)


@needs_mujoco
def test_fingertip_pads_touch_only_the_bottle():
    import mujoco

    from stage4_bimanual.bimanual import reset_scene

    m = reset_scene(100).model
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    for pad in ("a_fixed_tip_pad", "a_moving_tip_pad", "b_fixed_tip_pad", "b_moving_tip_pad"):
        assert m.geom_contype[gid(pad)] == 2 and m.geom_conaffinity[gid(pad)] == 2
    for bottle_geom in ("bottle_body", "bottle_geom"):
        assert m.geom_contype[gid(bottle_geom)] & 2
    for other in ("table_surface", "mug_geom", "drawer_handle", "plate_geom"):
        assert not (m.geom_contype[gid(other)] & 2), other


@needs_mujoco
def test_bimanual_pour_lands_in_the_mug_and_leaves_both_vessels_upright():
    from stage4_bimanual.bimanual import reset_scene
    from stage4_bimanual.primitives import PickBottlePrimitive, PickMugPrimitive, PourWaterPrimitive, tilt_deg
    from stage4_bimanual.trajectory import TrajectoryExecutor

    sim = reset_scene(100)
    executor = TrajectoryExecutor(sim.model, sim.data, contact_audit=sim.contact_audit)
    assert PickMugPrimitive(executor, sim).execute()
    pick = PickBottlePrimitive(executor, sim)
    assert pick.execute()
    assert 0.06 < pick.metrics["grasp_height_above_base_m"] < 0.09, "grasp must be on the body, not the neck"
    pour = PourWaterPrimitive(executor, sim)
    assert pour.execute(), pour.metrics.get("failure")
    metrics = pour.metrics
    assert metrics["hold_mouth_xy_offset_m"] < 0.03
    assert 0.0 < metrics["hold_mouth_height_above_rim_m"] < 0.09
    assert metrics["hold_bottle_tilt_deg"] > 85.0
    assert 8.0 < metrics["hold_mug_tilt_deg"] < 20.0
    assert metrics["bottle_end_tilt_deg"] < 5.0 and metrics["mug_end_tilt_deg"] < 5.0
    assert sim.contact_audit.ok, sim.contact_audit.summary()
    assert tilt_deg(sim.data.xmat[sim.model.body("water_bottle").id]) < 5.0


@needs_mujoco
def test_execute_reports_unknown_actions_as_failures_and_stops():
    from common.types import Action, ActionType
    from stage4_bimanual import execute, reset_scene

    sim = reset_scene(100)
    result = execute(
        [Action(step_id=1, action=ActionType.HANDOFF, arm="A", object="plate"),
         Action(step_id=2, action=ActionType.OPEN_DRAWER, arm="A", object="top_drawer")],
        sim,
    )
    assert result.action_results == {1: False, 2: False}
    assert not result.success
    assert "not attempted" in (result.error or "")
