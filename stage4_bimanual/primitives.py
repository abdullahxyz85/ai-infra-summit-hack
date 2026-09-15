"""Dynamic bimanual manipulation primitives with real physics and IK.

Every object manipulation is physically executed by the robot arm with:
  1. Elevated pre-grasp approach (no table sweeping or knocking objects)
  2. Descent / approach to exact physical contact
  3. Closed gripper clamping, welded only after verified contact
  4. Lift to clearance height
  5. Transport above the table
  6. Smooth descent and gentle placement
  7. Opening gripper before retraction

Motion pacing: every segment has a duration in seconds and the executor is
never asked to move a joint faster than JOINT_MAX_VELOCITY_RAD_S. These
trajectories are recorded as 25 Hz demonstrations for imitation learning, so
they must look like a careful human teleoperator, not a 40 ms snap.

Trajectory jitter: when ``sim.trajectory_jitter`` is > 0 every primitive samples
small, seed-deterministic variations of its free parameters (standoffs, lift
heights, tilt angle, hold time, timing). Object placement alone gives a scripted
expert very little between-episode variation; the jitter widens the state and
action distribution a learned policy sees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from stage4_bimanual.constants import (
    ALTITUDE_GRASP_PLATE,
    ALTITUDE_SAFE_TRANSIT,
    ARM_A_STANDBY,
    ARM_B_STANDBY,
    BOTTLE_APPROACH_OPEN,
    BOTTLE_BODY_GRASP_HEIGHT,
    BOTTLE_GRASP_CLOSE,
    BOTTLE_GRASP_PITCH_RAD,
    BOTTLE_GRASP_STANDOFF,
    BOTTLE_HEIGHT,
    BOTTLE_LIFT,
    GRIPPER_CLOSED,
    GRIPPER_HALF,
    GRIPPER_MAX_VELOCITY_RAD_S,
    GRIPPER_OPEN,
    JOINT_MAX_VELOCITY_RAD_S,
    MUG_HANDLE_OFFSET,
    MUG_HOLD_PITCH_RAD,
    MUG_POUR_STATION,
    MUG_RIM_OFFSET,
    MUG_SETDOWN_XY,
    MUG_TILT_DEG,
    PHYSICS_DT,
    PLATE_TABLE_XY,
    POUR_APPROACH_SIDE_OFFSET,
    POUR_HOLD_DURATION_S,
    POUR_MOUTH_CLEARANCE,
    POUR_MOUTH_INSET,
    POUR_TILT_DEG,
    POUR_TILT_DURATION_S,
    POUR_UNTILT_DURATION_S,
    ArmIdentifier,
)
from stage4_bimanual.kinematics import DLSInverseKinematics
from stage4_bimanual.trajectory import TrajectoryExecutor

TABLE_Z = 0.70
_ARM_SLOT = {"A": slice(0, 5), "B": slice(6, 11)}
_GRIP_SLOT = {"A": 5, "B": 11}


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _min_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def tilt_deg(xmat: np.ndarray) -> float:
    """Angle between a body's local +z and the world vertical, in degrees."""
    z = np.asarray(xmat).reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))


@dataclass(frozen=True)
class TrajectoryJitter:
    """Seed-deterministic variation of the scripted expert's free parameters."""

    grasp_dz: float = 0.0        # m, bottle body grasp height
    standoff: float = 0.0        # m, pre-grasp distance
    lift: float = 0.0            # m, lift after grasp
    side_offset: float = 0.0     # m, upright bottle waiting distance beside the mug
    clearance: float = 0.0       # m, mouth height above the rim
    inset: float = 0.0           # m, mouth offset from the mug axis
    tilt_deg: float = 0.0        # deg, full pour tilt
    mug_tilt_deg: float = 0.0    # deg, mug tilt toward the bottle
    hold_s: float = 0.0          # s, pour hold time
    time_scale: float = 1.0      # multiplies every segment duration
    station_dx: float = 0.0      # m, mug pour station
    station_dy: float = 0.0
    setdown_dx: float = 0.0      # m, mug set-down point
    setdown_dy: float = 0.0
    approach_dz: float = 0.0     # m, pre-grasp heights

    @classmethod
    def sample(cls, seed: int, scale: float = 1.0) -> "TrajectoryJitter":
        rng = np.random.RandomState(int(seed) * 7919 + 13)
        u = lambda w: float(rng.uniform(-w, w) * scale)  # noqa: E731
        return cls(
            grasp_dz=u(0.010), standoff=u(0.015), lift=u(0.020), side_offset=u(0.020),
            clearance=u(0.010), inset=u(0.008), tilt_deg=u(8.0), mug_tilt_deg=u(4.0),
            hold_s=u(0.3), time_scale=float(np.clip(1.0 + u(0.15), 0.7, 1.3)),
            station_dx=u(0.015), station_dy=u(0.015), setdown_dx=u(0.020), setdown_dy=u(0.020),
            approach_dz=u(0.020),
        )

    @classmethod
    def for_sim(cls, sim: Any) -> "TrajectoryJitter":
        scale = float(getattr(sim, "trajectory_jitter", 0.0) or 0.0)
        if scale <= 0.0:
            return cls()
        return cls.sample(int(getattr(sim, "seed", 0)), scale)


class BaseManipulationPrimitive(ABC):
    """Abstract base class for manipulation primitives using dynamic IK and physics."""

    def __init__(self, executor: TrajectoryExecutor, sim: Any = None):
        self.executor = executor
        self.sim = sim
        self.model = getattr(executor, "model", None) or getattr(sim, "model", None)
        self.data = getattr(executor, "data", None) or getattr(sim, "data", None)
        self.ik = (
            DLSInverseKinematics(self.model, self.data)
            if (self.model is not None and self.data is not None)
            else None
        )
        self.jitter = TrajectoryJitter.for_sim(sim)
        self.metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------ state queries
    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.copy(self.data.xpos[body_id]), np.copy(self.data.xmat[body_id]).reshape(3, 3)

    def site_pos(self, name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return np.copy(self.data.site_xpos[site_id])

    def joint_world_axis(self, name: str) -> np.ndarray:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return np.copy(self.data.xaxis[joint_id])

    def weld_active(self, weld_name: str) -> bool:
        weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
        return weld_id >= 0 and bool(self.data.eq_active[weld_id])

    def _fail(self, reason: str) -> bool:
        self.metrics["failure"] = reason
        print(f"[{type(self).__name__}] {reason}")
        return False

    # ------------------------------------------------------------------ pacing
    def _steps(self, target_ctrl: np.ndarray, duration_s: float) -> int:
        """Physics steps for a segment: at least duration_s, never faster than the joint speed limits."""
        delta = np.abs(np.asarray(target_ctrl, dtype=np.float64) - self.data.ctrl[: len(target_ctrl)])
        arm_delta = max(float(delta[0:5].max()), float(delta[6:11].max()))
        grip_delta = max(float(delta[5]), float(delta[11]))
        n = max(
            duration_s / PHYSICS_DT,
            arm_delta / (JOINT_MAX_VELOCITY_RAD_S * PHYSICS_DT),
            grip_delta / (GRIPPER_MAX_VELOCITY_RAD_S * PHYSICS_DT),
        )
        return int(max(10, np.ceil(n)))

    def move(self, ctrl: np.ndarray, duration_s: float) -> None:
        """Smoothly drive all actuators to ctrl over duration_s (scaled by the jitter time scale)."""
        self.executor.interpolate(ctrl, steps=self._steps(ctrl, duration_s * self.jitter.time_scale))

    # ------------------------------------------------------------------ IK
    def solve_ik(
        self,
        arm: ArmIdentifier,
        target_pos: np.ndarray | list[float],
        wrist_roll: float = 0.0,
        pitch: float | None = None,
        site: str | None = None,
        q_init: list[float] | None = None,
    ) -> list[float]:
        """Solve inverse kinematics targeting the specified arm end effector."""
        if self.ik is None:
            return [0.0] * 5
        _, q_sol, _ = self.ik.solve(arm, target_pos, wrist_roll=wrist_roll, pitch=pitch, site=site, q_init=q_init)
        return q_sol

    def solve_ik_checked(
        self,
        arm: ArmIdentifier,
        target_pos: np.ndarray | list[float],
        wrist_roll: float = 0.0,
        pitch: float | None = None,
        site: str | None = None,
        tolerance_m: float = 0.012,
        q_init: list[float] | None = None,
    ) -> tuple[list[float], float, bool]:
        """IK plus an explicit reachability verdict (residual above tolerance_m = not reachable)."""
        if self.ik is None:
            return [0.0] * 5, 999.0, False
        converged, q_sol, residual = self.ik.solve(
            arm, target_pos, wrist_roll=wrist_roll, pitch=pitch, site=site, q_init=q_init
        )
        return q_sol, residual, bool(converged or residual < tolerance_m)

    def move_line(
        self,
        arm: ArmIdentifier,
        ctrl: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        waypoints: int,
        duration_s: float,
        wrist_roll: float = 0.0,
        pitch: float | None = None,
        site: str | None = None,
        stop_when: Any = None,
    ) -> float:
        """Straight Cartesian line for one arm's site through IK waypoints; returns the worst IK residual."""
        worst = 0.0
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        for point in np.linspace(start, end, waypoints + 1)[1:]:
            q, residual, _ = self.solve_ik_checked(arm, point, wrist_roll, pitch=pitch, site=site)
            worst = max(worst, residual)
            ctrl[_ARM_SLOT[arm]] = q
            self.move(ctrl, duration_s / waypoints)
            if stop_when is not None and stop_when():
                break
        return worst

    # ------------------------------------------------------------------ welds
    def _weld_bodies(self, weld_name: str) -> tuple[int, int, int]:
        weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
        if weld_id < 0:
            return -1, -1, -1
        return weld_id, int(self.model.eq_obj1id[weld_id]), int(self.model.eq_obj2id[weld_id])

    def _in_subtree(self, body_id: int, root_id: int) -> bool:
        """A jaw is a child body of the gripper base in this model."""
        while body_id > 0:
            if body_id == root_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return body_id == root_id

    def gripper_contact_bodies(self, weld_name: str) -> set[int]:
        """Gripper-side bodies (base with fixed finger, moving jaw) currently touching the weld's object."""
        _, b1, b2 = self._weld_bodies(weld_name)
        if b1 < 0:
            return set()
        touching: set[int] = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if self._in_subtree(body_a, b1) and self._in_subtree(body_b, b2):
                touching.add(body_a)
            elif self._in_subtree(body_b, b1) and self._in_subtree(body_a, b2):
                touching.add(body_b)
        return touching

    def close_until_contact(
        self,
        arm: ArmIdentifier,
        ctrl: np.ndarray,
        weld_name: str,
        *,
        floor: float = 0.0,
        squeeze: float = 0.02,
    ) -> bool:
        """Close the jaw in small increments until BOTH jaws touch the object, then squeeze a little.

        Commanding a fixed closed angle drives the pads into the object with the
        full servo torque, which bends the grasp weld and rotates the object;
        stopping at contact is also a rule a learned policy can reproduce.
        """
        _, b1, _ = self._weld_bodies(weld_name)
        slot = _GRIP_SLOT[arm]
        g = float(ctrl[slot])
        while g > floor:
            g = max(floor, g - (0.2 if g > 1.0 else 0.06))
            ctrl[slot] = g
            self.move(ctrl, 0.08)
            touching = self.gripper_contact_bodies(weld_name)
            if b1 in touching and any(b != b1 for b in touching):
                ctrl[slot] = max(floor, g - squeeze)
                self.move(ctrl, 0.25)
                return True
        return False

    def attach_weld(self, weld_name: str, require_both_jaws: bool = False) -> bool:
        """Attach only after the two bodies are in real MuJoCo contact.

        A weld is an approximation for a stable grasp in this lightweight
        scene. It must never be used as a teleport: if the robot did not reach
        the handle/object, the primitive fails visibly instead of moving it.
        With require_both_jaws the fixed finger AND the moving jaw must each
        touch the object (a real pinch, not a one-sided push).
        """
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return False
        try:
            weld_id, b1, b2 = self._weld_bodies(weld_name)
            if weld_id < 0:
                return False
            gripper_bodies = self.gripper_contact_bodies(weld_name)
            if not gripper_bodies:
                print(f"[grasp] {weld_name} refused: gripper/object contact was not established")
                return False
            if require_both_jaws:
                moving = {bid for bid in gripper_bodies if bid != b1}
                if not moving or b1 not in gripper_bodies:
                    print(f"[grasp] {weld_name} refused: only one jaw touches the object")
                    return False

            r1 = self.data.xmat[b1].reshape(3, 3)
            delta_world = self.data.xpos[b2] - self.data.xpos[b1]
            rel_pos = r1.T @ delta_world
            rel_quat = _quat_mul(_quat_inv(self.data.xquat[b1]), self.data.xquat[b2])

            self.model.eq_data[weld_id, 3:6] = rel_pos
            self.model.eq_data[weld_id, 6:10] = rel_quat
            self.data.eq_active[weld_id] = 1
            mujoco.mj_forward(self.model, self.data)
            return True
        except Exception:
            return False

    def detach_weld(self, weld_name: str) -> None:
        """Release equality weld constraint."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return
        try:
            weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            if weld_id >= 0:
                self.data.eq_active[weld_id] = 0
                mujoco.mj_forward(self.model, self.data)
        except Exception:
            pass

    @abstractmethod
    def execute(self) -> bool:
        """Execute the manipulation primitive and return success status."""


class OpenDrawerPrimitive(BaseManipulationPrimitive):
    """Arm A approaches the D-handle from the front (-X), grips it, and pulls the drawer open."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # Ensure Arm B is parked safely in standby pose
        ctrl = np.copy(self.data.ctrl)
        ctrl[6:11] = ARM_B_STANDBY
        ctrl[11] = GRIPPER_OPEN

        # 1. Query dynamic handle position
        handle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site"
        )
        if handle_site_id >= 0:
            handle_pos = np.copy(self.data.site_xpos[handle_site_id])
        else:
            handle_pos = np.array([-0.02, -0.220, 0.732])

        # --- Vertical descent approach ---
        # The extended handle bar projects well in front of the cabinet face.
        # Descending from directly above keeps the palm clear of the roof
        # while staying inside the SO-101 IK workspace.

        # 2a. Move high above the handle (z=0.95 is reachable and clears all objects)
        above_handle = np.array([handle_pos[0], handle_pos[1], ALTITUDE_SAFE_TRANSIT])
        q_above = self.solve_ik("A", above_handle, wrist_roll=0.0)
        ctrl[0:5] = q_above
        ctrl[5] = GRIPPER_OPEN
        self.move(ctrl, 1.2)

        # 2b. Descend vertically to handle height via Cartesian waypoints so the joint-space
        # arc never swings forward into the cabinet or prematurely displaces the handle.
        z_waypoints = np.linspace(ALTITUDE_SAFE_TRANSIT, handle_pos[2], 6)[1:]
        for z_wp in z_waypoints:
            q_wp = self.solve_ik("A", [handle_pos[0], handle_pos[1], z_wp], wrist_roll=0.0)
            ctrl[0:5] = q_wp
            self.move(ctrl, 0.25)

        # 3. Close gripper firmly on handle & attach weld
        ctrl[5] = GRIPPER_CLOSED
        self.move(ctrl, 0.6)
        if not self.attach_weld("weld_drawer"):
            # Retry: nudge slightly toward the handle (+X in world = closer to cabinet)
            nudge_pos = handle_pos + np.array([0.008, 0.0, 0.0])
            q_nudge = self.solve_ik("A", nudge_pos, wrist_roll=0.0)
            ctrl[0:5] = q_nudge
            self.move(ctrl, 0.6)
            ctrl[5] = GRIPPER_CLOSED
            self.move(ctrl, 0.5)
            if not self.attach_weld("weld_drawer"):
                return self._fail("drawer handle grasp not established")

        # 4. Pull along -X by 7.5 cm — reliably exceeds the 4 cm threshold.
        pull_target = handle_pos - np.array([0.075, 0.0, 0.0])
        q_pull = self.solve_ik("A", pull_target, wrist_roll=0.0)
        ctrl[0:5] = q_pull
        self.move(ctrl, 1.5)

        drawer_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        if drawer_joint < 0:
            self.detach_weld("weld_drawer")
            return False
        slide = float(self.data.qpos[self.model.jnt_qposadr[drawer_joint]])
        if slide <= 0.04:
            self.detach_weld("weld_drawer")
            return self._fail(f"pull incomplete: slide={slide:.3f} m; need >0.040 m")

        # 5. Release and retract vertically. Loosen the pinch first: lifting with the
        # jaws still clamped on the bar drags the drawer 3-7 cm back toward closed.
        self.detach_weld("weld_drawer")
        ctrl[5] = 0.5
        self.move(ctrl, 0.4)

        # Lift straight up off the cylindrical handle with the jaws only slightly
        # parted, so the moving jaw cannot swing into the handle posts or the face.
        # Hold the gripper pitch the pull ended with: position-only IK lets the
        # pitch drift between waypoints, and the fingers then sweep forward into
        # the bar and push the drawer back closed.
        pitch_now = float(np.arcsin(np.clip(self.ik.gripper_pointing_axis("A")[2], -1.0, 1.0)))
        for z_wp in np.linspace(handle_pos[2], ALTITUDE_SAFE_TRANSIT, 6)[1:]:
            q_lift, _, ok = self.solve_ik_checked("A", [pull_target[0], pull_target[1], z_wp], 0.0, pitch=pitch_now)
            ctrl[0:5] = q_lift if ok else self.solve_ik("A", [pull_target[0], pull_target[1], z_wp], wrist_roll=0.0)
            self.move(ctrl, 0.2)

        # Once safely clear of the handle in the upper airspace, open gripper
        ctrl[5] = GRIPPER_OPEN
        self.move(ctrl, 0.4)

        final_slide = float(self.data.qpos[self.model.jnt_qposadr[drawer_joint]])
        if final_slide <= 0.04:
            return self._fail(f"drawer did not remain open after release: slide={final_slide:.3f} m")
        return True


class PickPlatePrimitive(BaseManipulationPrimitive):
    """Arm A reaches exposed plate inside opened drawer, pinches rim directly, and lifts vertically."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        # 1. Query physical plate location inside opened drawer tray
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        plate_pos = (
            np.copy(self.data.xpos[plate_id])
            if plate_id >= 0
            else np.array([0.11, -0.22, 0.720])
        )

        # Front rim of plate is at X = plate_pos[0] - 0.040
        rim_x = plate_pos[0] - 0.040
        rim_y = plate_pos[1]

        # 2. Approach from high safe transit altitude directly above plate rim
        ctrl = np.copy(self.data.ctrl)
        ctrl[0:5] = self.solve_ik("A", [rim_x, rim_y, ALTITUDE_SAFE_TRANSIT], wrist_roll=0.0)
        ctrl[5] = GRIPPER_OPEN
        self.move(ctrl, 1.0)

        # 3. Pure vertical descent to plate front rim via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_SAFE_TRANSIT, ALTITUDE_GRASP_PLATE, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [rim_x, rim_y, z_wp], wrist_roll=0.0)
            self.move(ctrl, 0.2)

        # 4. Close on the plate rim until both pads touch it, then squeeze a
        # little. A blind full close buries the pads 7-9 mm in the disc; the
        # embedded fixed pad then drags the plate over when it is released.
        closed = self.close_until_contact("A", ctrl, "weld_plate", floor=GRIPPER_CLOSED, squeeze=0.04)
        if not (closed and self.attach_weld("weld_plate", require_both_jaws=True)):
            ctrl[5] = GRIPPER_CLOSED
            self.move(ctrl, 0.5)
            if not self.attach_weld("weld_plate"):
                return self._fail("plate rim grasp not established")
        self.metrics["grasp_gripper_cmd"] = float(ctrl[5])
        self.detach_weld("weld_plate_drawer")

        # 5. Pure vertical lift straight up to safe transit altitude via Cartesian waypoints
        for z_wp in np.linspace(ALTITUDE_GRASP_PLATE, ALTITUDE_SAFE_TRANSIT, 6)[1:]:
            ctrl[0:5] = self.solve_ik("A", [rim_x, rim_y, z_wp], wrist_roll=0.0)
            self.move(ctrl, 0.2)

        return True


class PlacePlatePrimitive(BaseManipulationPrimitive):
    """Arm A transits plate to dining center at transit altitude, descends vertically, and parks."""

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        ctrl = np.copy(self.data.ctrl)

        # Calibrated pinch target so plate center lands accurately at (0.06, 0.00)
        pinch_target_xy = [0.020, 0.000]
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        plate_pos = np.copy(self.data.xpos[plate_id]) if plate_id >= 0 else np.array([0.06, -0.22, 0.95])

        # 1. Horizontal transit at safe transit altitude directly to dining center
        pinch_now = self.site_pos("a_pinch_site")
        for xy in np.linspace([pinch_now[0], pinch_now[1]], pinch_target_xy, 5)[1:]:
            ctrl[0:5] = self.solve_ik("A", [xy[0], xy[1], ALTITUDE_SAFE_TRANSIT], wrist_roll=0.0)
            self.move(ctrl, 0.3)

        # 2a. Descend to levelling height, then level the plate. The rim pinch
        # leaves the plate hanging perpendicular to the fingers (50-60 deg tilt
        # with the pitch the position-only IK happened to give), and a plate
        # released on its edge flips. Close to the base and low (<= 0.76 m) the
        # arm can point the fingers down, which is where the plate normal
        # becomes vertical; the raised far rim swings down to pinch height.
        level_z = ALTITUDE_GRASP_PLATE + 0.03
        for z_wp in np.linspace(ALTITUDE_SAFE_TRANSIT, level_z, 4)[1:]:
            ctrl[0:5] = self.solve_ik("A", [pinch_target_xy[0], pinch_target_xy[1], z_wp], wrist_roll=0.0)
            self.move(ctrl, 0.2)
        level_pitch = self._plate_levelling_pitch(plate_id)
        if level_pitch is not None:
            q, _, ok = self.solve_ik_checked("A", [pinch_target_xy[0], pinch_target_xy[1], level_z], 0.0, pitch=level_pitch)
            if ok:
                ctrl[0:5] = q
                self.move(ctrl, 1.0)
            else:
                level_pitch = None
        self.metrics["plate_tilt_after_levelling_deg"] = tilt_deg(self.data.xmat[plate_id])

        def ik_hold(target: list[float]) -> list[float]:
            if level_pitch is not None:
                q, _, ok = self.solve_ik_checked("A", target, 0.0, pitch=level_pitch)
                if ok:
                    return q
            return self.solve_ik("A", target, wrist_roll=0.0)

        # 2a'. Aim the plate CENTRE at the table destination using the live
        # pinch-to-centre offset (the rim pinch plus the pan rotation since the
        # pick put it wherever it is now), correcting twice.
        for _ in range(2):
            offset_xy = self.data.xpos[plate_id][:2] - self.site_pos("a_pinch_site")[:2]
            pinch_target_xy = [PLATE_TABLE_XY[0] - offset_xy[0], PLATE_TABLE_XY[1] - offset_xy[1]]
            ctrl[0:5] = ik_hold([pinch_target_xy[0], pinch_target_xy[1], level_z])
            self.move(ctrl, 0.5)

        # 2b. Vertical descent with the plate held level until it rests on the
        # table (the level plate's base is ~1.3 cm below the pinch, so a fixed
        # pinch altitude would drop it from 2 cm up).
        for z_landing in np.linspace(level_z, ALTITUDE_GRASP_PLATE - 0.03, 10)[1:]:
            ctrl[0:5] = ik_hold([pinch_target_xy[0], pinch_target_xy[1], z_landing])
            self.move(ctrl, 0.15)
            if float(self.data.xpos[plate_id][2]) <= TABLE_Z + 0.0015:
                break
        self.move(ctrl, 0.3)

        # 3. Release: open fully, then lift STRAIGHT UP with the pitch held. With
        # the fingers vertical the fixed pad's tip sits inside the plate disc;
        # moving sideways first would drag the plate and flip it.
        self.detach_weld("weld_plate")
        ctrl[5] = GRIPPER_OPEN
        self.move(ctrl, 0.5)
        pinch_now = self.site_pos("a_pinch_site")
        for z_up in np.linspace(pinch_now[2], pinch_now[2] + 0.08, 4)[1:]:
            ctrl[0:5] = ik_hold([pinch_target_xy[0], pinch_target_xy[1], z_up])
            self.move(ctrl, 0.2)

        # 4. Retreat and retract to safe transit altitude
        retreat_xy = [-0.015, 0.000]
        for z_up in np.linspace(pinch_now[2] + 0.08, ALTITUDE_SAFE_TRANSIT, 4)[1:]:
            ctrl[0:5] = self.solve_ik("A", [retreat_xy[0], retreat_xy[1], z_up], wrist_roll=0.0)
            self.move(ctrl, 0.25)

        # 5. Retract Arm A to parked standby pose clear of the central workspace
        ctrl[0:5] = ARM_A_STANDBY
        self.move(ctrl, 1.2)

        plate_pos = np.copy(self.data.xpos[plate_id])
        if abs(plate_pos[2] - TABLE_Z) > 0.03 or tilt_deg(self.data.xmat[plate_id]) > 15.0:
            return self._fail("plate is not resting flat on the table after release")
        return True

    def _plate_levelling_pitch(self, plate_id: int) -> float | None:
        """Gripper pitch (rad) at which the welded plate's normal points straight up.

        The plate is rigid in the gripper, so its normal in the gripper frame is
        fixed; sweep the gripper pitch (roll 0, same reach direction) and take
        the pitch whose predicted world normal is most vertical.
        """
        if plate_id < 0:
            return None
        gripper = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "a_gripper_base")
        g_rot = self.data.xmat[gripper].reshape(3, 3)
        n_local = g_rot.T @ self.data.xmat[plate_id].reshape(3, 3)[:, 2]
        r_hat = self.ik.radial_unit("A", self.site_pos("a_pinch_site"))
        z_hat = np.array([0.0, 0.0, 1.0])
        best_pitch, best_up = None, -1.0
        for pitch in np.linspace(-1.5, 0.3, 91):
            pointing = np.cos(pitch) * r_hat + np.sin(pitch) * z_hat
            x_axis = -np.sin(pitch) * r_hat + np.cos(pitch) * z_hat   # jaw closing axis at roll 0
            z_axis = -pointing
            y_axis = np.cross(z_axis, x_axis)
            rot = np.column_stack([x_axis, y_axis, z_axis])
            up = float((rot @ n_local)[2])
            if up > best_up:
                best_up, best_pitch = up, float(pitch)
        return best_pitch if best_up > 0.9 else None


class PickMugPrimitive(BaseManipulationPrimitive):
    """Arm B grasps the mug by its handle with a pitched gripper and holds it upright at the pour station.

    The gripper is pitched MUG_HOLD_PITCH_RAD below horizontal rather than
    vertical: with a vertical gripper the SO-101 wrist_flex sits on its joint
    limit at the station and the mug could never be tilted toward the bottle.
    The pitch is held constant by the IK, so the mug stays upright in transit.
    """

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True

        mug_pos, mug_rot = self.body_pose("mug")
        if tilt_deg(mug_rot) > 15.0:
            return self._fail("mug is not upright before the grasp")
        handle = mug_pos + mug_rot @ np.asarray(MUG_HANDLE_OFFSET)
        pitch = MUG_HOLD_PITCH_RAD
        jitter = self.jitter

        ctrl = np.copy(self.data.ctrl)
        ctrl[11] = GRIPPER_OPEN

        # 1. Pre-grasp above the handle, gripper already pitched
        pre_grasp = handle + np.array([0.0, 0.0, 0.10 + jitter.approach_dz])
        q, residual, ok = self.solve_ik_checked("B", pre_grasp, 0.0, pitch=pitch)
        if not ok:
            return self._fail(f"pre-grasp above the mug handle unreachable (residual {residual * 1000:.0f} mm)")
        ctrl[6:11] = q
        self.move(ctrl, 1.4)

        # 2. Descend onto the handle
        worst = self.move_line("B", ctrl, pre_grasp, handle, 5, 1.0, 0.0, pitch=pitch)
        if worst > 0.015:
            return self._fail(f"handle descent left a {worst * 1000:.0f} mm IK residual")

        # 3. Close on the handle; weld only once the pads really touch it
        ctrl[11] = GRIPPER_CLOSED
        self.move(ctrl, 0.7)
        if not self.attach_weld("weld_mug"):
            self.move(ctrl, 0.3)
            if not self.attach_weld("weld_mug"):
                return self._fail("mug handle grasp not established")

        # 4. Lift to the transit altitude (pitch held -> mug stays upright).
        # The weld froze whatever pinch-to-mug offset the grasp produced, so every
        # later target is expressed through that measured offset, not the nominal handle.
        station = np.array(MUG_POUR_STATION) + np.array([jitter.station_dx, jitter.station_dy, 0.0])
        pinch_now = self.site_pos("b_pinch_site")
        grip_offset = pinch_now - self.body_pose("mug")[0]        # pinch site relative to the mug base
        transit_z = station[2] + grip_offset[2] + 0.04 + jitter.lift
        lifted = pinch_now + np.array([0.0, 0.0, transit_z - pinch_now[2]])
        self.move_line("B", ctrl, pinch_now, lifted, 4, 1.0, 0.0, pitch=pitch)

        # 5. Transit to above the station and settle down to the hold height
        station_pinch = station + grip_offset
        above_station = np.array([station_pinch[0], station_pinch[1], transit_z])
        worst = self.move_line("B", ctrl, lifted, above_station, 5, 1.6, 0.0, pitch=pitch)
        # final settle re-measures the offset (the pan rotation during transit turns it slightly)
        station_pinch = station + (self.site_pos("b_pinch_site") - self.body_pose("mug")[0])
        worst = max(worst, self.move_line("B", ctrl, self.site_pos("b_pinch_site"), station_pinch, 3, 0.7, 0.0, pitch=pitch))
        if worst > 0.015:
            return self._fail(f"mug station unreachable (residual {worst * 1000:.0f} mm)")

        mug_pos, mug_rot = self.body_pose("mug")
        self.metrics.update(mug_tilt_deg=tilt_deg(mug_rot), mug_station_error_m=float(np.linalg.norm(mug_pos[:2] - station[:2])))
        if not self.weld_active("weld_mug"):
            return self._fail("mug weld inactive at the station")
        if tilt_deg(mug_rot) > 12.0:
            return self._fail(f"mug is tilted {tilt_deg(mug_rot):.0f} deg at the station")
        if np.linalg.norm(mug_pos[:2] - station[:2]) > 0.03:
            return self._fail("mug did not reach the pour station")
        return True


def _pour_side(ik: DLSInverseKinematics, grasp_pt: np.ndarray, rim: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Which side of the mug the bottle pours from, and the roll sign/grasp roll that gets it there.

    Returns (r_hat, t_pour, tilt_sign, roll_grasp): A's reach direction at the
    bottle, the tangential unit vector from the mug axis toward the bottle,
    the sign of the wrist-roll increment that swings the mouth toward the mug
    (+roll swings it toward the left of the reach direction), and the grasp
    roll (+-pi/2, jaws horizontal) chosen so the pour roll stays in range.
    """
    r_hat = ik.radial_unit("A", grasp_pt)
    t_hat = np.array([-r_hat[1], r_hat[0], 0.0])              # left of the reach direction
    r_b = ik.radial_unit("B", rim)                             # B's reach direction at the mug
    # The mug can only tip toward the far side of B's reach, so the bottle
    # pours from that side; pick the tangential side of A accordingly.
    t_pour = t_hat if float(t_hat @ r_b) > 0.0 else -t_hat
    tilt_sign = -1.0 if float(t_hat @ t_pour) > 0.0 else 1.0
    return r_hat, t_pour, tilt_sign, -tilt_sign * np.pi / 2


class PickBottlePrimitive(BaseManipulationPrimitive):
    """Arm A grasps the water bottle by its upper body from the side and lifts it.

    The fingers come in horizontally (pitched BOTTLE_GRASP_PITCH_RAD), the jaws
    close horizontally around the 48 mm body and stop at contact, and the weld
    is attached only when both jaws touch. The wrist roll is chosen so the
    later pour (a roll about the fingers' axis) swings the mouth toward the mug
    without leaving the joint range.
    """

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True
        jitter = self.jitter
        metrics = self.metrics

        bottle_pos, bottle_rot = self.body_pose("water_bottle")
        if tilt_deg(bottle_rot) > 15.0 or bottle_pos[2] > TABLE_Z + 0.02:
            return self._fail("bottle is not standing upright on the table")
        rim = self._rim() if self.weld_active("weld_mug") else np.array(MUG_POUR_STATION) + np.asarray(MUG_RIM_OFFSET)

        grasp_height = BOTTLE_BODY_GRASP_HEIGHT + jitter.grasp_dz
        grasp_pt = bottle_pos + np.array([0.0, 0.0, grasp_height])
        r_hat, t_pour, tilt_sign, roll_grasp = _pour_side(self.ik, grasp_pt, rim)
        pitch_g = BOTTLE_GRASP_PITCH_RAD
        side_site = "a_side_grasp_site"
        metrics.update(t_pour=t_pour.round(3).tolist(), tilt_sign=tilt_sign, roll_grasp=float(roll_grasp))

        ctrl = np.copy(self.data.ctrl)
        ctrl[5] = BOTTLE_APPROACH_OPEN

        # 1. approach the bottle body: high point behind it, down, then straight in
        standoff = BOTTLE_GRASP_STANDOFF + jitter.standoff
        pre_grasp = grasp_pt - standoff * r_hat
        high = pre_grasp + np.array([0.0, 0.0, 0.10 + jitter.approach_dz])
        q, residual, ok = self.solve_ik_checked("A", high, roll_grasp, pitch=pitch_g, site=side_site)
        if not ok:
            return self._fail(f"bottle approach point unreachable (residual {residual * 1000:.0f} mm)")
        ctrl[0:5] = q
        self.move(ctrl, 1.3)
        self.move_line("A", ctrl, high, pre_grasp, 3, 0.7, roll_grasp, pitch=pitch_g, site=side_site)
        worst = self.move_line("A", ctrl, pre_grasp, grasp_pt, 4, 0.8, roll_grasp, pitch=pitch_g, site=side_site)
        if worst > 0.015:
            return self._fail(f"bottle grasp point unreachable (residual {worst * 1000:.0f} mm)")

        # 2. close both jaws on the body
        closed = self.close_until_contact("A", ctrl, "weld_bottle", floor=BOTTLE_GRASP_CLOSE)
        if not (closed and self.attach_weld("weld_bottle", require_both_jaws=True)):
            ctrl[5] = BOTTLE_APPROACH_OPEN
            self.move(ctrl, 0.4)
            ctrl[0:5] = self.solve_ik("A", grasp_pt + 0.01 * r_hat, roll_grasp, pitch=pitch_g, site=side_site)
            self.move(ctrl, 0.4)
            closed = self.close_until_contact("A", ctrl, "weld_bottle", floor=BOTTLE_GRASP_CLOSE - 0.1)
            if not (closed and self.attach_weld("weld_bottle", require_both_jaws=True)):
                return self._fail("bottle body grasp not established by both jaws")
        metrics["grasp_gripper_cmd"] = float(ctrl[5])
        if tilt_deg(self.body_pose("water_bottle")[1]) > 10.0:
            return self._fail("bottle was knocked over while closing the gripper")
        metrics["grasp_height_above_base_m"] = float(self.site_pos(side_site)[2] - self.body_pose("water_bottle")[0][2])

        # 3. lift straight up
        lift = BOTTLE_LIFT + jitter.lift
        lifted = grasp_pt + np.array([0.0, 0.0, lift])
        self.move_line("A", ctrl, grasp_pt, lifted, 3, 0.8, roll_grasp, pitch=pitch_g, site=side_site)
        if tilt_deg(self.body_pose("water_bottle")[1]) > 20.0:
            return self._fail("bottle tilted while lifting")
        if self.sim is not None:
            self.sim.bottle_home = grasp_pt.copy()   # where the pour returns the bottle
        return True

    def _rim(self) -> np.ndarray:
        pos, rot = self.body_pose("mug")
        return pos + rot @ np.asarray(MUG_RIM_OFFSET)


class PourWaterPrimitive(BaseManipulationPrimitive):
    """Arm A pours the held bottle into the mug arm B holds, returns it, and both arms set down.

    Pour kinematics (all targets are live measurements, never assumed poses):
      * the bottle is held by its upper body between the fingertips (see
        PickBottlePrimitive), so the mouth is BOTTLE_HEIGHT - grasp height above the grasp;
      * the pour is a wrist ROLL: the bottle rotates about the fingers' axis,
        so the mouth swings sideways toward the mug while the arm compensates
        the grasp-point motion (closed-loop mouth tracking every substep);
      * the mouth path goes from "upright beside the mug" to "POUR_MOUTH_CLEARANCE
        above the rim, slightly toward the bottle" as the tilt reaches POUR_TILT_DEG;
      * arm B tips the mug MUG_TILT_DEG toward the bottle in the same substeps
        and keeps the rim centre where it was.
    Success requires the mouth to end over the mug at full tilt and both
    vessels to stand upright on the table afterwards.
    """

    def execute(self) -> bool:
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return True
        jitter = self.jitter
        metrics = self.metrics

        # ---------------------------------------------------------------- preconditions
        if not self.weld_active("weld_mug"):
            return self._fail("arm B is not holding the mug")
        if not self.weld_active("weld_bottle"):
            return self._fail("arm A is not holding the bottle (run PickBottlePrimitive first)")
        rim = self._rim()
        if rim[2] < TABLE_Z + 0.08:
            return self._fail("mug is not held above the table")
        bottle_pos, bottle_rot = self.body_pose("water_bottle")
        if tilt_deg(bottle_rot) > 20.0:
            return self._fail("bottle is not upright in the gripper")

        # ---------------------------------------------------------------- geometry (from the live grasp)
        side_site = "a_side_grasp_site"
        pitch_g = float(np.arcsin(np.clip(self.ik.gripper_pointing_axis("A")[2], -1.0, 1.0)))
        lifted = self.site_pos(side_site)
        home = getattr(self.sim, "bottle_home", None) if self.sim is not None else None
        grasp_pt = np.array(home, dtype=np.float64) if home is not None else lifted - np.array([0.0, 0.0, BOTTLE_LIFT])
        r_hat, t_pour, tilt_sign, _ = _pour_side(self.ik, grasp_pt, rim)
        roll_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "a_wrist_roll")
        roll_grasp = float(self.data.ctrl[4])
        standoff = BOTTLE_GRASP_STANDOFF + jitter.standoff
        theta_max = np.radians(POUR_TILT_DEG + jitter.tilt_deg)
        metrics.update(t_pour=t_pour.round(3).tolist(), tilt_sign=tilt_sign, roll_grasp=roll_grasp, pitch_deg=float(np.degrees(pitch_g)))
        ctrl = np.copy(self.data.ctrl)

        # Verify the roll direction numerically from the wrist axis and the bottle's up vector
        axis = self.joint_world_axis("a_wrist_roll")
        up = bottle_rot[:, 2]
        numeric_sign = float(np.sign((np.cross(axis, up)) @ (-t_pour))) or tilt_sign
        if numeric_sign != tilt_sign:
            print(f"[pour] roll direction re-derived from the wrist axis ({numeric_sign:+.0f})")
            tilt_sign = numeric_sign
        lo, hi = self.model.jnt_range[roll_joint]
        roll_end = roll_grasp + tilt_sign * theta_max
        if not (lo + 0.05 <= roll_end <= hi - 0.05):
            theta_max = float(min(theta_max, (hi - 0.05 - roll_grasp) if tilt_sign > 0 else (roll_grasp - lo - 0.05)))
            print(f"[pour] tilt limited to {np.degrees(theta_max):.0f} deg by the wrist roll range")

        # ---------------------------------------------------------------- 4. transit beside the mug
        rim = self._rim()
        clearance = POUR_MOUTH_CLEARANCE + jitter.clearance
        side = POUR_APPROACH_SIDE_OFFSET + jitter.side_offset
        mouth_start = rim + np.array([0.0, 0.0, clearance]) + side * t_pour
        mouth_end = rim + np.array([0.0, 0.0, clearance]) + (POUR_MOUTH_INSET + jitter.inset) * t_pour
        offset = self._mouth() - self.site_pos(side_site)         # mouth relative to the grasp point (upright)
        site_start = mouth_start - offset
        cruise = np.array([site_start[0], site_start[1], max(lifted[2], site_start[2] + 0.03)])
        worst = self.move_line("A", ctrl, lifted, cruise, 5, 1.5, roll_grasp, pitch=pitch_g, site=side_site)
        worst = max(worst, self.move_line("A", ctrl, cruise, site_start, 3, 0.7, roll_grasp, pitch=pitch_g, site=side_site))
        if worst > 0.02:
            return self._fail(f"pour position unreachable (residual {worst * 1000:.0f} mm)")

        # ---------------------------------------------------------------- 5. coordinated tilt / hold / untilt
        rim_goal = self._rim()
        mug_tilt = np.radians(max(0.0, MUG_TILT_DEG + jitter.mug_tilt_deg))
        r_b = self.ik.radial_unit("B", rim_goal)
        mug_sign = -1.0 if float(t_pour @ r_b) > 0.0 else 1.0     # more negative pitch dips the far rim
        pitch_b0 = MUG_HOLD_PITCH_RAD
        hold_s = max(0.4, POUR_HOLD_DURATION_S + jitter.hold_s)
        n_tilt, n_hold, n_untilt = 18, 8, 14
        profile = ([_min_jerk(k / n_tilt) for k in range(1, n_tilt + 1)]
                   + [1.0] * n_hold
                   + [1.0 - _min_jerk(k / n_untilt) for k in range(1, n_untilt + 1)])
        durations = ([POUR_TILT_DURATION_S / n_tilt] * n_tilt + [hold_s / n_hold] * n_hold
                     + [POUR_UNTILT_DURATION_S / n_untilt] * n_untilt)
        theta_prev = 0.0
        hold_records: list[tuple[float, float, float]] = []
        worst_a = worst_b = 0.0
        for idx, (s, dt) in enumerate(zip(profile, durations)):
            theta = theta_max * s
            mouth_goal = mouth_start + (mouth_end - mouth_start) * s
            axis = self.joint_world_axis("a_wrist_roll")
            offset = self._mouth() - self.site_pos(side_site)
            offset_pred = _rot_axis(axis, tilt_sign * (theta - theta_prev)) @ offset
            site_target = mouth_goal - offset_pred
            q, res_a, _ = self.solve_ik_checked("A", site_target, roll_grasp + tilt_sign * theta, pitch=pitch_g, site=side_site)
            ctrl[0:5] = q
            worst_a = max(worst_a, res_a)
            # arm B: tip the mug toward the bottle, keeping the rim centre still
            b_target = self.site_pos("b_pinch_site") + (rim_goal - self._rim())
            q_b, res_b, ok_b = self.solve_ik_checked("B", b_target, 0.0, pitch=pitch_b0 + mug_sign * mug_tilt * s)
            if ok_b:
                ctrl[6:11] = q_b
            worst_b = max(worst_b, res_b)
            self.move(ctrl, dt)
            theta_prev = theta
            if n_tilt <= idx < n_tilt + n_hold:
                mouth = self._mouth()
                rim_now = self._rim()
                hold_records.append((
                    float(np.linalg.norm(mouth[:2] - rim_now[:2])),
                    float(mouth[2] - rim_now[2]),
                    tilt_deg(self.body_pose("water_bottle")[1]),
                    tilt_deg(self.body_pose("mug")[1]),
                ))
        offsets = np.array(hold_records)
        metrics.update(
            hold_mouth_xy_offset_m=float(offsets[:, 0].mean()),
            hold_mouth_height_above_rim_m=float(offsets[:, 1].mean()),
            hold_bottle_tilt_deg=float(offsets[:, 2].mean()),
            hold_mug_tilt_deg=float(offsets[:, 3].mean()),
            worst_ik_residual_a_m=worst_a, worst_ik_residual_b_m=worst_b,
        )
        pour_ok = (offsets[:, 0].mean() < 0.03 and 0.0 < offsets[:, 1].mean() < 0.09
                   and offsets[:, 2].mean() > min(85.0, np.degrees(theta_max) - 15.0))

        # ---------------------------------------------------------------- 6. return the bottle to the table
        trace = metrics.setdefault("trace", [])

        def mark(label: str) -> None:
            b_pos, b_rot = self.body_pose("water_bottle")
            trace.append((label, round(float(self.data.time), 2), round(tilt_deg(b_rot), 1), round(float(b_pos[2]), 4)))

        mark("pour_done")
        site_now = self.site_pos(side_site)
        self.move_line("A", ctrl, site_now, cruise, 3, 0.7, roll_grasp, pitch=pitch_g, site=side_site)
        above_home = np.array([lifted[0], lifted[1], cruise[2]])
        self.move_line("A", ctrl, cruise, above_home, 5, 1.5, roll_grasp, pitch=pitch_g, site=side_site)
        self.move_line("A", ctrl, above_home, lifted, 2, 0.4, roll_grasp, pitch=pitch_g, site=side_site)
        mark("above_home")
        bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
        self.move_line(
            "A", ctrl, lifted, grasp_pt - np.array([0.0, 0.0, 0.004]), 10, 1.2, roll_grasp, pitch=pitch_g, site=side_site,
            stop_when=lambda: float(self.data.xpos[bottle_id][2]) <= TABLE_Z + 0.0015,
        )
        self.move(ctrl, 0.25)
        mark("set_down")
        self.detach_weld("weld_bottle")
        # Open well past the contact angle (GRIPPER_HALF would CLOSE a body grasp further)
        ctrl[5] = float(np.clip(ctrl[5] + 0.45, GRIPPER_HALF, GRIPPER_OPEN))
        self.move(ctrl, 0.5)
        mark("released")
        site_now = self.site_pos(side_site)
        self.move_line("A", ctrl, site_now, site_now - standoff * r_hat, 4, 0.7, roll_grasp, pitch=pitch_g, site=side_site)
        mark("retreated")
        ctrl[5] = GRIPPER_OPEN
        self.move(ctrl, 0.3)
        site_now = self.site_pos(side_site)
        self.move_line("A", ctrl, site_now, site_now + np.array([0.0, 0.0, 0.10]), 3, 0.6, roll_grasp, pitch=pitch_g, site=side_site)
        ctrl[0:5] = ARM_A_STANDBY
        self.move(ctrl, 1.4)
        mark("a_parked")

        # ---------------------------------------------------------------- 7. arm B sets the mug down upright
        setdown = np.array([MUG_SETDOWN_XY[0] + jitter.setdown_dx, MUG_SETDOWN_XY[1] + jitter.setdown_dy, TABLE_Z])
        handle_now = self.site_pos("b_pinch_site")
        grip_offset = handle_now - self.body_pose("mug")[0]                  # live pinch-to-mug offset
        setdown_handle = setdown + grip_offset
        # Transit low (mug base 3.5 cm above the table): at the station height arm B
        # is near full reach with this pitch and the solver piles the joints onto
        # their limits, folding the forearm onto the mug.
        transit_z = TABLE_Z + 0.035 + grip_offset[2]
        self.move_line("B", ctrl, handle_now, np.array([handle_now[0], handle_now[1], transit_z]), 3, 0.6, 0.0, pitch=pitch_b0)
        self.move_line("B", ctrl, np.array([handle_now[0], handle_now[1], transit_z]),
                       np.array([setdown_handle[0], setdown_handle[1], transit_z]), 5, 1.5, 0.0, pitch=pitch_b0)
        # the pan rotation during transit turned the grip offset: re-measure before descending
        setdown_handle = setdown + (self.site_pos("b_pinch_site") - self.body_pose("mug")[0])
        handle_now = self.site_pos("b_pinch_site")
        self.move_line("B", ctrl, handle_now, np.array([setdown_handle[0], setdown_handle[1], handle_now[2]]), 2, 0.4, 0.0, pitch=pitch_b0)
        mug_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        self.move_line(
            "B", ctrl, np.array([setdown_handle[0], setdown_handle[1], handle_now[2]]), setdown_handle - np.array([0.0, 0.0, 0.004]),
            10, 1.4, 0.0, pitch=pitch_b0, stop_when=lambda: float(self.data.xpos[mug_id][2]) <= TABLE_Z + 0.0015,
        )
        self.move(ctrl, 0.25)
        self.detach_weld("weld_mug")
        ctrl[11] = GRIPPER_HALF
        self.move(ctrl, 0.5)
        handle_now = self.site_pos("b_pinch_site")
        self.move_line("B", ctrl, handle_now, handle_now + np.array([0.0, 0.0, 0.10]), 4, 0.8, 0.0, pitch=pitch_b0)
        ctrl[11] = GRIPPER_OPEN
        self.move(ctrl, 0.3)
        ctrl[6:11] = ARM_B_STANDBY
        self.move(ctrl, 1.4)
        mark("b_parked")

        # ---------------------------------------------------------------- 8. postconditions
        bottle_pos, bottle_rot = self.body_pose("water_bottle")
        mug_pos, mug_rot = self.body_pose("mug")
        metrics.update(
            bottle_end_tilt_deg=tilt_deg(bottle_rot), mug_end_tilt_deg=tilt_deg(mug_rot),
            bottle_end_z=float(bottle_pos[2]), mug_end_z=float(mug_pos[2]),
            mug_setdown_error_m=float(np.linalg.norm(mug_pos[:2] - setdown[:2])),
        )
        if not pour_ok:
            return self._fail(
                f"pour missed: mouth {offsets[:, 0].mean() * 100:.1f} cm from the mug axis, "
                f"{offsets[:, 1].mean() * 100:.1f} cm above the rim, tilt {offsets[:, 2].mean():.0f} deg"
            )
        if tilt_deg(bottle_rot) > 10.0 or bottle_pos[2] > TABLE_Z + 0.02:
            return self._fail(f"bottle not standing upright after release (tilt {tilt_deg(bottle_rot):.0f} deg)")
        if tilt_deg(mug_rot) > 10.0 or mug_pos[2] > TABLE_Z + 0.02:
            return self._fail(f"mug not standing upright after release (tilt {tilt_deg(mug_rot):.0f} deg)")
        return True

    # ------------------------------------------------------------------ live geometry
    def _mouth(self) -> np.ndarray:
        pos, rot = self.body_pose("water_bottle")
        return pos + rot @ np.array([0.0, 0.0, BOTTLE_HEIGHT])

    def _rim(self) -> np.ndarray:
        pos, rot = self.body_pose("mug")
        return pos + rot @ np.asarray(MUG_RIM_OFFSET)
