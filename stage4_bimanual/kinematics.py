"""Kinematics utilities and Damped Least Squares (DLS) Inverse Kinematics for SO-101 arms.

The SO-101 is a 5-DoF arm: shoulder_pan (yaw) followed by three parallel pitch
joints (shoulder_lift, elbow_flex, wrist_flex) and a wrist_roll about the
gripper's pointing axis. Its gripper can therefore point in any direction
inside the vertical plane selected by shoulder_pan, at any pitch, and roll
about that direction. This solver exposes exactly those degrees of freedom:

* position of a site on the gripper (always),
* the pitch of the gripper pointing axis (optional, ``pitch=``),
* the wrist roll (preset, ``wrist_roll=``).
"""

from typing import Any
import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from stage4_bimanual.constants import (
    ARM_A_JOINTS,
    ARM_B_JOINTS,
    ArmIdentifier,
)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)


class DLSInverseKinematics:
    """Damped Least Squares IK solver for the 5-DoF SO-101 arms in MuJoCo."""

    def __init__(
        self,
        model: Any,
        data: Any,
        damping: float = 0.005,
        step_size: float = 0.4,
        max_iterations: int = 150,
        tolerance_m: float = 0.003,
        direction_weight_m: float = 0.06,
        direction_tolerance: float = 0.03,
    ):
        self.model = model
        self.data = data
        self.ik_data = mujoco.MjData(model) if (HAS_MUJOCO and model is not None) else None
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance_m = tolerance_m
        # A unit error in the pointing direction is weighted like this many metres
        # of position error; 0.06 m/rad keeps the two objectives comparable.
        self.direction_weight_m = direction_weight_m
        self.direction_tolerance = direction_tolerance  # |p_target - p| (rad, small angle)

    # ------------------------------------------------------------------ helpers
    def _site_id(self, arm: ArmIdentifier, site: str | None) -> int:
        prefix = arm.lower()
        candidates = [site] if site else [f"{prefix}_pinch_site", f"{prefix}_gripperframe"]
        for name in candidates:
            if name is None:
                continue
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                return site_id
        raise ValueError(f"Site '{candidates[0]}' not found in MuJoCo model.")

    def gripper_pointing_axis(self, arm: ArmIdentifier, data: Any | None = None) -> np.ndarray:
        """World-frame unit vector along the fingers (gripper base -z axis)."""
        data = self.data if data is None else data
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm.lower()}_gripper_base")
        return -data.xmat[body].reshape(3, 3)[:, 2]

    def pan_anchor(self, arm: ArmIdentifier) -> np.ndarray:
        """World position of the shoulder_pan axis (radial directions are measured from here)."""
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm.lower()}_shoulder_pan")
        return np.array(self.data.xanchor[joint], dtype=np.float64)

    def radial_unit(self, arm: ArmIdentifier, target_pos_m: np.ndarray) -> np.ndarray:
        """Horizontal unit vector from the arm's pan axis toward target (the plane the gripper works in)."""
        delta = np.asarray(target_pos_m, dtype=np.float64) - self.pan_anchor(arm)
        delta[2] = 0.0
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            p = self.gripper_pointing_axis(arm)
            p[2] = 0.0
            norm = float(np.linalg.norm(p))
            return p / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        return delta / norm

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        arm: ArmIdentifier,
        target_pos_m: np.ndarray | list[float] | tuple[float, float, float],
        wrist_roll: float = 0.0,
        *,
        pitch: float | None = None,
        site: str | None = None,
        q_init: list[float] | np.ndarray | None = None,
    ) -> tuple[bool, list[float], float]:
        """Compute joint angles (radians) placing ``site`` at ``target_pos_m``.

        Args:
            arm: "A" or "B".
            target_pos_m: world position for the site.
            wrist_roll: preset wrist roll (rad). With ``pitch`` given it is held
                fixed; otherwise it is only the starting value (legacy behaviour).
            pitch: optional elevation of the gripper pointing axis in radians:
                0 = fingers horizontal, -pi/2 = fingers pointing straight down.
                When given, shoulder_pan / shoulder_lift / elbow_flex / wrist_flex
                solve position + pitch together (4 equations, 4 joints).
            site: site name to place (default: ``<arm>_pinch_site``).
            q_init: optional starting guess for [pan, lift, elbow, wrist_flex] (rad).
                DLS converges to the solution branch nearest its start, so a
                primitive can pick e.g. the elbow-up branch with this.

        Performs IK on an isolated scratch data copy so the live simulation state
        (qpos, qvel, equality constraints) is never mutated or teleported.

        Returns:
            (converged, joint_angles[5], residual_distance_m)
        """
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return False, [0.0] * 5, 999.0

        if self.ik_data is None:
            self.ik_data = mujoco.MjData(self.model)

        # Synchronize scratch data with current live simulation configuration
        self.ik_data.qpos[:] = self.data.qpos[:]

        target = np.asarray(target_pos_m, dtype=np.float64)
        site_id = self._site_id(arm, site)
        gripper_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm.lower()}_gripper_base")

        joint_names = ARM_A_JOINTS if arm == "A" else ARM_B_JOINTS
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_names]
        jnt_qpos_indices = [self.model.jnt_qposadr[j] for j in joint_ids]
        jnt_dof_indices = [self.model.jnt_dofadr[j] for j in joint_ids]

        # Preset wrist roll target (and optional branch seed) on scratch data
        self.ik_data.qpos[jnt_qpos_indices[4]] = wrist_roll
        if q_init is not None:
            for slot, value in enumerate(list(q_init)[:4]):
                self.ik_data.qpos[jnt_qpos_indices[slot]] = float(value)

        use_pitch = pitch is not None
        active = list(range(4)) if use_pitch else list(range(5))  # joint slots in the Jacobian
        dof_cols = [jnt_dof_indices[i] for i in active]

        if use_pitch:
            radial = self.radial_unit(arm, target)
            p_target = np.array([np.cos(pitch) * radial[0], np.cos(pitch) * radial[1], np.sin(pitch)])
            p_target /= np.linalg.norm(p_target)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        converged = False
        w = self.direction_weight_m

        for _ in range(self.max_iterations):
            mujoco.mj_forward(self.model, self.ik_data)
            current_pos = self.ik_data.site_xpos[site_id]
            pos_error = target - current_pos
            err_norm = float(np.linalg.norm(pos_error))

            mujoco.mj_jacSite(self.model, self.ik_data, jacp, None, site_id)
            J = jacp[:, dof_cols]
            error = pos_error
            if use_pitch:
                p = -self.ik_data.xmat[gripper_body].reshape(3, 3)[:, 2]
                dir_error = p_target - p
                if err_norm < self.tolerance_m and float(np.linalg.norm(dir_error)) < self.direction_tolerance:
                    converged = True
                    break
                mujoco.mj_jacBody(self.model, self.ik_data, None, jacr, gripper_body)
                # dp = omega x p = -[p]_x omega  ->  J_dir = -[p]_x J_rot
                J_dir = -_skew(p) @ jacr[:, dof_cols]
                J = np.vstack([J, w * J_dir])
                error = np.concatenate([pos_error, w * dir_error])
            elif err_norm < self.tolerance_m:
                converged = True
                break

            # Damped Least Squares update: dq = J^T (J J^T + lambda^2 I)^-1 e
            lambda_matrix = (self.damping ** 2) * np.eye(J.shape[0])
            delta_q = J.T @ np.linalg.solve(J @ J.T + lambda_matrix, error)

            for k, slot in enumerate(active):
                q_idx = jnt_qpos_indices[slot]
                self.ik_data.qpos[q_idx] += self.step_size * delta_q[k]
                limit_range = self.model.jnt_range[joint_ids[slot]]
                self.ik_data.qpos[q_idx] = np.clip(self.ik_data.qpos[q_idx], limit_range[0], limit_range[1])

        mujoco.mj_forward(self.model, self.ik_data)
        final_pos = self.ik_data.site_xpos[site_id]
        final_error = float(np.linalg.norm(target - final_pos))
        joint_angles = [float(self.ik_data.qpos[idx]) for idx in jnt_qpos_indices]
        if use_pitch:
            joint_angles[4] = float(wrist_roll)

        return converged or (final_error < 0.01 and not use_pitch), joint_angles, final_error
