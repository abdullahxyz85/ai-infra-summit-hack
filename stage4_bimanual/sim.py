"""MuJoCo simulation wrapper and domain randomization engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from stage4_bimanual.safety import ContactAudit

try:
    import mujoco
    import numpy as np
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


@dataclass
class FakeSim:
    """Fallback stub when MuJoCo is not available."""
    seed: int
    contact_audit: Any = None
    trajectory_jitter: float = 0.0


# Bodies whose pose is randomized. Free-joint bodies move through qpos; the
# drawer cabinet is fixed to the world so it moves through model.body_pos, and
# the plate (welded inside its tray) moves with it.
_FREE_OBJECTS = ("mug", "water_bottle", "spoon", "fork")
_MASS_SCALED_BODIES = ("mug", "water_bottle", "plate", "spoon", "fork")

DEFAULT_RANDOMIZATION: dict[str, Any] = {
    "object_placement_cm": [-2.0, 2.0],
    "placement_cm_by_object": {},
    "object_yaw_deg": {},
    "lighting_intensity": [0.7, 1.3],
    "friction": [0.8, 1.2],
    "mass_scale": [0.9, 1.1],
    "trajectory_jitter": 0.0,
}


class DomainRandomizer:
    """Applies domain randomization per evaluation seed to prevent sim-to-real overfitting.

    Config keys (configs/default.yaml -> randomization):
      object_placement_cm: [lo, hi]      default +-xy jitter for mug, water_bottle, spoon, fork
      placement_cm_by_object: {name: [lo, hi]}   per-object override; may include drawer_unit
                                          (the plate inside the drawer moves with it)
      object_yaw_deg: {name: [lo, hi]}   yaw about +z for free objects (e.g. the mug handle)
      lighting_intensity: [lo, hi]       multiplier on main_light diffuse
      friction: [lo, hi]                 multiplier on every geom's sliding friction
      mass_scale: [lo, hi]               multiplier on the tableware masses (not the robot links)
      trajectory_jitter: float           scale of the scripted expert's seed-deterministic
                                         waypoint/timing variation (0 = nominal script)
    """

    @staticmethod
    def load_config(config_path: Path) -> dict[str, Any]:
        """Read randomization parameters from YAML config (defaults fill in missing keys)."""
        cfg: dict[str, Any] = dict(DEFAULT_RANDOMIZATION)
        if config_path.exists():
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                cfg.update(loaded.get("randomization", {}) or {})
            except Exception:
                pass
        return cfg

    @staticmethod
    def _range(cfg: dict[str, Any], key: str, default: list[float]) -> tuple[float, float]:
        value = cfg.get(key, default)
        try:
            lo, hi = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = float(default[0]), float(default[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    @classmethod
    def randomize(cls, model: Any, data: Any, seed: int, config_path: Path) -> None:
        """Apply seeded perturbations to object locations, lighting, friction, and mass."""
        if not HAS_MUJOCO:
            return

        rng = np.random.RandomState(seed)
        cfg = cls.load_config(config_path)
        default_cm = cls._range(cfg, "object_placement_cm", DEFAULT_RANDOMIZATION["object_placement_cm"])
        per_object = cfg.get("placement_cm_by_object") or {}
        yaw_cfg = cfg.get("object_yaw_deg") or {}

        def placement_m(name: str) -> tuple[float, float]:
            lo, hi = cls._range(per_object, name, list(default_cm)) if name in per_object else default_cm
            return lo / 100.0, hi / 100.0

        # 1. Tabletop object placement jitter (free-joint bodies)
        for obj_name in _FREE_OBJECTS:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id < 0:
                continue
            jnt_id = model.body_jntadr[body_id]
            if jnt_id < 0:
                continue
            q_adr = model.jnt_qposadr[jnt_id]
            lo, hi = placement_m(obj_name)
            data.qpos[q_adr] += rng.uniform(lo, hi)
            data.qpos[q_adr + 1] += rng.uniform(lo, hi)
            if obj_name in yaw_cfg:
                ylo, yhi = cls._range(yaw_cfg, obj_name, [0.0, 0.0])
                yaw = np.radians(rng.uniform(ylo, yhi))
                spin = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                current = np.copy(data.qpos[q_adr + 3:q_adr + 7])
                result = np.zeros(4)
                mujoco.mju_mulQuat(result, spin, current)
                data.qpos[q_adr + 3:q_adr + 7] = result

        # 1b. Drawer cabinet (fixed body) and the plate resting inside it move together
        if "drawer_unit" in per_object:
            drawer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drawer_unit")
            plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
            if drawer_id >= 0:
                lo, hi = placement_m("drawer_unit")
                dx, dy = rng.uniform(lo, hi), rng.uniform(lo, hi)
                model.body_pos[drawer_id][0] += dx
                model.body_pos[drawer_id][1] += dy
                if plate_id >= 0 and model.body_jntadr[plate_id] >= 0:
                    q_adr = model.jnt_qposadr[model.body_jntadr[plate_id]]
                    data.qpos[q_adr] += dx
                    data.qpos[q_adr + 1] += dy
        mujoco.mj_forward(model, data)

        # 2. Lighting intensity variation
        lo, hi = cls._range(cfg, "lighting_intensity", DEFAULT_RANDOMIZATION["lighting_intensity"])
        light_scale = rng.uniform(lo, hi)
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "main_light")
        if light_id >= 0:
            model.light_diffuse[light_id] = np.clip(model.light_diffuse[light_id] * light_scale, 0.2, 1.0)

        # 3. Contact friction scaling
        lo, hi = cls._range(cfg, "friction", DEFAULT_RANDOMIZATION["friction"])
        fric_scale = rng.uniform(lo, hi)
        model.geom_friction[:, 0] = np.clip(model.geom_friction[:, 0] * fric_scale, 0.2, 2.5)

        # 4. Tableware mass perturbation (robot link masses stay nominal: the
        #    position servos are tuned for them and a policy cannot observe them)
        lo, hi = cls._range(cfg, "mass_scale", DEFAULT_RANDOMIZATION["mass_scale"])
        mass_scale = rng.uniform(lo, hi)
        for name in _MASS_SCALED_BODIES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                model.body_mass[body_id] *= mass_scale

        # Settle simulation under gravity and zero velocities
        for _ in range(50):
            mujoco.mj_step(model, data)
        data.qvel[:] = 0.0


class MuJoCoSim:
    """Encapsulates MuJoCo simulation environment, rendering, and state queries."""

    def __init__(
        self,
        model: Any,
        data: Any,
        seed: int,
        contact_audit: ContactAudit | None = None,
        trajectory_jitter: float = 0.0,
    ):
        self.model = model
        self.data = data
        self.seed = seed
        self._renderer: Any = None
        self.contact_audit = contact_audit or ContactAudit()
        # Scale of the scripted expert's seed-deterministic waypoint/timing
        # variation (see primitives.TrajectoryJitter). 0 = nominal script.
        self.trajectory_jitter = float(trajectory_jitter)

    def set_weld(self, weld_name: str, active: bool) -> None:
        """Toggle an equality weld constraint dynamically."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return
        try:
            weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            if weld_id >= 0:
                self.data.eq_active[weld_id] = 1 if active else 0
        except Exception:
            pass

    @property
    def renderer(self) -> Any:
        """Lazy-initialize offscreen RGB camera renderer."""
        if self._renderer is None and HAS_MUJOCO and self.model is not None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        return self._renderer

    def step(self, steps: int = 1) -> None:
        """Advance physics simulation by the given number of timesteps."""
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)

    def get_camera_frame(self, camera_name: str = "overhead_cam") -> Any | None:
        """Render RGB image (H, W, 3) from the specified camera."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return None
        try:
            self.renderer.update_scene(self.data, camera=camera_name)
            return self.renderer.render()
        except Exception as err:
            print(f"[stage4_bimanual.sim] Camera render warning: {err}")
            return None

    def get_drawer_state(self) -> str:
        """Query the drawer linear slide joint displacement."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return "closed"
        try:
            drawer_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide"
            )
            if drawer_joint_id >= 0:
                qpos_addr = self.model.jnt_qposadr[drawer_joint_id]
                slide_dist = float(self.data.qpos[qpos_addr])
                return "open" if slide_dist > 0.04 else "closed"
        except Exception:
            pass
        return "closed"

    def get_object_positions(self) -> dict[str, tuple[float, float, float]]:
        """Return 3D Cartesian coordinates for all tracked scene items."""
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return {
                "plate": (0.05, 0.0, 0.715),
                "mug": (0.06, 0.18, 0.748),
                "water_bottle": (-0.02, -0.08, 0.78),
                "spoon": (0.18, 0.08, 0.705),
                "fork": (0.18, 0.02, 0.705),
            }

        tracked: dict[str, tuple[float, float, float]] = {}
        for obj_name in ["plate", "mug", "water_bottle", "spoon", "fork", "drawer_unit"]:
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name
                )
                if body_id >= 0:
                    pos = self.data.xpos[body_id]
                    tracked[obj_name] = (
                        round(float(pos[0]), 3),
                        round(float(pos[1]), 3),
                        round(float(pos[2]), 3),
                    )
            except Exception:
                pass
        return tracked
