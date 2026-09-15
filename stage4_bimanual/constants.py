"""Hardware and simulation constants for dual SO-101 robotic arms."""

from dataclasses import dataclass
from typing import Final, Literal

ArmIdentifier = Literal["A", "B"]

# Actuator and joint names
ARM_A_JOINTS: Final[list[str]] = [
    "a_shoulder_pan",
    "a_shoulder_lift",
    "a_elbow_flex",
    "a_wrist_flex",
    "a_wrist_roll",
]

ARM_B_JOINTS: Final[list[str]] = [
    "b_shoulder_pan",
    "b_shoulder_lift",
    "b_elbow_flex",
    "b_wrist_flex",
    "b_wrist_roll",
]

ARM_A_ACTUATOR_OFFSET: Final[int] = 0
ARM_B_ACTUATOR_OFFSET: Final[int] = 6

ARM_A_GRIPPER_ACTUATOR: Final[int] = 5
ARM_B_GRIPPER_ACTUATOR: Final[int] = 11

# Gripper control values (STS3215 actuator angle in radians)
GRIPPER_OPEN: Final[float] = 1.60
GRIPPER_CLOSED: Final[float] = -0.10
GRIPPER_HALF: Final[float] = 0.6


@dataclass(frozen=True)
class ArmRestPose:
    """Neutral resting configuration for SO-101 arm joints."""
    shoulder_pan: float = 0.0
    shoulder_lift: float = 0.0
    elbow_flex: float = 0.0
    wrist_flex: float = 0.0
    wrist_roll: float = 0.0
    gripper: float = GRIPPER_OPEN

    def as_array(self) -> list[float]:
        return [
            self.shoulder_pan,
            self.shoulder_lift,
            self.elbow_flex,
            self.wrist_flex,
            self.wrist_roll,
            self.gripper,
        ]


# Tabletop Standard Altitude Planes (meters)
ALTITUDE_SURFACE: Final[float] = 0.70
ALTITUDE_GRASP_PLATE: Final[float] = 0.730   # Gripper pinch site height for plate rim contact
ALTITUDE_GRASP_MUG: Final[float] = 0.765     # Pinch site height for mug upper body grasp
ALTITUDE_GRASP_BOTTLE: Final[float] = 0.860  # Pinch site height for bottle neck grasp
ALTITUDE_SAFE_TRANSIT: Final[float] = 0.95   # Unobstructed 3D airspace above all tabletop objects
ALTITUDE_APPROACH_HIGH: Final[float] = 0.910 # High waypoint for approaching tall obstacles (within kinematic reach)

# Standby configurations: compactly tucked back, clear of central tabletop workspace
ARM_A_STANDBY: Final[list[float]] = [-0.10, -0.80, 1.40, -0.60, 0.0]
ARM_B_STANDBY: Final[list[float]] = [ 0.10, -0.80, 1.40, -0.60, 0.0]

DRAWER_SLIDE_MAX_METERS: Final[float] = 0.15
DEFAULT_SUBSTEPS_PER_TRAJECTORY: Final[int] = 60

# ---------------------------------------------------------------------------
# Motion pacing. Scripted waypoints are demonstrations for a 25 Hz imitation
# policy, so every segment gets a duration and the executor never commands a
# joint faster than a real STS3215 moves under load.
PHYSICS_DT: Final[float] = 0.002
JOINT_MAX_VELOCITY_RAD_S: Final[float] = 2.0
GRIPPER_MAX_VELOCITY_RAD_S: Final[float] = 3.0

# Plate destination on the table (stage6_verify TABLE_DESTINATIONS["plate"], 4.5 cm tolerance)
PLATE_TABLE_XY: Final[tuple[float, float]] = (0.06, 0.00)

# ---------------------------------------------------------------------------
# Bimanual pour geometry (MuJoCo world frame, metres; table top at z = 0.70).
# Arm B holds the mug by its handle at the station; arm A grasps the bottle by
# its body from the side, moves it beside the mug and rolls its wrist to pour.
MUG_POUR_STATION: Final[tuple[float, float, float]] = (0.02, 0.05, 0.775)  # mug base centre while held
MUG_SETDOWN_XY: Final[tuple[float, float]] = (0.08, 0.16)                  # mug base centre after the pour (5 cm forearm clearance)
MUG_RIM_OFFSET: Final[tuple[float, float, float]] = (0.0, -0.018, 0.096)   # rim centre in the mug body frame
MUG_HANDLE_OFFSET: Final[tuple[float, float, float]] = (0.0, 0.050, 0.065)  # handle grasp point in the mug frame
MUG_TILT_DEG: Final[float] = 12.0                # mug tips toward the bottle while receiving the pour
# Arm B holds the mug handle with its fingers 65 deg below horizontal. A vertical
# gripper puts b_wrist_flex on its joint limit at the station (no tilt headroom).
MUG_HOLD_PITCH_RAD: Final[float] = -1.134

BOTTLE_BODY_RADIUS: Final[float] = 0.024
BOTTLE_HEIGHT: Final[float] = 0.160              # bottle base -> mouth
BOTTLE_BODY_GRASP_HEIGHT: Final[float] = 0.075   # side-grasp height above the bottle base (upper body)
BOTTLE_GRASP_PITCH_RAD: Final[float] = -0.35     # fingers point 20 deg below horizontal at the body grasp
BOTTLE_GRASP_STANDOFF: Final[float] = 0.06       # pre-grasp distance behind the bottle axis
BOTTLE_APPROACH_OPEN: Final[float] = 1.10        # jaw opening for the side approach (tip gap ~7 cm)
BOTTLE_GRASP_CLOSE: Final[float] = 0.30          # lowest jaw command while closing on a 48 mm body (contact ~0.75)
BOTTLE_LIFT: Final[float] = 0.09                 # vertical lift after the grasp

POUR_TILT_DEG: Final[float] = 105.0              # bottle tilt from vertical at full pour
POUR_MOUTH_CLEARANCE: Final[float] = 0.035       # mouth height above the rim at full pour
POUR_MOUTH_INSET: Final[float] = 0.012           # mouth offset from the mug axis toward the bottle
POUR_APPROACH_SIDE_OFFSET: Final[float] = 0.13   # upright bottle waits this far beside the mug axis
POUR_TILT_DURATION_S: Final[float] = 1.5
POUR_HOLD_DURATION_S: Final[float] = 1.0
POUR_UNTILT_DURATION_S: Final[float] = 1.2

