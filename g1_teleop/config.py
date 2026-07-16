"""Central configuration for the G1 vision-based teleoperation pipeline.

All tunable constants live here so experiments don't require touching logic.
Grouped by subsystem: paths, coordinate transform, smoothing, IK, workspace
limits, torso yaw, validity gating, and box spawning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH: str = r"D:\Charles_Aninon\Thesis Project\g1_vision_teleop\scene.xml"


# ─── ZED BODY_38 keypoint indices ─────────────────────────────────────────────
PELVIS: int = 0
LEFT_SHOULDER: int = 12
RIGHT_SHOULDER: int = 13
LEFT_ELBOW: int = 14
RIGHT_ELBOW: int = 15
LEFT_WRIST: int = 16
RIGHT_WRIST: int = 17

REQUIRED_ARM_KEYPOINTS: List[int] = [
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
]
# Pelvis is only needed for locomotion (not yet built). Keeping it out of the
# arm-teleop gate avoids freezing when the lower body is out of frame.
REQUIRED_KEYPOINTS: List[int] = REQUIRED_ARM_KEYPOINTS

SKELETON_PAIRS = [
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (PELVIS, LEFT_SHOULDER),
    (PELVIS, RIGHT_SHOULDER),
]


# ─── Coordinate transform (camera → robot frame) ──────────────────────────────
# ZED camera:  X=right, Y=down, Z=forward-into-scene
# G1 robot:    X=forward, Y=left, Z=up
# Empirically verified: Y_rob=+X_cam, Z_rob=-Y_cam, X_rob=-DEPTH_SCALE*Z_cam.
#
# DEPTH_SCALE controls how much forward/back reach is preserved. It was 0.3
# (crushing forward reach, causing arms to drift sideways and jitter). Noise is
# now handled by the depth_alpha temporal filter, so this can be near 1.0 to
# keep real reach. Lower slightly if forward reach becomes too noisy.
DEPTH_SCALE: float = 0.6


# ─── Joint names ──────────────────────────────────────────────────────────────
LEFT_ARM_JOINTS: List[str] = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS: List[str] = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# IK drives shoulder(3) + elbow(1); wrists are held at a natural pose.
N_IK_JOINTS: int = 4
WAIST_YAW_JOINT: str = "waist_yaw_joint"

# Seed pose for the IK nullspace: the natural joint configuration for reaching
# forward, per side, for the 4 IK joints [shoulder_pitch, shoulder_roll,
# shoulder_yaw, elbow]. The nullspace pulls the redundant DOF toward this so the
# solver settles on a consistent elbow-out/down pose instead of flipping between
# elbow-in and elbow-out (the folding-across-chest artifact). Tune live:
#   shoulder_pitch < 0 raises the arm forward; elbow > 0 bends it; roll spreads.
IK_SEED_LEFT  = [-0.3,  0.2, 0.0, 0.6]   # left arm: slight outward roll (+)
IK_SEED_RIGHT = [-0.3, -0.2, 0.0, 0.6]   # right arm: mirror roll (-)

WRIST_NATURAL: Dict[str, float] = {
    # Task-smart fixed wrist pose for bimanual grasping: palms angled inward
    # and slightly down so the hands are oriented to press a box in front.
    # Fixed (not IK-driven) to avoid the wrist-twist artifact. Tune these live:
    # roll rotates the palm, pitch angles the hand down, yaw turns it in/out.
    "left_wrist_roll_joint":   0.3,
    "left_wrist_pitch_joint":  0.2,
    "left_wrist_yaw_joint":    0.0,
    "right_wrist_roll_joint":  -0.3,
    "right_wrist_pitch_joint":  0.2,
    "right_wrist_yaw_joint":    0.0,
}

# Body names used to read live link positions.
LEFT_SHOULDER_BODY = "left_shoulder_pitch_link"
RIGHT_SHOULDER_BODY = "right_shoulder_pitch_link"
LEFT_ELBOW_BODY = "left_elbow_link"
RIGHT_ELBOW_BODY = "right_elbow_link"
LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"


@dataclass(frozen=True)
class SmoothingConfig:
    """Low-pass filter factors (thesis Eq 3.8: q = (1-a)*prev + a*new)."""
    arm_alpha: float = 0.6      # was 0.8 — a bit steadier, still responsive
    yaw_alpha: float = 0.3      # yaw is noisier, smooth harder
    depth_alpha: float = 0.5    # extra low-pass on the noisy forward/back axis
    max_coast_frames: int = 5   # hold last good pose through short tracking dropouts
    # One-Euro keypoint filter (Casiez et al. 2012): min_cutoff lower = smoother
    # when still; beta higher = more responsive during motion (less lag).
    euro_enabled: bool = True
    euro_min_cutoff: float = 2.0
    euro_beta: float = 0.08
    euro_freq: float = 30.0     # ZED body tracking runs ~30 Hz


@dataclass(frozen=True)
class IKConfig:
    """Damped least-squares IK solver parameters.

    Tuned for stability over raw responsiveness: higher damping and a stronger
    neutral bias make the solver settle to consistent joint solutions instead
    of churning between equivalent ones (the source of frame-to-frame jitter).
    """
    max_iter: int = 30
    tol: float = 1e-3
    step_size: float = 0.5
    damping: float = 0.12          # more damping = smoother, less churn
    neutral_weight: float = 0.03   # (legacy, unused by nullspace solver)
    nullspace_weight: float = 0.5  # pull toward seed pose in the nullspace (elbow-out)
    target_deadzone: float = 0.008  # (legacy, unused by stillness lock)
    still_enter: float = 0.020     # per-frame motion below 20mm counts as "still"
    still_break: float = 0.040     # exceed 40mm to unlock (One-Euro lowers source noise)
    still_frames: int = 6          # this many consecutive still frames -> relock


@dataclass(frozen=True)
class WorkspaceConfig:
    """Elbow workspace clamps that keep each arm on its own side (robot frame)."""
    elbow_y_min_left: float = 0.02    # left elbow stays >= 2cm left of center
    elbow_y_max_right: float = -0.02  # right elbow stays >= 2cm right of center
    elbow_x_min: float = -0.05        # elbows can't go >5cm behind shoulder


@dataclass(frozen=True)
class TorsoYawConfig:
    """Maps demonstrator torso yaw onto the G1 waist_yaw joint.

    sign flips turn direction (camera mirrors the demonstrator); if you turn
    left and the robot turns right, set sign = -1.0. scale dampens the motion.
    """
    sign: float = 1.0
    scale: float = 1.0


@dataclass(frozen=True)
class GatingConfig:
    """Tracking-validity gating thresholds.

    A frame is rejected (arms/yaw freeze at last good pose) when tracking is
    unreliable. This prevents self-occlusion and out-of-range poses from
    corrupting demonstrations.

    Thresholds are deliberately loose: gating should catch clearly broken
    frames (NaN, tracking dropout, limbs collapsed/exploded), not normal
    teleoperation. Tighten only if bad demos slip through.
    """
    confidence_min: float = 30.0        # per-keypoint ZED confidence floor
    max_segment_len: float = 1.5        # implausible if a limb segment exceeds this (m)
    min_segment_len: float = 0.03       # implausible if a segment collapses below this (m)
    max_facing_yaw: float = 2.5         # reject only near-full turn-around (rad, ~143deg)
    enable_yaw_gate: bool = False       # off until yaw sign is verified live


@dataclass(frozen=True)
class BoxConfig:
    """Box spawn parameters. Must match scene.xml platform_pickup position."""
    pickup_center: tuple = (1.8, 1.5)   # platform_pickup x, y (world frame)
    pickup_half: float = 0.13           # uniform sampling half-range on platform
    spawn_z: float = 0.56               # platform top (0.47) + box half-height (0.09)
    body_name: str = "box1"


@dataclass(frozen=True)
class ZEDConfig:
    """ZED camera / body-tracking runtime parameters."""
    resolution: str = "HD720"
    depth_mode: str = "NEURAL"
    confidence_threshold: int = 50


@dataclass(frozen=True)
class TeleopConfig:
    """Top-level config aggregating all subsystems."""
    model_path: str = MODEL_PATH
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    torso_yaw: TorsoYawConfig = field(default_factory=TorsoYawConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    zed: ZEDConfig = field(default_factory=ZEDConfig)