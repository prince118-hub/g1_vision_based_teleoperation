"""Geometric scaling: human arm keypoints to robot end-effector targets."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import WorkspaceConfig
from .transforms import apply_camera_rotation


def _unit_direction(a_cam: np.ndarray, b_cam: np.ndarray,
                    depth_scale: float) -> Optional[np.ndarray]:
    seg = b_cam - a_cam
    if np.linalg.norm(seg) < 1e-6:
        return None
    # Rotate the raw segment into the robot frame (depth_scale applied inside),
    # then normalize once. Scaling the raw segment rather than a pre-normalized
    # unit vector keeps the direction geometrically consistent and avoids
    # amplifying noise when the forward component is small.
    rotated = apply_camera_rotation(seg, depth_scale)
    n = np.linalg.norm(rotated)
    if n < 1e-6:
        return None
    return rotated / n


def compute_arm_targets(
    shoulder_cam: np.ndarray,
    elbow_cam: np.ndarray,
    wrist_cam: np.ndarray,
    upper_arm_len: float,
    forearm_len: float,
    robot_shoulder_world: np.ndarray,
    depth_scale: float,
    workspace: WorkspaceConfig,
    *,
    elbow_y_min: Optional[float] = None,
    elbow_y_max: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Scale human arm segment directions to G1 link lengths (thesis Eq 3.2-3.5).

    Directions are extracted from human keypoints, rotated into the robot frame,
    scaled to the robot's link lengths, and anchored at the live robot shoulder.
    Elbow targets are clamped to keep each arm on its own side of the chest.
    Returns (elbow_target, wrist_target) or (None, None) if input is degenerate.
    """
    u_upper = _unit_direction(shoulder_cam, elbow_cam, depth_scale)
    if u_upper is None:
        return None, None
    u_fore = _unit_direction(elbow_cam, wrist_cam, depth_scale)
    if u_fore is None:
        return None, None

    elbow_target = robot_shoulder_world + upper_arm_len * u_upper
    elbow_target[0] = max(elbow_target[0],
                          robot_shoulder_world[0] + workspace.elbow_x_min)
    if elbow_y_min is not None:
        elbow_target[1] = max(elbow_target[1], elbow_y_min)
    if elbow_y_max is not None:
        elbow_target[1] = min(elbow_target[1], elbow_y_max)

    wrist_target = elbow_target + forearm_len * u_fore
    return elbow_target, wrist_target