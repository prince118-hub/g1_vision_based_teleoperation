"""Camera-to-robot coordinate transforms."""
from __future__ import annotations

import numpy as np


def apply_camera_rotation(v_cam: np.ndarray, depth_scale: float) -> np.ndarray:
    """Rotate a direction vector from the ZED camera frame to the robot frame.

    Camera frame: X=right, Y=down, Z=forward-into-scene.
    Robot frame:  X=forward, Y=left, Z=up.

    Reaching toward the camera (decreasing Z_cam) maps to the robot arm going
    forward (+X_rob), hence the negation. depth_scale dampens forward/back
    motion to suppress depth-sensor noise.
    """
    return np.array([
        -depth_scale * v_cam[2],   # X_rob
        v_cam[0],                  # Y_rob
        -v_cam[1],                 # Z_rob
    ])


def torso_yaw_from_shoulders(
    left_shoulder_cam: np.ndarray,
    right_shoulder_cam: np.ndarray,
    depth_scale: float,
) -> float | None:
    """Estimate demonstrator torso yaw (radians) from the shoulder line.

    The right→left shoulder vector lies along +Y_rob when facing the camera
    square-on (yaw 0). Returns None if the shoulders are too close to give a
    reliable direction.
    """
    shoulder_line_cam = left_shoulder_cam - right_shoulder_cam
    if np.linalg.norm(shoulder_line_cam) < 1e-6:
        return None
    line_rob = apply_camera_rotation(shoulder_line_cam, depth_scale)
    # Facing square-on, the right->left shoulder line points to robot-right
    # (-Y_rob), so negate Y to place the baseline on the axis and read 0.
    # Turning tilts the line's forward (X_rob) component, giving signed yaw.
    yaw = np.arctan2(line_rob[0], -line_rob[1])
    return float((yaw + np.pi) % (2 * np.pi) - np.pi)
