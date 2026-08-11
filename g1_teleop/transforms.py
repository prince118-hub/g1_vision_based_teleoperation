"Camera-to-robot coordinate transforms"

import numpy as np

def apply_camera_rotation(v_cam: np.ndarray, depth_scale: float) -> np.ndarray:
    return np.array([
        -depth_scale * v_cam[2],   #x_rob
        v_cam[0],                  #y_rob
        -v_cam[1],                 #z_rob
    ])

def torso_yaw_from_shoulders(
    left_shoulder_cam: np.ndarray,
    right_shoulder_cam: np.ndarray,
    depth_scale: float,
) -> float | None:
    shoulder_line_cam = left_shoulder_cam - right_shoulder_cam
    if np.linalg.norm(shoulder_line_cam) < 1e-6:
        return None
    line_rob = apply_camera_rotation(shoulder_line_cam, depth_scale)
    yaw = np.arctan2(line_rob[0], -line_rob[1])
    return float((yaw + np.pi) % (2 * np.pi) - np.pi)   
    