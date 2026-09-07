import pyzed.sl as sl
import numpy as np

# ─── CONFIRMED BODY_38 INDICES ───────────────────────────────────────────────
PELVIS         = 0
LEFT_SHOULDER  = 12
RIGHT_SHOULDER = 13
LEFT_ELBOW     = 14
RIGHT_ELBOW    = 15
LEFT_WRIST     = 16
RIGHT_WRIST    = 17

# ─── ROTATION MATRIX (Equation 3.1 from proposal) ────────────────────────────
# Camera frame:      X=right,   Y=down,  Z=forward
# Robot frame:       X=forward, Y=left,  Z=up
# Mapping:
#   X_rob =  Z_cam
#   Y_rob = -X_cam
#   Z_rob = -Y_cam
R = np.array([
    [ 0,  0,  1],   # X_rob = Z_cam
    [-1,  0,  0],   # Y_rob = -X_cam
    [ 0, -1,  0],   # Z_rob = -Y_cam
])

def transform_to_robot_frame(p_cam):
    """Apply rotation matrix to convert camera frame → robot frame."""
    return R @ np.array(p_cam)

# ─── ZED SETUP ───────────────────────────────────────────────────────────────
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to open ZED: {err}")
    exit()

body_params = sl.BodyTrackingParameters()
body_params.enable_tracking = True
body_params.enable_body_fitting = True
body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
body_params.body_format = sl.BODY_FORMAT.BODY_38

err = zed.enable_body_tracking(body_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to enable body tracking: {err}")
    zed.close()
    exit()

body_runtime_params = sl.BodyTrackingRuntimeParameters()
body_runtime_params.detection_confidence_threshold = 50
bodies = sl.Bodies()

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
print("Coordinate Transformation Test")
print("Stand ~1.5m from camera. Move arms and verify robot frame values make sense.")
print("In robot frame: X=forward(depth), Y=left/right, Z=up/down")
print("-" * 80)

frame = 0
try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies, body_runtime_params)

            if bodies.is_new and len(bodies.body_list) > 0:
                kp = bodies.body_list[0].keypoint

                # extract raw camera frame positions
                p_cam = {
                    "pelvis"    : np.array(kp[PELVIS]),
                    "l_shoulder": np.array(kp[LEFT_SHOULDER]),
                    "r_shoulder": np.array(kp[RIGHT_SHOULDER]),
                    "l_elbow"   : np.array(kp[LEFT_ELBOW]),
                    "r_elbow"   : np.array(kp[RIGHT_ELBOW]),
                    "l_wrist"   : np.array(kp[LEFT_WRIST]),
                    "r_wrist"   : np.array(kp[RIGHT_WRIST]),
                }

                # apply coordinate transformation
                p_rob = {name: transform_to_robot_frame(pos)
                         for name, pos in p_cam.items()}

                # print every 15 frames to keep readable
                if frame % 15 == 0:
                    print(f"\n--- Frame {frame} ---")
                    print(f"{'Keypoint':<12} {'CAM (x,y,z)':<40} {'ROB (x,y,z)':<40}")
                    print(f"{'':12} {'(right,down,fwd)':<40} {'(fwd,left,up)':<40}")
                    print("-" * 90)
                    for name in p_cam:
                        c = p_cam[name]
                        r = p_rob[name]
                        cam_str = f"({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})"
                        rob_str = f"({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})"
                        print(f"{name:<12} {cam_str:<40} {rob_str:<40}")

                frame += 1

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    zed.disable_body_tracking()
    zed.close()