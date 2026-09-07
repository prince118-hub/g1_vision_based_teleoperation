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

# ─── ROTATION MATRIX (verified) ──────────────────────────────────────────────
R = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
])

def transform_to_robot_frame(p_cam):
    return R @ np.array(p_cam)

# ─── G1 LINK LENGTHS (from MJCF) ─────────────────────────────────────────────
# upper arm: shoulder_yaw_link to elbow_link
L_ua = np.sqrt(0.015783**2 + 0.0**2 + 0.080518**2)  # ≈ 0.0820m

# forearm: elbow_link to wrist_roll_link
L_fa = np.sqrt(0.1**2 + 0.00188791**2 + 0.01**2)    # ≈ 0.1001m

print(f"G1 upper arm length L_ua = {L_ua:.4f} m")
print(f"G1 forearm length  L_fa = {L_fa:.4f} m")

# ─── GEOMETRIC SCALING (Equations 3.2 - 3.5) ─────────────────────────────────
def geometric_scaling(p_sh, p_el, p_wr, L_ua, L_fa, p_sh_robot):
    """
    Scale human arm segments to G1 arm lengths.
    
    Args:
        p_sh: human shoulder position in robot frame
        p_el: human elbow position in robot frame  
        p_wr: human wrist position in robot frame
        L_ua: G1 upper arm length
        L_fa: G1 forearm length
        p_sh_robot: G1 shoulder position in robot frame (fixed)
    
    Returns:
        p_el_rob: scaled elbow target in robot frame
        p_wr_rob: scaled wrist target in robot frame
    """
    # Equation 3.2 - unit direction of upper arm
    ua_vec = p_el - p_sh
    ua_norm = np.linalg.norm(ua_vec)
    if ua_norm < 1e-6:
        return None, None
    u_ua = ua_vec / ua_norm

    # Equation 3.3 - unit direction of forearm
    fa_vec = p_wr - p_el
    fa_norm = np.linalg.norm(fa_vec)
    if fa_norm < 1e-6:
        return None, None
    u_fa = fa_vec / fa_norm

    # Equation 3.4 - scaled elbow position
    p_el_rob = p_sh_robot + L_ua * u_ua

    # Equation 3.5 - scaled wrist position
    p_wr_rob = p_el_rob + L_fa * u_fa

    return p_el_rob, p_wr_rob

# ─── G1 SHOULDER POSITIONS IN ROBOT FRAME (from MJCF, neutral pose) ──────────
# left_shoulder_pitch_link relative to torso, torso relative to pelvis
# approximate positions at neutral standing pose
G1_LEFT_SHOULDER  = np.array([0.0040,  0.1002,  0.2478])  # from MJCF
G1_RIGHT_SHOULDER = np.array([0.0040, -0.1002,  0.2478])  # symmetric

# ─── ZED SETUP ───────────────────────────────────────────────────────────────
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
zed.open(init_params)

body_params = sl.BodyTrackingParameters()
body_params.enable_tracking = True
body_params.enable_body_fitting = True
body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
body_params.body_format = sl.BODY_FORMAT.BODY_38
zed.enable_body_tracking(body_params)

body_runtime_params = sl.BodyTrackingRuntimeParameters()
body_runtime_params.detection_confidence_threshold = 50
bodies = sl.Bodies()

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
print("\nGeometric Scaling Test")
print("Move your arms and verify scaled targets track your motion direction")
print("-" * 80)

frame = 0
try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies, body_runtime_params)

            if bodies.is_new and len(bodies.body_list) > 0:
                kp = bodies.body_list[0].keypoint

                # check for nan in required keypoints
                required = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
                           RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]
                if any(np.any(np.isnan(kp[i])) for i in required):
                    continue

                # transform to robot frame
                p_l_sh = transform_to_robot_frame(kp[LEFT_SHOULDER])
                p_l_el = transform_to_robot_frame(kp[LEFT_ELBOW])
                p_l_wr = transform_to_robot_frame(kp[LEFT_WRIST])
                p_r_sh = transform_to_robot_frame(kp[RIGHT_SHOULDER])
                p_r_el = transform_to_robot_frame(kp[RIGHT_ELBOW])
                p_r_wr = transform_to_robot_frame(kp[RIGHT_WRIST])

                # apply geometric scaling
                l_el_target, l_wr_target = geometric_scaling(
                    p_l_sh, p_l_el, p_l_wr, L_ua, L_fa, G1_LEFT_SHOULDER)
                r_el_target, r_wr_target = geometric_scaling(
                    p_r_sh, p_r_el, p_r_wr, L_ua, L_fa, G1_RIGHT_SHOULDER)

                if frame % 15 == 0 and l_wr_target is not None:
                    print(f"\n--- Frame {frame} ---")
                    print(f"{'':20} {'HUMAN (rob frame)':<35} {'G1 SCALED TARGET':<35}")
                    print(f"  L wrist (human) : {p_l_wr}")
                    print(f"  L wrist (scaled): {l_wr_target}")
                    print(f"  R wrist (human) : {p_r_wr}")
                    print(f"  R wrist (scaled): {r_wr_target}")

                    # sanity: check scaled targets are within G1 reach
                    l_reach = np.linalg.norm(l_wr_target - G1_LEFT_SHOULDER)
                    r_reach = np.linalg.norm(r_wr_target - G1_RIGHT_SHOULDER)
                    max_reach = L_ua + L_fa  # ≈ 0.182m
                    print(f"  L reach: {l_reach:.3f}m (max: {max_reach:.3f}m) "
                          f"{'✓' if l_reach <= max_reach else '⚠ EXCEEDS REACH'}")
                    print(f"  R reach: {r_reach:.3f}m (max: {max_reach:.3f}m) "
                          f"{'✓' if r_reach <= max_reach else '⚠ EXCEEDS REACH'}")

                frame += 1

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    zed.disable_body_tracking()
    zed.close()