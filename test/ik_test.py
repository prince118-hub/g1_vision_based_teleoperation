import pyzed.sl as sl
import numpy as np
import mujoco
import mujoco.viewer
import cv2

# ─── KEYPOINT INDICES (BODY_38) ───────────────────────────────────────────────
PELVIS         = 0
LEFT_SHOULDER  = 12
RIGHT_SHOULDER = 13
LEFT_ELBOW     = 14
RIGHT_ELBOW    = 15
LEFT_WRIST     = 16
RIGHT_WRIST    = 17

SKELETON_PAIRS = [
    (LEFT_SHOULDER,  LEFT_ELBOW),
    (LEFT_ELBOW,     LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW,    RIGHT_WRIST),
    (LEFT_SHOULDER,  RIGHT_SHOULDER),
    (PELVIS,         LEFT_SHOULDER),
    (PELVIS,         RIGHT_SHOULDER),
]

WRIST_NATURAL = {
    "left_wrist_roll_joint":   0.0,
    "left_wrist_pitch_joint":  0.0,
    "left_wrist_yaw_joint":    0.0,
    "right_wrist_roll_joint":  0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint":   0.0,
}


# ─── COORDINATE TRANSFORM ─────────────────────────────────────────────────────
# ZED camera frame:  X=right, Y=down,  Z=forward (into scene)
# G1 robot frame:    X=forward, Y=left, Z=up
#
# Verified empirically:
#   Y_rob = +X_cam  (right arm → right arm, confirmed)
#   Z_rob = -Y_cam  (arm up → robot up, confirmed)
#   X_rob = depth scale * Z_cam (dampened to reduce depth noise artifacts)
#
DEPTH_SCALE = 0.3   # 0.0 = no forward/back mapping, 1.0 = full

def apply_R_cam(v_cam):
    """Rotate unit direction vector from camera frame to robot world frame.

    ZED camera: X=right, Y=down, Z=forward-into-scene
    G1 robot:   X=forward, Y=left, Z=up

    Depth sign: moving toward camera = decreasing Z_cam = negative Z_cam.
    We negate so that reaching TOWARD camera = robot arm goes FORWARD (+X_rob).
    """
    return np.array([
        -DEPTH_SCALE * v_cam[2],   # X_rob = -Z_cam (toward cam = forward)
                      v_cam[0],    # Y_rob = +X_cam  (verified correct)
                     -v_cam[1],    # Z_rob = -Y_cam  (up, verified correct)
    ])

# ─── SMOOTHING ────────────────────────────────────────────────────────────────
# Thesis Eq 3.8: q_smooth = (1-α)*q_prev + α*q_new
# α=0.8 means 80% new signal each frame — tracks fast, still filters jitter
ALPHA = 0.8

# ─── MUJOCO SETUP ─────────────────────────────────────────────────────────────
MODEL_PATH = r"D:\Charles_Aninon\Thesis Project\scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

def get_joint_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

def get_body_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

# Apply natural wrist orientation at startup
for joint_name, angle in WRIST_NATURAL.items():
    try:
        jid = get_joint_id(model, joint_name)
        data.qpos[model.jnt_qposadr[jid]] = angle
    except Exception:
        pass
mujoco.mj_forward(model, data)

# ─── BOX SPAWN / RESET ────────────────────────────────────────────────────────
# The box is a freejoint object added in scene.xml. Even with g1.xml's keyframe
# patched to include it, we place it here so we get per-episode randomization
# (thesis §3.3.1). MUST be called AFTER mj_resetDataKeyframe, otherwise the
# keyframe reset leaves the box at its fixed keyframe spot.
box_body_id = get_body_id(model, "box1")
box_jnt_id  = model.body_jntadr[box_body_id]
box_qadr    = model.jnt_qposadr[box_jnt_id]
box_dofadr  = model.jnt_dofadr[box_jnt_id]

PICKUP_CENTER = np.array([1.8, 1.5])   # platform_pickup x, y (world frame)
PICKUP_HALF   = 0.13                    # uniform sampling half-range on platform
BOX_SPAWN_Z   = 0.56                    # platform top (0.47) + box half-height (0.09)

def reset_box(randomize=True):
    """Place the box on the pickup platform. Call AFTER mj_resetDataKeyframe.
    randomize=True samples a new position each episode (thesis §3.3.1);
    randomize=False places it at the platform center (deterministic)."""
    if randomize:
        x = np.random.uniform(PICKUP_CENTER[0] - PICKUP_HALF, PICKUP_CENTER[0] + PICKUP_HALF)
        y = np.random.uniform(PICKUP_CENTER[1] - PICKUP_HALF, PICKUP_CENTER[1] + PICKUP_HALF)
    else:
        x, y = PICKUP_CENTER
    data.qpos[box_qadr:box_qadr+3]     = [x, y, BOX_SPAWN_Z]
    data.qpos[box_qadr+3:box_qadr+7]   = [1, 0, 0, 0]     # upright quaternion
    data.qvel[box_dofadr:box_dofadr+6] = 0                # zero linear + angular velocity
    mujoco.mj_forward(model, data)

# Deterministic placement for this teleop test. For data collection, call
# reset_box(randomize=True) at the start of each episode instead.
reset_box(randomize=False)

left_shoulder_body_id  = get_body_id(model, "left_shoulder_pitch_link")
right_shoulder_body_id = get_body_id(model, "right_shoulder_pitch_link")
left_elbow_body_id     = get_body_id(model, "left_elbow_link")
right_elbow_body_id    = get_body_id(model, "right_elbow_link")
left_wrist_body_id     = get_body_id(model, "left_wrist_yaw_link")
right_wrist_body_id    = get_body_id(model, "right_wrist_yaw_link")

# ─── LINK LENGTHS FROM MODEL (ground truth) ───────────────────────────────────
l_sh_pos = data.xpos[left_shoulder_body_id].copy()
l_el_pos = data.xpos[left_elbow_body_id].copy()
l_wr_pos = data.xpos[left_wrist_body_id].copy()
r_sh_pos = data.xpos[right_shoulder_body_id].copy()
r_el_pos = data.xpos[right_elbow_body_id].copy()
r_wr_pos = data.xpos[right_wrist_body_id].copy()

L_ua_left  = np.linalg.norm(l_el_pos - l_sh_pos)
L_fa_left  = np.linalg.norm(l_wr_pos - l_el_pos)
L_ua_right = np.linalg.norm(r_el_pos - r_sh_pos)
L_fa_right = np.linalg.norm(r_wr_pos - r_el_pos)

print("=== Link lengths from model ===")
print(f"  Left  upper arm : {L_ua_left:.4f} m")
print(f"  Left  forearm   : {L_fa_left:.4f} m")
print(f"  Right upper arm : {L_ua_right:.4f} m")
print(f"  Right forearm   : {L_fa_right:.4f} m")
print()
print("=== Shoulder positions at neutral pose ===")
print(f"  Left  shoulder world : {l_sh_pos}")
print(f"  Right shoulder world : {r_sh_pos}")
print()

# ─── ARM JOINT DEFINITIONS ────────────────────────────────────────────────────
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "left_elbow_joint",
    "left_wrist_roll_joint",     "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# IK only moves shoulder + elbow joints (first 4).
# Wrist joints are held at a natural pose — prevents twist artifact.
# Wrist joints are still recorded in state/action for completeness.
IK_LEFT_JOINTS  = LEFT_ARM_JOINTS[:4]   # shoulder x3 + elbow
IK_RIGHT_JOINTS = RIGHT_ARM_JOINTS[:4]

# Elbow workspace clamp — prevents arms going through the chest.
# Left  elbow Y must stay > ELBOW_Y_MIN_LEFT  (left of centerline)
# Right elbow Y must stay < ELBOW_Y_MAX_RIGHT (right of centerline)
# Values in robot world frame (meters). Tune if needed.
ELBOW_Y_MIN_LEFT  =  0.02   # left elbow must stay at least 2cm left of center
ELBOW_Y_MAX_RIGHT = -0.02   # right elbow must stay at least 2cm right of center
ELBOW_X_MIN       = -0.05   # elbows can't go more than 5cm behind shoulder

left_dof_ids   = [model.jnt_dofadr[get_joint_id(model, n)]  for n in LEFT_ARM_JOINTS]
right_dof_ids  = [model.jnt_dofadr[get_joint_id(model, n)]  for n in RIGHT_ARM_JOINTS]
left_qpos_ids  = [model.jnt_qposadr[get_joint_id(model, n)] for n in LEFT_ARM_JOINTS]
right_qpos_ids = [model.jnt_qposadr[get_joint_id(model, n)] for n in RIGHT_ARM_JOINTS]
left_limits    = [model.jnt_range[get_joint_id(model, n)]   for n in LEFT_ARM_JOINTS]
right_limits   = [model.jnt_range[get_joint_id(model, n)]   for n in RIGHT_ARM_JOINTS]

# IK-only subsets (shoulder + elbow, no wrist)
ik_left_dof_ids   = [model.jnt_dofadr[get_joint_id(model, n)]  for n in IK_LEFT_JOINTS]
ik_right_dof_ids  = [model.jnt_dofadr[get_joint_id(model, n)]  for n in IK_RIGHT_JOINTS]
ik_left_qpos_ids  = [model.jnt_qposadr[get_joint_id(model, n)] for n in IK_LEFT_JOINTS]
ik_right_qpos_ids = [model.jnt_qposadr[get_joint_id(model, n)] for n in IK_RIGHT_JOINTS]
ik_left_limits    = [model.jnt_range[get_joint_id(model, n)]   for n in IK_LEFT_JOINTS]
ik_right_limits   = [model.jnt_range[get_joint_id(model, n)]   for n in IK_RIGHT_JOINTS]

# Neutral pose for IK joints only (shoulder + elbow)
neutral_left_q  = np.array([data.qpos[qid] for qid in ik_left_qpos_ids])
neutral_right_q = np.array([data.qpos[qid] for qid in ik_right_qpos_ids])

# ─── GEOMETRIC SCALING ────────────────────────────────────────────────────────
def compute_targets(p_sh_cam, p_el_cam, p_wr_cam,
                    L_ua, L_fa, g1_shoulder_world,
                    elbow_y_min=None, elbow_y_max=None):
    """
    Extract direction vectors from human keypoints (camera frame),
    rotate to robot world frame, scale to G1 link lengths,
    anchor at the live G1 shoulder position.

    elbow_y_min / elbow_y_max: clamp elbow Y to prevent crossing centerline
    (arms going through chest). Left arm uses y_min, right arm uses y_max.
    """
    ua_cam = p_el_cam - p_sh_cam
    ua_norm = np.linalg.norm(ua_cam)
    if ua_norm < 1e-6:
        return None, None
    u_ua = apply_R_cam(ua_cam / ua_norm)

    fa_cam = p_wr_cam - p_el_cam
    fa_norm = np.linalg.norm(fa_cam)
    if fa_norm < 1e-6:
        return None, None
    u_fa = apply_R_cam(fa_cam / fa_norm)

    # re-normalise after rotation (DEPTH_SCALE < 1 shortens the vector)
    u_ua_len = np.linalg.norm(u_ua)
    u_fa_len = np.linalg.norm(u_fa)
    if u_ua_len < 1e-6 or u_fa_len < 1e-6:
        return None, None
    u_ua /= u_ua_len
    u_fa /= u_fa_len

    el_target = g1_shoulder_world + L_ua * u_ua
    wr_target = el_target         + L_fa * u_fa

    # ── Workspace clamp: prevent arms crossing through the chest ──────────────
    # Clamp elbow X (forward/back) so it can't go far behind the robot
    el_target[0] = max(el_target[0], g1_shoulder_world[0] + ELBOW_X_MIN)

    # Clamp elbow Y to keep each arm on its own side of the centerline
    if elbow_y_min is not None:
        el_target[1] = max(el_target[1], elbow_y_min)
    if elbow_y_max is not None:
        el_target[1] = min(el_target[1], elbow_y_max)

    # Recompute wrist target from clamped elbow so the forearm stays consistent
    wr_target = el_target + L_fa * u_fa

    return el_target, wr_target

# ─── INVERSE KINEMATICS ───────────────────────────────────────────────────────
def solve_ik(model, data,
             elbow_body_id, wrist_body_id,
             el_target, wr_target,
             qpos_ids, dof_ids, joint_limits,
             neutral_q,
             max_iter=30, tol=1e-3, step_size=0.5, damping=0.05,
             neutral_weight=0.01):
    """
    Damped least-squares IK targeting both elbow and wrist positions.

    elbow_body_id  — controls where the elbow goes (fixes the mirroring problem)
    wrist_body_id  — controls where the wrist goes
    neutral_weight — small pull toward neutral pose each iteration,
                     prevents joints from getting stuck at limits when
                     the arm is lowered (fixes the 'stuck in middle' problem)
    """
    jacp_el = np.zeros((3, model.nv))
    jacp_wr = np.zeros((3, model.nv))
    jacr    = np.zeros((3, model.nv))

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        el_pos = data.xpos[elbow_body_id].copy()
        wr_pos = data.xpos[wrist_body_id].copy()

        err_el = el_target - el_pos
        err_wr = wr_target - wr_pos
        if np.linalg.norm(err_el) < tol and np.linalg.norm(err_wr) < tol:
            break

        mujoco.mj_jac(model, data, jacp_el, jacr, el_pos, elbow_body_id)
        mujoco.mj_jac(model, data, jacp_wr, jacr, wr_pos, wrist_body_id)

        J   = np.vstack([jacp_el[:, dof_ids], jacp_wr[:, dof_ids]])
        err = np.concatenate([err_el, err_wr])

        # damped least-squares step
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)

        current_q = np.array([data.qpos[qid] for qid in qpos_ids])

        for i, (qid, lim) in enumerate(zip(qpos_ids, joint_limits)):
            # neutral bias: gently pull toward rest pose to avoid joint lock
            bias = neutral_weight * (neutral_q[i] - current_q[i])
            new_q = data.qpos[qid] + step_size * dq[i] + bias
            data.qpos[qid] = np.clip(new_q, lim[0], lim[1])

    mujoco.mj_forward(model, data)
    return np.array([data.qpos[qid] for qid in qpos_ids])

# ─── VISUALIZATION ────────────────────────────────────────────────────────────
def draw_skeleton(frame, kp_2d, conf, required_ids):
    for (a, b) in SKELETON_PAIRS:
        if kp_2d[a] is None or kp_2d[b] is None:
            continue
        cv2.line(frame, kp_2d[a], kp_2d[b], (0, 255, 0), 2)
    for idx in required_ids:
        if kp_2d[idx] is None:
            continue
        color = (0, 255, 0) if conf[idx] >= 50 else (0, 0, 255)
        cv2.circle(frame, kp_2d[idx], 6, color, -1)

def draw_debug(frame, status, l_el, r_el, l_ua, l_fa):
    y = 30
    def put(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 28
    color = (0, 255, 0) if "OK" in status else (0, 100, 255)
    put(f"Status: {status}", color)
    put(f"L_ua={l_ua:.3f}m  L_fa={l_fa:.3f}m  depth={DEPTH_SCALE}  alpha={ALPHA}",
        (200, 200, 200))
    if l_el is not None:
        put(f"L el tgt: {l_el[0]:.3f} {l_el[1]:.3f} {l_el[2]:.3f}")
        put(f"R el tgt: {r_el[0]:.3f} {r_el[1]:.3f} {r_el[2]:.3f}")

# ─── ZED SETUP ────────────────────────────────────────────────────────────────
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode        = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units  = sl.UNIT.METER
zed.open(init_params)

body_params = sl.BodyTrackingParameters()
body_params.enable_tracking     = True
body_params.enable_body_fitting = True
body_params.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
body_params.body_format         = sl.BODY_FORMAT.BODY_38
zed.enable_body_tracking(body_params)

body_runtime_params = sl.BodyTrackingRuntimeParameters()
body_runtime_params.detection_confidence_threshold = 50

bodies  = sl.Bodies()
img_zed = sl.Mat()

required = [PELVIS,
            LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST,
            RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]

prev_left_q  = neutral_left_q.copy()
prev_right_q = neutral_right_q.copy()
_diag_printed = False

print("Move your arms — G1 should mirror. Press Q in camera window to quit.")
print("Tests: raise right arm UP, then OUT to side, then lower back down.\n")

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            viewer.sync()
            continue

        zed.retrieve_image(img_zed, sl.VIEW.LEFT)
        frame = img_zed.get_data()[:, :, :3].copy()
        h, w  = frame.shape[:2]

        zed.retrieve_bodies(bodies, body_runtime_params)

        status   = "waiting for body"
        kp_2d    = [None] * 38
        conf_arr = [0]    * 38
        l_el_dbg = None
        r_el_dbg = None

        if bodies.is_new and len(bodies.body_list) > 0:
            body     = bodies.body_list[0]
            kp       = body.keypoint
            kp2d     = body.keypoint_2d
            conf_arr = body.keypoint_confidence

            for i in range(min(38, len(kp2d))):
                pt = kp2d[i]
                if pt is not None and not np.any(np.isnan(pt)):
                    kp_2d[i] = (int(np.clip(pt[0], 0, w - 1)),
                                 int(np.clip(pt[1], 0, h - 1)))

            nan_fail  = any(np.any(np.isnan(kp[i])) for i in required)
            conf_fail = any(conf_arr[i] < 50         for i in required)

            if not nan_fail and not conf_fail:
                status = "tracking OK"

                if not _diag_printed:
                    print("=== LIVE keypoint sample ===")
                    print(f"  LEFT_SHOULDER : {np.array(kp[LEFT_SHOULDER])}")
                    print(f"  LEFT_ELBOW    : {np.array(kp[LEFT_ELBOW])}")
                    print(f"  RIGHT_SHOULDER: {np.array(kp[RIGHT_SHOULDER])}")
                    print()
                    _diag_printed = True

                p_l_sh = np.array(kp[LEFT_SHOULDER],  dtype=float)
                p_l_el = np.array(kp[LEFT_ELBOW],     dtype=float)
                p_l_wr = np.array(kp[LEFT_WRIST],     dtype=float)
                p_r_sh = np.array(kp[RIGHT_SHOULDER], dtype=float)
                p_r_el = np.array(kp[RIGHT_ELBOW],    dtype=float)
                p_r_wr = np.array(kp[RIGHT_WRIST],    dtype=float)

                l_sh_world = data.xpos[left_shoulder_body_id].copy()
                r_sh_world = data.xpos[right_shoulder_body_id].copy()

                l_el_target, l_wr_target = compute_targets(
                    p_l_sh, p_l_el, p_l_wr,
                    L_ua_left, L_fa_left, l_sh_world,
                    elbow_y_min=ELBOW_Y_MIN_LEFT)   # left arm stays left of center

                r_el_target, r_wr_target = compute_targets(
                    p_r_sh, p_r_el, p_r_wr,
                    L_ua_right, L_fa_right, r_sh_world,
                    elbow_y_max=ELBOW_Y_MAX_RIGHT)  # right arm stays right of center

                if l_el_target is not None and r_el_target is not None:
                    l_el_dbg = l_el_target
                    r_el_dbg = r_el_target

                    raw_left_q = solve_ik(
                        model, data,
                        left_elbow_body_id,  left_wrist_body_id,
                        l_el_target, l_wr_target,
                        ik_left_qpos_ids, ik_left_dof_ids, ik_left_limits,
                        neutral_left_q)

                    raw_right_q = solve_ik(
                        model, data,
                        right_elbow_body_id, right_wrist_body_id,
                        r_el_target, r_wr_target,
                        ik_right_qpos_ids, ik_right_dof_ids, ik_right_limits,
                        neutral_right_q)

                    # low-pass filter (thesis Eq 3.8)
                    smooth_left_q  = (1 - ALPHA) * prev_left_q  + ALPHA * raw_left_q
                    smooth_right_q = (1 - ALPHA) * prev_right_q + ALPHA * raw_right_q

                    for i, qid in enumerate(ik_left_qpos_ids):
                        data.qpos[qid] = smooth_left_q[i]
                    for i, qid in enumerate(ik_right_qpos_ids):
                        data.qpos[qid] = smooth_right_q[i]

                    prev_left_q  = smooth_left_q.copy()
                    prev_right_q = smooth_right_q.copy()

                    mujoco.mj_forward(model, data)
                    status = "IK OK"

        draw_skeleton(frame, kp_2d, conf_arr, required)
        draw_debug(frame, status, l_el_dbg, r_el_dbg, L_ua_left, L_fa_left)

        cv2.imshow("ZED Body Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        viewer.sync()

cv2.destroyAllWindows()
zed.disable_body_tracking()
zed.close()
print("Done.")