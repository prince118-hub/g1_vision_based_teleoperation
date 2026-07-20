"""Vision-driven locomotion evaluation — SINGLE WINDOW, for presentation.

Same simulation as run_integrated_vision.py (walking policy on legs, ZED arm IK
on arms, locomotion command from a vision strategy) but rendered into ONE
window: camera feed on the left, MuJoCo on the right. Intended for showing an
adviser the failure modes rather than describing them.

    python run_integrated_combined.py pelvis   # thesis 3.3.4 as written
    python run_integrated_combined.py lean     # mode-switched joystick
    python run_integrated_combined.py keyboard # decoupled fallback

The overlay carries the evidence:

    TRAVELLED   distance the robot has moved from where it started. Under the
                pelvis strategy this climbs while the demonstrator stands
                still, which is the whole argument in one number.
    LOCO cmd    the live velocity command driving the legs.
    diagnostics strategy internals — pelvis visibility, trigger state, lean
                magnitudes — so the cause is visible alongside the effect.

An UNCOMMANDED banner appears whenever the robot is moving while the
demonstrator is not deliberately commanding motion. That is the moment worth
capturing on video.

Camera (click the window first):
  a/d rotate   w/s tilt   +/- zoom   r reset
  space  emergency stop and re-anchor
  q      quit
"""
import sys
import time

import cv2
import numpy as np
import mujoco
import torch

from g1_teleop.config import TeleopConfig
from g1_teleop.robot import G1Robot
from g1_teleop.teleop import TeleopController
from g1_teleop.zed_source import ZEDSource
from g1_teleop.overlay import draw_skeleton
from g1_teleop import config as C
from locomotion_input import PelvisVelocity, LeanJoystick, KeyboardCommand

# ── Paths (EDIT THESE) ────────────────────────────────────────────────────────
POLICY_PATH = r"D:\Charles_Aninon\Thesis Project\unitree_rl_gym\deploy\pre_train\g1\motion.pt"
SCENE_PATH = r"D:\Charles_Aninon\Thesis Project\g1_vision_teleop\scene.xml"

# ── Display ───────────────────────────────────────────────────────────────────
PANEL_H = 620
ZED_PANEL_W = 760
MJ_PANEL_W = 760
MUJOCO_W, MUJOCO_H = 960, 720      # offscreen render size (<= scene.xml offwidth)
WINDOW = "Vision locomotion evaluation — ZED | MuJoCo"

CAM_LOOKAT = [1.2, 0.0, 0.8]
CAM_DISTANCE = 4.5
CAM_AZIMUTH = 0.0
CAM_ELEVATION = -20.0

# ── Walking policy config ────────────────────────────────────────────────────
SIM_DT = 0.002
CONTROL_DECIMATION = 10
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS, NUM_OBS, GAIT_PERIOD = 12, 47, 0.8

IDLE_THRESHOLD, HOLD_KP, HOLD_MAX, HOLD_DEADBAND = 0.05, 0.8, 0.25, 0.02

# ── Grasp range ───────────────────────────────────────────────────────────────
# Measured arm reach is upper 0.200 + forearm 0.184 = 0.384 m from the shoulder,
# and the box sits about 0.20 m below shoulder height. That leaves roughly
# 0.33 m of horizontal reach, so the base has to stop within this band of the
# box or the arms cannot get to it.
GRASP_MIN, GRASP_MAX = 0.20, 0.34

LEG_QPOS, LEG_QVEL = slice(7, 19), slice(6, 18)
LEG_CTRL, ARM_CTRL = slice(0, 12), slice(12, 29)

UPPER_BODY_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])


def yaw_from_quat(quat):
    qw, qx, qy, qz = quat
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def world_to_body(v, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def fit_to_box(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    canvas = np.zeros((h, w, 3), dtype=img.dtype)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    return canvas


def text(img, s, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "pelvis"
    if which.startswith("p"):
        loco, loco_name = PelvisVelocity(), "PELVIS VELOCITY (thesis 3.3.4)"
    elif which.startswith("l"):
        loco, loco_name = LeanJoystick(), "LEAN JOYSTICK (mode-switched)"
    else:
        loco, loco_name = KeyboardCommand(), "KEYBOARD (decoupled)"

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[LEG_QPOS] = DEFAULT_ANGLES
    data.qvel[LEG_QVEL] = 0.0
    mujoco.mj_forward(model, data)
    start_xy = np.array(data.qpos[0:2], dtype=np.float64)

    cfg = TeleopConfig()
    twin = G1Robot(cfg)
    controller = TeleopController(twin, cfg)
    zed = ZEDSource(cfg.zed)

    twin_upper_qpos = []
    for name in UPPER_BODY_JOINTS:
        jid = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not found on twin: {name}")
        twin_upper_qpos.append(twin.model.jnt_qposadr[jid])

    policy = torch.jit.load(POLICY_PATH)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")

    try:
        renderer = mujoco.Renderer(model, height=MUJOCO_H, width=MUJOCO_W)
    except ValueError as e:
        print("Renderer init failed:", e)
        print("Increase <global offwidth/offheight> in scene.xml, or lower "
              "MUJOCO_W/MUJOCO_H here.")
        zed.close()
        return

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    scene_option = mujoco.MjvOption()

    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_leg_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)
    hold_target = np.array(data.qpos[0:2], dtype=np.float64)
    arm_targets = np.array(data.ctrl[ARM_CTRL], dtype=np.float32)
    counter = 0
    status_text, status_color = "waiting for ZED", (0, 100, 255)
    loco_diag = {}
    last_disp = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, ZED_PANEL_W + MJ_PANEL_W, PANEL_H)

    print(f"=== Vision locomotion evaluation ===")
    print(f"locomotion command source: {loco_name}")
    print("Click the window. a/d rotate, w/s tilt, +/- zoom, space stop, q quit.\n")

    try:
        while True:
            leg_q, leg_dq = data.qpos[LEG_QPOS], data.qvel[LEG_QVEL]
            data.ctrl[LEG_CTRL] = ((target_leg_pos - leg_q) * KPS
                                   + (0.0 - leg_dq) * KDS)
            data.ctrl[ARM_CTRL] = arm_targets
            mujoco.mj_step(model, data)
            counter += 1

            if counter % CONTROL_DECIMATION:
                continue

            # ── Velocity command: hold station when nothing is commanded ────
            if np.linalg.norm(cmd) < IDLE_THRESHOLD:
                err_world = hold_target - data.qpos[0:2]
                if np.linalg.norm(err_world) < HOLD_DEADBAND:
                    active_cmd = np.zeros(3, dtype=np.float32)
                else:
                    err_body = world_to_body(err_world, yaw_from_quat(data.qpos[3:7]))
                    active_cmd = np.zeros(3, dtype=np.float32)
                    active_cmd[0:2] = np.clip(HOLD_KP * err_body, -HOLD_MAX, HOLD_MAX)
            else:
                active_cmd = cmd.astype(np.float32)
                hold_target[:] = data.qpos[0:2]

            qj = (data.qpos[LEG_QPOS] - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj = data.qvel[LEG_QVEL] * DOF_VEL_SCALE
            t = counter * SIM_DT
            phase = (t % GAIT_PERIOD) / GAIT_PERIOD

            obs[:3] = data.qvel[3:6] * ANG_VEL_SCALE
            obs[3:6] = get_gravity_orientation(data.qpos[3:7])
            obs[6:9] = active_cmd * CMD_SCALE
            obs[9:9 + NUM_ACTIONS] = qj
            obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dqj
            obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = action
            obs[9 + 3 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS + 2] = [
                np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_leg_pos = action * ACTION_SCALE + DEFAULT_ANGLES

            # ── ZED: locomotion command + arm IK ────────────────────────────
            frame = zed.grab()

            if frame is not None and frame.keypoints_3d:
                outcome = controller.step(frame)
                status_text = ("IK OK" if outcome.applied
                               else f"HOLD: {outcome.reason.value}")
                status_color = (0, 255, 0) if outcome.applied else (0, 100, 255)
                arm_targets = np.array([twin.data.qpos[a] for a in twin_upper_qpos],
                                       dtype=np.float32)
                draw_skeleton(frame.image, frame.keypoints_2d,
                              frame.confidences, C.REQUIRED_KEYPOINTS)
            elif frame is not None:
                status_text, status_color = "no body detected", (0, 100, 255)

            # The command source is updated every tick regardless of tracking.
            # KeyboardCommand ignores keypoints entirely, so gating this on body
            # detection would make the arrow keys dead whenever you step out of
            # frame. Vision strategies handle missing keypoints internally.
            if isinstance(loco, KeyboardCommand):
                cmd[:] = loco()
            elif frame is not None and frame.keypoints_3d:
                cmd[:] = loco(frame.keypoints_3d)
                loco_diag = getattr(loco, "diag", {})

            if frame is None:
                # No new camera frame: still service the window so keyboard and
                # camera controls stay responsive between ZED frames.
                key_raw = cv2.waitKeyEx(1)
                if key_raw != -1 and (key_raw & 0xFF) == ord("q"):
                    break
                continue

            # ── Left panel: camera ──────────────────────────────────────────
            left = fit_to_box(frame.image, ZED_PANEL_W, PANEL_H)
            text(left, f"Status: {status_text}", (10, 28), 0.6, status_color)
            y = 56
            for k, v in loco_diag.items():
                if k == "strategy":
                    continue
                s = f"{k} = {v:+.4f}" if isinstance(v, float) else f"{k} = {v}"
                text(left, s, (10, y), 0.5, (200, 220, 255))
                y += 24

            # ── Right panel: simulation ─────────────────────────────────────
            renderer.update_scene(data, camera=cam, scene_option=scene_option)
            right = fit_to_box(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR),
                               MJ_PANEL_W, PANEL_H)

            travelled = float(np.linalg.norm(data.qpos[0:2] - start_xy))
            text(right, loco_name, (10, 28), 0.6, (255, 255, 0))
            text(right, f"TRAVELLED  {travelled:.2f} m", (10, 56), 0.7,
                 (0, 255, 255))
            text(right, f"LOCO cmd  fwd {cmd[0]:+.2f}  turn {cmd[2]:+.2f}",
                 (10, 84), 0.55)

            # ── Approach feedback ───────────────────────────────────────────
            # Distance from the robot's shoulders to the box, so the approach
            # can be judged from a number instead of timed by feel. The arm can
            # only reach about ARM_REACH, so this is the difference between a
            # graspable stop and one that is 10 cm short.
            box_xy = data.xpos[box_body_id][:2]
            base_xy = data.qpos[0:2]
            gap = float(np.linalg.norm(box_xy - base_xy))
            in_range = GRASP_MIN <= gap <= GRASP_MAX
            colour = (0, 255, 0) if in_range else (0, 200, 255)
            text(right, f"BOX DISTANCE  {gap:.2f} m", (10, 118), 0.7, colour)
            if in_range:
                text(right, "IN GRASP RANGE", (10, 150), 0.7, (0, 255, 0))
            elif gap > GRASP_MAX:
                text(right, f"too far  (walk {gap - GRASP_MAX:.2f} m closer)",
                     (10, 150), 0.55, (0, 200, 255))
            else:
                text(right, "too close  (back up)", (10, 150), 0.55, (0, 200, 255))

            if isinstance(loco, KeyboardCommand) and loco.precision:
                text(right, "PRECISION", (10, 182), 0.6, (255, 200, 0))

            # Flag motion that the demonstrator did not ask for.
            moving = last_disp is not None and abs(travelled - last_disp) > 0.004
            commanded = np.linalg.norm(cmd) > IDLE_THRESHOLD
            if moving and not commanded:
                text(right, "UNCOMMANDED MOTION", (10, 118), 0.75, (0, 80, 255))
            last_disp = travelled

            hint = "a/d rot  w/s tilt  +/- zoom  r reset  space stop  q quit"
            if isinstance(loco, KeyboardCommand):
                hint = ("arrows = walk/turn (hold)   p = precision   |   " + hint)
            text(right, hint, (10, PANEL_H - 14), 0.45, (200, 200, 200))

            cv2.imshow(WINDOW, np.hstack([left, right]))

            # waitKeyEx reports extended codes, so real arrow keys come through.
            # Plain waitKey masks them off, which is why arrows appeared dead.
            raw = cv2.waitKeyEx(1)
            key = raw & 0xFF if raw != -1 else 255

            # Arrow key codes differ by platform; accept the common ones plus
            # i/j/k/l as a guaranteed fallback.
            ARROW_UP = {2490368, 65362, 82}
            ARROW_DOWN = {2621440, 65364, 84}
            ARROW_LEFT = {2424832, 65361, 81}
            ARROW_RIGHT = {2555904, 65363, 83}

            if key == ord("q"):
                break
            elif key == ord("a"):
                cam.azimuth = (cam.azimuth - 5) % 360
            elif key == ord("d"):
                cam.azimuth = (cam.azimuth + 5) % 360
            elif key == ord("w"):
                cam.elevation = min(cam.elevation + 5, 89)
            elif key == ord("s"):
                cam.elevation = max(cam.elevation - 5, -89)
            elif key in (ord("+"), ord("=")):
                cam.distance = max(cam.distance - 0.3, 0.5)
            elif key in (ord("-"), ord("_")):
                cam.distance += 0.3
            elif key == ord("r"):
                cam.lookat[:] = CAM_LOOKAT
                cam.distance, cam.azimuth, cam.elevation = (
                    CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION)
            elif key == ord("p") and isinstance(loco, KeyboardCommand):
                loco.toggle_precision()
            elif key == 32:
                cmd[:] = 0.0
                hold_target[:] = data.qpos[0:2]
                if isinstance(loco, KeyboardCommand):
                    loco.cmd[:] = 0.0
                print("stopped and anchored")
            elif isinstance(loco, KeyboardCommand):
                # Walking: arrow keys or i/j/k/l.
                if raw in ARROW_UP or key == ord("i"):
                    loco.on_key(265)
                elif raw in ARROW_DOWN or key == ord("k"):
                    loco.on_key(264)
                elif raw in ARROW_LEFT or key == ord("j"):
                    loco.on_key(263)
                elif raw in ARROW_RIGHT or key == ord("l"):
                    loco.on_key(262)
                else:
                    continue
                print(f"cmd = forward {loco.cmd[0]:+.2f}  turn {loco.cmd[2]:+.2f}")
    finally:
        renderer.close()
        cv2.destroyAllWindows()
        zed.close()
        print("Done.")


if __name__ == "__main__":
    main()