"""Combined single-window teleoperation view.

Renders the MuJoCo scene offscreen and pastes it beside the ZED camera feed in
ONE OpenCV window — no MuJoCo UI panels, a clean side-by-side monitor. The
MuJoCo camera is fixed (set in code below), so there is no mouse camera control
here; for interactive inspection (rotate/zoom/click) run run_teleop.py instead.

Press Q to quit.
"""
from __future__ import annotations

import cv2
import numpy as np
import mujoco

from g1_teleop.config import TeleopConfig
from g1_teleop.robot import G1Robot
from g1_teleop.teleop import TeleopController
from g1_teleop.zed_source import ZEDSource
from g1_teleop.overlay import draw_skeleton, draw_status
from g1_teleop import config as C

# ── Layout / render settings ──────────────────────────────────────────────────
# Each panel has its own target width and a shared display height. Adjust these
# to change the default split, or use [ and ] keys at runtime to shift it live.
PANEL_H = 600                  # display height of both panels (px)
ZED_PANEL_W = 800              # left (ZED) panel target width (px)
MUJOCO_PANEL_W = 720           # right (MuJoCo) panel target width (px)
PANEL_W_STEP = 40              # px change per [ / ] key press
PANEL_W_MIN = 200              # don't let a panel shrink below this

# Offscreen RENDER resolution. Higher = sharper but slower. 960x720 with the
# anti-aliasing in scene.xml hits ~30fps on most GPUs while staying crisp. Raise
# toward 1200x900 only if your fps counter stays near 30.
MUJOCO_W, MUJOCO_H = 960, 720
WINDOW_NAME = "Teleop — ZED (left)  |  MuJoCo (right)"

# ── Fixed MuJoCo camera (behind the robot, looking forward). Same values we
#    tuned in the interactive viewer. Adjust here to reframe. ──────────────────
CAM_LOOKAT = [1.2, 0.0, 0.8]
CAM_DISTANCE = 4.0
CAM_AZIMUTH = 0.0
CAM_ELEVATION = -20.0

# ── Keyboard camera control steps ─────────────────────────────────────────────
AZIMUTH_STEP = 5.0     # degrees per key press
ELEVATION_STEP = 5.0   # degrees per key press
ZOOM_STEP = 0.3        # meters per key press
PAN_STEP = 0.15        # meters per key press (lookat shift)


def _make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    return cam


def _handle_camera_key(key: int, cam: mujoco.MjvCamera) -> bool:
    """Adjust the camera from a keypress. Returns True if the key was a camera
    control (so the caller knows it was handled). Keys:
      a / d : rotate left / right (azimuth)
      w / s : tilt up / down (elevation)
      + / - : zoom in / out (distance)
      arrow keys or i/j/k/l : pan the look-at point
      r     : reset to the default view
    """
    if key == ord('a'):
        cam.azimuth = (cam.azimuth - AZIMUTH_STEP) % 360
    elif key == ord('d'):
        cam.azimuth = (cam.azimuth + AZIMUTH_STEP) % 360
    elif key == ord('w'):
        cam.elevation = min(cam.elevation + ELEVATION_STEP, 89.0)
    elif key == ord('s'):
        cam.elevation = max(cam.elevation - ELEVATION_STEP, -89.0)
    elif key in (ord('+'), ord('=')):
        cam.distance = max(cam.distance - ZOOM_STEP, 0.5)
    elif key in (ord('-'), ord('_')):
        cam.distance = cam.distance + ZOOM_STEP
    elif key == ord('l'):
        cam.lookat[1] += PAN_STEP
    elif key == ord('j'):
        cam.lookat[1] -= PAN_STEP
    elif key == ord('i'):
        cam.lookat[2] += PAN_STEP
    elif key == ord('k'):
        cam.lookat[2] -= PAN_STEP
    elif key == ord('r'):
        cam.lookat[:] = CAM_LOOKAT
        cam.distance = CAM_DISTANCE
        cam.azimuth = CAM_AZIMUTH
        cam.elevation = CAM_ELEVATION
    else:
        return False
    return True


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def _draw_text(img, text, org, scale=0.55, color=(255, 255, 255)):
    """Draw text with a dark outline so it stays readable over bright or busy
    backgrounds (e.g. the light MuJoCo floor)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)          # outline
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)              # fill


def _fit_to_box(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize img to fill a width x height box, preserving aspect ratio and
    letterboxing (black bars) as needed. Lets each panel have an independent
    size without distorting the image."""
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((height, width, 3), dtype=img.dtype)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main() -> None:
    cfg = TeleopConfig()
    robot = G1Robot(cfg)
    controller = TeleopController(robot, cfg)
    zed = ZEDSource(cfg.zed)

    # Offscreen MuJoCo renderer. If this fails on framebuffer size, enlarge
    # <global offwidth=.. offheight=..> in scene.xml (must exceed MUJOCO_W/H).
    try:
        renderer = mujoco.Renderer(robot.model, height=MUJOCO_H, width=MUJOCO_W)
    except ValueError as e:
        print("Renderer init failed:", e)
        print("Fix: add offwidth/offheight to <global> in scene.xml, or lower "
              "MUJOCO_W/MUJOCO_H at the top of this file.")
        zed.close()
        return
    cam = _make_camera()
    scene_option = mujoco.MjvOption()

    print("=== Combined single-window teleop ===")
    print(f"  L upper/fore: {robot.upper_arm_left:.3f} / {robot.forearm_left:.3f}")
    print(f"  R upper/fore: {robot.upper_arm_right:.3f} / {robot.forearm_right:.3f}")
    print("Move your arms — G1 mirrors.")
    print("Camera:  a/d rotate  w/s tilt  +/- zoom  i/j/k/l pan  r reset")
    print("Layout:  [ / ] shrink/grow ZED panel   ; / ' shrink/grow MuJoCo panel")
    print("         , / . shorter/taller both      q quit\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cam_hint = "a/d rot  w/s tilt  +/- zoom  i/j/k/l pan  r reset"
    layout_hint = "[ ] ZED-w   ; ' MuJoCo-w   , . height   q quit"

    # Mutable panel sizes (start from the module defaults).
    zed_w = ZED_PANEL_W
    mj_w = MUJOCO_PANEL_W
    panel_h = PANEL_H

    import time
    _t_prev = time.time()
    _fps = 0.0
    _profile_t = time.time()   # for periodic timing printout

    try:
        while True:
            _t0 = time.time()
            frame = zed.grab()
            _t_grab = time.time()
            if frame is None:
                continue

            # FPS (exponential moving average).
            _now = time.time()
            dt = _now - _t_prev
            _t_prev = _now
            if dt > 0:
                _fps = 0.9 * _fps + 0.1 * (1.0 / dt)

            outcome = controller.step(frame)
            _t_step = time.time()
            status = ("IK OK" if outcome.applied else f"HOLD: {outcome.reason.value}")
            color = (0, 255, 0) if outcome.applied else (0, 100, 255)

            # ── Left panel: ZED feed with skeleton overlay ──────────────────
            draw_skeleton(frame.image, frame.keypoints_2d,
                          frame.confidences, C.REQUIRED_KEYPOINTS)
            draw_status(frame.image, [(f"Status: {status}", color)])
            left = _fit_to_box(frame.image, zed_w, panel_h)

            # ── Right panel: offscreen MuJoCo render ────────────────────────
            renderer.update_scene(robot.data, camera=cam, scene_option=scene_option)
            mj_rgb = renderer.render()                       # RGB uint8
            _t_render = time.time()
            mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)
            right = _fit_to_box(mj_bgr, mj_w, panel_h)

            # Per-stage timing printout ~once a second, to find the bottleneck.
            if time.time() - _profile_t > 1.0:
                _profile_t = time.time()
                print(f"[timing] grab={1000*(_t_grab-_t0):5.1f}ms  "
                      f"teleop={1000*(_t_step-_t_grab):5.1f}ms  "
                      f"render={1000*(_t_render-_t_step):5.1f}ms  "
                      f"fps={_fps:.1f}")

            # fps + hints on the sim panel (outlined = readable on any bg)
            _draw_text(right, f"{_fps:4.1f} fps", (10, 24), 0.6, (0, 255, 255))
            _draw_text(right, cam_hint, (10, right.shape[0] - 40), 0.5)
            _draw_text(right, layout_hint, (10, right.shape[0] - 16), 0.5,
                       color=(200, 220, 255))

            # ── Composite side by side ──────────────────────────────────────
            combined = np.hstack([left, right])
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Layout keys
            if key == ord('['):
                zed_w = max(PANEL_W_MIN, zed_w - PANEL_W_STEP)
            elif key == ord(']'):
                zed_w = zed_w + PANEL_W_STEP
            elif key == ord(';'):
                mj_w = max(PANEL_W_MIN, mj_w - PANEL_W_STEP)
            elif key == ord("'"):
                mj_w = mj_w + PANEL_W_STEP
            elif key == ord(','):
                panel_h = max(PANEL_W_MIN, panel_h - PANEL_W_STEP)
            elif key == ord('.'):
                panel_h = panel_h + PANEL_W_STEP
            elif key != 255:
                _handle_camera_key(key, cam)
    finally:
        renderer.close()
        cv2.destroyAllWindows()
        zed.close()
        print("Done.")


if __name__ == "__main__":
    main()