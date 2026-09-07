"""Entry point: live ZED-driven teleoperation of the G1 arms in MuJoCo.

Move your arms in front of the ZED and the G1 mirrors them. When tracking is
unreliable the robot holds its last good pose. The ZED tracking window and the
MuJoCo viewer are auto-positioned side by side on startup. Press Q to quit.
"""
from __future__ import annotations

import time

import cv2
import mujoco.viewer

from g1_teleop.config import TeleopConfig
from g1_teleop.robot import G1Robot
from g1_teleop.teleop import TeleopController
from g1_teleop.zed_source import ZEDSource
from g1_teleop.overlay import draw_skeleton, draw_status
from g1_teleop import config as C

# ── Side-by-side layout (pixels). ZED on the left half, MuJoCo on the right. ──
WINDOW_W = 900
WINDOW_H = 700
ZED_X, ZED_Y = 0, 0
MUJOCO_X, MUJOCO_Y = WINDOW_W + 8, 0
ZED_WINDOW_NAME = "ZED Body Tracking"


def _position_zed_window() -> None:
    cv2.namedWindow(ZED_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ZED_WINDOW_NAME, WINDOW_W, WINDOW_H)
    cv2.moveWindow(ZED_WINDOW_NAME, ZED_X, ZED_Y)


def _position_mujoco_window() -> None:
    """Move the MuJoCo viewer window to the right half. Windows-only; silently
    skipped on other platforms or if the window isn't found. Matches the window
    by a substring of its title, so it works regardless of the exact title
    format (e.g. 'MuJoCo : g1_29dof_rev_1_0 scene')."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found = []

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _cb(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if "MuJoCo" in title or "mujoco" in title:
                found.append(hwnd)
            return True

        user32.EnumWindows(EnumWindowsProc(_cb), 0)
        for hwnd in found:
            # SetWindowPos(hwnd, HWND_TOP=0, x, y, w, h, SWP_SHOWWINDOW=0x40)
            user32.SetWindowPos(hwnd, 0, MUJOCO_X, MUJOCO_Y,
                                WINDOW_W, WINDOW_H, 0x0040)
    except Exception:
        pass  # non-Windows or lookup failed — leave the window where it is


def main() -> None:
    cfg = TeleopConfig()
    robot = G1Robot(cfg)
    controller = TeleopController(robot, cfg, debug_gate=True)
    zed = ZEDSource(cfg.zed)

    print("=== Link lengths (m) ===")
    print(f"  L upper/fore: {robot.upper_arm_left:.3f} / {robot.forearm_left:.3f}")
    print(f"  R upper/fore: {robot.upper_arm_right:.3f} / {robot.forearm_right:.3f}")
    print("Move your arms — G1 mirrors. Q in the ZED window to quit.\n")

    _position_zed_window()

    try:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            # Set the camera explicitly on the viewer. The XML <global> camera is
            # not reliably applied by launch_passive, so we configure it here:
            # behind the robot (looking down +x toward the platforms), tilted
            # slightly down. azimuth 180 looks toward +x; elevation -20 tilts down.
            viewer.cam.lookat[:] = [1.2, 0.0, 0.8]   # point of interest (scene center)
            viewer.cam.distance = 4.0                # how far back the camera sits
            viewer.cam.azimuth = 0                 # behind robot, facing the boxes
            viewer.cam.elevation = -20               # slight downward tilt
            viewer.sync()

            # The viewer window can take a moment to appear; retry positioning.
            _positioned = False
            _start = time.time()

            while viewer.is_running():
                # Keep trying to place the MuJoCo window for the first ~3s.
                if not _positioned and time.time() - _start < 3.0:
                    _position_mujoco_window()
                    _position_zed_window()
                elif not _positioned:
                    _positioned = True

                frame = zed.grab()
                if frame is None:
                    viewer.sync()
                    continue

                outcome = controller.step(frame)
                status = ("IK OK" if outcome.applied else f"FROZEN: {outcome.reason.value}")
                color = (0, 255, 0) if outcome.applied else (0, 100, 255)

                draw_skeleton(frame.image, frame.keypoints_2d,
                              frame.confidences, C.REQUIRED_KEYPOINTS)
                draw_status(frame.image, [(f"Status: {status}", color)])

                cv2.imshow(ZED_WINDOW_NAME, frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                viewer.sync()
    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("Done.")


if __name__ == "__main__":
    main()