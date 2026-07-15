"""Entry point: live ZED-driven teleoperation of the G1 arms in MuJoCo.

Move your arms in front of the ZED and the G1 mirrors them. Turn your torso
left/right and the waist follows. When tracking is unreliable (occlusion,
sideways/back pose, low confidence) the robot freezes rather than following
corrupted data. Press Q in the camera window to quit.
"""
from __future__ import annotations

import cv2
import mujoco.viewer

from g1_teleop.config import TeleopConfig
from g1_teleop.robot import G1Robot
from g1_teleop.teleop import TeleopController
from g1_teleop.zed_source import ZEDSource
from g1_teleop.overlay import draw_skeleton, draw_status
from g1_teleop import config as C


def main() -> None:
    cfg = TeleopConfig()
    robot = G1Robot(cfg)
    controller = TeleopController(robot, cfg, debug_gate=True)
    zed = ZEDSource(cfg.zed)

    print("=== teleop build: independent-arms + shoulder-snapshot (no averaging) ===")
    print("=== Link lengths (m) ===")
    print(f"  L upper/fore: {robot.upper_arm_left:.3f} / {robot.forearm_left:.3f}")
    print(f"  R upper/fore: {robot.upper_arm_right:.3f} / {robot.forearm_right:.3f}")
    print("  (if L and R lengths are IDENTICAL you are running an old averaged file)")
    print("Move your arms — G1 mirrors. Q in camera window to quit.\n")

    try:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            while viewer.is_running():
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

                cv2.imshow("ZED Body Tracking", frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                viewer.sync()
    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("Done.")


if __name__ == "__main__":
    main()