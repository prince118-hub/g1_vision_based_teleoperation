"""Bare ZED tracking check — no MuJoCo, no gating, no IK.

Answers one question: is the ZED actually tracking a body, and are the arm
keypoints sane? Prints one line per frame with body count, confidence of the
key arm joints, and the measured upper-arm segment length in meters.

Run:  python zed_check.py    (Ctrl+C to stop)

What to look for:
  - "bodies=0"                -> camera sees no one. Tracking problem: check
                                 distance (2-3m), lighting, full torso in view.
  - conf values below ~40     -> weak tracking on those joints.
  - upper-arm length way off   -> keypoints jumping (should be ~0.20-0.35 m).
  - NaN                        -> ZED dropped that keypoint this frame.
"""
import numpy as np
import pyzed.sl as sl

LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 12, 14, 16

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.METER
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED open FAILED: {status}")
    raise SystemExit

bp = sl.BodyTrackingParameters()
bp.enable_tracking = True
bp.enable_body_fitting = True
bp.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
bp.body_format = sl.BODY_FORMAT.BODY_38
zed.enable_body_tracking(bp)

rt = sl.BodyTrackingRuntimeParameters()
rt.detection_confidence_threshold = 40

bodies = sl.Bodies()
print("Stand 2-3 m from the camera, full upper body visible. Ctrl+C to stop.\n")

try:
    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_bodies(bodies, rt)

        n = len(bodies.body_list) if bodies.body_list else 0
        if n == 0:
            print("bodies=0  (no one detected)")
            continue

        b = bodies.body_list[0]
        kp = b.keypoint
        conf = b.keypoint_confidence
        sh = np.array(kp[LEFT_SHOULDER], float)
        el = np.array(kp[LEFT_ELBOW], float)
        wr = np.array(kp[LEFT_WRIST], float)

        has_nan = any(np.any(np.isnan(x)) for x in (sh, el, wr))
        if has_nan:
            print(f"bodies={n}  NaN in L arm  conf(sh/el/wr)="
                  f"{conf[LEFT_SHOULDER]:.0f}/{conf[LEFT_ELBOW]:.0f}/{conf[LEFT_WRIST]:.0f}")
            continue

        upper = np.linalg.norm(el - sh)
        fore = np.linalg.norm(wr - el)
        print(f"bodies={n}  conf(sh/el/wr)="
              f"{conf[LEFT_SHOULDER]:.0f}/{conf[LEFT_ELBOW]:.0f}/{conf[LEFT_WRIST]:.0f}  "
              f"upper_arm={upper:.3f}m  forearm={fore:.3f}m")
except KeyboardInterrupt:
    pass
finally:
    zed.disable_body_tracking()
    zed.close()
    print("\nStopped.")