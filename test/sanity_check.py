import pyzed.sl as sl
import numpy as np

# --- ZED SETUP ---
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER  # important: ensure metric output

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to open ZED: {err}")
    exit()

# --- BODY TRACKING SETUP ---
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

# --- KEYPOINT INDEX FOR BODY_38 ---
LEFT_WRIST = 15   # verify this index against your SDK version

print("Starting live wrist tracking. Press Ctrl+C to stop.")
print(f"{'Frame':<8} {'X (left/right)':<20} {'Y (up/down)':<20} {'Z (depth)':<20} {'Confidence':<12}")
print("-" * 80)

frame = 0
try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies, body_runtime_params)

            if bodies.is_new and len(bodies.body_list) > 0:
                # take the first detected person
                body = bodies.body_list[0]
                keypoints = body.keypoint
                confidence = body.keypoint_confidence

                left_wrist = keypoints[LEFT_WRIST]
                conf = confidence[LEFT_WRIST]

                x, y, z = left_wrist

                # flag suspicious values
                metric_ok = 0.1 < abs(z) < 5.0   # Z should be 1.5-2m in your setup
                noise_flag = "⚠ CHECK" if not metric_ok else "OK"

                print(
                    f"{frame:<8} "
                    f"X={x:+.3f}m{'':<10} "
                    f"Y={y:+.3f}m{'':<10} "
                    f"Z={z:+.3f}m{'':<10} "
                    f"conf={conf:.1f}% {noise_flag}"
                )

                frame += 1

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    zed.disable_body_tracking()
    zed.close()