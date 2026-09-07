# G1 Vision-Based Teleoperation

State-based imitation learning pipeline for bimanual box pickup on the Unitree G1
humanoid robot in MuJoCo. This repository covers the demonstration-collection
front end: ZED stereo body tracking retargeted to the G1 arms via geometric
scaling and inverse kinematics, with torso-yaw following and tracking-validity
gating.

## Structure

```
g1_teleop/
├── run_teleop.py          # entry point: live ZED -> G1 teleoperation
└── g1_teleop/
    ├── config.py          # all tunable constants and dataclass configs
    ├── transforms.py      # camera -> robot frame rotation, torso yaw
    ├── gating.py          # tracking-validity gating (rejects bad frames)
    ├── retargeting.py     # geometric scaling: human keypoints -> arm targets
    ├── ik.py              # damped least-squares inverse kinematics
    ├── robot.py           # MuJoCo model wrapper, box reset, waist control
    ├── zed_source.py      # ZED SDK body-tracking wrapper
    ├── overlay.py         # OpenCV debug overlay
    └── teleop.py          # per-frame controller tying it all together
```

Separation of concerns keeps the hardware dependency (`zed_source`) and the
simulator dependency (`robot`) isolated, so the retargeting, gating, and
transform logic can be tested without either the ZED SDK or a display.

## Requirements

- Python 3.10+
- MuJoCo 3.7.0, NumPy, OpenCV (`pip install -r requirements.txt`)
- Stereolabs ZED SDK with the `pyzed` Python bindings (installed separately)
- `scene.xml` and `g1.xml` (Unitree G1 from MuJoCo Menagerie) — set the path in
  `config.py` (`MODEL_PATH`)

## Usage

```bash
python run_teleop.py
```

Move your arms in front of the ZED and the G1 mirrors them. Turn your torso and
the waist follows. When tracking is unreliable (self-occlusion, sideways/back
pose, low confidence, implausible limb lengths) the robot freezes at its last
good pose instead of following corrupted data.

## Configuration

Everything tunable lives in `config.py`. Common adjustments:

- `TorsoYawConfig.sign` — flip to `-1.0` if turning left makes the robot turn right
- `GatingConfig.max_facing_yaw` — how far the demonstrator can turn before frames are rejected
- `SmoothingConfig.arm_alpha` / `yaw_alpha` — tracking responsiveness vs. jitter
- `BoxConfig.pickup_center` — must match `platform_pickup` in `scene.xml`

## Status

Demonstration collection front end. Locomotion trigger, physics-stepped grasping,
and demonstration logging are planned next.
