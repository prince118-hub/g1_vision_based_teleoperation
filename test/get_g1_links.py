import mujoco
import numpy as np

# load your G1 model - adjust path to wherever your xml is
model = mujoco.MjModel.from_xml_path(
    r"D:\Charles_Aninon\Thesis Project\g1_29dof_rev_1_0.xml"
)
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)

# get shoulder, elbow, wrist positions in neutral pose
# adjust body names to match your MJCF
bodies_to_check = [
    "left_shoulder_pitch_link",
    "left_elbow_link", 
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]

print("G1 body positions in neutral pose:")
for name in bodies_to_check:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[body_id]
        print(f"  {name}: {pos}")
    except:
        print(f"  {name}: NOT FOUND - check name")