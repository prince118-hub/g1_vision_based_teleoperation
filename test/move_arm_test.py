import time
import math
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()

    while viewer.is_running():

        t = time.time() - start_time

        # Left leg
        data.ctrl[0] = 0.5 * math.sin(t * 2)      # left hip pitch
        data.ctrl[3] = 0.8 * math.sin(t * 2)      # left knee

        # Right leg (opposite phase)
        data.ctrl[6] = -0.5 * math.sin(t * 2)     # right hip pitch
        data.ctrl[9] = -0.8 * math.sin(t * 2)     # right knee

        # Arm swing
        data.ctrl[15] = -0.4 * math.sin(t * 2)    # left shoulder
        data.ctrl[22] =  0.4 * math.sin(t * 2)    # right shoulder

        mujoco.mj_step(model, data)
        viewer.sync()