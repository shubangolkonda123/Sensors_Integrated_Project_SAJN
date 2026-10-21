import time
from pathlib import Path

import mujoco
import mujoco.viewer


model_path = (
    Path(__file__).parent
    / "universal_robots_ur3e"
    / "ur3e.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Load the saved home configuration and actuator targets.
home_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)
if home_id == -1:
    raise ValueError("The XML is missing the 'home' keyframe.")

mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

# Preserve the arm's six home targets.
arm_targets = data.ctrl[:6].copy()

previous_phase = None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.perf_counter()

        # Repeat: open, close, open, close...
        # Each phase lasts two seconds of simulation time.
        phase = int(data.time / 2.0) % 2

        if phase == 0:
            finger_target = 0.03
            status = "OPEN"
        else:
            finger_target = 0.0
            status = "CLOSE"

        data.ctrl[:6] = arm_targets
        data.actuator("left_gripper").ctrl[0] = finger_target
        data.actuator("right_gripper").ctrl[0] = finger_target

        if phase != previous_phase:
            print(f"Gripper: {status}")
            previous_phase = phase

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = (
            model.opt.timestep
            - (time.perf_counter() - step_start)
        )
        if remaining > 0:
            time.sleep(remaining)