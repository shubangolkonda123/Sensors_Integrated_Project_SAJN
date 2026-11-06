import time
from pathlib import Path

import mujoco
import mujoco.viewer
import roboticstoolbox as rtb
robot = rtb.models.DH.UR3()




model_path = (
    Path(__file__).parent
    / "universal_robots_ur3e"
    / "ur3e.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Set the starting pose once, without overriding the sliders afterward.
home_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)

if home_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, home_id)

mujoco.mj_forward(model, data)

camera_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "wrist_camera",
)

renderer = mujoco.Renderer(model, height=240, width=320)
next_capture_time = 0.0
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id

    while viewer.is_running():
        step_start = time.perf_counter()

        # Leave actuator commands to the viewer's control sliders.
        mujoco.mj_step(model, data)
        # Capture an RGB image once per second of simulation time.
        if data.time >= next_capture_time:
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="wrist_camera")
            rgb = renderer.render().copy()

            print(
                "Camera image:",
                rgb.shape,
                "pixel range:",
                int(rgb.min()),
                "to",
                int(rgb.max()),
            )

            next_capture_time = data.time + 1.0
        viewer.sync()

        remaining = (
            model.opt.timestep
            - (time.perf_counter() - step_start)
        )

        if remaining > 0:
            time.sleep(remaining)