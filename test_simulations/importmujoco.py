import mujoco
import mujoco.viewer
from pathlib import Path

# Path to the UR3e MJCF model
model_path = Path(__file__).parent / "universal_robots_ur3e" / "ur3e.xml"

# Load the MuJoCo model and create simulation data
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Launch passive viewer and run simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()