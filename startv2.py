import time
from pathlib import Path

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

MODEL_PATH = (
    Path(__file__).parent
    / "universal_robots_ur3e"
    / "ur3e.xml"
)

CAMERA_NAME = "wrist_camera"
WIDTH = 320
HEIGHT = 240
CAMERA_PERIOD = 1.0 / 15.0

# Display/sensor working range for this tabletop prototype.
MIN_DEPTH = 0.02
MAX_DEPTH = 2.0


class MuJoCoRGBD:
    def __init__(self, model, camera_name, width, height):
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(
            model,
            height=height,
            width=width,
        )

    def capture(self, data):
        # RGB and depth are captured from the same simulation state.
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(
            data,
            camera=self.camera_name,
        )
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(
            data,
            camera=self.camera_name,
        )
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        return rgb, depth

    def close(self):
        self.renderer.close()


def make_depth_display(depth):
    valid = (
        np.isfinite(depth)
        & (depth > MIN_DEPTH)
        & (depth < MAX_DEPTH)
    )

    # Fixed scale: nearby surfaces appear brighter.
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = (
        255.0
        * (
            1.0
            - (depth[valid] - MIN_DEPTH)
            / (MAX_DEPTH - MIN_DEPTH)
        )
    ).astype(np.uint8)

    coloured = cv2.applyColorMap(
        scaled,
        cv2.COLORMAP_TURBO,
    )
    coloured[~valid] = 0

    return coloured, valid

def rgbd_to_cloud(rgb, depth, K):
    height, width = depth.shape

    uu, vv = np.meshgrid(
        np.arange(width),
        np.arange(height),
    )

    good = (
        np.isfinite(depth)
        & (depth > MIN_DEPTH)
        & (depth < MAX_DEPTH)
    )

    uv1 = np.stack([
        uu[good],
        vv[good],
        np.ones(np.count_nonzero(good)),
    ])

    dvalid = depth[good]

    # Same back-projection equation as Week 4.
    PC = (np.linalg.inv(K) @ uv1) * dvalid

    # Keep matching colours for each 3D point.
    colours = rgb[good].astype(np.float64) / 255.0

    # Return points as N x 3.
    return PC.T, colours
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)
if home_id == -1:
    raise ValueError("Missing 'home' keyframe.")

camera_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    CAMERA_NAME,
)
if camera_id == -1:
    raise ValueError(f"Missing camera: {CAMERA_NAME}")

mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

camera = MuJoCoRGBD(
    model,
    CAMERA_NAME,
    WIDTH,
    HEIGHT,
)

next_capture = 0.0
next_report = 0.0

print("Manual joint control remains enabled.")
print("RGB: colour image.")
print("Depth: coloured display of metric depth.")
print("Press Q in an OpenCV window to exit.")
# Ideal pinhole intrinsics for your current fovy-only XML camera.
fovy = np.deg2rad(model.cam_fovy[camera_id])

fy = (HEIGHT / 2.0) / np.tan(fovy / 2.0)
fx = fy

# Pixel-centre convention for the rendered image.
cx = (WIDTH - 1) / 2.0
cy = (HEIGHT - 1) / 2.0

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
])

print("Camera matrix K:")
print(K)
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

        while viewer.is_running():
            step_start = time.perf_counter()

            # No scripted actuator commands.
            # The viewer sliders remain in control.
            mujoco.mj_step(model, data)

            if data.time >= next_capture:
                mujoco.mj_forward(model, data)
                rgb, depth = camera.capture(data)
                points_camera, cloud_colours = rgbd_to_cloud(
                    rgb,
                    depth,
                    K,
                )

                depth_display, valid = make_depth_display(depth)

                # OpenCV displays BGR, while MuJoCo returns RGB.
                rgb_display = cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR,
                )

                cv2.imshow("Wrist RGB", rgb_display)
                cv2.imshow(
                    "Wrist depth — black means outside range",
                    depth_display,
                )

                if data.time >= next_report:
                    if np.any(valid):
                        print(
                            f"RGB {rgb.shape} | "
                            f"depth {depth.shape} | "
                            f"valid depth: "
                            f"{depth[valid].min():.3f}–"
                            f"{depth[valid].max():.3f} m"
                        )
                    else:
                        print(
                            "No depth within the selected "
                            f"{MIN_DEPTH}–{MAX_DEPTH} m range."
                        )
                    print("Point cloud:", points_camera.shape)
                    next_report = data.time + 1.0
                    
                next_capture = data.time + CAMERA_PERIOD

            viewer.sync()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            remaining = (
                model.opt.timestep
                - (time.perf_counter() - step_start)
            )
            if remaining > 0:
                time.sleep(remaining)

finally:
    camera.close()
    cv2.destroyAllWindows()
# Display the last captured cloud after the simulation closes.
if "points_camera" in globals() and len(points_camera) > 0:
    # Subsample for faster interactive plotting.
    points = points_camera[::3]
    colours = cloud_colours[::3]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Display optical coordinates as:
    # horizontal = X right
    # depth      = Z forward
    # vertical   = -Y up
    displayed = np.column_stack([
        points[:, 0],
        points[:, 2],
        -points[:, 1],
    ])

    ax.scatter(
        displayed[:, 0],
        displayed[:, 1],
        displayed[:, 2],
        c=colours,
        s=1,
        linewidths=0,
    )

    ax.set_xlabel("X right (m)")
    ax.set_ylabel("Z forward (m)")
    ax.set_zlabel("-Y up (m)")
    ax.set_title("Wrist RGB-D cloud — camera coordinates")

    ax.set_box_aspect(
        np.maximum(np.ptp(displayed, axis=0), 0.001)
    )

    plt.tight_layout()
    plt.show()
else:
    print("No point cloud was captured.")