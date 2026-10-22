import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import mujoco
import mujoco.viewer


# ================= SETTINGS =================

MODEL_PATH = (
    Path(__file__).parent
    / "universal_robots_ur3e"
    / "ur3e.xml"
)

CAMERA_NAME = "wrist_camera"
WIDTH = 320
HEIGHT = 240

# Perception updates at up to 5 Hz of simulation time.
PERCEPTION_PERIOD = 0.2

MIN_DEPTH = 0.02
MAX_DEPTH = 2.0

# Excludes nearby gripper geometry for your current observation pose.
WORKSPACE_MIN_DEPTH = 0.20

VOXEL_SIZE = 0.003
PLANE_THRESHOLD = 0.003

# Candidate target surfaces must lie this far toward the camera
# from the fitted table plane.
MIN_OBJECT_HEIGHT = 0.006
MAX_OBJECT_HEIGHT = 0.08

MIN_TARGET_PIXELS = 12


# ================= RGB-D CAMERA =================

class MuJoCoRGBD:
    def __init__(self, model, name, width, height):
        self.name = name
        self.renderer = mujoco.Renderer(
            model,
            height=height,
            width=width,
        )

    def capture(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        depth = self.renderer.render().copy()

        self.renderer.disable_depth_rendering()
        return rgb, depth

    def close(self):
        self.renderer.close()


# ================= PERCEPTION =================

class TabletopPerception:
    def __init__(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)

    def backproject(self, depth):
        height, width = depth.shape

        u, v = np.meshgrid(
            np.arange(width),
            np.arange(height),
        )

        pixels = np.stack([
            u,
            v,
            np.ones_like(u),
        ], axis=-1)

        rays = pixels @ self.K_inv.T

        # Organized H x W x 3 cloud in optical coordinates:
        # X right, Y down, Z forward.
        return rays * depth[..., None]

    def process(self, rgb, depth):
        xyz = self.backproject(depth)

        valid = (
            np.isfinite(depth)
            & (depth > MIN_DEPTH)
            & (depth < MAX_DEPTH)
        )

        workspace = valid & (depth > WORKSPACE_MIN_DEPTH)

        result = {
            "status": "Insufficient workspace points",
            "plane": None,
            "table_mask": np.zeros(depth.shape, dtype=bool),
            "target_mask": np.zeros(depth.shape, dtype=bool),
            "surface_position": None,
            "bbox": None,
            "n_points": int(np.count_nonzero(valid)),
            "plane_fraction": None,
        }

        points = xyz[workspace]

        if len(points) < 100:
            return result

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(VOXEL_SIZE)

        if len(cloud.points) < 50:
            return result

        plane, inliers = cloud.segment_plane(
            distance_threshold=PLANE_THRESHOLD,
            ransac_n=3,
            num_iterations=500,
        )

        plane = np.asarray(plane, dtype=float)
        plane /= np.linalg.norm(plane[:3])

        # Point the normal toward the camera origin.
        # For a visible tabletop this makes the object side positive.
        if plane[3] < 0:
            plane = -plane

        fraction = len(inliers) / len(cloud.points)
        result["plane_fraction"] = fraction

        if fraction < 0.35:
            result["status"] = "No dominant table-plane candidate"
            return result

        result["plane"] = plane

        signed_distance = xyz @ plane[:3] + plane[3]

        table_mask = (
            workspace
            & (np.abs(signed_distance) < PLANE_THRESHOLD)
        )

        above_table = (
            workspace
            & (signed_distance > MIN_OBJECT_HEIGHT)
            & (signed_distance < MAX_OBJECT_HEIGHT)
        )

        result["table_mask"] = table_mask

        # Red occupies both ends of OpenCV's hue interval.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        red_low = cv2.inRange(
            hsv,
            np.array([0, 80, 40], dtype=np.uint8),
            np.array([12, 255, 255], dtype=np.uint8),
        )

        red_high = cv2.inRange(
            hsv,
            np.array([168, 80, 40], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )

        red = (red_low | red_high) > 0
        candidate = (red & above_table).astype(np.uint8)

        # Select the largest connected red region above the plane.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )

        if count <= 1:
            result["status"] = "Table found; no red target above it"
            return result

        label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_TARGET_PIXELS:
            result["status"] = "Red target region too small"
            return result

        target_mask = labels == label
        target_points = xyz[target_mask]

        # This is a robust visible-surface location,
        # NOT a reconstructed object centre.
        surface_position = np.median(target_points, axis=0)

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        result.update({
            "status": "Target detected",
            "target_mask": target_mask,
            "surface_position": surface_position,
            "bbox": (x, y, w, h),
        })

        return result


# ================= DISPLAYS =================

def depth_display(depth):
    valid = (
        np.isfinite(depth)
        & (depth > MIN_DEPTH)
        & (depth < MAX_DEPTH)
    )

    scaled = np.zeros(depth.shape, dtype=np.uint8)

    scaled[valid] = (
        255
        * (
            1
            - (depth[valid] - MIN_DEPTH)
            / (MAX_DEPTH - MIN_DEPTH)
        )
    ).astype(np.uint8)

    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def detection_display(rgb, result):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    target = result["target_mask"]
    if np.any(target):
        image[target] = (
            0.5 * image[target]
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

    if result["bbox"] is not None:
        x, y, w, h = result["bbox"]
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1,
        )

    cv2.putText(
        image,
        result["status"],
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    position = result["surface_position"]

    if position is not None:
        text = (
            f"Camera XYZ: {position[0]:+.3f}, "
            f"{position[1]:+.3f}, {position[2]:.3f} m"
        )

        cv2.putText(
            image,
            text,
            (5, HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image


# ================= SIMULATION =================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "home"
)

camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
)

if home_id == -1:
    raise ValueError("Missing 'home' keyframe.")

if camera_id == -1:
    raise ValueError(f"Missing '{CAMERA_NAME}' camera.")

mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

# Ideal intrinsics for your existing fovy-only XML camera.
fovy = np.deg2rad(model.cam_fovy[camera_id])
focal_pixels = (HEIGHT / 2) / np.tan(fovy / 2)

K = np.array([
    [focal_pixels, 0, (WIDTH - 1) / 2],
    [0, focal_pixels, (HEIGHT - 1) / 2],
    [0, 0, 1],
], dtype=float)

camera = MuJoCoRGBD(model, CAMERA_NAME, WIDTH, HEIGHT)
perception = TabletopPerception(K)

next_capture = 0.0
next_report = 0.0

print("Camera matrix K:\n", K)
print("Manual sliders remain enabled.")
print("Press Q in an image window to quit.")
print("Reported XYZ is in optical camera coordinates.")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

        while viewer.is_running():
            start = time.perf_counter()

            # Actuator inputs remain controlled by viewer sliders.
            mujoco.mj_step(model, data)

            if data.time >= next_capture:
                mujoco.mj_forward(model, data)

                rgb, depth = camera.capture(data)
                result = perception.process(rgb, depth)

                cv2.imshow(
                    "RGB-D target detection",
                    detection_display(rgb, result),
                )

                cv2.imshow(
                    "Metric depth",
                    depth_display(depth),
                )

                masks = np.zeros(
                    (HEIGHT, WIDTH, 3),
                    dtype=np.uint8,
                )

                masks[result["table_mask"]] = [160, 160, 160]
                masks[result["target_mask"]] = [0, 255, 0]

                cv2.imshow(
                    "Segmentation: grey table, green target",
                    masks,
                )

                if data.time >= next_report:
                    print(
                        f"{result['status']} | "
                        f"cloud points: {result['n_points']}"
                    )

                    if result["surface_position"] is not None:
                        print(
                            "  Visible target surface XYZ (m):",
                            np.round(
                                result["surface_position"], 4
                            ),
                        )

                    next_report = data.time + 1.0

                next_capture = data.time + PERCEPTION_PERIOD

            viewer.sync()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            remaining = (
                model.opt.timestep
                - (time.perf_counter() - start)
            )

            if remaining > 0:
                time.sleep(remaining)

finally:
    camera.close()
    cv2.destroyAllWindows()