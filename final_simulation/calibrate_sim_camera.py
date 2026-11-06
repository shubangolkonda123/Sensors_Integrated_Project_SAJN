"""
calibrate_sim_camera.py

Automatic checkerboard intrinsic calibration for the simulated UR3 wrist camera.

Run normally from VS Code:
    python calibrate_sim_camera.py

Optional:
    python calibrate_sim_camera.py --model path/to/ur3e.xml
    python calibrate_sim_camera.py --views 20 --test-views 5 --seed 7

No original XML changes are required.

Outputs:
    results/sim_camera_calibration/
        camera_calibration.npz
        calibration_report.json
        images/
        detected/

Notes:
- Robot positioning is kinematic, not a dynamic controller experiment.
- IK and sampled table-collision checks are used for camera positioning.
- Calibration uses detected corners from rendered RGB images.
- MuJoCo camera parameters are consulted only AFTER calibration,
  for evaluation.
- Distortion is fixed at zero because this experiment uses MuJoCo's
  ideal pinhole rendering, with no added lens-distortion simulation.
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

WIDTH = 320
HEIGHT = 240

# Inner corners: 8 columns, 5 rows.
# Actual pattern: 9 columns, 6 rows of squares.
COLS = 8
ROWS = 5
SQUARE = 0.015

BOARD_CENTER = np.array([0.31, 0.08, 0.227])
BOARD_THICKNESS = 0.0005
BOARD_BORDER = 0.015

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

CAMERA_NAME = "wrist_camera"

# Camera optical axes relative to MuJoCo's camera axes.
OPTICAL_CONVERSION = np.diag([1.0, -1.0, -1.0])


def require_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing model element: {name}")
    return index


def default_model_path():
    folder = Path(__file__).resolve().parent
    candidates = [
        folder / "universal_robots_ur3e" / "ur3e.xml",
        folder.parent / "universal_robots_ur3e" / "ur3e.xml",
        folder / "ur3e.xml",
    ]

    for path in candidates:
        if path.is_file():
            return path

    return candidates[0]


def vector_text(values):
    return " ".join(f"{float(value):.10g}" for value in values)


# ------------------------------------------------------------
# CREATE CHECKERBOARD IN A COPY OF THE XML
# ------------------------------------------------------------

def load_model_with_checkerboard(xml_path):
    root = ET.parse(xml_path).getroot()
    folder = xml_path.parent

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")

    # from_xml_string has no original XML directory.
    # Resolve asset directories before compiling the modified XML.
    assetdir = compiler.get("assetdir", "")
    meshdir = compiler.get("meshdir", assetdir)
    texturedir = compiler.get("texturedir", assetdir)

    compiler.set("meshdir", str((folder / meshdir).resolve()))
    compiler.set("texturedir", str((folder / texturedir).resolve()))

    for include in root.iter("include"):
        filename = include.get("file")
        if filename:
            include.set("file", str((folder / filename).resolve()))

    world = root.find("worldbody")
    if world is None:
        raise ValueError("XML has no worldbody.")

    # Hide the target block during calibration without changing qpos layout.
    block = world.find(".//body[@name='target_block']")
    if block is not None:
        for geom in block.iter("geom"):
            geom.set("rgba", "1 0 0 0")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    for name, rgba in [
        ("calibration_white", "1 1 1 1"),
        ("calibration_black", "0.01 0.01 0.01 1"),
    ]:
        ET.SubElement(
            asset,
            "material",
            name=name,
            rgba=rgba,
            emission="0.7",
            specular="0",
            shininess="0",
        )

    board = ET.SubElement(
        world,
        "body",
        name="calibration_board",
        pos=vector_text(BOARD_CENTER),
    )

    pattern_width = (COLS + 1) * SQUARE
    pattern_height = (ROWS + 1) * SQUARE

    # White border behind the alternating squares.
    ET.SubElement(
        board,
        "geom",
        name="calibration_board_border",
        type="box",
        pos="0 0 -0.0007",
        size=vector_text([
            pattern_width / 2 + BOARD_BORDER,
            pattern_height / 2 + BOARD_BORDER,
            0.0005,
        ]),
        material="calibration_white",
        contype="0",
        conaffinity="0",
    )

    for row in range(ROWS + 1):
        for col in range(COLS + 1):
            x = (col - COLS / 2) * SQUARE
            y = (row - ROWS / 2) * SQUARE

            material = (
                "calibration_black"
                if (row + col) % 2 == 0
                else "calibration_white"
            )

            ET.SubElement(
                board,
                "geom",
                name=f"calibration_square_{row}_{col}",
                type="box",
                pos=vector_text([x, y, 0]),
                size=vector_text([
                    SQUARE / 2,
                    SQUARE / 2,
                    BOARD_THICKNESS,
                ]),
                material=material,
                contype="0",
                conaffinity="0",
            )

    xml_string = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_string)


# ------------------------------------------------------------
# KINEMATIC CAMERA POSITIONING
# ------------------------------------------------------------

class CameraPosePlanner:
    def __init__(self, model, home_id):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.data, home_id)

        joint_ids = [
            require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINTS
        ]

        self.addresses = model.jnt_qposadr[joint_ids]
        self.lower = model.jnt_range[joint_ids, 0] + 1e-4
        self.upper = model.jnt_range[joint_ids, 1] - 1e-4

        self.camera_id = require_id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
        )
        self.table_id = require_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "tabletop"
        )

        self.home = self.data.qpos[self.addresses].copy()

    def set_q(self, q):
        self.data.qpos[self.addresses] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def camera_pose(self, q):
        self.set_q(q)

        position = self.data.cam_xpos[self.camera_id].copy()
        rotation = (
            self.data.cam_xmat[self.camera_id].reshape(3, 3)
            @ OPTICAL_CONVERSION
        )

        return position, rotation

    def solve(self, position, rotation, initial_q):
        def residual(q):
            current_position, current_rotation = self.camera_pose(q)

            rotation_error = Rotation.from_matrix(
                rotation @ current_rotation.T
            ).as_rotvec()

            return np.concatenate([
                current_position - position,
                0.15 * rotation_error,
            ])

        solution = least_squares(
            residual,
            np.clip(initial_q, self.lower, self.upper),
            bounds=(self.lower, self.upper),
            max_nfev=180,
            diff_step=1e-5,
        )

        error = residual(solution.x)

        if np.linalg.norm(error[:3]) > 0.003:
            return None

        if np.linalg.norm(error[3:]) / 0.15 > 0.04:
            return None

        return solution.x.copy()

    def path_clear_of_table(self, start, goal):
        # Sampled table-only check, not a complete collision planner.
        for q in np.linspace(start, goal, 24):
            self.set_q(q)

            for contact in self.data.contact[:self.data.ncon]:
                if self.table_id not in (contact.geom1, contact.geom2):
                    continue

                other = (
                    contact.geom2
                    if contact.geom1 == self.table_id
                    else contact.geom1
                )

                body_id = self.model.geom_bodyid[other]
                body_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )

                if body_name == "target_block":
                    continue

                if contact.dist < -0.0005:
                    return False

        return True


def looking_at_rotation(position, target, roll):
    forward = target - position
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)

    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward])

    return rotation @ Rotation.from_euler("z", roll).as_matrix()


def animate_move(model, data, addresses, start, goal, viewer, seconds):
    steps = max(2, int(seconds * 40))

    for fraction in np.linspace(0, 1, steps):
        if not viewer.is_running():
            return False

        blend = fraction * fraction * (3 - 2 * fraction)
        data.qpos[addresses] = start + blend * (goal - start)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(seconds / steps)

    return True


# ------------------------------------------------------------
# IMAGE DETECTION AND EVALUATION
# ------------------------------------------------------------

def detect_corners(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            (COLS, ROWS),
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE,
        )

        if found:
            return True, corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(
        gray,
        (COLS, ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if not found:
        return False, None

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.001,
        ),
    )

    return True, corners


def validation_errors(object_points, image_points, K, distortion):
    errors = []

    for corners in image_points:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners,
            K,
            distortion,
        )

        if not ok:
            errors.append(None)
            continue

        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, K, distortion
        )

        difference = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        rms = np.sqrt(np.mean(np.sum(difference**2, axis=1)))
        errors.append(float(rms))

    return errors


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument("--test-views", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=250)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--move-seconds", type=float, default=0.4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sim_camera_calibration"),
    )
    args = parser.parse_args()

    if args.views < 10 or args.test_views < 1:
        parser.error("Use at least 10 training views and 1 test view.")

    if args.move_seconds < 0:
        parser.error("--move-seconds cannot be negative.")

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output = args.output.resolve()
    images_dir = output / "images"
    detected_dir = output / "detected"
    images_dir.mkdir(parents=True, exist_ok=True)
    detected_dir.mkdir(parents=True, exist_ok=True)

    seed = (
        args.seed
        if args.seed is not None
        else int(np.random.SeedSequence().entropy)
    )
    rng = np.random.default_rng(seed)

    print("Loading:", model_path)
    print("Seed:", seed)
    print("Checkerboard inner corners:", (COLS, ROWS))
    print("Square size:", SQUARE, "m")
    print("Robot motion is kinematic for this calibration experiment.")

    model = load_model_with_checkerboard(model_path)
    home_id = require_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    planner = CameraPosePlanner(model, home_id)
    current_q = planner.home.copy()

    object_points = np.zeros((COLS * ROWS, 3), dtype=np.float32)
    object_points[:, :2] = (
        np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE
    )

    collected = []
    accepted_q = []
    total_needed = args.views + args.test_views

    counts = {
        "ik_failed": 0,
        "table_path_rejected": 0,
        "detection_failed": 0,
        "similar_pose_rejected": 0,
        "accepted": 0,
    }

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = BOARD_CENTER
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

            for attempt in range(args.attempts):
                if not viewer.is_running():
                    break

                if len(collected) >= total_needed:
                    break

                # Vary position, distance, tilt and roll.
                position = BOARD_CENTER + np.array([
                    rng.uniform(-0.085, 0.085),
                    rng.uniform(-0.085, 0.085),
                    rng.uniform(0.18, 0.29),
                ])

                target = BOARD_CENTER + np.array([
                    rng.uniform(-0.025, 0.025),
                    rng.uniform(-0.020, 0.020),
                    BOARD_THICKNESS,
                ])

                desired_rotation = looking_at_rotation(
                    position,
                    target,
                    rng.uniform(-0.35, 0.35),
                )

                goal = None
                for initial in (current_q, planner.home):
                    goal = planner.solve(
                        position, desired_rotation, initial
                    )
                    if goal is not None:
                        break

                if goal is None:
                    counts["ik_failed"] += 1
                    continue

                if accepted_q and min(
                    np.linalg.norm(goal - q) for q in accepted_q
                ) < 0.10:
                    counts["similar_pose_rejected"] += 1
                    continue

                if not planner.path_clear_of_table(current_q, goal):
                    counts["table_path_rejected"] += 1
                    continue

                moved = animate_move(
                    model,
                    data,
                    planner.addresses,
                    current_q,
                    goal,
                    viewer,
                    args.move_seconds,
                )

                if not moved:
                    break

                current_q = goal.copy()

                renderer.update_scene(data, camera=CAMERA_NAME)
                rgb = renderer.render().copy()

                found, corners = detect_corners(rgb)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                display = bgr.copy()

                if found:
                    cv2.drawChessboardCorners(
                        display, (COLS, ROWS), corners, True
                    )

                    number = len(collected)
                    cv2.imwrite(
                        str(images_dir / f"view_{number:03d}.png"), bgr
                    )
                    cv2.imwrite(
                        str(detected_dir / f"view_{number:03d}.png"),
                        display,
                    )

                    collected.append(corners.copy())
                    accepted_q.append(goal.copy())
                    counts["accepted"] += 1

                    print(
                        f"Captured {len(collected)}/{total_needed} "
                        f"(candidate {attempt + 1})"
                    )
                else:
                    counts["detection_failed"] += 1
                    print(
                        f"Candidate {attempt + 1}: "
                        "checkerboard not fully detected."
                    )

                cv2.putText(
                    display,
                    f"Views: {len(collected)}/{total_needed}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("Automatic simulated calibration", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        renderer.close()
        cv2.destroyAllWindows()

    print("Capture diagnostics:", counts)

        # Prefer the requested view count, but allow calibration with
    # at least 10 training images plus held-out validation images.
    minimum_training_views = 10
    available_views = len(collected)

    if available_views < minimum_training_views + 1:
        raise RuntimeError(
            f"Only {available_views} usable checkerboard views were "
            "collected. Need at least 10 training views and "
            "1 held-out validation view. "
            f"Capture diagnostics: {counts}"
        )

    # Preserve the requested test count where possible, while
    # always keeping at least 10 images for fitting.
    actual_test_views = min(
        args.test_views,
        available_views - minimum_training_views,
    )

    actual_training_views = min(
        args.views,
        available_views - actual_test_views,
    )

    if available_views < total_needed:
        print(
            f"\nWARNING: Collected {available_views}/{total_needed} "
            "requested views."
        )
        print(
            "Proceeding with "
            f"{actual_training_views} training views and "
            f"{actual_test_views} held-out validation views."
        )

    args.views = actual_training_views
    args.test_views = actual_test_views

    # Randomly separate fitting and held-out images.
    order = rng.permutation(len(collected))
    train_indices = order[:args.views]
    test_indices = order[args.views:args.views + args.test_views]

    train_images = [collected[i] for i in train_indices]
    test_images = [collected[i] for i in test_indices]
    train_objects = [object_points.copy() for _ in train_images]

    # Ideal pinhole camera experiment: no simulated lens distortion.
    # Focal lengths and principal point remain unknown and are fitted.
    flags = (
        cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )

    rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(
        train_objects,
        train_images,
        (WIDTH, HEIGHT),
        None,
        None,
        flags=flags,
    )

    if not np.all(np.isfinite(K)):
        raise RuntimeError("Calibration returned non-finite parameters.")

    train_errors = validation_errors(
        object_points, train_images, K, distortion
    )
    test_errors = validation_errors(
        object_points, test_images, K, distortion
    )

    # Evaluation only: compare fitted parameters with the renderer model.
    camera_id = planner.camera_id
    fovy = np.deg2rad(model.cam_fovy[camera_id])
    focal = (HEIGHT / 2) / np.tan(fovy / 2)

    # Pixel-centre convention used by OpenCV image coordinates.
    K_reference = np.array([
        [focal, 0, (WIDTH - 1) / 2],
        [0, focal, (HEIGHT - 1) / 2],
        [0, 0, 1],
    ])

    valid_test_errors = [
        error for error in test_errors if error is not None
    ]

    report = {
        "seed": seed,
        "image_size": [WIDTH, HEIGHT],
        "inner_corners": [COLS, ROWS],
        "square_size_m": SQUARE,
        "motion_mode": "kinematic joint positioning",
        "distortion_model": "fixed zero: ideal pinhole renderer",
        "capture_diagnostics": counts,
        "training_indices": train_indices.tolist(),
        "held_out_indices": test_indices.tolist(),
        "calibration_rms_px": float(rms),
        "training_per_view_rms_px": train_errors,
        "held_out_per_view_rms_px": test_errors,
        "held_out_mean_rms_px": (
            float(np.mean(valid_test_errors))
            if valid_test_errors else None
        ),
        "K_estimated": K.tolist(),
        "Dist": distortion.ravel().tolist(),
        "K_renderer_reference": K_reference.tolist(),
        "K_difference": (K - K_reference).tolist(),
        "validation_note": (
            "Held-out board poses are fitted with solvePnP using fixed "
            "calibrated intrinsics. This tests reprojection consistency, "
            "not independent depth accuracy."
        ),
    }

    np.savez(
        output / "camera_calibration.npz",
        K=K,
        Dist=distortion,
        Rms=float(rms),
        image_size=np.array([WIDTH, HEIGHT]),
        square_size_m=SQUARE,
        inner_corners=np.array([COLS, ROWS]),
        joint_poses=np.array(accepted_q),
    )

    (output / "calibration_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nCALIBRATION COMPLETE")
    print("Estimated camera matrix:\n", K)
    print("Distortion:\n", distortion)
    print(f"Calibration RMS: {rms:.4f} pixels")
    print("Held-out per-view RMS:", test_errors)
    print("Renderer reference K:\n", K_reference)
    print("Difference:\n", K - K_reference)
    print("Results saved to:", output)


if __name__ == "__main__":
    main()