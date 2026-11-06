"""
Vision-guided UR3e pick prototype.

Pipeline:
    wrist RGB-D camera
      -> camera calibration
      -> depth-based block localisation
      -> block position measurement
      -> joint Kalman observer / block estimator
      -> LQR state-space joint controller
      -> MuJoCo UR3e pick

Ground-truth policy:
    The perception, estimation, planning and control pipeline uses only:
      - measured joint angles,
      - wrist RGB images,
      - wrist depth images,
      - model-based forward kinematics.

    Simulator ground truth for the block is used only inside Evaluation
    for post-run metrics. It is never passed to Sensors, PickTask,
    EstimatedJointControl, or any planner/controller function.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import mujoco
import mujoco.viewer
import roboticstoolbox as rtb

from scipy.linalg import solve_discrete_are
from scipy.optimize import least_squares
from scipy.signal import cont2discrete
from scipy.spatial.transform import Rotation


# ============================================================
# SETTINGS
# ============================================================

CONTROL_DT = 0.01
VISION_DT = 0.15

WIDTH = 320
HEIGHT = 240

# Joint encoder noise.
ENCODER_SIGMA = 0.0005  # rad

# RGB-D depth sensor model.
# This must not be zero: the estimator must operate on noisy depth.
DEPTH_NOISE_STD = 0.004  # m, approximately 4 mm standard deviation
DEPTH_DROPOUT_PROBABILITY = 0.01

# Valid depth range.
DEPTH_MIN = 0.02
DEPTH_MAX = 1.5

FINGER_OPEN = 0.03
FINGER_CLOSED = 0.0

APPROACH_CLEARANCE = 0.08
LIFT_DISTANCE = 0.08

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

GRIPPER_ACTUATORS = [
    "left_gripper",
    "right_gripper",
]

# Desired grasp orientation:
# local Z points downward, local X remains world X.
DOWNWARD_ROTATION = np.diag([1.0, -1.0, -1.0])


def require_id(model, object_type, name):
    index = mujoco.mj_name2id(model, object_type, name)
    if index < 0:
        raise ValueError(f"Missing model element: {name}")
    return index


def default_model_path():
    script_folder = Path(__file__).resolve().parent

    candidates = [
        script_folder / "universal_robots_ur3e" / "ur3e.xml",
        script_folder.parent / "universal_robots_ur3e" / "ur3e.xml",
        script_folder / "ur3e.xml",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


# ============================================================
# SECTION 1 — CAMERA CALIBRATION
# ============================================================

class WristCameraCalibration:
    """
    Intrinsic calibration for the wrist-mounted RGB-D camera.

    The camera matrix is derived from MuJoCo's vertical field of view and
    the rendered image size. For a real camera, replace this with a
    checkerboard calibration using cv2.calibrateCamera.

    The camera optical-frame convention is:
        X: image right
        Y: image down
        Z: forward, away from camera
    """

    def __init__(self, model, camera_id, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.camera_id = camera_id

        fovy_rad = np.deg2rad(model.cam_fovy[camera_id])
        focal_pixels = (height / 2.0) / np.tan(fovy_rad / 2.0)

        self.K = np.array(
            [
                [focal_pixels, 0.0, (width - 1) / 2.0],
                [0.0, focal_pixels, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        u, v = np.meshgrid(
            np.arange(width),
            np.arange(height),
        )

        pixels = np.stack(
            [u, v, np.ones_like(u)],
            axis=-1,
        )

        self.rays = pixels @ np.linalg.inv(self.K).T

    def deproject(self, depth_image):
        """
        Convert a depth image into 3D points in the camera optical frame.

        Returns an HxWx3 array.
        """
        return self.rays * depth_image[..., None]


# ============================================================
# SECTION 2 — MODEL-BASED KINEMATICS
# ============================================================

class Kinematics:
    """
    Model-based kinematics used by perception, planning and control.

    This class does not read the live target-block pose.
    It only computes gripper/camera poses from a supplied joint vector.
    """

    def __init__(self, model, home_id):
        self.model = model
        self.data = mujoco.MjData(model)

        mujoco.mj_resetDataKeyframe(
            model,
            self.data,
            home_id,
        )

        self.joint_ids = np.array(
            [
                require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINTS
            ]
        )

        self.actuator_ids = np.array(
            [
                require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATORS
            ]
        )

        self.q_addresses = model.jnt_qposadr[self.joint_ids]
        self.v_addresses = model.jnt_dofadr[self.joint_ids]

        self.grasp_site_id = require_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "grasp_site",
        )

        self.camera_id = require_id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "wrist_camera",
        )

        self.table_geom_id = require_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "tabletop",
        )

        self.lower = np.maximum(
            model.jnt_range[self.joint_ids, 0],
            model.actuator_ctrlrange[self.actuator_ids, 0],
        ) + 1e-5

        self.upper = np.minimum(
            model.jnt_range[self.joint_ids, 1],
            model.actuator_ctrlrange[self.actuator_ids, 1],
        ) - 1e-5

    def update(self, q, velocity=None):
        self.data.qpos[self.q_addresses] = q
        self.data.qvel[:] = 0.0

        if velocity is not None:
            self.data.qvel[self.v_addresses] = velocity

        mujoco.mj_forward(self.model, self.data)

    def tcp_position(self, q):
        self.update(q)
        return self.data.site_xpos[self.grasp_site_id].copy()

    def camera_pose(self, q):
        """
        Return wrist-camera pose in the robot base frame.

        The returned rotation maps camera optical-frame coordinates into
        base-frame coordinates.
        """
        self.update(q)

        position = self.data.cam_xpos[self.camera_id].copy()
        rotation_mujoco = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()

        # Convert MuJoCo camera axes to the optical convention:
        # X right, Y down, Z forward.
        rotation_optical = rotation_mujoco @ np.diag([1.0, -1.0, -1.0])

        return position, rotation_optical

    def inverse_kinematics(self, target_position, q_initial):
        target_position = np.asarray(target_position, dtype=float)

        def residual(q):
            self.update(q)

            position = self.data.site_xpos[self.grasp_site_id]
            rotation = self.data.site_xmat[self.grasp_site_id].reshape(3, 3)

            position_error = position - target_position
            rotation_error = Rotation.from_matrix(
                DOWNWARD_ROTATION @ rotation.T
            ).as_rotvec()

            return np.concatenate(
                [
                    position_error,
                    0.15 * rotation_error,
                ]
            )

        result = least_squares(
            residual,
            np.clip(q_initial, self.lower, self.upper),
            bounds=(self.lower, self.upper),
            max_nfev=200,
            diff_step=1e-5,
        )

        error = residual(result.x)
        position_error = np.linalg.norm(error[:3])
        angle_error = np.linalg.norm(error[3:]) / 0.15

        if position_error > 0.005 or angle_error > 0.08:
            raise RuntimeError(
                "IK failed: "
                f"position error={position_error:.4f} m, "
                f"orientation error={np.degrees(angle_error):.2f} deg"
            )

        return result.x.copy()

    def segment_clear_of_table(self, q_start, q_end):
        """
        Sampled geometric check against the table only.

        This is not a complete collision planner.
        """
        for q in np.linspace(q_start, q_end, 12):
            self.update(q)

            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]

                if self.table_geom_id not in (contact.geom1, contact.geom2):
                    continue

                other_geom = (
                    contact.geom2
                    if contact.geom1 == self.table_geom_id
                    else contact.geom1
                )

                body_id = self.model.geom_bodyid[other_geom]
                body_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                )

                if (
                    body_name != "target_block"
                    and contact.dist < -0.0005
                ):
                    return False

        return True


# ============================================================
# SECTION 3 — RGB-D SENSING AND DEPTH-BASED LOCALISATION
# ============================================================

class Sensors:
    """
    Wrist RGB-D sensing and depth-based target-block localisation.

    Block-related inputs:
        - RGB image,
        - depth image.

    This class must never receive the simulator's true block pose.
    """

    def __init__(self, model, kinematics, rng):
        self.kinematics = kinematics
        self.rng = rng

        self.renderer = mujoco.Renderer(
            model,
            height=HEIGHT,
            width=WIDTH,
        )

        self.calibration = WristCameraCalibration(
            model=model,
            camera_id=kinematics.camera_id,
            width=WIDTH,
            height=HEIGHT,
        )

    def encoder(self, plant):
        """
        Simulated joint-angle encoder measurement.

        This is a robot proprioceptive sensor, not block ground truth.
        """
        q_measured = plant.qpos[self.kinematics.q_addresses].copy()
        q_measured += self.rng.normal(
            0.0,
            ENCODER_SIGMA,
            6,
        )

        return q_measured

    def capture_rgbd(
        self,
        plant,
        noise_sigma=DEPTH_NOISE_STD,
        dropout=DEPTH_DROPOUT_PROBABILITY,
    ):
        """
        Capture wrist RGB and depth, then apply a realistic depth-noise
        model.

        The default noise is non-zero. A zero-noise depth sensor would not
        provide a meaningful robustness test for the estimator.
        """
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(
            plant,
            camera="wrist_camera",
        )
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(
            plant,
            camera="wrist_camera",
        )
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        valid = (
            np.isfinite(depth)
            & (depth > DEPTH_MIN)
            & (depth < DEPTH_MAX)
        )

        if noise_sigma > 0.0:
            depth[valid] += self.rng.normal(
                0.0,
                noise_sigma,
                np.count_nonzero(valid),
            )

        dropped = self.rng.random(depth.shape) < dropout
        depth[~valid | dropped] = 0.0

        return rgb, depth

    def find_target(self, rgb, depth, q_estimated):
        """
        Depth-based localisation of the red target block.

        Steps:
        1. Deproject valid depth pixels into the camera optical frame.
        2. Transform them into the robot base frame using model-based
           wrist-camera pose.
        3. Restrict points to the known tabletop workspace.
        4. Estimate the tabletop plane using RANSAC.
        5. Detect red block pixels above the table plane.
        6. Estimate the block XY centre from the visible point cloud.
        7. Place the grasp height just above the estimated table surface.

        Returns:
            grasp_position: 3-vector in base frame, or None.
            target_mask: HxW uint8 mask for visualisation.
        """
        empty_mask = np.zeros(depth.shape, dtype=np.uint8)

        valid = (
            np.isfinite(depth)
            & (depth > 0.15)
            & (depth < DEPTH_MAX)
        )

        points_camera = self.calibration.deproject(depth)

        camera_position, camera_rotation = self.kinematics.camera_pose(
            q_estimated
        )

        points_base = (
            points_camera @ camera_rotation.T
            + camera_position
        )

        # Broad tabletop workspace.
        # This is a static workspace prior, not a block-position measurement.
        workspace = (
            valid
            & (points_base[..., 0] > 0.12)
            & (points_base[..., 0] < 0.56)
            & (np.abs(points_base[..., 1]) < 0.27)
            & (points_base[..., 2] > 0.17)
            & (points_base[..., 2] < 0.36)
        )

        points = points_base[workspace]

        if len(points) < 150:
            return None, empty_mask

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(0.003)

        if len(cloud.points) < 50:
            return None, empty_mask

        plane, inliers = cloud.segment_plane(
            distance_threshold=0.003,
            ransac_n=3,
            num_iterations=350,
        )

        plane = np.asarray(plane, dtype=float)
        plane /= np.linalg.norm(plane[:3])

        if plane[2] < 0.0:
            plane = -plane

        if plane[2] < 0.9 or len(inliers) < 0.35 * len(cloud.points):
            return None, empty_mask

        signed_distance = points_base @ plane[:3] + plane[3]

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        red_low = cv2.inRange(
            hsv,
            (0, 80, 40),
            (12, 255, 255),
        )

        red_high = cv2.inRange(
            hsv,
            (168, 80, 40),
            (179, 255, 255),
        )

        red = (red_low | red_high) > 0

        candidate = (
            red
            & workspace
            & (signed_distance > 0.008)
            & (signed_distance < 0.07)
        ).astype(np.uint8)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )

        if count < 2:
            return None, empty_mask

        label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        if stats[label, cv2.CC_STAT_AREA] < 12:
            return None, empty_mask

        target_mask = labels == label
        object_points = points_base[target_mask]

        # Robust XY centre from visible block extents.
        xy = (
            np.percentile(
                object_points[:, :2],
                5,
                axis=0,
            )
            + np.percentile(
                object_points[:, :2],
                95,
                axis=0,
            )
        ) / 2.0

        table_height = -(
            plane[0] * xy[0]
            + plane[1] * xy[1]
            + plane[3]
        ) / plane[2]

        # Specific to the 4 cm cube and finger geometry.
        grasp_position = np.array(
            [
                xy[0],
                xy[1],
                table_height + 0.037,
            ]
        )

        return (
            grasp_position,
            target_mask.astype(np.uint8) * 255,
        )

    def close(self):
        self.renderer.close()


# ============================================================
# SECTION 4 — JOINT KALMAN OBSERVER AND LQR CONTROLLER
# ============================================================

class EstimatedJointControl:
    """
    State-space joint controller with a Kalman observer.

    State:
        x = [q1...q6, qdot1...qdot6]

    Measurement:
        y = [q1...q6]

    The joint angles are measured by simulated encoders.
    The joint velocities are unmeasured and are estimated by the Kalman
    observer.

    The controller is a discrete LQR state-feedback law:
        u = q_reference - K (x_hat - x_reference)

    It uses the estimated state, never simulator ground truth.
    """

    def __init__(self, model, kinematics, q_initial):
        self.model = model
        self.kinematics = kinematics

        kinematics.update(q_initial)

        mass = np.zeros(
            (model.nv, model.nv),
            dtype=np.float64,
        )

        mujoco.mj_fullM(
            model,
            kinematics.data,
            mass,
        )

        arm_mass = mass[
            np.ix_(
                kinematics.v_addresses,
                kinematics.v_addresses,
            )
        ]

        kp = model.actuator_gainprm[
            kinematics.actuator_ids, 0
        ]

        kd = -model.actuator_biasprm[
            kinematics.actuator_ids, 2
        ]

        if np.any(kp <= 0.0) or np.any(kd < 0.0):
            raise ValueError(
                "Expected the affine position actuators "
                "from the supplied XML."
            )

        A_continuous = np.block(
            [
                [
                    np.zeros((6, 6)),
                    np.eye(6),
                ],
                [
                    -np.linalg.solve(arm_mass, np.diag(kp)),
                    -np.linalg.solve(arm_mass, np.diag(kd)),
                ],
            ]
        )

        B_continuous = np.vstack(
            [
                np.zeros((6, 6)),
                np.linalg.solve(arm_mass, np.diag(kp)),
            ]
        )

        C = np.hstack(
            [
                np.eye(6),
                np.zeros((6, 6)),
            ]
        )

        self.A, self.B, self.C, _, _ = cont2discrete(
            (
                A_continuous,
                B_continuous,
                C, None
            ),
            CONTROL_DT,
        )

        Q_control = np.diag(
            [400.0] * 6 + [1.0] * 6
        )

        R_control = 0.5 * np.eye(6)

        S = solve_discrete_are(
            self.A,
            self.B,
            Q_control,
            R_control,
        )

        self.K = np.linalg.solve(
            R_control + self.B.T @ S @ self.B,
            self.B.T @ S @ self.A,
        )

        # Initial estimated state.
        self.x = np.concatenate(
            [
                q_initial,
                np.zeros(6),
            ]
        )

        # Initial state covariance.
        self.P = np.diag(
            [ENCODER_SIGMA**2] * 6
            + [0.05**2] * 6
        )

        # Process-noise input matrix.
        G = np.vstack(
            [
                0.5 * CONTROL_DT**2 * np.eye(6),
                CONTROL_DT * np.eye(6),
            ]
        )

        self.Q = (
            G @ (4.0 * np.eye(6)) @ G.T
            + 1e-10 * np.eye(12)
        )

        self.R = ENCODER_SIGMA**2 * np.eye(6)

        self.u = q_initial.copy()
        self.first_update = True

        controllability = np.hstack(
            [
                np.linalg.matrix_power(self.A, i) @ self.B
                for i in range(12)
            ]
        )

        observability = np.vstack(
            [
                self.C @ np.linalg.matrix_power(self.A, i)
                for i in range(12)
            ]
        )

        self.analysis = {
            "states": 12,
            "controllability_rank": int(
                np.linalg.matrix_rank(controllability)
            ),
            "observability_rank": int(
                np.linalg.matrix_rank(observability)
            ),
            "max_closed_loop_eigenvalue_magnitude": float(
                np.max(
                    np.abs(
                        np.linalg.eigvals(self.A - self.B @ self.K)
                    )
                )
            ),
        }

    def update(self, measured_q, q_reference):
        """
        Run one Kalman predict/update cycle, then compute the LQR command.

        Inputs:
            measured_q: noisy encoder measurement of joint angles.
            q_reference: desired joint angles from the pick planner.

        Returns:
            commanded joint positions.
        """
        if not self.first_update:
            self.x = self.A @ self.x + self.B @ self.u
            self.P = self.A @ self.P @ self.A.T + self.Q

        self.first_update = False

        innovation = measured_q - self.C @ self.x

        innovation_covariance = (
            self.C @ self.P @ self.C.T
            + self.R
        )

        gain = np.linalg.solve(
            innovation_covariance,
            self.C @ self.P,
        ).T

        self.x += gain @ innovation

        J = np.eye(12) - gain @ self.C

        self.P = (
            J @ self.P @ J.T
            + gain @ self.R @ gain.T
        )

        self.P = (self.P + self.P.T) / 2.0

        reference_state = np.concatenate(
            [
                q_reference,
                np.zeros(6),
            ]
        )

        command = q_reference - self.K @ (
            self.x - reference_state
        )

        command = np.clip(
            command,
            q_reference - 0.15,
            q_reference + 0.15,
        )

        command = np.clip(
            command,
            self.u - 0.012,
            self.u + 0.012,
        )

        self.u = np.clip(
            command,
            self.kinematics.lower,
            self.kinematics.upper,
        )

        return self.u.copy()

    def estimated_bias_compensation(self):
        """
        Gravity/bias compensation computed from the estimated joint state.

        This does not read the true plant state.
        """
        self.kinematics.update(
            self.x[:6],
            self.x[6:],
        )

        return self.kinematics.data.qfrc_bias[
            self.kinematics.v_addresses
        ].copy()


# ============================================================
# SECTION 5 — PICKING STATE MACHINE
# ============================================================

class PickTask:
    """
    Vision-guided pick state machine.

    The only block information used here is `target`, produced by
    Sensors.find_target from RGB-D data.
    """

    def __init__(self, kinematics, q_initial):
        self.kinematics = kinematics

        self.q_reference = q_initial.copy()
        self.q_goal = q_initial.copy()

        self.state = "OBSERVE"
        self.entered = 0.0

        self.last_seen = -np.inf
        self.last_plan = -np.inf

        self.grasp_position = None
        self.valid_count = 0

        self.finger_target = FINGER_OPEN

        self.trajectory = None
        self.trajectory_start = 0.0

        self.abort_reason = ""

    def transition(self, state, t):
        print(
            f"{t:6.2f}s: "
            f"{self.state} -> {state}"
        )

        self.state = state
        self.entered = t
        self.trajectory = None

    def abort(self, reason, t, q_estimated):
        self.abort_reason = reason

        self.q_reference = q_estimated.copy()
        self.q_goal = q_estimated.copy()

        self.transition("ABORT", t)
        print("ABORT:", reason)

    def update_vision(self, target, t):
        """
        Update the vision-derived grasp position.

        `target` is the depth-based camera measurement, not ground truth.
        """
        if target is None:
            self.valid_count = 0
            return

        self.last_seen = t
        self.valid_count += 1

        if self.grasp_position is None:
            self.grasp_position = target.copy()
        else:
            self.grasp_position = (
                0.7 * self.grasp_position
                + 0.3 * target
            )

    def plan_motion(
        self,
        target_position,
        t,
        q_estimated,
        duration,
    ):
        goal = self.kinematics.inverse_kinematics(
            target_position,
            q_estimated,
        )

        if np.max(np.abs(goal - q_estimated)) > 1.5:
            raise RuntimeError(
                "IK branch jump: refusing motion."
            )

        if not self.kinematics.segment_clear_of_table(
            q_estimated,
            goal,
        ):
            raise RuntimeError(
                "Proposed joint path intersects the table."
            )

        self.q_goal = goal

        times = np.linspace(
            0.0,
            duration,
            max(
                3,
                int(duration / CONTROL_DT) + 1,
            ),
        )

        self.trajectory = rtb.jtraj(
            self.q_reference,
            goal,
            times,
        ).q

        self.trajectory_start = t

    def update(self, t, q_estimated):
        if self.trajectory is not None:
            index = min(
                len(self.trajectory) - 1,
                max(
                    0,
                    int(
                        (t - self.trajectory_start)
                        / CONTROL_DT
                    ),
                ),
            )

            self.q_reference = self.trajectory[index].copy()

        settled = (
            np.max(np.abs(q_estimated - self.q_goal)) < 0.025
        )

        elapsed = t - self.entered

        try:
            if self.state == "OBSERVE":
                if self.valid_count >= 4:
                    self.transition("PREGRASP", t)

                elif elapsed > 12.0:
                    raise RuntimeError(
                        "Target not acquired "
                        "from the observation pose."
                    )

            elif self.state == "PREGRASP":
                if t - self.last_seen > 2.0:
                    raise RuntimeError(
                        "Target lost during approach."
                    )

                target = self.grasp_position + np.array(
                    [
                        0.0,
                        0.0,
                        APPROACH_CLEARANCE,
                    ]
                )

                if t - self.last_plan > 0.8:
                    self.plan_motion(
                        target,
                        t,
                        q_estimated,
                        2.0,
                    )
                    self.last_plan = t

                position_error = np.linalg.norm(
                    self.kinematics.tcp_position(q_estimated) - target
                )

                if position_error < 0.012 and settled:
                    self.transition("DESCEND", t)
                    self.last_plan = -np.inf

                elif elapsed > 25.0:
                    raise RuntimeError(
                        "Pregrasp timeout."
                    )

            elif self.state == "DESCEND":
                if t - self.last_seen > 5.0:
                    raise RuntimeError(
                        "Target measurement stale during descent."
                    )

                if self.trajectory is None:
                    self.plan_motion(
                        self.grasp_position,
                        t,
                        q_estimated,
                        4.0,
                    )
                    self.last_plan = t

                elif (
                    elapsed < 2.0
                    and t - self.last_plan > 0.8
                    and t - self.last_seen < 0.3
                ):
                    self.plan_motion(
                        self.grasp_position,
                        t,
                        q_estimated,
                        max(1.0, 4.0 - elapsed),
                    )
                    self.last_plan = t

                position_error = np.linalg.norm(
                    self.kinematics.tcp_position(q_estimated)
                    - self.grasp_position
                )

                if elapsed > 4.0 and position_error < 0.008:
                    self.transition("CLOSE", t)

                elif elapsed > 10.0:
                    raise RuntimeError(
                        "Descent timeout."
                    )

            elif self.state == "CLOSE":
                fraction = min(1.0, elapsed / 1.5)

                self.finger_target = FINGER_OPEN * (1.0 - fraction)

                if elapsed > 2.5:
                    self.transition("LIFT", t)

                    self.plan_motion(
                        self.grasp_position
                        + np.array(
                            [
                                0.0,
                                0.0,
                                LIFT_DISTANCE,
                            ]
                        ),
                        t,
                        q_estimated,
                        4.0,
                    )

            elif self.state == "LIFT":
                self.finger_target = FINGER_CLOSED

                if elapsed > 5.0 and settled:
                    self.transition("HOLD", t)

                elif elapsed > 12.0:
                    raise RuntimeError(
                        "Lift timeout."
                    )

        except RuntimeError as error:
            self.abort(
                str(error),
                t,
                q_estimated,
            )

        return (
            self.q_reference.copy(),
            self.finger_target,
        )


# ============================================================
# SECTION 6 — EVALUATION: GROUND TRUTH ISOLATED HERE
# ============================================================

class Evaluation:
    """
    Evaluation-only module.

    This is the only class permitted to read simulator ground truth for
    the target block. Its outputs are used for metrics and diagnostics,
    not for perception, estimation, planning or control.
    """

    def __init__(
        self,
        model,
        kinematics,
        plant,
        directory,
    ):
        self.kinematics = kinematics
        self.directory = directory
        self.rows = []

        self.block_id = require_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_block",
        )

        self.initial_height = float(
            plant.xpos[self.block_id, 2]
        )

    def record(
        self,
        plant,
        controller,
        q_reference,
        state,
    ):
        q_true = plant.qpos[self.kinematics.q_addresses].copy()
        v_true = plant.qvel[self.kinematics.v_addresses].copy()

        true_state = np.concatenate([q_true, v_true])

        error = controller.x - true_state

        try:
            nees = float(
                error
                @ np.linalg.solve(controller.P, error)
            )
        except np.linalg.LinAlgError:
            nees = float("nan")

        row = {
            "time": float(plant.time),
            "state": state,
            "block_height": float(
                plant.xpos[self.block_id, 2]
            ),
            "nees": nees,
        }

        for i in range(6):
            values = {
                "q": q_true[i],
                "v": v_true[i],
                "qhat": controller.x[i],
                "vhat": controller.x[i + 6],
                "qref": q_reference[i],
                "u": controller.u[i],
                "q_sigma": np.sqrt(
                    max(controller.P[i, i], 0.0)
                ),
                "v_sigma": np.sqrt(
                    max(controller.P[i + 6, i + 6], 0.0)
                ),
            }

            for name, value in values.items():
                row[f"{name}{i + 1}"] = float(value)

        self.rows.append(row)

    def save(self, task, controller, camera_matrix):
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.rows:
            with (self.directory / "timeseries.csv").open(
                "w",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.rows[0].keys(),
                )
                writer.writeheader()
                writer.writerows(self.rows)

        hold_heights = [
            row["block_height"]
            for row in self.rows
            if row["state"] == "HOLD"
        ]

        lifted = (
            len(hold_heights) >= 100
            and min(hold_heights[-100:])
            > self.initial_height + 0.06
        )

        summary = {
            "task_state": task.state,
            "abort_reason": task.abort_reason,
            "held_lift_detected": bool(lifted),
            "initial_object_height_m": self.initial_height,
            "model_analysis": controller.analysis,
            "depth_noise_std_m": DEPTH_NOISE_STD,
            "depth_dropout_probability": DEPTH_DROPOUT_PROBABILITY,
            
        }

        if self.rows:
            metrics = [
                (
                    "joint_position_estimation_rmse_rad",
                    "qhat",
                    "q",
                ),
                (
                    "joint_velocity_estimation_rmse_rad_s",
                    "vhat",
                    "v",
                ),
                (
                    "joint_tracking_rmse_rad",
                    "q",
                    "qref",
                ),
            ]

            for label, first, second in metrics:
                errors = np.array(
                    [
                        [
                            row[f"{first}{i}"]
                            - row[f"{second}{i}"]
                            for i in range(1, 7)
                        ]
                        for row in self.rows
                    ]
                )

                summary[label] = np.sqrt(
                    np.mean(errors**2, axis=0)
                ).tolist()

        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        np.savez(
            self.directory / "design.npz",
            A=controller.A,
            B=controller.B,
            C=controller.C,
            K=controller.K,
            Qprocess=controller.Q,
            Rmeasurement=controller.R,
            Kcamera=camera_matrix,
        )

        print("\nTRIAL SUMMARY")
        print(json.dumps(summary, indent=2))


# ============================================================
# SECTION 7 — MAIN CONTROL LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--randomize",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--depth-noise",
        type=float,
        default=DEPTH_NOISE_STD,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEPTH_DROPOUT_PROBABILITY,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=65.0,
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/pick_trial"),
    )

    args = parser.parse_args()

    if args.depth_noise <= 0.0:
        parser.error(
            "--depth-noise must be greater than zero. "
            "A zero-noise depth sensor does not test state estimation."
        )

    if not 0.0 <= args.dropout <= 1.0:
        parser.error(
            "--dropout must be between 0 and 1."
        )

    model_path = args.model.resolve()

    print("Loading XML:", model_path)
    print("MuJoCo version:", mujoco.__version__)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"XML not found: {model_path}"
        )

    rng = np.random.default_rng(args.seed)

    model = mujoco.MjModel.from_xml_path(str(model_path))

    physics_steps = int(round(CONTROL_DT / model.opt.timestep))

    if (
        physics_steps < 1
        or not np.isclose(
            physics_steps * model.opt.timestep,
            CONTROL_DT,
        )
    ):
        raise ValueError(
            "Physics timestep must divide "
            "the 0.01 s control period exactly."
        )

    home_id = require_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    plant = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(
        model,
        plant,
        home_id,
    )

    if args.randomize:
        block_joint_id = require_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "target_block_joint",
        )

        address = model.jnt_qposadr[block_joint_id]

        plant.qpos[address:address + 7] = [
            rng.uniform(0.30, 0.40),
            rng.uniform(-0.04, 0.04),
            0.246,
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        mujoco.mj_forward(model, plant)

    kinematics = Kinematics(
        model,
        home_id,
    )

    # Initialise controller from a noisy encoder measurement.
    q_initial_measured = (
        plant.qpos[kinematics.q_addresses].copy()
        + rng.normal(
            0.0,
            ENCODER_SIGMA,
            6,
        )
    )

    controller = EstimatedJointControl(
        model,
        kinematics,
        q_initial_measured,
    )

    task = PickTask(
        kinematics,
        controller.x[:6],
    )

    evaluation = Evaluation(
        model,
        kinematics,
        plant,
        args.results,
    )

    finger_actuator_ids = [
        require_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in GRIPPER_ACTUATORS
    ]

    print(
        "MODEL ANALYSIS:",
        controller.analysis,
    )

    sensors = None
    next_vision = 0.0

    try:
        sensors = Sensors(
            model,
            kinematics,
            rng,
        )

        print("Camera matrix:\n", sensors.calibration.K)
        print(
            "Depth noise std:",
            args.depth_noise,
            "m",
        )
        print(
            "Depth dropout probability:",
            args.dropout,
        )
        print("Automatic arm control is active.")
        print("Press Q in the RGB window to exit.")

        with mujoco.viewer.launch_passive(
            model,
            plant,
        ) as viewer:
            viewer.cam.lookat[:] = [
                0.25,
                0.0,
                0.25,
            ]
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

            while (
                viewer.is_running()
                and plant.time < args.duration
            ):
                step_start = time.perf_counter()
                t = float(plant.time)

                if t >= next_vision:
                    mujoco.mj_forward(
                        model,
                        plant,
                    )

                    rgb, depth = sensors.capture_rgbd(
                        plant,
                        args.depth_noise,
                        args.dropout,
                    )

                    # Camera-only block measurement.
                    target, mask = sensors.find_target(
                        rgb,
                        depth,
                        controller.x[:6],
                    )

                    if task.state in (
                        "OBSERVE",
                        "PREGRASP",
                        "DESCEND",
                    ):
                        task.update_vision(
                            target,
                            t,
                        )

                    display = cv2.cvtColor(
                        rgb,
                        cv2.COLOR_RGB2BGR,
                    )

                    display[mask > 0] = [
                        0,
                        255,
                        0,
                    ]

                    cv2.putText(
                        display,
                        task.state,
                        (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow(
                        "Wrist RGB: green target",
                        display,
                    )

                    next_vision = t + VISION_DT

                q_reference, finger_target = task.update(
                    t,
                    controller.x[:6],
                )

                command = controller.update(
                    sensors.encoder(plant),
                    q_reference,
                )

                plant.ctrl[kinematics.actuator_ids] = command
                plant.ctrl[finger_actuator_ids] = finger_target

                # Estimated-model gravity compensation.
                plant.qfrc_applied[:] = 0.0
                plant.qfrc_applied[kinematics.v_addresses] = (
                    controller.estimated_bias_compensation()
                )

                # Ground truth is recorded for evaluation only.
                evaluation.record(
                    plant,
                    controller,
                    q_reference,
                    task.state,
                )

                for _ in range(physics_steps):
                    mujoco.mj_step(
                        model,
                        plant,
                    )

                viewer.sync()

                if (
                    cv2.waitKey(1) & 0xFF
                ) == ord("q"):
                    break

                if (
                    task.state in ("HOLD", "ABORT")
                    and plant.time - task.entered > 3.0
                ):
                    break

                remaining = (
                    CONTROL_DT
                    - (time.perf_counter() - step_start)
                )

                if remaining > 0.0:
                    time.sleep(remaining)

    finally:
        if sensors is not None:
            sensors.close()

        cv2.destroyAllWindows()

        if sensors is not None:
            evaluation.save(
                task,
                controller,
                sensors.calibration.K,
            )


if __name__ == "__main__":
    main()