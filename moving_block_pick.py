"""
Moving-block vision-guided UR3e pick with a block EKF.

Task flow:
1. The block starts at a randomised position within the robot workspace.
2. The block moves sideways in the -y direction at a fixed velocity.
3. The UR3e hovers above and follows the moving block using the block EKF.
4. Once the block stops, its velocity is zeroed exactly once.
   After that, the block is left alone so the gripper can hold it
   naturally during DESCEND, CLOSE, LIFT and HOLD.

Ground truth policy:
    Block ground truth is used only in Evaluation.
    The EKF, tracking controller, planner and picker use only:
      - encoder joint angles,
      - wrist RGB-D camera measurements,
      - the constant-velocity block motion model.
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
VISION_DT = 0.10

WIDTH = 320
HEIGHT = 240

ENCODER_SIGMA = 0.0005

# Non-zero RGB-D depth noise.
DEPTH_NOISE_STD = 0.004
DEPTH_DROPOUT_PROBABILITY = 0.01

FINGER_OPEN = 0.03
FINGER_CLOSED = 0.0

# Tracking-phase settings.
# The block moves sideways in -y so that it remains within reach.
BLOCK_SPEED = np.array([0.0, -0.018, 0.0])  # m/s
BLOCK_MOVE_DURATION = 12.0

# After the block stops, wait briefly before locking the grasp target.
BLOCK_STOP_SETTLE_TIME = 0.5

HOVER_CLEARANCE = 0.10

# Original static-pick settings.
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
    Wrist RGB-D camera intrinsics and pixel ray directions.

    The intrinsic matrix is derived from MuJoCo's vertical field of view.
    For real hardware, replace this with checkerboard calibration.
    """

    def __init__(self, model, camera_id, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height

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

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
        self.rays = pixels @ np.linalg.inv(self.K).T

    def deproject(self, depth_image):
        """Convert depth image to 3D camera-frame points."""
        return self.rays * depth_image[..., None]


# ============================================================
# SECTION 2 — UR3e KINEMATICS
# ============================================================

class Kinematics:
    """
    UR3e forward kinematics, camera pose, IK and table-collision check.

    This class does not read target-block ground truth.
    """

    def __init__(self, model, home_id):
        self.model = model
        self.data = mujoco.MjData(model)

        mujoco.mj_resetDataKeyframe(model, self.data, home_id)

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
            model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self.camera_id = require_id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera"
        )
        self.table_geom_id = require_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "tabletop"
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
        self.update(q)

        position = self.data.cam_xpos[self.camera_id].copy()
        rotation_mujoco = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()
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
                [position_error, 0.15 * rotation_error]
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
        for q in np.linspace(q_start, q_end, 12):
            self.update(q)

            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]

                if self.table_geom_id not in (
                    contact.geom1,
                    contact.geom2,
                ):
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
# SECTION 3 — RGB-D SENSING AND BLOCK LOCALISATION
# ============================================================

class Sensors:
    """
    Wrist RGB-D sensing and depth-based block localisation.

    This returns the original grasp position used by the working
    static-pick pipeline:

        grasp_position = table_height + 0.037

    During TRACK, this measurement is also fed to the block EKF.
    After the block stops, the latest measurement is locked and used
    directly for DESCEND -> CLOSE -> LIFT.
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
        )

    def encoder(self, plant):
        q_measured = plant.qpos[self.kinematics.q_addresses].copy()
        q_measured += self.rng.normal(0.0, ENCODER_SIGMA, 6)
        return q_measured

    def capture_rgbd(
        self,
        plant,
        noise_sigma=DEPTH_NOISE_STD,
        dropout=DEPTH_DROPOUT_PROBABILITY,
    ):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(plant, camera="wrist_camera")
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(plant, camera="wrist_camera")
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        valid = (
            np.isfinite(depth)
            & (depth > 0.02)
            & (depth < 1.5)
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
        Original depth-based block localisation.

        Returns:
            grasp_position: grasp-site target in base frame, or None.
            target_mask: RGB mask for visualisation.
        """
        empty_mask = np.zeros(depth.shape, dtype=np.uint8)

        valid = (
            np.isfinite(depth)
            & (depth > 0.15)
            & (depth < 1.5)
        )

        points_camera = self.calibration.deproject(depth)

        camera_position, camera_rotation = self.kinematics.camera_pose(
            q_estimated
        )

        points_base = (
            points_camera @ camera_rotation.T
            + camera_position
        )

        workspace = (
            valid
            & (points_base[..., 0] > 0.0)
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

        red_low = cv2.inRange(hsv, (0, 80, 40), (12, 255, 255))
        red_high = cv2.inRange(hsv, (168, 80, 40), (179, 255, 255))
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

        # Original working estimate: approximate XY centre from the
        # visible extents of the block points.
        xy = (
            np.percentile(object_points[:, :2], 2, axis=0)
            + np.percentile(object_points[:, :2], 95, axis=0)
        ) / 2.0

        table_height = -(
            plane[0] * xy[0]
            + plane[1] * xy[1]
            + plane[3]
        ) / plane[2]

        # Original grasp position used by the working static pick.
        grasp_position = np.array(
            [
                xy[0],
                xy[1],
                table_height + 0.037,
            ]
        )

        return grasp_position, target_mask.astype(np.uint8) * 255

    def close(self):
        self.renderer.close()


# ============================================================
# SECTION 4 — BLOCK EKF
# ============================================================

class BlockConstantVelocityEKF:
    """
    EKF used only while the block is moving.

    State:
        x = [px, py, pz, vx, vy, vz]

    Motion model:
        Constant velocity.

    Measurement:
        RGB-D grasp-position measurement:
        z = [px, py, pz]

    Once the block stops, the EKF is no longer used for grasping.
    """

    def __init__(self, dt=VISION_DT):
        self.dt = dt
        self.initialised = False

        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e3

        # Constant-velocity motion model.
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Camera observes position directly.
        self.H = np.hstack(
            [
                np.eye(3),
                np.zeros((3, 3)),
            ]
        )

        # Process noise: block may accelerate slightly.
        self.Q = np.diag(
            [
                (0.001)**2, (0.001)**2, (0.001)**2,
                (0.015)**2, (0.015)**2, (0.015)**2,
            ]
        )

        # RGB-D position measurement noise.
        self.R = np.diag(
            [
                (0.006)**2,
                (0.006)**2,
                (0.005)**2,
            ]
        )

    def reset_from_measurement(self, z):
        self.x[:3] = np.asarray(z, dtype=float)
        self.x[3:] = 0.0

        self.P = np.diag(
            [
                (0.010)**2, (0.010)**2, (0.010)**2,
                (0.100)**2, (0.100)**2, (0.100)**2,
            ]
        )

        self.initialised = True

    def predict(self):
        if not self.initialised:
            return

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if z is None:
            return

        z = np.asarray(z, dtype=float)

        if not self.initialised:
            self.reset_from_measurement(z)
            return

        innovation = z - self.H @ self.x

        innovation_covariance = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        kalman_gain = np.linalg.solve(
            innovation_covariance,
            self.H @ self.P,
        ).T

        self.x = self.x + kalman_gain @ innovation

        joseph = np.eye(6) - kalman_gain @ self.H
        self.P = (
            joseph @ self.P @ joseph.T
            + kalman_gain @ self.R @ kalman_gain.T
        )
        self.P = (self.P + self.P.T) / 2.0

    def stop_block(self):
        """
        Called when the known moving phase ends.

        This prevents the constant-velocity model from continuing to
        predict motion after the block has stopped.
        """
        if self.initialised:
            self.x[3:] = 0.0

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    @property
    def covariance(self):
        return self.P.copy()


# ============================================================
# SECTION 5 — JOINT KALMAN OBSERVER AND LQR CONTROLLER
# ============================================================

class EstimatedJointControl:
    """
    12-state joint Kalman observer plus LQR state-feedback controller.

    State:
        [six joint angles, six joint velocities]

    Measurement:
        [six encoder joint angles]

    The LQR controller uses the estimated state, not simulator ground truth.
    """

    def __init__(self, model, kinematics, q_initial):
        self.model = model
        self.kinematics = kinematics

        kinematics.update(q_initial)

        mass = np.zeros((model.nv, model.nv), dtype=np.float64)
        mujoco.mj_fullM(model, kinematics.data, mass)

        arm_mass = mass[
            np.ix_(
                kinematics.v_addresses,
                kinematics.v_addresses,
            )
        ]

        kp = model.actuator_gainprm[kinematics.actuator_ids, 0]
        kd = -model.actuator_biasprm[kinematics.actuator_ids, 2]

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
            (A_continuous, B_continuous, C, None),
            CONTROL_DT,
        )

        Q_control = np.diag([400.0] * 6 + [1.0] * 6)
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

        self.x = np.concatenate([q_initial, np.zeros(6)])
        self.P = np.diag(
            [ENCODER_SIGMA**2] * 6 + [0.05**2] * 6
        )

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
            [q_reference, np.zeros(6)]
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
        self.kinematics.update(
            self.x[:6],
            self.x[6:],
        )

        return self.kinematics.data.qfrc_bias[
            self.kinematics.v_addresses
        ].copy()


# ============================================================
# SECTION 6 — TRACKING AND PICK STATE MACHINE
# ============================================================

class MovingBlockPickTask:
    """
    States:
        TRACK     - hover above and follow the moving block using EKF
        DESCEND   - descend to a locked RGB-D grasp position
        CLOSE     - close gripper
        LIFT      - lift block
        HOLD      - successful pick
        ABORT     - failure
    """

    def __init__(self, kinematics, q_initial):
        self.kinematics = kinematics

        self.q_reference = q_initial.copy()
        self.q_goal = q_initial.copy()

        self.state = "TRACK"
        self.entered = 0.0

        self.last_seen = -np.inf
        self.last_plan = -np.inf

        # Latest raw RGB-D grasp-position measurement.
        self.target_position = None

        # Grasp target locked when DESCEND begins.
        self.descent_target = None

        self.valid_count = 0

        self.finger_target = FINGER_OPEN

        self.trajectory = None
        self.trajectory_start = 0.0

        self.abort_reason = ""

    def transition(self, state, t):
        print(f"{t:6.2f}s: {self.state} -> {state}")
        self.state = state
        self.entered = t
        self.trajectory = None

    def abort(self, reason, t, q_estimated):
        self.abort_reason = reason
        self.q_reference = q_estimated.copy()
        self.q_goal = q_estimated.copy()
        self.transition("ABORT", t)
        print("ABORT:", reason)

    def update_vision(self, grasp_position, t):
        if grasp_position is None:
            self.valid_count = 0
            return

        self.last_seen = t
        self.valid_count += 1

        # Store the latest raw camera measurement.
        self.target_position = grasp_position.copy()

    def plan_motion(self, target_position, t, q_estimated, duration):
        goal = self.kinematics.inverse_kinematics(
            target_position,
            q_estimated,
        )

        if np.max(np.abs(goal - q_estimated)) > 1.5:
            raise RuntimeError("IK branch jump: refusing motion.")

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
            max(3, int(duration / CONTROL_DT) + 1),
        )

        self.trajectory = rtb.jtraj(
            self.q_reference,
            goal,
            times,
        ).q

        self.trajectory_start = t

    def update(self, t, q_estimated, block_ekf, block_is_moving):
        if self.trajectory is not None:
            index = min(
                len(self.trajectory) - 1,
                max(
                    0,
                    int((t - self.trajectory_start) / CONTROL_DT),
                ),
            )
            self.q_reference = self.trajectory[index].copy()

        settled = (
            np.max(np.abs(q_estimated - self.q_goal)) < 0.025
        )

        elapsed = t - self.entered

        # Time since the block stopped moving.
        time_since_stop = t - BLOCK_MOVE_DURATION

        try:
            if self.state == "TRACK":
                if self.target_position is None:
                    if elapsed > 12.0:
                        raise RuntimeError(
                            "Moving block not acquired."
                        )
                    return self.q_reference.copy(), self.finger_target

                # Hover above the EKF block estimate while moving.
                hover_target = (
                    block_ekf.position
                    + np.array([0.0, 0.0, HOVER_CLEARANCE])
                )

                if t - self.last_plan > 0.5:
                    self.plan_motion(
                        hover_target,
                        t,
                        q_estimated,
                        0.8,
                    )
                    self.last_plan = t

                # Once the block stops and tracking is stable,
                # lock the latest RGB-D grasp position and descend.
                if (
                    not block_is_moving
                    and time_since_stop > BLOCK_STOP_SETTLE_TIME
                    and t - self.last_seen < 0.3
                    and self.valid_count > 0
                    and settled
                ):
                    # Lock the grasp target while the camera still has
                    # a clear view of the block.
                    self.descent_target = self.target_position.copy()

                    self.transition("DESCEND", t)
                    self.last_plan = -np.inf

                elif elapsed > 35.0:
                    raise RuntimeError("Tracking timeout.")

            elif self.state == "DESCEND":
                # The block is stationary and the grasp target is locked.
                # Do NOT update it from the camera during descent, because
                # the changing wrist-camera viewpoint and gripper occlusion
                # make the measured centre drift.
                if self.descent_target is None:
                    raise RuntimeError(
                        "Descent target was not locked."
                    )

                grasp_target = self.descent_target

                if self.trajectory is None:
                    self.plan_motion(
                        grasp_target,
                        t,
                        q_estimated,
                        4.0,
                    )
                    self.last_plan = t

                position_error = np.linalg.norm(
                    self.kinematics.tcp_position(q_estimated)
                    - grasp_target
                )

                if elapsed > 4.0 and position_error < 0.008:
                    self.transition("CLOSE", t)

                elif elapsed > 10.0:
                    raise RuntimeError("Descent timeout.")

            elif self.state == "CLOSE":
                # Original gripper closing behaviour.
                fraction = min(1.0, elapsed / 1.5)
                self.finger_target = FINGER_OPEN * (1.0 - fraction)

                if elapsed > 2.5:
                    self.transition("LIFT", t)

                    # Lift vertically from the locked grasp target.
                    self.plan_motion(
                        self.descent_target
                        + np.array([0.0, 0.0, LIFT_DISTANCE]),
                        t,
                        q_estimated,
                        4.0,
                    )

            elif self.state == "LIFT":
                self.finger_target = FINGER_CLOSED

                if elapsed > 5.0 and settled:
                    self.transition("HOLD", t)

                elif elapsed > 12.0:
                    raise RuntimeError("Lift timeout.")

        except RuntimeError as error:
            self.abort(str(error), t, q_estimated)

        return self.q_reference.copy(), self.finger_target


# ============================================================
# SECTION 7 — EVALUATION: GROUND TRUTH ISOLATED HERE
# ============================================================

class Evaluation:
    """
    Evaluation-only ground-truth access.

    This class never provides information to the EKF, tracking controller,
    planner or picker.
    """

    def __init__(self, model, kinematics, plant, directory):
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
        block_ekf,
        q_reference,
        state,
    ):
        q_true = plant.qpos[self.kinematics.q_addresses].copy()
        v_true = plant.qvel[self.kinematics.v_addresses].copy()

        true_state = np.concatenate([q_true, v_true])
        joint_error = controller.x - true_state

        try:
            nees = float(
                joint_error
                @ np.linalg.solve(controller.P, joint_error)
            )
        except np.linalg.LinAlgError:
            nees = float("nan")

        block_position_true = plant.xpos[self.block_id].copy()
        block_position_error = (
            block_ekf.position - block_position_true
            if block_ekf.initialised
            else np.full(3, np.nan)
        )

        row = {
            "time": float(plant.time),
            "state": state,
            "block_height": float(block_position_true[2]),
            "nees": nees,
            "block_error_x": float(block_position_error[0]),
            "block_error_y": float(block_position_error[1]),
            "block_error_z": float(block_position_error[2]),
        }

        for i in range(6):
            row[f"q{i + 1}"] = float(q_true[i])
            row[f"v{i + 1}"] = float(v_true[i])
            row[f"qhat{i + 1}"] = float(controller.x[i])
            row[f"vhat{i + 1}"] = float(controller.x[i + 6])
            row[f"qref{i + 1}"] = float(q_reference[i])

        self.rows.append(row)

    def save(self, task, controller, block_ekf, camera_matrix):
        self.directory.mkdir(parents=True, exist_ok=True)

        if self.rows:
            with (self.directory / "timeseries.csv").open(
                "w", newline=""
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
            "block_ekf_final_position": (
                block_ekf.position.tolist()
                if block_ekf.initialised
                else None
            ),
            "block_ekf_final_velocity": (
                block_ekf.velocity.tolist()
                if block_ekf.initialised
                else None
            ),
            "joint_model_analysis": controller.analysis,
            "depth_noise_std_m": DEPTH_NOISE_STD,
            "depth_dropout_probability": DEPTH_DROPOUT_PROBABILITY,
        }

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
            block_F=block_ekf.F,
            block_H=block_ekf.H,
            block_Q=block_ekf.Q,
            block_R=block_ekf.R,
        )

        print("\nTRIAL SUMMARY")
        print(json.dumps(summary, indent=2))


# ============================================================
# SECTION 8 — MAIN LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--seed", type=int, default=1)
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
    parser.add_argument("--duration", type=float, default=70.0)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/moving_block_pick"),
    )

    args = parser.parse_args()

    if args.depth_noise <= 0.0:
        parser.error(
            "--depth-noise must be greater than zero."
        )

    model_path = args.model.resolve()

    print("Loading XML:", model_path)
    print("MuJoCo version:", mujoco.__version__)

    rng = np.random.default_rng(args.seed)

    model = mujoco.MjModel.from_xml_path(str(model_path))

    physics_steps = int(round(CONTROL_DT / model.opt.timestep))

    if not np.isclose(
        physics_steps * model.opt.timestep,
        CONTROL_DT,
    ):
        raise ValueError(
            "Physics timestep must divide the control period exactly."
        )

    home_id = require_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    plant = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, plant, home_id)
    mujoco.mj_forward(model, plant)

    # Identify block free-joint velocity DOFs.
    block_joint_id = require_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "target_block_joint",
    )
    block_velocity_address = model.jnt_dofadr[block_joint_id]

    kinematics = Kinematics(model, home_id)

    q_initial_measured = (
        plant.qpos[kinematics.q_addresses].copy()
        + rng.normal(0.0, ENCODER_SIGMA, 6)
    )

    controller = EstimatedJointControl(
        model,
        kinematics,
        q_initial_measured,
    )

    block_ekf = BlockConstantVelocityEKF(dt=VISION_DT)

    task = MovingBlockPickTask(
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

    print("MODEL ANALYSIS:", controller.analysis)

    sensors = None
    next_vision = 0.0
    block_stop_handled = False

    try:
        sensors = Sensors(model, kinematics, rng)

        print("Camera matrix:\n", sensors.calibration.K)
        print("Block speed:", BLOCK_SPEED, "m/s")
        print("Block movement duration:", BLOCK_MOVE_DURATION, "s")
        print("Depth noise std:", args.depth_noise, "m")

        with mujoco.viewer.launch_passive(
            model,
            plant,
        ) as viewer:
            viewer.cam.lookat[:] = [0.25, 0.0, 0.25]
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

            while (
                viewer.is_running()
                and plant.time < args.duration
            ):
                step_start = time.perf_counter()
                t = float(plant.time)

                # Block moves for a fixed period, then stops.
                block_is_moving = t < BLOCK_MOVE_DURATION

                if block_is_moving:
                    # Drive the block only during its moving phase.
                    plant.qvel[
                        block_velocity_address: block_velocity_address + 3
                    ] = BLOCK_SPEED
                elif not block_stop_handled:
                    # Negate / zero the block velocity exactly once
                    # when the moving phase ends.
                    plant.qvel[
                        block_velocity_address: block_velocity_address + 3
                    ] = 0.0

                    # Also stop the EKF's constant-velocity prediction.
                    block_ekf.stop_block()

                    block_stop_handled = True

                if t >= next_vision:
                    mujoco.mj_forward(model, plant)

                    rgb, depth = sensors.capture_rgbd(
                        plant,
                        args.depth_noise,
                        args.dropout,
                    )

                    grasp_measurement, mask = sensors.find_target(
                        rgb,
                        depth,
                        controller.x[:6],
                    )

                    # While the block is moving, update the EKF.
                    if block_is_moving:
                        block_ekf.predict()
                        block_ekf.update(grasp_measurement)

                    # Always store the latest raw RGB-D measurement.
                    # After the block stops, this is used to lock the
                    # descent target.
                    task.update_vision(grasp_measurement, t)

                    display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    display[mask > 0] = [0, 255, 0]

                    cv2.putText(
                        display,
                        f"{task.state} | EKF tracking",
                        (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow(
                        "Wrist RGB: green block detection",
                        display,
                    )

                    next_vision = t + VISION_DT

                q_reference, finger_target = task.update(
                    t,
                    controller.x[:6],
                    block_ekf,
                    block_is_moving,
                )

                command = controller.update(
                    sensors.encoder(plant),
                    q_reference,
                )

                plant.ctrl[kinematics.actuator_ids] = command
                plant.ctrl[finger_actuator_ids] = finger_target

                plant.qfrc_applied[:] = 0.0
                plant.qfrc_applied[kinematics.v_addresses] = (
                    controller.estimated_bias_compensation()
                )

                evaluation.record(
                    plant,
                    controller,
                    block_ekf,
                    q_reference,
                    task.state,
                )

                for _ in range(physics_steps):
                    mujoco.mj_step(model, plant)

                viewer.sync()

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
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
                block_ekf,
                sensors.calibration.K,
            )


if __name__ == "__main__":
    main()