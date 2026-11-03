"""
transforms.py - small helpers for 4x4 homogeneous transforms.

Naming convention used everywhere in this project:
    T_A_B  = pose of frame B expressed in frame A
    so  P_A = T_A_B @ P_B   (point in B -> point in A)
    and T_A_C = T_A_B @ T_B_C   (chain them by matching the inner letters)
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R, P):
    """Build a 4x4 transform from a 3x3 rotation R and a 3-vector position P."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = P
    return T


def inv_T(T):
    """Fast inverse of a rigid transform (uses R^T instead of a full matrix inverse)."""
    R = T[:3, :3]
    P = T[:3, 3]
    T_Inv = np.eye(4)
    T_Inv[:3, :3] = R.T
    T_Inv[:3, 3] = -R.T @ P
    return T_Inv


def transform_point(T, P):
    """Apply transform T to a single 3D point P."""
    return T[:3, :3] @ P + T[:3, 3]


def rotvec_to_R(Rot_Vec):
    """Axis-angle vector (what the UR pendant shows as Rx,Ry,Rz) -> rotation matrix."""
    return Rotation.from_rotvec(Rot_Vec).as_matrix()


def R_to_rotvec(R):
    """Rotation matrix -> axis-angle vector. Its length is the rotation angle in radians."""
    return Rotation.from_matrix(R).as_rotvec()


def ur_pose_to_T(Pose):
    """UR pose [x, y, z, rx, ry, rz] (metres, axis-angle) -> 4x4 transform."""
    Pose = np.asarray(Pose, dtype=float)
    return make_T(rotvec_to_R(Pose[3:]), Pose[:3])


def rot_z(Angle):
    c, s = np.cos(Angle), np.sin(Angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def down_orientation_like(R_Current):
    """
    Return a rotation whose z-axis points straight down (-z of the base),
    keeping the current yaw (direction of the tool x-axis in the horizontal plane).
    Used as the grasp orientation so the robot never has to spin the wrist a lot.
    """
    X = R_Current[:, 0].copy()
    X[2] = 0.0
    if np.linalg.norm(X) < 1e-6:          # tool x-axis is vertical: pick any yaw
        X = np.array([1.0, 0.0, 0.0])
    X = X / np.linalg.norm(X)
    Z = np.array([0.0, 0.0, -1.0])
    Y = np.cross(Z, X)
    return np.column_stack([X, Y, Z])
