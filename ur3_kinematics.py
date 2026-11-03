"""
ur3_kinematics.py - forward kinematics and Jacobian for the UR3 / UR3e.

Uses the standard Denavit-Hartenberg parameters published by Universal Robots.
This is the same model you should use in the MuJoCo simulation, so the sim and
hardware share one kinematic model (good for the sim-vs-hardware comparison).

NOTE on frames: the DH "base" frame below is rotated 180 deg about z compared with
the "Base" frame shown on the UR pendant (x and y have opposite signs). Our code
only ever uses this DH frame, computed from the joint angles, so it is consistent.
Just remember it if you compare numbers against the pendant screen.
"""
import numpy as np

DH_PARAMS = {
    # d [m], a [m], alpha [rad] for joints 1..6 (standard DH)
    "UR3": dict(
        D=[0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819],
        A=[0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0],
        Alpha=[np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0],
    ),
    "UR3e": dict(
        D=[0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921],
        A=[0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0],
        Alpha=[np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0],
    ),
}


def dh_transform(A, Alpha, D, Theta):
    """One standard-DH link transform T_{i-1,i}."""
    ct, st = np.cos(Theta), np.sin(Theta)
    ca, sa = np.cos(Alpha), np.sin(Alpha)
    return np.array([
        [ct, -st * ca, st * sa, A * ct],
        [st, ct * ca, -ct * sa, A * st],
        [0.0, sa, ca, D],
        [0.0, 0.0, 0.0, 1.0],
    ])


class UR3_Kinematics:
    def __init__(self, Model="UR3", T_Flange_Tool=None):
        P = DH_PARAMS[Model]
        self.D = np.array(P["D"])
        self.A = np.array(P["A"])
        self.Alpha = np.array(P["Alpha"])
        # Tool = gripper fingertip centre, fixed offset from the flange
        self.T_Flange_Tool = np.eye(4) if T_Flange_Tool is None else T_Flange_Tool

    def fk_chain(self, Q):
        """Return [T_0_0, T_0_1, ..., T_0_6]: every joint frame in the base frame."""
        T = np.eye(4)
        Frames = [T.copy()]
        for i in range(6):
            T = T @ dh_transform(self.A[i], self.Alpha[i], self.D[i], Q[i])
            Frames.append(T.copy())
        return Frames

    def fk_flange(self, Q):
        """T_Base_Flange (the tool flange, where the camera bracket mounts)."""
        return self.fk_chain(Q)[-1]

    def fk_tool(self, Q):
        """T_Base_Tool (gripper fingertip centre)."""
        return self.fk_flange(Q) @ self.T_Flange_Tool

    def jacobian_tool(self, Q):
        """
        6x6 geometric Jacobian of the TOOL point, expressed in the base frame.
        [v; w] = J @ qd, where v = linear velocity of the tool point, w = angular velocity.
        Column i:  linear part  = z_{i-1} x (p_tool - p_{i-1})
                   angular part = z_{i-1}
        """
        Frames = self.fk_chain(Q)
        P_Tool = (Frames[-1] @ self.T_Flange_Tool)[:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            Z = Frames[i][:3, 2]
            P = Frames[i][:3, 3]
            J[:3, i] = np.cross(Z, P_Tool - P)
            J[3:, i] = Z
        return J
