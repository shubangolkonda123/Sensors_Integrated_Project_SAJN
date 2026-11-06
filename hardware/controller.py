"""
controller.py - the two control loops.                      (Part D)

OUTER loop: PBVS (position-based visual servoing)
    The camera gives the object position. We turn the pose error between the
    gripper and the goal (above / at the object) into a desired tool velocity,
    then into joint speeds with the Jacobian (damped least squares).

INNER loop: per-joint LQR on the state-space joint model x = [q, qd]
    Tracks the joint reference produced by the outer loop, using the OBSERVER's
    state estimate (never the simulator's ground truth).

    u = qd_ref - K (x_hat - x_ref)
    -> if we're exactly on the reference, we just send the feed-forward speed qd_ref.
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from joint_model import discrete_model
from transforms import R_to_rotvec


class Joint_LQR:
    def __init__(self, Tau, Dt, Q, R):
        self.Ad, self.Bd, _ = discrete_model(Tau, Dt)
        P = solve_discrete_are(self.Ad, self.Bd, Q, R)
        self.K = np.linalg.solve(R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)   # 1x2
        self.Closed_Loop_Eigs = np.linalg.eigvals(self.Ad - self.Bd @ self.K)

    def compute(self, X_Hat, Q_Ref, Qd_Ref):
        """X_Hat: 6x2 observer state. Returns commanded joint speeds (6,)."""
        Err = X_Hat - np.column_stack([Q_Ref, Qd_Ref])    # 6x2
        return Qd_Ref - Err @ self.K[0]


class PBVS_Controller:
    def __init__(self, Gain_Pos, Gain_Rot, Max_Lin, Max_Rot):
        self.Gain_Pos, self.Gain_Rot = Gain_Pos, Gain_Rot
        self.Max_Lin, self.Max_Rot = Max_Lin, Max_Rot

    def compute(self, T_Base_Tool, Goal_Pos, Goal_R, Speed_Factor=1.0):
        """Returns tool twist V = [v; w] (base frame) and the position/rotation error sizes.
        Speed_Factor lowers the speed LIMITS only (the gain stays the same, so the
        final convergence near the goal is not slowed down)."""
        E_Pos = Goal_Pos - T_Base_Tool[:3, 3]
        E_Rot = R_to_rotvec(Goal_R @ T_Base_Tool[:3, :3].T)   # rotation still needed, base frame
        V = self.Gain_Pos * E_Pos
        W = self.Gain_Rot * E_Rot
        # Clip speeds while keeping direction
        Max_Lin, Max_Rot = self.Max_Lin * Speed_Factor, self.Max_Rot * Speed_Factor
        if np.linalg.norm(V) > Max_Lin:
            V *= Max_Lin / np.linalg.norm(V)
        if np.linalg.norm(W) > Max_Rot:
            W *= Max_Rot / np.linalg.norm(W)
        return np.hstack([V, W]), np.linalg.norm(E_Pos), np.linalg.norm(E_Rot)


def resolved_rate(J, V, Damping, Max_Joint_Speed):
    """Tool twist -> joint speeds with damped least squares:
         qd = J^T (J J^T + lambda^2 I)^-1 V
    Damping keeps speeds bounded near singularities (costs a little accuracy there)."""
    Qd = J.T @ np.linalg.solve(J @ J.T + Damping ** 2 * np.eye(6), V)
    Peak = np.max(np.abs(Qd))
    if Peak > Max_Joint_Speed:
        Qd *= Max_Joint_Speed / Peak          # scale ALL joints so the path stays straight
    return Qd


def manipulability(J):
    """Yoshikawa measure sqrt(det(J J^T)). Close to 0 = near a singularity."""
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
