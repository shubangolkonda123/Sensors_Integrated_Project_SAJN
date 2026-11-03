"""
joint_model.py - state-space model of ONE UR3 joint under speedJ control.   (Part A)

Physical reasoning:
  speedJ sends a velocity command u. The UR's own servo loop makes the joint speed
  follow u, approximately like a first-order lag with time constant tau:
        qdd = (u - qd) / tau
  and of course  qd = d/dt q.

State x = [q, qd]   (joint angle, joint speed)
Input u = commanded joint speed (what we send with speedJ)
Output y = q        (encoder)

Continuous time:
    A = [[0, 1],          B = [[0],          C = [1, 0]
         [0, -1/tau]]          [1/tau]]

All 6 joints share this model (tau identified with identify_joint_lag.py).
Run this file directly to print the controllability / observability analysis.
"""
import numpy as np
from scipy.linalg import expm


def continuous_model(Tau):
    A = np.array([[0.0, 1.0], [0.0, -1.0 / Tau]])
    B = np.array([[0.0], [1.0 / Tau]])
    C = np.array([[1.0, 0.0]])
    return A, B, C


def discrete_model(Tau, Dt):
    """Exact zero-order-hold discretisation (the command is held for Dt by speedJ)."""
    A, B, C = continuous_model(Tau)
    # Trick: expm of the augmented matrix gives both Ad and Bd in one go
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2:] = B
    Md = expm(M * Dt)
    return Md[:2, :2], Md[:2, 2:], C


def analyse(Tau, Dt):
    A, B, C = continuous_model(Tau)
    Ad, Bd, _ = discrete_model(Tau, Dt)
    Ctrb = np.hstack([Bd, Ad @ Bd])
    Obsv = np.vstack([C, C @ Ad])
    Report = {
        "A": A, "B": B, "C": C, "Ad": Ad, "Bd": Bd,
        "open_loop_eigs": np.linalg.eigvals(A),
        "ctrb_rank": np.linalg.matrix_rank(Ctrb),
        "obsv_rank": np.linalg.matrix_rank(Obsv),
        "ctrb_cond": np.linalg.cond(Ctrb),
        "obsv_cond": np.linalg.cond(Obsv),
    }
    # Observability from VELOCITY only (C = [0, 1]): q is NOT observable that way,
    # which is why we must use the encoder angle, not just the speed.
    C_Vel = np.array([[0.0, 1.0]])
    Report["obsv_rank_from_qd_only"] = np.linalg.matrix_rank(np.vstack([C_Vel, C_Vel @ Ad]))
    return Report


if __name__ == "__main__":
    import config as Cfg
    R = analyse(Cfg.JOINT_LAG_TAU, Cfg.DT)
    np.set_printoptions(precision=5, suppress=True)
    print(f"tau = {Cfg.JOINT_LAG_TAU} s, dt = {Cfg.DT} s")
    print("A =\n", R["A"], "\nB =\n", R["B"])
    print("Ad =\n", R["Ad"], "\nBd =\n", R["Bd"])
    print("open-loop eigenvalues:", R["open_loop_eigs"], " (0 = integrator, -1/tau = servo lag)")
    print(f"controllability rank = {R['ctrb_rank']}/2   (cond {R['ctrb_cond']:.1f})")
    print(f"observability rank   = {R['obsv_rank']}/2   from q (encoder)   (cond {R['obsv_cond']:.1f})")
    print(f"observability rank   = {R['obsv_rank_from_qd_only']}/2   from qd only -> angle not observable")
