"""Part A: rebuilds the joint model from moving_block_pick.py and analyses it.
Run: python final_simulation/model_analysis.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.signal import cont2discrete

DT = 0.01
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
NAMES = ["pan", "lift", "elbow", "wrist 1", "wrist 2", "wrist 3"]
ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "static" / "images"

Model = mujoco.MjModel.from_xml_path(str(ROOT / "universal_robots_ur3e" / "ur3e.xml"))
Data = mujoco.MjData(Model)
Home = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_KEY, "home")
Joint_Ids = [mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
Q_Addr, V_Addr = Model.jnt_qposadr[Joint_Ids], Model.jnt_dofadr[Joint_Ids]
Act = [mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATORS]
Kp = Model.actuator_gainprm[Act, 0]
Kd = -Model.actuator_biasprm[Act, 2]
Torque_Limit = Model.actuator_forcerange[Act, 1]
mujoco.mj_resetDataKeyframe(Model, Data, Home)
Q_Home = Data.qpos[Q_Addr].copy()


def mass_matrix(Q):
    mujoco.mj_resetDataKeyframe(Model, Data, Home)
    Data.qpos[Q_Addr] = Q
    mujoco.mj_forward(Model, Data)
    M = np.zeros((Model.nv, Model.nv))
    mujoco.mj_fullM(Model, Data, M)
    return M[np.ix_(V_Addr, V_Addr)]


def joint_model(Q):
    """Same model as moving_block_pick.py: x = [q, qd], u = commanded angles."""
    M = mass_matrix(Q)
    Ac = np.block([[np.zeros((6, 6)), np.eye(6)],
                   [-np.linalg.solve(M, np.diag(Kp)), -np.linalg.solve(M, np.diag(Kd))]])
    Bc = np.vstack([np.zeros((6, 6)), np.linalg.solve(M, np.diag(Kp))])
    C = np.hstack([np.eye(6), np.zeros((6, 6))])
    A, B, C, _, _ = cont2discrete((Ac, Bc, C, None), DT)
    return A, B, C, Ac, M


def lqr(A, B):
    Q, R = np.diag([400.0] * 6 + [1.0] * 6), 0.5 * np.eye(6)
    S = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)


def ctrb(A, B):
    return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(len(A))])


def obsv(A, C):
    return np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(len(A))])


def radius(A):
    return float(max(abs(np.linalg.eigvals(A))))


def mujoco_step(Q0, Step, N=150):
    """Nonlinear arm with gravity compensation, command = Q0 + Step."""
    D = mujoco.MjData(Model)
    mujoco.mj_resetDataKeyframe(Model, D, Home)
    D.qpos[Q_Addr] = Q0
    D.ctrl[Act] = Q0
    mujoco.mj_forward(Model, D)
    Q, Tau = [D.qpos[Q_Addr].copy()], []
    for _ in range(N):
        D.ctrl[Act] = Q0 + Step
        D.qfrc_applied[:] = 0
        D.qfrc_applied[V_Addr] = D.qfrc_bias[V_Addr]
        for _ in range(round(DT / Model.opt.timestep)):
            mujoco.mj_step(Model, D)
        Q.append(D.qpos[Q_Addr].copy())
        Tau.append(D.actuator_force[Act].copy())
    return np.array(Q), np.array(Tau)


def linear_step(A, B, Q0, Step, N=150):
    X, Q = np.zeros(12), [Q0.copy()]
    for _ in range(N):
        X = A @ X + B @ Step
        Q.append(Q0 + X[:6])
    return np.array(Q)


def rise_settle(T, Y):
    """Y normalised 0 -> 1. Returns 10-90% rise time and 2% settling time."""
    Rise = T[np.argmax(Y >= 0.9)] - T[np.argmax(Y >= 0.1)]
    Settle = T[np.nonzero(abs(Y - 1) > 0.02)[0][-1] + 1]
    return round(float(Rise), 2), round(float(Settle), 2)


A, B, C, Ac, M = joint_model(Q_Home)
K = lqr(A, B)
R = {"kp": Kp.tolist(), "kd": Kd.tolist(), "torque_limit": Torque_Limit.tolist(),
     "M_diag": np.diag(M).round(3).tolist(),
     "armature_share_pct": (100 * Model.dof_armature[V_Addr] / np.diag(M)).round(1).tolist(),
     "M_max_offdiag": float(np.max(abs(M - np.diag(np.diag(M))) / np.sqrt(np.outer(np.diag(M), np.diag(M))))),
     "lqr_gain_diag": np.diag(K[:, :6]).round(2).tolist(),
     "lqr_radius": radius(A - B @ K),
     "eigs_continuous": sorted(np.linalg.eigvals(Ac).real.round(2).tolist(), reverse=True)}

# Controllability / observability: rank, PBH, and Gramians in physical units
# (inputs scaled by the 0.15 rad command limit, outputs by 0.5 mrad encoder noise).
R["rank_ctrb"] = int(np.linalg.matrix_rank(ctrb(A, B)))
R["rank_obsv"] = int(np.linalg.matrix_rank(obsv(A, C)))
R["pbh_min"] = int(min(min(np.linalg.matrix_rank(np.hstack([L * np.eye(12) - A, B])),
                           np.linalg.matrix_rank(np.vstack([L * np.eye(12) - A, C])))
                       for L in np.linalg.eigvals(A)))
Wc = solve_discrete_lyapunov(A, 0.15 ** 2 * B @ B.T)
Wo = solve_discrete_lyapunov(A.T, C.T @ C / 0.0005 ** 2)
for Key, W in [("ctrb", Wc), ("obsv", Wo)]:
    E, V = np.linalg.eigh(W)
    i = int(np.argmax(abs(V[:, 0])))
    R[f"gramian_{Key}_condition"] = float(E[-1] / E[0])
    R[f"gramian_{Key}_weakest"] = ("q " if i < 6 else "qd ") + NAMES[i % 6]

R["lost_encoder"] = []
for j in range(6):
    Cj = np.delete(C, j, axis=0)
    Sv = np.linalg.svd(obsv(A, Cj), compute_uv=False)
    Woj = solve_discrete_lyapunov(A.T, Cj.T @ Cj / 0.0005 ** 2)
    R["lost_encoder"].append({"joint": NAMES[j], "rank": int(np.linalg.matrix_rank(obsv(A, Cj))),
                              "sv_ratio": float(Sv[-1] / Sv[0]),
                              "best_std_mrad": float(np.sqrt(abs(np.linalg.pinv(Woj)[j, j])) * 1000)})

# Pose dependence: model frozen at home vs 300 random poses within +-0.6 rad
Rng = np.random.default_rng(0)
Slowest, Home_Gain, Redesigned, M_Change = [], [], [], []
for _ in range(300):
    Ap, Bp, _, Acp, Mp = joint_model(np.clip(Q_Home + Rng.uniform(-0.6, 0.6, 6), -6.28, 6.28))
    Slowest.append(max(np.linalg.eigvals(Acp).real))
    Home_Gain.append(radius(Ap - Bp @ K))
    Redesigned.append(radius(Ap - Bp @ lqr(Ap, Bp)))
    M_Change.append(100 * max(abs(np.diag(Mp) / np.diag(M) - 1)))
R["pose_M_change_pct_median_max"] = [float(np.median(M_Change)), float(max(M_Change))]
R["pose_slowest_min_max"] = [float(min(Slowest)), float(max(Slowest))]
R["pose_radius_home_gain_max"] = max(Home_Gain)

# Cube constant-velocity model (Part C settings) and growth of uncertainty when blind
F = np.block([[np.eye(3), 0.1 * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
H = np.hstack([np.eye(3), np.zeros((3, 3))])
Qc = np.diag([0.001 ** 2] * 3 + [0.015 ** 2] * 3)
R["cube_rank_obsv"] = int(np.linalg.matrix_rank(obsv(F, H)))
R["cube_rank_obsv_xy_only"] = int(np.linalg.matrix_rank(obsv(F, H[:2])))
P = solve_discrete_are(F.T, H.T, Qc, np.diag([0.006 ** 2, 0.006 ** 2, 0.005 ** 2]))
R["cube_std_mm"] = {"0.0s": float(np.sqrt(P[0, 0]) * 1000)}
for k in range(1, 31):
    P = F @ P @ F.T + Qc
    if k in (5, 10, 20):
        R["cube_std_mm"][f"{k / 10}s"] = float(np.sqrt(P[0, 0]) * 1000)

# Validation: linear model vs MuJoCo for steps on lift (heavy) and wrist 2 (light)
T = np.arange(151) * DT
Q_Ext = Q_Home + np.array([0, 0.5, -0.6, 0, 0, 0])
R["validation"] = []
Fig, Ax = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for c, (Pose, Q0, Size) in enumerate([("home", Q_Home, 0.05), ("home", Q_Home, 0.30), ("extended", Q_Ext, 0.05)]):
    for Jt, Colour in [(1, "C0"), (4, "C3")]:
        Step = np.zeros(6)
        Step[Jt] = Size
        Q_Mj, Tau = mujoco_step(Q0, Step)
        Y_Mj = (Q_Mj[:, Jt] - Q0[Jt]) / Size
        Y_Lin = (linear_step(A, B, Q0, Step)[:, Jt] - Q0[Jt]) / Size   # always the home model
        Y_First = 1 - np.exp(-Kp[Jt] / Kd[Jt] * T)                      # qd = (Kp/Kd)(u - q)
        Err, Err_First = 100 * (Y_Mj - Y_Lin), 100 * (Y_Mj - Y_First)
        R["validation"].append({
            "pose": Pose, "joint": NAMES[Jt], "step": Size,
            "rms_pct": float(np.sqrt(np.mean(Err ** 2))), "peak_pct": float(max(abs(Err))),
            "first_order_rms_pct": float(np.sqrt(np.mean(Err_First ** 2))),
            "first_order_peak_pct": float(max(abs(Err_First))),
            "other_joints_pct": float(100 * np.max(abs(np.delete(Q_Mj - Q0, Jt, axis=1))) / Size),
            "peak_torque_pct": float(100 * max(abs(Tau[:, Jt])) / Torque_Limit[Jt]),
            "rise_settle_mujoco": rise_settle(T, Y_Mj), "rise_settle_model": rise_settle(T, Y_Lin)})
        Ax[0, c].plot(T, Y_Mj, color=Colour, lw=2.5, alpha=0.5, label=f"{NAMES[Jt]}: MuJoCo")
        Ax[0, c].plot(T, Y_Lin, "k--", lw=1)
        Ax[1, c].plot(T, Err, color=Colour, label=f"{NAMES[Jt]}: 12-state model")
        Ax[1, c].plot(T, Err_First, color=Colour, ls=":", label=f"{NAMES[Jt]}: first-order model")
    Ax[0, c].plot([], [], "k--", lw=1, label="12-state linear model")
    Ax[0, c].set_title(f"{Size:.2f} rad step, {Pose} pose", fontsize=10)
    Ax[0, c].legend(fontsize=8, loc="lower right")
    Ax[1, c].legend(fontsize=7, loc="upper right")
    Ax[1, c].set_xlabel("time (s)")
for a in Ax.flat:
    a.grid(alpha=0.3)
Ax[0, 0].set_ylabel("normalised joint angle")
Ax[1, 0].set_ylabel("MuJoCo minus model (% of step)")
Fig.suptitle("Linear joint model vs nonlinear MuJoCo arm (open loop, gravity compensated)")
Fig.tight_layout()
Fig.savefig(IMAGES / "01_model_step_validation.png", dpi=150)

Fig, Ax = plt.subplots(1, 2, figsize=(11, 4))
Ax[0].hist(Slowest, bins=30)
Ax[0].axvline(max(np.linalg.eigvals(Ac).real), color="k", ls="--", label="home pose")
Ax[0].set(xlabel="slowest open-loop eigenvalue, real part (rad/s)", ylabel="number of poses",
          title="Slowest open-loop mode across 300 poses (home ± 0.6 rad)")
Ax[1].hist(Home_Gain, bins=30, color="C1", alpha=0.8, label="home-pose LQR gain")
Ax[1].hist(Redesigned, bins=30, color="C2", alpha=0.6, label="gain redesigned per pose")
Ax[1].axvline(R["lqr_radius"], color="k", ls="--", label="home pose")
Ax[1].set(xlabel="closed-loop spectral radius |λ|max (stable if < 1)", title="Is the home-pose LQR still good elsewhere?")
for a in Ax:
    a.legend(fontsize=8)
Fig.tight_layout()
Fig.savefig(IMAGES / "03_model_pose_dependence.png", dpi=150)

(Path(__file__).parent / "model_analysis_results.json").write_text(json.dumps(R, indent=2))
print(json.dumps(R, indent=2))
