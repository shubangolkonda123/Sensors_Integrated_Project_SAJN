"""Part B: frame diagram at the home pose, and how joint-angle error moves the cube estimate.
Run: python final_simulation/frame_diagram.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
NAMES = ["pan", "lift", "elbow", "wrist 1", "wrist 2", "wrist 3"]
OPTICAL = np.diag([1.0, -1.0, -1.0])   # MuJoCo camera axes -> optical axes (x right, y down, z forward)

Model = mujoco.MjModel.from_xml_path(str(ROOT / "universal_robots_ur3e" / "ur3e.xml"))
Data = mujoco.MjData(Model)
mujoco.mj_resetDataKeyframe(Model, Data, mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_KEY, "home"))
Q_Addr = Model.jnt_qposadr[[mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]]
Cam = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
Flange = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
Grasp = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
Q_Home = Data.qpos[Q_Addr].copy()


def make_T(R, P):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, P
    return T


def frames(Q):
    """Base-frame poses of the flange, camera (optical) and grasp point for joint angles Q."""
    Data.qpos[Q_Addr] = Q
    mujoco.mj_forward(Model, Data)
    return {"F": make_T(Data.site_xmat[Flange].reshape(3, 3), Data.site_xpos[Flange]),
            "C": make_T(Data.cam_xmat[Cam].reshape(3, 3) @ OPTICAL, Data.cam_xpos[Cam]),
            "T": make_T(Data.site_xmat[Grasp].reshape(3, 3), Data.site_xpos[Grasp])}


T_Home = frames(Q_Home)
Cube = np.array([0.30, 0.10, 0.262])            # grasp point: table (z = 0.225) + 37 mm
T_FC = np.linalg.inv(T_Home["F"]) @ T_Home["C"]  # camera on the flange: fixed by the mount
T_FT = np.linalg.inv(T_Home["F"]) @ T_Home["T"]

# A cube point seen by the camera is mapped to the base with the ESTIMATED joints:
# p_B = T_BC(q_hat) @ p_C. Joint error dq therefore moves the estimate by J dq.
P_Cam = np.linalg.inv(T_Home["C"]) @ np.append(Cube, 1)
J = np.zeros((3, 6))
for j in range(6):
    dq = np.zeros(6)
    dq[j] = 1e-4
    J[:, j] = ((frames(Q_Home + dq)["C"] @ P_Cam)[:3] - Cube) / 1e-4
frames(Q_Home)
R = {"T_BC_home": T_Home["C"].round(4).tolist(), "T_FC": T_FC.round(4).tolist(), "T_FT": T_FT.round(4).tolist(),
     "camera_to_cube_m": float(np.linalg.norm(P_Cam[:3])),
     "mm_per_mrad": dict(zip(NAMES, np.linalg.norm(J, axis=0).round(3).tolist()))}
for Label, Sigma in [("filter_0.18mrad", 0.00018), ("raw_encoder_0.5mrad", 0.0005)]:
    R[f"cube_std_mm_{Label}"] = float(np.sqrt(np.trace(J @ J.T) * Sigma ** 2) * 1000)
R["cube_shift_mm_lift_20mrad"] = float(np.linalg.norm(J[:, 1]) * 0.020 * 1000)

# Figure: whole scene (left) and the wrist close up (right)
Fig = plt.figure(figsize=(13, 6.5))
Bodies = ["base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "gripper"]
Arm = np.array([Data.xpos[mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_BODY, b)] for b in Bodies])
Frames = [("{B} base", np.eye(4), 0.08), ("{F} flange", T_Home["F"], 0.03), ("{C} camera", T_Home["C"], 0.03),
          ("{T} grasp point", T_Home["T"], 0.03), ("{O} cube", make_T(np.eye(3), Cube), 0.04)]
Wrist = T_Home["C"][:3, 3]
for i, (Title, Centre, Half) in enumerate([("Whole scene", np.array([0.12, 0.02, 0.3]), 0.42),
                                           ("Wrist close up", Wrist, 0.08)]):
    Ax = Fig.add_subplot(1, 2, i + 1, projection="3d")
    if i == 0:
        Ax.plot(*Arm.T, "o-", color="grey", lw=4, ms=5, alpha=0.5)
        X, Y = np.meshgrid([0.15, 0.55], [-0.25, 0.25])
        Ax.plot_surface(X, Y, np.full_like(X, 0.225), color="tan", alpha=0.3)
    Ax.plot(*np.vstack([Wrist, Cube]).T, "m--", lw=1.5, label="camera ray to cube")
    for Name, T, L in Frames:
        L = L if i == 0 else L / 2
        for k, Colour in enumerate("rgb"):
            Ax.quiver(*T[:3, 3], *(L * T[:3, k]), color=Colour, lw=2)
        if i == 1 or Name[1] in "BO":
            Ax.text(*(T[:3, 3] + [0, 0, 0.012 if i else 0.03]), Name, fontsize=9)
    Ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=Title,
           xlim=Centre[0] + [-Half, Half], ylim=Centre[1] + [-Half, Half], zlim=Centre[2] + [-Half, Half])
    Ax.view_init(elev=22, azim=-60)
    Ax.legend(loc="upper left", fontsize=8)
Fig.suptitle("Frames at the home pose (red = x, green = y, blue = z)")
Fig.tight_layout()
Fig.savefig(ROOT / "static" / "images" / "06_frame_diagram.png", dpi=150)

(Path(__file__).parent / "frame_results.json").write_text(json.dumps(R, indent=2))
print(json.dumps(R, indent=2))
