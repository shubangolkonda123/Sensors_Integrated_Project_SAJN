"""
calibrate_handeye.py - finds T_Flange_Cam: where the camera sits on the wrist.   (Part B)

"Eye-in-hand" calibration. A checkerboard is fixed on the table (it must NOT move).
You put the robot in freedrive, move it to ~15-20 different poses where the camera
sees the whole board (vary the angles a lot, tilt 15-30 deg in different directions),
and press SPACE at each one. For each pose we record:
    T_Base_Flange   from the joint angles (forward kinematics)
    T_Cam_Board     from the image (solvePnP on the checkerboard corners)
Then OpenCV solves  AX = XB  for X = T_Flange_Cam.

We run FOUR different solvers and compare them: if they agree to ~1-2 mm the
calibration is trustworthy. We also report how consistently the board position is
predicted across all poses (it should be the same point every time).

FIXED CAMERA ("eye-to-hand", our setup, use --fixed):
    The camera stays on its stand and a SMALL checkerboard is taped flat and rigid
    to the gripper (it must not wobble). The same maths works with the robot poses
    inverted:
        inv(T_Base_Flange_i) @ T_Base_Cam @ T_Cam_Board_i = T_Flange_Board  (constant)
    so the solver returns X = T_Base_Cam, and the "spread" is how consistently the
    board position on the gripper is predicted (should be a few mm).
    Saved to calib/eye_to_hand.npz.

Usage:
    python calibrate_handeye.py --selftest                 # wrist-camera maths check (no robot)
    python calibrate_handeye.py --selftest --fixed         # fixed-camera maths check (no robot)
    python calibrate_handeye.py --fixed --cols 5 --rows 4 --square 0.02   # real robot + RealSense
Keys during capture:  SPACE = capture pose,  c = compute & save,  q = quit
"""
import argparse
import os
import numpy as np
import cv2
import config as Cfg
from transforms import make_T, inv_T, rotvec_to_R
from ur3_kinematics import UR3_Kinematics

METHODS = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_park_numpy(T_Base_Flange_List, T_Cam_Board_List):
    """
    Our own implementation of Park & Martin (1994), independent of OpenCV.
    The board is fixed, so for any two poses i, j:
        T_Base_Flange_i X T_Cam_Board_i = T_Base_Flange_j X T_Cam_Board_j
    =>  A X = X B   with  A = inv(Tb_j) Tb_i   (flange motion)
                          B = Tc_j inv(Tc_i)   (camera motion)
    Rotation:    M = sum(beta alpha^T), alpha = log(R_A), beta = log(R_B);  R_X = (M^T M)^(-1/2) M^T
    Translation: (R_A - I) t_X = R_X t_B - t_A, stacked over all pairs, least squares.
    """
    from transforms import R_to_rotvec
    N = len(T_Base_Flange_List)
    Pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    A_List, B_List = [], []
    for i, j in Pairs:
        A_List.append(inv_T(T_Base_Flange_List[j]) @ T_Base_Flange_List[i])
        B_List.append(T_Cam_Board_List[j] @ inv_T(T_Cam_Board_List[i]))
    M = np.zeros((3, 3))
    for A, B in zip(A_List, B_List):
        M += np.outer(R_to_rotvec(B[:3, :3]), R_to_rotvec(A[:3, :3]))
    W, V = np.linalg.eigh(M.T @ M)
    R_X = V @ np.diag(1 / np.sqrt(W)) @ V.T @ M.T
    C = np.vstack([A[:3, :3] - np.eye(3) for A in A_List])
    D = np.hstack([R_X @ B[:3, 3] - A[:3, 3] for A, B in zip(A_List, B_List)])
    t_X = np.linalg.lstsq(C, D, rcond=None)[0]
    return make_T(R_X, t_X)


def solve_handeye(T_Base_Flange_List, T_Cam_Board_List):
    """Returns {method: (T_Flange_Cam, spread_mean, spread_max)}."""
    R_G = [T[:3, :3] for T in T_Base_Flange_List]
    t_G = [T[:3, 3] for T in T_Base_Flange_List]
    R_T = [T[:3, :3] for T in T_Cam_Board_List]
    t_T = [T[:3, 3] for T in T_Cam_Board_List]
    Solutions = {"Park (ours)": solve_park_numpy(T_Base_Flange_List, T_Cam_Board_List)}
    if hasattr(cv2, "calibrateHandEye"):          # missing from some OpenCV 5 builds
        for Name, Method in METHODS.items():
            R, t = cv2.calibrateHandEye(R_G, t_G, R_T, t_T, method=Method)
            Solutions[Name] = make_T(R, t.ravel())
    else:
        print("[handeye] cv2.calibrateHandEye not available, using our Park solver only")
    Results = {}
    for Name, X in Solutions.items():
        # Consistency: board position in the base frame predicted from every pose.
        # Perfect calibration -> all identical. Spread = calibration quality.
        Board_In_Base = np.array([(Tb @ X @ Tc)[:3, 3] for Tb, Tc in zip(T_Base_Flange_List, T_Cam_Board_List)])
        Spread = np.linalg.norm(Board_In_Base - Board_In_Base.mean(0), axis=1)
        Results[Name] = (X, float(Spread.mean()), float(Spread.max()))
    return Results


def report_and_save(Results, Path, Key="T_Flange_Cam"):
    print("\nMethod        tx[mm]   ty[mm]   tz[mm]   board spread mean/max [mm]")
    for Name, (X, Mean, Max) in Results.items():
        print(f"{Name:12s} {X[0, 3] * 1000:8.2f} {X[1, 3] * 1000:8.2f} {X[2, 3] * 1000:8.2f}"
              f"      {Mean * 1000:.2f} / {Max * 1000:.2f}")
    Best = min(Results, key=lambda k: Results[k][1])
    T = Results[Best][0]
    Ts = np.array([R[0][:3, 3] for R in Results.values()])
    print(f"\nSolvers disagree by up to {np.ptp(Ts, axis=0).max() * 1000:.2f} mm in translation.")
    print(f"Saving '{Best}' (lowest board spread) to {Path}")
    os.makedirs(os.path.dirname(Path) or ".", exist_ok=True)
    np.savez(Path, **{Key: T}, Method=Best, All={k: v[0] for k, v in Results.items()})
    return T


def solve_eye_to_hand(T_Base_Flange_List, T_Cam_Board_List):
    """Fixed camera: same solvers, robot poses inverted. Returns {method: (T_Base_Cam, mean, max)}."""
    return solve_handeye([inv_T(T) for T in T_Base_Flange_List], T_Cam_Board_List)


def selftest_fixed():
    """Fake poses with a KNOWN fixed camera + board on the gripper; solvers must recover it."""
    Rng = np.random.default_rng(2)
    Kin = UR3_Kinematics(Cfg.ROBOT_MODEL)
    T_Base_Cam_True = Cfg.T_BASE_CAM_DEFAULT
    T_Flange_Board = make_T(rotvec_to_R([0.0, 0.3, 0.0]), [0.0, 0.05, 0.08])
    A, B = [], []
    for _ in range(18):
        Q = Cfg.HOME_Q + Rng.normal(0, 0.25, 6)
        Tb = Kin.fk_flange(Q)
        Tc = inv_T(T_Base_Cam_True) @ Tb @ T_Flange_Board
        Tc = Tc @ make_T(rotvec_to_R(Rng.normal(0, 0.003, 3)), Rng.normal(0, 0.0007, 3))
        A.append(Tb)
        B.append(Tc)
    Res = solve_eye_to_hand(A, B)
    for Name, (X, Mean, _) in Res.items():
        Err = np.linalg.norm(X[:3, 3] - T_Base_Cam_True[:3, 3]) * 1000
        print(f"selftest --fixed {Name:11s}: camera position error {Err:.2f} mm, spread {Mean * 1000:.2f} mm")
    print("selftest passed" if all(np.linalg.norm(v[0][:3, 3] - T_Base_Cam_True[:3, 3]) < 0.006
                                   for v in Res.values()) else "selftest FAILED")


def selftest():
    """Fake poses with a KNOWN camera mount + noise; the solvers must recover it."""
    Rng = np.random.default_rng(1)
    Kin = UR3_Kinematics(Cfg.ROBOT_MODEL)
    X_True = make_T(rotvec_to_R([0.02, -0.03, 0.01]), [0.004, -0.068, 0.032])
    T_Base_Board = make_T(rotvec_to_R([0, 0, 0.3]), [-0.25, -0.12, 0.0])
    A, B = [], []
    for _ in range(18):
        Q = Cfg.HOME_Q + Rng.normal(0, 0.15, 6)
        Tb = Kin.fk_flange(Q)
        Tc = inv_T(Tb @ X_True) @ T_Base_Board
        # add realistic noise: 0.3 mm / 0.1 deg on the board pose
        Tc = Tc @ make_T(rotvec_to_R(Rng.normal(0, 0.002, 3)), Rng.normal(0, 0.0003, 3))
        A.append(Tb)
        B.append(Tc)
    Res = solve_handeye(A, B)
    for Name, (X, _, _) in Res.items():
        Err = np.linalg.norm(X[:3, 3] - X_True[:3, 3]) * 1000
        print(f"selftest {Name:11s}: translation error {Err:.2f} mm")
    print("selftest passed" if all(np.linalg.norm(v[0][:3, 3] - X_True[:3, 3]) < 0.003
                                   for v in Res.values()) else "selftest FAILED")


def capture(Args):
    from robot import RTDE_Robot
    from camera import RealSense_Camera
    Kin = UR3_Kinematics(Cfg.ROBOT_MODEL)
    Robot = RTDE_Robot(Cfg.ROBOT_IP)
    Cam = RealSense_Camera(Cfg.CAMERA_WIDTH, Cfg.CAMERA_HEIGHT, Cfg.CAMERA_FPS)
    K, Dist = Cam.Intrinsics.K(), Cam.Intrinsics.Dist
    # 3D corner positions on the board (board frame, z = 0)
    Obj = np.zeros((Args.cols * Args.rows, 3), np.float32)
    Obj[:, :2] = np.mgrid[0:Args.cols, 0:Args.rows].T.reshape(-1, 2) * Args.square
    A, B = [], []
    Robot.teach_mode(True)
    print("Freedrive ON. Move the arm by hand. SPACE = capture, c = compute, q = quit")
    try:
        while True:
            F = Cam.get_frame()
            if F is None:
                continue
            Img = F[0]
            Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            Found, Corners = cv2.findChessboardCorners(Gray, (Args.cols, Args.rows))
            Show = Img.copy()
            if Found:
                Corners = cv2.cornerSubPix(Gray, Corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
                cv2.drawChessboardCorners(Show, (Args.cols, Args.rows), Corners, Found)
            cv2.putText(Show, f"captured: {len(A)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("hand-eye", Show)
            Key = cv2.waitKey(1) & 0xFF
            if Key == ord(" ") and Found:
                Ok, Rv, Tv = cv2.solvePnP(Obj, Corners, K, Dist)
                if Ok:
                    A.append(Kin.fk_flange(Robot.get_q()))
                    B.append(make_T(cv2.Rodrigues(Rv)[0], Tv.ravel()))
                    print(f"captured pose {len(A)}")
            elif Key == ord("c"):
                if len(A) < 8:
                    print("need at least 8 poses (15-20 is better)")
                    continue
                if Args.fixed:
                    report_and_save(solve_eye_to_hand(A, B), Cfg.EYE_TO_HAND_FILE, Key="T_Base_Cam")
                else:
                    report_and_save(solve_handeye(A, B), Cfg.HANDEYE_FILE)
                break
            elif Key == ord("q"):
                break
    finally:
        Robot.teach_mode(False)
        Robot.close()
        Cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--selftest", action="store_true")
    P.add_argument("--fixed", action="store_true", help="camera on a stand, board on the gripper")
    P.add_argument("--cols", type=int, default=9, help="inner corners along the board width")
    P.add_argument("--rows", type=int, default=6, help="inner corners along the board height")
    P.add_argument("--square", type=float, default=0.025, help="square size in metres (MEASURE it)")
    A = P.parse_args()
    if A.selftest:
        selftest_fixed() if A.fixed else selftest()
    else:
        capture(A)
