"""
main_pick.py - vision-guided pick on the UR3 (real or mock).          (Part E: integration)

Run:
    python main_pick.py --trials 5                     # mock (default switches in config.py)
    python main_pick.py --trials 10 --seed 3           # different random object positions
    python main_pick.py --trials 5 --occlude           # robustness: object hidden during approach
    python main_pick.py --trials 5 --disturb           # robustness: joint bump during approach
    python main_pick.py --trials 5 --noise-scale 3     # robustness: 3x more depth noise than modelled
    python main_pick.py --trials 3 --record --show     # save annotated video / live window

Each trial:
    HOME -> SEARCH -> APPROACH -> DESCEND -> GRASP -> LIFT -> LOWER -> RELEASE -> RETREAT -> DONE
      SEARCH   : robot still, Kalman filter collects measurements until uncertainty is small
      APPROACH : PBVS to a point above the object; camera keeps updating the estimate
      DESCEND  : estimate is FROZEN (locked) and used blind. Wrist camera: too close to see.
                 Fixed camera: the arm arriving above the object can block the view.
      GRASP    : close gripper, check it holds something
      LIFT/LOWER/RELEASE/RETREAT : lift, put it back where it was, let go, move up

Ground truth (mock only) is used ONLY for scoring after the trial. Search this file
for "GROUND TRUTH" to see every place it appears.

Press Ctrl-C at any time: the robot is stopped before the program exits.
On the real robot ALWAYS have a hand on the e-stop / teach pendant.
"""
import argparse
import os
import signal
import time
import numpy as np
import cv2

import config as Cfg
from transforms import transform_point, down_orientation_like
from ur3_kinematics import UR3_Kinematics
from detection import Colour_Depth_Detector, draw_detection
from estimator import Target_KF, Joint_Observer
from controller import Joint_LQR, PBVS_Controller, resolved_rate, manipulability
from hardware import Hardware, load_camera_pose
from trial_logger import Trial_Logger, append_summary

STOP_REQUESTED = False


def _on_ctrl_c(_Sig, _Frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[main] Ctrl-C: stopping robot...")


class Abort(Exception):
    pass


def in_workspace(P):
    return (np.all(P >= Cfg.WORKSPACE_MIN) and np.all(P <= Cfg.WORKSPACE_MAX)
            and np.hypot(P[0], P[1]) >= Cfg.WORKSPACE_MIN_RADIUS)


def run_trial(Trial_Id, Hw, Kin, T_Cam_Calib, Observer, Lqr, Args, Gt_Object_Top=None):
    """Runs one pick. Gt_Object_Top = GROUND TRUTH object top-centre (mock only, scoring only)."""
    Scale = Args.speed_scale
    Max_Qd = Cfg.MAX_JOINT_SPEED * Scale
    Detector = Colour_Depth_Detector(Hw.Camera.Intrinsics)
    Kf = Target_KF(Cfg.TARGET_PROCESS_STD, Cfg.NIS_GATE, Cfg.MAX_CONSEC_REJECTS)
    Pbvs = PBVS_Controller(Cfg.PBVS_GAIN_POS, Cfg.PBVS_GAIN_ROT,
                           Cfg.MAX_LIN_SPEED * Scale, Cfg.MAX_ROT_SPEED * Scale)
    Logger = Trial_Logger(os.path.join(Args.log_dir, f"trial_{Trial_Id:03d}.csv"))
    Video = None
    if Args.record:
        Video = cv2.VideoWriter(os.path.join(Args.log_dir, f"trial_{Trial_Id:03d}.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), Cfg.CAMERA_FPS,
                                (Cfg.CAMERA_WIDTH, Cfg.CAMERA_HEIGHT))

    # ---------- go home, reset everything ----------
    Hw.Gripper.open()
    Hw.Robot.move_j(Cfg.HOME_Q, Cfg.HOME_SPEED * Scale, Cfg.HOME_ACC)
    Q0 = Hw.Robot.get_q()
    Observer.reset(Q0)
    Q_Ref, U_Prev = Q0.copy(), np.zeros(6)

    State, Phase_Start, Settle = "SEARCH", 0.0, 0
    Goal_R = Locked_Target = Locked_Std = Grasp_Tool_Pos = None
    Held, Reason, Disturbed = False, "", False
    Overruns = 0
    T_Start = time.perf_counter()
    t = 0.0

    def next_state(New):
        nonlocal State, Phase_Start, Settle
        print(f"[trial {Trial_Id}] t={t:6.2f}s  {State} -> {New}")
        State, Phase_Start, Settle = New, t, 0

    def pause_for_gripper(Action):
        """Stop the arm, run a (blocking) gripper action, then resync references."""
        nonlocal Q_Ref, U_Prev
        Hw.Robot.stop(Cfg.STOP_DECEL)
        Action()
        Q = Hw.Robot.get_q()
        Observer.reset(Q)
        Q_Ref, U_Prev = Q.copy(), np.zeros(6)

    try:
        while State != "DONE":
            Loop_Start = time.perf_counter()

            # ===== safety checks =====
            if STOP_REQUESTED:
                raise Abort("user stop (Ctrl-C)")
            if not Hw.Robot.is_safe():
                raise Abort("robot protective/emergency stop")

            # ===== 1. read sensors =====
            Q_Meas = Hw.Robot.get_q()
            Qd_Meas = Hw.Robot.get_qd()           # logged for checking the observer ONLY
            X_Hat = Observer.step(Q_Meas, U_Prev)
            T_Base_Tool = Kin.fk_tool(Q_Meas)
            if T_Base_Tool[2, 3] < Cfg.TOOL_Z_MIN:
                raise Abort(f"tool below safe height (z={T_Base_Tool[2, 3]:.3f})")

            # ===== 2. vision + target Kalman filter =====
            # The filter runs while its estimate is in use (SEARCH, APPROACH). After that
            # the target is locked and the filter is frozen; when the object is hidden
            # during APPROACH, predict() keeps running so the uncertainty grows honestly.
            Vision_On = State in ("SEARCH", "APPROACH")
            if Vision_On:
                Kf.predict()
            Meas_Base, Det = None, None
            Frame = Hw.Camera.get_frame()
            if Frame is not None:
                Colour, Depth = Frame
                # Frame chain (Part B: frame management)
                if Cfg.CAMERA_MOUNT == "fixed":
                    T_Base_Cam = T_Cam_Calib                              # camera -> base, constant
                    Det = Detector.detect(Colour, Depth, T_Base_Cam)     # top-face method
                else:
                    T_Base_Cam = Kin.fk_flange(Q_Meas) @ T_Cam_Calib      # camera -> flange -> base
                    Det = Detector.detect(Colour, Depth)
                if Det is not None and Vision_On and Det.P_Cam[2] > Cfg.VISION_MIN_DEPTH:
                    Meas_Base = transform_point(T_Base_Cam, Det.P_Cam)
                    R_Bc = T_Base_Cam[:3, :3]
                    Kf.update(Meas_Base, R_Bc @ Det.R_Cam @ R_Bc.T)   # covariance rotates too
                if Video is not None or Args.show:
                    Est = Kf.X if Kf.initialised else np.full(3, np.nan)
                    Text = f"{State}  est={np.round(Est, 3)}  2sd={np.round(2 * Kf.std() * 1000, 1)}mm"
                    Img = draw_detection(Colour, Det, Text)
                    if Gt_Object_Top is not None:   # GROUND TRUTH: display only
                        cv2.putText(Img, f"truth={np.round(Gt_Object_Top, 3)}", (10, 46),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if Video is not None:
                        Video.write(Img)
                    if Args.show:
                        cv2.imshow("UR3 pick", Img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise Abort("user stop (q)")

            # ===== 3. state machine: decide the goal =====
            Elapsed = t - Phase_Start
            Goal_Pos, Speed_Factor = None, 1.0
            Up = lambda h: np.array([0.0, 0.0, h])

            if State == "SEARCH":
                if (Kf.initialised and Kf.N_Updates >= Cfg.SEARCH_MIN_UPDATES
                        and np.all(Kf.std() < Cfg.SEARCH_MAX_STD)):
                    if not in_workspace(Kf.X):
                        raise Abort(f"target {np.round(Kf.X, 3)} outside workspace")
                    Goal_R = down_orientation_like(T_Base_Tool[:3, :3])
                    next_state("APPROACH")
                elif Elapsed > Cfg.SEARCH_TIMEOUT:
                    raise Abort("no reliable detection during search")
            if State == "APPROACH":
                Goal_Pos = Kf.X + Up(Cfg.APPROACH_HEIGHT)
                if Args.occlude and Hw.Mock_World:
                    Hw.Camera.Object_Visible = Elapsed < 0.5   # hide object after 0.5 s
                if Args.disturb and Hw.Is_Mock and not Disturbed and Elapsed > 1.0:
                    Hw.Robot.Q[0] += 0.05                      # 3 deg bump on the base joint
                    Disturbed = True
                    print(f"[trial {Trial_Id}] disturbance applied")
            elif State == "DESCEND":
                Goal_Pos = Locked_Target + Up(Cfg.GRASP_Z_OFFSET)
                Speed_Factor = Cfg.DESCEND_SPEED_FACTOR
            elif State == "LIFT":
                Goal_Pos = Grasp_Tool_Pos + Up(Cfg.LIFT_HEIGHT)
            elif State == "LOWER":
                Goal_Pos = Grasp_Tool_Pos + Up(0.002)
                Speed_Factor = Cfg.DESCEND_SPEED_FACTOR
            elif State == "RETREAT":
                Goal_Pos = Grasp_Tool_Pos + Up(Cfg.APPROACH_HEIGHT)
            elif State == "GRASP":
                Grasp_Tool_Pos = T_Base_Tool[:3, 3].copy()
                pause_for_gripper(Hw.Gripper.close)
                Held = Hw.Gripper.object_detected()
                print(f"[trial {Trial_Id}] gripper reports object held: {Held}")
                next_state("LIFT")
                continue
            elif State == "RELEASE":
                pause_for_gripper(Hw.Gripper.open)
                next_state("RETREAT")
                continue

            # ===== 4. control: PBVS outer loop -> LQR inner loop =====
            Pos_Err = Rot_Err = np.nan
            J = Kin.jacobian_tool(Q_Meas)
            if Goal_Pos is not None:
                V, Pos_Err, Rot_Err = Pbvs.compute(T_Base_Tool, Goal_Pos, Goal_R, Speed_Factor)
                Qd_Ref = resolved_rate(J, V, Cfg.DLS_DAMPING, Max_Qd)
                Settle = Settle + 1 if (Pos_Err < Cfg.POS_TOL and Rot_Err < Cfg.ROT_TOL) else 0
            else:
                Qd_Ref = np.zeros(6)

            Q_Ref = Q_Ref + Qd_Ref * Cfg.DT
            Q_Ref = np.clip(Q_Ref, X_Hat[:, 0] - Cfg.REF_LEASH, X_Hat[:, 0] + Cfg.REF_LEASH)
            U = np.clip(Lqr.compute(X_Hat, Q_Ref, Qd_Ref), -Max_Qd, Max_Qd)
            Hw.Robot.speed_j(U, Cfg.SPEEDJ_ACC, Cfg.DT)
            U_Prev = U

            Logger.log(t=round(t, 4), state=State, q=Q_Meas, qd_meas=Qd_Meas, qd_hat=X_Hat[:, 1],
                       q_ref=Q_Ref, u=U, tool=T_Base_Tool[:3, 3], goal=Goal_Pos,
                       est=Kf.X, est_std=Kf.std() if Kf.initialised else None, meas=Meas_Base,
                       nis=Kf.Last_Nis, pos_err=Pos_Err, rot_err=Rot_Err, manip=manipulability(J),
                       gt=Gt_Object_Top)   # GROUND TRUTH logged for evaluation only

            # ===== 5. transitions when a goal is reached =====
            if Settle >= Cfg.SETTLE_CYCLES:
                if State == "APPROACH":
                    Locked_Target, Locked_Std = Kf.X.copy(), Kf.std()
                    if Hw.Mock_World:
                        Hw.Camera.Object_Visible = True
                    next_state("DESCEND")
                elif State == "DESCEND":
                    next_state("GRASP")
                elif State == "LIFT":
                    next_state("LOWER")
                elif State == "LOWER":
                    next_state("RELEASE")
                elif State == "RETREAT":
                    next_state("DONE")
            if State not in ("SEARCH",) and t - Phase_Start > Cfg.PHASE_TIMEOUT:
                raise Abort(f"timeout in {State}")

            # ===== 6. timing =====
            if Hw.Is_Mock:
                t += Cfg.DT                         # simulated time, run as fast as possible
            else:
                # One slow loop is harmless (speedJ holds the last command for a moment,
                # and the robot stops itself if commands stop). A long freeze or many slow
                # loops in a row is not: then abort.
                Spent = time.perf_counter() - Loop_Start
                Overruns = Overruns + 1 if Spent > Cfg.DT + Cfg.MAX_LOOP_OVERRUN else 0
                if Spent > Cfg.MAX_LOOP_FREEZE or Overruns >= Cfg.MAX_CONSEC_OVERRUNS:
                    raise Abort(f"control loop too slow ({Spent * 1000:.0f} ms, {Overruns} slow loops in a row)")
                time.sleep(max(0.0, Cfg.DT - Spent))
                t = time.perf_counter() - T_Start
        Success = Held
        Reason = "completed" if Held else "completed but gripper empty"
    except Abort as E:
        Hw.Robot.stop(Cfg.STOP_DECEL)
        Success, Reason = False, f"ABORT: {E}"
        print(f"[trial {Trial_Id}] {Reason}")
    finally:
        Hw.Robot.stop(Cfg.STOP_DECEL)
        Logger.close()
        if Video is not None:
            Video.release()

    # ---------- score the trial ----------
    Result = dict(trial=Trial_Id, success=int(Success), reason=Reason, duration=round(t, 2),
                  kf_updates=Kf.N_Updates, kf_rejects=Kf.N_Rejects,
                  occlude=int(Args.occlude), disturb=int(Args.disturb), noise_scale=Args.noise_scale)
    for i, Ax in enumerate("xyz"):
        Result[f"locked_std_{Ax}_mm"] = round(Locked_Std[i] * 1000, 2) if Locked_Std is not None else ""
    if Gt_Object_Top is not None and Locked_Target is not None:      # GROUND TRUTH scoring
        Est_Err = Locked_Target - Gt_Object_Top
        for i, Ax in enumerate("xyz"):
            Result[f"est_err_{Ax}_mm"] = round(Est_Err[i] * 1000, 2)
        if Grasp_Tool_Pos is not None:
            Grasp_Err = Grasp_Tool_Pos - (Gt_Object_Top + np.array([0, 0, Cfg.GRASP_Z_OFFSET]))
            for i, Ax in enumerate("xyz"):
                Result[f"grasp_err_{Ax}_mm"] = round(Grasp_Err[i] * 1000, 2)
    return Result


def main():
    P = argparse.ArgumentParser(description="UR3 vision-guided pick (Track B hardware)")
    P.add_argument("--trials", type=int, default=3)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--speed-scale", type=float, default=Cfg.SPEED_SCALE)
    P.add_argument("--log-dir", default="logs")
    P.add_argument("--occlude", action="store_true", help="mock: hide object during approach")
    P.add_argument("--disturb", action="store_true", help="mock: bump the base joint during approach")
    P.add_argument("--noise-scale", type=float, default=1.0, help="mock: true depth noise x this")
    P.add_argument("--record", action="store_true", help="save annotated camera video per trial")
    P.add_argument("--show", action="store_true", help="live camera window (press q to stop)")
    Args = P.parse_args()
    if Args.speed_scale > 1.0:
        raise SystemExit("speed-scale above 1.0 is not allowed")
    os.makedirs(Args.log_dir, exist_ok=True)
    signal.signal(signal.SIGINT, _on_ctrl_c)

    Kin = UR3_Kinematics(Cfg.ROBOT_MODEL, Cfg.T_FLANGE_TOOL)
    T_Cam_Calib = load_camera_pose()
    Hw = Hardware(Kin, T_Cam_Calib, Seed=Args.seed)
    Observer = Joint_Observer(Cfg.JOINT_LAG_TAU, Cfg.DT, Cfg.OBSERVER_POLES, Cfg.HOME_Q)
    Lqr = Joint_LQR(Cfg.JOINT_LAG_TAU, Cfg.DT, Cfg.LQR_Q, Cfg.LQR_R)
    print(f"[main] LQR gain K = {np.round(Lqr.K, 3)}, closed-loop eigs = {np.round(Lqr.Closed_Loop_Eigs, 3)}")
    print(f"[main] observer gain L = {np.round(Observer.L.ravel(), 3)}")
    Rng = np.random.default_rng(Args.seed)
    Summary_Path = os.path.join(Args.log_dir, "summary.csv")

    try:
        for Trial_Id in range(Args.trials):
            Gt = None
            if Hw.Mock_World:
                # Randomised object position (Track B requirement). GROUND TRUTH for scoring.
                Xy = Rng.uniform(Cfg.MOCK_OBJECT_REGION_MIN, Cfg.MOCK_OBJECT_REGION_MAX)
                Gt = np.array([Xy[0], Xy[1], Cfg.TABLE_Z + Cfg.OBJECT_HEIGHT])
                Hw.Camera.Object_Top = Gt
                Hw.Camera.Object_Visible = True
                Hw.Camera.Noise_Scale = Args.noise_scale
                # Mock gripper "succeeds" if the fingertips are within 12 mm sideways and
                # 15 mm vertically of the true grasp point (roughly a 2F-85 on a 40 mm object)
                def Grasp_Check(G=Gt):
                    Err = Kin.fk_tool(Hw.Robot.get_q())[:3, 3] - (G + np.array([0, 0, Cfg.GRASP_Z_OFFSET]))
                    return np.linalg.norm(Err[:2]) < 0.012 and abs(Err[2]) < 0.015
                Hw.Gripper.Grasp_Check = Grasp_Check
            else:
                input(f"\nTrial {Trial_Id}: place the object somewhere new, step back, press Enter...")
            Result = run_trial(Trial_Id, Hw, Kin, T_Cam_Calib, Observer, Lqr, Args, Gt)
            if not Hw.Mock_World and Cfg.GRIPPER_TYPE == "mock":
                Result["success"] = int(input("Did the grasp succeed? [y/n] ").strip().lower() == "y")
            append_summary(Summary_Path, Result)
            print(f"[main] result: {Result}")
            if STOP_REQUESTED:
                break
    finally:
        Hw.close()
        cv2.destroyAllWindows()
    print(f"[main] summary written to {Summary_Path}")


if __name__ == "__main__":
    main()
