"""
robot.py - real UR3 interface through ur_rtde  (pip install ur_rtde).

ur_rtde talks to the robot over Ethernet using UR's Real-Time Data Exchange.
  - RTDEReceiveInterface : reads joint angles/speeds and safety status
  - RTDEControlInterface : sends motion commands (uploads a small script to the robot)

Robot setup on the pendant (ask at induction):
  - robot in Remote Control mode (e-Series) or Program not running (CB3)
  - your laptop on the same subnet as ROBOT_IP

If the lab uses ROS instead, write a class with the same methods
(get_q, get_qd, speed_j, move_j, stop, is_safe, teach_mode, close)
and switch to it in hardware.py. Nothing else changes.
"""
import numpy as np


class RTDE_Robot:
    def __init__(self, Ip):
        import rtde_control
        import rtde_receive
        print(f"[robot] connecting to UR at {Ip} ...")
        self.Rtde_R = rtde_receive.RTDEReceiveInterface(Ip)
        self.Rtde_C = rtde_control.RTDEControlInterface(Ip)
        print("[robot] connected")

    # ---------- reading ----------
    def get_q(self):
        """Measured joint angles [rad], from the encoders."""
        return np.array(self.Rtde_R.getActualQ())

    def get_qd(self):
        """Measured joint speeds [rad/s]. Our observer estimates these too; we
        log both so the project page can show the observer is correct."""
        return np.array(self.Rtde_R.getActualQd())

    def is_safe(self):
        """False if the robot is protective-stopped or e-stopped."""
        return not (self.Rtde_R.isProtectiveStopped() or self.Rtde_R.isEmergencyStopped())

    # ---------- commanding ----------
    def speed_j(self, Qd, Acc, Dt):
        """Joint velocity command. The robot keeps this speed until the next command
        (and stops itself if commands stop arriving)."""
        self.Rtde_C.speedJ(list(map(float, Qd)), float(Acc), float(Dt))

    def move_j(self, Q, Speed, Acc):
        """Blocking joint-space move (used only to go to the home pose)."""
        self.Rtde_C.moveJ(list(map(float, Q)), float(Speed), float(Acc))

    def stop(self, Decel=2.0):
        self.Rtde_C.speedStop(float(Decel))

    def teach_mode(self, On):
        """Freedrive: lets you push the arm by hand (used during hand-eye calibration)."""
        if On:
            self.Rtde_C.teachMode()
        else:
            self.Rtde_C.endTeachMode()

    def close(self):
        try:
            self.Rtde_C.speedStop(2.0)
            self.Rtde_C.stopScript()
        finally:
            self.Rtde_C.disconnect()
            self.Rtde_R.disconnect()
