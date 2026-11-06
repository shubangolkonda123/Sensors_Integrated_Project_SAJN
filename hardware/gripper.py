"""
gripper.py - gripper interfaces. Every gripper class has: open(), close(), object_detected().

Robotiq 2F-85/2F-140 with the Robotiq URCap installed on the UR listens on TCP
port 63352 for simple text commands ("SET POS 255" etc). That is what we use here.
"""
import socket
import time


class Robotiq_Gripper:
    PORT = 63352

    def __init__(self, Robot_Ip, Speed=150, Force=60):
        self.Sock = socket.create_connection((Robot_Ip, self.PORT), timeout=2.0)
        self._set("ACT", 1)                       # activate
        time.sleep(1.5)
        self._set("SPE", Speed)                   # 0-255
        self._set("FOR", Force)                   # 0-255, keep low for fragile objects
        self._set("GTO", 1)                       # "go to" enabled
        print("[gripper] Robotiq activated")

    def _send(self, Cmd):
        self.Sock.sendall((Cmd + "\n").encode())
        return self.Sock.recv(1024).decode().strip()

    def _set(self, Var, Value):
        return self._send(f"SET {Var} {int(Value)}")

    def _get(self, Var):
        Reply = self._send(f"GET {Var}")         # e.g. "OBJ 2"
        return int(Reply.split()[-1])

    def _move(self, Pos, Timeout=3.0):
        self._set("POS", Pos)
        T0 = time.time()
        time.sleep(0.2)
        while self._get("OBJ") == 0 and time.time() - T0 < Timeout:   # 0 = still moving
            time.sleep(0.05)

    def open(self):
        self._move(0)

    def close(self):
        self._move(255)

    def object_detected(self):
        """OBJ = 1 or 2 means the fingers stopped on something (an object is held)."""
        return self._get("OBJ") in (1, 2)

    def shutdown(self):
        self.Sock.close()


class OnRobot_Gripper:
    """
    OnRobot RG2 on the lab UR3e, through the onRobot library (as in the lab repo's
    example 15). The library sends XML-RPC requests to the OnRobot URCap on the robot
    (port 41414). An RTDE control session must already be open, so create the robot
    (RTDE_Robot) BEFORE this gripper - hardware.py already does that.

    Install:  pip install "git+https://github.com/RianPolMacCionaoith/onRobot.git@main#subdirectory=onRobot"
    Pendant:  Installation -> Tool/IO -> OnRobot must be selected.
    """

    def __init__(self, Robot_Ip, Gripper_Id=0):
        import config as Cfg
        import onRobot.gripper as Rg_Lib
        self.Cfg = Cfg
        self.Rg = Rg_Lib.RG(Robot_Ip, Gripper_Id)
        Width = self.Rg.get_rg_width()
        print(f"[gripper] RG2 connected, current width {Width:.1f} mm")

    def _move(self, Width_Mm, Timeout=4.0):
        """Command a width, then wait until the fingers stop moving."""
        self.Rg.rg_grip(float(Width_Mm), float(self.Cfg.RG2_FORCE_N))
        T0 = time.time()
        Last = self.Rg.get_rg_width()
        while time.time() - T0 < Timeout:
            time.sleep(0.25)
            Now = self.Rg.get_rg_width()
            if abs(Now - Last) < 0.3:        # stopped (reached target or hit the object)
                break
            Last = Now
        return self.Rg.get_rg_width()

    def open(self):
        self._move(self.Cfg.RG2_OPEN_WIDTH_MM)

    def close(self):
        self.Last_Width = self._move(self.Cfg.RG2_CLOSED_WIDTH_MM)
        print(f"[gripper] closed, width {self.Last_Width:.1f} mm")

    def object_detected(self):
        """Closing towards 10 mm but stopping wider than ~20 mm means the cube is between the fingers."""
        return self.Rg.get_rg_width() > self.Cfg.RG2_HELD_MIN_WIDTH_MM

    def shutdown(self):
        pass
