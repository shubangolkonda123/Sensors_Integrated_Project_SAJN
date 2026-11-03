"""
hardware.py - builds the robot / camera / gripper objects from the switches in config.py,
and loads the camera calibration. This is the ONLY file that knows which hardware is real.
"""
import os
import numpy as np
import config as Cfg
from transforms import make_T


def load_camera_pose():
    """
    Returns the camera calibration for the mount chosen in config.py:
      CAMERA_MOUNT = "fixed" -> T_Base_Cam   (camera pose in the robot base frame, constant)
      CAMERA_MOUNT = "wrist" -> T_Flange_Cam (camera pose on the flange, moves with the arm)
    """
    if Cfg.CAMERA_MOUNT == "fixed":
        if os.path.exists(Cfg.EYE_TO_HAND_FILE):
            print(f"[calib] loaded fixed-camera calibration from {Cfg.EYE_TO_HAND_FILE}")
            return np.load(Cfg.EYE_TO_HAND_FILE)["T_Base_Cam"]
        print("[calib] WARNING: no fixed-camera calibration found, using T_BASE_CAM_DEFAULT")
        return Cfg.T_BASE_CAM_DEFAULT.copy()
    return load_T_Flange_Cam()


def load_T_Flange_Cam():
    """Hand-eye calibration result if it exists, otherwise the rough default."""
    if os.path.exists(Cfg.HANDEYE_FILE):
        T = np.load(Cfg.HANDEYE_FILE)["T_Flange_Cam"]
        print(f"[calib] loaded hand-eye calibration from {Cfg.HANDEYE_FILE}")
        return T
    print("[calib] WARNING: no hand-eye calibration found, using T_FLANGE_CAM_DEFAULT")
    return Cfg.T_FLANGE_CAM_DEFAULT.copy()


class Hardware:
    def __init__(self, Kin, T_Cam_Calib, Seed=0):
        """T_Cam_Calib = T_Base_Cam (fixed camera) or T_Flange_Cam (wrist camera)."""
        if Cfg.USE_MOCK_CAMERA and not Cfg.USE_MOCK_ROBOT:
            print("[hw] mock camera + real robot: URSim test mode (fake object drawn from the arm's pose)")
        if not Cfg.USE_MOCK_CAMERA and Cfg.USE_MOCK_ROBOT:
            print("[hw] real camera + mock robot: OK for testing detection only (arm pose is fake)")

        # ---- robot ----
        if Cfg.USE_MOCK_ROBOT:
            from mock_hardware import Mock_Robot
            self.Robot = Mock_Robot(Cfg.HOME_Q, Cfg.JOINT_LAG_TAU, Seed=Seed)
        else:
            from robot import RTDE_Robot
            self.Robot = RTDE_Robot(Cfg.ROBOT_IP)

        # ---- camera ----
        if Cfg.USE_MOCK_CAMERA:
            from mock_hardware import Mock_Camera
            # The mock's TRUE camera mount is slightly different from our calibration,
            # just like reality: this creates a realistic small bias to diagnose.
            T_True = T_Cam_Calib @ make_T(np.eye(3), [Cfg.MOCK_HANDEYE_ERROR, -Cfg.MOCK_HANDEYE_ERROR, 0])
            if Cfg.CAMERA_MOUNT == "fixed":
                self.Camera = Mock_Camera(self.Robot, Kin, Seed=Seed + 1, T_Base_Cam_True=T_True)
            else:
                self.Camera = Mock_Camera(self.Robot, Kin, T_True, Seed=Seed + 1)
        else:
            from camera import RealSense_Camera
            Custom = Cfg.INTRINSICS_FILE if Cfg.USE_CUSTOM_INTRINSICS else None
            self.Camera = RealSense_Camera(Cfg.CAMERA_WIDTH, Cfg.CAMERA_HEIGHT, Cfg.CAMERA_FPS, Custom)

        # ---- gripper ----
        if Cfg.GRIPPER_TYPE == "robotiq":
            from gripper import Robotiq_Gripper
            self.Gripper = Robotiq_Gripper(Cfg.ROBOT_IP)
        elif Cfg.GRIPPER_TYPE == "onrobot":
            from gripper import OnRobot_Gripper
            self.Gripper = OnRobot_Gripper(Cfg.ROBOT_IP)   # needs the RTDE session opened above
        else:
            from mock_hardware import Mock_Gripper
            self.Gripper = Mock_Gripper()

        self.Is_Mock = Cfg.USE_MOCK_ROBOT      # robot is simulated in Python (time runs as fast as possible)
        self.Mock_World = Cfg.USE_MOCK_CAMERA  # fake object/camera: ground truth exists for scoring

    def close(self):
        for Part in (self.Robot, self.Camera):
            try:
                Part.close()
            except Exception as E:
                print(f"[hw] error while closing: {E}")
