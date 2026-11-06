"""
mock_hardware.py - fake robot, camera and gripper so the WHOLE pipeline can be run
and debugged on a laptop before going near the real arm.

They have exactly the same methods as the real classes, so main_pick.py can't tell
the difference. The mock world knows the true object position: that is GROUND TRUTH
and it is only ever used (a) to draw the fake camera image and (b) to score the
result afterwards. The controller never sees it.
"""
import numpy as np
import cv2
import config as Cfg
from camera import Camera_Intrinsics
from transforms import transform_point, inv_T


class Mock_Robot:
    """Joint velocity follows the speedJ command like the real UR:
    acceleration-limited ramp, then a first-order lag with time constant Tau."""

    def __init__(self, Q0, Tau, Encoder_Noise=1e-5, Seed=0):
        self.Q = np.array(Q0, dtype=float)
        self.Qd = np.zeros(6)
        self.Cmd = np.zeros(6)
        self.Tau = Tau
        self.Encoder_Noise = Encoder_Noise
        self.Rng = np.random.default_rng(Seed)
        self.Sim_Time = 0.0

    def get_q(self):
        return self.Q + self.Rng.normal(0, self.Encoder_Noise, 6)

    def get_qd(self):
        return self.Qd.copy()

    def is_safe(self):
        return True

    def speed_j(self, Qd_Cmd, Acc, Dt, Substeps=5):
        h = Dt / Substeps
        for _ in range(Substeps):
            # 1) speedJ ramps its internal command at most Acc rad/s^2
            Step = np.clip(np.asarray(Qd_Cmd) - self.Cmd, -Acc * h, Acc * h)
            self.Cmd += Step
            # 2) the joint velocity loop lags behind the command
            self.Qd += (self.Cmd - self.Qd) * h / self.Tau
            self.Q += self.Qd * h
        self.Sim_Time += Dt

    def move_j(self, Q, Speed, Acc):
        self.Q = np.array(Q, dtype=float)   # teleport: fine for a mock
        self.Qd[:] = 0.0
        self.Cmd[:] = 0.0

    def stop(self, Decel=2.0):
        self.Cmd[:] = 0.0
        self.Qd[:] = 0.0

    def teach_mode(self, On):
        pass

    def close(self):
        pass


class Mock_Camera:
    """Renders a grey table with a red disk (the object's top face) as seen from the
    camera, plus a noisy depth image. The camera is either on the wrist
    (T_Flange_Cam_True given) or fixed on a stand (T_Base_Cam_True given).
    Limitation: the fake object is only its top face and the arm is not drawn, so
    side faces and the arm blocking the view are NOT simulated."""

    def __init__(self, Robot, Kin, T_Flange_Cam_True=None, Seed=0, T_Base_Cam_True=None):
        I = Cfg.MOCK_INTRINSICS
        self.Intrinsics = Camera_Intrinsics(I["Fx"], I["Fy"], I["Cx"], I["Cy"],
                                            Cfg.CAMERA_WIDTH, Cfg.CAMERA_HEIGHT)
        self.Robot = Robot
        self.Kin = Kin
        self.T_Flange_Cam_True = T_Flange_Cam_True
        self.T_Base_Cam_True = T_Base_Cam_True
        self.Rng = np.random.default_rng(Seed)
        self.Every_N = max(1, round(Cfg.CONTROL_HZ / Cfg.CAMERA_FPS))
        self.Calls = 0
        self.Object_Top = None            # GROUND TRUTH, set by main_pick for each trial
        self.Object_Visible = True        # can be switched off to test occlusion
        self.Min_Range = 0.12             # real stereo cameras can't see closer than this
        self.Noise_Scale = 1.0            # >1 = true noise worse than the filter's model (robustness test)
        H, W = Cfg.CAMERA_HEIGHT, Cfg.CAMERA_WIDTH
        U, V = np.meshgrid(np.arange(W), np.arange(H))
        # Unit-depth ray for every pixel, in the camera frame
        self.Rays = np.stack([(U - I["Cx"]) / I["Fx"], (V - I["Cy"]) / I["Fy"], np.ones_like(U, float)], -1)
        # Pre-generated noise images (generating fresh ones every frame is slow)
        self.Depth_Noise_Bank = self.Rng.standard_normal((8, H, W)).astype(np.float32)
        self.Colour_Noise_Bank = self.Rng.normal(0, 4, (8, H, W, 3)).astype(np.int16)

    def _plane_depth(self, T_Base_Cam, Plane_Z):
        """Depth image of a horizontal plane z = Plane_Z (ray-plane intersection)."""
        R, P = T_Base_Cam[:3, :3], T_Base_Cam[:3, 3]
        Dir_Z = self.Rays @ R[2, :]                 # z-component of each ray in the base frame
        with np.errstate(divide="ignore"):
            Depth = (Plane_Z - P[2]) / Dir_Z
        Depth[~np.isfinite(Depth) | (Depth < 0)] = 0.0
        return Depth

    def get_frame(self):
        self.Calls += 1
        if self.Calls % self.Every_N:
            return None
        H, W = Cfg.CAMERA_HEIGHT, Cfg.CAMERA_WIDTH
        if self.T_Base_Cam_True is not None:
            T_Base_Cam = self.T_Base_Cam_True                       # fixed stand
        else:
            T_Base_Cam = self.Kin.fk_flange(self.Robot.get_q()) @ self.T_Flange_Cam_True
        T_Cam_Base = inv_T(T_Base_Cam)

        Colour = np.zeros((H, W, 3), np.uint8)
        Colour[:] = Cfg.MOCK_TABLE_COLOUR
        Depth = self._plane_depth(T_Base_Cam, Cfg.TABLE_Z)

        if self.Object_Top is not None and self.Object_Visible:
            # Project the circular top face into the image as a polygon
            Ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            Rim = self.Object_Top + Cfg.MOCK_OBJECT_RADIUS * np.stack([np.cos(Ang), np.sin(Ang), 0 * Ang], 1)
            Rim_Cam = np.array([transform_point(T_Cam_Base, p) for p in Rim])
            if np.all(Rim_Cam[:, 2] > 0.01):
                I = self.Intrinsics
                Px = np.stack([I.Fx * Rim_Cam[:, 0] / Rim_Cam[:, 2] + I.Cx,
                               I.Fy * Rim_Cam[:, 1] / Rim_Cam[:, 2] + I.Cy], 1)
                Mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(Mask, [np.round(Px).astype(np.int32)], 1)
                Colour[Mask > 0] = Cfg.MOCK_OBJECT_COLOUR
                Top_Depth = self._plane_depth(T_Base_Cam, self.Object_Top[2])
                Depth[Mask > 0] = Top_Depth[Mask > 0]

        # Depth noise: per-pixel noise + a per-frame shared error (real stereo error
        # is spatially correlated, so averaging many pixels does NOT remove it all)
        Sigma = (Cfg.DEPTH_NOISE_K * Depth ** 2 + Cfg.DEPTH_NOISE_MIN) * self.Noise_Scale
        k = self.Rng.integers(8)
        Depth = Depth + self.Depth_Noise_Bank[k] * Sigma * 2.0 + self.Rng.normal(0, 1) * Sigma
        Depth[Depth < self.Min_Range] = 0.0
        Colour = np.clip(Colour.astype(np.int16) + self.Colour_Noise_Bank[self.Rng.integers(8)],
                         0, 255).astype(np.uint8)
        return Colour, Depth.astype(np.float32)

    def close(self):
        pass


class Mock_Gripper:
    def __init__(self):
        self.Closed = False
        self.Holding = False
        self.Grasp_Check = None   # main_pick can set a function that decides success

    def open(self):
        self.Closed, self.Holding = False, False

    def close(self):
        self.Closed = True
        self.Holding = bool(self.Grasp_Check()) if self.Grasp_Check else True

    def object_detected(self):
        return self.Holding

    def shutdown(self):
        pass
