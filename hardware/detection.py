"""
detection.py - finds the target object in the colour image and turns it into a
3D point (in the CAMERA frame) with a covariance.     (Part B: sensing)

Steps:
  1. HSV colour threshold -> binary mask
  2. clean the mask (morphological open/close), keep the biggest blob
  3. blob centroid (u, v) in pixels, corrected for lens distortion
  4. depth Z = MEDIAN of the valid depth pixels inside the blob
       (median, not mean: robust to holes and edge pixels that hit the table)
  5. back-project with the pinhole model:  X = (u - cx) Z / fx,  Y = (v - cy) Z / fy
  6. covariance from the sensor noise model in config.py

FIXED (side-on) camera: the camera sees the TOP and the SIDES of the cube, so the
median depth of the blob is a point somewhere on the visible surface, not the top
centre (up to ~2 cm wrong). When T_Base_Cam is given, step 4-5 are replaced by:
  4b. back-project EVERY valid blob pixel to 3D and move it to the robot base frame
  4c. keep only the highest points (the top face, within TOP_FACE_BAND of the top)
  4d. top centre = median x, y of those points, z = top height
  4e. move that point back into the camera frame, so the rest of the pipeline is unchanged
"""
from dataclasses import dataclass
import numpy as np
import cv2
import config as Cfg


@dataclass
class Detection:
    P_Cam: np.ndarray      # 3D top-centre of the object in the camera optical frame [m]
    R_Cam: np.ndarray      # 3x3 measurement covariance in the camera frame [m^2]
    U: float               # pixel centroid (for drawing)
    V: float
    Area: int              # blob area in pixels
    Valid_Frac: float      # fraction of blob pixels with valid depth


class Colour_Depth_Detector:
    def __init__(self, Intrinsics):
        self.Intr = Intrinsics
        self.K = Intrinsics.K()
        self.Dist = Intrinsics.Dist
        self.Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.Last_Mask = None

    def make_mask(self, Colour_Bgr):
        Hsv = cv2.cvtColor(Colour_Bgr, cv2.COLOR_BGR2HSV)
        Mask = np.zeros(Hsv.shape[:2], np.uint8)
        for Lo, Hi in Cfg.HSV_RANGES:
            Mask |= cv2.inRange(Hsv, np.array(Lo), np.array(Hi))
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, self.Kernel)    # remove speckles
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_CLOSE, self.Kernel)   # fill small holes
        return Mask

    def detect(self, Colour_Bgr, Depth_M, T_Base_Cam=None):
        """Returns a Detection, or None if the object isn't reliably visible.
        T_Base_Cam: give it for the FIXED camera to use the top-face method."""
        Mask = self.make_mask(Colour_Bgr)
        self.Last_Mask = Mask
        Contours, _ = cv2.findContours(Mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not Contours:
            return None
        Blob = max(Contours, key=cv2.contourArea)
        Area = cv2.contourArea(Blob)
        if Area < Cfg.MIN_BLOB_AREA:
            return None

        M = cv2.moments(Blob)
        U, V = M["m10"] / M["m00"], M["m01"] / M["m00"]

        # Depth: only the pixels inside the blob, shrunk a little to avoid edges
        Blob_Mask = np.zeros_like(Mask)
        cv2.drawContours(Blob_Mask, [Blob], -1, 255, thickness=-1)
        Blob_Mask = cv2.erode(Blob_Mask, self.Kernel)
        Depths = Depth_M[Blob_Mask > 0]
        if Depths.size == 0:
            return None
        Valid = Depths[(Depths > Cfg.DEPTH_MIN) & (Depths < Cfg.DEPTH_MAX)]
        Valid_Frac = Valid.size / Depths.size
        if Valid_Frac < Cfg.MIN_VALID_DEPTH_FRAC:
            return None
        if T_Base_Cam is not None:
            P_Cam = self._top_centre_cam(Blob_Mask, Depth_M, T_Base_Cam)
            if P_Cam is None:
                return None
            X, Y, Z = P_Cam
        else:
            Z = float(np.median(Valid))
            # Undistort the centroid (RealSense colour distortion is tiny, but be correct)
            Pt = cv2.undistortPoints(np.array([[[U, V]]], np.float64), self.K, self.Dist, P=self.K)
            Uu, Vu = Pt[0, 0]
            X = (Uu - self.Intr.Cx) * Z / self.Intr.Fx
            Y = (Vu - self.Intr.Cy) * Z / self.Intr.Fy
            P_Cam = np.array([X, Y, Z])

        # Noise model -> covariance (camera frame: x right, y down, z forward)
        Sigma_Z = Cfg.DEPTH_NOISE_K * Z ** 2 + Cfg.DEPTH_NOISE_MIN
        Sigma_X = Cfg.PIXEL_NOISE * Z / self.Intr.Fx
        Sigma_Y = Cfg.PIXEL_NOISE * Z / self.Intr.Fy
        # x and y also inherit some depth error (they are scaled by Z)
        Sigma_X = np.hypot(Sigma_X, abs(X) / Z * Sigma_Z)
        Sigma_Y = np.hypot(Sigma_Y, abs(Y) / Z * Sigma_Z)
        R_Cam = np.diag([Sigma_X ** 2, Sigma_Y ** 2, Sigma_Z ** 2])

        return Detection(P_Cam, R_Cam, U, V, int(Area), Valid_Frac)

    def _top_centre_cam(self, Blob_Mask, Depth_M, T_Base_Cam):
        """Fixed camera: top-centre of the object from its 3D points (see module docstring)."""
        V_Pix, U_Pix = np.nonzero(Blob_Mask)
        Z = Depth_M[V_Pix, U_Pix].astype(float)
        Ok = (Z > Cfg.DEPTH_MIN) & (Z < Cfg.DEPTH_MAX)
        if Ok.sum() < 20:
            return None
        Uv = np.stack([U_Pix[Ok], V_Pix[Ok]], 1).astype(np.float64).reshape(-1, 1, 2)
        Und = cv2.undistortPoints(Uv, self.K, self.Dist, P=self.K).reshape(-1, 2)
        Z = Z[Ok]
        P_Cam = np.stack([(Und[:, 0] - self.Intr.Cx) * Z / self.Intr.Fx,
                          (Und[:, 1] - self.Intr.Cy) * Z / self.Intr.Fy, Z], 1)
        R, T = T_Base_Cam[:3, :3], T_Base_Cam[:3, 3]
        P_Base = P_Cam @ R.T + T
        # Find the top-face height. Start at the 90th percentile (ignores noisy spikes),
        # then re-centre on the MEDIAN of the points near it a few times. Using the
        # highest points directly would be biased upwards by depth noise.
        Z_Top = np.percentile(P_Base[:, 2], 90)
        for _ in range(5):
            Near = np.abs(P_Base[:, 2] - Z_Top) < 2 * Cfg.TOP_FACE_BAND
            if Near.sum() < 10:
                return None
            Z_Top = np.median(P_Base[Near, 2])
        Top = P_Base[np.abs(P_Base[:, 2] - Z_Top) < 2 * Cfg.TOP_FACE_BAND]
        if len(Top) < 10:
            return None
        Centre_Base = np.array([np.median(Top[:, 0]), np.median(Top[:, 1]), Z_Top])
        return R.T @ (Centre_Base - T)                    # back to the camera frame


def draw_detection(Colour_Bgr, Det, Text=""):
    """Overlay for the demo video: crosshair on the detected centroid."""
    Img = Colour_Bgr.copy()
    if Det is not None:
        C = (int(Det.U), int(Det.V))
        cv2.drawMarker(Img, C, (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
        cv2.putText(Img, f"Z={Det.P_Cam[2]:.3f} m", (C[0] + 12, C[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if Text:
        cv2.putText(Img, Text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return Img
