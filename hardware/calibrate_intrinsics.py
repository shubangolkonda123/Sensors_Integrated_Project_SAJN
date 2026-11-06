"""
calibrate_intrinsics.py - checks the camera's intrinsic calibration.        (Part B)

RealSense cameras ship with accurate factory intrinsics. The brief asks for camera
calibration, so this script does a proper checkerboard calibration and COMPARES it
with the factory values. Report both, and the reprojection error, on the project page.

Hold the checkerboard (or move the camera) through ~20 varied views: close/far,
tilted, in all corners of the image.  SPACE = capture,  c = calibrate,  q = quit

    python calibrate_intrinsics.py --cols 9 --rows 6 --square 0.025
"""
import argparse
import os
import numpy as np
import cv2
import config as Cfg
from camera import RealSense_Camera

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--cols", type=int, default=9)
    P.add_argument("--rows", type=int, default=6)
    P.add_argument("--square", type=float, default=0.025)
    A = P.parse_args()

    Cam = RealSense_Camera(Cfg.CAMERA_WIDTH, Cfg.CAMERA_HEIGHT, Cfg.CAMERA_FPS)
    Obj = np.zeros((A.cols * A.rows, 3), np.float32)
    Obj[:, :2] = np.mgrid[0:A.cols, 0:A.rows].T.reshape(-1, 2) * A.square
    Obj_Pts, Img_Pts = [], []
    try:
        while True:
            F = Cam.get_frame()
            if F is None:
                continue
            Gray = cv2.cvtColor(F[0], cv2.COLOR_BGR2GRAY)
            Found, Corners = cv2.findChessboardCorners(Gray, (A.cols, A.rows))
            Show = F[0].copy()
            if Found:
                Corners = cv2.cornerSubPix(Gray, Corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
                cv2.drawChessboardCorners(Show, (A.cols, A.rows), Corners, Found)
            cv2.putText(Show, f"views: {len(Img_Pts)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("intrinsics", Show)
            Key = cv2.waitKey(1) & 0xFF
            if Key == ord(" ") and Found:
                Obj_Pts.append(Obj)
                Img_Pts.append(Corners)
            elif Key == ord("c") and len(Img_Pts) >= 10:
                Rms, K, Dist, _, _ = cv2.calibrateCamera(Obj_Pts, Img_Pts, Gray.shape[::-1], None, None)
                Kf = Cam.Intrinsics.K()
                print(f"\nreprojection RMS error: {Rms:.3f} px  (good: < 0.5 px)")
                print("           checkerboard   factory")
                for Name, Idx in (("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))):
                    print(f"  {Name}:     {K[Idx]:10.2f}  {Kf[Idx]:10.2f}   diff {K[Idx] - Kf[Idx]:+.2f}")
                print("  distortion:", np.round(Dist.ravel(), 4))
                os.makedirs("calib", exist_ok=True)
                np.savez(Cfg.INTRINSICS_FILE, K=K, Dist=Dist, Rms=Rms)
                print(f"saved to {Cfg.INTRINSICS_FILE} (set USE_CUSTOM_INTRINSICS=True to use it)")
                break
            elif Key == ord("q"):
                break
    finally:
        Cam.close()
        cv2.destroyAllWindows()
