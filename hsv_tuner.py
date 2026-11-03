"""
hsv_tuner.py - sliders to find the HSV colour range of your object under the LAB lighting.
Do this at the start of every lab session (lighting changes the colour a lot).

    python hsv_tuner.py
Move the sliders until ONLY the object is white in the mask window, then copy the
printed range into HSV_RANGES in config.py. Press q to quit.
"""
import numpy as np
import cv2
import config as Cfg
from camera import RealSense_Camera

if __name__ == "__main__":
    Cam = RealSense_Camera(Cfg.CAMERA_WIDTH, Cfg.CAMERA_HEIGHT, Cfg.CAMERA_FPS)
    cv2.namedWindow("mask")
    for Name, Max, Init in (("H lo", 179, 0), ("H hi", 179, 10), ("S lo", 255, 120),
                            ("S hi", 255, 255), ("V lo", 255, 70), ("V hi", 255, 255)):
        cv2.createTrackbar(Name, "mask", Init, Max, lambda _v: None)
    try:
        while True:
            F = Cam.get_frame()
            if F is None:
                continue
            V = [cv2.getTrackbarPos(n, "mask") for n in ("H lo", "S lo", "V lo", "H hi", "S hi", "V hi")]
            Mask = cv2.inRange(cv2.cvtColor(F[0], cv2.COLOR_BGR2HSV), np.array(V[:3]), np.array(V[3:]))
            cv2.imshow("colour", F[0])
            cv2.imshow("mask", Mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"HSV_RANGES = [({tuple(V[:3])}, {tuple(V[3:])})]")
                break
    finally:
        Cam.close()
        cv2.destroyAllWindows()
