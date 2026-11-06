"""
measure_depth_noise.py - measures the camera's depth noise so DEPTH_NOISE_K / R are
backed by data, not guessed.                                     (Part B / Part C)

Put the target object on the table. The script moves the arm straight down in steps
(different camera heights), and at each height records N detections while still.
For each height it reports the std of the measured depth. Then it fits
    sigma_z = k * Z^2 + sigma_min
and prints k and sigma_min to copy into config.py. Plot saved to calib/depth_noise.png.

    python measure_depth_noise.py              # uses the switches in config.py (works on mock too)
"""
import numpy as np
import config as Cfg
from ur3_kinematics import UR3_Kinematics
from hardware import Hardware, load_T_Flange_Cam
from detection import Colour_Depth_Detector
from controller import resolved_rate

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Kin = UR3_Kinematics(Cfg.ROBOT_MODEL, Cfg.T_FLANGE_TOOL)
    Hw = Hardware(Kin, load_T_Flange_Cam())
    if Hw.Is_Mock:
        Hw.Camera.Object_Top = np.array([-0.24, -0.12, Cfg.TABLE_Z + Cfg.OBJECT_HEIGHT])
    Det = Colour_Depth_Detector(Hw.Camera.Intrinsics)
    Heights, Means, Stds = [], [], []
    try:
        Hw.Robot.move_j(Cfg.HOME_Q, Cfg.HOME_SPEED * 0.5, Cfg.HOME_ACC)
        for Step in range(5):
            Zs = []
            while len(Zs) < 100:
                F = Hw.Camera.get_frame()
                if F is None:
                    Hw.Robot.speed_j(np.zeros(6), Cfg.SPEEDJ_ACC, Cfg.DT)
                    continue
                D = Det.detect(*F)
                if D is not None:
                    Zs.append(D.P_Cam[2])
            Means.append(np.mean(Zs))
            Stds.append(np.std(Zs))
            print(f"Z = {Means[-1]:.3f} m   sigma_z = {Stds[-1] * 1000:.2f} mm")
            # move DOWN 3.5 cm slowly (moving up hits the UR3 reach limit) with straight-line velocity control
            T0 = Kin.fk_tool(Hw.Robot.get_q())[:3, 3]
            for _ in range(int(2.0 / Cfg.DT)):
                Q = Hw.Robot.get_q()
                E = T0 + [0, 0, -0.035] - Kin.fk_tool(Q)[:3, 3]
                V = np.hstack([np.clip(2 * E, -0.03, 0.03), np.zeros(3)])
                Hw.Robot.speed_j(resolved_rate(Kin.jacobian_tool(Q), V, 0.02, 0.3), Cfg.SPEEDJ_ACC, Cfg.DT)
            Hw.Robot.stop()
        Z, S = np.array(Means), np.array(Stds)
        # Least squares fit of  S = k Z^2 + s0
        Coef, *_ = np.linalg.lstsq(np.column_stack([Z ** 2, np.ones_like(Z)]), S, rcond=None)
        print(f"\nfit: DEPTH_NOISE_K = {Coef[0]:.4f},  DEPTH_NOISE_MIN = {Coef[1]:.4f}  -> copy into config.py")
        print("(the mock's noise is what config says, so on mock this just checks the method)")
        Zf = np.linspace(Z.min() * 0.9, Z.max() * 1.1, 50)
        plt.plot(Z, S * 1000, "o", label="measured")
        plt.plot(Zf, (Coef[0] * Zf ** 2 + Coef[1]) * 1000, label="fit  k Z^2 + s0")
        plt.xlabel("depth Z [m]"); plt.ylabel("std of object depth [mm]"); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig("calib/depth_noise.png", dpi=150)
    finally:
        Hw.close()
