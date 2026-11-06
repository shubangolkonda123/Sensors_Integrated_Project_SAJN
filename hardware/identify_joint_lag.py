"""
identify_joint_lag.py - measures tau, the time constant of the UR's velocity loop. (Part A)

The joint model in joint_model.py assumes   qdd = (u - qd) / tau.
This script checks that assumption and finds tau:
  - starts at HOME_Q, sends a small velocity STEP to joint 6 (the wrist, safest:
    it only spins the tool) and records the measured joint speed
  - fits the first-order step response  qd(t) = u (1 - exp(-t / tau))
  - repeats the step a few times and reports mean +- std of tau
Copy the result into JOINT_LAG_TAU in config.py. Plot saved to calib/joint_lag.png.

Note: speedJ also has an acceleration limit, so we use a high acceleration here
(the ramp would otherwise dominate the response).

    python identify_joint_lag.py           # uses config switches (mock or real)
"""
import time
import numpy as np
from scipy.optimize import curve_fit
import config as Cfg
from ur3_kinematics import UR3_Kinematics
from hardware import Hardware, load_T_Flange_Cam

JOINT = 5           # wrist 3
STEP = 0.3          # rad/s
DURATION = 0.6      # s
ACC = 20.0          # rad/s^2, high so the acceleration ramp doesn't hide the lag
HZ = 125            # sample fast for a good fit


def step_model(t, Tau, Delay):
    return STEP * (1 - np.exp(-np.clip(t - Delay, 0, None) / Tau))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Kin = UR3_Kinematics(Cfg.ROBOT_MODEL, Cfg.T_FLANGE_TOOL)
    Hw = Hardware(Kin, load_T_Flange_Cam())
    Dt = 1.0 / HZ
    Taus = []
    try:
        for Rep in range(4):
            Hw.Robot.move_j(Cfg.HOME_Q, Cfg.HOME_SPEED * 0.5, Cfg.HOME_ACC)
            Sign = 1 if Rep % 2 == 0 else -1
            U = np.zeros(6)
            U[JOINT] = Sign * STEP
            T, Qd = [], []
            T0 = time.perf_counter()
            for k in range(int(DURATION * HZ)):
                Start = time.perf_counter()
                Hw.Robot.speed_j(U, ACC, Dt)
                T.append(k * Dt)
                Qd.append(Sign * Hw.Robot.get_qd()[JOINT])
                if not Hw.Is_Mock:
                    time.sleep(max(0, Dt - (time.perf_counter() - Start)))
            Hw.Robot.stop(5.0)
            T, Qd = np.array(T), np.array(Qd)
            (Tau, Delay), _ = curve_fit(step_model, T, Qd, p0=[0.05, 0.005], bounds=([0.001, 0], [1, 0.1]))
            Taus.append(Tau)
            print(f"step {Rep}: tau = {Tau * 1000:.1f} ms, delay = {Delay * 1000:.1f} ms")
            plt.plot(T, Qd, ".", ms=3, label=f"measured {Rep}")
            plt.plot(T, step_model(T, Tau, Delay), "-", lw=1)
        print(f"\nJOINT_LAG_TAU = {np.mean(Taus):.4f}   (std {np.std(Taus) * 1000:.1f} ms) -> copy into config.py")
        if Hw.Is_Mock:
            print(f"(mock true tau = {Cfg.JOINT_LAG_TAU}; the acceleration ramp adds a little)")
        plt.xlabel("time [s]"); plt.ylabel("joint 6 speed [rad/s]"); plt.grid(alpha=0.3)
        plt.title("speedJ step response and first-order fit")
        plt.savefig("calib/joint_lag.png", dpi=150)
    finally:
        Hw.close()
