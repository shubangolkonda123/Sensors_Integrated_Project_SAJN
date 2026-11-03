"""
estimator.py - the two estimators in the pipeline.          (Part C)

1. Target_KF       : Kalman filter for the object's 3D position in the BASE frame.
                     Fuses many noisy camera measurements into one estimate + uncertainty.
                     Keeps predicting (with growing uncertainty) when the object is out of view.

2. Joint_Observer  : Luenberger observer estimating joint speeds qd from encoder angles q
                     (the "unmeasured state" in the brief). The UR also reports qd, which we
                     do NOT feed in: we log it only to check the observer is right.
"""
import numpy as np
from scipy.signal import place_poles
from joint_model import discrete_model


class Target_KF:
    """
    State x = [px, py, pz] of the object's top-centre in the robot base frame.
    Model:  x_{k+1} = x_k + w,   w ~ N(0, Q)     (object is static)
    Meas.:  z_k     = x_k + v,   v ~ N(0, R_k)   (R_k comes from the detection, rotated into base)
    """

    def __init__(self, Process_Std, Nis_Gate, Max_Consec_Rejects):
        self.Q = np.eye(3) * Process_Std ** 2
        self.Nis_Gate = Nis_Gate
        self.Max_Consec_Rejects = Max_Consec_Rejects
        self.reset()

    def reset(self):
        self.X = None
        self.P = None
        self.N_Updates = 0
        self.N_Rejects = 0
        self.Consec_Rejects = 0
        self.Last_Nis = np.nan

    @property
    def initialised(self):
        return self.X is not None

    def predict(self):
        self.Last_Nis = np.nan              # only steps with a measurement get an NIS value
        if self.initialised:
            self.P = self.P + self.Q        # F = I, so only the covariance changes

    def update(self, Z, R):
        """Z: measured position (base frame). R: its 3x3 covariance. Returns True if accepted."""
        if not self.initialised:
            self.X, self.P = Z.copy(), R.copy()
            self.N_Updates = 1
            return True
        Y = Z - self.X                      # innovation
        S = self.P + R                      # innovation covariance (H = I)
        Nis = float(Y @ np.linalg.solve(S, Y))
        self.Last_Nis = Nis
        if Nis > self.Nis_Gate:             # outlier gate: too unlikely to be real
            self.N_Rejects += 1
            self.Consec_Rejects += 1
            if self.Consec_Rejects >= self.Max_Consec_Rejects:
                print("[kf] many consecutive rejects -> object probably moved, re-initialising")
                self.reset()
                self.update(Z, R)
            return False
        self.Consec_Rejects = 0
        K = self.P @ np.linalg.inv(S)       # Kalman gain
        self.X = self.X + K @ Y
        I_K = np.eye(3) - K
        self.P = I_K @ self.P @ I_K.T + K @ R @ K.T   # Joseph form: stays symmetric & positive
        self.N_Updates += 1
        return True

    def std(self):
        return np.sqrt(np.diag(self.P)) if self.initialised else np.full(3, np.nan)


class Joint_Observer:
    """
    Current-estimator form (predict, then correct with the new measurement):
        x_pred = Ad x_hat + Bd u_prev
        x_hat  = x_pred + L (q_meas - C x_pred)
    Same 2-state model for all 6 joints, so X_Hat is 6x2: column 0 = q, column 1 = qd.
    """

    def __init__(self, Tau, Dt, Poles_Continuous, Q0):
        self.Ad, self.Bd, self.C = discrete_model(Tau, Dt)
        Poles_Discrete = np.exp(np.array(Poles_Continuous) * Dt)
        # Observer gain by pole placement on the dual system (Ad^T, C^T)
        self.L = place_poles(self.Ad.T, self.C.T, Poles_Discrete).gain_matrix.T   # 2x1
        self.X_Hat = np.column_stack([np.asarray(Q0, float), np.zeros(6)])

    def reset(self, Q0):
        self.X_Hat = np.column_stack([np.asarray(Q0, float), np.zeros(6)])

    def step(self, Q_Meas, U_Prev):
        X_Pred = self.X_Hat @ self.Ad.T + np.outer(U_Prev, self.Bd[:, 0])
        Innov = Q_Meas - X_Pred[:, 0]
        self.X_Hat = X_Pred + np.outer(Innov, self.L[:, 0])
        return self.X_Hat

    @property
    def Q_Hat(self):
        return self.X_Hat[:, 0]

    @property
    def Qd_Hat(self):
        return self.X_Hat[:, 1]
