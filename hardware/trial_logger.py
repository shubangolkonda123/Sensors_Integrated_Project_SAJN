"""
trial_logger.py - writes one CSV per trial (every control step) and one summary row
per trial. These files are the evidence for the quantitative results on the project page.
"""
import csv
import os
import numpy as np


def _Expand(Name, N):
    return [f"{Name}_{i}" for i in range(N)]


COLUMNS = (["t", "state"] + _Expand("q", 6) + _Expand("qd_meas", 6) + _Expand("qd_hat", 6)
           + _Expand("q_ref", 6) + _Expand("u", 6) + _Expand("tool", 3) + _Expand("goal", 3)
           + _Expand("est", 3) + _Expand("est_std", 3) + _Expand("meas", 3)
           + ["nis", "pos_err", "rot_err", "manip"] + _Expand("gt", 3))


class Trial_Logger:
    def __init__(self, Path):
        os.makedirs(os.path.dirname(Path) or ".", exist_ok=True)
        self.File = open(Path, "w", newline="")
        self.Writer = csv.writer(self.File)
        self.Writer.writerow(COLUMNS)

    def log(self, **Row):
        Values = []
        for Key in ("t", "state"):
            Values.append(Row[Key])
        for Key, N in (("q", 6), ("qd_meas", 6), ("qd_hat", 6), ("q_ref", 6), ("u", 6),
                       ("tool", 3), ("goal", 3), ("est", 3), ("est_std", 3), ("meas", 3)):
            V = Row.get(Key)
            Values += list(np.round(V, 6)) if V is not None else [np.nan] * N
        for Key in ("nis", "pos_err", "rot_err", "manip"):
            Values.append(round(float(Row.get(Key, np.nan)), 6))
        Gt = Row.get("gt")
        Values += list(np.round(Gt, 6)) if Gt is not None else [np.nan] * 3
        self.Writer.writerow(Values)

    def close(self):
        self.File.close()


def append_summary(Path, Result):
    New = not os.path.exists(Path)
    with open(Path, "a", newline="") as F:
        W = csv.DictWriter(F, fieldnames=list(Result.keys()))
        if New:
            W.writeheader()
        W.writerow(Result)
