"""
plot_trial.py - figures for the project page from the logs.

    python plot_trial.py logs/trial_000.csv            # one trial: 4 panels
    python plot_trial.py --summary logs/summary.csv    # error statistics over all trials

Panels for one trial:
  1. target estimate +- 2 sigma vs raw measurements (and ground truth on mock)
  2. joint speed: observer estimate vs the robot's own measurement (observer check)
  3. tool position error to the current goal, with the phases shaded
  4. NIS of the target filter vs the 95% gate (filter consistency)
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as Cfg


def plot_trial(Path):
    D = pd.read_csv(Path)
    T = D["t"].values
    Fig, Ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    # 1. estimate +- 2 sigma, x axis shown (change to y/z as needed)
    for i, (Name, Col) in enumerate(zip("xyz", ["C0", "C1", "C2"])):
        Est, Std = D[f"est_{i}"].values, D[f"est_std_{i}"].values
        Off = np.nanmean(Est)            # plot relative to the mean so all axes fit
        Ax[0].plot(T, (Est - Off) * 1000, Col, label=f"{Name} estimate")
        Ax[0].fill_between(T, (Est - Off - 2 * Std) * 1000, (Est - Off + 2 * Std) * 1000, color=Col, alpha=0.2)
        Ax[0].plot(T, (D[f"meas_{i}"] - Off) * 1000, ".", color=Col, ms=2, alpha=0.5)
        if D[f"gt_{i}"].notna().any():
            Ax[0].plot(T, (D[f"gt_{i}"] - Off) * 1000, "--", color=Col, lw=1)
    Ax[0].set_ylabel("target pos. - mean [mm]")
    Ax[0].set_title("Target KF: estimate (line), +-2 sigma (band), measurements (dots), truth (dashed, mock)")
    Ax[0].legend(fontsize=8, ncol=3)
    Ax[0].set_ylim(-15, 15)

    # 2. observer check on the joint with the most motion
    J = int(np.argmax([np.nanstd(D[f"qd_meas_{j}"]) for j in range(6)]))
    Ax[1].plot(T, D[f"qd_meas_{J}"], label=f"joint {J + 1} measured by robot")
    Ax[1].plot(T, D[f"qd_hat_{J}"], "--", label="observer estimate")
    Ax[1].plot(T, D[f"u_{J}"], ":", label="command u")
    Rms = np.sqrt(np.nanmean((D[f"qd_hat_{J}"] - D[f"qd_meas_{J}"]) ** 2))
    Ax[1].set_ylabel("joint speed [rad/s]")
    Ax[1].set_title(f"Joint observer vs measured speed (RMS diff {Rms * 1000:.1f} mrad/s)")
    Ax[1].legend(fontsize=8)

    # 3. position error with phases shaded
    Ax[2].semilogy(T, D["pos_err"] * 1000)
    Ax[2].axhline(Cfg.POS_TOL * 1000, color="k", ls=":", lw=1, label="tolerance")
    States = D["state"].values
    Edges = [0] + [k for k in range(1, len(States)) if States[k] != States[k - 1]] + [len(States) - 1]
    for k, (a, b) in enumerate(zip(Edges[:-1], Edges[1:])):
        if k % 2:                                    # shade every second phase
            for A_ in Ax:
                A_.axvspan(T[a], T[b], color="grey", alpha=0.08)
        Ax[2].text(T[a], 0.95, States[a], fontsize=7, rotation=90, va="top",
                   transform=Ax[2].get_xaxis_transform())
    Ax[2].set_ylabel("tool pos. error [mm]")
    Ax[2].set_title("PBVS: distance from tool to current goal")
    Ax[2].legend(fontsize=8)

    # 4. NIS
    Ax[3].plot(T, D["nis"], ".", ms=3)
    Ax[3].axhline(Cfg.NIS_GATE, color="r", ls="--", lw=1, label="95% gate (chi2, 3 dof)")
    Ax[3].axhline(3.0, color="g", ls=":", lw=1, label="expected mean (3)")
    Ax[3].set_ylabel("NIS")
    Ax[3].set_xlabel("time [s]")
    Ax[3].set_title(f"Filter consistency: mean NIS {np.nanmean(D['nis']):.2f}")
    Ax[3].legend(fontsize=8)

    Fig.tight_layout()
    Out = os.path.splitext(Path)[0] + ".png"
    Fig.savefig(Out, dpi=140)
    print(f"saved {Out}")


def plot_summary(Path):
    S = pd.read_csv(Path)
    print(f"trials: {len(S)}   success rate: {S['success'].mean() * 100:.0f}%")
    Cols = [c for c in S.columns if c.startswith(("est_err", "grasp_err", "locked_std"))]
    if Cols:
        print(S[Cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
    if "grasp_err_x_mm" in S:
        Fig, Ax = plt.subplots(figsize=(5, 5))
        Ax.scatter(S["grasp_err_x_mm"], S["grasp_err_y_mm"], c=S["success"], cmap="RdYlGn", vmin=0, vmax=1)
        Ax.add_patch(plt.Circle((0, 0), 12, fill=False, ls="--", color="grey"))
        Ax.axhline(0, color="k", lw=0.5); Ax.axvline(0, color="k", lw=0.5)
        Ax.set_aspect("equal"); Ax.set_xlabel("grasp error x [mm]"); Ax.set_ylabel("grasp error y [mm]")
        Ax.set_title("Grasp position error (dashed = 12 mm finger tolerance)")
        Out = os.path.splitext(Path)[0] + "_grasp.png"
        Fig.savefig(Out, dpi=140)
        print(f"saved {Out}")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("path")
    P.add_argument("--summary", action="store_true")
    A = P.parse_args()
    plot_summary(A.path) if A.summary else plot_trial(A.path)
