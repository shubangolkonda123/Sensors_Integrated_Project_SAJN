"""Part E: summarise repeated evaluation trials.
Run each trial first, e.g. for seeds 1-10:
  python evaluated_calibrated_pick.py --no-show --seed 1 --results trials/nominal_s1 --calibration-folder trials/nominal_s1/calibration
Then: python summarise_trials.py trials"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOCAL_REF = 207.846   # renderer focal length (px)
Trials = Path(sys.argv[1] if len(sys.argv) > 1 else "trials")
Rows = []
for D in sorted((p for p in Trials.iterdir() if (p / "metrics.json").is_file()), key=lambda p: int(p.name.rsplit("_s", 1)[1])):
    M = json.loads((D / "metrics.json").read_text())
    S = json.loads((D / "summary.json").read_text())
    Cal = json.loads((D / "calibration" / "calibration_report.json").read_text())
    T = np.genfromtxt(D / "timeseries.csv", delimiter=",", names=True, dtype=None, encoding=None)
    i = np.argmax(T["state"] == "DESCEND")   # grasp target is locked here
    K = np.array(Cal["K_estimated"])
    Rows.append({
        "trial": D.name, "success": int(S["task_state"] == "HOLD" and S["held_lift_detected"]),
        "focal_px": K[[0, 1], [0, 1]].mean(), "cx_err_px": K[0, 2] - 159.5, "cy_err_px": K[1, 2] - 119.5,
        "calib_rms_px": Cal["calibration_rms_px"], "heldout_rms_px": Cal["held_out_mean_rms_px"],
        "joint_q_rmse_mrad": 1000 * np.mean(M["joint_position_rad"]["rmse"]),
        "joint_qd_rmse_mrad_s": 1000 * np.mean(M["joint_velocity_rad_s"]["rmse"]),
        "joint_anees": M["joint_ANEES"], "track_rmse_max_mrad": 1000 * max(M["joint_tracking_rmse_rad"]),
        "cube_xy_rmse_mm": 1000 * M["object_xy_vector_rmse_m"], "cube_raw_xy_rmse_mm": 1000 * M["paired_raw_xy_vector_rmse_m"],
        "cube_anees": M["object_xy_ANEES"], "vision_fraction": M["active_vision_detection_fraction"],
        "grasp_target_err_mm": 1000 * np.hypot(T["block_error_x"][i], T["block_error_y"][i])})

with open(Trials / "trials_summary.csv", "w", newline="") as F:
    W = csv.DictWriter(F, fieldnames=Rows[0].keys())
    W.writeheader()
    W.writerows(Rows)
for Key in list(Rows[0])[1:]:
    V = np.array([r[Key] for r in Rows])
    print(f"{Key:22s} mean {V.mean():8.3f}  min {V.min():8.3f}  max {V.max():8.3f}")

F = np.array([r["focal_px"] for r in Rows])
E = np.array([r["cube_xy_rmse_mm"] for r in Rows])
G = np.array([r["grasp_target_err_mm"] for r in Rows])
Slope, Offset = np.polyfit(F, E, 1)
print(f"correlation focal vs cube error {np.corrcoef(F, E)[0, 1]:.2f}, slope {Slope:.2f} mm/px")

Fig, Ax = plt.subplots(1, 2, figsize=(11, 4))
Seeds = [int(r["trial"].rsplit("_s", 1)[1]) for r in Rows]
Ax[0].bar(Seeds, G, color="C0", label="grasp target error at lock")
Ax[0].bar(Seeds, E, color="none", edgecolor="k", label="cube XY RMSE while tracking")
Ax[0].set(xlabel="trial (seed)", ylabel="error (mm)", title=f"{sum(r['success'] for r in Rows)}/{len(Rows)} picks succeeded")
Ax[0].legend(fontsize=8, ncol=2, loc="upper center")
Ax[0].set_ylim(0, 3)
Ax[1].scatter(F, E, label="cube XY RMSE")
Ax[1].scatter(F, G, marker="x", label="grasp target error")
Ax[1].plot(np.sort(F), Slope * np.sort(F) + Offset, "C0--", lw=1)
Ax[1].axvline(FOCAL_REF, color="k", ls=":", label="true focal length")
Ax[1].set(xlabel="calibrated focal length (px)", ylabel="error (mm)", title="Cube error follows the calibration")
Ax[1].legend(fontsize=8)
for a in Ax:
    a.grid(alpha=0.3)
Fig.tight_layout()
Fig.savefig(Path(__file__).resolve().parent.parent / "static" / "images" / "08_repeated_trials.png", dpi=150)
