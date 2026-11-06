"""
evaluate_calibrated_pick.py

Run calibrated moving-block picking with evaluation-only ground truth
and lab-style figures.

Required local files:
    calibrated_moving_block_pick.py
    calibrate_sim_camera.py
    moving_block_pick.py

Normal run:
    python evaluate_calibrated_pick.py

Reproducible trial:
    python evaluate_calibrated_pick.py --seed 1

Change physical depth noise:
    python evaluate_calibrated_pick.py --seed 1 --depth-noise 0.008

Change assumed BLOCK FILTER measurement covariance:
    python evaluate_calibrated_pick.py --seed 1 --block-r-scale 0.1

Change assumed BLOCK FILTER process covariance:
    python evaluate_calibrated_pick.py --seed 1 --block-q-scale 10

The scaling arguments change the block estimator's assumptions.
They do not change physical sensor noise or controller LQR weights.

Outputs are saved to a unique evaluation directory:
    evaluation_log.npz
    joint_metrics.csv
    metrics.json
    figures/*.png
    plus the existing picker's result files.

All figures are generated after the simulation ends.
They are then displayed unless --no-show is supplied.

Ground truth is read only inside the evaluation class.
No ground-truth data is returned to the estimator or controller.

Object evaluation:
- XY estimates are compared with the target body's XY origin.
- Z estimates are compared with the picker's grasp-height convention,
  tabletop top + 0.037 m, NOT the target body's centre height.
- Object-filter metrics are computed only at new vision timestamps
  while the filter is active in the moving TRACK phase.
- The frozen object estimate after motion stops is not scored as a
  continuing observer during grasp/lift.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.stats import chi2

import moving_block_pick as pick
import calibrated_moving_block_pick as calibrated_launcher


def run_main(module, arguments):
    original = sys.argv[:]
    try:
        sys.argv = [str(module.__file__), *map(str, arguments)]
        module.main()
    finally:
        sys.argv = original


def rms(values, axis=0):
    return np.sqrt(np.mean(np.asarray(values) ** 2, axis=axis))


def nees(error, covariance):
    try:
        return float(error @ np.linalg.solve(covariance, error))
    except np.linalg.LinAlgError:
        return np.nan


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def metric_dict(error, sigma):
    error = np.asarray(error)
    sigma = np.asarray(sigma)

    return {
        "rmse": np.asarray(rms(error)).tolist(),
        "mean_error": np.mean(error, axis=0).tolist(),
        "maximum_absolute_error": np.max(
            np.abs(error), axis=0
        ).tolist(),
        "fraction_inside_2sigma": np.mean(
            np.abs(error) <= 2 * sigma, axis=0
        ).tolist(),
    }


def add_phase_lines(ax, times, states):
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1

    for index in changes:
        ax.axvline(
            times[index],
            color="0.65",
            linestyle=":",
            linewidth=0.8,
        )


def style_axis(ax):
    ax.grid(True, linestyle="--", alpha=0.35)


def install_diagnostics(output, show, q_scale, r_scale):
    original_sensors = pick.Sensors
    original_evaluation = pick.Evaluation
    original_block_filter = pick.BlockConstantVelocityEKF

    cache = {
        "encoder": None,
        "vision_sequence": 0,
        "measurement": None,
    }

    class LoggedSensors(original_sensors):
        def encoder(self, plant):
            measured = super().encoder(plant)
            cache["encoder"] = measured.copy()
            return measured

        def find_target(self, rgb, depth, q_estimated):
            measurement, mask = super().find_target(
                rgb, depth, q_estimated
            )
            cache["vision_sequence"] += 1
            cache["measurement"] = (
                None if measurement is None
                else np.asarray(measurement).copy()
            )
            return measurement, mask

    class TunedBlockFilter(original_block_filter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.Q = self.Q * q_scale
            self.R = self.R * r_scale

    class DiagnosticEvaluation(original_evaluation):
        def __init__(self, model, kinematics, plant, directory):
            super().__init__(
                model, kinematics, plant, directory
            )

            self.model = model
            self.diagnostic_rows = []
            self.object_rows = []
            self.last_vision_sequence = 0

            self.table_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "tabletop",
            )
            if self.table_id < 0:
                raise ValueError("Missing tabletop geometry.")

        def record(
            self,
            plant,
            controller,
            block_ekf,
            q_reference,
            state,
        ):
            # Retain the original picker's logging.
            super().record(
                plant,
                controller,
                block_ekf,
                q_reference,
                state,
            )

            q_true = plant.qpos[
                self.kinematics.q_addresses
            ].copy()

            v_true = plant.qvel[
                self.kinematics.v_addresses
            ].copy()

            x_true = np.concatenate([q_true, v_true])
            x_estimated = controller.x.copy()
            P = controller.P.copy()

            encoder = (
                np.full(6, np.nan)
                if cache["encoder"] is None
                else cache["encoder"].copy()
            )

            self.diagnostic_rows.append({
                "time": float(plant.time),
                "state": str(state),
                "truth": x_true,
                "estimate": x_estimated,
                "P": P,
                "reference": np.asarray(q_reference).copy(),
                "encoder": encoder,
                "command": np.asarray(controller.u).copy(),
                "nees": nees(x_estimated - x_true, P),
            })

            sequence = cache["vision_sequence"]
            if sequence == self.last_vision_sequence:
                return

            self.last_vision_sequence = sequence

            # The current code stops updating the block EKF after
            # the moving phase, so do not score it during lifting.
            active = (
                block_ekf.initialised
                and state == "TRACK"
                and plant.time < pick.BLOCK_MOVE_DURATION
            )

            if not active:
                return

            body_position = plant.xpos[self.block_id].copy()

            # World-frame velocity at the body's COM.
            # For the supplied symmetric block, its COM and body
            # origin coincide.
            spatial_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                plant,
                mujoco.mjtObj.mjOBJ_BODY,
                self.block_id,
                spatial_velocity,
                0,
            )
            body_velocity = spatial_velocity[3:].copy()

            # The perception output is a grasp target, not the
            # geometric block centre. Match that Z convention.
            table_rotation = plant.geom_xmat[
                self.table_id
            ].reshape(3, 3)
            table_top = (
                plant.geom_xpos[self.table_id]
                + table_rotation[:, 2]
                * self.model.geom_size[self.table_id, 2]
            )

            target_truth = body_position.copy()
            target_truth[2] = table_top[2] + 0.037

            measurement = (
                np.full(3, np.nan)
                if cache["measurement"] is None
                else cache["measurement"].copy()
            )

            block_state = block_ekf.x.copy()
            block_P = block_ekf.P.copy()
            xy_error = block_state[:2] - target_truth[:2]

            self.object_rows.append({
                "time": float(plant.time),
                "truth": target_truth,
                "velocity_truth": body_velocity,
                "estimate": block_state,
                "P": block_P,
                "measurement": measurement,
                "xy_nees": nees(xy_error, block_P[:2, :2]),
            })

        def save(
            self,
            task,
            controller,
            block_ekf,
            camera_matrix,
        ):
            super().save(
                task, controller, block_ekf, camera_matrix
            )

            if not self.diagnostic_rows:
                print("No diagnostic samples were recorded.")
                return

            self.create_results(task)

        def create_results(self, task):
            directory = Path(output)
            figures = directory / "figures"
            figures.mkdir(parents=True, exist_ok=True)

            rows = self.diagnostic_rows

            t = np.array([r["time"] for r in rows])
            states = np.array([r["state"] for r in rows])
            truth = np.stack([r["truth"] for r in rows])
            estimate = np.stack([r["estimate"] for r in rows])
            P = np.stack([r["P"] for r in rows])
            reference = np.stack([r["reference"] for r in rows])
            encoder = np.stack([r["encoder"] for r in rows])
            command = np.stack([r["command"] for r in rows])
            joint_nees = np.array([r["nees"] for r in rows])

            error = estimate - truth
            sigma = np.sqrt(
                np.maximum(
                    np.diagonal(P, axis1=1, axis2=2), 0
                )
            )
            tracking_error = truth[:, :6] - reference

            metrics = {
                "task_final_state": str(task.state),
                "abort_reason": str(task.abort_reason),
                "samples": len(t),
                "duration_s": float(t[-1] - t[0]),
                "block_Q_scale": q_scale,
                "block_R_scale": r_scale,
                "joint_position_rad": metric_dict(
                    error[:, :6], sigma[:, :6]
                ),
                "joint_velocity_rad_s": metric_dict(
                    error[:, 6:], sigma[:, 6:]
                ),
                "joint_tracking_rmse_rad": rms(
                    tracking_error
                ).tolist(),
                "joint_ANEES": finite_mean(joint_nees),
                "joint_NEES_dimension": 12,
                "command_increment_rms_rad": (
                    rms(np.diff(command, axis=0)).tolist()
                    if len(command) > 1 else None
                ),
                "command_note": (
                    "Actuator position command and its increments; "
                    "not torque, energy or mechanical effort."
                ),
                "object_reference_note": (
                    "XY uses block body origin. Z uses horizontal "
                    "tabletop top + 0.037 m, matching grasp convention."
                ),
            }

            arrays = {
                "time": t,
                "state": states,
                "joint_truth": truth,
                "joint_estimate": estimate,
                "joint_covariance": P,
                "joint_reference": reference,
                "encoder": encoder,
                "command": command,
                "joint_nees": joint_nees,
            }

            def save_figure(fig, name):
                fig.tight_layout()
                fig.savefig(
                    figures / name,
                    dpi=180,
                    bbox_inches="tight",
                )

            # 1. True/estimated/reference joint positions.
            fig, axes = plt.subplots(
                3, 2, figsize=(14, 10), sharex=True
            )
            fig.suptitle(
                "Joint positions: true, estimated and reference"
            )

            for i, ax in enumerate(axes.flat):
                ax.plot(
                    t, np.degrees(truth[:, i]),
                    "k-", label="True",
                )
                ax.plot(
                    t, np.degrees(estimate[:, i]),
                    color="tab:red", linestyle="--",
                    label="Observer",
                )
                ax.plot(
                    t, np.degrees(reference[:, i]),
                    color="tab:green", linestyle=":",
                    label="Reference",
                )
                ax.set_ylabel(f"Joint {i + 1} (deg)")
                ax.set_xlabel("Time (s)")
                add_phase_lines(ax, t, states)
                style_axis(ax)

            axes.flat[0].legend()
            save_figure(fig, "01_joint_positions.png")

            # 2. Joint position error with uncertainty.
            fig, axes = plt.subplots(
                3, 2, figsize=(14, 10), sharex=True
            )
            fig.suptitle(
                "Joint position estimation error and ±2σ"
            )

            for i, ax in enumerate(axes.flat):
                bound = np.degrees(2 * sigma[:, i])
                ax.fill_between(
                    t, -bound, bound,
                    color="tab:blue", alpha=0.2,
                    label="±2σ",
                )
                ax.plot(
                    t, np.degrees(error[:, i]),
                    color="tab:blue", label="Estimate − truth",
                )
                ax.axhline(0, color="k", linewidth=0.6)
                ax.set_ylabel(f"Joint {i + 1} error (deg)")
                ax.set_xlabel("Time (s)")
                style_axis(ax)

            axes.flat[0].legend()
            save_figure(fig, "02_joint_position_errors.png")

            # 3. Unmeasured joint velocities.
            fig, axes = plt.subplots(
                3, 2, figsize=(14, 10), sharex=True
            )
            fig.suptitle("Joint velocities: true versus observer")

            for i, ax in enumerate(axes.flat):
                j = i + 6
                ax.plot(
                    t, np.degrees(truth[:, j]),
                    "k-", label="True",
                )
                ax.plot(
                    t, np.degrees(estimate[:, j]),
                    color="tab:red", label="Observer",
                )
                lower = np.degrees(
                    estimate[:, j] - 2 * sigma[:, j]
                )
                upper = np.degrees(
                    estimate[:, j] + 2 * sigma[:, j]
                )
                ax.fill_between(
                    t, lower, upper,
                    color="tab:red", alpha=0.15,
                    label="Observer ±2σ",
                )
                ax.set_ylabel(f"Joint {i + 1} (deg/s)")
                ax.set_xlabel("Time (s)")
                style_axis(ax)

            axes.flat[0].legend()
            save_figure(fig, "03_joint_velocities.png")

            # 4. Actual tracking performance.
            fig, axes = plt.subplots(
                2, 1, figsize=(13, 8), sharex=True
            )

            for i in range(6):
                axes[0].plot(
                    t, np.degrees(tracking_error[:, i]),
                    label=f"Joint {i + 1}",
                )
                axes[1].plot(
                    t, command[:, i] - reference[:, i],
                    label=f"Joint {i + 1}",
                )

            axes[0].set_title("True joint tracking error")
            axes[0].set_ylabel("True − reference (deg)")
            axes[1].set_title(
                "Actuator position correction—not torque"
            )
            axes[1].set_ylabel("Command − reference (rad)")
            axes[1].set_xlabel("Time (s)")

            for ax in axes:
                add_phase_lines(ax, t, states)
                style_axis(ax)
                ax.legend(ncol=3)

            save_figure(fig, "04_tracking_and_commands.png")

            # 5. Joint consistency.
            fig, ax = plt.subplots(figsize=(13, 4))
            lower, upper = chi2.ppf([0.025, 0.975], df=12)

            ax.semilogy(
                t, np.maximum(joint_nees, 1e-12),
                color="tab:purple", label="Joint NEES",
            )
            ax.axhspan(
                lower, upper, color="tab:green",
                alpha=0.12, label="Pointwise 95% reference interval",
            )
            ax.axhline(
                12, color="k", linestyle="--",
                label="Expected mean under ideal assumptions: 12",
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("NEES")
            ax.set_title(
                "Joint observer consistency—temporally correlated samples"
            )
            ax.legend()
            style_axis(ax)
            save_figure(fig, "05_joint_consistency.png")

            # Object plots use only new active vision timestamps.
            if self.object_rows:
                objects = self.object_rows

                ot = np.array([r["time"] for r in objects])
                target = np.stack([r["truth"] for r in objects])
                velocity = np.stack([
                    r["velocity_truth"] for r in objects
                ])
                bx = np.stack([r["estimate"] for r in objects])
                bP = np.stack([r["P"] for r in objects])
                measured = np.stack([
                    r["measurement"] for r in objects
                ])
                bnees = np.array([
                    r["xy_nees"] for r in objects
                ])

                bsigma = np.sqrt(np.maximum(
                    np.diagonal(bP, axis1=1, axis2=2), 0
                ))
                position_error = bx[:, :3] - target
                valid_measurements = np.all(
                    np.isfinite(measured), axis=1
                )

                metrics["object_position_m"] = metric_dict(
                    position_error, bsigma[:, :3]
                )
                metrics["object_velocity_xy_m_s"] = metric_dict(
                    bx[:, 3:5] - velocity[:, :2],
                    bsigma[:, 3:5],
                )
                metrics["object_xy_ANEES"] = finite_mean(bnees)
                metrics["object_xy_NEES_dimension"] = 2
                metrics["active_vision_detection_fraction"] = float(
                    np.mean(valid_measurements)
                )
                metrics["object_xy_vector_rmse_m"] = float(
                    np.sqrt(np.mean(np.sum(
                        position_error[:, :2] ** 2, axis=1
                    )))
                )

                if np.any(valid_measurements):
                    raw_error = (
                        measured[valid_measurements, :2]
                        - target[valid_measurements, :2]
                    )
                    filtered_error = position_error[
                        valid_measurements, :2
                    ]

                    metrics["paired_raw_xy_vector_rmse_m"] = float(
                        np.sqrt(np.mean(np.sum(raw_error**2, axis=1)))
                    )
                    metrics["paired_filtered_xy_vector_rmse_m"] = float(
                        np.sqrt(np.mean(np.sum(
                            filtered_error**2, axis=1
                        )))
                    )

                arrays.update({
                    "object_time": ot,
                    "object_target_truth": target,
                    "object_velocity_truth": velocity,
                    "object_estimate": bx,
                    "object_covariance": bP,
                    "object_measurement": measured,
                    "object_xy_nees": bnees,
                })

                fig, axes = plt.subplots(
                    3, 2, figsize=(14, 11), sharex=True
                )

                for i, name in enumerate(["X", "Y", "Grasp Z"]):
                    ax = axes[i, 0]
                    ax.plot(
                        ot, 1000 * target[:, i],
                        "k-", label="True reference",
                    )
                    ax.plot(
                        ot, 1000 * bx[:, i],
                        color="tab:red", label="Filter",
                    )
                    ax.scatter(
                        ot, 1000 * measured[:, i],
                        s=12, alpha=0.5,
                        color="tab:blue", label="RGB-D",
                    )
                    ax.set_ylabel(f"{name} (mm)")
                    style_axis(ax)

                    ax = axes[i, 1]
                    bound = 2000 * bsigma[:, i]
                    ax.fill_between(
                        ot, -bound, bound,
                        alpha=0.2, color="tab:blue",
                        label="±2σ",
                    )
                    ax.plot(
                        ot, 1000 * position_error[:, i],
                        color="tab:blue", label="Estimate − reference",
                    )
                    ax.axhline(0, color="k", linewidth=0.6)
                    ax.set_ylabel(f"{name} error (mm)")
                    style_axis(ax)

                axes[0, 0].legend()
                axes[0, 1].legend()
                axes[-1, 0].set_xlabel("Time (s)")
                axes[-1, 1].set_xlabel("Time (s)")
                fig.suptitle(
                    "Moving-phase target localisation and uncertainty"
                )
                save_figure(fig, "06_object_position_errors.png")

                fig, axes = plt.subplots(
                    2, 2, figsize=(14, 9)
                )

                ax = axes[0, 0]
                ax.plot(
                    target[:, 0], target[:, 1],
                    "k-", label="True XY",
                )
                ax.plot(
                    bx[:, 0], bx[:, 1],
                    color="tab:red", label="Estimated XY",
                )
                ax.scatter(
                    measured[:, 0], measured[:, 1],
                    s=12, alpha=0.5, label="RGB-D",
                )
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xlabel("Base X (m)")
                ax.set_ylabel("Base Y (m)")
                ax.set_title("Object trajectory")
                ax.legend()

                for i, colour in [(0, "tab:blue"), (1, "tab:orange")]:
                    axes[0, 1].plot(
                        ot, velocity[:, i],
                        color=colour, linestyle="--",
                        label=f"True {'XY'[i]}",
                    )
                    axes[0, 1].plot(
                        ot, bx[:, i + 3],
                        color=colour,
                        label=f"Estimated {'XY'[i]}",
                    )

                axes[0, 1].set_title("Object velocity")
                axes[0, 1].set_ylabel("Velocity (m/s)")
                axes[0, 1].set_xlabel("Time (s)")
                axes[0, 1].legend()

                low, high = chi2.ppf([0.025, 0.975], df=2)
                axes[1, 0].semilogy(
                    ot, np.maximum(bnees, 1e-12),
                    label="XY NEES",
                )
                axes[1, 0].axhspan(
                    low, high, alpha=0.15,
                    color="tab:green",
                    label="Pointwise 95% reference interval",
                )
                axes[1, 0].axhline(
                    2, color="k", linestyle="--",
                    label="Ideal expected mean: 2",
                )
                axes[1, 0].set_title("XY consistency")
                axes[1, 0].set_xlabel("Time (s)")
                axes[1, 0].legend()

                axes[1, 1].step(
                    ot, valid_measurements.astype(int),
                    where="mid",
                )
                axes[1, 1].set_ylim(-0.1, 1.1)
                axes[1, 1].set_yticks([0, 1])
                axes[1, 1].set_title(
                    "Vision availability after filter initialisation"
                )
                axes[1, 1].set_xlabel("Time (s)")
                axes[1, 1].set_ylabel("Valid detection")

                for ax in axes.flat:
                    style_axis(ax)

                save_figure(fig, "07_object_dashboard.png")

            np.savez_compressed(
                directory / "evaluation_log.npz", **arrays
            )

            with (directory / "joint_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    "joint",
                    "angle_estimation_rmse_deg",
                    "velocity_estimation_rmse_deg_s",
                    "true_tracking_rmse_deg",
                    "angle_2sigma_coverage",
                ])

                for i in range(6):
                    writer.writerow([
                        i + 1,
                        float(np.degrees(rms(error[:, i]))),
                        float(np.degrees(rms(error[:, i + 6]))),
                        float(np.degrees(rms(tracking_error[:, i]))),
                        float(np.mean(
                            np.abs(error[:, i]) <= 2 * sigma[:, i]
                        )),
                    ])

            (directory / "metrics.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )

            print("\nEVALUATION RESULTS")
            print("Final state:", task.state)
            print("Joint angle RMSE (deg):")
            print(np.degrees(rms(error[:, :6])))
            print("Joint velocity RMSE (deg/s):")
            print(np.degrees(rms(error[:, 6:])))
            print("Joint ANEES:", metrics["joint_ANEES"])
            print("Results:", directory.resolve())

            if show:
                plt.show()
            else:
                plt.close("all")

    pick.Sensors = LoggedSensors
    pick.BlockConstantVelocityEKF = TunedBlockFilter
    pick.Evaluation = DiagnosticEvaluation


def main():
    parser = argparse.ArgumentParser(
        description="Calibrated picking with lab-style evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--block-q-scale", type=float, default=1.0)
    parser.add_argument("--block-r-scale", type=float, default=1.0)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--results", type=Path, default=None)

    args, forwarded = parser.parse_known_args()

    if args.block_q_scale <= 0 or args.block_r_scale <= 0:
        parser.error("Covariance scale factors must be positive.")

    output = args.results
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = (
            Path(__file__).resolve().parent
            / "results"
            / f"evaluation_{stamp}"
        )

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "block_Q_scale": args.block_q_scale,
        "block_R_scale": args.block_r_scale,
        "forwarded_arguments": forwarded,
        "ground_truth_policy": "evaluation only",
    }
    (output / "run_configuration.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    install_diagnostics(
        output=output,
        show=not args.no_show,
        q_scale=args.block_q_scale,
        r_scale=args.block_r_scale,
    )

    run_main(
        calibrated_launcher,
        [*forwarded, "--results", output],
    )


if __name__ == "__main__":
    main()