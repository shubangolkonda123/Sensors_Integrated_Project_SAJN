"""
calibrated_moving_block_pick.py

Run this file instead of moving_block_pick.py.

Workflow
--------
1. Check for a saved simulated-camera calibration.
2. Run calibrate_sim_camera.py if calibration is missing or requested.
3. Load the checkerboard-estimated camera matrix and distortion.
4. Replace moving_block_pick.WristCameraCalibration.
5. Start the existing moving-block pick pipeline.

Required files in the same folder
--------------------------------
calibrated_moving_block_pick.py
calibrate_sim_camera.py
moving_block_pick.py

The robot XML and mesh assets stay in their existing locations.

Examples
--------
Normal run:
    python calibrated_moving_block_pick.py

Force new calibration:
    python calibrated_moving_block_pick.py --recalibrate

Set the robot XML:
    python calibrated_moving_block_pick.py --model path/to/ur3e.xml

Pass arguments to the picker:
    python calibrated_moving_block_pick.py --seed 7 --duration 90

Notes
-----
- The checkerboard exists only in the calibration scene.
- Picking reloads the original XML, without the generated checkerboard.
- The saved intrinsic calibration is still used during picking.
- Existing picking, randomisation, observer and controller logic
  remain in moving_block_pick.py.
- Use the updated calibration script that permits a reduced number
  of views when sufficient training and validation images exist.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import calibrate_sim_camera as calibration_task
import moving_block_pick as pick_task


# ------------------------------------------------------------
# DEFAULT SETTINGS
# ------------------------------------------------------------

SCRIPT_FOLDER = Path(__file__).resolve().parent

DEFAULT_CALIBRATION_FOLDER = (
    SCRIPT_FOLDER / "results" / "sim_camera_calibration"
)

CALIBRATION_FILENAME = "camera_calibration.npz"

# Set True to capture new checkerboard images every time you run.
RECALIBRATE_EVERY_RUN = True

CALIBRATION_TRAINING_VIEWS = 20
CALIBRATION_TEST_VIEWS = 5
CALIBRATION_ATTEMPTS = 600


# ------------------------------------------------------------
# RUN AN EXISTING SCRIPT'S MAIN FUNCTION
# ------------------------------------------------------------

def run_script_main(module, arguments):
    original_argv = sys.argv[:]

    try:
        sys.argv = [
            str(module.__file__),
            *[str(argument) for argument in arguments],
        ]
        module.main()
    finally:
        sys.argv = original_argv


# ------------------------------------------------------------
# LOAD AND VALIDATE SAVED CALIBRATION
# ------------------------------------------------------------

def load_calibration(calibration_file, expected_size):
    calibration_file = Path(calibration_file)

    if not calibration_file.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_file}"
        )

    with np.load(
        calibration_file,
        allow_pickle=False,
    ) as saved:
        required_fields = {"K", "Dist", "image_size"}
        missing_fields = required_fields.difference(saved.files)

        if missing_fields:
            raise ValueError(
                "Calibration is missing fields: "
                f"{sorted(missing_fields)}"
            )

        K = np.asarray(
            saved["K"],
            dtype=np.float64,
        ).copy()

        distortion = np.asarray(
            saved["Dist"],
            dtype=np.float64,
        ).reshape(-1, 1).copy()

        size_values = np.asarray(
            saved["image_size"],
            dtype=int,
        ).ravel()

        if size_values.size != 2:
            raise ValueError(
                "image_size must contain [width, height]."
            )

        image_size = tuple(int(value) for value in size_values)

        rms = None
        if "Rms" in saved.files:
            rms_values = np.asarray(
                saved["Rms"],
                dtype=float,
            ).ravel()

            if rms_values.size != 1:
                raise ValueError(
                    "Rms must contain one scalar value."
                )

            rms = float(rms_values[0])

    if K.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 camera matrix; received {K.shape}."
        )

    if not np.all(np.isfinite(K)):
        raise ValueError(
            "Camera matrix contains non-finite values."
        )

    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(
            "Camera focal lengths must be positive."
        )

    if not np.allclose(
        K[2],
        [0.0, 0.0, 1.0],
        atol=1e-8,
    ):
        raise ValueError(
            "Invalid final row in camera matrix."
        )

    if distortion.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            "Unexpected number of distortion coefficients: "
            f"{distortion.size}"
        )

    if not np.all(np.isfinite(distortion)):
        raise ValueError(
            "Distortion coefficients contain non-finite values."
        )

    if image_size != tuple(expected_size):
        raise ValueError(
            f"Calibration resolution is {image_size}, but picking "
            f"uses {tuple(expected_size)}. Use matching resolutions "
            "and recalibrate."
        )

    if rms is not None and not np.isfinite(rms):
        raise ValueError(
            "Calibration RMS is not finite."
        )

    return {
        "K": K,
        "Dist": distortion,
        "image_size": image_size,
        "Rms": rms,
        "source": str(calibration_file.resolve()),
    }


# ------------------------------------------------------------
# CAMERA CLASS USED BY THE PICKING PIPELINE
# ------------------------------------------------------------

def install_calibrated_camera(calibration):
    """
    Install the calibrated camera before pick_task.main() creates
    its Sensors object.

    This preserves the interface of WristCameraCalibration:
        camera.K
        camera.rays
        camera.deproject(depth_image)
    """

    expected_width, expected_height = calibration["image_size"]

    class CalibratedWristCamera:
        def __init__(
            self,
            model,
            camera_id,
            width=pick_task.WIDTH,
            height=pick_task.HEIGHT,
        ):
            self.width = int(width)
            self.height = int(height)

            if (self.width, self.height) != (
                expected_width,
                expected_height,
            ):
                raise ValueError(
                    "Requested camera dimensions do not match "
                    "the saved intrinsic calibration."
                )

            # Keep constructor compatibility with the original class.
            # Do not derive intrinsics from model.cam_fovy here.
            self.camera_id = camera_id

            self.K = calibration["K"].copy()
            self.Dist = calibration["Dist"].copy()
            self.distortion_coeffs = self.Dist
            self.rms = calibration["Rms"]
            self.source = calibration["source"]

            u, v = np.meshgrid(
                np.arange(self.width, dtype=np.float32),
                np.arange(self.height, dtype=np.float32),
            )

            pixel_coordinates = np.stack(
                [u, v],
                axis=-1,
            ).reshape(-1, 1, 2)

            normalized_coordinates = cv2.undistortPoints(
                pixel_coordinates,
                self.K,
                self.Dist,
            ).reshape(
                self.height,
                self.width,
                2,
            )

            self.rays = np.concatenate(
                [
                    normalized_coordinates,
                    np.ones(
                        (self.height, self.width, 1),
                        dtype=normalized_coordinates.dtype,
                    ),
                ],
                axis=-1,
            ).astype(np.float64)

        def deproject(self, depth_image):
            depth_image = np.asarray(
                depth_image,
                dtype=np.float64,
            )

            expected_shape = (
                self.height,
                self.width,
            )

            if depth_image.shape != expected_shape:
                raise ValueError(
                    f"Expected depth image shape {expected_shape}; "
                    f"received {depth_image.shape}."
                )

            return self.rays * depth_image[..., None]

    pick_task.WristCameraCalibration = CalibratedWristCamera


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the simulated wrist camera, then run "
            "moving-block picking with the estimated intrinsics."
        ),
        allow_abbrev=False,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=pick_task.default_model_path(),
        help="Path to the original robot XML.",
    )

    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Capture new images and replace saved calibration.",
    )

    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Prepare/check calibration without running picking.",
    )

    parser.add_argument(
        "--calibration-folder",
        type=Path,
        default=DEFAULT_CALIBRATION_FOLDER,
    )

    parser.add_argument(
        "--calibration-views",
        type=int,
        default=CALIBRATION_TRAINING_VIEWS,
    )

    parser.add_argument(
        "--calibration-test-views",
        type=int,
        default=CALIBRATION_TEST_VIEWS,
    )

    parser.add_argument(
        "--calibration-attempts",
        type=int,
        default=CALIBRATION_ATTEMPTS,
    )

    # Unknown arguments are forwarded unchanged to the existing picker.
    args, picking_arguments = parser.parse_known_args()

    if args.calibration_views < 10:
        parser.error(
            "--calibration-views must be at least 10."
        )

    if args.calibration_test_views < 1:
        parser.error(
            "--calibration-test-views must be at least 1."
        )

    if args.calibration_attempts < 1:
        parser.error(
            "--calibration-attempts must be positive."
        )

    model_path = args.model.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Robot XML not found: {model_path}"
        )

    calibration_folder = args.calibration_folder.resolve()
    calibration_file = (
        calibration_folder / CALIBRATION_FILENAME
    )

    picking_size = (
        int(pick_task.WIDTH),
        int(pick_task.HEIGHT),
    )

    calibration_size = (
        int(calibration_task.WIDTH),
        int(calibration_task.HEIGHT),
    )

    if calibration_size != picking_size:
        raise ValueError(
            f"Calibration uses {calibration_size}, while picking "
            f"uses {picking_size}. Set WIDTH and HEIGHT consistently "
            "in both scripts."
        )

    print("\n" + "=" * 64)
    print("CALIBRATED MOVING-BLOCK PICK")
    print("=" * 64)
    print("Robot XML:", model_path)
    print("Image resolution:", picking_size)
    print("Calibration file:", calibration_file)
    print("Checkerboard during picking: absent")

    recalibrate = (
        args.recalibrate
        or RECALIBRATE_EVERY_RUN
        or not calibration_file.is_file()
    )

    calibration = None

    if not recalibrate:
        try:
            calibration = load_calibration(
                calibration_file,
                picking_size,
            )
        except (ValueError, KeyError, OSError, EOFError) as error:
            print("\nSaved calibration cannot be used:")
            print(error)
            print("Starting a fresh calibration.")
            recalibrate = True

    if recalibrate:
        print("\nSTAGE 1: SIMULATED CHECKERBOARD CALIBRATION")
        print(
            "The calibration scene contains the checkerboard."
        )
        print(
            "Complete capture and calibration before picking starts."
        )

        run_script_main(
            calibration_task,
            [
                "--model",
                model_path,
                "--views",
                args.calibration_views,
                "--test-views",
                args.calibration_test_views,
                "--attempts",
                args.calibration_attempts,
                "--output",
                calibration_folder,
            ],
        )

        # If calibration raises an error, execution stops above.
        # Do not silently fall back to model-derived intrinsics.
        calibration = load_calibration(
            calibration_file,
            picking_size,
        )
    else:
        print("\nSTAGE 1: REUSING SAVED CALIBRATION")

    print("\nEstimated camera matrix K:")
    print(calibration["K"])

    print("\nDistortion coefficients:")
    print(calibration["Dist"].ravel())

    if calibration["Rms"] is not None:
        print(
            "\nTraining reprojection RMS: "
            f"{calibration['Rms']:.4f} pixels"
        )

    print("\nCalibration source:")
    print(calibration["source"])

    if args.calibration_only:
        print("\nCalibration is ready. Picking was not started.")
        return

    install_calibrated_camera(calibration)

    # Store which calibration this launcher used.
    # This supplements the picker's own result files.
    usage_record = {
        "model_path": str(model_path),
        "calibration_source": calibration["source"],
        "image_size": list(calibration["image_size"]),
        "K": calibration["K"].tolist(),
        "Dist": calibration["Dist"].ravel().tolist(),
        "training_rms_px": calibration["Rms"],
        "checkerboard_present_during_pick": False,
        "picking_arguments": picking_arguments,
    }

    calibration_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    (calibration_folder / "last_pick_calibration.json").write_text(
        json.dumps(usage_record, indent=2),
        encoding="utf-8",
    )

    print("\nSTAGE 2: MOVING-BLOCK PICK")
    print(
        "Loading the original scene without the checkerboard."
    )
    print(
        "RGB-D localisation will use the estimated camera matrix."
    )
    print(
        "The existing moving-block controller and estimators "
        "remain unchanged."
    )

    run_script_main(
        pick_task,
        [
            "--model",
            model_path,
            *picking_arguments,
        ],
    )


if __name__ == "__main__":
    main()