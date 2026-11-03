# UR3 vision-guided pick: hardware pipeline (Track B)

Python pipeline for running the Track B pick on the **real UR3e** in the UTS Mechatronics
Lab. Every hardware piece is behind a small wrapper, and there is a full **mock mode**
(fake robot, camera and gripper) so the whole loop can be run and debugged on a laptop
before going near the arm.

**Our lab setup** (matches the lab repo [UTS-RI/UR3_RG2_rosbridge](https://github.com/UTS-RI/UR3_RG2_rosbridge)):

| Part | What we use | Setting in `config.py` |
|---|---|---|
| Arm | UR3e, `ur_rtde` over an Ethernet cable, pendant in **Remote** | `ROBOT_MODEL = "UR3e"`, `ROBOT_IP` |
| Gripper | OnRobot RG2 via the lab's `onRobot` library | `GRIPPER_TYPE = "onrobot"` |
| Camera | Intel RealSense D435i on a **fixed stand** beside the robot (eye-to-hand) | `CAMERA_MOUNT = "fixed"` |

The brief describes a wrist-mounted camera. We could not mount one, so the camera is
fixed on a stand; state this on the project page. Consequences:
- the camera pose `T_Base_Cam` is **constant** (one calibration, no joint-angle error in the camera pose),
- the camera never gets too close to the object (no minimum-range problem),
- but the **arm can block the view** of the object, and a bumped stand ruins the calibration
  (tape/clamp it down, recalibrate if it moves),
- it sees the top AND sides of the cube, so `detection.py` finds the **top face** in 3D
  (base frame) instead of using the median depth of the blob.
The wrist-camera code is still there: `CAMERA_MOUNT = "wrist"`.

---

## 1. Quick start (mock mode, no hardware)

```bash
pip install -r requirements.txt        # ur_rtde / pyrealsense2 only needed for real hardware
python joint_model.py                  # Part A: model + controllability / observability
python main_pick.py --trials 5         # 5 picks, random object positions
python plot_trial.py logs/trial_000.csv
python plot_trial.py --summary logs/summary.csv
```

Robustness runs (Part E):
```bash
python main_pick.py --trials 10 --occlude      --log-dir logs_occlude   # object hidden during approach
python main_pick.py --trials 10 --disturb      --log-dir logs_disturb   # 3 deg bump on the base joint
python main_pick.py --trials 10 --noise-scale 3 --log-dir logs_noise    # 3x more depth noise than modelled
```

---

## 2. How the pipeline works

```
 RealSense ──colour+depth──► detection.py ──P_Cam, R_Cam──► frame chain ──► Target_KF ──► object estimate
                              (HSV blob,                  camera→flange→base    (static KF,      + covariance
                               median depth)              (hand-eye calib)       NIS gating)          │
                                                                                                       ▼
 UR3 encoders ──q──► Joint_Observer ──x̂=[q,q̇]──►  Joint_LQR  ◄──q_ref, q̇_ref── PBVS + Jacobian ◄── goal pose
                     (Luenberger)                 (inner loop)                  (outer loop)
                                                       │
                                                       └──u (speedJ)──► UR3
```

State machine per trial (`main_pick.py`):
`SEARCH → APPROACH → DESCEND → GRASP → LIFT → LOWER → RELEASE → RETREAT`

| Phase | What happens | Camera |
|---|---|---|
| SEARCH | Arm still at home, filter collects measurements until every axis std < 4 mm | on |
| APPROACH | PBVS to 12 cm above the object | on, keeps refining |
| DESCEND | Estimate is **locked**; descend blind (camera too close to see) | off |
| GRASP / LIFT / LOWER / RELEASE / RETREAT | Close, lift, put back, let go, move up | off |

**Ground truth** (mock only) is used to draw the fake image and to score trials afterwards.
Search `main_pick.py` for `GROUND TRUTH` to see every place it appears. The controller only
ever uses the filter estimate and the observer state.

---

## 3. Files and where they map to the marking criteria

| File | What it does | Criterion |
|---|---|---|
| `config.py` | Every tunable number, with where it comes from | all |
| `joint_model.py` | State-space model of a speedJ-controlled joint, ZOH discretisation, ctrb/obsv ranks | **A** |
| `identify_joint_lag.py` | Step test to measure τ of the UR velocity loop | **A** |
| `ur3_kinematics.py` | UR3/UR3e DH forward kinematics + geometric Jacobian | A/D |
| `camera.py` | RealSense wrapper: aligned depth, spatial + temporal filters, factory intrinsics | **B** |
| `calibrate_intrinsics.py` | Checkerboard calibration, compared with factory intrinsics | **B** |
| `calibrate_handeye.py` | `--fixed`: eye-to-hand calibration (T_Base_Cam, board on the gripper). Without it: eye-in-hand (T_Flange_Cam). Own Park solver + OpenCV solvers compared | **B** |
| `measure_depth_noise.py` | Measures σ_z at several heights and fits σ_z = kZ² + σ₀ (backs up R) | **B/C** |
| `hsv_tuner.py` | Sliders to tune the colour threshold under lab lighting | B |
| `detection.py` | HSV blob → centroid → median depth → 3D point + covariance | **B** |
| `estimator.py` | `Target_KF` (object position) and `Joint_Observer` (joint speeds) | **C** |
| `controller.py` | `Joint_LQR` inner loop, `PBVS_Controller` outer loop, damped least squares | **D** |
| `main_pick.py` | State machine, safety checks, logging, robustness switches | **E** |
| `plot_trial.py` | Project-page figures: estimate ±2σ, observer check, tracking error, NIS | C/D/E |
| `robot.py`, `gripper.py` | Real ur_rtde wrapper; OnRobot RG2 (lab) and Robotiq grippers | — |
| `mock_hardware.py`, `hardware.py` | Mock devices and the switch between mock/real | — |

### Design choices to defend in the viva

- **Joint model.** speedJ is a velocity command, so the plant is a velocity servo, not a
  torque-driven link: `q̈ = (u − q̇)/τ`. Controllable (rank 2). Observable from q
  (rank 2) but **not** from q̇ alone (rank 1), which is why the observer uses the encoders.
- **Observer.** The UR also reports q̇. We deliberately don't feed it in; we log it to
  validate the observer (panel 2 of the trial plot).
- **LQR weights.** `Q = diag(400, 1)`, `R = 1` gives K ≈ [16.7, 0.93], closed-loop poles
  ≈ 0.70 ± 0.08j (discrete). Position error is penalised far more than velocity error.
  The limit on the gains is the speed clip and the 50 Hz loop rate.
- **Measurement noise R.** From a stereo depth model, σ_z = kZ² + σ₀, rotated into the
  base frame each step (`R_base = R_bc R_cam R_bcᵀ`). Measure k and σ₀ on the real camera.
- **Process noise Q.** Object is static, but Q > 0 absorbs kinematic and hand-eye error
  as the arm moves, and makes uncertainty grow honestly when the object is out of view.
- **Outlier gating.** Measurements with NIS > 7.81 (χ², 3 DOF, 95%) are rejected. After
  15 rejects in a row the filter assumes the object moved and re-initialises.
- **"What happens when the object leaves the view?"** During APPROACH the filter keeps
  predicting and P grows (see `--occlude`). During DESCEND the camera is below its
  minimum range, so the estimate is locked by design.

### Results in mock mode with the fixed camera (UR3e + RG2 geometry)

8 trials each, seed 11: nominal 8/8, joint bump 8/8, occlusion 8/8, 3x depth noise 8/8.
Estimate error ≈ 2 mm in x/y/z, which is the 2 mm calibration error built into the mock.
`REF_LEASH` was reduced from 0.05 to 0.02 rad: with the UR3e and the longer RG2, a 3° bump
on the base joint left the reference 0.05 rad away from the arm and the approach oscillated
until it timed out (2 of 3 bump trials failed). Good example of a diagnosed failure.

Mock limitations: the fake camera draws only the cube's top face and does not draw the arm,
so side faces and the arm blocking the view are not tested in mock.

### Earlier results (wrist camera, UR3 geometry)

| Test (mock) | Success | Grasp error x / y / z (mean) |
|---|---|---|
| Nominal, 5 trials | 5/5 | 2.0 / −2.0 / 2.0 mm |
| Occlusion during approach | 2/2 | ≈ 2 / −2 / 1.8 mm |
| Joint bump during approach | 3/3 | ≈ 2 / −2 / 2.2 mm |
| 3× depth noise | 3/3 | ≈ 1.9 / −2.0 / 2.5 mm, ~40–45 NIS rejects per trial |

The consistent ~2 mm x/y error is the 2 mm hand-eye error built into the mock
(`MOCK_HANDEYE_ERROR`). The filter's own std is only ~0.4 mm, so it is **overconfident**:
it can't see a calibration bias. That is a real effect you should expect on hardware too,
and it is a good example of tracing a failure to a specific cause (Distinction band).

---

## 4. Testing without the robot

| Level | What it tests | `config.py` switches |
|---|---|---|
| 1. Mock | All your logic: detection, filter, observer, controller, state machine | `USE_MOCK_ROBOT=True`, `USE_MOCK_CAMERA=True` |
| 2. URSim + fake camera | The real `ur_rtde` connection, real-time loop, UR's own motion limits | `USE_MOCK_ROBOT=False`, `USE_MOCK_CAMERA=True`, `ROBOT_IP="127.0.0.1"` |
| 3. Real RealSense, no arm | Camera, colour tuning, depth noise (if you can borrow one) | `USE_MOCK_CAMERA=False` |

**URSim** is Universal Robots' official simulator. It runs the same controller software
as the real robot, so `ur_rtde` can't tell the difference. With Docker installed:

```bash
docker run --rm -it -p 6080:6080 -p 29999:29999 -p 30001-30004:30001-30004 universalrobots/ursim_e-series
```

Open http://localhost:6080/vnc.html, turn the robot on and release the brakes, and set it to
Remote Control. If it is an e-Series image, use `ROBOT_MODEL = "UR3e"`, or pick UR3 in the
installation settings. Then run `python main_pick.py --trials 3`. In this mode the trials run in real
time (about 30 s each) and the camera image is faked from the simulated arm's pose.

## 5. Bringing it up on the real robot (do these in order)

**Lab-specific setup** (from the lab repo README): laptop Ethernet set to IP `192.168.0.100`,
mask `255.255.255.0`, gateway `192.168.0.1` (Wi-Fi unchanged); robot IP is printed under the
big number next to the arm, check with `ping`; on the pendant: Installation → Tool/IO →
OnRobot selected, ON → START to release brakes, then select **Remote**.

**Fixed camera calibration:** print a small checkerboard (e.g. 6 x 5 squares of 20 mm =
5 x 4 inner corners), glue it to stiff card and tape it rigidly to the gripper. Then
`python calibrate_handeye.py --fixed --cols 5 --rows 4 --square 0.020` (measure the
squares with calipers). Move the arm in freedrive to 15-20 poses where the camera sees the
board, **tilting and rotating the wrist a lot**. Expect a spread of a few mm.
`python calibrate_handeye.py --selftest --fixed` checks the maths without hardware.

**Keep the arm out of the camera's view during SEARCH:** check that at `HOME_Q` the arm does
not block the camera's line of sight to the table. If it does, change `HOME_Q`.

> ⚠️ Always have a team member with a hand on the e-stop. Start with
> `--speed-scale 0.3`. Clear the workspace. Never stand inside the robot's reach while it runs.

**At the induction, find out:**
1. Robot model (UR3 or UR3e) and IP address → `ROBOT_MODEL`, `ROBOT_IP`
2. Is it controlled by ur_rtde, or only through ROS? (If ROS only, write a `ROS_Robot`
   class with the same methods as `RTDE_Robot` and switch to it in `hardware.py`.)
3. Camera model (D435 / D405 / other) and how it's mounted
4. Gripper model and whether the Robotiq URCap is installed (port 63352)
5. Table height in the robot base frame → `TABLE_Z`
6. Whether `HOME_Q` is a safe pose on that table

**Bring-up steps:**

| # | Step | Command | Switches in `config.py` |
|---|---|---|---|
| 1 | Camera only: check images and tune colour | `python hsv_tuner.py` | — |
| 2 | Intrinsics check | `python calibrate_intrinsics.py --square 0.025` | — |
| 3 | Connect arm, check it reads joints | `python -c "from robot import RTDE_Robot; r=RTDE_Robot('IP'); print(r.get_q())"` | — |
| 4 | Joint lag τ (only wrist 3 moves) | `python identify_joint_lag.py` | `USE_MOCK_ROBOT=False` |
| 5 | Camera calibration (fixed stand) | `python calibrate_handeye.py --fixed --cols 5 --rows 4 --square 0.020` | — |
| 6 | Depth noise | `python measure_depth_noise.py` | both real |
| 6b | Gripper only | `python -c "from gripper import OnRobot_Gripper as G; from robot import RTDE_Robot as R; import config as C; r=R(C.ROBOT_IP); g=G(C.ROBOT_IP); g.open(); g.close(); print(g.object_detected()); r.close()"` | — |
| 7 | Dry pick, no gripper, high above table | raise `TABLE_Z` by 5 cm temporarily, `--speed-scale 0.3` | `GRIPPER_TYPE="mock"` |
| 8 | Full pick | `python main_pick.py --trials 10 --show --record` | all real |

Measure the checkerboard square size with calipers; an error there scales every
calibration result.

---

## 6. Safety features in the code

- Speed limits on every joint, tool linear and angular speed, scaled by `--speed-scale` (max 1.0)
- Abort if the fingertips go below `TOOL_Z_MIN`, or the target is outside the workspace box
- Abort on protective stop / e-stop, phase timeout, a control loop that freezes (>0.3 s) or is slow 5 times in a row
- Ctrl-C or `q` in the live window always stops the arm before exiting
- The reference can't run more than 0.05 rad ahead of the robot (no wind-up)
- ur_rtde's speedJ stops the arm by itself if commands stop arriving

---

## 7. Sim vs hardware comparison

`trial_XXX.csv` has the same columns in mock and on hardware, so run the same analysis on
both (and on the MuJoCo sim logs, if you match the column names). On hardware there is no
ground truth: use grasp success rate, the locked-estimate std, NIS statistics, observer vs
measured q̇, and (optionally) a ruler measurement of where the object really was.

## 8. Acknowledgements

Structure adapted from the 41014 tutorials. Uses ur_rtde (SDU Robotics), Intel
librealsense, OpenCV and SciPy. Generated with AI assistance (declare on the project page).
