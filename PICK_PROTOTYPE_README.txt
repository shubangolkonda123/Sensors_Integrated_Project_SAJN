INTEGRATED TRACK B PROTOTYPE - READ BEFORE RUNNING

Status
------
The Python files were syntax-checked only. They were NOT run against your XML,
meshes, MuJoCo/OpenGL installation, observation pose or gripper. Successful
execution, stable control and successful grasping have NOT been demonstrated.
Do not describe this as completed assessment work or hardware-ready software.
Simulation only. Never connect it to a physical robot.

Required existing model
-----------------------
Your current ur3e.xml plus assets, with the exact names used in the conversation:
- six arm joints and six affine position actuators
- two sliding finger joints and left_gripper/right_gripper position actuators
- grasp_site, wrist_camera, tabletop, target_block and target_block_joint
- home keyframe containing all 15 qpos and 8 actuator controls
- saved home pose looking at the table
The script does not create or modify those XML/assets on disk.
The table is assumed horizontal near z=0.225 m in the robot-base/world frame.
The target is assumed a red, axis-aligned 4 cm cube. Initial target rotation is
not randomised. A fixed downward gripper orientation is assumed.

Installation
------------
Use your existing activated venv. Install the accompanying requirements file.
Do not install several OpenCV distributions into the same environment.
If cv2 already works, avoid reinstalling competing OpenCV packages.

Example commands (put the Python files beside your current start.py)
-------------------------------------------------------------------
python integrated_pick_prototype.py --model universal_robots_ur3e/ur3e.xml
python plot_pick_results.py results/pick_trial

Randomised trial
----------------
python integrated_pick_prototype.py --model universal_robots_ur3e/ur3e.xml --randomize --seed 2 --results results/trial_2

Varied sensor conditions (metres and probability)
------------------------------------------------
python integrated_pick_prototype.py --model universal_robots_ur3e/ur3e.xml --randomize --seed 3 --depth-noise 0.002 --dropout 0.05 --results results/trial_3
Use distinct output directories: existing CSV/JSON files are overwritten.

Architecture
------------
Sensors: sampled noisy joint positions, RGB and optical-axis metric depth.
Perception: depth back-projection, estimated camera transform, workspace crop,
RANSAC support plane, colour + height target selection. No live true object pose
is used by perception. Camera transform comes from model kinematics evaluated at
estimated joint angles, not the live plant camera pose.
Target: robust visible extents in XY plus a known-cube/plane grasp-height rule.
This is not general 6D object-pose or unseen-shape reconstruction.
Observer: 12-state joint-position/joint-velocity Kalman filter.
Plant model: frozen-configuration inertia and existing affine position-servo
stiffness/damping, discretised at 100 Hz. Fingers/couplings/model changes and
actuator saturation are approximations. Model mismatch must be evaluated.
Control: discrete LQR outer joint-position-servo feedback, clipped and rate-limited.
The model-based bias force is computed using estimated q/qdot in separate model
data. It is applied as an additional simulated arm input, not read from true
plant bias forces. This interface is NOT a demonstrated hardware capability.
Motion: model-based numerical IK (SciPy) and Peter Corke Toolbox joint trajectories.
No ETS is used. The stock Toolbox UR3 DH model is NOT used: its correspondence
with the XML was never established. A DH/robot.tool implementation must replace
this IK backend after independent pose validation if that is a project requirement.
Visual approach: object measurements update PREGRASP and early DESCEND references.
Final descent/closure may use a recent frozen reference for up to five seconds.
This is a constrained PBVS-style prototype, not a fully analysed visual servo.
Task: OBSERVE -> PREGRASP -> DESCEND -> CLOSE -> LIFT -> HOLD; failures -> ABORT.
Collision checks: sampled candidate configurations against table only. NOT a
continuous collision planner, self-collision guarantee or hardware safety system.
ABORT holds estimated current position. Close the viewer/Q to terminate.

Evaluation
----------
Only Evaluation reads true joint velocity and object height for scoring/logging.
Sensor adapter reads true joint positions to synthesise encoder measurements.
Initialisation/trial setup may set simulator ground truth.
CSV contains true/estimated joints, references, controls, uncertainty, NEES and
object height. JSON reports RMSE and whether a >6 cm lift was held for the last
100 HOLD samples. This is a limited task metric, not a force/contact-based grasp
verification sensor available to the robot.
NPZ contains model and design matrices. Plot script shows estimate uncertainty
alongside ground truth and references. Numerical rank checks need interpretation
and conditioning analysis; a printed rank is not a complete modelling report.

What you must still do
---------------------
1. Run and debug the actual model; verify every frame, joint and actuator mapping.
2. Validate camera intrinsics/extrinsics/depth with calibration evidence.
3. Derive/justify the state-space model, sampling, observer covariances and gains.
4. Evaluate tracking, estimation consistency, control effort and failure modes.
5. Repeat seeded trials and disturbances; report both successes and failures.
6. Establish actual hardware command/measurement interfaces with teaching staff.
7. Demonstrate simulation to staff and attend required induction before hardware.
8. Produce your own reproducible GitHub page, five-minute demonstration, team
   contribution statement, source acknowledgements and AI-use declaration.
9. Prepare every member for the viva. Do not submit code you cannot explain.
No performance results, video, calibration report or hardware results are supplied
or fabricated by these files.
