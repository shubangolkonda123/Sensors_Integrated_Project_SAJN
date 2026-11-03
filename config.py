"""
config.py - EVERY tunable number for the hardware pipeline lives here.

Change values here, never inside the other files. Each value has a comment saying
where it comes from, so you can defend it in the report and the viva
("Where do the values in Q and R come from?").

Things marked  >>> FILL IN AT INDUCTION <<<  depend on the lab hardware.
"""
import numpy as np
from transforms import make_T

# =====================================================================
# 1. HARDWARE SWITCHES
#    Start with everything on mock. Turn on one real device at a time.
# =====================================================================
USE_MOCK_ROBOT = True        # False = real UR3 through ur_rtde
USE_MOCK_CAMERA = True       # False = real Intel RealSense (needs the real robot too)
GRIPPER_TYPE = "mock"        # "mock" | "onrobot" (lab RG2) | "robotiq"

# UTS Mechatronics Lab setup (from the lab repo UTS-RI/UR3_RG2_rosbridge):
#   UR3e, ur_rtde over an Ethernet cable, laptop Ethernet IP 192.168.0.100 / 255.255.255.0,
#   OnRobot RG2 gripper through the onRobot library, pendant set to Remote.
ROBOT_IP = "192.168.0.250"   # >>> FILL IN: the IP printed under the big number next to the arm <<<
ROBOT_MODEL = "UR3e"         # the lab arm is a UR3e

# Where the camera is:
#   "fixed" = on a stand next to the robot, not moving (eye-to-hand)  <- our setup
#   "wrist" = mounted on the robot's wrist (eye-in-hand)
CAMERA_MOUNT = "fixed"

# =====================================================================
# 2. TIMING
# =====================================================================
CONTROL_HZ = 50              # control loop rate. ur_rtde speedJ is happy at 50-125 Hz
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30
DT = 1.0 / CONTROL_HZ

# =====================================================================
# 3. SAFETY LIMITS  (start slow; raise SPEED_SCALE only after it works)
# =====================================================================
SPEED_SCALE = 0.5            # multiplies every speed limit below. Use 0.3 for first real run
MAX_JOINT_SPEED = 0.6        # rad/s, hard clip on every joint velocity command
MAX_LIN_SPEED = 0.08         # m/s, tool linear speed limit
MAX_ROT_SPEED = 0.4          # rad/s, tool angular speed limit
SPEEDJ_ACC = 1.0             # rad/s^2, acceleration passed to speedJ
STOP_DECEL = 2.0             # rad/s^2, deceleration used when stopping

TABLE_Z = 0.0                # base-frame height of the table surface [m]  >>> MEASURE <<<
TOOL_Z_MIN = TABLE_Z + 0.005 # abort if the fingertips ever go below this

# Box in the base frame where a target is allowed to be (outside = abort, don't move)
WORKSPACE_MIN = np.array([-0.45, -0.40, TABLE_Z - 0.02])
WORKSPACE_MAX = np.array([0.45, 0.40, TABLE_Z + 0.20])
WORKSPACE_MIN_RADIUS = 0.15  # too close to the base column = unreachable / collision

PHASE_TIMEOUT = 25.0         # s, any phase that takes longer than this aborts
MAX_LOOP_OVERRUN = 0.05      # s, a loop this much longer than DT counts as "slow"
MAX_CONSEC_OVERRUNS = 5      # abort after this many slow loops in a row
MAX_LOOP_FREEZE = 0.3        # s, abort immediately if a single loop takes this long

# =====================================================================
# 4. POSES AND GEOMETRY
# =====================================================================
# Observation pose: camera looking straight down at the table, ~0.38 m above it.
# Flange ends up at about (-0.23, -0.11, 0.41) in the DH base frame.
# >>> CHECK ON THE PENDANT that this pose is clear of the lab table/fixtures <<<
HOME_Q = np.array([0.0, -1.75, 1.25, -1.071, -np.pi / 2, 0.0])
HOME_SPEED, HOME_ACC = 0.3, 0.5          # moveJ speed (rad/s) / acceleration (rad/s^2)

# Flange -> centre between the RG2 fingertips. Roughly 0.20 m for an RG2.
# >>> MEASURE with a ruler from the flange face to the middle of the finger pads <<<
T_FLANGE_TOOL = make_T(np.eye(3), [0.0, 0.0, 0.20])

# OnRobot RG2 commands (same values as the lab's example 15)
RG2_OPEN_WIDTH_MM = 80.0        # open wide enough for the 40 mm cube with margin
RG2_CLOSED_WIDTH_MM = 10.0      # target when closing; it stops early on the cube
RG2_FORCE_N = 20.0              # gripping force
RG2_HELD_MIN_WIDTH_MM = 20.0    # stopped wider than this after closing = something is held

# Flange -> camera optical frame. Used ONLY if calib/handeye.npz doesn't exist yet.
# Rough guess: camera 7 cm off the flange axis, looking along the flange z-axis.
T_FLANGE_CAM_DEFAULT = make_T(np.eye(3), [0.0, -0.07, 0.03])
HANDEYE_FILE = "calib/handeye.npz"


def look_at(Cam_Pos, Target):
    """Camera pose (optical frame: x right, y down, z forward) looking from Cam_Pos at Target."""
    Z = np.asarray(Target, float) - np.asarray(Cam_Pos, float)
    Z /= np.linalg.norm(Z)
    X = np.cross(Z, [0.0, 0.0, 1.0])     # horizontal, so the image is not rolled
    X /= np.linalg.norm(X)
    Y = np.cross(Z, X)
    return make_T(np.column_stack([X, Y, Z]), Cam_Pos)


# Base -> camera optical frame for the FIXED camera. Used ONLY until
# calib/eye_to_hand.npz exists (run calibrate_handeye.py --fixed).
# Rough guess: stand ~0.6 m to the side of the work area, camera ~0.5 m up, looking down at it.
T_BASE_CAM_DEFAULT = look_at([-0.25, -0.72, 0.50], [-0.24, -0.12, 0.0])
EYE_TO_HAND_FILE = "calib/eye_to_hand.npz"
INTRINSICS_FILE = "calib/intrinsics.npz"
USE_CUSTOM_INTRINSICS = False   # True = use your checkerboard intrinsics instead of factory ones

# Pick geometry, all relative to the estimated TOP-CENTRE of the object
OBJECT_HEIGHT = 0.04            # m  >>> MEASURE your object <<<
APPROACH_HEIGHT = 0.12          # pre-grasp: this far above the object top
GRASP_Z_OFFSET = -OBJECT_HEIGHT / 2   # grasp at mid-height of the object
LIFT_HEIGHT = 0.10              # how high to lift after grasping
DESCEND_SPEED_FACTOR = 0.4      # slow down for the final descent

POS_TOL = 0.003                 # m, "goal reached" position tolerance
ROT_TOL = 0.03                  # rad, "goal reached" orientation tolerance
SETTLE_CYCLES = 10              # must be inside tolerance this many loops in a row

# =====================================================================
# 5. SENSING  (Part B)
# =====================================================================
# HSV colour ranges for the target object (OpenCV H is 0-179). Red wraps around 0,
# so it needs two ranges.  >>> RE-TUNE under lab lighting with hsv_tuner.py <<<
HSV_RANGES = [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))]
MIN_BLOB_AREA = 150             # px, smaller blobs are noise
MIN_VALID_DEPTH_FRAC = 0.3      # need at least 30% of blob pixels with valid depth
DEPTH_MIN, DEPTH_MAX = 0.10, 1.0   # m, depth outside this range is treated as invalid

# Depth noise model for a stereo camera: sigma_z grows with distance squared.
# sigma_z = DEPTH_NOISE_K * Z^2 + DEPTH_NOISE_MIN  (Intel quotes ~2% at 2 m for D435)
# >>> Verify: hold the camera still over the object, log 300 frames, take std <<<
DEPTH_NOISE_K = 0.005           # 1/m
DEPTH_NOISE_MIN = 0.001         # m
PIXEL_NOISE = 1.5               # px, std of the blob centroid
VISION_MIN_DEPTH = 0.20         # m, below this the camera is too close; stop trusting it
TOP_FACE_BAND = 0.008           # m, fixed camera: points within this of the highest point = top face

# =====================================================================
# 6. ESTIMATION  (Part C)
# =====================================================================
# Target Kalman filter: the object is static, so a constant-position model.
# Q is small but non-zero to absorb kinematic/hand-eye error as the arm moves.
TARGET_PROCESS_STD = 0.0002     # m per control step -> Q = std^2 * I  (~1.4 mm/sqrt(s) random walk)
NIS_GATE = 7.815                # chi-square 95% for 3 DOF: reject measurements above this
MAX_CONSEC_REJECTS = 15         # this many rejects in a row = object moved -> re-initialise
SEARCH_MIN_UPDATES = 15         # measurements needed before we trust the estimate
SEARCH_MAX_STD = 0.004          # m, every axis std must be below this to start moving
SEARCH_TIMEOUT = 10.0           # s, give up if no good estimate by then

# Joint model: speedJ is a velocity command; the UR's internal loop makes the
# actual velocity follow it like a first-order lag with time constant tau.
# >>> IDENTIFY with identify_joint_lag.py on the real arm <<<
JOINT_LAG_TAU = 0.06            # s

# Luenberger observer poles (continuous time, rad/s). ~3-5x faster than the controller.
OBSERVER_POLES = [-40.0, -45.0]

# =====================================================================
# 7. CONTROL  (Part D)
# =====================================================================
# Per-joint LQR on x = [q, qd], u = commanded joint speed.
LQR_Q = np.diag([400.0, 1.0])   # penalise position error much more than velocity error
LQR_R = np.array([[1.0]])       # penalise control effort
REF_LEASH = 0.02               # rad, reference can't run further than this ahead of the robot.
                                # Was 0.05: with the UR3e + RG2 that let a 3 deg bump leave a
                                # 0.05 rad gap and the PBVS loop oscillated (timeout). 0.02 fixed it.

# PBVS outer loop: tool velocity = gain * pose error (then clipped to speed limits)
PBVS_GAIN_POS = 2.0             # 1/s
PBVS_GAIN_ROT = 2.0             # 1/s
DLS_DAMPING = 0.02              # damped least squares near singularities

# =====================================================================
# 8. MOCK SETTINGS (only used when USE_MOCK_* is True)
# =====================================================================
MOCK_OBJECT_RADIUS = 0.02                        # m, top face radius of the fake object
MOCK_OBJECT_REGION_MIN = np.array([-0.32, -0.20])  # x, y range for random object placement
MOCK_OBJECT_REGION_MAX = np.array([-0.16, -0.04])
MOCK_HANDEYE_ERROR = 0.002      # m, the "true" camera pose differs from calibration by this
MOCK_TABLE_COLOUR = (90, 90, 90)
MOCK_OBJECT_COLOUR = (20, 20, 200)   # BGR red
MOCK_INTRINSICS = dict(Fx=615.0, Fy=615.0, Cx=320.0, Cy=240.0)  # typical D435 colour @640x480
