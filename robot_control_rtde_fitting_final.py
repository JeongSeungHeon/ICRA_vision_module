"""Dual-camera grasp-target follow script adapted for UR5 RTDE control."""

import argparse
import queue
import sys
import time
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections import deque

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

import cv2 as cv
import numpy as np
import yaml

from calibration.extrinsics import load_transform_chain
from object_pt_extraction.segmentation_engine import (
    SegmentationEngine,
    parse_prompt_classes,
)
from perception.fusion import PerceptionFusion
#from perception.fill_level_estimator import FillLevelEstimator
from perception.grasp_target import GraspTargetPlanner
from perception.hand_relative_fallback import HandRelativeFallbackTracker
from perception.hand_selector import HandSelector
from perception.hand_worker import HandWorkerCam0, HandWorkerCam1
from perception.object_merger import ObjectMerger
from perception.object_worker import ObjectWorkerCam0, ObjectWorkerCam1
from perception.silhouette_constraint import SilhouetteObservation
from perception.shape_fitting_tracker_v2 import ShapeFittingTracker
from perception.target_predictor import TargetPredictor
from robot.rtde_controller import RtdeController
from system.dual_sensor_hub import DualSensorHub
from system.shared_state import (
    GRIPPER_CLOSE,
    GRIPPER_HOLD,
    GRIPPER_OPEN,
    ROBOT_CMD_HOLD,
    ROBOT_CMD_MOVE_TO_POSITION,
    ROBOT_CMD_SERVO_TO_POSITION,
    ROBOT_CMD_STOP,
    RobotCommandState,
)
from video_record import HandoverVideoRecorderService
from utils.debug_3d_recorder import Debug3DRecorder
from utils.handover_metadata import HandoverMetadataRecorder
from utils.runtime_profiler import RuntimeProfiler


# =========================
# Constants
# =========================
DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
DEFAULT_WORKSPACE_MM = {
    "x": (-600.0, 800.0),
    "y": (-200.0, 800.0),
    "z": (0.0, 800.0),
}

# object motion trigger
REFERENCE_LOCK_COUNT = 8
MOTION_TRIGGER_MM = 40.0
MOTION_TRIGGER_Z_MM = 40.0

# control loop
DEFAULT_CONTROL_HZ = 30.0
MAX_XY_SPEED_MM_S = 250.0 # 80
MAX_Z_SPEED_MM_S = 250.0  # 80

# EEF target offset from detected object center (robot base frame)
EEF_X_OFFSET_MM = -320.0
EEF_Y_OFFSET_MM = 0.0

# grasp / place behavior
HOVER_Z_OFFSET_MM = 0
DESCEND_EXTRA_MM = 0.0
BACKOFF_X_MM = 75.0
DEFAULT_POST_RELEASE_Z_OFFSET_MM = 0.0
HOME_PLACE_X_OFFSET_MM = 0.0
HOME_PLACE_Y_OFFSET_MM = -3.0
HOME_PLACE_MIN_Z_MM = 30.0
PRE_RELEASE_MIN_Z_EPSILON_MM = 1e-3
GRASP_POINT_Y_OFFSET_MM = 10.0
PLACE_Z_GRASP_BUFFER_FRAMES = 5
PLACE_Z_MIN_VALID_SAMPLES = 3
GRIPPER_FORCE_STOP_DELTA_N = 100000
GRIPPER_FORCE_STOP_MIN_ELAPSED_S = 0.12
DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD = 200
DEFAULT_GRIPPER_POSITION_STALL_ENABLED = True
DEFAULT_GRIPPER_POSITION_STALL_STABLE_READS = 3
DEFAULT_GRIPPER_POSITION_STALL_TOLERANCE = 1
DEFAULT_GRIPPER_POSITION_STALL_MIN_ELAPSED_S = 0.12
RELEASE_PARAMETER_MM = 80.0
DEFAULT_TACTILE_PORT = "/dev/ttyACM0"
DEFAULT_TACTILE_NUM_MAGS = 5
DEFAULT_TACTILE_BASELINE_SAMPLES = 5
DEFAULT_TACTILE_STARTUP_DELAY_S = 1.0
DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD = 30.0
DEFAULT_TACTILE_EXTRA_GRASP_POS = 10
DEFAULT_TACTILE_RELEASE_DELTA_THRESHOLD = 50.0
DEFAULT_TACTILE_RELEASE_REF_DELAY_S = 0.3
DEFAULT_TACTILE_RELEASE_TIMEOUT_S = 1.0
DEFAULT_TACTILE_RELEASE_DESCENT_STEP_MM = 2.0
DEFAULT_TACTILE_RELEASE_DESCENT_POLL_DT_S = 0.03
DEFAULT_TACTILE_AUTO_BASELINE_RESET_AFTER_OPEN_S = 1.5
DEFAULT_TACTILE_RELEASE_STOP_TIMING_DEBUG = False
DEFAULT_TACTILE_RELEASE_STOP_SPEED_THRESHOLD_MPS = 0.002
DEFAULT_TACTILE_RELEASE_STOP_MONITOR_TIMEOUT_S = 1.0
DEFAULT_TACTILE_RELEASE_STOP_MONITOR_POLL_DT_S = 0.005
DEFAULT_POST_BACKOFF_STOP_CHECK_ENABLED = True
DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS = 0.002
DEFAULT_POST_BACKOFF_STOP_TIMEOUT_S = 1.0
DEFAULT_POST_BACKOFF_STOP_POLL_DT_S = 0.01
DEFAULT_POST_BACKOFF_STOP_REQUIRE_CONFIRMED = False
HOME_JOINTS_DEG = [0.0, -135.0, 135.0, 0.0, 90.0, 0.0]
HOME_JOINT_TOLERANCE_DEG = 1.0
HOME_JOINT_SPEED_RAD_S = 0.5
HOME_JOINT_ACCELERATION_RAD_S2 = 0.5
VIDEO_RECORDER_SERIAL = "335522072904"
VIDEO_RECORDER_PORT = 5000

ROBOT_REQ_INIT_ROBOT = "INIT_ROBOT"
ROBOT_REQ_START_FOLLOW = "START_FOLLOW"
ROBOT_REQ_STOP_FOLLOW = "STOP_FOLLOW"
ROBOT_REQ_START_GRASP_PLACE = "START_GRASP_PLACE"
ROBOT_REQ_RESET_HOME = "RESET_HOME"
ROBOT_REQ_SAVE_AND_STOP = "SAVE_AND_STOP"
ROBOT_REQ_EMERGENCY_STOP = "EMERGENCY_STOP"
ROBOT_REQ_SHUTDOWN = "SHUTDOWN"

ROBOT_URGENT_REQUESTS = {
    ROBOT_REQ_RESET_HOME,
    ROBOT_REQ_SAVE_AND_STOP,
    ROBOT_REQ_EMERGENCY_STOP,
    ROBOT_REQ_SHUTDOWN,
}

ROBOT_STATE_IDLE = "IDLE"
ROBOT_STATE_INITIALIZING = "INITIALIZING"
ROBOT_STATE_FOLLOWING = "FOLLOWING"
ROBOT_STATE_GRASPING = "GRASPING"
ROBOT_STATE_RETURNING = "RETURNING"
ROBOT_STATE_PLACING = "PLACING"
ROBOT_STATE_RESETTING = "RESETTING"
ROBOT_STATE_DONE = "DONE"
ROBOT_STATE_ERROR = "ERROR"
ROBOT_STATE_STOPPING = "STOPPING"


@dataclass
class RobotRequest:
    type: str
    payload: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    request_id: int = 0


@dataclass
class RobotActionContext:
    fitted_points_base: object = None
    grasp_point_base: object = None
    object_label: object = None
    template_axes_base: object = None


@dataclass
class RobotStatus:
    state: str = ROBOT_STATE_IDLE
    last_error: object = None
    is_connected: bool = False
    using_mock: bool = False
    active_request: object = None
    active_request_id: int = 0
    last_robot_pose: object = None
    last_command_type: object = None
    grasp_ok: object = None
    task_done_epoch: int = 0
    reset_done_epoch: int = 0
    task_ready_epoch: int = 0


def parse_args():
    """Parse CLI switches for perception, robot follow, debugging, and profiling."""
    parser = argparse.ArgumentParser(
        description="Run dual-camera perception, build a grasp target from hand pose + merged object cloud, and follow it with UR5 RTDE."
    )
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="Model name or local weights path.")
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=None,
        help="Text prompt classes for YOLOE, e.g. --prompt person bus or --prompt person,bus",
    )
    parser.add_argument("--serial", default=None, help="Legacy single-camera option. Ignored in dual-camera grasp-target mode.")
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RealSense stream FPS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. cpu, 0, 0,1.")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional class id filter.")
    parser.add_argument(
        "--select-mode",
        choices=["all_instances", "highest_score", "class_filter"],
        default="all_instances",
        help="Instance selection policy for downstream processing.",
    )
    parser.add_argument(
        "--select-class",
        nargs="*",
        default=None,
        help="Optional class-name filter used with selection.",
    )
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on supported devices.")
    #parser.add_argument("--show-depth", action="store_true", help="Show a second depth preview window.")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="Upper bound for depth visualization.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config used by the existing UR5 RTDE controller.",
    )

    # UR5 RTDE follow options
    parser.add_argument("--enable-follow", action="store_true", help="Enable UR5 RTDE follow mode.")
    parser.add_argument("--robot-ip", type=str, default="192.168.56.101", help="Optional UR5 IP override.")
    parser.add_argument("--min-valid-count", type=int, default=3, help="Min consecutive valid detections before follow.")
    parser.add_argument("--target-timeout-s", type=float, default=0.5, help="Stop following if target is stale.")
    parser.add_argument("--workspace-x", nargs=2, type=float, default=None, help="Workspace X limits in mm.")
    parser.add_argument("--workspace-y", nargs=2, type=float, default=None, help="Workspace Y limits in mm.")
    parser.add_argument("--workspace-z", nargs=2, type=float, default=None, help="Workspace Z limits in mm.")
    parser.add_argument("--move-to-base", action="store_true", help="Reserved for compatibility; HOME now uses joint targets.")
    parser.add_argument("--open-gripper", action="store_true", help="Open gripper during init.")
    parser.add_argument("--verbose-robot", action="store_true", help="Print detailed robot command logs.")
    parser.add_argument("--follow-z", dest="follow_z", action="store_true", help="Enable z-axis follow.")
    parser.add_argument("--no-follow-z", dest="follow_z", action="store_false", help="Disable z-axis follow.")
    parser.set_defaults(follow_z=None)
    parser.add_argument("--control-hz", type=float, default=None, help="Servo loop rate. Defaults to config or 30 Hz.")
    parser.add_argument("--position-tolerance-m", type=float, default=0.01, help="Move completion tolerance in meters.")
    parser.add_argument("--move-timeout-s", type=float, default=10.0, help="Timeout for blocking move steps.")
    parser.add_argument("--gripper-close-timeout-s", type=float, default=2.0, help="Timeout for force/current grasp verification.")
    parser.add_argument("--gripper-release-dwell-s", type=float, default=0.5, help="Dwell after opening gripper.")
    parser.add_argument(
        "--enable-pre-release-descend-before-open",
        dest="pre_release_descend_before_open",
        action="store_true",
        help="Descend in Z before opening the gripper at PLACE. Defaults to config.",
    )
    parser.add_argument(
        "--pre-release-descend-m",
        type=float,
        default=None,
        help="Optional pre-release descend distance in meters. Defaults to RELEASE_PARAMETER_MM.",
    )
    parser.set_defaults(pre_release_descend_before_open=None)
    parser.add_argument("--follow-handoff-timeout-s", type=float, default=5.0, help="Timeout to wait for the follow thread to release robot control before pregrasp.")
    parser.add_argument(
        "--enable-target-prediction",
        dest="enable_target_prediction",
        action="store_true",
        help="Use Kalman prediction to bridge short target dropouts during FOLLOW.",
    )
    parser.add_argument(
        "--disable-target-prediction",
        dest="enable_target_prediction",
        action="store_false",
        help="Disable target prediction and fall back to raw target only.",
    )
    parser.set_defaults(enable_target_prediction=True)
    parser.add_argument(
        "--prediction-max-horizon-s",
        type=float,
        default=0.25,
        help="Maximum dropout duration bridged by predicted targets.",
    )
    parser.add_argument(
        "--prediction-process-noise-mm-s2",
        type=float,
        default=800.0,
        help="Kalman process noise acceleration scale for xy prediction.",
    )
    parser.add_argument(
        "--prediction-measurement-noise-mm",
        type=float,
        default=25.0,
        help="Kalman measurement noise scale for xy updates.",
    )
    parser.add_argument(
        "--prediction-max-xy-speed-mm-s",
        type=float,
        default=200.0,
        help="Maximum xy prediction speed before velocity clipping.",
    )
    parser.add_argument(
        "--prediction-reinit-jump-mm",
        type=float,
        default=120.0,
        help="Reinitialize the Kalman state if a new measurement jumps too far from prediction.",
    )
    parser.add_argument(
        "--3d-debug",
        dest="debug_3d",
        action="store_true",
        help="Enable in-memory 3D debug recording. Press 'd' to save and reset.",
    )
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="With --3d-debug, include raw color/depth frames in saved 3D debug recordings.",
    )
    parser.add_argument(
        "--debug-3d-dir",
        type=str,
        default="output/debug_3d",
        help="Directory where saved 3D debug recordings are written.",
    )
    parser.add_argument(
        "--debug-3d-max-object-points",
        type=int,
        default=8000,
        help="Maximum raw object cloud points stored per debug frame.",
    )
    parser.add_argument(
        "--debug-3d-max-template-points",
        type=int,
        default=8000,
        help="Maximum fitted template cloud points stored per debug frame.",
    )
    parser.add_argument(
        "--no-debug-3d-template-axes",
        dest="debug_3d_template_axes",
        action="store_false",
        help="Do not store fitted template local x/y/z axes in 3D debug recordings.",
    )
    parser.set_defaults(debug_3d_template_axes=True)
    parser.add_argument(
        "--disable-debug-3d-recording",
        action="store_true",
        help="Backward-compatible alias that keeps 3D debug recording disabled.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Enable task video recording. Press 's' to finish/save using the configured recorder flow.",
    )
    parser.add_argument(
        "--profile-runtime",
        action="store_true",
        help="Collect runtime CPU/RAM/GPU and per-stage timing logs. Press 's' to save the current session.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="output/runtime_profile",
        help="Directory where runtime profiling CSV/JSON files are written.",
    )
    parser.add_argument(
        "--profile-sample-interval-s",
        type=float,
        default=1.0,
        help="Resource sampling interval for runtime profiling.",
    )
    parser.add_argument(
        "--profile-print-every-s",
        type=float,
        default=5.0,
        help="Print a short runtime profiling summary every N seconds. Use 0 to disable.",
    )
    return parser.parse_args()


def load_yaml_config(path):
    """Load the YAML runtime config used by cameras, perception, safety, and RTDE."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_video_recorder_web_ui_enabled(config):
    """Return whether task video recording should use the web UI finalize flow."""
    video_cfg = dict((config or {}).get("video_recording", {}) or {})
    web_ui_cfg = dict(video_cfg.get("web_ui", {}) or {})
    return bool(web_ui_cfg.get("enabled", True))


def get_video_recorder_tactile_logging_config(config):
    """Return tactile sidecar logging switches for the task video recorder."""
    video_cfg = dict((config or {}).get("video_recording", {}) or {})
    tactile_cfg = dict(video_cfg.get("tactile_logging", {}) or {})
    return {
        "enabled": bool(tactile_cfg.get("enabled", True)),
        "csv_enabled": bool(tactile_cfg.get("csv_enabled", True)),
        "rerun_enabled": bool(tactile_cfg.get("rerun_enabled", True)),
        "rerun_live": bool(tactile_cfg.get("rerun_live", False)),
    }


def apply_config_defaults(args, config):
    """Fill CLI defaults from config while keeping explicit command-line values."""
    robot_cfg = config.get("robot", {})
    live_cfg = robot_cfg.get("live_follow", {})
    return_sequence_cfg = robot_cfg.get("return_sequence", {})
    tactile_cfg = robot_cfg.get("tactile", {})
    rtde_cfg = robot_cfg.get("rtde", {})
    safety_cfg = config.get("safety", {})
    workspace_cfg = safety_cfg.get("workspace_bounds_m", {})

    if args.robot_ip is None:
        args.robot_ip = rtde_cfg.get("robot_ip")
    if args.control_hz is None:
        args.control_hz = float(live_cfg.get("control_hz", DEFAULT_CONTROL_HZ))
    if args.follow_z is None:
        args.follow_z = bool(live_cfg.get("follow_z", False))
    args.post_release_z_offset_mm = (
        float(return_sequence_cfg.get("post_release_z_offset_m", DEFAULT_POST_RELEASE_Z_OFFSET_MM / 1000.0)) * 1000.0
    )
    if args.pre_release_descend_before_open is None:
        args.pre_release_descend_before_open = bool(return_sequence_cfg.get("pre_release_descend_enabled", False))
    pre_release_descend_m = (
        args.pre_release_descend_m
        if args.pre_release_descend_m is not None
        else return_sequence_cfg.get("pre_release_descend_m", RELEASE_PARAMETER_MM / 1000.0)
    )
    args.pre_release_descend_mm = max(0.0, float(pre_release_descend_m) * 1000.0)
    args.post_backoff_stop_check_enabled = bool(
        return_sequence_cfg.get("post_backoff_stop_check_enabled", DEFAULT_POST_BACKOFF_STOP_CHECK_ENABLED)
    )
    args.post_backoff_stop_speed_threshold_mps = max(
        0.0,
        float(
            return_sequence_cfg.get(
                "post_backoff_stop_speed_threshold_mps",
                DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS,
            )
        ),
    )
    args.post_backoff_stop_timeout_s = max(
        0.0,
        float(return_sequence_cfg.get("post_backoff_stop_timeout_s", DEFAULT_POST_BACKOFF_STOP_TIMEOUT_S)),
    )
    args.post_backoff_stop_poll_dt_s = max(
        0.0,
        float(return_sequence_cfg.get("post_backoff_stop_poll_dt_s", DEFAULT_POST_BACKOFF_STOP_POLL_DT_S)),
    )
    args.post_backoff_stop_require_confirmed = bool(
        return_sequence_cfg.get(
            "post_backoff_stop_require_confirmed",
            DEFAULT_POST_BACKOFF_STOP_REQUIRE_CONFIRMED,
        )
    )

    for axis_name in ("x", "y", "z"):
        arg_name = f"workspace_{axis_name}"
        current_value = getattr(args, arg_name)
        if current_value is not None:
            continue
        config_bounds = workspace_cfg.get(axis_name)
        if config_bounds is not None and len(config_bounds) == 2:
            mm_bounds = [float(config_bounds[0]) * 1000.0, float(config_bounds[1]) * 1000.0]
            setattr(args, arg_name, mm_bounds)
        else:
            setattr(args, arg_name, list(DEFAULT_WORKSPACE_MM[axis_name]))

    args.tactile_enabled = bool(tactile_cfg.get("enabled", False))
    args.tactile_port = str(tactile_cfg.get("port", DEFAULT_TACTILE_PORT))
    args.tactile_num_mags = max(1, int(tactile_cfg.get("num_mags", DEFAULT_TACTILE_NUM_MAGS)))
    args.tactile_baseline_samples = max(
        1,
        int(tactile_cfg.get("baseline_samples", DEFAULT_TACTILE_BASELINE_SAMPLES)),
    )
    args.tactile_startup_delay_s = max(
        0.0,
        float(tactile_cfg.get("startup_delay_s", DEFAULT_TACTILE_STARTUP_DELAY_S)),
    )
    args.tactile_contact_norm_threshold = max(
        0.0,
        float(tactile_cfg.get("contact_norm_threshold", DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD)),
    )
    args.tactile_extra_grasp_pos = max(
        0,
        int(tactile_cfg.get("extra_grasp_pos", DEFAULT_TACTILE_EXTRA_GRASP_POS)),
    )
    args.tactile_release_delta_threshold = max(
        0.0,
        float(tactile_cfg.get("release_delta_threshold", DEFAULT_TACTILE_RELEASE_DELTA_THRESHOLD)),
    )
    args.tactile_release_ref_delay_s = max(
        0.0,
        float(tactile_cfg.get("release_ref_delay_s", DEFAULT_TACTILE_RELEASE_REF_DELAY_S)),
    )
    # args.tactile_release_timeout_s = max(
    #     0.0,
    #     float(tactile_cfg.get("release_timeout_s", DEFAULT_TACTILE_RELEASE_TIMEOUT_S)),
    # )
    args.tactile_release_descent_min_z_mm = max(
        HOME_PLACE_MIN_Z_MM,
        float(tactile_cfg.get("release_descent_min_z_mm", HOME_PLACE_MIN_Z_MM)),
    )
    args.tactile_release_descent_step_mm = max(
        0.1,
        float(tactile_cfg.get("release_descent_step_mm", DEFAULT_TACTILE_RELEASE_DESCENT_STEP_MM)),
    )
    args.tactile_release_descent_poll_dt_s = max(
        0.0,
        float(tactile_cfg.get("release_descent_poll_dt_s", DEFAULT_TACTILE_RELEASE_DESCENT_POLL_DT_S)),
    )
    args.tactile_auto_baseline_reset_after_open_s = max(
        0.0,
        float(
            tactile_cfg.get(
                "auto_baseline_reset_after_open_s",
                DEFAULT_TACTILE_AUTO_BASELINE_RESET_AFTER_OPEN_S,
            )
        ),
    )
    args.tactile_release_stop_timing_debug = bool(
        tactile_cfg.get("release_stop_timing_debug", DEFAULT_TACTILE_RELEASE_STOP_TIMING_DEBUG)
    )
    args.tactile_release_stop_speed_threshold_mps = max(
        0.0,
        float(
            tactile_cfg.get(
                "release_stop_speed_threshold_mps",
                DEFAULT_TACTILE_RELEASE_STOP_SPEED_THRESHOLD_MPS,
            )
        ),
    )
    args.tactile_release_stop_monitor_timeout_s = max(
        0.0,
        float(
            tactile_cfg.get(
                "release_stop_monitor_timeout_s",
                DEFAULT_TACTILE_RELEASE_STOP_MONITOR_TIMEOUT_S,
            )
        ),
    )
    args.tactile_release_stop_monitor_poll_dt_s = max(
        0.0,
        float(
            tactile_cfg.get(
                "release_stop_monitor_poll_dt_s",
                DEFAULT_TACTILE_RELEASE_STOP_MONITOR_POLL_DT_S,
            )
        ),
    )
    args.tactile_debug = bool(tactile_cfg.get("debug", True))

    return args


def render_depth(depth_image_m, max_depth_m):
    max_depth_m = max(max_depth_m, 1e-6)
    clipped = np.clip(depth_image_m, 0.0, max_depth_m)
    scaled = (255.0 * clipped / max_depth_m).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)


class AnySkinTactileManager:
    """Small AnySkin reader used only when robot.tactile.enabled is true."""

    def __init__(self, args):
        self.enabled = bool(getattr(args, "tactile_enabled", False))
        self.port = str(getattr(args, "tactile_port", DEFAULT_TACTILE_PORT))
        self.num_mags = max(1, int(getattr(args, "tactile_num_mags", DEFAULT_TACTILE_NUM_MAGS)))
        self.baseline_samples = max(1, int(getattr(args, "tactile_baseline_samples", DEFAULT_TACTILE_BASELINE_SAMPLES)))
        self.startup_delay_s = max(0.0, float(getattr(args, "tactile_startup_delay_s", DEFAULT_TACTILE_STARTUP_DELAY_S)))
        self.contact_norm_threshold = max(
            0.0,
            float(getattr(args, "tactile_contact_norm_threshold", DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD)),
        )
        self.debug = bool(getattr(args, "tactile_debug", True))
        self.lock = threading.RLock()
        self.stream = None
        self.baseline = None
        self.latest = np.zeros(self.num_mags * 3, dtype=np.float32)
        self.latest_norm = 0.0
        self.last_error = None
        self.status = "disabled"
        self.release_reference_norm = None
        self.release_delta_norm = None
        self.release_status = "off"
        self.last_sample_perf_s = np.nan
        self.last_sample_unix_s = np.nan

    def start(self):
        if not self.enabled:
            return
        try:
            from anyskin import AnySkinProcess
        except Exception as exc:
            raise RuntimeError(
                "robot.tactile.enabled is true, but the anyskin package could not be imported. "
                "Install AnySkin dependencies or set robot.tactile.enabled: false."
            ) from exc

        try:
            self.stream = AnySkinProcess(num_mags=self.num_mags, port=self.port)
            self.stream.start()
            if self.startup_delay_s > 0.0:
                time.sleep(self.startup_delay_s)
            self.reset_baseline()
            self.status = "ready"
            if self.debug:
                print(f"[Tactile] AnySkin stream started on {self.port} ({self.num_mags} mags).")
        except Exception as exc:
            self.last_error = str(exc)
            self.status = "error"
            self.close()
            raise RuntimeError(f"Failed to start AnySkin tactile reader on {self.port}: {exc}") from exc

    def reset_baseline(self):
        with self.lock:
            if self.stream is None:
                return False
            try:
                baseline_data = self.stream.get_data(num_samples=self.baseline_samples)
                baseline_data = np.asarray(baseline_data, dtype=np.float32)
                if baseline_data.ndim != 2 or baseline_data.shape[1] < self.num_mags * 3 + 1:
                    raise RuntimeError(f"Unexpected AnySkin baseline shape: {baseline_data.shape}")
                self.baseline = np.mean(baseline_data[:, 1 : 1 + self.num_mags * 3], axis=0)
                self.latest = np.zeros(self.num_mags * 3, dtype=np.float32)
                self.latest_norm = 0.0
                self.last_sample_perf_s = time.perf_counter()
                self.last_sample_unix_s = time.time()
                self.last_error = None
                if self.debug:
                    print("[Tactile] Baseline reset.")
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self.status = "read_error"
                print(f"[WARN] Tactile baseline reset failed: {exc}")
                return False

    def read(self):
        with self.lock:
            if not self.enabled or self.stream is None:
                return self.latest.copy()
            if self.baseline is None:
                self.reset_baseline()
            try:
                sensor_data = self.stream.get_data(num_samples=1)[0]
                sensor_data = np.asarray(sensor_data, dtype=np.float32)[1 : 1 + self.num_mags * 3]
                if sensor_data.shape[0] != self.num_mags * 3:
                    raise RuntimeError(f"Unexpected AnySkin sample length: {sensor_data.shape[0]}")
                self.latest = sensor_data - self.baseline
                self.latest_norm = float(np.linalg.norm(self.latest))
                self.last_sample_perf_s = time.perf_counter()
                self.last_sample_unix_s = time.time()
                self.last_error = None
                if self.status == "read_error":
                    self.status = "ready"
                return self.latest.copy()
            except Exception as exc:
                self.last_error = str(exc)
                self.status = "read_error"
                print(f"[WARN] Tactile read failed: {exc}")
                return self.latest.copy()

    def total_norm(self):
        with self.lock:
            self.read()
            return float(self.latest_norm)

    def snapshot(self, *, refresh=True):
        with self.lock:
            if refresh:
                self.read()
            return {
                "values": np.asarray(self.latest, dtype=np.float32).copy(),
                "num_mags": int(self.num_mags),
                "total_norm": float(self.latest_norm),
                "release_ref_norm": self.release_reference_norm,
                "release_delta_norm": self.release_delta_norm,
                "status": str(self.release_status),
                "error": self.last_error,
                "timestamp_perf_s": float(self.last_sample_perf_s),
                "timestamp_unix_s": float(self.last_sample_unix_s),
            }

    def clear_runtime_state(self, *, release_status="reset_cleared"):
        with self.lock:
            self.release_reference_norm = None
            self.release_delta_norm = None
            self.release_status = str(release_status)
            self.latest = np.zeros(self.num_mags * 3, dtype=np.float32)
            self.latest_norm = 0.0
            self.last_sample_perf_s = time.perf_counter()
            self.last_sample_unix_s = time.time()
            self.last_error = None

    def set_release_reference(self, reference_norm):
        with self.lock:
            self.release_reference_norm = None if reference_norm is None else float(reference_norm)
            self.release_delta_norm = None
            self.release_status = "armed" if reference_norm is not None else "off"

    def update_release_delta(self, current_norm):
        with self.lock:
            if self.release_reference_norm is None:
                self.release_delta_norm = None
                return None
            self.release_delta_norm = abs(float(current_norm) - float(self.release_reference_norm))
            return self.release_delta_norm

    def close(self):
        with self.lock:
            stream = self.stream
            self.stream = None
        if stream is None:
            return
        try:
            stream.pause_streaming()
        except Exception:
            pass
        try:
            stream.join()
        except Exception:
            pass
        self.status = "closed"
        if self.debug:
            print("[Tactile] AnySkin stream stopped.")


def clamp_value(v, low, high):
    return max(low, min(high, v))


def clamp_pose_mm(x_mm, y_mm, z_mm, args):
    x_mm = clamp_value(x_mm, args.workspace_x[0], args.workspace_x[1])
    y_mm = clamp_value(y_mm, args.workspace_y[0], args.workspace_y[1])
    z_mm = clamp_value(z_mm, args.workspace_z[0], args.workspace_z[1])
    return x_mm, y_mm, z_mm


def meters_to_mm(values):
    return np.asarray(values, dtype=np.float32) * 1000.0


def mm_to_m_tuple(values):
    vec = np.asarray(values, dtype=np.float32).reshape(3) / 1000.0
    return tuple(float(v) for v in vec)


def compute_dynamic_eef_target(reference_xyz_mm, eef_xyz_mm):
    """Blend the x-offset down as the tool approaches the tracked reference point."""
    reference_xyz_mm = np.asarray(reference_xyz_mm, dtype=np.float32).reshape(3)
    eef_xyz_mm = np.asarray(eef_xyz_mm, dtype=np.float32).reshape(3)

    dist_xy = float(np.linalg.norm(reference_xyz_mm[:2] - eef_xyz_mm[:2]))
    far_offset_x = EEF_X_OFFSET_MM
    near_offset_x = 0.0
    near_d = 40.0
    far_d = abs(float(EEF_X_OFFSET_MM)) + near_d

    if dist_xy >= far_d:
        offset_x = far_offset_x
    elif dist_xy <= near_d:
        offset_x = near_offset_x
    else:
        t = (dist_xy - near_d) / max(far_d - near_d, 1e-6)
        offset_x = t * far_offset_x + (1.0 - t) * near_offset_x

    target_xyz_mm = reference_xyz_mm.copy()
    target_xyz_mm[0] += offset_x
    target_xyz_mm[1] += EEF_Y_OFFSET_MM
    return target_xyz_mm.astype(np.float32), float(offset_x), dist_xy


def get_close_range_step_mm(ref_err_xyz, max_step_mm, max_step_z_mm):
    """Use smaller servo steps near the object for gentler final alignment."""
    dist_xy = float(np.linalg.norm(np.asarray(ref_err_xyz, dtype=np.float32)[:2]))
    if dist_xy < 95.0:
        return 5, 2.2, dist_xy
    return float(max_step_mm), float(max_step_z_mm), dist_xy


def compute_close_range_ref_step_xyz(ref_err_xyz, max_step_xy, max_step_z, *, follow_z):
    """Compute one bounded servo step, staging close XY motion on a dominant axis."""
    ref_err_xyz = np.asarray(ref_err_xyz, dtype=np.float32).reshape(3)
    ref_step_xyz = np.zeros(3, dtype=np.float32)
    dist_xy = float(np.linalg.norm(ref_err_xyz[:2]))
    dominant_axis = None

    # Stage the final XY approach along one axis to reduce diagonal tip collisions.
    if dist_xy < 65.0 and abs(float(ref_err_xyz[0])) > 15.0 and abs(float(ref_err_xyz[1])) > 15.0:
        dominant_axis = 0 if abs(float(ref_err_xyz[0])) >= abs(float(ref_err_xyz[1])) else 1
        ref_step_xyz[dominant_axis] = np.clip(ref_err_xyz[dominant_axis], -max_step_xy, max_step_xy)
    else:
        ref_step_xyz[0:2] = np.clip(ref_err_xyz[0:2], -max_step_xy, max_step_xy)

    if follow_z:
        ref_step_xyz[2] = np.clip(ref_err_xyz[2], -max_step_z, max_step_z)

    return ref_step_xyz, dominant_axis



def get_home_joints_rad():
    """Return the configured HOME joint target in radians."""
    return tuple(float(np.deg2rad(value)) for value in HOME_JOINTS_DEG)


def make_robot_command(command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode="manual"):
    """Create the shared robot command object expected by RtdeController.step()."""
    return RobotCommandState(
        command_type=command_type,
        target_position_base=target_position_base,
        fixed_orientation_base=fixed_orientation_base,
        gripper_action=gripper_action,
        source_mode=source_mode,
        stop_requested=(command_type == ROBOT_CMD_STOP),
        timestamp=time.time(),
        valid=True,
    )


def send_robot_command( controller, command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode="manual"):
    """Send one robot command through the RTDE controller with a fresh timestamp."""
    command = make_robot_command(
        command_type,
        target_position_base=target_position_base,
        fixed_orientation_base=fixed_orientation_base,
        gripper_action=gripper_action,
        source_mode=source_mode,
    )
    return controller.step(command, now_timestamp=command.timestamp)


def init_rtde(args):
    """Connect RTDE, validate robot state, and open the gripper for task startup."""
    print(f"[INFO] Connecting to UR5 RTDE using config {args.config} ...")
    controller = RtdeController.from_config(args.config)
    if args.robot_ip:
        controller.robot_ip = args.robot_ip

    # This standalone script keeps orientation from the live robot pose, which is returned as rotvec.
    controller.fixed_orientation_format = "rotvec"
    controller.connect()

    state = controller.read_robot_state(now_timestamp=time.time())
    if not state.is_connected:
        raise RuntimeError(f"Failed to connect RTDE controller. last_error={state.last_error}")
    if state.actual_tcp_pose_base is None:
        raise RuntimeError("RTDE connected but actual_tcp_pose_base is unavailable.")

    pose = state.actual_tcp_pose_base
    print(
        "[INFO] Current pose base: "
        f"x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}, "
        f"rx={pose[3]:.4f}, ry={pose[4]:.4f}, rz={pose[5]:.4f}"
    )

    if args.move_to_base:
        print("[INFO] --move-to-base is ignored; HOME is defined by HOME_JOINTS_DEG.")

    print("[INFO] Opening gripper at startup...")
    startup_open_ok = False
    if hasattr(controller, "open_gripper_blocking"):
        try:
            startup_open_ok = bool(controller.open_gripper_blocking())
        except Exception as exc:
            print(f"[WARN] Direct startup gripper open failed: {exc}")
            startup_open_ok = False
    if not startup_open_ok:
        send_robot_command(controller, ROBOT_CMD_HOLD, gripper_action=GRIPPER_OPEN, source_mode="startup_open")
    time.sleep(0.2)

    if args.open_gripper:
        print("[INFO] --open-gripper specified; startup open was already applied.")

    return controller


def safe_stop_rtde(controller):
    if controller is None:
        return
    try:
        send_robot_command(controller, ROBOT_CMD_STOP, source_mode="manual_stop")
    except Exception:
        pass


def disconnect_rtde(controller):
    if controller is None:
        return
    try:
        controller.close()
    except Exception:
        pass


class FollowSharedState:
    """Thread-safe task state shared by the perception loop and robot follow thread."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.target_predictor = TargetPredictor(
            process_noise_mm_s2=args.prediction_process_noise_mm_s2,
            measurement_noise_mm=args.prediction_measurement_noise_mm,
            max_velocity_xy_mm_s=args.prediction_max_xy_speed_mm_s,
            reinit_jump_mm=args.prediction_reinit_jump_mm,
        )

        self.follow_enabled = args.enable_follow
        self.latest_target_xyz_mm = None # 로봇 eef가 따라갈 목표 위치 (mm 단위)
        self.latest_grasp_xyz_mm = None # 실제 object내의 grasp point 위치 (mm 단위)
        self.latest_measurement_source = "none" # "measured" or "hand_fallback"
        self.latest_target_t = 0.0 # 마지막으로 타겟이 업데이트된 시간 (초 단위)
        self.last_measured_target_xyz_mm = None # 마지막으로 측정된 타겟 위치 (mm 단위)
        self.last_measured_target_t = 0.0 # 마지막으로 타겟이 측정된 시간 (초 단위)
        self.valid_detection_streak = 0 # 연속적으로 유효한 타겟이 감지된 횟수
        self.prediction_armed = False # 칼만 필터 예측이 활성화되어 있는지 여부
        self.predicted_target_xyz_mm = None # 칼만 필터로 예측된 타겟 위치 (mm 단위)
        self.predicted_target_t = 0.0 # 마지막으로 예측된 타겟 위치가 업데이트된 시간 (초 단위)
        self.prediction_age_s = None # 현재 예측된 타겟 위치가 얼마나 오래되었는지 (초 단위)
        self.control_target_xyz_mm = None # 실제로 로봇이 따라가도록 명령된 타겟 위치 (mm 단위)
        self.target_source = "none" # "measured", "predicted", or "hand_fallback" 중 하나로, control_target_xyz_mm의 출처를 나타냄

        self.reference_object_xy_mm = None # 로봇이 따라갈 때 참조하는 object의 xy 위치 (mm 단위)
        self.reference_object_xyz_mm = None # 로봇이 따라갈 때 참조하는 object의 xyz 위치 (mm 단위)
        self.reference_locked = False # 참조 위치가 고정되어 있는지 여부. True이면 reference_object_xyz_mm이 로봇의 고정된 z와 orientation과 함께 follow 제약으로 사용됨.
        self.motion_triggered = False # follow 모드에서 로봇이 실제로 움직이기 시작했는지 여부
        self.reference_streak = 0 # 연속적으로 참조 위치가 유효한 타겟으로 업데이트된 횟수
        self.initial_object_label = None # follow 모드가 시작ㅂ될 때 참조로 사용된 object의 라벨 (디버그용)

        # self.fixed_z_mm = None # follow 모드에서 로봇의 z 위치를 고정하는 경우의 고정된 z 값 (mm 단위)
        self.fixed_orientation_base = None # follow 모드에서 로봇의 orientation을 고정하는 경우의 고정된 orientation (base 좌표계, rotvec 형식)
        self.initial_pose_base = None # follow 모드가 시작될 때 로봇의 초기 pose (base 좌표계, x/y/z in mm + rotvec)

        self.stop_event = threading.Event() # follow 스레드가 종료되어야 할 때 설정되는 이벤트
        self.follow_pause_requested = False  # follow 스레드가 일시 중지되어야 할 때 설정되는 플래그. follow_idle_event와 함께 사용됨.
        self.follow_idle_event = threading.Event() # follow 스레드가 현재 유휴 상태(즉, 로봇이 움직이지 않고 제어 명령을 기다리는 상태)인지 나타내는 이벤트. follow_pause_requested가 True일 때 follow_idle_event가 set되면 follow 스레드는 제어를 반환할 준비가 된 것으로 간주함. 초기값은 set된 상태로 시작하여, follow 모드가 활성화되고 제어 명령이 주어지면 clear됨. follow 모드가 비활성화되거나 일시 중지 요청이 있을 때 다시 set됨. follow_idle_event는 주로 pregrasp 단계에서 follow 스레드가 로봇 제어를 반환할 때까지 기다리는 데 사용됨.
        self.follow_idle_event.set() # follow 스레드가 초기에는 유휴 상태로 시작하도록 설정

        self.home_object_xyz_mm = None # 홈 위치에서의 object의 xyz 위치 (mm 단위)
        self.home_object_locked = False # 홈 위치에서 object 위치가 고정되어 있는지 여부. True이면 home_object_xyz_mm이 홈 위치에서의 object 위치로 간주되고, follow 모드에서 참조로 사용될 수 있음.
        self.home_pose_buffer = deque(maxlen=8) # 홈 위치에서의 최근 로봇 pose 버퍼 (base 좌표계, x/y/z in mm + rotvec)
        self.home_object_pixel = None # 홈 위치에서 object의 pixel 위치 (u/v in pixels)
        self.home_pixel_buffer = deque(maxlen=15) #  홈 위치에서의 최근 object pixel 위치 버퍼 (u/v in pixels)

        self.object_stopped = False # 로봇이 object를 따라가다가 멈춰야 하는 상황이 발생했는지 여부. True이면 follow 모드에서 로봇이 움직이지 않고 제어 명령을 기다리는 상태로 전환됨.
        self.stop_pose_buffer = deque(maxlen=15) # object이 멈춰야 하는 상황이 발생했을 때의 최근 로봇 pose 버퍼 (base 좌표계, x/y/z in mm + rotvec)

        self.latest_object_xyz_mm = None # 로봇이 따라가고 있는 object의 최신 xyz 위치 (mm 단위). follow 모드에서 로봇이 실제로 따라가는 타겟 위치와는 다를 수 있음. 디버그용으로 사용됨.
        self.pregrasp_started = False # pregrasp 단계가 시작되었는지 여부. True이면 follow 모드에서 로봇이 pregrasp 제약으로 움직이기 시작했음을 나타냄.
        self.grasp_closed = False # grasp이 닫혔는지 여부. True이면 follow 모드에서 로봇이 grasp 제약으로 움직이기 시작했음을 나타냄.
        self.grasp_offset_xyz_mm = None # grasp이 닫힌 후에 로봇과 object 사이의 xyz offset (mm 단위). follow 모드에서 로봇이 grasp 제약으로 움직일 때 참조로 사용됨.
        self.recent_grasp_z_mm_buffer = deque(maxlen=PLACE_Z_GRASP_BUFFER_FRAMES) # 최근 grasp z 샘플 버퍼 (mm 단위). grasp이 닫힌 후에 place 높이 추정에 사용됨.
        self.recent_template_bottom_z_mm_buffer = deque(maxlen=PLACE_Z_GRASP_BUFFER_FRAMES) # 최근 fitted template bottom z 샘플 버퍼 (mm 단위). grasp이 닫힌 후에 place 높이 추정에 사용됨.
        self.frozen_place_z_mm = None # grasp이 닫힌 후에 place 높이로 고정된 z 값 (mm 단위). follow 모드에서 로봇이 place 제약으로 움직일 때 참조로 사용됨. None이면 아직 고정되지 않은 상태를 나타냄.
        self.frozen_place_z_raw_mm = None # 고정된 place z의 원시값 (mm 단위). grasp z 샘플과 fitted template bottom z 샘플의 중앙값을 계산하여 place z로 고정할 때, 이 값은 중앙값 계산에 사용된 원시 샘플의 중앙값을 나타냄. 디버그용으로 사용됨.
        self.frozen_place_z_grasp_median_mm = None # 고정된 place z의 grasp z 샘플 중앙값 (mm 단위). grasp이 닫힌 후에 place 높이로 고정할 때, 이 값은 중앙값 계산에 사용된 grasp z 샘플의 중앙값을 나타냄. 디버그용으로 사용됨.
        self.frozen_place_z_template_bottom_median_mm = None # 고정된 place z의 fitted template bottom z 샘플 중앙값 (mm 단위). grasp이 닫힌 후에 place 높이로 고정할 때, 이 값은 중앙값 계산에 사용된 fitted template bottom z 샘플의 중앙값을 나타냄. 디버그용으로 사용됨.
        self.frozen_place_z_used_fallback = True # 고정된 place z가 fallback 값(예: grasp z 샘플 중앙값)으로 고정되었는지 여부. True이면 place z가 grasp z 샘플 중앙값과 같은 fallback 값으로 고정되었음을 나타냄. False이면 place z가 grasp z 샘플과 fitted template bottom z 샘플의 중앙값 계산 결과로 고정되었음을 나타냄. 디버그용으로 사용됨.
        self.task_state = "FOLLOW" # 현재 태스크 상태를 나타내는 문자열. 예: "FOLLOW", "PREGRASP", "GRASP", "PLACE", "RETURN", 등. follow 모드에서 로봇이 어떤 단계의 제약으로 움직이고 있는지를 나타냄.
        self.task_epoch = 0 # 태스크 상태가 변경될 때마다 증가하는 카운터. 디버그용으로 사용됨.

    def set_fixed_pose_from_robot(self, controller):
        """Capture z and orientation from the robot for follow-mode constraints."""
        state = controller.read_robot_state(now_timestamp=time.time())
        pose = state.actual_tcp_pose_base
        if pose is None:
            raise RuntimeError("Failed to get current robot pose for fixed pose.")
        with self.lock:
            # self.fixed_z_mm = float(pose[2] * 1000.0)
            self.fixed_orientation_base = tuple(float(v) for v in pose[3:6])
            self.initial_pose_base = tuple(float(v) for v in pose)
        print(
            "[INFO] Fixed pose set from current robot pose: "
            # f"z={self.fixed_z_mm:.2f} mm, "
            f"rotvec=({pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f})"
        )

    def _reset_prediction_locked(self, *, reset_arm=False):
        self.target_predictor.reset()
        self.predicted_target_xyz_mm = None
        self.predicted_target_t = 0.0
        self.prediction_age_s = None
        self.control_target_xyz_mm = None
        self.target_source = "none"
        if reset_arm:
            self.prediction_armed = False

    def set_task_state(self, task_state, *, reset_prediction=False, reset_arm=False):
        with self.lock:
            self.task_state = str(task_state)
            if reset_prediction:
                self._reset_prediction_locked(reset_arm=reset_arm)

    def set_follow_enabled(self, follow_enabled, *, reset_prediction=True, reset_arm=True):
        with self.lock:
            self.follow_enabled = bool(follow_enabled)
            if reset_prediction:
                self._reset_prediction_locked(reset_arm=reset_arm)
            print(f"[INFO] follow_enabled = {self.follow_enabled}")

    def toggle_follow(self):
        with self.lock:
            self.follow_enabled = not self.follow_enabled
            self._reset_prediction_locked(reset_arm=True)
            print(f"[INFO] follow_enabled = {self.follow_enabled}")

    def stop_follow(self):
        with self.lock:
            self.follow_enabled = False
            self._reset_prediction_locked(reset_arm=True)
            print("[INFO] Robot follow stopped.")

    def clear_target(self, *, reset_prediction=False, reset_arm=False):
        with self.lock:
            self.latest_target_xyz_mm = None
            self.latest_grasp_xyz_mm = None
            self.latest_measurement_source = "none"
            self.valid_detection_streak = 0
            self.reference_streak = 0
            self.control_target_xyz_mm = None
            self.target_source = "none"
            if reset_prediction:
                self._reset_prediction_locked(reset_arm=reset_arm)

    def update_place_z_samples(
        self,
        *,
        grasp_point_base=None,
        object_point_base=None,
        fitted_points_base=None,
    ):
        """Collect recent grasp and fitted-template z samples for later place height."""
        grasp_z_mm = None
        source_point = grasp_point_base if grasp_point_base is not None else object_point_base
        if source_point is not None:
            source_point_arr = np.asarray(source_point, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(source_point_arr)):
                grasp_z_mm = float(source_point_arr[2] * 1000.0)

        template_bottom_z_mm = None
        if fitted_points_base is not None:
            fitted_points = np.asarray(fitted_points_base, dtype=np.float32).reshape((-1, 3))
            if len(fitted_points) > 0:
                valid_z = fitted_points[np.isfinite(fitted_points[:, 2]), 2]
                if len(valid_z) > 0:
                    template_bottom_z_mm = float(np.min(valid_z) * 1000.0)

        with self.lock:
            if grasp_z_mm is not None:
                self.recent_grasp_z_mm_buffer.append(grasp_z_mm)
            if template_bottom_z_mm is not None:
                self.recent_template_bottom_z_mm_buffer.append(template_bottom_z_mm)

    def finalize_place_z_from_recent_samples(self):
        """Freeze the place height estimate after grasp closes."""
        with self.lock:
            self.frozen_place_z_mm = None
            self.frozen_place_z_raw_mm = None
            self.frozen_place_z_grasp_median_mm = None
            self.frozen_place_z_template_bottom_median_mm = None
            self.frozen_place_z_used_fallback = True

            grasp_samples = np.asarray(self.recent_grasp_z_mm_buffer, dtype=np.float32)
            template_bottom_samples = np.asarray(self.recent_template_bottom_z_mm_buffer, dtype=np.float32)
            if (
                len(grasp_samples) < PLACE_Z_MIN_VALID_SAMPLES
                or len(template_bottom_samples) < PLACE_Z_MIN_VALID_SAMPLES
            ):
                return {
                    "valid": False,
                    "reason": "insufficient_samples",
                    "grasp_sample_count": int(len(grasp_samples)),
                    "template_bottom_sample_count": int(len(template_bottom_samples)),
                    "grasp_z_median_mm": None,
                    "template_bottom_z_median_mm": None,
                    "raw_place_z_mm": None,
                    "place_z_mm": None,
                }

            grasp_z_median_mm = float(np.median(grasp_samples))
            template_bottom_z_median_mm = float(np.median(template_bottom_samples))
            raw_place_z_mm = float(grasp_z_median_mm - template_bottom_z_median_mm + RELEASE_PARAMETER_MM)
            place_z_mm = max(raw_place_z_mm, HOME_PLACE_MIN_Z_MM)

            self.frozen_place_z_mm = place_z_mm
            self.frozen_place_z_raw_mm = raw_place_z_mm
            self.frozen_place_z_grasp_median_mm = grasp_z_median_mm
            self.frozen_place_z_template_bottom_median_mm = template_bottom_z_median_mm
            self.frozen_place_z_used_fallback = False
            return {
                "valid": True,
                "reason": "ok",
                "grasp_sample_count": int(len(grasp_samples)),
                "template_bottom_sample_count": int(len(template_bottom_samples)),
                "grasp_z_median_mm": grasp_z_median_mm,
                "template_bottom_z_median_mm": template_bottom_z_median_mm,
                "raw_place_z_mm": raw_place_z_mm,
                "place_z_mm": place_z_mm,
            }

    def request_follow_pause(self):
        with self.lock:
            self.follow_pause_requested = True

    def clear_follow_pause(self):
        with self.lock:
            self.follow_pause_requested = False

    def reset_for_restart(self, *, follow_enabled=None):
        """Clear task-local state so the next handover starts from a clean epoch."""
        with self.lock:
            if follow_enabled is None:
                follow_enabled = self.args.enable_follow
            self.follow_enabled = bool(follow_enabled)
            self.latest_target_xyz_mm = None # 로봇 eef가 따라갈 목표 위치 (mm 단위)
            self.latest_grasp_xyz_mm = None # 실제 object내의 grasp point 위치 (mm 단위)
            self.latest_measurement_source = "none" # "measured" or "hand_fallback"
            self.latest_target_t = 0.0
            self.last_measured_target_xyz_mm = None
            self.last_measured_target_t = 0.0
            self.valid_detection_streak = 0
            self.prediction_armed = False
            self.reference_object_xy_mm = None
            self.reference_object_xyz_mm = None
            self.reference_locked = False
            self.motion_triggered = False
            self.reference_streak = 0
            self.initial_object_label = None
            # self.fixed_z_mm = None
            self.fixed_orientation_base = None
            self.initial_pose_base = None
            self.follow_pause_requested = False
            self.home_object_xyz_mm = None
            self.home_object_locked = False
            self.home_pose_buffer.clear()
            self.home_object_pixel = None
            self.home_pixel_buffer.clear()
            self.object_stopped = False
            self.stop_pose_buffer.clear()
            self.latest_object_xyz_mm = None
            self.pregrasp_started = False
            self.grasp_closed = False
            self.grasp_offset_xyz_mm = None
            self.recent_grasp_z_mm_buffer.clear()
            self.recent_template_bottom_z_mm_buffer.clear()
            self.frozen_place_z_mm = None
            self.frozen_place_z_raw_mm = None
            self.frozen_place_z_grasp_median_mm = None
            self.frozen_place_z_template_bottom_median_mm = None
            self.frozen_place_z_used_fallback = True
            self.task_state = "FOLLOW"
            self.task_epoch += 1
            self._reset_prediction_locked(reset_arm=True)
        self.follow_idle_event.set()
        print(f"[INFO] Follow/task state reset. follow_enabled={self.follow_enabled}")

    def set_follow_thread_idle(self, is_idle):
        if is_idle:
            self.follow_idle_event.set()
        else:
            self.follow_idle_event.clear()

    def wait_for_follow_idle(self, timeout_s):
        return self.follow_idle_event.wait(timeout=max(float(timeout_s), 0.0))

    def update_target(
        self,
        grasp_xyz_m,
        object_xyz_m=None,
        pixel_xy=None,
        eef_xyz_mm=None,
        measurement_source="measured",
        object_label=None,
    ):
        """Update measured/fallback target state and arm follow once motion is detected."""
        if object_xyz_m is None:
            self.clear_target(reset_prediction=False, reset_arm=False)
            return

        measurement_source = str(measurement_source or "measured").strip().lower()
        if measurement_source not in {"measured", "hand_fallback"}:
            measurement_source = "measured"
        normalized_object_label = None
        if object_label is not None:
            normalized_object_label = str(object_label).strip().lower() or None

        object_xyz_mm = np.asarray(object_xyz_m, dtype=np.float32)[:3] * 1000.0
        object_xy_mm = object_xyz_mm[:2].copy()
        grasp_xyz_mm = None
        tracking_reference_xyz_mm = object_xyz_mm.copy()
        target_xyz_mm = None
        if grasp_xyz_m is not None:
            grasp_xyz_mm = np.asarray(grasp_xyz_m, dtype=np.float32)[:3] * 1000.0
            tracking_reference_xyz_mm = grasp_xyz_mm.copy()

        if eef_xyz_mm is None:
            target_xyz_mm = tracking_reference_xyz_mm.copy()
            target_xyz_mm[0] += EEF_X_OFFSET_MM
            target_xyz_mm[1] += EEF_Y_OFFSET_MM
        else:
            target_xyz_mm, dynamic_offset_x, dist_xy = compute_dynamic_eef_target(
                tracking_reference_xyz_mm,
                eef_xyz_mm,
            )
            if self.args.verbose_robot:
                target_kind = "grasp" if grasp_xyz_mm is not None else "object"
                print(
                    f"[ROBOT] tracking={target_kind}, dynamic_offset_x={dynamic_offset_x:.1f} mm, "
                    f"eef_target_dist_xy={dist_xy:.1f} mm"
                )

        self.try_lock_home_pose(object_xyz_mm, pixel_xy)
        self.update_stop_state(object_xyz_mm)

        with self.lock:
            current_perf = time.perf_counter()
            self.latest_object_xyz_mm = object_xyz_mm.copy()
            self.latest_grasp_xyz_mm = None if grasp_xyz_mm is None else grasp_xyz_mm.copy()
            self.latest_measurement_source = measurement_source

            if not self.reference_locked:
                self.reference_streak += 1
                if self.reference_streak >= REFERENCE_LOCK_COUNT:
                    self.reference_object_xy_mm = object_xy_mm.copy()
                    self.reference_object_xyz_mm = object_xyz_mm.copy()
                    self.reference_locked = True
                    self.motion_triggered = False
                    print(f"[INFO] Reference object position locked: {self.reference_object_xyz_mm}")
                self.latest_target_xyz_mm = None
                self.valid_detection_streak = 0
                self._reset_prediction_locked(reset_arm=True)
                return

            if (
                measurement_source == "measured"
                and not self.motion_triggered
                and normalized_object_label is not None
                and self.initial_object_label != normalized_object_label
            ):
                self.initial_object_label = normalized_object_label
                print(
                    "[INFO] Cached initial segmentation label for grasp threshold: "
                    f"label={self.initial_object_label}"
                )

            if not self.motion_triggered:
                move_dist_mm = float(np.linalg.norm(object_xy_mm - self.reference_object_xy_mm))
                move_dist_z_mm = 0.0
                if self.reference_object_xyz_mm is not None:
                    move_dist_z_mm = float(abs(object_xyz_mm[2] - self.reference_object_xyz_mm[2]))
                if move_dist_mm >= MOTION_TRIGGER_MM or move_dist_z_mm >= MOTION_TRIGGER_Z_MM:
                    self.motion_triggered = True
                    print(
                        "[INFO] Object motion detected: "
                        f"xy={move_dist_mm:.2f} mm, z={move_dist_z_mm:.2f} mm -> follow start"
                    )

            self.latest_target_xyz_mm = target_xyz_mm.copy()
            self.valid_detection_streak += 1
            self.latest_target_t = current_perf
            if measurement_source == "measured":
                self.last_measured_target_xyz_mm = target_xyz_mm.copy()
                self.last_measured_target_t = current_perf
            if self.valid_detection_streak >= int(self.args.min_valid_count):
                self.prediction_armed = True
            if (
                measurement_source == "measured"
                and self.args.enable_target_prediction
                and self.task_state == "FOLLOW"
            ):
                self.target_predictor.update(target_xyz_mm, current_perf)

    def _refresh_tracking_targets_locked(self, now_perf):
        """Select the freshest control target, falling back to short-horizon prediction."""
        measurement_age_s = None
        if self.last_measured_target_t > 0.0:
            measurement_age_s = max(float(now_perf) - float(self.last_measured_target_t), 0.0)

        self.predicted_target_xyz_mm = None
        self.predicted_target_t = 0.0
        self.prediction_age_s = None
        self.control_target_xyz_mm = None
        self.target_source = "none"

        if self.task_state != "FOLLOW":
            return measurement_age_s

        if self.args.enable_target_prediction and self.target_predictor.has_state():
            predicted = self.target_predictor.predict(now_perf)
            if predicted is not None and predicted.valid:
                self.predicted_target_xyz_mm = predicted.xyz_mm.copy()
                self.predicted_target_t = float(now_perf)
                self.prediction_age_s = float(predicted.prediction_age_s)

        raw_target = None if self.latest_target_xyz_mm is None else self.latest_target_xyz_mm.copy()
        raw_is_fresh = (
            raw_target is not None
            and self.latest_target_t > 0.0
            and (float(now_perf) - float(self.latest_target_t)) <= float(self.args.target_timeout_s)
        )
        if raw_is_fresh:
            self.control_target_xyz_mm = raw_target
            self.target_source = str(self.latest_measurement_source or "measured")
            return measurement_age_s

        predicted_is_valid = (
            self.args.enable_target_prediction
            and self.prediction_armed
            and self.predicted_target_xyz_mm is not None
            and self.prediction_age_s is not None
            and float(self.prediction_age_s) <= float(self.args.prediction_max_horizon_s)
        )
        if predicted_is_valid:
            self.control_target_xyz_mm = self.predicted_target_xyz_mm.copy()
            self.target_source = "predicted"

        return measurement_age_s

    def get_snapshot(self, now_perf=None):
        """Return a consistent copy of follow/task state for readers outside the lock."""
        with self.lock:
            now_perf = time.perf_counter() if now_perf is None else float(now_perf)
            measurement_age_s = self._refresh_tracking_targets_locked(now_perf)
            return {
                "follow_enabled": self.follow_enabled,
                "latest_target_xyz_mm": None if self.latest_target_xyz_mm is None else self.latest_target_xyz_mm.copy(),
                "latest_grasp_xyz_mm": None if self.latest_grasp_xyz_mm is None else self.latest_grasp_xyz_mm.copy(),
                "measurement_source": self.latest_measurement_source,
                "latest_target_t": self.latest_target_t,
                "last_measured_target_xyz_mm": None if self.last_measured_target_xyz_mm is None else self.last_measured_target_xyz_mm.copy(),
                "last_measured_target_t": self.last_measured_target_t,
                "valid_detection_streak": self.valid_detection_streak,
                "prediction_armed": self.prediction_armed,
                "predicted_target_xyz_mm": None if self.predicted_target_xyz_mm is None else self.predicted_target_xyz_mm.copy(),
                "predicted_target_t": self.predicted_target_t,
                "prediction_age_s": self.prediction_age_s,
                "control_target_xyz_mm": None if self.control_target_xyz_mm is None else self.control_target_xyz_mm.copy(),
                "target_source": self.target_source,
                "measurement_age_s": measurement_age_s,
                "motion_triggered": self.motion_triggered,
                "follow_pause_requested": self.follow_pause_requested,
                # "fixed_z_mm": self.fixed_z_mm,
                "fixed_orientation_base": None if self.fixed_orientation_base is None else tuple(self.fixed_orientation_base),
                "latest_object_xyz_mm": None if self.latest_object_xyz_mm is None else self.latest_object_xyz_mm.copy(),
                "initial_object_label": self.initial_object_label,
                "object_stopped": self.object_stopped,
                "pregrasp_started": self.pregrasp_started,
                "initial_pose_base": None if self.initial_pose_base is None else tuple(self.initial_pose_base),
                "task_state": self.task_state,
                "task_epoch": self.task_epoch,
            }

    def try_lock_home_pose(self, object_xyz_mm, pixel_xy):
        """Lock the stable start pose used later as the delivery location."""
        if self.home_object_locked:
            return

        self.home_pose_buffer.append(object_xyz_mm.copy())
        if pixel_xy is not None:
            self.home_pixel_buffer.append(np.array(pixel_xy, dtype=np.int32))

        if len(self.home_pose_buffer) < self.home_pose_buffer.maxlen:
            return

        buf = np.stack(self.home_pose_buffer, axis=0)
        xyz_range = buf.max(axis=0) - buf.min(axis=0)

        stable_xy = (xyz_range[0] < 15.0) and (xyz_range[1] < 15.0)
        stable_z = xyz_range[2] < 18.0

        if stable_xy and stable_z:
            self.home_object_xyz_mm = buf.mean(axis=0)

            if len(self.home_pixel_buffer) > 0:
                pix_buf = np.stack(self.home_pixel_buffer, axis=0)
                home_pix = np.median(pix_buf, axis=0).astype(np.int32)
                self.home_object_pixel = (int(home_pix[0]), int(home_pix[1]))
            else:
                self.home_object_pixel = None

            self.home_object_locked = True
            print(f"[INFO] Home object position locked: {self.home_object_xyz_mm}")
            if self.home_object_pixel is not None:
                print(f"[INFO] Home object pixel locked: {self.home_object_pixel}")

    def update_stop_state(self, object_xyz_mm):
        """Detect when the object has settled after motion starts."""
        if not self.motion_triggered:
            self.object_stopped = False
            self.stop_pose_buffer.clear()
            return

        self.stop_pose_buffer.append(object_xyz_mm.copy())

        if len(self.stop_pose_buffer) < self.stop_pose_buffer.maxlen:
            self.object_stopped = False
            return

        buf = np.stack(self.stop_pose_buffer, axis=0)
        xyz_range = buf.max(axis=0) - buf.min(axis=0)

        stable_xy = (xyz_range[0] < 30.0) and (xyz_range[1] < 30.0)
        stable_z = xyz_range[2] < 50.0

        if stable_xy and stable_z:
            if not self.object_stopped:
                print(f"[INFO] Object STOPPED detected. xyz_range={xyz_range}")
            self.object_stopped = True
        else:
            self.object_stopped = False

    def is_pregrasp_pose_reached(
        self,
        eef_pose_base,
        x_tol_mm=210.0,
        y_tol_mm=30.0,
        z_tol_mm=30.0,
    ):
        """Check whether a robot pose is close enough to close the gripper."""
        snapshot = self.get_snapshot()
        target_xyz = snapshot["latest_grasp_xyz_mm"]
        if target_xyz is None:
            target_xyz = snapshot["latest_object_xyz_mm"]

        if target_xyz is None or eef_pose_base is None:
            return False

        eef_x, eef_y, eef_z = meters_to_mm(eef_pose_base[:3])
        obj_x, obj_y, obj_z = target_xyz[:3]

        dx = float(obj_x - eef_x)
        dy = float(obj_y - eef_y)
        dz = float(obj_z - eef_z)

        x_ok = abs(dx) <= x_tol_mm
        y_ok = abs(dy) <= y_tol_mm
        z_ok = abs(dz) <= z_tol_mm

        # print(
        #     f"[DEBUG] final grasp window | "
        #     f"|dx|={abs(dx):.1f} <= {x_tol_mm:.1f} -> {x_ok}, "
        #     f"|dy|={abs(dy):.1f} <= {y_tol_mm:.1f} -> {y_ok}, "
        #     f"|dz|={abs(dz):.1f} <= {z_tol_mm:.1f} -> {z_ok}"
        # )
        return x_ok and y_ok and z_ok

    def should_start_pregrasp(
        self,
        controller,
        x_tol_mm=210.0,
        y_tol_mm=30.0,
        z_tol_mm=30.0,
    ):
        """Compatibility wrapper for tests and legacy callers."""
        state = controller.read_robot_state(now_timestamp=time.time())
        return self.is_pregrasp_pose_reached(
            state.actual_tcp_pose_base,
            x_tol_mm=x_tol_mm,
            y_tol_mm=y_tol_mm,
            z_tol_mm=z_tol_mm,
        )

def robot_control_loop(controller, shared_state, args):
    """Servo the UR5 toward the current control target while follow mode is active."""
    interval = 1.0 / max(float(args.control_hz), 1e-6)
    max_step_mm = MAX_XY_SPEED_MM_S / max(float(args.control_hz), 1e-6)
    max_step_z_mm = MAX_Z_SPEED_MM_S / max(float(args.control_hz), 1e-6)
    last_sent_pose_mm = None
    ref_target_xyz_mm = None
    was_active = False

    while not shared_state.stop_event.is_set():
        start_t = time.time()
        snap = shared_state.get_snapshot()
        control_target_xyz_mm = snap["control_target_xyz_mm"]
        target_source = snap["target_source"]

        active = True
        if snap["follow_pause_requested"]:
            active = False
        elif not snap["follow_enabled"]:
            active = False
        elif control_target_xyz_mm is None:
            active = False
        elif target_source != "predicted" and snap["valid_detection_streak"] < args.min_valid_count and not snap["prediction_armed"]:
            active = False
        elif target_source == "predicted" and not snap["prediction_armed"]:
            active = False
        elif not snap["motion_triggered"]:
            active = False
        elif target_source == "predicted" and (
            snap["prediction_age_s"] is None or float(snap["prediction_age_s"]) > float(args.prediction_max_horizon_s)
        ):
            active = False
        # elif snap["fixed_z_mm"] is None or snap["fixed_orientation_base"] is None:
        #     active = False
        elif snap["fixed_orientation_base"] is None:
            active = False

        if not active:
            if was_active:
                safe_stop_rtde(controller)
                ref_target_xyz_mm = None
                last_sent_pose_mm = None
            was_active = False
            shared_state.set_follow_thread_idle(True)
            time.sleep(max(0.0, interval - (time.time() - start_t)))
            continue

        shared_state.set_follow_thread_idle(False)

        target_xyz_mm = control_target_xyz_mm
        # fixed_z_mm = snap["fixed_z_mm"]
        fixed_orientation_base = snap["fixed_orientation_base"]

        if ref_target_xyz_mm is None:
            state = controller.read_robot_state(now_timestamp=time.time())
            pose = state.actual_tcp_pose_base
            if pose is None:
                time.sleep(max(0.0, interval - (time.time() - start_t)))
                continue
            ref_target_xyz_mm = meters_to_mm(pose[:3]).astype(np.float32)
            print(f"[INFO] ref_target initialized from current EEF xyz: {ref_target_xyz_mm}")

        ref_err_xyz = target_xyz_mm - ref_target_xyz_mm
        max_step_xy, max_step_z, dist_xy = get_close_range_step_mm(
            ref_err_xyz,
            max_step_mm,
            max_step_z_mm,
        )
        ref_step_xyz, dominant_axis = compute_close_range_ref_step_xyz(
            ref_err_xyz,
            max_step_xy,
            max_step_z,
            follow_z=bool(args.follow_z),
        )

        # if not args.follow_z:
        #     ref_target_xyz_mm[2] = fixed_z_mm

        ref_target_xyz_mm = ref_target_xyz_mm + ref_step_xyz
        cmd_z = float(ref_target_xyz_mm[2]) # if args.follow_z else float(fixed_z_mm)
        cmd_x, cmd_y, cmd_z = clamp_pose_mm(float(ref_target_xyz_mm[0]), float(ref_target_xyz_mm[1]), cmd_z, args)
        pose_mm = np.array([cmd_x, cmd_y, cmd_z], dtype=np.float32)

        if last_sent_pose_mm is not None:
            pos_delta = np.linalg.norm(pose_mm - last_sent_pose_mm)
            if pos_delta < 0.2:
                time.sleep(max(0.0, interval - (time.time() - start_t)))
                continue

        target_position_base = mm_to_m_tuple(pose_mm)
        try:
            send_robot_command(
                controller,
                ROBOT_CMD_SERVO_TO_POSITION,
                target_position_base=target_position_base,
                fixed_orientation_base=fixed_orientation_base,
                gripper_action=GRIPPER_HOLD,
                source_mode="follow_servo",
            )
            was_active = True
            last_sent_pose_mm = pose_mm
            # if args.verbose_robot:
            #     print(
            #         f"[ROBOT] source={target_source}, "
            #         f"raw_mm={snap['latest_target_xyz_mm']}, "
            #         f"control_mm={target_xyz_mm}, "
            #         f"ref_mm={ref_target_xyz_mm}, "
            #         f"close_range_dist_xy={dist_xy:.1f}, "
            #         f"stage_axis={'xy' if dominant_axis is None else ('x' if dominant_axis == 0 else 'y')}, "
            #         f"cmd_m={target_position_base}"
            #     )
        except Exception as exc:
            print(f"[WARN] servo command failed: {exc}")

        elapsed = time.time() - start_t
        time.sleep(max(0.0, interval - elapsed))


class RobotWorker:
    """Single thread that owns all RTDE and gripper commands."""

    def __init__(self, args, shared_state, metadata_recorder=None, tactile_manager=None):
        # 로봇 제어 주기, timeout, gripper/tactile 설정 등 실행 옵션을 보관한다.
        self.args = args
        # perception main loop와 target/follow 상태를 주고받는 공유 상태 객체다.
        self.shared_state = shared_state
        # grasp/place 결과와 task 이벤트를 기록하는 metadata recorder다.
        self.metadata_recorder = metadata_recorder
        # tactile grasp 판정 또는 tactile 로그 저장에 사용할 manager다.
        self.tactile_manager = tactile_manager
        # RTDE controller 인스턴스이며, init 전이나 disconnect 후에는 None이다.
        self.controller = None
        # main thread가 submit한 RobotRequest를 worker thread가 순서대로 처리하는 큐다.
        self._request_queue = queue.Queue()
        # 외부에서 get_status()로 읽는 로봇 worker 상태 snapshot이다.
        self._status = RobotStatus()
        # _status를 여러 thread가 동시에 읽고 쓰지 않도록 보호하는 lock이다.
        self._status_lock = threading.Lock()
        # 긴 동작 중 reset/stop/shutdown 같은 요청이 들어왔을 때 취소 신호로 사용한다.
        self._cancel_event = threading.Event()
        # worker thread의 main loop 종료 조건으로 사용하는 event다.
        self._shutdown_event = threading.Event()
        # 실제 로봇 명령을 소유하는 background thread 핸들이다.
        self._thread = None
        # request에 고유 ID를 붙이기 위한 단조 증가 counter다.
        self._request_seq = 0
        # request ID 발급이 thread-safe 하도록 보호하는 lock이다.
        self._request_seq_lock = threading.Lock()
        # follow servo 중 마지막으로 로봇에 보낸 TCP 목표 위치[mm]다.
        self._last_sent_pose_mm = None
        # follow servo가 한 번에 움직일 reference target 위치[mm]다.
        self._ref_target_xyz_mm = None
        # 직전 follow tick에서 실제 servo 명령이 활성 상태였는지 추적한다.
        self._was_follow_active = False
        # control_hz에 맞춰 다음 follow tick을 실행할 wall-clock 시각이다.
        self._next_follow_tick_t = 0.0
        # FOLLOWING이 아닐 때도 주기적으로 robot state를 읽기 위한 마지막 read 시각이다.
        self._last_status_read_t = 0.0

    def start(self):
        # 이미 worker thread가 살아 있으면 중복으로 시작하지 않는다.
        if self._thread is not None and self._thread.is_alive():
            return
        # 모든 RTDE/gripper 명령은 이 daemon thread 안에서만 실행된다.
        self._thread = threading.Thread(target=self._run, name="robot-worker", daemon=True)
        # background worker loop를 시작한다.
        self._thread.start()

    def join(self, timeout=None):
        # 종료 시 main thread가 worker thread 정리를 기다릴 때 사용한다.
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, request):
        # 문자열 등으로 들어온 요청도 RobotRequest 형태로 정규화한다.
        if not isinstance(request, RobotRequest):
            request = RobotRequest(str(request))
        # payload는 queue에 넣기 전에 shallow copy해서 호출자 변경과 분리한다.
        payload = {} if request.payload is None else dict(request.payload)
        # 요청마다 고유 request_id를 부여한다.
        with self._request_seq_lock:
            self._request_seq += 1
            request_id = self._request_seq
        # worker queue에 들어갈 immutable에 가까운 request 객체를 새로 만든다.
        queued_request = RobotRequest(
            type=str(request.type),
            payload=payload,
            created_at=float(request.created_at),
            request_id=request_id,
        )
        # 긴 로봇 동작을 끊어야 하는 긴급 요청이면 cancel_event를 먼저 세운다.
        if queued_request.type in ROBOT_URGENT_REQUESTS:
            self._cancel_event.set()
        # worker thread가 처리할 수 있도록 request queue에 넣는다.
        self._request_queue.put(queued_request)
        # 호출자가 로그나 디버깅에 사용할 수 있게 request_id를 반환한다.
        return request_id

    def get_status(self):
        # RobotStatus를 복사해서 반환해 외부 thread가 내부 상태를 직접 바꾸지 못하게 한다.
        with self._status_lock:
            return replace(self._status)

    def _set_status(self, **updates):
        # 여러 status field를 lock 안에서 원자적으로 갱신한다.
        with self._status_lock:
            for key, value in updates.items():
                setattr(self._status, key, value)

    def _bump_status_counter(self, field_name):
        # task_ready/reset_done/task_done 같은 edge-trigger 이벤트 epoch를 1 증가시킨다.
        with self._status_lock:
            setattr(self._status, field_name, int(getattr(self._status, field_name)) + 1)
            return int(getattr(self._status, field_name))

    def _current_state(self):
        # 현재 robot worker state만 안전하게 읽는다.
        with self._status_lock:
            return self._status.state

    def _set_active_request(self, request):
        # 지금 처리 중인 request type/id를 status에 노출한다.
        self._set_status(active_request=request.type, active_request_id=int(request.request_id))

    def _clear_active_request(self):
        # request 처리가 끝나면 active request 표시를 비운다.
        self._set_status(active_request=None, active_request_id=0)

    def _reset_follow_tracking(self):
        # follow servo의 이전 목표와 활성 상태를 모두 초기화한다.
        self._last_sent_pose_mm = None
        self._ref_target_xyz_mm = None
        self._was_follow_active = False
        # robot worker가 servo 명령을 쥐고 있지 않음을 shared state에 알린다.
        self.shared_state.set_follow_thread_idle(True)

    def _publish_robot_state(self, state):
        # blocking move/gripper helper 안에서 읽은 RTDE state를 main loop가 볼 수 있게 publish한다.
        if state is None:
            return
        updates = {}
        if hasattr(state, "actual_tcp_pose_base"):
            pose = getattr(state, "actual_tcp_pose_base", None)
            updates["last_robot_pose"] = None if pose is None else tuple(float(v) for v in pose)
        if hasattr(state, "is_connected"):
            updates["is_connected"] = bool(getattr(state, "is_connected", False))
        using_mock = getattr(state, "using_mock", getattr(self.controller, "using_mock", None))
        if using_mock is not None:
            updates["using_mock"] = bool(using_mock)
        if hasattr(state, "last_error"):
            updates["last_error"] = getattr(state, "last_error", None)
        if updates:
            self._set_status(**updates)
        self._last_status_read_t = time.time()

    def _read_robot_state(self):
        # controller가 없으면 연결되지 않은 상태로 status를 갱신한다.
        if self.controller is None:
            self._set_status(is_connected=False, using_mock=False)
            return None
        # RTDE controller에서 현재 TCP pose, 연결 상태, 에러 등을 읽는다.
        try:
            state = self.controller.read_robot_state(now_timestamp=time.time())
        except Exception as exc:
            # read 실패는 status에 에러로 남기고 이번 frame은 상태 없음으로 처리한다.
            self._set_status(last_error=str(exc), is_connected=False)
            return None

        # 외부 main loop가 볼 수 있도록 최신 robot status를 publish한다.
        self._publish_robot_state(state)
        return state

    def _send_robot_command(self, command_type, **kwargs):
        # controller가 없으면 명령을 보낼 대상이 없으므로 no-op 처리한다.
        if self.controller is None:
            return None
        # 마지막으로 보낸 command type을 status에 남긴다.
        self._set_status(last_command_type=command_type)
        # 실제 RTDE/gripper command helper로 명령을 전달한다.
        return send_robot_command(self.controller, command_type, **kwargs)

    def _safe_stop(self, source_mode="worker_stop"):
        # controller가 없으면 멈출 로봇 연결도 없다.
        if self.controller is None:
            return
        # stop 명령 자체가 실패해도 worker가 죽지 않도록 보호한다.
        try:
            self._send_robot_command(ROBOT_CMD_STOP, source_mode=source_mode)
        except Exception as exc:
            self._set_status(last_error=str(exc))

    def _disconnect(self):
        # 이미 disconnect 상태면 아무것도 하지 않는다.
        if self.controller is None:
            return
        # RTDE 연결과 gripper/controller 리소스를 닫는다.
        disconnect_rtde(self.controller)
        self.controller = None
        # 외부 status에도 연결 해제 상태를 반영한다.
        self._set_status(is_connected=False, using_mock=False, last_robot_pose=None)

    def _publish_task_ready(self):
        # main loop가 새 trial 시작을 감지하도록 task_ready_epoch를 증가시킨다.
        self._bump_status_counter("task_ready_epoch")

    def _run(self):
        # shutdown event가 세워질 때까지 request 처리와 follow servo를 반복한다.
        while not self._shutdown_event.is_set():
            try:
                # request queue를 짧게 기다려서 명령 처리 지연과 follow 주기를 균형 있게 유지한다.
                request = self._request_queue.get(timeout=0.01)
                # 들어온 request type에 맞는 handler를 실행한다.
                self._handle_request(request)
                # shutdown request를 처리했다면 worker loop를 즉시 빠져나간다.
                if request.type == ROBOT_REQ_SHUTDOWN:
                    break
                # request를 처리한 tick에서는 아래 follow_once를 건너뛰고 다음 loop로 간다.
                continue
            except queue.Empty:
                # 처리할 request가 없으면 현재 state에 따라 follow 또는 status polling을 한다.
                pass

            # FOLLOWING 상태에서는 control_hz에 맞춰 target servo를 한 tick 실행한다.
            if self._current_state() == ROBOT_STATE_FOLLOWING:
                self._follow_once()
            # FOLLOWING이 아니어도 연결이 살아 있으면 10Hz 정도로 robot state를 갱신한다.
            elif self.controller is not None and time.time() - self._last_status_read_t >= 0.1:
                self._read_robot_state()

        # loop를 빠져나오면 shutdown event를 확실히 세워 외부 상태와 맞춘다.
        self._shutdown_event.set()

    def _handle_request(self, request):
        # 현재 처리 중인 request를 status에 표시한다.
        self._set_active_request(request)
        try:
            # 로봇 연결과 HOME 이동을 수행한다.
            if request.type == ROBOT_REQ_INIT_ROBOT:
                self._handle_init_robot()
            # perception target을 따라가는 servo follow를 시작한다.
            elif request.type == ROBOT_REQ_START_FOLLOW:
                self._handle_start_follow()
            # follow servo를 정지하고 idle로 돌아간다.
            elif request.type == ROBOT_REQ_STOP_FOLLOW:
                self._handle_stop_follow()
            # pregrasp 도달 후 gripper close, return, place sequence를 수행한다.
            elif request.type == ROBOT_REQ_START_GRASP_PLACE:
                self._handle_start_grasp_place(request.payload)
            # gripper를 열고 HOME으로 돌아간 뒤 새 task 준비 상태로 reset한다.
            elif request.type == ROBOT_REQ_RESET_HOME:
                self._handle_reset_home()
            # 현재 task 저장 흐름을 위해 follow를 멈추고 idle로 둔다.
            elif request.type == ROBOT_REQ_SAVE_AND_STOP:
                self._handle_save_and_stop()
            # 즉시 follow를 멈추고 stop command를 보낸다.
            elif request.type == ROBOT_REQ_EMERGENCY_STOP:
                self._handle_emergency_stop()
            # worker 종료와 controller disconnect를 수행한다.
            elif request.type == ROBOT_REQ_SHUTDOWN:
                self._handle_shutdown()
            # 알 수 없는 request type은 reject하고 status에 에러를 남긴다.
            else:
                self._reject_request(request, f"unknown request type {request.type}")
        finally:
            # handler 성공/실패와 관계없이 active request 표시는 지운다.
            self._clear_active_request()

    def _reject_request(self, request, reason):
        # 현재 state에서 처리할 수 없는 요청을 명시적으로 기록한다.
        message = f"Rejected robot request {request.type}: {reason}"
        print(f"[WARN] {message}")
        self._set_status(last_error=message)

    def _handle_init_robot(self):
        # 이미 controller가 있으면 중복 초기화를 하지 않는다.
        if self.controller is not None:
            self._set_status(last_error=None)
            return
        # 이전 긴급 취소 신호가 남아 있으면 초기화 전에 해제한다.
        self._cancel_event.clear()
        # 초기화 중임을 외부 status에 알린다.
        self._set_status(state=ROBOT_STATE_INITIALIZING, last_error=None, grasp_ok=None)
        try:
            # RTDE controller를 생성하고 로봇/그리퍼 연결을 초기화한다.
            self.controller = init_rtde(self.args)
            # 초기 연결 상태와 TCP pose를 읽어 status에 반영한다.
            self._read_robot_state()
            # task 시작 전 HOME joint pose로 이동한다.
            home_ok = move_robot_to_home_pose(
                self.controller,
                self.args,
                cancel_event=self._cancel_event,
                on_state_read=self._publish_robot_state,
            )
            # HOME 이동이 취소된 경우 idle 상태로 돌아간다.
            if home_ok is False:
                self._set_status(state=ROBOT_STATE_IDLE, last_error="robot initialization cancelled")
                return
            # 현재 로봇 TCP orientation을 follow 중 고정 orientation 기준으로 저장한다.
            self.shared_state.set_fixed_pose_from_robot(self.controller)
            # HOME 이동 후 최신 pose를 다시 읽는다.
            self._read_robot_state()
            # main loop가 metadata/video recording 시작을 준비하도록 task_ready를 publish한다.
            self._publish_task_ready()
            # 초기화가 끝났으므로 idle 상태로 전환한다.
            self._set_status(state=ROBOT_STATE_IDLE, last_error=None)
        except Exception as exc:
            # 초기화 실패는 로봇 제어 state를 ERROR로 전환한다.
            self._set_status(state=ROBOT_STATE_ERROR, last_error=str(exc))
            print(f"[WARN] Robot initialization failed: {exc}")

    def _handle_start_follow(self):
        # follow는 controller 초기화 이후에만 시작할 수 있다.
        if self.controller is None:
            self._reject_request(RobotRequest(ROBOT_REQ_START_FOLLOW), "robot is not initialized")
            return
        # grasp/place/reset/stop 같은 blocking sequence 중에는 follow 시작을 거부한다.
        state = self._current_state()
        if state in {ROBOT_STATE_GRASPING, ROBOT_STATE_RETURNING, ROBOT_STATE_PLACING, ROBOT_STATE_RESETTING, ROBOT_STATE_STOPPING}:
            self._reject_request(RobotRequest(ROBOT_REQ_START_FOLLOW), f"state is {state}")
            return
        # 이전 취소 신호를 해제하고 follow 가능 상태로 shared state를 설정한다.
        self._cancel_event.clear()
        self.shared_state.clear_follow_pause()
        self.shared_state.set_follow_enabled(True, reset_prediction=True, reset_arm=True)
        # UI/debug용 task state를 FOLLOW로 표시한다.
        self.shared_state.set_task_state("FOLLOW", reset_prediction=False, reset_arm=False)
        # 이전 follow 기준점과 마지막 명령을 지운다.
        self._reset_follow_tracking()
        # 다음 loop에서 바로 follow tick이 실행되도록 tick timestamp를 초기화한다.
        self._next_follow_tick_t = 0.0
        # worker state를 FOLLOWING으로 전환한다.
        self._set_status(state=ROBOT_STATE_FOLLOWING, last_error=None, grasp_ok=None)

    def _handle_stop_follow(self):
        # shared state에 follow 중지를 요청해 main/follow 양쪽 상태를 맞춘다.
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        # 실제 servo 명령을 보낸 적이 있다면 로봇에 stop command를 보낸다.
        if self._was_follow_active:
            self._safe_stop(source_mode="stop_follow")
        # follow 내부 기준점을 초기화하고 idle 상태를 publish한다.
        self._reset_follow_tracking()
        # FOLLOWING 상태에서 온 stop 요청이면 worker state를 IDLE로 바꾼다.
        if self._current_state() == ROBOT_STATE_FOLLOWING:
            self._set_status(state=ROBOT_STATE_IDLE)

    def _handle_start_grasp_place(self, payload):
        # grasp/place sequence는 초기화된 controller가 있어야 실행할 수 있다.
        if self.controller is None:
            self._reject_request(RobotRequest(ROBOT_REQ_START_GRASP_PLACE), "robot is not initialized")
            return
        # follow 중 pregrasp에 도달한 순간에만 grasp/place 요청을 받아들인다.
        if self._current_state() != ROBOT_STATE_FOLLOWING:
            self._reject_request(RobotRequest(ROBOT_REQ_START_GRASP_PLACE), f"state is {self._current_state()}")
            return

        # 요청 처리 직전 실제 로봇 pose를 다시 읽어 최종 pregrasp 도달 여부를 확인한다.
        robot_state = self._read_robot_state()
        robot_pose = None if robot_state is None else getattr(robot_state, "actual_tcp_pose_base", None)
        # shared_state 기준 final pose check가 실패하면 grasp를 시작하지 않는다.
        if not self.shared_state.is_pregrasp_pose_reached(robot_pose):
            self._set_status(last_error="grasp request ignored because final pose check failed")
            return

        # main loop가 넘긴 geometry/action context를 RobotActionContext로 정규화한다.
        context = payload.get("context")
        if context is None:
            context = RobotActionContext()
        elif isinstance(context, dict):
            context = RobotActionContext(**context)

        # follow servo에서 grasp/place blocking sequence로 제어권을 넘기기 위해 follow를 멈춘다.
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        self._safe_stop(source_mode="pregrasp_handoff")
        self._reset_follow_tracking()

        # tactile grasp 판정이 있으면 geometry 기반 threshold 대신 config 기본값을 사용한다.
        tactile_enabled = self.tactile_manager is not None and bool(getattr(self.tactile_manager, "enabled", False))
        if tactile_enabled:
            reset_gripper_position_threshold_to_config_default(self.controller)
        else:
            # tactile이 없으면 fitted geometry로 gripper close 완료 threshold를 조정한다.
            configure_gripper_position_threshold_from_geometry(
                self.controller,
                context.fitted_points_base,
                context.grasp_point_base,
                object_label=context.object_label,
                template_axes_base=context.template_axes_base,
            )

        # gripper close 단계로 state를 바꾸고 grasp 판정값을 초기화한다.
        self._set_status(state=ROBOT_STATE_GRASPING, grasp_ok=None, last_error=None)
        # 그리퍼를 닫고 position/tactile 조건으로 grasp 성공 여부를 판정한다.
        grasp_ok = execute_gripper_close(
            self.controller,
            timeout_s=self.args.gripper_close_timeout_s,
            verbose=True,
            metadata_recorder=self.metadata_recorder,
            tactile_manager=self.tactile_manager,
            tactile_contact_threshold=self.args.tactile_contact_norm_threshold,
            tactile_extra_grasp_pos=self.args.tactile_extra_grasp_pos,
            cancel_event=self._cancel_event,
            on_state_read=self._publish_robot_state,
        )
        # grasp 판정 결과를 status에 기록한다.
        self._set_status(grasp_ok=bool(grasp_ok))
        print(f"[INFO] grasp_ok = {grasp_ok}")
        # 도중에 cancel/reset/shutdown이 들어왔으면 로봇을 멈추고 idle로 돌아간다.
        if self._cancel_event.is_set():
            self._safe_stop(source_mode="grasp_cancelled")
            self._set_status(state=ROBOT_STATE_IDLE)
            return
        # grasp 검증이 실패하면 ERROR 상태로 전환하고 place sequence는 실행하지 않는다.
        if not grasp_ok:
            self._set_status(state=ROBOT_STATE_ERROR, last_error="gripper close did not verify grasp")
            return

        # grasp 성공 후 현재 TCP와 object target 간 offset을 저장해 place 계산에 사용한다.
        save_grasp_offset(self.controller, self.shared_state, on_state_read=self._publish_robot_state)
        # offset 저장 직후 취소가 들어왔는지 다시 확인한다.
        if self._cancel_event.is_set():
            self._safe_stop(source_mode="post_grasp_cancelled")
            self._set_status(state=ROBOT_STATE_IDLE)
            return

        # return/place sequence 상태를 외부에 알린다.
        self._set_status(state=ROBOT_STATE_RETURNING)
        self._set_status(state=ROBOT_STATE_PLACING)
        # object를 들고 return pose로 이동한 뒤 place 동작을 실행한다.
        place_ok = execute_return_and_place(
            self.controller,
            self.shared_state,
            self.args,
            metadata_recorder=self.metadata_recorder,
            tactile_manager=self.tactile_manager,
            cancel_event=self._cancel_event,
            on_state_read=self._publish_robot_state,
        )
        # place 중 취소되면 stop 후 idle로 돌아간다.
        if self._cancel_event.is_set():
            self._safe_stop(source_mode="place_cancelled")
            self._set_status(state=ROBOT_STATE_IDLE)
            return
        # return/place 실패는 ERROR로 보고 main loop가 추가 grasp 요청을 막게 한다.
        if not place_ok:
            self._set_status(state=ROBOT_STATE_ERROR, last_error="return/place failed")
            return

        # main loop가 task 완료를 감지하도록 task_done_epoch를 증가시킨다.
        self._bump_status_counter("task_done_epoch")
        # 전체 grasp/place sequence가 정상 종료됐음을 표시한다.
        self._set_status(state=ROBOT_STATE_DONE, last_error=None)

    def _handle_reset_home(self):
        # reset 요청 자체는 새로운 sequence이므로 이전 cancel 신호를 지운다.
        self._cancel_event.clear()
        # reset 진행 중임을 status에 반영한다.
        self._set_status(state=ROBOT_STATE_RESETTING, last_error=None, grasp_ok=None)
        try:
            # follow servo를 먼저 정지해 reset sequence가 로봇 제어권을 갖게 한다.
            self.shared_state.request_follow_pause()
            self.shared_state.stop_follow()
            self._safe_stop(source_mode="reset_home")
            self._reset_follow_tracking()
            # controller가 연결되어 있으면 gripper open과 HOME 이동까지 수행한다.
            if self.controller is not None:
                execute_gripper_open(
                    self.controller,
                    dwell_s=self.args.gripper_release_dwell_s,
                    cancel_event=self._cancel_event,
                )
                # HOME joint pose로 복귀한다.
                home_ok = move_robot_to_home_pose(
                    self.controller,
                    self.args,
                    cancel_event=self._cancel_event,
                    on_state_read=self._publish_robot_state,
                )
                # HOME 이동 취소 시 idle로만 돌아간다.
                if home_ok is False:
                    self._set_status(state=ROBOT_STATE_IDLE, last_error="reset cancelled")
                    return
                # tactile, shared perception/follow 상태를 새 trial 기준으로 초기화한다.
                reset_tactile_state_for_system_reset(self.tactile_manager)
                self.shared_state.reset_for_restart(follow_enabled=self.args.enable_follow)
                # reset 후 현재 robot pose에서 고정 orientation 기준을 다시 잡는다.
                self.shared_state.set_fixed_pose_from_robot(self.controller)
                self.shared_state.clear_follow_pause()
                # 최신 robot pose를 status에 반영한다.
                self._read_robot_state()
                # main loop가 새 metadata/video task를 시작하게 task_ready를 publish한다.
                self._publish_task_ready()
                # enable_follow 설정에 따라 reset 직후 follow를 재개하거나 idle로 둔다.
                self._set_status(state=ROBOT_STATE_FOLLOWING if self.args.enable_follow else ROBOT_STATE_IDLE)
            # controller가 없으면 로봇 동작 없이 software state만 reset한다.
            else:
                reset_tactile_state_for_system_reset(self.tactile_manager)
                self.shared_state.reset_for_restart(follow_enabled=False)
                self._set_status(state=ROBOT_STATE_IDLE)
            # main loop가 perception/debug/profile reset 후처리를 하도록 reset_done_epoch를 증가시킨다.
            self._bump_status_counter("reset_done_epoch")
        except Exception as exc:
            # reset 실패는 안전하게 ERROR 상태로 노출한다.
            self._set_status(state=ROBOT_STATE_ERROR, last_error=str(exc))
            print(f"[WARN] Robot reset failed: {exc}")

    def _handle_save_and_stop(self):
        # 저장 단축키 처리 중에는 follow를 멈추는 STOPPING 상태로 둔다.
        self._set_status(state=ROBOT_STATE_STOPPING)
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        # servo motion이 남아 있을 수 있으므로 stop command를 보낸다.
        self._safe_stop(source_mode="save_and_stop")
        # follow 내부 상태와 cancel 신호를 정리한 뒤 idle로 둔다.
        self._reset_follow_tracking()
        self._cancel_event.clear()
        self._set_status(state=ROBOT_STATE_IDLE)

    def _handle_emergency_stop(self):
        # 긴급 정지는 저장/복귀 없이 즉시 follow를 끊고 stop command를 보낸다.
        self._set_status(state=ROBOT_STATE_STOPPING)
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        self._safe_stop(source_mode="emergency_stop")
        self._reset_follow_tracking()

    def _handle_shutdown(self):
        # 프로그램 종료 시 follow를 멈추고 controller 연결까지 닫는다.
        self._set_status(state=ROBOT_STATE_STOPPING)
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        self._safe_stop(source_mode="shutdown")
        self._reset_follow_tracking()
        self._disconnect()
        self._shutdown_event.set()

    def _follow_once(self):
        # follow servo는 control_hz를 넘지 않도록 tick 간격을 제한한다.
        now = time.time()
        interval = 1.0 / max(float(self.args.control_hz), 1e-6)
        if now < self._next_follow_tick_t:
            return
        self._next_follow_tick_t = now + interval

        # 최신 robot pose와 shared perception target snapshot을 읽는다.
        robot_state = self._read_robot_state()
        snap = self.shared_state.get_snapshot()
        # shared_state가 smoothing/prediction을 거쳐 만든 최종 follow target[mm]이다.
        control_target_xyz_mm = snap["control_target_xyz_mm"]
        # target이 measured인지 predicted인지에 따라 안전 조건이 달라진다.
        target_source = snap["target_source"]

        # 아래 조건 중 하나라도 실패하면 이번 tick에서는 servo 명령을 보내지 않는다.
        active = True
        # grasp/place/reset 등에서 pause를 요청한 경우 follow를 멈춘다.
        if snap["follow_pause_requested"]:
            active = False
        # follow 자체가 꺼져 있으면 비활성화한다.
        elif not snap["follow_enabled"]:
            active = False
        # 따라갈 target이 아직 없으면 비활성화한다.
        elif control_target_xyz_mm is None:
            active = False
        # measured target은 최소 유효 detection streak을 만족하거나 prediction이 arm된 뒤에만 사용한다.
        elif target_source != "predicted" and snap["valid_detection_streak"] < self.args.min_valid_count and not snap["prediction_armed"]:
            active = False
        # predicted target은 prediction이 명시적으로 arm된 경우에만 사용한다.
        elif target_source == "predicted" and not snap["prediction_armed"]:
            active = False
        # hand/object motion trigger 전에는 로봇이 움직이지 않도록 막는다.
        elif not snap["motion_triggered"]:
            active = False
        # prediction target이 너무 오래된 경우 안전하게 follow를 끊는다.
        elif target_source == "predicted" and (
            snap["prediction_age_s"] is None or float(snap["prediction_age_s"]) > float(self.args.prediction_max_horizon_s)
        ):
            active = False
        # elif snap["fixed_z_mm"] is None or snap["fixed_orientation_base"] is None:
        #     active = False
        elif snap["fixed_orientation_base"] is None:
            active = False

        # active 조건을 만족하지 못하면 필요 시 stop을 보내고 idle 상태를 publish한다.
        if not active:
            # 직전 tick까지 servo가 나가고 있었다면 로봇에 stop command를 한 번 보낸다.
            if self._was_follow_active:
                self._safe_stop(source_mode="follow_inactive")
                self._ref_target_xyz_mm = None
                self._last_sent_pose_mm = None
            # 이번 tick부터는 follow 명령이 비활성 상태임을 기록한다.
            self._was_follow_active = False
            self.shared_state.set_follow_thread_idle(True)
            return

        # 이 tick에서는 로봇 servo 제어권을 worker가 사용 중임을 표시한다.
        self.shared_state.set_follow_thread_idle(False)

        # fixed_z_mm = snap["fixed_z_mm"]
        # follow 중 자세는 초기 robot pose에서 잡은 orientation을 고정해서 사용한다.
        fixed_orientation_base = snap["fixed_orientation_base"]
        # reference target이 없으면 현재 TCP 위치를 시작점으로 초기화한다.
        if self._ref_target_xyz_mm is None:
            pose = None if robot_state is None else getattr(robot_state, "actual_tcp_pose_base", None)
            # 현재 robot pose를 읽지 못하면 안전하게 이번 tick을 건너뛴다.
            if pose is None:
                return
            self._ref_target_xyz_mm = meters_to_mm(pose[:3]).astype(np.float32)
            print(f"[INFO] ref_target initialized from current EEF xyz: {self._ref_target_xyz_mm}")

        # control_hz 기준 한 tick에서 허용할 최대 XY/Z 이동량을 계산한다.
        max_step_mm = MAX_XY_SPEED_MM_S / max(float(self.args.control_hz), 1e-6)
        max_step_z_mm = MAX_Z_SPEED_MM_S / max(float(self.args.control_hz), 1e-6)
        # 현재 reference와 perception target 사이의 오차[mm]다.
        ref_err_xyz = control_target_xyz_mm - self._ref_target_xyz_mm
        # close-range에서는 XY 거리와 Z 속도 제한을 반영한 step limit을 얻는다.
        max_step_xy, max_step_z, dist_xy = get_close_range_step_mm(
            ref_err_xyz,
            max_step_mm,
            max_step_z_mm,
        )
        # dominant axis와 follow_z 설정을 고려해 이번 tick에서 이동할 reference step을 계산한다.
        ref_step_xyz, dominant_axis = compute_close_range_ref_step_xyz(
            ref_err_xyz,
            max_step_xy,
            max_step_z,
            follow_z=bool(self.args.follow_z),
        )

        # if not self.args.follow_z:
        #     self._ref_target_xyz_mm[2] = fixed_z_mm

        # reference target을 target 방향으로 한 step 전진시킨다.
        self._ref_target_xyz_mm = self._ref_target_xyz_mm + ref_step_xyz
        # 현재 코드는 follow_z 여부와 관계없이 reference z를 command z로 사용한다.
        cmd_z = float(self._ref_target_xyz_mm[2]) # if self.args.follow_z else float(fixed_z_mm)
        # workspace/safety boundary에 맞춰 command pose를 clamp한다.
        cmd_x, cmd_y, cmd_z = clamp_pose_mm(float(self._ref_target_xyz_mm[0]), float(self._ref_target_xyz_mm[1]), cmd_z, self.args)
        # robot command helper에 넘기기 위한 xyz[mm] 배열이다.
        pose_mm = np.array([cmd_x, cmd_y, cmd_z], dtype=np.float32)

        # 이전에 보낸 pose와 거의 같으면 불필요한 servo command를 보내지 않는다.
        if self._last_sent_pose_mm is not None:
            pos_delta = np.linalg.norm(pose_mm - self._last_sent_pose_mm)
            if pos_delta < 0.2:
                return

        # RTDE 명령은 meter 단위를 사용하므로 mm target을 m tuple로 변환한다.
        target_position_base = mm_to_m_tuple(pose_mm)
        # servo command 전송 실패가 worker thread 종료로 이어지지 않도록 보호한다.
        try:
            self._send_robot_command(
                ROBOT_CMD_SERVO_TO_POSITION,
                target_position_base=target_position_base,
                fixed_orientation_base=fixed_orientation_base,
                gripper_action=GRIPPER_HOLD,
                source_mode="follow_servo",
            )
            # servo 명령이 성공했음을 기록해 비활성화 시 stop을 보낼 수 있게 한다.
            self._was_follow_active = True
            # 다음 tick에서 중복 명령을 줄이기 위해 마지막 command pose를 저장한다.
            self._last_sent_pose_mm = pose_mm
            # if self.args.verbose_robot:
            #     print(
            #         f"[ROBOT] source={target_source}, "
            #         f"raw_mm={snap['latest_target_xyz_mm']}, "
            #         f"control_mm={control_target_xyz_mm}, "
            #         f"ref_mm={self._ref_target_xyz_mm}, "
            #         f"close_range_dist_xy={dist_xy:.1f}, "
            #         f"stage_axis={'xy' if dominant_axis is None else ('x' if dominant_axis == 0 else 'y')}, "
            #         f"cmd_m={target_position_base}"
            #     )
        except Exception as exc:
            # servo 전송 실패를 status와 콘솔 로그에 남긴다.
            self._set_status(last_error=str(exc))
            print(f"[WARN] servo command failed: {exc}")


def compute_tcp_speed_norms(state):
    """Return linear/angular TCP speed norms from a RobotState-like object."""
    speed = getattr(state, "actual_tcp_speed", None)
    if speed is None or len(speed) < 3:
        return None, None
    try:
        linear_values = np.asarray(speed[:3], dtype=np.float64)
        linear_norm = float(np.linalg.norm(linear_values))
        angular_norm = None
        if len(speed) >= 6:
            angular_values = np.asarray(speed[3:6], dtype=np.float64)
            angular_norm = float(np.linalg.norm(angular_values))
        return linear_norm, angular_norm
    except Exception:
        return None, None


def read_rtde_is_steady(controller):
    """Return RTDE isSteady() when available; otherwise None."""
    rtde_control = getattr(controller, "_rtde_control", None)
    is_steady = getattr(rtde_control, "isSteady", None)
    if not callable(is_steady):
        return None
    try:
        return bool(is_steady())
    except Exception:
        return None


def wait_until_robot_stopped(
    controller,
    *,
    timeout_s,
    speed_threshold_mps=DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS,
    poll_dt=DEFAULT_POST_BACKOFF_STOP_POLL_DT_S,
    cancel_event=None,
    on_state_read=None,
    source_mode="robot_stop_wait",
):
    """Poll robot state until TCP speed or RTDE steady status indicates a stop."""
    del source_mode
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_rtde(controller)
        return False
    if not hasattr(controller, "read_robot_state"):
        return False

    deadline = time.time() + max(0.0, float(timeout_s))
    threshold = max(0.0, float(speed_threshold_mps))
    poll_dt = max(0.0, float(poll_dt))

    while True:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_rtde(controller)
            return False
        try:
            state = controller.read_robot_state(now_timestamp=time.time())
            if on_state_read is not None:
                on_state_read(state)
        except Exception:
            state = None

        if state is not None:
            linear_speed, _angular_speed = compute_tcp_speed_norms(state)
            if linear_speed is not None and linear_speed <= threshold:
                return True

        is_steady = read_rtde_is_steady(controller)
        if is_steady is True:
            return True

        if time.time() >= deadline:
            return False
        time.sleep(poll_dt)


def wait_until_target_reached(
    controller,
    target_position_base,
    *,
    timeout_s,
    tolerance_m,
    poll_dt=0.05,
    cancel_event=None,
    on_state_read=None,
):
    """Poll the robot pose until a blocking move reaches its target or times out."""
    deadline = time.time() + timeout_s
    target = np.asarray(target_position_base, dtype=np.float32).reshape(3)

    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_rtde(controller)
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        pose = state.actual_tcp_pose_base
        if pose is not None:
            current = np.asarray(pose[:3], dtype=np.float32).reshape(3)
            error = float(np.linalg.norm(current - target))
            if error <= tolerance_m:
                return True
        time.sleep(poll_dt)
    return False


def wait_until_joint_target_reached(
    controller,
    target_joints_rad,
    *,
    timeout_s,
    tolerance_rad,
    poll_dt=0.05,
    cancel_event=None,
    on_state_read=None,
):
    """Poll the robot joints until the target is reached or the move times out."""
    deadline = time.time() + timeout_s
    target = np.asarray(target_joints_rad, dtype=np.float64).reshape(6)

    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            if hasattr(controller, "stop_joint_motion"):
                controller.stop_joint_motion()
            else:
                safe_stop_rtde(controller)
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        joints = getattr(state, "joint_positions", None)
        if joints is not None:
            current = np.asarray(joints, dtype=np.float64).reshape(6)
            if np.all(np.isfinite(current)):
                max_error = float(np.max(np.abs(current - target)))
                if max_error <= tolerance_rad:
                    return True
        time.sleep(poll_dt)
    return False


def move_robot_and_wait(
    controller,
    target_position_base,
    fixed_orientation_base,
    *,
    timeout_s,
    tolerance_m,
    source_mode,
    cancel_event=None,
    on_state_read=None,
):
    """Issue one blocking position move and wait for completion."""
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_rtde(controller)
        return False
    send_robot_command(
        controller,
        ROBOT_CMD_MOVE_TO_POSITION,
        target_position_base=target_position_base,
        fixed_orientation_base=fixed_orientation_base,
        gripper_action=GRIPPER_HOLD,
        source_mode=source_mode,
    )
    return wait_until_target_reached(
        controller,
        target_position_base,
        timeout_s=timeout_s,
        tolerance_m=tolerance_m,
        cancel_event=cancel_event,
        on_state_read=on_state_read,
    )


def stop_follow_for_handoff(shared_state, timeout_s):
    """Pause the follow thread before the main thread takes over for grasp/place."""
    shared_state.request_follow_pause()
    released = shared_state.wait_for_follow_idle(timeout_s)
    if not released:
        print(f"[WARN] Follow thread did not release robot control within {timeout_s:.2f}s")
        return False
    print("[INFO] Follow thread released robot control for pregrasp handoff")
    return True


def move_robot_to_home_pose(controller, args, cancel_event=None, on_state_read=None):
    """Move the robot to the configured HOME joint target before or between tasks."""
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_rtde(controller)
        return False
    home_joints_rad = get_home_joints_rad()
    home_text = ", ".join(f"{value:.1f}" for value in HOME_JOINTS_DEG)
    print(
        "[INFO] Moving robot to HOME joints: "
        f"deg=[{home_text}], speed={HOME_JOINT_SPEED_RAD_S:.2f} rad/s, "
        f"accel={HOME_JOINT_ACCELERATION_RAD_S2:.2f} rad/s^2"
    )
    if not hasattr(controller, "move_to_joint_positions"):
        raise RuntimeError("RTDE controller does not support joint HOME moves.")

    started = bool(
        controller.move_to_joint_positions(
            home_joints_rad,
            speed_rad_s=HOME_JOINT_SPEED_RAD_S,
            acceleration_rad_s2=HOME_JOINT_ACCELERATION_RAD_S2,
            async_move=True,
        )
    )
    if not started:
        raise RuntimeError("Failed to start HOME joint move.")

    ok = wait_until_joint_target_reached(
        controller,
        home_joints_rad,
        timeout_s=args.move_timeout_s,
        tolerance_rad=float(np.deg2rad(HOME_JOINT_TOLERANCE_DEG)),
        cancel_event=cancel_event,
        on_state_read=on_state_read,
    )
    if not ok:
        if hasattr(controller, "stop_joint_motion"):
            controller.stop_joint_motion()
        else:
            safe_stop_rtde(controller)
        raise RuntimeError("Failed to reach HOME joints during startup.")
    print("[INFO] HOME joints reached")
    return True


def resolve_default_gripper_position_threshold(gripper_cfg):
    """Resolve the fallback gripper close completion threshold."""
    gripper_cfg = dict(gripper_cfg or {})
    return int(
        gripper_cfg.get("position_complete_threshold", DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD)
    )


def resolve_gripper_position_stall_detection_config(controller):
    """Resolve gripper position stall fallback settings from controller config."""
    config = getattr(controller, "config", {}) or {}
    gripper_cfg = dict(config.get("robot", {}).get("gripper", {}) or {})
    stall_cfg = dict(gripper_cfg.get("position_stall_detection", {}) or {})
    return {
        "enabled": bool(stall_cfg.get("enabled", DEFAULT_GRIPPER_POSITION_STALL_ENABLED)),
        "stable_reads_required": max(
            1,
            int(stall_cfg.get("stable_reads_required", DEFAULT_GRIPPER_POSITION_STALL_STABLE_READS)),
        ),
        "tolerance": max(
            0,
            int(stall_cfg.get("tolerance", DEFAULT_GRIPPER_POSITION_STALL_TOLERANCE)),
        ),
        "min_elapsed_s": max(
            0.0,
            float(stall_cfg.get("min_elapsed_s", DEFAULT_GRIPPER_POSITION_STALL_MIN_ELAPSED_S)),
        ),
    }


def _gripper_geometry_cfg(gripper_cfg):
    gripper_cfg = dict(gripper_cfg or {})
    geometry_cfg = gripper_cfg.get("position_threshold_geometry", {})
    return dict(geometry_cfg) if isinstance(geometry_cfg, dict) else {}


def _estimate_xy_max_diameter_m(points_xy, max_points=128):
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape((-1, 2))
    if len(points_xy) == 0:
        return None
    if len(points_xy) > int(max_points):
        sample_indices = np.linspace(0, len(points_xy) - 1, int(max_points), dtype=np.int32)
        points_xy = points_xy[sample_indices]
    if len(points_xy) == 1:
        return 0.0
    deltas = points_xy[:, None, :] - points_xy[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    return float(np.max(distances))


def _estimate_pca_section_widths_m(points_xy):
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape((-1, 2))
    if len(points_xy) < 2:
        return None
    centered = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    if not np.all(np.isfinite(covariance)):
        return None
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    projected = centered @ basis
    extents = np.ptp(projected, axis=0)
    if len(extents) < 2 or not np.all(np.isfinite(extents)):
        return None
    long_width = float(np.max(extents))
    short_width = float(np.min(extents))
    return short_width, long_width


def _normalize_gripper_label(label):
    return "" if label is None else str(label).strip().lower()


def _local_width_axis_for_label(geometry_cfg, label):
    axis_by_label = geometry_cfg.get("local_width_axis_by_label", {})
    if not isinstance(axis_by_label, dict):
        return None

    normalized_label = _normalize_gripper_label(label)
    for configured_label, configured_axis in axis_by_label.items():
        if _normalize_gripper_label(configured_label) == normalized_label:
            axis = str(configured_axis).strip().lower()
            return axis if axis in {"x", "y", "z"} else None
    return None


def _estimate_local_axis_width_m(slice_points, template_axes_base, axis_name):
    points = np.asarray(slice_points, dtype=np.float64).reshape((-1, 3))
    try:
        axes = np.asarray(template_axes_base, dtype=np.float64).reshape((3, 3))
    except Exception:
        return None
    axis_index = {"x": 0, "y": 1, "z": 2}.get(str(axis_name).strip().lower())
    if axis_index is None or len(points) == 0 or not np.all(np.isfinite(axes)):
        return None

    axis = axes[axis_index]
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    axis = axis / norm
    projected = points @ axis.reshape(3, 1)
    projected = projected.reshape(-1)
    if len(projected) == 0 or not np.all(np.isfinite(projected)):
        return None
    return float(np.max(projected) - np.min(projected))


def _select_template_slice_points(fitted_points_base, grasp_point_base, geometry_cfg):
    if fitted_points_base is None:
        return None, "missing_geometry"
    points = np.asarray(fitted_points_base, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0 or grasp_point_base is None:
        return None, "missing_geometry"

    grasp = np.asarray(grasp_point_base, dtype=np.float64).reshape(-1)
    if len(grasp) < 3 or not np.isfinite(grasp[:3]).all():
        return None, "missing_grasp"
    z_values = points[:, 2]
    finite_mask = np.isfinite(z_values) & np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite_mask]
    if len(points) == 0:
        return None, "missing_geometry"

    grasp_z = float(grasp[2])
    slice_band_m = max(float(geometry_cfg.get("slice_band_m", 0.005)), 0.0)
    min_slice_points = max(int(geometry_cfg.get("min_slice_points", 8)), 1)
    half_band_m = 0.5 * slice_band_m
    if half_band_m > 0.0:
        z_delta = np.abs(points[:, 2] - grasp_z)
        slice_points = points[z_delta <= half_band_m]
    else:
        slice_points = np.empty((0, 3), dtype=np.float64)

    if len(slice_points) >= min_slice_points:
        return slice_points, "band"

    fallback_count = max(int(geometry_cfg.get("fallback_nearest_points", 64)), min_slice_points)
    nearest_count = min(fallback_count, len(points))
    if nearest_count <= 0:
        return None, "missing_geometry"
    nearest_indices = np.argsort(np.abs(points[:, 2] - grasp_z))[:nearest_count]
    nearest_points = points[nearest_indices]
    if len(nearest_points) < min_slice_points:
        return None, "insufficient_slice_points"
    return nearest_points, "nearest_z"


def estimate_gripper_template_width_cm(
    fitted_points_base,
    grasp_point_base,
    gripper_cfg,
    *,
    object_label=None,
    template_axes_base=None,
):
    """Estimate template width/diameter at the grasp z position in centimeters."""
    geometry_cfg = _gripper_geometry_cfg(gripper_cfg)
    slice_points, slice_source = _select_template_slice_points(
        fitted_points_base,
        grasp_point_base,
        geometry_cfg,
    )
    if slice_points is None:
        return None, {
            "reason": slice_source,
            "slice_source": slice_source,
            "slice_point_count": 0,
        }

    local_axis = _local_width_axis_for_label(geometry_cfg, object_label)
    if local_axis is not None:
        if template_axes_base is None:
            return None, {
                "reason": "missing_template_axes",
                "slice_source": slice_source,
                "slice_point_count": int(len(slice_points)),
                "width_mode": f"local_axis_{local_axis}",
                "label": _normalize_gripper_label(object_label) or None,
            }
        width_m = _estimate_local_axis_width_m(slice_points, template_axes_base, local_axis)
        if width_m is None or not np.isfinite(width_m):
            return None, {
                "reason": "invalid_local_axis_width",
                "slice_source": slice_source,
                "slice_point_count": int(len(slice_points)),
                "width_mode": f"local_axis_{local_axis}",
                "label": _normalize_gripper_label(object_label) or None,
            }
        width_cm = float(width_m * 100.0)
        return width_cm, {
            "reason": "ok",
            "slice_source": slice_source,
            "slice_point_count": int(len(slice_points)),
            "width_mode": f"local_axis_{local_axis}",
            "local_axis": local_axis,
            "label": _normalize_gripper_label(object_label) or None,
            "width_cm": width_cm,
        }

    points_xy = slice_points[:, :2]
    diameter_m = _estimate_xy_max_diameter_m(points_xy)
    pca_widths = _estimate_pca_section_widths_m(points_xy)
    width_mode = "diameter"
    width_m = diameter_m
    anisotropic_ratio = max(float(geometry_cfg.get("anisotropic_ratio", 1.25)), 1.0)
    if pca_widths is not None:
        short_width_m, long_width_m = pca_widths
        if short_width_m > 1e-9 and long_width_m / short_width_m >= anisotropic_ratio:
            width_m = short_width_m
            width_mode = "pca_short_axis"

    if width_m is None or not np.isfinite(width_m):
        return None, {
            "reason": "invalid_width",
            "slice_source": slice_source,
            "slice_point_count": int(len(slice_points)),
        }

    width_cm = float(width_m * 100.0)
    return width_cm, {
        "reason": "ok",
        "slice_source": slice_source,
        "slice_point_count": int(len(slice_points)),
        "width_mode": width_mode,
        "width_cm": width_cm,
    }


def resolve_gripper_position_threshold_from_geometry(
    gripper_cfg,
    fitted_points_base,
    grasp_point_base,
    *,
    object_label=None,
    template_axes_base=None,
):
    """Resolve gripper close completion threshold from fitted template geometry."""
    gripper_cfg = dict(gripper_cfg or {})
    default_threshold = resolve_default_gripper_position_threshold(gripper_cfg)
    geometry_cfg = _gripper_geometry_cfg(gripper_cfg)
    if not bool(geometry_cfg.get("enabled", True)):
        return default_threshold, {
            "reason": "disabled",
            "threshold_float": float(default_threshold),
            "threshold": int(default_threshold),
            "fallback_threshold": int(default_threshold),
        }

    width_cm, debug = estimate_gripper_template_width_cm(
        fitted_points_base,
        grasp_point_base,
        gripper_cfg,
        object_label=object_label,
        template_axes_base=template_axes_base,
    )
    if width_cm is None:
        debug.update(
            {
                "threshold_float": float(default_threshold),
                "threshold": int(default_threshold),
                "fallback_threshold": int(default_threshold),
            }
        )
        return default_threshold, debug

    formula_opening_cm = float(geometry_cfg.get("formula_opening_cm", 9.0))
    if not np.isfinite(formula_opening_cm) or formula_opening_cm <= 1e-9:
        formula_opening_cm = 9.0
    alpha_cm = float(geometry_cfg.get("alpha_cm", 0.1))
    threshold_float = (formula_opening_cm - (float(width_cm) + alpha_cm)) * 255.0 / formula_opening_cm
    clamp_min = float(geometry_cfg.get("clamp_min", 0.0))
    clamp_max = float(geometry_cfg.get("clamp_max", 255.0))
    if clamp_min > clamp_max:
        clamp_min, clamp_max = clamp_max, clamp_min
    threshold_clamped = float(np.clip(threshold_float, clamp_min, clamp_max))
    threshold = int(np.floor(threshold_clamped + 0.5 + 1e-9))
    debug.update(
        {
            "alpha_cm": alpha_cm,
            "formula_opening_cm": formula_opening_cm,
            "threshold_float": threshold_float,
            "threshold_clamped_float": threshold_clamped,
            "threshold": threshold,
            "fallback_threshold": int(default_threshold),
        }
    )
    return threshold, debug


def configure_gripper_position_threshold_from_geometry(
    controller,
    fitted_points_base,
    grasp_point_base,
    *,
    object_label=None,
    template_axes_base=None,
):
    """Select a gripper completion threshold from the fitted template grasp section."""
    if controller is None:
        return None

    controller_config = getattr(controller, "config", {}) or {}
    gripper_cfg = (
        controller_config
        .get("robot", {})
        .get("gripper", {})
    )
    position_threshold, debug = resolve_gripper_position_threshold_from_geometry(
        gripper_cfg,
        fitted_points_base,
        grasp_point_base,
        object_label=object_label,
        template_axes_base=template_axes_base,
    )

    controller.gripper_position_complete_threshold = int(position_threshold)
    reason = debug.get("reason", "unknown")
    width_cm = debug.get("width_cm")
    width_text = "n/a" if width_cm is None else f"{float(width_cm):.3f}cm"
    threshold_float = float(debug.get("threshold_float", position_threshold))
    threshold_clamped = float(debug.get("threshold_clamped_float", position_threshold))
    print(
        "[INFO] Gripper position threshold configured from geometry: "
        f"threshold_float={threshold_float:.3f}, "
        f"threshold_clamped={threshold_clamped:.3f}, "
        f"threshold={int(position_threshold)}, "
        f"width={width_text}, "
        f"mode={debug.get('width_mode', 'fallback')}, "
        f"label={debug.get('label', _normalize_gripper_label(object_label) or 'unknown')}, "
        f"slice_source={debug.get('slice_source', 'n/a')}, "
        f"slice_points={int(debug.get('slice_point_count', 0))}, "
        f"reason={reason}"
    )
    return int(position_threshold)


def reset_gripper_position_threshold_to_config_default(controller):
    """Use the static gripper threshold fallback without vision/template geometry."""
    if controller is None:
        return None

    controller_config = getattr(controller, "config", {}) or {}
    gripper_cfg = (
        controller_config
        .get("robot", {})
        .get("gripper", {})
    )
    position_threshold = int(
        gripper_cfg.get(
            "position_complete_threshold",
            DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD,
        )
    )
    controller.gripper_position_complete_threshold = int(position_threshold)
    print(
        "[INFO] Tactile mode enabled; skipping vision/template gripper threshold geometry. "
        f"Using static fallback position threshold={int(position_threshold)}."
    )
    return int(position_threshold)


def start_task_video_recording(video_recorder, task_ready_timestamp):
    if video_recorder is None:
        return
    ok, message = video_recorder.start_recording_for_task(task_start_timestamp_iso=task_ready_timestamp)
    level = "[INFO]" if ok else "[WARN]"
    print(f"{level} Video recorder: {message}")


def open_video_recorder_ui(video_recorder):
    if video_recorder is None:
        return
    opened, url = video_recorder.open_browser()
    if opened:
        print(f"[INFO] Video recorder UI opened: {url}")
    else:
        print(f"[INFO] Open the video recorder UI to finalize save: {url}")


def discard_video_recording_for_reset(video_recorder):
    if video_recorder is None:
        return
    ok, message = video_recorder.discard_pending_recording()
    if ok:
        print(f"[INFO] Video recorder reset discard: {message}")
    elif message != "nothing to discard":
        print(f"[WARN] Video recorder reset discard failed: {message}")
    else:
        print("[INFO] Video recorder reset discard: nothing to discard")

    try:
        status = video_recorder.get_status()
    except Exception as exc:
        print(f"[WARN] Video recorder status check failed after reset: {exc}")
        return

    has_live_frame = bool(status.get("has_live_frame"))
    last_frame_age_ms = status.get("last_frame_age_ms")
    if has_live_frame:
        age_text = "-" if last_frame_age_ms is None else f"{int(last_frame_age_ms)} ms"
        print(f"[INFO] Video recorder kept live stream active after reset (last_frame_age={age_text}).")
    else:
        print("[WARN] Video recorder has no live frame after reset; next task recording will wait for the next frame.")


def reset_tactile_state_for_system_reset(tactile_manager):
    if tactile_manager is None or not bool(getattr(tactile_manager, "enabled", False)):
        return False

    if hasattr(tactile_manager, "clear_runtime_state"):
        tactile_manager.clear_runtime_state(release_status="reset_cleared")
    else:
        num_mags = max(1, int(getattr(tactile_manager, "num_mags", DEFAULT_TACTILE_NUM_MAGS)))
        tactile_manager.release_reference_norm = None
        tactile_manager.release_delta_norm = None
        tactile_manager.release_status = "reset_cleared"
        tactile_manager.latest = np.zeros(num_mags * 3, dtype=np.float32)
        tactile_manager.latest_norm = 0.0
        tactile_manager.last_error = None

    baseline_reset = False
    if hasattr(tactile_manager, "reset_baseline"):
        baseline_reset = bool(tactile_manager.reset_baseline())

    if hasattr(tactile_manager, "clear_runtime_state"):
        tactile_manager.clear_runtime_state(release_status="reset_ready" if baseline_reset else "reset_cleared")
    else:
        tactile_manager.release_reference_norm = None
        tactile_manager.release_delta_norm = None
        tactile_manager.release_status = "reset_ready" if baseline_reset else "reset_cleared"
    print(
        "[Tactile] Reset state for system reset: "
        f"baseline_reset={baseline_reset}, ref cleared, delta cleared."
    )
    return baseline_reset


def reset_perception_pipeline_for_system_reset(pipeline):
    """Reset perception-only state after the robot worker finishes a reset."""
    if pipeline is None:
        return

    shape_fitting_tracker = pipeline.get("shape_fitting_tracker")
    if shape_fitting_tracker is not None and hasattr(shape_fitting_tracker, "reset"):
        shape_fitting_tracker.reset()
        print("[INFO] Shape fitting tracker reset. ICP initialization will restart.")

    object_merger = pipeline.get("object_merger")
    if object_merger is not None and hasattr(object_merger, "reset_initial_centroid"):
        object_merger.reset_initial_centroid()

    for object_worker_key in ("object_worker_cam0", "object_worker_cam1"):
        object_worker = pipeline.get(object_worker_key)
        if object_worker is not None and hasattr(object_worker, "reset"):
            object_worker.reset()
    print("[INFO] Object workers reset. Temporal class locks will be reacquired.")

    for hand_worker_key in ("hand_worker_cam0", "hand_worker_cam1"):
        hand_worker = pipeline.get(hand_worker_key)
        if hand_worker is not None and hasattr(hand_worker, "reset"):
            hand_worker.reset()
    print("[INFO] Hand workers reset. MediaPipe tracking and hand smoothing will restart.")

    hand_selector = pipeline.get("hand_selector")
    if hand_selector is not None and hasattr(hand_selector, "reset"):
        hand_selector.reset()
        print("[INFO] Hand selector reset. Locked hand camera will be reacquired.")

    fusion = pipeline.get("fusion")
    if fusion is not None and hasattr(fusion, "reset"):
        fusion.reset()
        print("[INFO] Perception fusion reset. Filtered hand/object centers cleared.")

    hand_relative_fallback = pipeline.get("hand_relative_fallback")
    if hand_relative_fallback is not None and hasattr(hand_relative_fallback, "reset"):
        hand_relative_fallback.reset()


def reset_system_to_start_state(
    controller,
    shared_state,
    args,
    metadata_recorder=None,
    pipeline=None,
    video_recorder=None,
    tactile_manager=None,
):
    """Reset perception, robot pose, metadata, and recorder state for a new task."""
    print("[INFO] Reset requested: returning to startup state")
    task_start_perf = None
    discard_video_recording_for_reset(video_recorder)
    reset_perception_pipeline_for_system_reset(pipeline)

    if controller is None:
        reset_tactile_state_for_system_reset(tactile_manager)
        shared_state.reset_for_restart(follow_enabled=False)
        return

    shared_state.request_follow_pause()
    shared_state.stop_follow()
    shared_state.wait_for_follow_idle(args.follow_handoff_timeout_s)
    safe_stop_rtde(controller)

    execute_gripper_open(controller, dwell_s=args.gripper_release_dwell_s)
    move_robot_to_home_pose(controller, args)
    reset_tactile_state_for_system_reset(tactile_manager)

    shared_state.reset_for_restart(follow_enabled=args.enable_follow)
    shared_state.set_fixed_pose_from_robot(controller)
    if metadata_recorder is not None:
        task_ready_timestamp = metadata_recorder.mark_task_ready(shared_state)
        task_start_perf = time.perf_counter()
        print(f"[INFO] Metadata task start timestamp={task_ready_timestamp}")
        start_task_video_recording(video_recorder, task_ready_timestamp)
    shared_state.clear_follow_pause()
    print("[INFO] Reset complete. System is back at startup state.")
    return task_start_perf


def stop_gripper_motion_safely(controller, reason):
    if not hasattr(controller, "stop_gripper_motion"):
        return
    try:
        controller.stop_gripper_motion()
    except Exception as exc:
        print(f"[WARN] Failed to stop gripper after {reason}: {exc}")


def note_robot_first_contact(metadata_recorder):
    if metadata_recorder is None:
        return
    first_contact_timestamp = metadata_recorder.note_robot_first_contact()
    if first_contact_timestamp is not None:
        print(f"[INFO] Robot first contact timestamp={first_contact_timestamp}")


def print_tactile_frame_log(tactile_manager, *, stage, stage_time_s=None, **fields):
    if tactile_manager is None or not bool(getattr(tactile_manager, "enabled", False)):
        return

    num_mags = max(1, int(getattr(tactile_manager, "num_mags", DEFAULT_TACTILE_NUM_MAGS)))
    tactile_snapshot = None
    if hasattr(tactile_manager, "snapshot"):
        tactile_snapshot = tactile_manager.snapshot(refresh=False)
    tactile_values = getattr(tactile_manager, "latest", []) if tactile_snapshot is None else tactile_snapshot.get("values", [])
    tactile_data = np.asarray(tactile_values, dtype=np.float32).flatten()
    expected_values = num_mags * 3
    if tactile_data.size < expected_values:
        padded = np.zeros((expected_values,), dtype=np.float32)
        padded[: tactile_data.size] = tactile_data
        tactile_data = padded
    tactile_data = tactile_data[:expected_values].reshape(num_mags, 3)

    time_text = "-" if stage_time_s is None else f"{float(stage_time_s):.3f}"
    ref_norm = getattr(tactile_manager, "release_reference_norm", None) if tactile_snapshot is None else tactile_snapshot.get("release_ref_norm")
    ref_text = "-" if ref_norm is None else f"{float(ref_norm):.3f}"
    delta_norm = getattr(tactile_manager, "release_delta_norm", None) if tactile_snapshot is None else tactile_snapshot.get("release_delta_norm")
    delta_text = "-" if delta_norm is None else f"{float(delta_norm):.3f}"
    status = str(getattr(tactile_manager, "release_status", "off") if tactile_snapshot is None else tactile_snapshot.get("status", "off"))
    norm = float(getattr(tactile_manager, "latest_norm", 0.0) if tactile_snapshot is None else tactile_snapshot.get("total_norm", 0.0))
    field_text = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    # mag_text = " ".join(
    #     f"M{idx}=({values[0]:.3f},{values[1]:.3f},{values[2]:.3f})"
    #     for idx, values in enumerate(tactile_data)
    # )
    print(
        "[TACTILE_FRAME] "
        f"stage={stage} stage_time_s={time_text} norm={norm:.3f} ref={ref_text} "
        f"delta={delta_text} status={status} {field_text}",
        flush=True,
    )


def execute_gripper_close(
    controller,
    timeout_s=2.0,
    poll_dt=0.05,
    verbose=True,
    metadata_recorder=None,
    tactile_manager=None,
    tactile_contact_threshold=None,
    tactile_extra_grasp_pos=None,
    cancel_event=None,
    on_state_read=None,
):
    """Close the gripper and stop when force or position indicates contact."""
    if verbose:
        print("[INFO] GRIPPER CLOSE start")

    if cancel_event is not None and cancel_event.is_set():
        stop_gripper_motion_safely(controller, "cancel before close")
        return False

    baseline_state = controller.read_robot_state(now_timestamp=time.time())
    if on_state_read is not None:
        on_state_read(baseline_state)
    baseline_force_norm = baseline_state.tcp_force_norm_n
    if verbose:
        print(f"[INFO] Pre-close baseline force_norm={baseline_force_norm}")

    close_started = False
    if hasattr(controller, "start_gripper_close"):
        try:
            close_started = bool(controller.start_gripper_close())
        except Exception as exc:
            print(f"[WARN] Failed to start async gripper close: {exc}")
            close_started = False

    if not close_started:
        send_robot_command(
            controller,
            ROBOT_CMD_HOLD,
            gripper_action=GRIPPER_CLOSE,
            source_mode="gripper_close",
        )

    deadline = time.time() + timeout_s
    loop_start = time.time()
    force_threshold = float(getattr(controller, "min_tcp_force_norm_n", 8.0))
    stall_cfg = resolve_gripper_position_stall_detection_config(controller)
    last_position_value = None
    stable_position_reads = 0
    tactile_enabled = tactile_manager is not None and bool(getattr(tactile_manager, "enabled", False))
    tactile_contact_threshold = (
        DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD
        if tactile_contact_threshold is None
        else float(tactile_contact_threshold)
    )
    tactile_extra_grasp_pos = (
        DEFAULT_TACTILE_EXTRA_GRASP_POS
        if tactile_extra_grasp_pos is None
        else max(0, int(tactile_extra_grasp_pos))
    )
    tactile_contact_detected = False
    tactile_contact_position = None
    tactile_extra_target_position = None
    if tactile_enabled:
        tactile_manager.release_status = "close_monitoring"

    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            stop_gripper_motion_safely(controller, "cancel")
            if tactile_enabled:
                tactile_manager.release_status = "close_cancelled"
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        force_norm = state.tcp_force_norm_n
        elapsed = time.time() - loop_start
        force_delta = None
        if baseline_force_norm is not None and force_norm is not None:
            force_delta = float(force_norm) - float(baseline_force_norm)

        force_triggered = False
        if elapsed >= GRIPPER_FORCE_STOP_MIN_ELAPSED_S and force_norm is not None:
            if baseline_force_norm is None:
                force_triggered = float(force_norm) >= force_threshold
            else:
                force_triggered = (
                    float(force_norm) >= force_threshold
                    and force_delta is not None
                    and force_delta >= GRIPPER_FORCE_STOP_DELTA_N
                )

        gripper_close_state = None
        position_triggered = False
        position_stall_triggered = False
        position_value_int = None
        closed_position_int = 255
        if hasattr(controller, "get_gripper_close_state"):
            try:
                gripper_close_state = controller.get_gripper_close_state()
                if gripper_close_state is not None:
                    position_value = gripper_close_state.get("position")
                    closed_position = gripper_close_state.get("closed_position")
                    if closed_position is not None:
                        closed_position_int = int(closed_position)
                    if position_value is not None:
                        position_value_int = int(position_value)
                        position_threshold = int(getattr(controller, "gripper_position_complete_threshold", DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD))
                        position_triggered = position_value_int >= position_threshold
                        if last_position_value is None:
                            stable_position_reads = 1
                        elif abs(position_value_int - last_position_value) <= stall_cfg["tolerance"]:
                            stable_position_reads += 1
                        else:
                            stable_position_reads = 1
                        last_position_value = position_value_int
                        position_stall_triggered = (
                            bool(stall_cfg["enabled"])
                            and elapsed >= stall_cfg["min_elapsed_s"]
                            and stable_position_reads >= stall_cfg["stable_reads_required"]
                        )
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Failed to read gripper close state: {exc}")

        tactile_norm = None
        tactile_triggered = False
        tactile_extra_done = False
        if tactile_enabled:
            tactile_norm = tactile_manager.total_norm()
            print_tactile_frame_log(
                tactile_manager,
                stage="gripper_close",
                stage_time_s=elapsed,
                gripper_pos=position_value_int,
            )
            if not tactile_contact_detected and tactile_norm >= tactile_contact_threshold:
                tactile_contact_detected = True
                tactile_triggered = True
                tactile_contact_position = position_value_int
                if position_value_int is None:
                    stop_gripper_motion_safely(controller, "tactile trigger without gripper position")
                    note_robot_first_contact(metadata_recorder)
                    tactile_manager.release_status = "close_tactile_stop"
                    print(
                        "[INFO] Tactile contact detected during close "
                        f"(norm={tactile_norm:.3f} >= {tactile_contact_threshold:.3f}); "
                        "gripper position unavailable, stopping close."
                    )
                    return True

                tactile_extra_target_position = min(
                    int(position_value_int) + int(tactile_extra_grasp_pos),
                    int(closed_position_int),
                )
                tactile_manager.release_status = "close_extra"
                print(
                    "[INFO] Tactile contact detected during close: "
                    f"norm={tactile_norm:.3f}, contact_pos={position_value_int}, "
                    f"extra={tactile_extra_grasp_pos}, target_pos={tactile_extra_target_position}."
                )

            if tactile_contact_detected and tactile_extra_target_position is not None:
                tactile_extra_done = (
                    position_value_int is not None
                    and int(position_value_int) >= int(tactile_extra_target_position)
                )
                if tactile_extra_done:
                    stop_gripper_motion_safely(controller, "tactile extra close")
                    note_robot_first_contact(metadata_recorder)
                    tactile_manager.release_status = "close_done"
                    print(
                        "[INFO] Tactile extra close target reached. "
                        f"position={position_value_int}, target={tactile_extra_target_position}. "
                        "Stopping gripper close and finishing grasp stage."
                    )
                    return True
                if position_triggered:
                    position_triggered = False

        if verbose:
            tactile_text = ""
            if tactile_enabled:
                tactile_text = (
                    f", tactile_norm={tactile_norm}, "
                    f"tactile_triggered={tactile_triggered}, "
                    f"tactile_target={tactile_extra_target_position}"
                )
            print(
                "[GRIPPER] "
                f"force_norm={state.tcp_force_norm_n}, "
                f"force_delta={force_delta}, "
                f"mean_joint_current={state.mean_joint_current_a}, "
                f"verified={state.grasp_verified_force_current}, "
                f"force_triggered={force_triggered}, "
                f"position_triggered={position_triggered}, "
                f"position_stall_triggered={position_stall_triggered}, "
                f"stable_position_reads={stable_position_reads}, "
                f"gripper_state={gripper_close_state}"
                f"{tactile_text}"
            )
        if force_triggered:
            stop_gripper_motion_safely(controller, "force trigger")
            note_robot_first_contact(metadata_recorder)
            delta_str = "n/a" if force_delta is None else f"{force_delta:.3f}"
            print(
                f"[INFO] Force rise detected during close (abs={float(force_norm):.3f} N, delta={delta_str} N). "
                "Stopping gripper close."
            )
            return True
        if position_triggered:
            stop_gripper_motion_safely(controller, "position trigger")
            note_robot_first_contact(metadata_recorder)
            position_threshold = int(
                getattr(controller, "gripper_position_complete_threshold", DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD)
            )
            print(
                f"[INFO] Gripper position reached threshold >= {position_threshold}. Stopping gripper close and finishing grasp stage."
            )
            return True
        if position_stall_triggered:
            stop_gripper_motion_safely(controller, "position stall trigger")
            note_robot_first_contact(metadata_recorder)
            if tactile_enabled and tactile_contact_detected:
                tactile_manager.release_status = "close_stall_after_contact"
            print(
                "[INFO] Gripper position stalled "
                f"(position={position_value_int}, stable_reads={stable_position_reads}, "
                f"tolerance={stall_cfg['tolerance']}). "
                "Stopping gripper close and finishing grasp stage."
            )
            return True
        time.sleep(poll_dt)

    stop_gripper_motion_safely(controller, "close timeout")
    if tactile_enabled:
        tactile_manager.release_status = "close_timeout"
    print("[WARN] Grasp could not be verified from RTDE force/current.")
    return False


def execute_gripper_open(controller, dwell_s=0.5, metadata_recorder=None, cancel_event=None):
    """Open the gripper and optionally record the last-contact timestamp."""
    if cancel_event is not None and cancel_event.is_set():
        stop_gripper_motion_safely(controller, "cancel before open")
        return False
    if metadata_recorder is not None:
        last_contact_timestamp = metadata_recorder.note_robot_last_contact()
        if last_contact_timestamp is not None:
            print(f"[INFO] Robot last contact timestamp={last_contact_timestamp}")
    send_robot_command(
        controller,
        ROBOT_CMD_HOLD,
        gripper_action=GRIPPER_OPEN,
        source_mode="gripper_open",
    )
    deadline = time.time() + max(dwell_s, 0.0)
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            stop_gripper_motion_safely(controller, "cancel during open dwell")
            return False
        time.sleep(min(0.05, max(deadline - time.time(), 0.0)))
    return True


def capture_tactile_release_reference(tactile_manager, delay_s=0.0, cancel_event=None):
    """Capture the held-object tactile norm used to gate release timing."""
    if tactile_manager is None or not bool(getattr(tactile_manager, "enabled", False)):
        return None
    if cancel_event is not None and cancel_event.is_set():
        return None
    delay_s = max(0.0, float(delay_s))
    if delay_s > 0.0:
        tactile_manager.release_status = "ref_delay"
        deadline = time.time() + delay_s
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return None
            time.sleep(min(0.05, max(deadline - time.time(), 0.0)))
    reference_norm = tactile_manager.total_norm()
    tactile_manager.set_release_reference(reference_norm)
    print_tactile_frame_log(tactile_manager, stage="release_reference", stage_time_s=0.0)
    print(f"[Tactile] Release reference captured: norm={reference_norm:.3f}")
    return reference_norm


def execute_tactile_release_descent(
    controller,
    tactile_manager,
    fixed_orientation_base,
    start_pose_mm,
    args,
    cancel_event=None,
    on_state_read=None,
):
    """Descend in -Z until tactile release trigger or the configured lower bound."""
    # 시작 pose[mm]에서 x, y, z만 float으로 꺼낸다.
    start_x, start_y, start_z = [float(v) for v in start_pose_mm[:3]]
    # tactile release 하강이 내려갈 수 있는 최저 z[mm]다.
    # HOME_PLACE_MIN_Z_MM보다 낮아지지 않게 해서 바닥/홈 안전 높이를 지킨다.
    min_z_mm = max(
        HOME_PLACE_MIN_Z_MM,
        float(getattr(args, "tactile_release_descent_min_z_mm", HOME_PLACE_MIN_Z_MM)),
    )
    # tactile 값과 로봇 pose를 확인하는 polling 주기[sec]다.
    poll_dt = max(
        0.0,
        float(getattr(args, "tactile_release_descent_poll_dt_s", DEFAULT_TACTILE_RELEASE_DESCENT_POLL_DT_S)),
    )
    # release reference 대비 tactile norm 변화량이 이 값보다 커지면 release trigger로 판단한다.
    delta_threshold = max(
        0.0,
        float(getattr(args, "tactile_release_delta_threshold", DEFAULT_TACTILE_RELEASE_DELTA_THRESHOLD)),
    )
    release_stop_timing_debug = bool(
        getattr(args, "tactile_release_stop_timing_debug", DEFAULT_TACTILE_RELEASE_STOP_TIMING_DEBUG)
    )
    release_stop_speed_threshold_mps = max(
        0.0,
        float(
            getattr(
                args,
                "tactile_release_stop_speed_threshold_mps",
                DEFAULT_TACTILE_RELEASE_STOP_SPEED_THRESHOLD_MPS,
            )
        ),
    )
    release_stop_monitor_timeout_s = max(
        0.0,
        float(
            getattr(
                args,
                "tactile_release_stop_monitor_timeout_s",
                DEFAULT_TACTILE_RELEASE_STOP_MONITOR_TIMEOUT_S,
            )
        ),
    )
    release_stop_monitor_poll_dt_s = max(
        0.0,
        float(
            getattr(
                args,
                "tactile_release_stop_monitor_poll_dt_s",
                DEFAULT_TACTILE_RELEASE_STOP_MONITOR_POLL_DT_S,
            )
        ),
    )
    # 로봇 위치 도달 판정 tolerance를 meter에서 millimeter로 변환한다.
    tolerance_mm = max(0.0, float(getattr(args, "position_tolerance_m", 0.0)) * 1000.0)
    # 현재 release 후보 pose다. 시작 z가 min_z보다 낮으면 min_z로 올려 안전 범위 안에 둔다.
    current_pose_mm = (start_x, start_y, max(start_z, min_z_mm))
    # tactile descent 로그의 상대 시간을 계산하기 위한 시작 시각이다.
    descent_start_unix = time.time()
    descent_start_perf = time.perf_counter()
    # 하강 move가 너무 오래 걸릴 때 빠져나오기 위한 deadline이다.
    deadline = descent_start_unix + max(0.0, float(getattr(args, "move_timeout_s", 0.0)))
    # tactile manager에 현재 release 하강 중임을 기록한다.
    tactile_manager.release_status = "release_descending"
    # 하강 시작 전에 이미 cancel 요청이 있으면 로봇을 멈추고 현재 pose에서 종료한다.
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_rtde(controller)
        tactile_manager.release_status = "release_cancelled"
        return {
            "triggered": False,
            "timed_out": False,
            "reached_min_z": False,
            "release_pose_mm": current_pose_mm,
        }

    # 하강 시작 조건을 콘솔에 남긴다.
    print(
        "[Tactile] Release descent start: "
        f"start=({start_x:.1f}, {start_y:.1f}, {start_z:.1f}), "
        f"target_z={min_z_mm:.1f}, "
        f"delta_threshold={delta_threshold:.3f}."
    )

    # 로봇에게 현재 x/y와 고정 orientation을 유지한 채 min_z까지 내려가라고 명령한다.
    # 아래 while loop는 이 이동이 진행되는 동안 tactile trigger를 계속 감시한다.
    send_robot_command(
        controller,
        ROBOT_CMD_MOVE_TO_POSITION,
        target_position_base=mm_to_m_tuple([start_x, start_y, min_z_mm]),
        fixed_orientation_base=fixed_orientation_base,
        gripper_action=GRIPPER_HOLD,
        source_mode="tactile_release_descent",
    )

    def fmt_debug_value(value, precision=6, suffix=""):
        if value is None:
            return "-"
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(value_float):
            return "-"
        return f"{value_float:.{precision}f}{suffix}"

    def log_release_stop_timing(event, perf_s=None, unix_s=None, **values):
        if not release_stop_timing_debug:
            return
        now_perf = time.perf_counter() if perf_s is None else float(perf_s)
        now_unix = time.time() if unix_s is None else float(unix_s)
        elapsed_s = now_perf - descent_start_perf
        fields = [
            f"event={event}",
            f"t_rel_s={elapsed_s:.6f}",
            f"unix_s={now_unix:.6f}",
        ]
        for key, value in values.items():
            fields.append(f"{key}={value}")
        print("[TactileTiming] " + " ".join(fields), flush=True)

    def monitor_release_stop(trigger_info, stop_issue_perf, stop_return_perf):
        if not release_stop_timing_debug:
            return

        trigger_perf = trigger_info.get("trigger_perf_s")
        tcp_stop_perf = None
        steady_perf = None
        last_linear_speed = None
        last_angular_speed = None
        last_is_steady = None
        read_error = None
        deadline_perf = time.perf_counter() + release_stop_monitor_timeout_s

        while time.perf_counter() <= deadline_perf:
            state_read_unix = time.time()
            try:
                state = controller.read_robot_state(now_timestamp=state_read_unix)
                if on_state_read is not None:
                    on_state_read(state)
                linear_speed, angular_speed = compute_tcp_speed_norms(state)
                last_linear_speed = linear_speed
                last_angular_speed = angular_speed
            except Exception as exc:
                read_error = repr(exc)
                linear_speed = None
                angular_speed = None

            is_steady = read_rtde_is_steady(controller)
            observed_perf = time.perf_counter()
            observed_unix = time.time()
            if is_steady is not None:
                last_is_steady = is_steady

            if tcp_stop_perf is None and linear_speed is not None and linear_speed <= release_stop_speed_threshold_mps:
                tcp_stop_perf = observed_perf
                log_release_stop_timing(
                    "tcp_speed_stopped",
                    perf_s=observed_perf,
                    unix_s=observed_unix,
                    linear_speed_mps=fmt_debug_value(linear_speed),
                    angular_speed=fmt_debug_value(angular_speed),
                    threshold_mps=fmt_debug_value(release_stop_speed_threshold_mps),
                    command_to_tcp_stop_s=fmt_debug_value(tcp_stop_perf - stop_issue_perf),
                    trigger_to_tcp_stop_s=fmt_debug_value(tcp_stop_perf - trigger_perf),
                )

            if steady_perf is None and is_steady is True:
                steady_perf = observed_perf
                log_release_stop_timing(
                    "isSteady_true",
                    perf_s=observed_perf,
                    unix_s=observed_unix,
                    linear_speed_mps=fmt_debug_value(linear_speed),
                    command_to_isSteady_s=fmt_debug_value(steady_perf - stop_issue_perf),
                    trigger_to_isSteady_s=fmt_debug_value(steady_perf - trigger_perf),
                )

            if tcp_stop_perf is not None and steady_perf is not None:
                break

            if release_stop_monitor_poll_dt_s <= 0.0:
                time.sleep(0.0)
            else:
                time.sleep(release_stop_monitor_poll_dt_s)

        if tcp_stop_perf is None:
            log_release_stop_timing(
                "tcp_speed_stop_timeout",
                linear_speed_mps=fmt_debug_value(last_linear_speed),
                angular_speed=fmt_debug_value(last_angular_speed),
                threshold_mps=fmt_debug_value(release_stop_speed_threshold_mps),
                timeout_s=fmt_debug_value(release_stop_monitor_timeout_s),
                read_error=read_error if read_error is not None else "-",
            )
        if steady_perf is None:
            log_release_stop_timing(
                "isSteady_timeout",
                isSteady=last_is_steady if last_is_steady is not None else "-",
                timeout_s=fmt_debug_value(release_stop_monitor_timeout_s),
            )

        log_release_stop_timing(
            "actual_stop_summary",
            trigger_to_command_s=fmt_debug_value(stop_issue_perf - trigger_perf),
            command_call_s=fmt_debug_value(stop_return_perf - stop_issue_perf),
            command_to_tcp_stop_s=fmt_debug_value(None if tcp_stop_perf is None else tcp_stop_perf - stop_issue_perf),
            command_to_isSteady_s=fmt_debug_value(None if steady_perf is None else steady_perf - stop_issue_perf),
            trigger_to_tcp_stop_s=fmt_debug_value(None if tcp_stop_perf is None else tcp_stop_perf - trigger_perf),
            trigger_to_isSteady_s=fmt_debug_value(None if steady_perf is None else steady_perf - trigger_perf),
        )

    def stop_for_release_trigger(trigger_info):
        stop_issue_perf = time.perf_counter()
        stop_issue_unix = time.time()
        log_release_stop_timing(
            "stop_command_issue",
            perf_s=stop_issue_perf,
            unix_s=stop_issue_unix,
            trigger_to_command_s=fmt_debug_value(stop_issue_perf - trigger_info.get("trigger_perf_s")),
        )
        safe_stop_rtde(controller)
        stop_return_perf = time.perf_counter()
        stop_return_unix = time.time()
        log_release_stop_timing(
            "stop_command_return",
            perf_s=stop_return_perf,
            unix_s=stop_return_unix,
            command_call_s=fmt_debug_value(stop_return_perf - stop_issue_perf),
            trigger_to_return_s=fmt_debug_value(stop_return_perf - trigger_info.get("trigger_perf_s")),
        )
        monitor_release_stop(trigger_info, stop_issue_perf, stop_return_perf)

    def read_current_pose_mm():
        # RTDE에서 현재 TCP pose를 읽는다.
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        pose = state.actual_tcp_pose_base
        # pose를 못 읽으면 None을 반환해 바깥 loop에서 안전 정지한다.
        if pose is None:
            return None
        # controller pose는 meter 단위이므로 mm 단위 tuple로 변환한다.
        pose_mm = meters_to_mm(pose[:3])
        return tuple(float(v) for v in pose_mm[:3])

    def check_trigger(pose_mm):
        # tactile sensor 전체 norm을 읽는다.
        current_norm = tactile_manager.total_norm()
        # release reference 대비 현재 norm 변화량을 갱신하고 가져온다.
        delta_norm = tactile_manager.update_release_delta(current_norm)
        # 현재 하강 단계의 tactile frame log를 남긴다.
        print_tactile_frame_log(
            tactile_manager,
            stage="release_descent",
            stage_time_s=time.perf_counter() - descent_start_perf,
            z_mm=f"{pose_mm[2]:.1f}",
        )
        # 변화량이 threshold를 넘으면 물체가 놓일 조건이 되었다고 판단한다.
        if delta_norm is not None and delta_norm > delta_threshold:
            trigger_perf = time.perf_counter()
            trigger_unix = time.time()
            sample_perf = getattr(tactile_manager, "last_sample_perf_s", np.nan)
            sample_unix = getattr(tactile_manager, "last_sample_unix_s", np.nan)
            sample_age_s = None
            if sample_perf is not None:
                try:
                    sample_perf_float = float(sample_perf)
                    if np.isfinite(sample_perf_float):
                        sample_age_s = trigger_perf - sample_perf_float
                except (TypeError, ValueError):
                    sample_age_s = None
            tactile_manager.release_status = "release_triggered"
            print(
                "[Tactile] Release trigger detected during descent: "
                f"norm={current_norm:.3f}, ref={tactile_manager.release_reference_norm:.3f}, "
                f"delta={delta_norm:.3f} > {delta_threshold:.3f}."
            )
            log_release_stop_timing(
                "trigger_detected",
                perf_s=trigger_perf,
                unix_s=trigger_unix,
                descent_s=fmt_debug_value(trigger_perf - descent_start_perf),
                norm=fmt_debug_value(current_norm, precision=3),
                ref=fmt_debug_value(tactile_manager.release_reference_norm, precision=3),
                delta=fmt_debug_value(delta_norm, precision=3),
                threshold=fmt_debug_value(delta_threshold, precision=3),
                z_mm=fmt_debug_value(pose_mm[2], precision=1),
                tactile_sample_unix_s=fmt_debug_value(sample_unix),
                tactile_sample_age_s=fmt_debug_value(sample_age_s),
            )
            return {
                "trigger_perf_s": trigger_perf,
                "trigger_unix_s": trigger_unix,
                "current_norm": float(current_norm),
                "reference_norm": tactile_manager.release_reference_norm,
                "delta_norm": float(delta_norm),
                "delta_threshold": float(delta_threshold),
                "pose_mm": tuple(float(v) for v in pose_mm[:3]),
                "sample_perf_s": sample_perf,
                "sample_unix_s": sample_unix,
                "sample_age_s": sample_age_s,
            }
        # 아직 release trigger 조건을 만족하지 못했다.
        return None

    # move_to_position이 진행되는 동안 cancel, pose read, tactile trigger, min_z, timeout을 반복 확인한다.
    while True:
        # 외부 reset/shutdown 등이 들어오면 즉시 하강을 멈추고 실패 결과를 반환한다.
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_rtde(controller)
            tactile_manager.release_status = "release_cancelled"
            return {
                "triggered": False,
                "timed_out": False,
                "reached_min_z": False,
                "release_pose_mm": current_pose_mm,
            }
        # 현재 로봇 TCP 위치[mm]를 읽는다.
        pose_mm = read_current_pose_mm()
        # pose를 읽지 못하면 더 내려가는 것이 위험하므로 멈추고 현재 후보 pose에서 open하도록 반환한다.
        if pose_mm is None:
            tactile_manager.release_status = "release_descent_pose_missing"
            safe_stop_rtde(controller)
            print(
                "[WARN] Tactile release descent could not read current TCP pose; "
                "opening gripper at last known target."
            )
            return {
                "triggered": False,
                "timed_out": True,
                "reached_min_z": False,
                "release_pose_mm": current_pose_mm,
            }
        # 마지막으로 읽은 실제 TCP pose를 release 후보 pose로 갱신한다.
        current_pose_mm = pose_mm

        # tactile 변화량이 threshold를 넘었는지 확인한다.
        trigger_info = check_trigger(current_pose_mm)
        if trigger_info is not None:
            # trigger가 잡히면 하강 이동을 멈추고 현재 위치를 gripper open 위치로 반환한다.
            stop_for_release_trigger(trigger_info)
            return {
                "triggered": True,
                "timed_out": False,
                "reached_min_z": False,
                "release_pose_mm": current_pose_mm,
            }

        # tactile trigger가 없어도 최저 z에 도달하면 더 내려가지 않고 현재 위치에서 open한다.
        if current_pose_mm[2] <= min_z_mm + tolerance_mm:
            tactile_manager.release_status = "release_min_z_open"
            safe_stop_rtde(controller)
            print(
                "[Tactile] Release descent reached min z without trigger; opening gripper. "
                f"z={current_pose_mm[2]:.1f}, min_z={min_z_mm:.1f}."
            )
            return {
                "triggered": False,
                "timed_out": False,
                "reached_min_z": True,
                "release_pose_mm": current_pose_mm,
            }

        # 지정 timeout을 넘기면 이동을 멈추고 현재 위치에서 open하도록 반환한다.
        if time.time() >= deadline:
            tactile_manager.release_status = "release_descent_move_timeout"
            safe_stop_rtde(controller)
            print(
                "[WARN] Tactile release descent move timed out; opening gripper at current pose. "
                f"z={current_pose_mm[2]:.1f}, min_z={min_z_mm:.1f}."
            )
            return {
                "triggered": False,
                "timed_out": True,
                "reached_min_z": False,
                "release_pose_mm": current_pose_mm,
            }
        # 설정된 polling 주기만큼 쉬었다가 다시 tactile/pose 상태를 확인한다.
        if poll_dt > 0.0:
            time.sleep(poll_dt)


def reset_tactile_baseline_after_open(tactile_manager, delay_s, cancel_event=None):
    if tactile_manager is None or not bool(getattr(tactile_manager, "enabled", False)):
        return
    if cancel_event is not None and cancel_event.is_set():
        return
    delay_s = max(0.0, float(delay_s))
    if delay_s <= 0.0:
        return
    tactile_manager.release_status = "baseline_reset_delay"
    deadline = time.time() + delay_s
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            return
        time.sleep(min(0.05, max(deadline - time.time(), 0.0)))
    if tactile_manager.reset_baseline():
        tactile_manager.release_status = "baseline_reset"


def save_grasp_offset(controller, shared_state, on_state_read=None):
    """Store object-to-tool offset and freeze place-height data after grasp."""
    snap = shared_state.get_snapshot()
    obj_xyz = snap["latest_object_xyz_mm"]
    if obj_xyz is None:
        print("[WARN] Cannot save grasp offset: no latest object xyz.")
        return False

    state = controller.read_robot_state(now_timestamp=time.time())
    if on_state_read is not None:
        on_state_read(state)
    cur_pose = state.actual_tcp_pose_base
    if cur_pose is None:
        print("[WARN] Cannot read EEF pose for grasp offset.")
        return False

    eef_xyz = meters_to_mm(cur_pose[:3])
    grasp_offset_xyz = obj_xyz - eef_xyz

    with shared_state.lock:
        shared_state.grasp_offset_xyz_mm = grasp_offset_xyz
        shared_state.grasp_closed = True
    place_z_result = shared_state.finalize_place_z_from_recent_samples()
    shared_state.set_task_state("GRASPED", reset_prediction=True, reset_arm=True)

    print(f"[INFO] grasp_offset_xyz_mm saved: {grasp_offset_xyz}")
    if place_z_result["valid"]:
        print(
            "[INFO] frozen_place_z_mm saved: "
            f"grasp_z_med={place_z_result['grasp_z_median_mm']:.1f}, "
            f"template_bottom_z_med={place_z_result['template_bottom_z_median_mm']:.1f}, "
            f"raw_place_z={place_z_result['raw_place_z_mm']:.1f}, "
            f"place_z={place_z_result['place_z_mm']:.1f}"
        )
    else:
        print(
            "[WARN] frozen_place_z_mm fallback armed: "
            f"reason={place_z_result['reason']}, "
            f"grasp_samples={place_z_result['grasp_sample_count']}, "
            f"template_bottom_samples={place_z_result['template_bottom_sample_count']}"
        )
    return True


def compute_place_target(shared_state):
    """Compute the delivery tool target from home pose and saved grasp offset."""
    with shared_state.lock:
        home_xyz = None if shared_state.home_object_xyz_mm is None else shared_state.home_object_xyz_mm.copy()
        grasp_offset = None if shared_state.grasp_offset_xyz_mm is None else shared_state.grasp_offset_xyz_mm.copy()
        frozen_place_z_mm = shared_state.frozen_place_z_mm
        frozen_place_z_raw_mm = shared_state.frozen_place_z_raw_mm
        frozen_place_z_grasp_median_mm = shared_state.frozen_place_z_grasp_median_mm
        frozen_place_z_template_bottom_median_mm = shared_state.frozen_place_z_template_bottom_median_mm

    if home_xyz is None:
        return None, {
            "used_fallback": True,
            "fallback_reason": "no_home_xyz",
            "grasp_z_median_mm": None,
            "template_bottom_z_median_mm": None,
            "raw_place_z_mm": None,
        }

    if grasp_offset is not None:
        home_xyz[0] -= grasp_offset[0]
    home_xyz[0] += HOME_PLACE_X_OFFSET_MM
    home_xyz[1] += HOME_PLACE_Y_OFFSET_MM
    debug = {
        "used_fallback": frozen_place_z_mm is None,
        "fallback_reason": "home_z" if frozen_place_z_mm is None else "none",
        "grasp_z_median_mm": frozen_place_z_grasp_median_mm,
        "template_bottom_z_median_mm": frozen_place_z_template_bottom_median_mm,
        "raw_place_z_mm": float(home_xyz[2]) if frozen_place_z_raw_mm is None else float(frozen_place_z_raw_mm),
    }
    if frozen_place_z_mm is not None:
        home_xyz[2] = float(frozen_place_z_mm)
    home_xyz[2] = max(float(home_xyz[2]), HOME_PLACE_MIN_Z_MM)
    return home_xyz, debug


def compute_pre_release_descend_target_mm(place_x, place_y, place_z, args):
    """Resolve the optional descend target used immediately before gripper open."""
    enabled = bool(getattr(args, "pre_release_descend_before_open", False))
    descend_mm = max(0.0, float(getattr(args, "pre_release_descend_mm", RELEASE_PARAMETER_MM)))
    place_x = float(place_x)
    place_y = float(place_y)
    place_z = float(place_z)
    if not enabled or descend_mm <= 1e-9:
        return (place_x, place_y, place_z), {
            "enabled": False,
            "descend_mm": descend_mm,
            "unclamped_z_mm": place_z,
            "target_z_mm": place_z,
            "home_guard_applied": False,
        }

    unclamped_z = place_z - descend_mm
    min_release_z = HOME_PLACE_MIN_Z_MM + PRE_RELEASE_MIN_Z_EPSILON_MM
    workspace_z = getattr(args, "workspace_z", DEFAULT_WORKSPACE_MM["z"])
    workspace_z_low = float(workspace_z[0])
    workspace_z_high = float(workspace_z[1])
    guarded_low = max(min_release_z, workspace_z_low)
    target_z = clamp_value(unclamped_z, guarded_low, workspace_z_high)
    if target_z <= HOME_PLACE_MIN_Z_MM:
        target_z = min_release_z

    target_x = clamp_value(place_x, args.workspace_x[0], args.workspace_x[1])
    target_y = clamp_value(place_y, args.workspace_y[0], args.workspace_y[1])
    home_guard_applied = unclamped_z < min_release_z or target_z <= HOME_PLACE_MIN_Z_MM
    return (target_x, target_y, target_z), {
        "enabled": True,
        "descend_mm": descend_mm,
        "unclamped_z_mm": unclamped_z,
        "target_z_mm": target_z,
        "home_guard_applied": bool(home_guard_applied),
    }


def execute_return_and_place(
    controller,
    shared_state,
    args,
    metadata_recorder=None,
    tactile_manager=None,
    cancel_event=None,
    on_state_read=None,
):
    """Run the post-grasp return, release, backoff, and HOME sequence."""
    # 함수 시작 시 이미 cancel 요청이 들어온 상태라면 로봇을 멈추고 실패로 종료한다.
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_rtde(controller)
        return False
    # shared_state에 저장된 grasp offset/place z 샘플을 이용해 최종 place 목표 EEF 위치[mm]를 계산한다.
    target_eef_xyz, place_target_debug = compute_place_target(shared_state)
    # place 목표를 계산할 수 없으면 이후 이동 경로를 만들 수 없으므로 중단한다.
    if target_eef_xyz is None:
        print("[WARN] Cannot compute place target.")
        return False

    # 현재 공유 상태 snapshot에서 로봇 자세 고정값과 초기 pose를 가져온다.
    snap = shared_state.get_snapshot()
    # place/return 동안 TCP orientation을 고정하기 위한 base 좌표계 orientation이다.
    fixed_orientation_base = snap["fixed_orientation_base"]
    # HOME joint 복귀가 실패했을 때 fallback으로 돌아갈 초기 TCP pose다.
    initial_pose_base = snap["initial_pose_base"]
    # 고정 orientation이 없으면 위치 이동 명령을 만들 수 없으므로 중단한다.
    if fixed_orientation_base is None:
        print("[WARN] No fixed orientation.")
        return False

    # tactile manager가 있고 enabled이면 놓기 직전 tactile 기반 하강 로직을 사용한다.
    tactile_enabled = tactile_manager is not None and bool(getattr(tactile_manager, "enabled", False))

    def move_with_optional_cancel(target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
        # move_robot_and_wait에 넘길 공통 이동 옵션을 모은다.
        kwargs = {
            "timeout_s": timeout_s,
            "tolerance_m": tolerance_m,
            "source_mode": source_mode,
        }
        # cancel_event가 있으면 blocking move 중간에도 취소할 수 있도록 전달한다.
        if cancel_event is not None:
            kwargs["cancel_event"] = cancel_event
        if on_state_read is not None:
            kwargs["on_state_read"] = on_state_read
        # 지정한 target position/orientation으로 이동 명령을 보내고 도달 여부를 기다린다.
        return move_robot_and_wait(
            controller,
            target_position_base,
            fixed_orientation_base,
            **kwargs,
        )

    # compute_place_target이 만든 기본 place 목표 좌표[mm]를 분리한다.
    target_x, target_y, target_z = target_eef_xyz
    # 목표 좌표가 workspace 범위를 넘지 않도록 clamp한다.
    target_x, target_y, target_z = clamp_pose_mm(target_x, target_y, target_z, args)

    # 물체를 든 상태로 바로 place 지점에 가지 않고, 먼저 위쪽 hover 지점으로 이동한다.
    hover_z = target_z + HOVER_Z_OFFSET_MM
    # hover 지점도 workspace 범위 안에 들어오도록 보정한다.
    hover_x, hover_y, hover_z = clamp_pose_mm(target_x, target_y, hover_z, args)

    # 실제 release 직전까지 내려갈 place z를 계산한다.
    place_z = target_z + DESCEND_EXTRA_MM
    # place 위치 역시 workspace/safety boundary 안으로 제한한다.
    place_x, place_y, place_z = clamp_pose_mm(target_x, target_y, place_z, args)
    # gripper open 직전 추가 하강이 켜져 있으면 release pose를 따로 계산한다.
    release_pose_mm, pre_release_debug = compute_pre_release_descend_target_mm(place_x, place_y, place_z, args)
    # release pose를 x/y/z 변수로 분리해 이후 tactile 또는 gripper open 위치로 사용한다.
    release_x, release_y, release_z = release_pose_mm

    # place z 계산에 사용된 grasp z median을 로그 문자열로 만든다.
    grasp_z_med_text = (
        "-"
        if place_target_debug["grasp_z_median_mm"] is None
        else f"{float(place_target_debug['grasp_z_median_mm']):.1f}"
    )
    # template bottom z median을 로그 문자열로 만든다.
    template_bottom_z_med_text = (
        "-"
        if place_target_debug["template_bottom_z_median_mm"] is None
        else f"{float(place_target_debug['template_bottom_z_median_mm']):.1f}"
    )
    # 보정 전 raw place z를 로그 문자열로 만든다.
    raw_place_z_text = (
        "-"
        if place_target_debug["raw_place_z_mm"] is None
        else f"{float(place_target_debug['raw_place_z_mm']):.1f}"
    )

    # return/place 시퀀스에서 사용할 주요 target을 콘솔에 출력한다.
    print(f"[INFO] RETURN hover target: ({hover_x:.1f}, {hover_y:.1f}, {hover_z:.1f})")
    print(f"[INFO] PLACE target: ({place_x:.1f}, {place_y:.1f}, {place_z:.1f})")
    # place z가 어떤 기준으로 계산됐는지 디버그 정보를 출력한다.
    print(
        "[INFO] PLACE z debug: "
        f"grasp_z_med={grasp_z_med_text} "
        f"template_bottom_z_med={template_bottom_z_med_text} "
        f"raw_place_z={raw_place_z_text} "
        f"fallback={place_target_debug['used_fallback']}"
    )
    # tactile release가 없고 pre-release descend 옵션이 켜졌다면 고정 거리 하강 정보를 출력한다.
    if pre_release_debug["enabled"] and not tactile_enabled:
        print(
            "[INFO] PRE-release descend target: "
            f"({release_x:.1f}, {release_y:.1f}, {release_z:.1f}), "
            f"descend={pre_release_debug['descend_mm']:.1f} mm, "
            f"unclamped_z={pre_release_debug['unclamped_z_mm']:.1f} mm, "
            f"home_guard_applied={pre_release_debug['home_guard_applied']}"
        )
    # tactile release가 켜져 있으면 고정 하강 거리 대신 tactile 조건 기반 하강을 사용한다고 출력한다.
    elif tactile_enabled:
        print(
            "[INFO] Tactile release descent enabled. "
            f"Ignoring fixed pre-release descend distance={pre_release_debug['descend_mm']:.1f} mm; "
            f"min_z={float(getattr(args, 'tactile_release_descent_min_z_mm', HOME_PLACE_MIN_Z_MM)):.1f} mm."
        )

    # gripper를 열기 전 기본 이동 경로: hover 위치로 복귀한 뒤 place 위치로 내려간다.
    move_sequence = [
        ("return_hover", [hover_x, hover_y, hover_z]),
        ("return_place", [place_x, place_y, place_z]),
    ]
    # 각 이동 단계는 timeout/tolerance/cancel을 적용해 순서대로 실행한다.
    for source_mode, pose_mm in move_sequence:
        # 이동 시작 전 cancel 요청이 있으면 즉시 정지하고 실패 처리한다.
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_rtde(controller)
            return False
        # mm 단위 target을 m 단위로 바꿔 로봇 이동을 실행하고 도달 여부를 기다린다.
        ok = move_with_optional_cancel(
            mm_to_m_tuple(pose_mm),
            fixed_orientation_base,
            timeout_s=args.move_timeout_s,
            tolerance_m=args.position_tolerance_m,
            source_mode=source_mode,
        )
        # 특정 이동 단계가 timeout되면 전체 return/place 시퀀스를 실패로 끝낸다.
        if not ok:
            print(f"[WARN] Move timed out during {source_mode}.")
            return False

    # tactile release 모드에서는 놓기 전 현재 tactile 값을 기준값으로 캡처한다.
    if tactile_enabled:
        capture_tactile_release_reference(
            tactile_manager,
            delay_s=getattr(args, "tactile_release_ref_delay_s", DEFAULT_TACTILE_RELEASE_REF_DELAY_S),
            cancel_event=cancel_event,
        )
        # tactile reference 캡처 중 cancel되었으면 로봇을 멈추고 종료한다.
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_rtde(controller)
            return False
        # tactile 기반 하강 함수에 cancel_event를 선택적으로 넘기기 위한 kwargs다.
        descent_kwargs = {}
        if cancel_event is not None:
            descent_kwargs["cancel_event"] = cancel_event
        if on_state_read is not None:
            descent_kwargs["on_state_read"] = on_state_read
        # tactile 변화량을 보면서 release에 적절한 위치까지 추가 하강한다.
        descent_result = execute_tactile_release_descent(
            controller,
            tactile_manager,
            fixed_orientation_base,
            [place_x, place_y, place_z],
            args,
            **descent_kwargs,
        )
        # tactile 하강 결과로 결정된 실제 release pose를 이후 open/backoff 기준으로 사용한다.
        release_x, release_y, release_z = descent_result["release_pose_mm"]
    # tactile이 없고 pre-release descend가 켜져 있으면 고정 거리만큼 release pose로 이동한다.
    elif pre_release_debug["enabled"]:
        ok = move_with_optional_cancel(
            mm_to_m_tuple([release_x, release_y, release_z]),
            fixed_orientation_base,
            timeout_s=args.move_timeout_s,
            tolerance_m=args.position_tolerance_m,
            source_mode="pre_release_descend_before_open",
        )
        # pre-release 하강 이동이 실패하면 gripper를 열지 않고 종료한다.
        if not ok:
            print("[WARN] Pre-release descend move timed out.")
            return False

    # release pose에서 gripper를 열어 물체를 놓는다.
    execute_gripper_open(
        controller,
        dwell_s=args.gripper_release_dwell_s,
        metadata_recorder=metadata_recorder,
        cancel_event=cancel_event,
    )
    # gripper open 중 cancel이 들어왔으면 이후 후퇴/HOME 동작 없이 실패 처리한다.
    if cancel_event is not None and cancel_event.is_set():
        return False
    # tactile 모드에서는 물체를 놓은 뒤 baseline을 다시 잡아 다음 task에 영향을 줄인다.
    if tactile_enabled:
        reset_tactile_baseline_after_open(
            tactile_manager,
            getattr(
                args,
                "tactile_auto_baseline_reset_after_open_s",
                DEFAULT_TACTILE_AUTO_BASELINE_RESET_AFTER_OPEN_S,
            ),
            cancel_event=cancel_event,
        )

    # release 이후 위쪽으로 살짝 올라갈 z 목표를 계산한다.
    post_release_z = max(release_z + float(args.post_release_z_offset_mm), HOME_PLACE_MIN_Z_MM)
    # post-release 위치도 workspace 범위 안으로 제한한다.
    post_release_x, post_release_y, post_release_z = clamp_pose_mm(release_x, release_y, post_release_z, args)
    # 설정된 post-release z offset이 있으면 실제 상승 이동을 수행한다.
    if abs(float(args.post_release_z_offset_mm)) > 1e-6:
        print(
            "[INFO] POST-release Z target: "
            f"({post_release_x:.1f}, {post_release_y:.1f}, {post_release_z:.1f}), "
            f"offset={float(args.post_release_z_offset_mm):.1f} mm"
        )
        # gripper open 후 물체와의 간섭을 줄이기 위해 z 방향으로 이동한다.
        ok = move_with_optional_cancel(
            mm_to_m_tuple([post_release_x, post_release_y, post_release_z]),
            fixed_orientation_base,
            timeout_s=args.move_timeout_s,
            tolerance_m=args.position_tolerance_m,
            source_mode="post_release_z_move",
        )
        # post-release 이동 실패 시 전체 시퀀스를 실패로 끝낸다.
        if not ok:
            print("[WARN] Post-release Z move timed out.")
            return False

    # place 위치에서 x 방향으로 물러나 HOME 복귀 전 물체와 거리를 둔다.
    backoff_x = place_x - BACKOFF_X_MM
    # backoff target도 workspace/safety boundary 안으로 제한한다.
    backoff_x, backoff_y, backoff_z = clamp_pose_mm(backoff_x, place_y, post_release_z, args)
    # 물체를 놓은 뒤 후퇴 이동을 실행한다.
    ok = move_with_optional_cancel(
        mm_to_m_tuple([backoff_x, backoff_y, backoff_z]),
        fixed_orientation_base,
        timeout_s=args.move_timeout_s,
        tolerance_m=args.position_tolerance_m,
        source_mode="return_backoff",
    )
    # 후퇴 이동이 실패하면 HOME 복귀를 시도하지 않고 실패 처리한다.
    if not ok:
        print("[WARN] Back off move timed out.")
        return False

    # backoff moveL이 위치 tolerance만 만족한 상태에서 바로 moveJ로 전환되지 않도록 실제 정지를 확인한다.
    if bool(getattr(args, "post_backoff_stop_check_enabled", DEFAULT_POST_BACKOFF_STOP_CHECK_ENABLED)):
        stop_confirmed = wait_until_robot_stopped(
            controller,
            timeout_s=getattr(args, "post_backoff_stop_timeout_s", DEFAULT_POST_BACKOFF_STOP_TIMEOUT_S),
            speed_threshold_mps=getattr(
                args,
                "post_backoff_stop_speed_threshold_mps",
                DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS,
            ),
            poll_dt=getattr(args, "post_backoff_stop_poll_dt_s", DEFAULT_POST_BACKOFF_STOP_POLL_DT_S),
            cancel_event=cancel_event,
            on_state_read=on_state_read,
            source_mode="post_backoff_stop_check",
        )
        if cancel_event is not None and cancel_event.is_set():
            return False
        if stop_confirmed:
            print("[INFO] POST-backoff stop confirmed before HOME moveJ.")
        else:
            message = "[WARN] POST-backoff stop was not confirmed before HOME moveJ."
            if bool(
                getattr(
                    args,
                    "post_backoff_stop_require_confirmed",
                    DEFAULT_POST_BACKOFF_STOP_REQUIRE_CONFIRMED,
                )
            ):
                print(message + " Aborting return sequence because confirmation is required.")
                return False
            print(message + " Continuing with HOME moveJ.")

    # 정상 경로에서는 joint HOME pose로 복귀한다.
    try:
        home_kwargs = {}
        if cancel_event is not None:
            home_kwargs["cancel_event"] = cancel_event
        if on_state_read is not None:
            home_kwargs["on_state_read"] = on_state_read
        home_ok = move_robot_to_home_pose(controller, args, **home_kwargs)
        # HOME 이동이 cancel로 False를 반환하면 실패 처리한다.
        if home_ok is False:
            return False
    # HOME joint 복귀가 예외로 실패하면 초기 TCP pose로 fallback 복귀를 시도한다.
    except Exception as exc:
        print(f"[WARN] Return to HOME joints failed: {exc}")
        # 초기 pose 기록이 없으면 fallback 이동도 불가능하다.
        if initial_pose_base is None:
            return False
        # 초기 pose에서 position과 orientation을 분리한다.
        initial_target = tuple(float(v) for v in initial_pose_base[:3])
        initial_orientation = tuple(float(v) for v in initial_pose_base[3:6])
        # HOME 대신 초기 TCP pose로 복귀한다.
        ok = move_with_optional_cancel(
            initial_target,
            initial_orientation,
            timeout_s=args.move_timeout_s,
            tolerance_m=args.position_tolerance_m,
            source_mode="return_initial_pose",
        )
        # 초기 pose 복귀도 실패하면 전체 시퀀스를 실패로 끝낸다.
        if not ok:
            print("[WARN] Return to initial pose timed out.")
            return False

    # metadata recorder가 있으면 최종 delivery location을 기록한다.
    if metadata_recorder is not None:
        home_xyz = None if shared_state.home_object_xyz_mm is None else shared_state.home_object_xyz_mm.copy()
        metadata_recorder.note_delivery_location(home_xyz)

    # task state를 DONE으로 바꾸고 prediction/arm 상태를 다음 task를 위해 초기화한다.
    shared_state.set_task_state("DONE", reset_prediction=True, reset_arm=True)

    # return/place 전체 시퀀스가 성공했음을 알린다.
    print("[INFO] RETURN + PLACE done")
    return True


def configure_object_worker_from_args(worker, args, prompt_classes):
    """Apply CLI model/prompt/selection settings to an object worker."""
    effective_prompt_classes = prompt_classes if prompt_classes else list(getattr(worker.segmentation_engine, "prompt_classes", []))
    preprocess_config = dict(getattr(worker.segmentation_engine, "preprocess_config", {}) or {})
    worker.segmentation_engine = SegmentationEngine(
        model_name=args.model,
        prompt_classes=effective_prompt_classes,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        classes=args.classes,
        half=args.half,
        retina_masks=True,
        preprocess_config=preprocess_config,
    )
    worker.selection_mode = args.select_mode
    if args.select_class is not None:
        worker.selection_class_names = list(args.select_class)


def build_dual_perception_pipeline(args):
    """Construct all camera, perception, fusion, fitting, and debug pipeline objects."""
    sensor_hub = DualSensorHub.from_config(args.config)
    sensor_hub.width = int(args.width)
    sensor_hub.height = int(args.height)
    sensor_hub.fps = int(args.fps)

    object_worker_cam0 = ObjectWorkerCam0.from_config(args.config)
    object_worker_cam1 = ObjectWorkerCam1.from_config(args.config)
    prompt_classes = parse_prompt_classes(args.prompt)
    configure_object_worker_from_args(object_worker_cam0, args, prompt_classes)
    configure_object_worker_from_args(object_worker_cam1, args, prompt_classes)

    hand_worker_cam0 = HandWorkerCam0.from_config(args.config)
    hand_worker_cam1 = HandWorkerCam1.from_config(args.config)
    hand_selector = HandSelector.from_config(args.config)
    object_merger = ObjectMerger.from_config(args.config)
    shape_fitting_tracker = ShapeFittingTracker.from_config(args.config)
    fusion = PerceptionFusion.from_config(args.config)
    grasp_planner = GraspTargetPlanner.from_config(args.config)
    hand_relative_fallback = HandRelativeFallbackTracker.from_config(args.config)
    # fill_level_estimator = FillLevelEstimator.from_config(args.config)
    transform_chain = load_transform_chain(args.config)
    t_cam0_base = np.linalg.inv(transform_chain.t_base_cam0).astype(np.float32)
    t_cam1_base = np.linalg.inv(transform_chain.t_base_cam1).astype(np.float32)

    return {
        "sensor_hub": sensor_hub,
        "object_worker_cam0": object_worker_cam0,
        "object_worker_cam1": object_worker_cam1,
        "hand_worker_cam0": hand_worker_cam0,
        "hand_worker_cam1": hand_worker_cam1,
        "hand_selector": hand_selector,
        "object_merger": object_merger,
        "shape_fitting_tracker": shape_fitting_tracker,
        "fusion": fusion,
        "grasp_planner": grasp_planner,
        "hand_relative_fallback": hand_relative_fallback,
        #"fill_level_estimator": fill_level_estimator,
        "transform_chain": transform_chain,
        "t_cam0_base": t_cam0_base,
        "t_cam1_base": t_cam1_base,
        "prompt_classes": prompt_classes,
    }


def build_fitted_merged_object(raw_merged_object, shape_fitting_state):
    """Expose the shape-fitted template cloud through the merger state interface."""
    if not shape_fitting_state.valid:
        return replace(
            raw_merged_object,
            object_detected=False,
            centroid_base=None,
            merged_point_count=0,
            merged_points_base=[],
            valid=False,
        )

    fitted_points = np.asarray(shape_fitting_state.fitted_points_base, dtype=np.float32).reshape((-1, 3))
    return replace(
        raw_merged_object,
        object_detected=True,
        label=shape_fitting_state.label or raw_merged_object.label,
        centroid_base=shape_fitting_state.centroid_base,
        merged_point_count=int(len(fitted_points)),
        merged_points_base=[tuple(float(v) for v in point) for point in fitted_points],
        valid=True,
    )


def project_base_point_to_cam0(point_base, intrinsics, t_cam0_base, width: int, height: int):
    """Project one base-frame point into a camera image for overlay drawing."""
    if point_base is None or intrinsics is None:
        return None
    point = np.asarray(point_base, dtype=np.float32).reshape(3)
    point_h = np.concatenate([point, np.array([1.0], dtype=np.float32)], axis=0)
    point_cam = (t_cam0_base @ point_h.reshape(4, 1)).reshape(-1)[:3]
    z = float(point_cam[2])
    if not np.isfinite(z) or z <= 1e-6:
        return None
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    u = int(round((float(point_cam[0]) * fx / z) + cx))
    v = int(round((float(point_cam[1]) * fy / z) + cy))
    if u < 0 or u >= int(width) or v < 0 or v >= int(height):
        return None
    return (u, v)


def project_base_points_to_cam0(points_base, intrinsics, t_cam0_base, width: int, height: int, max_points: int = 1500):
    """Project a sampled base-frame point cloud into camera pixels for preview."""
    if points_base is None or intrinsics is None:
        return np.empty((0, 2), dtype=np.int32)
    points = np.asarray(points_base, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride]
    ones = np.ones((len(points), 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=1)
    points_cam = (t_cam0_base @ points_h.T).T[:, :3]
    zs = points_cam[:, 2]
    valid = np.isfinite(zs) & (zs > 1e-6)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)
    points_cam = points_cam[valid]
    zs = zs[valid]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    us = np.round((points_cam[:, 0] * fx / zs) + cx).astype(np.int32)
    vs = np.round((points_cam[:, 1] * fy / zs) + cy).astype(np.int32)
    in_bounds = (us >= 0) & (us < int(width)) & (vs >= 0) & (vs < int(height))
    if not np.any(in_bounds):
        return np.empty((0, 2), dtype=np.int32)
    return np.stack([us[in_bounds], vs[in_bounds]], axis=1)


def _build_silhouette_observation(camera_id, frame_bundle, object_worker, t_base_cam):
    """Package the latest per-camera segmentation mask for shape fitting rerank."""
    object_debug = getattr(object_worker, "last_debug", None)
    mask = None if object_debug is None else getattr(object_debug, "combined_mask", None)
    if mask is None:
        return None
    mask_array = np.asarray(mask)
    if mask_array.size == 0 or not np.any(mask_array):
        return None

    intrinsics = getattr(frame_bundle, "intrinsics", None) or {}
    required_keys = ("fx", "fy", "cx", "cy")
    if any(key not in intrinsics for key in required_keys):
        return None

    image_shape = tuple(int(value) for value in frame_bundle.color_image.shape[:2])
    return SilhouetteObservation(
        camera_id=int(camera_id),
        mask=mask_array,
        intrinsics=dict(intrinsics),
        t_base_cam=np.asarray(t_base_cam, dtype=np.float32),
        image_shape=(image_shape[0], image_shape[1]),
        weight=1.0,
    )


def build_silhouette_observations(snapshot, pipeline):
    """Build cam0/cam1 silhouette observations when masks and calibration are available."""
    transform_chain = pipeline.get("transform_chain")
    if transform_chain is None:
        return []

    observations = []
    cam0_observation = _build_silhouette_observation(
        0,
        snapshot.cam0,
        pipeline.get("object_worker_cam0"),
        transform_chain.t_base_cam0,
    )
    if cam0_observation is not None:
        observations.append(cam0_observation)

    cam1_observation = _build_silhouette_observation(
        1,
        snapshot.cam1,
        pipeline.get("object_worker_cam1"),
        transform_chain.t_base_cam1,
    )
    if cam1_observation is not None:
        observations.append(cam1_observation)
    return observations


def choose_point(primary, fallback=None):
    return primary if primary is not None else fallback


def offset_point_base_mm(point_base, *, x_mm=0.0, y_mm=0.0, z_mm=0.0):
    if point_base is None:
        return None
    point = np.asarray(point_base, dtype=np.float32).reshape(3).copy()
    point[0] += float(x_mm) / 1000.0
    point[1] += float(y_mm) / 1000.0
    point[2] += float(z_mm) / 1000.0
    return tuple(float(v) for v in point)


def append_debug_3d_frame(
    debug_3d_recorder,
    *,
    snapshot,
    current_time,
    loop_perf,
    record_elapsed_s,
    current_task_epoch,
    pipeline,
    selected_hand,
    merged_object,
    shape_fitting_state,
    object_point_base,
    grasp_point_base,
    eef_pose_base,
    measurement_source,
    tactile_manager=None,
):
    """Record one synchronized perception/debug frame for later 3D inspection."""
    tactile_snapshot = None
    if tactile_manager is not None and bool(getattr(tactile_manager, "enabled", False)):
        try:
            if hasattr(tactile_manager, "snapshot"):
                tactile_snapshot = tactile_manager.snapshot(refresh=True)
            else:
                if hasattr(tactile_manager, "total_norm"):
                    tactile_manager.total_norm()
                values = np.asarray(getattr(tactile_manager, "latest", []), dtype=np.float32).flatten()
                tactile_snapshot = {
                    "values": values,
                    "num_mags": int(getattr(tactile_manager, "num_mags", max(values.size // 3, 0))),
                    "total_norm": float(getattr(tactile_manager, "latest_norm", float(np.linalg.norm(values)))),
                    "release_ref_norm": getattr(tactile_manager, "release_reference_norm", None),
                    "release_delta_norm": getattr(tactile_manager, "release_delta_norm", None),
                    "status": str(getattr(tactile_manager, "release_status", getattr(tactile_manager, "status", ""))),
                    "error": getattr(tactile_manager, "last_error", None),
                    "timestamp_perf_s": float(getattr(tactile_manager, "last_sample_perf_s", np.nan)),
                    "timestamp_unix_s": float(getattr(tactile_manager, "last_sample_unix_s", np.nan)),
                }
        except Exception as exc:
            tactile_snapshot = {
                "values": np.empty((0,), dtype=np.float32),
                "num_mags": int(getattr(tactile_manager, "num_mags", 0)),
                "total_norm": np.nan,
                "release_ref_norm": getattr(tactile_manager, "release_reference_norm", None),
                "release_delta_norm": getattr(tactile_manager, "release_delta_norm", None),
                "status": "snapshot_error",
                "error": str(exc),
            }
    debug_3d_recorder.append_frame(
        frame_index=int(snapshot.pair_index),
        timestamp_unix_s=current_time,
        timestamp_perf_s=loop_perf,
        record_elapsed_s=record_elapsed_s,
        task_epoch=current_task_epoch,
        selected_hand=selected_hand,
        hand_debug_cam0=getattr(pipeline["hand_worker_cam0"], "last_debug", None),
        hand_debug_cam1=getattr(pipeline["hand_worker_cam1"], "last_debug", None),
        raw_merged_object=merged_object,
        shape_fitting_state=shape_fitting_state,
        object_point_base=object_point_base,
        grasp_point_base=grasp_point_base,
        eef_pose_base=eef_pose_base,
        measurement_source=measurement_source,
        hand_selector_debug=getattr(pipeline["hand_selector"], "last_debug", None),
        tactile_snapshot=tactile_snapshot,
        snapshot=snapshot if debug_3d_recorder.save_images else None,
    )


def build_runtime_profile_context(args, yolo_model, pipeline):
    """Capture static run settings that should be written with profiler output."""
    return {
        "config_path": str(Path(args.config).resolve()),
        "yolo_model": yolo_model,
        "model": args.model,
        "device": args.device,
        "half": bool(args.half),
        "imgsz": int(args.imgsz),
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "select_mode": args.select_mode,
        "select_class": list(args.select_class or []),
        "enable_follow": bool(args.enable_follow),
        "control_hz": float(args.control_hz),
        "prompt_classes": list(pipeline.get("prompt_classes", [])),
    }


def collect_runtime_profile_metrics(
    *,
    snapshot,
    pipeline,
    object_cam0,
    object_cam1,
    merged_object,
    shape_fitting_state,
    selected_hand,
    fusion_state,
    grasp_target,
    measurement_source,
    instant_fps,
    smoothed_fps,
):
    """Collect per-frame perception and timing metrics for runtime profiling."""
    object_debug_cam0 = getattr(pipeline["object_worker_cam0"], "last_debug", None)
    object_debug_cam1 = getattr(pipeline["object_worker_cam1"], "last_debug", None)
    shape_fit_debug = getattr(pipeline["shape_fitting_tracker"], "last_debug", None)
    return {
        "instant_fps": float(instant_fps),
        "smoothed_fps": float(smoothed_fps),
        "camera_pair_delta_ms": float(getattr(snapshot, "timestamp_delta_ms", 0.0)),
        "camera_pair_within_sync_tolerance": bool(getattr(snapshot, "within_sync_tolerance", False)),
        "cam0_yolo_infer_ms": getattr(object_debug_cam0, "infer_ms", None),
        "cam1_yolo_infer_ms": getattr(object_debug_cam1, "infer_ms", None),
        "cam0_object_detected": bool(getattr(object_cam0, "object_detected", False)),
        "cam1_object_detected": bool(getattr(object_cam1, "object_detected", False)),
        "cam0_object_points": int(getattr(object_cam0, "point_count", 0)),
        "cam1_object_points": int(getattr(object_cam1, "point_count", 0)),
        "merged_object_valid": bool(getattr(merged_object, "valid", False)),
        "merged_object_points": int(getattr(merged_object, "merged_point_count", 0)),
        "merged_object_label": getattr(merged_object, "label", None),
        "shape_fit_valid": bool(getattr(shape_fitting_state, "valid", False)),
        "shape_fit_initialized": bool(getattr(shape_fitting_state, "initialized", False)),
        "shape_fit_reason": getattr(shape_fitting_state, "reason", None),
        "shape_fit_icp_time_ms": getattr(shape_fit_debug, "icp_time_ms", None),
        "shape_fit_icp_fitness": getattr(shape_fit_debug, "icp_fitness", None),
        "shape_fit_icp_rmse": getattr(shape_fit_debug, "icp_rmse", None),
        "shape_fit_z_rotation_deg": getattr(shape_fit_debug, "z_rotation_deg", None),
        "shape_fit_silhouette_enabled": getattr(shape_fit_debug, "silhouette_enabled", False),
        "shape_fit_silhouette_candidate_count": getattr(shape_fit_debug, "silhouette_candidate_count", 0),
        "shape_fit_silhouette_valid_camera_count": getattr(shape_fit_debug, "silhouette_valid_camera_count", 0),
        "shape_fit_silhouette_loss": getattr(shape_fit_debug, "silhouette_loss", None),
        "shape_fit_silhouette_outside_loss": getattr(shape_fit_debug, "silhouette_outside_loss", None),
        "shape_fit_silhouette_robust_3d_loss": getattr(shape_fit_debug, "robust_3d_loss", None),
        "shape_fit_rerank_changed_candidate": getattr(shape_fit_debug, "rerank_changed_candidate", False),
        "selected_hand_valid": bool(getattr(selected_hand, "valid", False)),
        "selected_hand_camera": getattr(selected_hand, "selected_camera", None),
        "fusion_object_fresh": bool(getattr(fusion_state, "object_fresh", False)),
        "fusion_hand_fresh": bool(getattr(fusion_state, "hand_fresh", False)),
        "fusion_hand_approach_latched": bool(getattr(fusion_state, "hand_approach_latched", False)),
        "grasp_target_valid": bool(getattr(grasp_target, "valid", False)),
        "measurement_source": str(measurement_source or "none"),
    }


def save_runtime_profile(runtime_profiler, *, reason):
    if not getattr(runtime_profiler, "enabled", False):
        return
    try:
        paths = runtime_profiler.save_current_session(reason=reason)
    except Exception as exc:
        print(f"[WARN] Runtime profile save failed: {exc}")
        return
    if paths:
        print(f"[INFO] Runtime profile saved: {paths.get('summary')}")


def discard_runtime_profile(runtime_profiler, *, reason):
    if not getattr(runtime_profiler, "enabled", False):
        return
    runtime_profiler.discard_current_session(reason=reason)


def render_camera_mask_preview(
    snapshot,
    pipeline,
    object_worker,
    selected_hand,
    fusion_state,
    display_grasp_point,
    *,
    camera_label,
):
    """Render the secondary camera preview with object mask and key projected points."""
    frame_bundle = snapshot.cam0 if camera_label == "cam0" else snapshot.cam1
    image_bgr = np.asarray(frame_bundle.color_image).copy()

    object_debug = getattr(object_worker, "last_debug", None)
    combined_mask = getattr(object_debug, "combined_mask", None)
    if combined_mask is not None:
        mask_bool = np.asarray(combined_mask, dtype=bool)
        if mask_bool.shape[:2] == image_bgr.shape[:2]:
            overlay = np.zeros_like(image_bgr, dtype=np.uint8)
            overlay[mask_bool] = np.array([255, 180, 0], dtype=np.uint8)
            image_bgr = cv.addWeighted(image_bgr, 1.0, overlay, 0.35, 0.0)

    camera_transform = pipeline["t_cam0_base"] if camera_label == "cam0" else pipeline["t_cam1_base"]
    hand_point = choose_point(fusion_state.filtered_hand_center_base, selected_hand.palm_center_base)
    draw_specs = [
        (display_grasp_point, (0, 255, 0), "grasp"),
        (hand_point, (255, 120, 0), "hand"),
    ]
    for point_base, color_bgr, _label in draw_specs:
        pixel = project_base_point_to_cam0(
            point_base,
            frame_bundle.intrinsics,
            camera_transform,
            image_bgr.shape[1],
            image_bgr.shape[0],
        )
        if pixel is None:
            continue
        cv.circle(image_bgr, pixel, 6, color_bgr, -1, cv.LINE_AA)
    return image_bgr


def format_record_clock(elapsed_s):
    if elapsed_s is None:
        return None
    elapsed_s = max(float(elapsed_s), 0.0)
    minutes = int(elapsed_s // 60.0)
    seconds = int(elapsed_s % 60.0)
    tenths = int((elapsed_s - int(elapsed_s)) * 10.0)
    return f"REC {minutes:02d}:{seconds:02d}.{tenths:d}"


def draw_record_clock_overlay(image_bgr, elapsed_s):
    """Draw the task recording clock onto a preview frame."""
    clock_text = format_record_clock(elapsed_s)
    if not clock_text:
        return image_bgr

    origin = (18, 42)
    cv.putText(image_bgr, clock_text, origin, cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_bgr, clock_text, origin, cv.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 255), 2, cv.LINE_AA)
    return image_bgr


def draw_tactile_status_overlay(image_bgr, tactile_manager):
    if tactile_manager is None or not bool(getattr(tactile_manager, "enabled", False)):
        return image_bgr

    tactile_snapshot = tactile_manager.snapshot(refresh=False) if hasattr(tactile_manager, "snapshot") else None
    norm_value = getattr(tactile_manager, "latest_norm", 0.0) if tactile_snapshot is None else tactile_snapshot.get("total_norm", 0.0)
    norm_text = f"{float(norm_value):.1f}"
    ref_norm = getattr(tactile_manager, "release_reference_norm", None) if tactile_snapshot is None else tactile_snapshot.get("release_ref_norm")
    ref_text = "-" if ref_norm is None else f"{float(ref_norm):.1f}"
    delta_norm = getattr(tactile_manager, "release_delta_norm", None) if tactile_snapshot is None else tactile_snapshot.get("release_delta_norm")
    delta_text = "-" if delta_norm is None else f"{float(delta_norm):.1f}"
    status = str(getattr(tactile_manager, "release_status", "off") if tactile_snapshot is None else tactile_snapshot.get("status", "off"))
    text = f"Tactile norm={norm_text} ref={ref_text} d={delta_text} {status}"

    origin = (12, 74)
    cv.putText(image_bgr, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image_bgr, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv.LINE_AA)
    return image_bgr


def render_cam0_perception_debug(
    snapshot,
    pipeline,
    merged_object,
    selected_hand,
    fusion_state,
    display_object_point,
    display_grasp_point,
    shared_state,
    record_elapsed_s=None,
    tactile_manager=None,
):
    """Render the main cam0 preview with mask, object cloud, hand, grasp, and HOME."""
    image_bgr = np.asarray(snapshot.cam0.color_image).copy()

    object_debug = getattr(pipeline["object_worker_cam0"], "last_debug", None)
    combined_mask = getattr(object_debug, "combined_mask", None)
    if combined_mask is not None:
        mask_bool = np.asarray(combined_mask, dtype=bool)
        if mask_bool.shape[:2] == image_bgr.shape[:2]:
            overlay = np.zeros_like(image_bgr, dtype=np.uint8)
            overlay[mask_bool] = np.array([0, 180, 255], dtype=np.uint8)
            image_bgr = cv.addWeighted(image_bgr, 1.0, overlay, 0.35, 0.0)

    merged_pixels = project_base_points_to_cam0(
        merged_object.merged_points_base,
        snapshot.cam0.intrinsics,
        pipeline["t_cam0_base"],
        image_bgr.shape[1],
        image_bgr.shape[0],
        max_points=1800,
    )
    if len(merged_pixels) > 0:
        point_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        point_mask[merged_pixels[:, 1], merged_pixels[:, 0]] = 255
        point_mask = cv.dilate(point_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        image_bgr[point_mask > 0] = np.array([0, 0, 255], dtype=np.uint8)

    object_point = display_object_point
    hand_point = choose_point(fusion_state.filtered_hand_center_base, selected_hand.palm_center_base)
    draw_specs = [
        (object_point, (0, 140, 255), "obj"),
        (display_grasp_point, (0, 255, 0), "grasp"),
        (hand_point, (255, 120, 0), "hand"),
    ]
    for point_base, color_bgr, label in draw_specs:
        pixel = project_base_point_to_cam0(
            point_base,
            snapshot.cam0.intrinsics,
            pipeline["t_cam0_base"],
            image_bgr.shape[1],
            image_bgr.shape[0],
        )
        if pixel is None:
            continue
        cv.circle(image_bgr, pixel, 6, color_bgr, -1, cv.LINE_AA)
        cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv.LINE_AA)

    home_pixel = shared_state.home_object_pixel if shared_state.home_object_locked else None
    if home_pixel is not None:
        cv.circle(image_bgr, home_pixel, 5, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(image_bgr, "HOME", (home_pixel[0] + 10, home_pixel[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image_bgr, "HOME", (home_pixel[0] + 10, home_pixel[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv.LINE_AA)

    image_bgr = draw_record_clock_overlay(image_bgr, record_elapsed_s)
    image_bgr = draw_tactile_status_overlay(image_bgr, tactile_manager)
    return image_bgr


def main():
    """Run the full dual-camera handover loop and handle keyboard controls."""
    # 커맨드라인 인자를 읽어 기본 실행 옵션을 만든다.
    args = parse_args()
    # 인자로 지정된 YAML 설정 파일을 로드한다.
    config = load_yaml_config(args.config)
    # YAML에 들어 있는 기본값을 커맨드라인 인자에 반영한다.
    args = apply_config_defaults(args, config)

    # 듀얼 카메라 모드에서는 개별 시리얼 인자를 사용하지 않으므로 경고만 출력한다.
    if args.serial is not None:
        print("[WARN] --serial is ignored in the dual-camera grasp-target mode. Camera selection comes from configs/handover.yaml")

    # 카메라, 검출기, 융합기, 로봇 좌표 변환 등 전체 perception 파이프라인을 구성한다.
    pipeline = build_dual_perception_pipeline(args)
    # 두 카메라 프레임을 동기화해서 공급하는 센서 허브를 가져온다.
    sensor_hub = pipeline["sensor_hub"]
    # 촉각 센서 옵션이 켜져 있으면 AnySkin tactile manager를 생성한다.
    tactile_manager = AnySkinTactileManager(args) if bool(getattr(args, "tactile_enabled", False)) else None
    # tactile manager가 생성된 경우 별도 수집 루프를 시작한다.
    if tactile_manager is not None:
        tactile_manager.start()
    # 듀얼 카메라 스트리밍을 시작한다.
    sensor_hub.start()

    # 런타임 로그에 남길 YOLO 모델 이름을 prompt class 정보까지 포함해 만든다.
    yolo_model = args.model if not pipeline["prompt_classes"] else f"{args.model} ({','.join(pipeline['prompt_classes'])})"
    # gpu, cpu 리소스 속도 파악
    # 프레임별 처리 시간과 시스템 리소스 사용량을 기록할 profiler를 준비한다.
    runtime_profiler = RuntimeProfiler(
        # --profile-runtime 옵션이 켜진 경우에만 profiler가 실제로 동작한다.
        enabled=bool(args.profile_runtime),
        # 저장될 runtime profile 파일의 출력 디렉터리다.
        output_dir=args.profile_dir,
        # CPU/GPU 샘플링 주기다.
        sample_interval_s=args.profile_sample_interval_s,
        # 콘솔에 profiler 요약을 출력할 주기다.
        print_every_s=args.profile_print_every_s,
        # 실행 조건, 모델, 카메라 설정 등 profile 메타데이터를 함께 저장한다.
        run_context=build_runtime_profile_context(args, yolo_model, pipeline),
    )
    # profiler가 켜져 있으면 저장 단축키를 사용자에게 알려준다.
    if runtime_profiler.enabled:
        print(f"[INFO] Runtime profiler enabled. Press 's' to save logs to {args.profile_dir}.")
    
    # cam0 디버그 화면의 OpenCV 창 이름이다.
    cam0_window_name = "cam0_grasp_target_follow"
    # cam1 미리보기 화면의 OpenCV 창 이름이다.
    cam1_window_name = "cam1_view"
    
    # depth_window_name = "cam0_depth"

    # 지수 이동 평균으로 부드럽게 표시할 FPS 값이다.
    smoothed_fps = 0.0
    # 이전 루프가 끝난 perf_counter 시각을 저장해 순간 FPS를 계산한다.
    last_loop_time = time.perf_counter()
    # previous_hand_approach = False

    # 메인 스레드와 로봇 스레드가 공유하는 target/follow 상태 객체다.
    shared_state = FollowSharedState(args)
    # task ready, geometry, completion 같은 handover 메타데이터를 저장하는 recorder다.
    metadata_recorder = HandoverMetadataRecorder()
    # 영상 녹화 서비스는 옵션이 켜졌을 때만 생성하므로 우선 None으로 둔다.
    video_recorder = None
    # 로봇 제어 worker 역시 follow 옵션이 켜졌을 때만 생성하므로 우선 None으로 둔다.
    robot_worker = None
    # 로봇 worker가 없거나 아직 상태를 못 받은 경우를 위한 기본 상태 객체다.
    robot_status = RobotStatus()
    # task_ready 이벤트를 중복 처리하지 않기 위해 마지막으로 처리한 epoch를 보관한다.
    last_task_ready_epoch = 0
    # reset 완료 이벤트를 중복 처리하지 않기 위해 마지막으로 처리한 epoch를 보관한다.
    last_reset_done_epoch = 0
    # task_done 이벤트를 중복 처리하지 않기 위해 마지막으로 처리한 epoch를 보관한다.
    last_task_done_epoch = 0
    # reset이 끝났을 때 runtime profile을 폐기할 사유를 임시로 저장한다.
    pending_reset_reason = None
    # grasp/place 요청이 이미 들어갔는지 추적해 중복 명령을 막는다.
    grasp_request_pending = False
    # 현재 perception fallback 상태가 어느 task epoch에 해당하는지 추적한다.
    active_task_epoch = None
    # 현재 task의 녹화 시작 perf_counter 시각이다.
    task_record_start_perf = None
    # 3D 디버그가 요청됐고 비활성화 옵션이 없을 때만 recorder를 켠다.
    debug_3d_enabled = bool(args.debug_3d) and not bool(args.disable_debug_3d_recording)
    # raw image 저장은 3D debug recorder가 있어야 가능하므로 잘못된 조합을 경고한다.
    if args.save_image and not debug_3d_enabled:
        print("[WARN] --save-image requires --3d-debug; raw image/depth capture is disabled.")
    # 3D debug recorder는 옵션이 켜졌을 때만 생성한다.
    debug_3d_recorder = None
    # 3D debug recording이 활성화된 경우 recorder를 구성한다.
    if debug_3d_enabled:
        debug_3d_recorder = Debug3DRecorder(
            # 3D debug 로그가 저장될 디렉터리다.
            output_dir=args.debug_3d_dir,
            # 이 블록에 들어왔으므로 recorder를 활성 상태로 만든다.
            enabled=True,
            # raw color/depth 프레임까지 저장할지 결정한다.
            save_images=bool(args.save_image),
            # object point cloud 저장량을 제한한다.
            max_object_points=args.debug_3d_max_object_points,
            # template point cloud 저장량을 제한한다.
            max_template_points=args.debug_3d_max_template_points,
            # template axis 정보까지 저장할지 결정한다.
            record_template_axes=bool(args.debug_3d_template_axes),
        )
        # d 키로 저장 및 reset할 수 있음을 콘솔에 안내한다.
        print(f"[INFO] 3D debug recorder armed. Press 'd' to save to {args.debug_3d_dir} and reset.")
        # raw image 저장이 켜진 경우 추가 저장 정보를 안내한다.
        if debug_3d_recorder.save_images:
            print("[INFO] 3D debug recorder will include raw color/depth frames.")

    # 영상 recorder의 web UI 사용 여부를 YAML 설정에서 가져온다.
    video_recorder_web_ui_enabled = get_video_recorder_web_ui_enabled(config)
    # 영상 recorder와 함께 촉각 로그를 남길지 YAML 설정에서 가져온다.
    video_recorder_tactile_logging = get_video_recorder_tactile_logging_config(config)
    # --record-video 옵션이 켜진 경우 녹화 서비스를 시작한다.
    if args.record_video:
        # 녹화 장치나 서버 초기화 실패가 전체 perception loop를 죽이지 않도록 보호한다.
        try:
            # handover 영상, 메타데이터, 촉각 로그를 관리하는 recorder service를 만든다.
            video_recorder = HandoverVideoRecorderService(
                # 녹화용 카메라 또는 장치의 고정 시리얼이다.
                serial=VIDEO_RECORDER_SERIAL,
                # web UI 또는 recorder server가 사용할 포트다.
                port=VIDEO_RECORDER_PORT,
                # 녹화와 task metadata를 연결하기 위한 recorder다.
                metadata_recorder=metadata_recorder,
                # web UI를 띄울지 여부다.
                web_ui_enabled=video_recorder_web_ui_enabled,
                # tactile 로그를 함께 저장하기 위한 manager다.
                tactile_manager=tactile_manager,
                # tactile logging 전체 enable flag다.
                tactile_logging_enabled=video_recorder_tactile_logging["enabled"],
                # tactile CSV 저장 여부다.
                tactile_csv_enabled=video_recorder_tactile_logging["csv_enabled"],
                # tactile rerun 로그 저장 여부다.
                tactile_rerun_enabled=video_recorder_tactile_logging["rerun_enabled"],
                # tactile rerun live streaming 여부다.
                tactile_rerun_live=video_recorder_tactile_logging["rerun_live"],
            )
            # recorder 서버를 실제로 시작하고 상태 문자열을 받는다.
            recorder_status = video_recorder.start_server()
            # web UI가 켜져 있으면 UI 접속 준비 상태를 출력한다.
            if video_recorder.is_web_ui_enabled:
                print(f"[INFO] Video recorder UI ready: {recorder_status}")
            # web UI가 꺼져 있으면 s 키로 바로 저장하는 모드임을 출력한다.
            else:
                print(f"[INFO] Video recorder ready; web UI disabled; press s to save directly. {recorder_status}")
        # recorder 초기화 실패 시 경고만 출력하고 나머지 시스템은 계속 실행한다.
        except Exception as exc:
            video_recorder = None
            print(f"[WARN] Failed to start video recorder service: {exc}")

    # 로봇 follow 제어가 켜진 경우 별도 worker thread를 생성한다.
    if args.enable_follow:
        robot_worker = RobotWorker(
            # 로봇 제어에 필요한 실행 옵션이다.
            args,
            # 메인 perception loop와 공유할 target/follow 상태다.
            shared_state,
            # 로봇 task 진행 상태를 metadata로 남기기 위한 recorder다.
            metadata_recorder=metadata_recorder,
            # grasp/place 중 tactile 정보를 참조할 수 있게 전달한다.
            tactile_manager=tactile_manager,
        )
        # 로봇 worker thread를 시작한다.
        robot_worker.start()
        # 로봇 연결과 초기 상태 설정을 요청한다.
        robot_worker.submit(RobotRequest(ROBOT_REQ_INIT_ROBOT))
        # 초기화 이후 target following을 시작하도록 요청한다.
        robot_worker.submit(RobotRequest(ROBOT_REQ_START_FOLLOW))

    # 예외나 q 종료가 발생해도 아래 finally에서 장치와 thread를 정리한다.
    try:
        # q/ESC 입력이 들어올 때까지 실시간 handover loop를 계속 돈다.
        while True:
            # wall-clock timestamp는 로그와 외부 기록용으로 사용한다.
            current_time = time.time()
            # perf_counter timestamp는 프레임 내부 경과 시간 계산에 사용한다.
            loop_perf = time.perf_counter()
            # 전체 loop 처리 시간을 profiler에 넣기 위한 시작 시각이다.
            loop_total_start = time.perf_counter()
            # 공유 상태를 읽을 때는 robot worker와의 경쟁을 피하기 위해 lock을 잡는다.
            with shared_state.lock:
                # 현재 task epoch를 읽어 frame과 metadata를 같은 task 단위로 묶는다.
                current_task_epoch = int(shared_state.task_epoch)
                # 로봇 motion이 이미 trigger됐는지 읽어 fallback 판단에 사용한다.
                motion_triggered = bool(shared_state.motion_triggered)
            # 로봇 worker가 있을 때는 상태 epoch와 에러 상태를 매 프레임 확인한다.
            if robot_worker is not None:
                # worker가 유지하는 최신 로봇 상태 snapshot을 가져온다.
                robot_status = robot_worker.get_status()
                # 로봇 reset 완료 epoch가 바뀌면 메인 thread 쪽 perception state도 reset한다.
                if robot_status.reset_done_epoch != last_reset_done_epoch:
                    # 같은 reset 이벤트를 다시 처리하지 않도록 epoch를 갱신한다.
                    last_reset_done_epoch = int(robot_status.reset_done_epoch)
                    # object/hand/fusion/fallback 등 perception pipeline 내부 상태를 초기화한다.
                    reset_perception_pipeline_for_system_reset(pipeline)
                    # reset 후 이전 3D debug buffer가 섞이지 않게 비운다.
                    if debug_3d_recorder is not None:
                        debug_3d_recorder.clear()
                    # reset 요청 때문에 profile을 버려야 한다면 여기서 폐기한다.
                    if pending_reset_reason is not None:
                        discard_runtime_profile(runtime_profiler, reason=pending_reset_reason)
                        pending_reset_reason = None
                    # reset 이후에는 이전 grasp 요청 상태를 무효화한다.
                    grasp_request_pending = False
                    # 메인 thread perception reset이 끝났음을 알린다.
                    print("[INFO] Main-thread perception reset complete after robot reset.")
                # task ready epoch가 바뀌면 새 handover trial 기록을 시작한다.
                if robot_status.task_ready_epoch != last_task_ready_epoch:
                    # 같은 task_ready 이벤트를 중복 처리하지 않도록 epoch를 갱신한다.
                    last_task_ready_epoch = int(robot_status.task_ready_epoch)
                    # metadata recorder에 task 시작 시각과 shared state snapshot을 기록한다.
                    task_ready_timestamp = metadata_recorder.mark_task_ready(shared_state)
                    # 영상과 debug 로그의 task-relative elapsed time 기준점을 잡는다.
                    task_record_start_perf = time.perf_counter()
                    # metadata 시작 timestamp를 콘솔에 남긴다.
                    print(f"[INFO] Metadata task start timestamp={task_ready_timestamp}")
                    # 영상 녹화가 켜져 있으면 task 시작 timestamp에 맞춰 녹화를 시작한다.
                    start_task_video_recording(video_recorder, task_ready_timestamp)
                # task done epoch가 바뀌면 grasp 요청 pending 상태를 해제한다.
                if robot_status.task_done_epoch != last_task_done_epoch:
                    last_task_done_epoch = int(robot_status.task_done_epoch)
                    grasp_request_pending = False
                # 로봇 에러 상태에서는 새 grasp 요청을 막는다.
                if robot_status.state == ROBOT_STATE_ERROR:
                    grasp_request_pending = False
            # profiler에 새 frame 기록을 시작한다.
            runtime_profiler.start_frame(timestamp_unix_s=current_time, task_epoch=current_task_epoch)
            # task epoch가 바뀌면 이전 task의 hand-relative fallback state를 초기화한다.
            if current_task_epoch != active_task_epoch:
                pipeline["hand_relative_fallback"].reset()
                active_task_epoch = current_task_epoch

            # 두 카메라에서 시간 동기화된 frame pair를 읽는 구간을 측정한다.
            with runtime_profiler.stage("read_pair"):
                snapshot = sensor_hub.read_next_pair()

            # cam0에서 object detection/segmentation을 수행한다.
            with runtime_profiler.stage("object_cam0"):
                object_cam0 = pipeline["object_worker_cam0"].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            # cam1에서 object detection/segmentation을 수행한다.
            with runtime_profiler.stage("object_cam1"):
                object_cam1 = pipeline["object_worker_cam1"].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
            # cam0에서 hand detector/tracker를 수행한다.
            with runtime_profiler.stage("hand_cam0"):
                hand_cam0 = pipeline["hand_worker_cam0"].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            # cam1에서 hand detector/tracker를 수행한다.
            with runtime_profiler.stage("hand_cam1"):
                hand_cam1 = pipeline["hand_worker_cam1"].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
            # 두 카메라의 hand/object 상태를 하나의 상태로 합치는 구간이다.
            with runtime_profiler.stage("merge"):
                # 두 카메라 중 현재 가장 신뢰할 수 있는 hand state를 선택한다.
                selected_hand = pipeline["hand_selector"].process_states(hand_cam0, hand_cam1)
                # 두 카메라 object state를 base 좌표계 기준 object state로 병합한다.
                merged_object = pipeline["object_merger"].process_states(
                    object_cam0,
                    object_cam1,
                    # hand_approach_detected=previous_hand_approach,
                )
            # 병합된 object point cloud에 형상 fitting/tracking을 수행한다.
            with runtime_profiler.stage("shape_fit"):
                silhouette_observations = build_silhouette_observations(snapshot, pipeline)
                shape_fitting_state = pipeline["shape_fitting_tracker"].process(
                    merged_object,
                    silhouette_observations=silhouette_observations,
                )
                # fitting 결과를 metadata recorder에 업데이트한다.
                metadata_recorder.update_geometry(shape_fitting_state, now_perf=loop_perf)
            # cam0 object worker가 남긴 debug 정보를 가져온다.
            object_debug_cam0 = getattr(pipeline["object_worker_cam0"], "last_debug", None)
            # fill-level 추정 등에 사용할 수 있는 cam0 object mask를 꺼낸다.
            cam0_mask = None if object_debug_cam0 is None else getattr(object_debug_cam0, "combined_mask", None)
            # with runtime_profiler.stage("fill_level"):
            #     fill_estimate = pipeline["fill_level_estimator"].estimate_fill_level_from_cam0(
            #         color_image_bgr=snapshot.cam0.color_image,
            #         depth_image_m=snapshot.cam0.depth_image_m,
            #         intrinsics=snapshot.cam0.intrinsics,
            #         container_mask=cam0_mask,
            #         camera_to_base=pipeline["transform_chain"].t_base_cam0,
            #         label=object_cam0.label,
            #     )
            #     metadata_recorder.update_fill_and_mass(
            #         fill_estimate,
            #         shape_fitting_state=shape_fitting_state,
            #         now_perf=loop_perf,
            #     )
            # object-hand fusion과 grasp target 계산을 같은 profiler stage로 묶는다.
            with runtime_profiler.stage("fusion_grasp"):
                # shape fitting 결과가 있으면 merged object에 fitted geometry를 반영한다.
                fitted_merged_object = build_fitted_merged_object(merged_object, shape_fitting_state)
                # object와 hand 상태를 융합해 hand 접근, latch, filtering 상태를 계산한다.
                fusion_state = pipeline["fusion"].process_states(
                    fitted_merged_object,
                    selected_hand,
                    now_timestamp=current_time,
                )
                # 융합된 상태를 바탕으로 로봇이 잡을 목표 grasp point를 계산한다.
                grasp_target = pipeline["grasp_planner"].process_states(
                    fitted_merged_object,
                    selected_hand,
                    fusion_state,
                )
                # previous_hand_approach = bool(
                #     fusion_state.hand_approach_detected or fusion_state.hand_approach_latched
                # )

                # filtering된 centroid가 있으면 우선 사용하고, 없으면 fitted object centroid를 사용한다.
                measured_object_point_base = choose_point(
                    fusion_state.filtered_object_centroid_base,
                    fitted_merged_object.centroid_base,
                )
                # planner가 유효한 grasp target을 냈을 때만 grasp point를 사용한다.
                measured_grasp_point_base = grasp_target.target_position_base if grasp_target.valid else None
                # 실제 로봇 grasp 위치 보정을 위해 base 좌표계 y축 offset을 적용한다.
                measured_grasp_point_base = offset_point_base_mm(
                    measured_grasp_point_base,
                    y_mm=GRASP_POINT_Y_OFFSET_MM,
                )
            # task recording이 시작된 뒤 현재 frame이 몇 초 지났는지 계산한다.
            frame_record_elapsed_s = (
                None
                if task_record_start_perf is None
                else max(float(loop_perf) - float(task_record_start_perf), 0.0)
            )
            # object/grasp 측정이 끊겼을 때 hand-relative fallback을 계산한다.
            with runtime_profiler.stage("fallback"):
                hand_relative_fallback_state = pipeline["hand_relative_fallback"].process(
                    # 실제 측정된 object 위치다.
                    measured_object_position_base=measured_object_point_base,
                    # 실제 측정된 grasp 위치다.
                    measured_grasp_position_base=measured_grasp_point_base,
                    # 현재 선택된 hand state다.
                    selected_hand=selected_hand,
                    # object-hand fusion 결과다.
                    fusion_state=fusion_state,
                    # 로봇 motion trigger 여부다.
                    motion_triggered=motion_triggered,
                    # 현재 wall-clock timestamp다.
                    now_timestamp=current_time,
                    # 현재 동기화 frame id다.
                    frame_id=snapshot.pair_index,
                    # task recording 기준 elapsed time이다.
                    record_elapsed_s=frame_record_elapsed_s,
                )

            # 기본적으로는 measured object point를 최종 object point로 사용한다.
            object_point_base = measured_object_point_base
            # 기본적으로는 measured grasp point를 최종 grasp point로 사용한다.
            grasp_point_base = measured_grasp_point_base
            # 최종 target이 실제 측정값에서 왔음을 표시한다.
            measurement_source = "measured"
            # object 측정이 끊겼고 fallback이 유효하면 hand-relative 예측값으로 대체한다.
            if measured_object_point_base is None and hand_relative_fallback_state.valid:
                object_point_base = hand_relative_fallback_state.object_position_base
                grasp_point_base = hand_relative_fallback_state.grasp_position_base
                measurement_source = "hand_fallback"

            # 최종 object base 좌표를 cam0 이미지 픽셀로 재투영해 overlay와 shared state에 사용한다.
            object_pixel = project_base_point_to_cam0(
                object_point_base,
                snapshot.cam0.intrinsics,
                pipeline["t_cam0_base"],
                snapshot.cam0.color_image.shape[1],
                snapshot.cam0.color_image.shape[0],
            )

            # 현재 end-effector pose는 필요할 때만 robot status에서 읽는다.
            eef_pose_base = None
            # shared state에는 end-effector xyz를 mm 단위로 넣기 위해 별도 변수로 둔다.
            eef_xyz_mm = None
            # object target이 있거나 3D debug 기록이 켜져 있으면 로봇 pose가 필요하다.
            need_robot_pose = object_point_base is not None or debug_3d_recorder is not None
            # 로봇 worker가 있고 pose가 필요한 frame에서만 status를 다시 읽는다.
            if robot_worker is not None and need_robot_pose:
                # robot status read 시간도 profiler stage로 기록한다.
                with runtime_profiler.stage("robot_read"):
                    robot_status = robot_worker.get_status()
                # status에 최신 robot pose가 있으면 tuple(float) 형태로 정규화한다.
                if robot_status.last_robot_pose is not None:
                    eef_pose_base = tuple(float(v) for v in robot_status.last_robot_pose)
                    # robot pose의 xyz[m]를 xyz[mm]로 변환한다.
                    eef_xyz_mm = meters_to_mm(eef_pose_base[:3])

            # 3D debug recorder가 켜져 있으면 현재 frame의 모든 중간 결과를 buffer에 쌓는다.
            if debug_3d_recorder is not None:
                # debug frame append 시간도 profiler에 기록한다.
                with runtime_profiler.stage("debug_3d"):
                    append_debug_3d_frame(
                        # 누적 저장할 recorder 객체다.
                        debug_3d_recorder,
                        # 현재 듀얼 카메라 frame pair다.
                        snapshot=snapshot,
                        # frame의 wall-clock timestamp다.
                        current_time=current_time,
                        # frame의 perf_counter timestamp다.
                        loop_perf=loop_perf,
                        # task recording 기준 elapsed time이다.
                        record_elapsed_s=frame_record_elapsed_s,
                        # 현재 task epoch다.
                        current_task_epoch=current_task_epoch,
                        # perception pipeline과 calibration 정보를 포함한다.
                        pipeline=pipeline,
                        # 선택된 hand state다.
                        selected_hand=selected_hand,
                        # 병합된 object state다.
                        merged_object=merged_object,
                        # shape fitting 결과다.
                        shape_fitting_state=shape_fitting_state,
                        # 최종 object target base 좌표다.
                        object_point_base=object_point_base,
                        # 최종 grasp target base 좌표다.
                        grasp_point_base=grasp_point_base,
                        # 현재 end-effector pose다.
                        eef_pose_base=eef_pose_base,
                        # measured인지 fallback인지 저장한다.
                        measurement_source=measurement_source,
                        # 촉각 상태도 함께 저장할 수 있게 전달한다.
                        tactile_manager=tactile_manager,
                    )

            # 최종 object target이 있으면 shared state를 갱신하고 필요 시 grasp/place를 시작한다.
            if object_point_base is not None:
                # target update 구간 시간을 profiler에 기록한다.
                with runtime_profiler.stage("target_update"):
                    # place z 보정을 위해 grasp/object/fitted point 샘플을 shared state에 누적한다.
                    shared_state.update_place_z_samples(
                        grasp_point_base=grasp_point_base,
                        object_point_base=object_point_base,
                        fitted_points_base=shape_fitting_state.fitted_points_base,
                    )

                    # 로봇 worker가 따라갈 최신 target을 shared state에 기록한다.
                    shared_state.update_target(
                        grasp_point_base,
                        object_point_base,
                        object_pixel,
                        eef_xyz_mm=eef_xyz_mm,
                        measurement_source=measurement_source,
                        object_label=merged_object.label,
                    )
                # 로봇이 pregrasp pose에 도달했고 아직 grasp 요청이 없으면 바로 grasp/place를 요청한다.
                if (
                    robot_worker is not None
                    and not grasp_request_pending
                    and robot_status.state == ROBOT_STATE_FOLLOWING
                    and shared_state.is_pregrasp_pose_reached(robot_status.last_robot_pose)
                ):
                    # direct grasp trigger를 콘솔에 남긴다.
                    print("[INFO] DIRECT GRASP trigger")
                    # robot worker에 넘길 fitted point cloud copy를 준비한다.
                    fitted_points_copy = None
                    # fitted point가 있으면 thread 간 공유 부작용을 피하기 위해 numpy copy를 만든다.
                    if shape_fitting_state.fitted_points_base is not None:
                        fitted_points_copy = np.asarray(shape_fitting_state.fitted_points_base, dtype=np.float32).copy()
                    # grasp point도 tuple copy로 정규화한다.
                    grasp_point_copy = None if grasp_point_base is None else tuple(float(v) for v in grasp_point_base)
                    # template axis 정보가 있으면 함께 넘긴다.
                    template_axes = getattr(shape_fitting_state, "template_axes_base", None)
                    # axis 정보 역시 numpy copy로 만들어 worker에 안전하게 전달한다.
                    template_axes_copy = None if template_axes is None else np.asarray(template_axes, dtype=np.float32).copy()
                    # grasp/place 동작에 필요한 geometry context를 하나로 묶는다.
                    action_context = RobotActionContext(
                        fitted_points_base=fitted_points_copy,
                        grasp_point_base=grasp_point_copy,
                        object_label=shape_fitting_state.label,
                        template_axes_base=template_axes_copy,
                    )
                    # robot worker에 grasp/place 시작 요청을 보낸다.
                    robot_worker.submit(
                        RobotRequest(
                            ROBOT_REQ_START_GRASP_PLACE,
                            payload={"context": action_context},
                        )
                    )
                    # 같은 pregrasp 상태에서 중복 요청하지 않도록 pending flag를 세운다.
                    grasp_request_pending = True
            # object target이 없으면 prediction/arm reset 없이 target만 비운다.
            else:
                shared_state.clear_target(reset_prediction=False, reset_arm=False)

            # FPS 계산 기준이 되는 현재 perf_counter 시각이다.
            now = time.perf_counter()
            # 직전 loop와의 시간 차로 순간 FPS를 계산한다.
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            # 화면 표시와 로그 안정성을 위해 순간 FPS를 지수 이동 평균으로 smoothing한다.
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            # 다음 frame FPS 계산을 위해 현재 시각을 저장한다.
            last_loop_time = now
            # task 시작 후 render 시점까지의 elapsed time을 계산한다.
            record_elapsed_s = None if task_record_start_perf is None else max(now - task_record_start_perf, 0.0)

            # OpenCV 화면 렌더링 구간을 profiler에 기록한다.
            with runtime_profiler.stage("render"):
                # 촉각 manager가 있으면 total norm을 계산해 최신 overlay 값으로 갱신한다.
                if tactile_manager is not None:
                    tactile_manager.total_norm()
                # cam0 화면에 object/hand/grasp/follow 상태를 overlay한 debug 이미지를 만든다.
                annotated = render_cam0_perception_debug(
                    snapshot,
                    pipeline,
                    fitted_merged_object,
                    selected_hand,
                    fusion_state,
                    object_point_base,
                    grasp_point_base,
                    shared_state,
                    record_elapsed_s=record_elapsed_s,
                    tactile_manager=tactile_manager,
                )
                # cam1 화면에는 mask와 grasp preview 중심의 보조 debug 이미지를 만든다.
                cam1_preview = render_camera_mask_preview(
                    snapshot,
                    pipeline,
                    pipeline["object_worker_cam1"],
                    selected_hand,
                    fusion_state,
                    grasp_point_base,
                    camera_label="cam1",
                )
                # cam0 debug 화면을 표시한다.
                cv.imshow(cam0_window_name, annotated)
                # cam1 preview 화면을 표시한다.
                cv.imshow(cam1_window_name, cam1_preview)
                # if args.show_depth:
                #     cv.imshow(depth_window_name, render_depth(snapshot.cam0.depth_image_m, args.depth_max_m))

            # OpenCV 키 입력 대기 시간도 profiler stage로 기록한다.
            with runtime_profiler.stage("wait_key"):
                # 1ms 동안 키 입력을 받고 하위 8bit만 사용한다.
                key = cv.waitKey(1) & 0xFF
            # frame 전체 loop 처리 시간을 profiler stage로 추가한다.
            runtime_profiler.add_stage_ms("loop_total", (time.perf_counter() - loop_total_start) * 1000.0)
            # 이번 frame의 timing과 perception metric을 runtime profile에 기록한다.
            runtime_profiler.record_frame(
                # 동기화 frame pair index다.
                frame_index=int(snapshot.pair_index),
                # frame의 wall-clock timestamp다.
                timestamp_unix_s=current_time,
                # 현재 task epoch다.
                task_epoch=current_task_epoch,
                # smoothing된 FPS다.
                fps=smoothed_fps,
                # object/hand/fusion/grasp 관련 상세 metric을 모은다.
                metrics=collect_runtime_profile_metrics(
                    snapshot=snapshot,
                    pipeline=pipeline,
                    object_cam0=object_cam0,
                    object_cam1=object_cam1,
                    merged_object=merged_object,
                    shape_fitting_state=shape_fitting_state,
                    selected_hand=selected_hand,
                    fusion_state=fusion_state,
                    grasp_target=grasp_target,
                    measurement_source=measurement_source,
                    instant_fps=instant_fps,
                    smoothed_fps=smoothed_fps,
                ),
            )
            # ESC 또는 q를 누르면 main loop를 빠져나간다.
            if key in (27, ord("q")):
                break
            # f 키는 follow pause/resume을 토글한다.
            if key == ord("f"):
                # 로봇 worker가 있으면 worker에게 follow 시작/정지를 요청한다.
                if robot_worker is not None:
                    # 현재 follow enabled 상태를 shared state snapshot에서 읽는다.
                    follow_enabled = bool(shared_state.get_snapshot()["follow_enabled"])
                    # 현재 상태에 따라 stop 또는 start request를 선택한다.
                    request_type = ROBOT_REQ_STOP_FOLLOW if follow_enabled else ROBOT_REQ_START_FOLLOW
                    # 선택한 follow request를 robot worker에 보낸다.
                    robot_worker.submit(RobotRequest(request_type))
                # 로봇 없이 perception만 돌리는 경우 shared state만 직접 토글한다.
                else:
                    shared_state.clear_follow_pause()
                    shared_state.toggle_follow()
            # r 키는 저장하지 않고 startup state로 reset한다.
            elif key == ord("r"):
                # reset 요청을 콘솔에 남긴다.
                print("[INFO] Reset requested: returning to startup state")
                # 진행 중인 영상 녹화가 있으면 저장하지 않고 버린다.
                discard_video_recording_for_reset(video_recorder)
                # task recording elapsed time 기준점을 제거한다.
                task_record_start_perf = None
                # 로봇 worker가 있으면 로봇을 home/reset sequence로 보낸다.
                if robot_worker is not None:
                    # reset 완료 후 profiler를 폐기할 사유를 저장한다.
                    pending_reset_reason = "r_key_reset"
                    # robot worker에 reset home request를 보낸다.
                    robot_worker.submit(RobotRequest(ROBOT_REQ_RESET_HOME))
                # 로봇 worker가 없으면 메인 thread에서 모든 상태를 즉시 reset한다.
                else:
                    reset_tactile_state_for_system_reset(tactile_manager)
                    shared_state.reset_for_restart(follow_enabled=False)
                    reset_perception_pipeline_for_system_reset(pipeline)
                    # debug recorder가 있으면 누적 buffer를 비운다.
                    if debug_3d_recorder is not None:
                        debug_3d_recorder.clear()
                    # 저장하지 않을 reset이므로 runtime profile도 폐기한다.
                    discard_runtime_profile(runtime_profiler, reason="r_key_reset")
            # d 키는 3D debug buffer를 저장한 뒤 reset한다.
            elif key == ord("d"):
                # 3D debug recorder가 꺼져 있으면 저장할 것이 없으므로 경고 후 다음 frame으로 간다.
                if debug_3d_recorder is None:
                    print("[WARN] 3D debug recording is disabled; launch with --3d-debug.")
                    continue
                # 저장 성공 시 경로를 담을 변수다.
                debug_save_path = None
                # debug 저장 실패가 reset까지 이어지지 않도록 보호한다.
                try:
                    # 누적된 3D debug frame을 디스크에 저장한다.
                    debug_save_path = debug_3d_recorder.save()
                    # 저장된 경로를 콘솔에 출력한다.
                    print(f"[INFO] 3D debug recording saved: {debug_save_path}")
                # 저장 실패 시 buffer를 유지하고 reset을 건너뛴다.
                except Exception as exc:
                    print(f"[WARN] 3D debug save failed; keeping buffered frames and skipping reset: {exc}")
                    continue
                # 저장이 끝난 buffer는 다음 trial과 섞이지 않게 비운다.
                debug_3d_recorder.clear()
                # debug 저장 reset에서는 일반 영상 녹화는 폐기한다.
                discard_video_recording_for_reset(video_recorder)
                # task recording elapsed time 기준점을 제거한다.
                task_record_start_perf = None
                # 로봇 worker가 있으면 reset은 worker를 통해 수행한다.
                if robot_worker is not None:
                    # reset 완료 후 profiler를 폐기할 사유를 저장한다.
                    pending_reset_reason = "d_key_debug_reset"
                    # robot worker에 reset home request를 보낸다.
                    robot_worker.submit(RobotRequest(ROBOT_REQ_RESET_HOME))
                # 로봇 worker가 없으면 메인 thread에서 즉시 reset한다.
                else:
                    reset_tactile_state_for_system_reset(tactile_manager)
                    shared_state.reset_for_restart(follow_enabled=False)
                    reset_perception_pipeline_for_system_reset(pipeline)
                    # debug reset으로 끝난 profile은 저장하지 않고 폐기한다.
                    discard_runtime_profile(runtime_profiler, reason="d_key_debug_reset")
            # s 키는 현재 task를 저장하고 follow를 멈춘다.
            elif key == ord("s"):
                # 로봇 worker가 있으면 저장 후 정지 sequence를 worker에 맡긴다.
                if robot_worker is not None:
                    robot_worker.submit(RobotRequest(ROBOT_REQ_SAVE_AND_STOP))
                # 로봇 worker가 없으면 shared state에서 follow를 직접 멈춘다.
                else:
                    shared_state.request_follow_pause()
                    shared_state.stop_follow()
                # metadata recorder가 있으면 completion row를 CSV에 append한다.
                if metadata_recorder is not None:
                    csv_path = metadata_recorder.record_completion()
                    # 새 metadata가 저장된 경우 경로를 출력한다.
                    if csv_path is not None:
                        print(f"[INFO] Handover metadata appended: {csv_path}")
                    # 이미 저장됐거나 저장할 task가 없으면 안내만 한다.
                    else:
                        print("[INFO] Handover metadata was already saved or no task metadata is available.")
                # 영상 recorder가 있으면 녹화를 마무리한다.
                if video_recorder is not None:
                    # web UI 모드에서는 pending 상태로 넘겨 사용자가 UI에서 확인/저장하게 한다.
                    if video_recorder.is_web_ui_enabled:
                        ok, message = video_recorder.finish_recording_to_pending()
                    # web UI가 없으면 현재 녹화를 바로 저장한다.
                    else:
                        ok, message = video_recorder.finish_and_save_plain_recording()
                    # recorder 결과에 따라 로그 레벨을 나눈다.
                    level = "[INFO]" if ok else "[WARN]"
                    # recorder 처리 결과 메시지를 출력한다.
                    print(f"{level} Video recorder: {message}")
                    # web UI 모드에서는 저장 후보를 확인할 UI를 연다.
                    if video_recorder.is_web_ui_enabled:
                        open_video_recorder_ui(video_recorder)
                # s 키 저장 시 runtime profile을 디스크에 저장한다.
                save_runtime_profile(runtime_profiler, reason="s_key")

    # loop 종료, 예외, KeyboardInterrupt 등 어떤 경우에도 리소스를 정리한다.
    finally:
        # runtime profiler 내부 sampling thread/file handle을 닫는다.
        runtime_profiler.close()
        # 3D debug recorder가 있으면 열려 있는 writer나 buffer를 정리한다.
        if debug_3d_recorder is not None:
            debug_3d_recorder.close()
        # 공유 stop event를 세워 background 루프들이 종료 조건을 볼 수 있게 한다.
        shared_state.stop_event.set()
        # 로봇 worker가 있으면 shutdown request를 보내고 thread 종료를 기다린다.
        if robot_worker is not None:
            robot_worker.submit(RobotRequest(ROBOT_REQ_SHUTDOWN))
            robot_worker.join(timeout=2.0)
        # tactile manager가 있으면 센서 수집 thread와 연결을 닫는다.
        if tactile_manager is not None:
            tactile_manager.close()
        # hand worker와 sensor hub는 중첩 finally로 최대한 모두 닫히게 한다.
        try:
            # cam0 hand worker를 닫는다.
            pipeline["hand_worker_cam0"].close()
        finally:
            try:
                # cam1 hand worker를 닫는다.
                pipeline["hand_worker_cam1"].close()
            finally:
                # hand worker close 중 예외가 나도 카메라 스트리밍은 반드시 멈춘다.
                sensor_hub.stop()
        # 영상 recorder service가 떠 있으면 서버와 장치를 정리한다.
        if video_recorder is not None:
            video_recorder.stop()
        # 열려 있는 모든 OpenCV 창을 닫는다.
        cv.destroyAllWindows()


# 이 파일을 직접 실행했을 때만 main loop를 시작한다.
if __name__ == "__main__":
    main()
