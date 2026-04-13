"""Dual-camera grasp-target follow script adapted for UR5 RTDE control."""

import argparse
import csv
import sys
import time
import pickle
import threading
from dataclasses import replace
from pathlib import Path
from collections import deque
from datetime import datetime

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
from perception.fdct_depth_completion import (
    bilateral_filter_depth,
    FDCTDepthCompleter,
    FDCTDepthCompletionConfig,
    format_depth_completion_stats,
    resolve_checkpoint as resolve_fdct_checkpoint,
)
from perception.fusion import PerceptionFusion
from perception.grasp_target import GraspTargetPlanner
from perception.hand_relative_fallback import HandRelativeFallbackTracker
from perception.hand_selector import HandSelector
from perception.hand_worker import HandWorkerCam0, HandWorkerCam1
from perception.object_merger import ObjectMerger
from perception.object_worker import ObjectWorkerCam0, ObjectWorkerCam1
from perception.shape_fitting_tracker import ShapeFittingTracker
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
from utils.handover_metadata import HandoverMetadataRecorder
from utils.realsense_stream import list_realsense_serials


# =========================
# Constants
# =========================
DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
DEFAULT_CALIB_PATH = Path("camera_parameters/c0_to_robot.pckl")
DEFAULT_WORKSPACE_MM = {
    "x": (-600.0, 800.0),
    "y": (-200.0, 800.0),
    "z": (0.0, 800.0),
}

# object motion trigger
REFERENCE_LOCK_COUNT = 8
MOTION_TRIGGER_MM = 30.0

# control loop
DEFAULT_CONTROL_HZ = 30.0
MAX_XY_SPEED_MM_S = 200.0 # 80
MAX_Z_SPEED_MM_S = 250.0  # 80

# EEF target offset from detected object center (robot base frame)
EEF_X_OFFSET_MM = -270.0
EEF_Y_OFFSET_MM = 0.0

# grasp / place behavior
PREGRASP_X_OFFSET_MM = -120.0
HOVER_Z_OFFSET_MM = 30.0
DESCEND_EXTRA_MM = 5.0
SAFE_LIFT_EXTRA_MM = 40.0
BACKOFF_X_MM = 100.0
GRIPPER_FORCE_STOP_DELTA_N = 10.0
GRIPPER_FORCE_STOP_MIN_ELAPSED_S = 0.12
DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD = 200
BASE_POSE = {
    "x": 208.0,
    "y": 102.0,
    "z": 308.0,
    "roll": 90.0,
    "pitch": 0.0,
    "yaw": 90.0,
}


def parse_args():
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
    parser.add_argument("--show-depth", action="store_true", help="Show a second depth preview window.")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="Upper bound for depth visualization.")
    parser.add_argument(
        "--enable-fdct-depth",
        dest="fdct_depth_enabled",
        action="store_true",
        help="Use FDCT completed depth for object point clouds.",
    )
    parser.add_argument(
        "--disable-fdct-depth",
        dest="fdct_depth_enabled",
        action="store_false",
        help="Use raw RealSense depth for object point clouds.",
    )
    parser.set_defaults(fdct_depth_enabled=None)
    parser.add_argument(
        "--fdct-cameras",
        choices=["cam0", "cam1", "both", "none"],
        default=None,
        help="Camera selection for FDCT object depth completion. Defaults to config.",
    )
    parser.add_argument(
        "--fdct-checkpoint",
        default=None,
        help="Path to FDCT checkpoint. Defaults to perception.depth_completion.fdct.checkpoint.",
    )
    parser.add_argument(
        "--fdct-device",
        default=None,
        help="Torch device for FDCT: auto, cpu, cuda, cuda:0, etc. Defaults to config.",
    )
    parser.add_argument(
        "--fdct-debug-stats",
        action="store_true",
        help="Print FDCT raw/completed depth stats while running.",
    )

    # 3D point options
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config used by the existing UR5 RTDE controller.",
    )
    parser.add_argument(
        "--calib-pkl",
        default=None,
        help="Path to camera-to-robot 4x4 transform pickle. Defaults to config/existing repo calibration.",
    )
    parser.add_argument(
        "--point-mode",
        choices=["median", "mean", "centroid_depth"],
        default="median",
        help="Representative 3D point extraction mode.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05, help="Minimum valid depth.")
    parser.add_argument("--max-valid-depth-m", type=float, default=2.0, help="Maximum valid depth.")
    parser.add_argument("--ema-alpha", type=float, default=0.1, help="EMA smoothing factor for 3D point.")

    # UR5 RTDE follow options
    parser.add_argument("--enable-follow", action="store_true", help="Enable UR5 RTDE follow mode.")
    parser.add_argument("--robot-ip", type=str, default="192.168.56.101", help="Optional UR5 IP override.")
    parser.add_argument("--min-valid-count", type=int, default=3, help="Min consecutive valid detections before follow.")
    parser.add_argument("--target-timeout-s", type=float, default=0.5, help="Stop following if target is stale.")
    parser.add_argument("--workspace-x", nargs=2, type=float, default=None, help="Workspace X limits in mm.")
    parser.add_argument("--workspace-y", nargs=2, type=float, default=None, help="Workspace Y limits in mm.")
    parser.add_argument("--workspace-z", nargs=2, type=float, default=None, help="Workspace Z limits in mm.")
    parser.add_argument("--move-to-base", action="store_true", help="Reserved for compatibility; current pose is used as baseline.")
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
        "--target-log-dir",
        type=str,
        default="logs/target_points",
        help="Directory where per-task target point histories are saved as CSV.",
    )
    return parser.parse_args()


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_config_defaults(args, config):
    robot_cfg = config.get("robot", {})
    live_cfg = robot_cfg.get("live_follow", {})
    rtde_cfg = robot_cfg.get("rtde", {})
    safety_cfg = config.get("safety", {})
    workspace_cfg = safety_cfg.get("workspace_bounds_m", {})
    cameras_cfg = config.get("cameras", {})
    cam0_cfg = cameras_cfg.get("cam0", {})
    fdct_cfg = config.get("perception", {}).get("depth_completion", {}).get("fdct", {})

    if args.robot_ip is None:
        args.robot_ip = rtde_cfg.get("robot_ip")
    if args.control_hz is None:
        args.control_hz = float(live_cfg.get("control_hz", DEFAULT_CONTROL_HZ))
    if args.follow_z is None:
        args.follow_z = bool(live_cfg.get("follow_z", False))

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

    if args.calib_pkl is None:
        args.calib_pkl = cam0_cfg.get("extrinsics_file", str(DEFAULT_CALIB_PATH))

    if args.fdct_depth_enabled is None:
        args.fdct_depth_enabled = bool(fdct_cfg.get("enabled", False))
    if args.fdct_cameras is None:
        args.fdct_cameras = str(fdct_cfg.get("cameras", "both")).strip().lower()
    if args.fdct_checkpoint is None:
        args.fdct_checkpoint = str(fdct_cfg.get("checkpoint", "FDCT/TransCG.tar"))
    if args.fdct_device is None:
        args.fdct_device = str(fdct_cfg.get("device", "auto"))
    args.fdct_net_width = int(fdct_cfg.get("net_width", 320))
    args.fdct_net_height = int(fdct_cfg.get("net_height", 240))
    args.fdct_depth_min = float(fdct_cfg.get("depth_min", 0.3))
    args.fdct_depth_max = float(fdct_cfg.get("depth_max", 1.5))
    args.fdct_depth_norm = float(fdct_cfg.get("depth_norm", 1.0))
    args.fdct_depth_coeff = float(fdct_cfg.get("depth_coeff", 10.0))
    args.fdct_inpaint = bool(fdct_cfg.get("inpaint", True))
    args.fdct_bilateral_enabled = bool(fdct_cfg.get("bilateral_enabled", True))
    args.fdct_bilateral_radius = max(0, int(fdct_cfg.get("bilateral_radius", 2)))
    args.fdct_bilateral_sigma_space = float(fdct_cfg.get("bilateral_sigma_space", 2.0))
    args.fdct_bilateral_zfar = float(fdct_cfg.get("bilateral_zfar", 100.0))
    args.fdct_debug_every = max(1, int(fdct_cfg.get("debug_every", 30)))
    args.fdct_fallback_to_raw = bool(fdct_cfg.get("fallback_to_raw", True))
    args.fdct_debug_stats = bool(args.fdct_debug_stats or fdct_cfg.get("debug_stats", False))

    return args


def pick_serial(serial):
    if serial:
        return serial
    serials = list_realsense_serials()
    if not serials:
        raise RuntimeError("No RealSense devices detected.")
    return serials[0]


def render_depth(depth_image_m, max_depth_m):
    max_depth_m = max(max_depth_m, 1e-6)
    clipped = np.clip(depth_image_m, 0.0, max_depth_m)
    scaled = (255.0 * clipped / max_depth_m).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)


def parse_fdct_camera_ids(camera_selection):
    selection = str(camera_selection or "none").strip().lower()
    if selection == "none":
        return set()
    if selection == "cam0":
        return {0}
    if selection == "cam1":
        return {1}
    if selection == "both":
        return {0, 1}
    raise ValueError(f"Unsupported --fdct-cameras value: {camera_selection}")


def build_fdct_depth_completer(args):
    if not args.fdct_depth_enabled or args.fdct_cameras == "none":
        return None

    checkpoint_path = resolve_fdct_checkpoint(args.fdct_checkpoint)
    config = FDCTDepthCompletionConfig(
        checkpoint_path=checkpoint_path,
        width=int(args.width),
        height=int(args.height),
        net_width=int(args.fdct_net_width),
        net_height=int(args.fdct_net_height),
        depth_min=float(args.fdct_depth_min),
        depth_max=float(args.fdct_depth_max),
        depth_norm=float(args.fdct_depth_norm),
        depth_coeff=float(args.fdct_depth_coeff),
        inpaint=bool(args.fdct_inpaint),
    )

    try:
        completer = FDCTDepthCompleter(config, device_arg=args.fdct_device)
    except Exception as exc:
        print(f"[WARN] FDCT depth completion disabled; failed to initialize: {exc}")
        return None

    print(
        "[INFO] FDCT depth completion enabled for "
        f"{args.fdct_cameras}: checkpoint={checkpoint_path}, device={completer.device}"
    )
    return completer


def _should_log_fdct_event(pipeline, key, every=30):
    counts = pipeline.setdefault("fdct_event_counts", {})
    counts[key] = int(counts.get(key, 0)) + 1
    return counts[key] <= 3 or counts[key] % max(1, int(every)) == 0


def apply_fdct_depth_to_object_frames(snapshot, pipeline, args):
    completer = pipeline.get("fdct_depth_completer")
    camera_ids = pipeline.get("fdct_camera_ids", set())
    if completer is None or not camera_ids:
        return snapshot.cam0, snapshot.cam1

    object_frames = {0: snapshot.cam0, 1: snapshot.cam1}
    for camera_id in sorted(camera_ids):
        frame_bundle = object_frames[camera_id]
        try:
            result = completer.complete(frame_bundle.color_image, frame_bundle.depth_image_m)
        except Exception as exc:
            if _should_log_fdct_event(pipeline, f"cam{camera_id}:exception", args.fdct_debug_every):
                print(f"[WARN] cam{camera_id} FDCT failed; using raw depth for object frame: {exc}")
            if not args.fdct_fallback_to_raw:
                raise
            continue

        if result is None:
            if _should_log_fdct_event(pipeline, f"cam{camera_id}:no_valid_depth", args.fdct_debug_every):
                print(f"[WARN] cam{camera_id} FDCT skipped; no valid depth after preprocessing.")
            continue

        filtered_depth_m = result.completed_depth_m.astype(np.float32, copy=False)
        if args.fdct_bilateral_enabled:
            try:
                filtered_depth_m = bilateral_filter_depth(
                    filtered_depth_m,
                    radius=args.fdct_bilateral_radius,
                    zfar=args.fdct_bilateral_zfar,
                    sigma_space=args.fdct_bilateral_sigma_space,
                )
            except Exception as exc:
                if _should_log_fdct_event(pipeline, f"cam{camera_id}:bilateral_exception", args.fdct_debug_every):
                    print(f"[WARN] cam{camera_id} bilateral filter failed; using FDCT output: {exc}")
                filtered_depth_m = result.completed_depth_m.astype(np.float32, copy=False)

        object_frames[camera_id] = replace(
            frame_bundle,
            depth_image_m=filtered_depth_m.astype(np.float32, copy=False),
        )
        if args.fdct_debug_stats and _should_log_fdct_event(pipeline, f"cam{camera_id}:stats", args.fdct_debug_every):
            stats = format_depth_completion_stats(
                frame_bundle.depth_image_m,
                result.completed_depth_m,
                args.fdct_depth_min,
                args.fdct_depth_max,
                filtered_depth_m=filtered_depth_m,
            )
            print(f"[FDCT] cam{camera_id} elapsed={result.elapsed_ms:.1f}ms {stats}")

    return object_frames[0], object_frames[1]


def overlay_status(frame, serial, model_name, fps, infer_ms, summary, follow_lines=None):
    lines = [
        f"model: {model_name}",
    ]
    if follow_lines is not None:
        lines.extend(follow_lines)

    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 26)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


def load_camera_to_robot_transform(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] calibration file not found: {path}")
        return None

    with open(p, "rb") as f:
        transform = pickle.load(f)

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise RuntimeError(f"Invalid transform shape: {transform.shape}, expected (4, 4)")
    print(f"[INFO] Loaded calibration: {path}")
    return transform


def get_color_intrinsics(camera, frame_bundle):
    candidates = []
    for obj in [frame_bundle, camera]:
        if obj is None:
            continue
        for attr in ["color_intrinsics", "intrinsics", "rs_intrinsics", "color_rs_intrinsics"]:
            if hasattr(obj, attr):
                candidates.append(getattr(obj, attr))

    for intr in candidates:
        if all(hasattr(intr, name) for name in ["fx", "fy", "ppx", "ppy"]):
            return float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

        if isinstance(intr, dict):
            if "fx" in intr and "fy" in intr:
                cx = intr["cx"] if "cx" in intr else intr.get("ppx")
                cy = intr["cy"] if "cy" in intr else intr.get("ppy")
                if cx is not None and cy is not None:
                    return float(intr["fx"]), float(intr["fy"]), float(cx), float(cy)

        if isinstance(intr, (list, tuple)) and len(intr) >= 4:
            return float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])

    raise RuntimeError(
        "Could not find color intrinsics in RealSenseCamera/frame_bundle. "
        "Please patch get_color_intrinsics() to match your wrapper."
    )


def erode_mask(mask, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def mask_to_point_cloud(mask, depth_image_m, fx, fy, cx, cy, min_depth_m, max_depth_m):
    valid = (
        mask.astype(bool)
        & np.isfinite(depth_image_m)
        & (depth_image_m > min_depth_m)
        & (depth_image_m < max_depth_m)
    )

    vs, us = np.where(valid)
    if len(us) == 0:
        return None, None, None

    zs = depth_image_m[vs, us].astype(np.float32)

    z_med = float(np.median(zs))
    z_keep = np.abs(zs - z_med) < 0.03
    if np.count_nonzero(z_keep) == 0:
        return None, None, None

    us = us[z_keep]
    vs = vs[z_keep]
    zs = zs[z_keep]

    xs = (us.astype(np.float32) - cx) * zs / fx
    ys = (vs.astype(np.float32) - cy) * zs / fy
    points_3d = np.stack([xs, ys, zs], axis=1)
    return points_3d, us, vs


def compute_object_3d_point(instance, depth_image_m, fx, fy, cx, cy, point_mode, min_depth_m, max_depth_m):
    mask = erode_mask(instance.mask, ksize=5)

    if point_mode == "centroid_depth":
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        u = int(np.median(xs))
        v = int(np.median(ys))

        h, w = depth_image_m.shape[:2]
        x0 = max(0, u - 2)
        x1 = min(w, u + 3)
        y0 = max(0, v - 2)
        y1 = min(h, v + 3)

        patch = depth_image_m[y0:y1, x0:x1]
        patch_valid = patch[
            np.isfinite(patch)
            & (patch > min_depth_m)
            & (patch < max_depth_m)
        ]
        if patch_valid.size == 0:
            return None

        z = float(np.median(patch_valid))
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return {
            "pixel": (u, v),
            "camera_xyz": np.array([x, y, z], dtype=np.float32),
            "num_points": int(patch_valid.size),
        }

    points_3d, us, vs = mask_to_point_cloud(mask, depth_image_m, fx, fy, cx, cy, min_depth_m, max_depth_m)
    if points_3d is None or len(points_3d) == 0:
        return None

    if point_mode == "mean":
        center_3d = np.mean(points_3d, axis=0)
    else:
        center_3d = np.median(points_3d, axis=0)

    deltas = points_3d - center_3d[None, :]
    dist2 = np.sum(deltas * deltas, axis=1)
    best_idx = int(np.argmin(dist2))

    u = int(us[best_idx])
    v = int(vs[best_idx])

    return {
        "pixel": (u, v),
        "camera_xyz": center_3d.astype(np.float32),
        "num_points": int(len(points_3d)),
    }


def smooth_point(current_xyz, previous_xyz, alpha):
    if current_xyz is None:
        return previous_xyz
    if previous_xyz is None:
        return current_xyz
    return alpha * current_xyz + (1.0 - alpha) * previous_xyz


def camera_to_robot_point(camera_xyz, camera_to_robot):
    if camera_to_robot is None or camera_xyz is None:
        return None
    p_cam = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2], 1.0], dtype=np.float64).reshape(4, 1)
    p_robot = (camera_to_robot @ p_cam).reshape(-1)
    return p_robot[:3].astype(np.float32)


def draw_point_overlay(frame, point_info, robot_xyz=None, home_pixel=None):
    if point_info is None:
        return
    u, v = point_info["pixel"]
    cam_xyz = point_info["camera_xyz"]

    cv.circle(frame, (u, v), 5, (0, 255, 255), -1, cv.LINE_AA)

    text_x, text_y = 10, 440
    lines = [f"cam xyz: [{cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f}] m"]
    if robot_xyz is not None:
        lines.append(f"robot xyz: [{robot_xyz[0]:.3f}, {robot_xyz[1]:.3f}, {robot_xyz[2]:.3f}] m")

    for i, text in enumerate(lines):
        org = (text_x, text_y + i * 25)
        cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)

    if home_pixel is not None:
        hu, hv = home_pixel
        cv.circle(frame, (hu, hv), 5, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(frame, "HOME", (hu + 10, hv - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, "HOME", (hu + 10, hv - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv.LINE_AA)


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
    dist_xy = float(np.linalg.norm(np.asarray(ref_err_xyz, dtype=np.float32)[:2]))
    if dist_xy < 55.0:
        return 5, 2.2, dist_xy
    return float(max_step_mm), float(max_step_z_mm), dist_xy


def compute_close_range_ref_step_xyz(ref_err_xyz, max_step_xy, max_step_z, *, follow_z):
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


def rpy_degrees_to_rotvec(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rotation = rot_z @ rot_y @ rot_x

    trace = float(np.trace(rotation))
    cos_theta = max(min((trace - 1.0) * 0.5, 1.0), -1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return (0.0, 0.0, 0.0)

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        diag = np.diag(rotation)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0))
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return (0.0, 0.0, 0.0)
    axis = axis / axis_norm
    rotvec = axis * theta
    return (float(rotvec[0]), float(rotvec[1]), float(rotvec[2]))


def get_base_pose_target(controller):
    state = controller.read_robot_state(now_timestamp=time.time())
    pose = state.actual_tcp_pose_base
    if pose is None:
        raise RuntimeError("Failed to read current TCP pose for BASE_POSE move.")
    return (
        mm_to_m_tuple([BASE_POSE["x"], BASE_POSE["y"], BASE_POSE["z"]]),
        tuple(float(v) for v in pose[3:6]),
    )


def make_robot_command(command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode="manual"):
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


def send_robot_command(controller, command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode="manual"):
    command = make_robot_command(
        command_type,
        target_position_base=target_position_base,
        fixed_orientation_base=fixed_orientation_base,
        gripper_action=gripper_action,
        source_mode=source_mode,
    )
    return controller.step(command, now_timestamp=command.timestamp)


def init_rtde(args):
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
        print("[INFO] move-to-base requested, but this RTDE version uses the current pose as the fixed baseline.")

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
        self.latest_target_xyz_mm = None
        self.latest_grasp_xyz_mm = None
        self.latest_measurement_source = "none"
        self.latest_target_t = 0.0
        self.last_measured_target_xyz_mm = None
        self.last_measured_target_t = 0.0
        self.valid_detection_streak = 0
        self.prediction_armed = False
        self.predicted_target_xyz_mm = None
        self.predicted_target_t = 0.0
        self.prediction_age_s = None
        self.control_target_xyz_mm = None
        self.target_source = "none"

        self.reference_object_xy_mm = None
        self.reference_locked = False
        self.motion_triggered = False
        self.reference_streak = 0

        self.fixed_z_mm = None
        self.fixed_orientation_base = None
        self.initial_pose_base = None

        self.stop_event = threading.Event()
        self.follow_pause_requested = False
        self.follow_idle_event = threading.Event()
        self.follow_idle_event.set()

        self.home_object_xyz_mm = None
        self.home_object_locked = False
        self.home_pose_buffer = deque(maxlen=15)
        self.home_object_pixel = None
        self.home_pixel_buffer = deque(maxlen=15)

        self.object_stopped = False
        self.stop_pose_buffer = deque(maxlen=15)

        self.latest_object_xyz_mm = None
        self.pregrasp_started = False
        self.grasp_closed = False
        self.grasp_offset_xyz_mm = None
        self.task_state = "FOLLOW"
        self.task_epoch = 0

    def set_fixed_pose_from_robot(self, controller):
        state = controller.read_robot_state(now_timestamp=time.time())
        pose = state.actual_tcp_pose_base
        if pose is None:
            raise RuntimeError("Failed to get current robot pose for fixed pose.")
        with self.lock:
            self.fixed_z_mm = float(pose[2] * 1000.0)
            self.fixed_orientation_base = tuple(float(v) for v in pose[3:6])
            self.initial_pose_base = tuple(float(v) for v in pose)
        print(
            "[INFO] Fixed pose set from current robot pose: "
            f"z={self.fixed_z_mm:.2f} mm, "
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

    def request_follow_pause(self):
        with self.lock:
            self.follow_pause_requested = True

    def clear_follow_pause(self):
        with self.lock:
            self.follow_pause_requested = False

    def reset_for_restart(self, *, follow_enabled=None):
        with self.lock:
            if follow_enabled is None:
                follow_enabled = self.args.enable_follow
            self.follow_enabled = bool(follow_enabled)
            self.latest_target_xyz_mm = None
            self.latest_grasp_xyz_mm = None
            self.latest_measurement_source = "none"
            self.latest_target_t = 0.0
            self.last_measured_target_xyz_mm = None
            self.last_measured_target_t = 0.0
            self.valid_detection_streak = 0
            self.prediction_armed = False
            self.reference_object_xy_mm = None
            self.reference_locked = False
            self.motion_triggered = False
            self.reference_streak = 0
            self.fixed_z_mm = None
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

    def update_target(self, grasp_xyz_m, object_xyz_m=None, pixel_xy=None, eef_xyz_mm=None, measurement_source="measured"):
        if object_xyz_m is None:
            self.clear_target(reset_prediction=False, reset_arm=False)
            return

        measurement_source = str(measurement_source or "measured").strip().lower()
        if measurement_source not in {"measured", "hand_fallback"}:
            measurement_source = "measured"

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
                    self.reference_locked = True
                    self.motion_triggered = False
                    print(f"[INFO] Reference object position locked: {self.reference_object_xy_mm}")
                self.latest_target_xyz_mm = None
                self.valid_detection_streak = 0
                self._reset_prediction_locked(reset_arm=True)
                return

            move_dist_mm = float(np.linalg.norm(object_xy_mm - self.reference_object_xy_mm))
            if not self.motion_triggered and move_dist_mm >= MOTION_TRIGGER_MM:
                self.motion_triggered = True
                print(f"[INFO] Object motion detected: {move_dist_mm:.2f} mm -> follow start")

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
                "fixed_z_mm": self.fixed_z_mm,
                "fixed_orientation_base": None if self.fixed_orientation_base is None else tuple(self.fixed_orientation_base),
                "latest_object_xyz_mm": None if self.latest_object_xyz_mm is None else self.latest_object_xyz_mm.copy(),
                "object_stopped": self.object_stopped,
                "pregrasp_started": self.pregrasp_started,
                "initial_pose_base": None if self.initial_pose_base is None else tuple(self.initial_pose_base),
                "task_state": self.task_state,
                "task_epoch": self.task_epoch,
            }

    def get_follow_status_text(self):
        snapshot = self.get_snapshot()
        mode = "ON" if snapshot["follow_enabled"] else "OFF"
        streak = snapshot["valid_detection_streak"]
        trig = "ON" if snapshot["motion_triggered"] else "WAIT"
        stopped = "YES" if snapshot["object_stopped"] else "NO"
        home = "LOCKED" if self.home_object_locked else "SEARCH"
        raw_target = "NONE" if snapshot["latest_target_xyz_mm"] is None else (
            f"[{snapshot['latest_target_xyz_mm'][0]:.1f}, {snapshot['latest_target_xyz_mm'][1]:.1f}, {snapshot['latest_target_xyz_mm'][2]:.1f}]"
        )
        control_target = "NONE" if snapshot["control_target_xyz_mm"] is None else (
            f"[{snapshot['control_target_xyz_mm'][0]:.1f}, {snapshot['control_target_xyz_mm'][1]:.1f}, {snapshot['control_target_xyz_mm'][2]:.1f}]"
        )
        prediction_age = "-" if snapshot["prediction_age_s"] is None else f"{snapshot['prediction_age_s']:.3f}s"

        return [
            f"follow: {mode} | trigger: {trig} | stopped: {stopped}",
            f"home: {home} | streak: {streak} | src: {snapshot['target_source']} | raw: {snapshot['measurement_source']}",
            f"raw_mm: {raw_target}",
            f"ctrl_mm: {control_target} | pred_age: {prediction_age}",
        ]

    def try_lock_home_pose(self, object_xyz_mm, pixel_xy):
        if self.home_object_locked:
            return

        self.home_pose_buffer.append(object_xyz_mm.copy())
        if pixel_xy is not None:
            self.home_pixel_buffer.append(np.array(pixel_xy, dtype=np.int32))

        if len(self.home_pose_buffer) < self.home_pose_buffer.maxlen:
            return

        buf = np.stack(self.home_pose_buffer, axis=0)
        xyz_range = buf.max(axis=0) - buf.min(axis=0)

        stable_xy = (xyz_range[0] < 5.0) and (xyz_range[1] < 5.0)
        stable_z = xyz_range[2] < 8.0

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

    def should_start_pregrasp(
        self,
        controller,
        x_tol_mm=180.0,
        y_tol_mm=30.0,
        z_tol_mm=30.0,
    ):
        snapshot = self.get_snapshot()
        target_xyz = snapshot["latest_grasp_xyz_mm"]
        if target_xyz is None:
            target_xyz = snapshot["latest_object_xyz_mm"]

        if target_xyz is None:
            return False

        state = controller.read_robot_state(now_timestamp=time.time())
        cur_pose = state.actual_tcp_pose_base
        if cur_pose is None:
            return False

        eef_x, eef_y, eef_z = meters_to_mm(cur_pose[:3])
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


class TaskTargetLogger:
    def __init__(self, shared_state, output_dir, sample_hz):
        self.shared_state = shared_state
        self.output_dir = Path(output_dir)
        self.sample_hz = max(float(sample_hz), 1e-6)
        self.sample_interval_s = 1.0 / self.sample_hz
        self.stop_event = threading.Event()
        self.thread = None

        self._lock = threading.Lock()
        self._active = False
        self._session_epoch = None
        self._session_index = 0
        self._sample_index = 0
        self._session_started_wall_time = None
        self._session_started_perf = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        self._completed_epochs = set()

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with self._lock:
            self._finalize_locked("shutdown")

    def abort_session(self, reason="aborted"):
        with self._lock:
            self._finalize_locked(reason)

    def _run(self):
        while not self.stop_event.is_set():
            sample_perf = time.perf_counter()
            sample_wall = time.time()
            snapshot = self.shared_state.get_snapshot(now_perf=sample_perf)

            with self._lock:
                if self._active and snapshot["task_epoch"] != self._session_epoch:
                    self._finalize_locked("reset")

                can_open = (
                    (not self._active)
                    and snapshot["task_state"] == "FOLLOW"
                    and snapshot["last_measured_target_t"] > 0.0
                    and int(snapshot["task_epoch"]) not in self._completed_epochs
                )
                if can_open:
                    self._open_locked(snapshot["task_epoch"], sample_wall, sample_perf)

                if self._active:
                    self._write_row_locked(sample_wall, sample_perf, snapshot)
                    if snapshot["task_state"] == "DONE":
                        self._finalize_locked("done")

            self.stop_event.wait(self.sample_interval_s)

    def _open_locked(self, task_epoch, sample_wall, sample_perf):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session_index += 1
        self._sample_index = 0
        self._session_epoch = int(task_epoch)
        self._session_started_wall_time = float(sample_wall)
        self._session_started_perf = float(sample_perf)
        timestamp = datetime.fromtimestamp(sample_wall).strftime("%Y%m%d_%H%M%S")
        self._csv_path = self.output_dir / f"target_points_task{self._session_index:03d}_{timestamp}.csv"
        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "sample_index",
                "wall_time_iso",
                "elapsed_s",
                "task_epoch",
                "task_state",
                "target_source",
                "measurement_age_s",
                "prediction_age_s",
                "raw_target_x_mm",
                "raw_target_y_mm",
                "raw_target_z_mm",
                "control_target_x_mm",
                "control_target_y_mm",
                "control_target_z_mm",
            ]
        )
        self._active = True
        print(f"[INFO] Target point logging started: {self._csv_path}")

    def _write_row_locked(self, sample_wall, sample_perf, snapshot):
        if self._csv_writer is None:
            return

        self._sample_index += 1
        raw_target = snapshot["latest_target_xyz_mm"]
        control_target = snapshot["control_target_xyz_mm"]
        raw_values = ["NaN", "NaN", "NaN"] if raw_target is None else [f"{float(v):.6f}" for v in raw_target]
        control_values = ["NaN", "NaN", "NaN"] if control_target is None else [f"{float(v):.6f}" for v in control_target]
        measurement_age = "NaN" if snapshot["measurement_age_s"] is None else f"{float(snapshot['measurement_age_s']):.6f}"
        prediction_age = "NaN" if snapshot["prediction_age_s"] is None else f"{float(snapshot['prediction_age_s']):.6f}"
        elapsed_s = float(sample_perf - self._session_started_perf)
        self._csv_writer.writerow(
            [
                self._sample_index,
                datetime.fromtimestamp(sample_wall).isoformat(timespec="milliseconds"),
                f"{elapsed_s:.6f}",
                snapshot["task_epoch"],
                snapshot["task_state"],
                snapshot["target_source"],
                measurement_age,
                prediction_age,
                *raw_values,
                *control_values,
            ]
        )
        self._csv_file.flush()

    def _finalize_locked(self, reason):
        if not self._active:
            return
        session_epoch = self._session_epoch
        csv_path = self._csv_path
        if self._csv_file is not None:
            self._csv_file.close()
        self._active = False
        self._session_epoch = None
        self._session_started_wall_time = None
        self._session_started_perf = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        if reason == "done" and session_epoch is not None:
            self._completed_epochs.add(int(session_epoch))
        print(f"[INFO] Target point logging finished ({reason}): {csv_path}")


def robot_control_loop(controller, shared_state, args):
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
        elif snap["fixed_z_mm"] is None or snap["fixed_orientation_base"] is None:
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
        fixed_z_mm = snap["fixed_z_mm"]
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

        if not args.follow_z:
            ref_target_xyz_mm[2] = fixed_z_mm

        ref_target_xyz_mm = ref_target_xyz_mm + ref_step_xyz
        cmd_z = float(ref_target_xyz_mm[2]) if args.follow_z else float(fixed_z_mm)
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
            if args.verbose_robot:
                print(
                    f"[ROBOT] source={target_source}, "
                    f"raw_mm={snap['latest_target_xyz_mm']}, "
                    f"control_mm={target_xyz_mm}, "
                    f"ref_mm={ref_target_xyz_mm}, "
                    f"close_range_dist_xy={dist_xy:.1f}, "
                    f"stage_axis={'xy' if dominant_axis is None else ('x' if dominant_axis == 0 else 'y')}, "
                    f"cmd_m={target_position_base}"
                )
        except Exception as exc:
            print(f"[WARN] servo command failed: {exc}")

        elapsed = time.time() - start_t
        time.sleep(max(0.0, interval - elapsed))


def wait_until_target_reached(controller, target_position_base, *, timeout_s, tolerance_m, poll_dt=0.05):
    deadline = time.time() + timeout_s
    target = np.asarray(target_position_base, dtype=np.float32).reshape(3)

    while time.time() < deadline:
        state = controller.read_robot_state(now_timestamp=time.time())
        pose = state.actual_tcp_pose_base
        if pose is not None:
            current = np.asarray(pose[:3], dtype=np.float32).reshape(3)
            error = float(np.linalg.norm(current - target))
            if error <= tolerance_m:
                return True
        time.sleep(poll_dt)
    return False


def move_robot_and_wait(controller, target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
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
    )


def stop_follow_for_handoff(controller, shared_state, timeout_s):
    shared_state.request_follow_pause()
    released = shared_state.wait_for_follow_idle(timeout_s)
    if not released:
        print(f"[WARN] Follow thread did not release robot control within {timeout_s:.2f}s")
        return False
    print("[INFO] Follow thread released robot control for pregrasp handoff")
    return True


def move_robot_to_home_pose(controller, args):
    base_target, base_orientation = get_base_pose_target(controller)
    print(
        "[INFO] Moving robot to HOME position while keeping current orientation: "
        f"x={BASE_POSE['x']:.1f}, y={BASE_POSE['y']:.1f}, z={BASE_POSE['z']:.1f}"
    )
    ok = move_robot_and_wait(
        controller,
        base_target,
        base_orientation,
        timeout_s=args.move_timeout_s,
        tolerance_m=args.position_tolerance_m,
        source_mode="startup_home_pose",
    )
    if not ok:
        raise RuntimeError("Failed to reach HOME pose during startup.")
    print("[INFO] HOME pose reached")


def reset_system_to_start_state(controller, shared_state, args, target_logger=None, metadata_recorder=None):
    print("[INFO] Reset requested: returning to startup state")
    if target_logger is not None:
        target_logger.abort_session("reset")
    if controller is None:
        shared_state.reset_for_restart(follow_enabled=False)
        return

    shared_state.request_follow_pause()
    shared_state.stop_follow()
    shared_state.wait_for_follow_idle(args.follow_handoff_timeout_s)
    safe_stop_rtde(controller)

    execute_gripper_open(controller, dwell_s=args.gripper_release_dwell_s)
    move_robot_to_home_pose(controller, args)

    shared_state.reset_for_restart(follow_enabled=args.enable_follow)
    shared_state.set_fixed_pose_from_robot(controller)
    if metadata_recorder is not None:
        task_ready_timestamp = metadata_recorder.mark_task_ready(shared_state)
        print(f"[INFO] Metadata task start timestamp={task_ready_timestamp}")
    shared_state.clear_follow_pause()
    print("[INFO] Reset complete. System is back at startup state.")


def execute_pregrasp_x_only(controller, shared_state, args):
    snap = shared_state.get_snapshot()
    grasp_xyz = snap["latest_grasp_xyz_mm"]
    obj_xyz = snap["latest_object_xyz_mm"]
    fixed_orientation_base = snap["fixed_orientation_base"]

    approach_xyz = grasp_xyz if grasp_xyz is not None else obj_xyz
    if approach_xyz is None or fixed_orientation_base is None:
        print("[WARN] No grasp/object pose available for pregrasp.")
        return False

    state = controller.read_robot_state(now_timestamp=time.time())
    cur_pose = state.actual_tcp_pose_base
    if cur_pose is None:
        print("[WARN] Failed to read current robot pose.")
        return False

    cur_x, cur_y, cur_z = meters_to_mm(cur_pose[:3])
    obj_x, obj_y, obj_z = approach_xyz

    target_x = obj_x + PREGRASP_X_OFFSET_MM
    target_y = cur_y
    target_z = cur_z
    target_x, target_y, target_z = clamp_pose_mm(target_x, target_y, target_z, args)

    print("[INFO] PREGRASP start")
    print(f"[INFO] current pose: x={cur_x:.1f}, y={cur_y:.1f}, z={cur_z:.1f}")
    print(f"[INFO] object xyz:  x={obj_x:.1f}, y={obj_y:.1f}, z={obj_z:.1f}")
    print(f"[INFO] pregrasp target: x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f}")

    shared_state.stop_follow()
    if not stop_follow_for_handoff(controller, shared_state, args.follow_handoff_timeout_s):
        return False

    ok = move_robot_and_wait(
        controller,
        mm_to_m_tuple([target_x, target_y, target_z]),
        fixed_orientation_base,
        timeout_s=args.move_timeout_s,
        tolerance_m=args.position_tolerance_m,
        source_mode="pregrasp",
    )
    if not ok:
        print("[WARN] Pregrasp move timed out.")
        return False

    print("[INFO] PREGRASP reached")
    return True


def execute_gripper_close(controller, timeout_s=2.0, poll_dt=0.05, verbose=True, metadata_recorder=None):
    if verbose:
        print("[INFO] GRIPPER CLOSE start")

    baseline_state = controller.read_robot_state(now_timestamp=time.time())
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
    while time.time() < deadline:
        state = controller.read_robot_state(now_timestamp=time.time())
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
        if hasattr(controller, "get_gripper_close_state"):
            try:
                gripper_close_state = controller.get_gripper_close_state()
                if gripper_close_state is not None:
                    position_value = gripper_close_state.get("position")
                    if position_value is not None:
                        position_threshold = int(getattr(controller, "gripper_position_complete_threshold", DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD))
                        position_triggered = int(position_value) >= position_threshold
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Failed to read gripper close state: {exc}")

        if verbose:
            print(
                "[GRIPPER] "
                f"force_norm={state.tcp_force_norm_n}, "
                f"force_delta={force_delta}, "
                f"mean_joint_current={state.mean_joint_current_a}, "
                f"verified={state.grasp_verified_force_current}, "
                f"force_triggered={force_triggered}, "
                f"position_triggered={position_triggered}, "
                f"gripper_state={gripper_close_state}"
            )
        if force_triggered:
            if hasattr(controller, "stop_gripper_motion"):
                try:
                    controller.stop_gripper_motion()
                except Exception as exc:
                    print(f"[WARN] Failed to stop gripper after force trigger: {exc}")
            if metadata_recorder is not None:
                first_contact_timestamp = metadata_recorder.note_robot_first_contact()
                if first_contact_timestamp is not None:
                    print(f"[INFO] Robot first contact timestamp={first_contact_timestamp}")
            delta_str = "n/a" if force_delta is None else f"{force_delta:.3f}"
            print(
                f"[INFO] Force rise detected during close (abs={float(force_norm):.3f} N, delta={delta_str} N). "
                "Stopping gripper close."
            )
            return True
        if position_triggered:
            if hasattr(controller, "stop_gripper_motion"):
                try:
                    controller.stop_gripper_motion()
                except Exception as exc:
                    print(f"[WARN] Failed to stop gripper after position trigger: {exc}")
            if metadata_recorder is not None:
                first_contact_timestamp = metadata_recorder.note_robot_first_contact()
                if first_contact_timestamp is not None:
                    print(f"[INFO] Robot first contact timestamp={first_contact_timestamp}")
            position_threshold = int(
                getattr(controller, "gripper_position_complete_threshold", DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD)
            )
            print(
                f"[INFO] Gripper position reached threshold >= {position_threshold}. Stopping gripper close and finishing grasp stage."
            )
            return True
        time.sleep(poll_dt)

    if hasattr(controller, "stop_gripper_motion"):
        try:
            controller.stop_gripper_motion()
        except Exception:
            pass
    print("[WARN] Grasp could not be verified from RTDE force/current.")
    return False


def execute_gripper_open(controller, dwell_s=0.5, metadata_recorder=None):
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
    time.sleep(max(dwell_s, 0.0))
    return True


def save_grasp_offset(controller, shared_state):
    snap = shared_state.get_snapshot()
    obj_xyz = snap["latest_object_xyz_mm"]
    if obj_xyz is None:
        print("[WARN] Cannot save grasp offset: no latest object xyz.")
        return False

    state = controller.read_robot_state(now_timestamp=time.time())
    cur_pose = state.actual_tcp_pose_base
    if cur_pose is None:
        print("[WARN] Cannot read EEF pose for grasp offset.")
        return False

    eef_xyz = meters_to_mm(cur_pose[:3])
    grasp_offset_xyz = obj_xyz - eef_xyz

    with shared_state.lock:
        shared_state.grasp_offset_xyz_mm = grasp_offset_xyz
        shared_state.grasp_closed = True
    shared_state.set_task_state("GRASPED", reset_prediction=True, reset_arm=True)

    print(f"[INFO] grasp_offset_xyz_mm saved: {grasp_offset_xyz}")
    return True


def compute_place_target(shared_state):
    with shared_state.lock:
        home_xyz = None if shared_state.home_object_xyz_mm is None else shared_state.home_object_xyz_mm.copy()
        grasp_offset = None if shared_state.grasp_offset_xyz_mm is None else shared_state.grasp_offset_xyz_mm.copy()

    if home_xyz is None or grasp_offset is None:
        return None

    return home_xyz - grasp_offset


def execute_return_and_place(controller, shared_state, args, metadata_recorder=None):
    target_eef_xyz = compute_place_target(shared_state)
    if target_eef_xyz is None:
        print("[WARN] Cannot compute place target.")
        return False

    snap = shared_state.get_snapshot()
    fixed_orientation_base = snap["fixed_orientation_base"]
    initial_pose_base = snap["initial_pose_base"]
    if fixed_orientation_base is None:
        print("[WARN] No fixed orientation.")
        return False

    target_x, target_y, target_z = target_eef_xyz
    target_x, target_y, target_z = clamp_pose_mm(target_x, target_y, target_z, args)

    hover_z = target_z + HOVER_Z_OFFSET_MM
    hover_x, hover_y, hover_z = clamp_pose_mm(target_x, target_y, hover_z, args)

    place_z = target_z + DESCEND_EXTRA_MM
    place_x, place_y, place_z = clamp_pose_mm(target_x, target_y, place_z, args)

    print(f"[INFO] RETURN hover target: ({hover_x:.1f}, {hover_y:.1f}, {hover_z:.1f})")
    print(f"[INFO] PLACE target: ({place_x:.1f}, {place_y:.1f}, {place_z:.1f})")

    move_sequence = [
        ("return_hover", [hover_x, hover_y, hover_z]),
        ("return_place", [place_x, place_y, place_z]),
    ]
    for source_mode, pose_mm in move_sequence:
        ok = move_robot_and_wait(
            controller,
            mm_to_m_tuple(pose_mm),
            fixed_orientation_base,
            timeout_s=args.move_timeout_s,
            tolerance_m=args.position_tolerance_m,
            source_mode=source_mode,
        )
        if not ok:
            print(f"[WARN] Move timed out during {source_mode}.")
            return False

    execute_gripper_open(
        controller,
        dwell_s=args.gripper_release_dwell_s,
        metadata_recorder=metadata_recorder,
    )

    backoff_x = place_x - BACKOFF_X_MM
    backoff_x, backoff_y, backoff_z = clamp_pose_mm(backoff_x, place_y, place_z, args)
    ok = move_robot_and_wait(
        controller,
        mm_to_m_tuple([backoff_x, backoff_y, backoff_z]),
        fixed_orientation_base,
        timeout_s=args.move_timeout_s,
        tolerance_m=args.position_tolerance_m,
        source_mode="return_backoff",
    )
    if not ok:
        print("[WARN] Back off move timed out.")
        return False

    base_target, base_orientation = get_base_pose_target(controller)
    if base_target is not None:
        ok = move_robot_and_wait(
            controller,
            base_target,
            base_orientation,
            timeout_s=args.move_timeout_s,
            tolerance_m=args.position_tolerance_m,
            source_mode="return_base_pose",
        )
        if not ok:
            print("[WARN] Return to BASE_POSE timed out.")
            return False
    elif initial_pose_base is not None:
        initial_target = tuple(float(v) for v in initial_pose_base[:3])
        initial_orientation = tuple(float(v) for v in initial_pose_base[3:6])
        ok = move_robot_and_wait(
            controller,
            initial_target,
            initial_orientation,
            timeout_s=args.move_timeout_s,
            tolerance_m=args.position_tolerance_m,
            source_mode="return_initial_pose",
        )
        if not ok:
            print("[WARN] Return to initial pose timed out.")
            return False

    if metadata_recorder is not None:
        home_xyz = None if shared_state.home_object_xyz_mm is None else shared_state.home_object_xyz_mm.copy()
        metadata_recorder.note_delivery_location(home_xyz)

    shared_state.set_task_state("DONE", reset_prediction=True, reset_arm=True)
    if metadata_recorder is not None:
        csv_path = metadata_recorder.record_completion()
        if csv_path is not None:
            print(f"[INFO] Handover metadata appended: {csv_path}")

    print("[INFO] RETURN + PLACE done")
    return True


def configure_object_worker_from_args(worker, args, prompt_classes):
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
    transform_chain = load_transform_chain(args.config)
    t_cam0_base = np.linalg.inv(transform_chain.t_base_cam0).astype(np.float32)
    t_cam1_base = np.linalg.inv(transform_chain.t_base_cam1).astype(np.float32)
    fdct_camera_ids = parse_fdct_camera_ids(args.fdct_cameras)
    fdct_depth_completer = build_fdct_depth_completer(args) if fdct_camera_ids else None

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
        "transform_chain": transform_chain,
        "t_cam0_base": t_cam0_base,
        "t_cam1_base": t_cam1_base,
        "fdct_camera_ids": fdct_camera_ids,
        "fdct_depth_completer": fdct_depth_completer,
        "fdct_event_counts": {},
        "prompt_classes": prompt_classes,
    }


def build_fitted_merged_object(raw_merged_object, shape_fitting_state):
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


def format_vec3(values):
    if values is None:
        return "None"
    vec = tuple(float(v) for v in values[:3])
    return "(%.3f, %.3f, %.3f)" % vec


def choose_point(primary, fallback=None):
    return primary if primary is not None else fallback


def _summarize_class_names(class_names):
    if not class_names:
        return "-"
    counts = {}
    for class_name in class_names:
        key = str(class_name)
        counts[key] = counts.get(key, 0) + 1
    parts = []
    for key in sorted(counts):
        count = counts[key]
        parts.append(f"{key}x{count}" if count > 1 else key)
    return ",".join(parts)


def get_segmentation_class_summary(object_worker):
    debug = getattr(object_worker, "last_debug", None)
    if debug is None:
        return "-", "-"
    all_summary = _summarize_class_names(getattr(debug, "all_class_names", ()))
    selected_summary = _summarize_class_names(getattr(debug, "selected_class_names", ()))
    return all_summary, selected_summary


def render_camera_mask_preview(
    snapshot,
    pipeline,
    object_worker,
    selected_hand,
    fusion_state,
    grasp_target,
    display_grasp_point,
    *,
    camera_label,
):
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
    for point_base, color_bgr, label in draw_specs:
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
        cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv.LINE_AA)

    title = f"{camera_label} view"
    cv.putText(image_bgr, title, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image_bgr, title, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    all_summary, selected_summary = get_segmentation_class_summary(object_worker)
    info_lines = [
        f"seg all: {all_summary}",
        f"seg selected: {selected_summary}",
    ]
    for line_index, text_line in enumerate(info_lines):
        origin = (12, 54 + line_index * 22)
        cv.putText(image_bgr, text_line, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image_bgr, text_line, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    return image_bgr


def render_cam0_perception_debug(
    snapshot,
    pipeline,
    merged_object,
    raw_merged_object,
    shape_fitting_state,
    selected_hand,
    fusion_state,
    grasp_target,
    display_object_point,
    display_grasp_point,
    hand_relative_fallback_state,
    shared_state,
    model_label,
    fps,
):
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

    lines = [
        f"model: {model_label}",
        f"fps: {fps:.1f}",
        f"merged_obj={bool(merged_object.valid)} points={merged_object.merged_point_count} proj={len(merged_pixels)}",
        (
            "shape_fit="
            f"{bool(shape_fitting_state.valid)} init={bool(shape_fitting_state.initialized)} "
            f"template={shape_fitting_state.template_id or '-'} "
            f"scale={'-' if shape_fitting_state.scale is None else f'{shape_fitting_state.scale:.3f}'}"
        ),
        (
            "shape_fit_reason="
            f"{shape_fitting_state.reason} raw_points={raw_merged_object.merged_point_count} "
            f"fitted_points={len(np.asarray(shape_fitting_state.fitted_points_base, dtype=np.float32).reshape((-1, 3)))}"
        ),
        f"hand={bool(selected_hand.valid)} cam={selected_hand.selected_camera} handed={selected_hand.handedness}",
        (
            "grasp_valid="
            f"{bool(grasp_target.valid)} grasp={format_vec3(grasp_target.target_position_base)} "
            f"grasp_hold={bool(getattr(grasp_target, 'used_temporal_hold', False))} "
            f"cand_idx={int(getattr(grasp_target, 'selected_candidate_index', -1))} "
            f"xy_lock={bool(getattr(grasp_target, 'xy_locked_to_centroid', False))}"
        ),
        (
            "hand_fallback="
            f"{bool(getattr(hand_relative_fallback_state, 'valid', False))} "
            f"obj={format_vec3(getattr(hand_relative_fallback_state, 'object_position_base', None))} "
            f"grasp={format_vec3(getattr(hand_relative_fallback_state, 'grasp_position_base', None))} "
            f"age={'-' if getattr(hand_relative_fallback_state, 'dropout_age_s', None) is None else f'{float(hand_relative_fallback_state.dropout_age_s):.3f}s'} "
            f"reason={getattr(hand_relative_fallback_state, 'reason', None)}"
        ),
        f"object={format_vec3(object_point)}",
        f"hand={format_vec3(hand_point)}",
    ]
    cam0_all_summary, cam0_selected_summary = get_segmentation_class_summary(pipeline["object_worker_cam0"])
    lines.append(f"cam0 seg all={cam0_all_summary}")
    lines.append(f"cam0 seg selected={cam0_selected_summary}")
    lines.extend(shared_state.get_follow_status_text())

    for line_index, text_line in enumerate(lines):
        origin = (12, 28 + line_index * 24)
        cv.putText(image_bgr, text_line, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image_bgr, text_line, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    return image_bgr


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    args = apply_config_defaults(args, config)

    if args.serial is not None:
        print("[WARN] --serial is ignored in the dual-camera grasp-target mode. Camera selection comes from configs/handover.yaml")

    pipeline = build_dual_perception_pipeline(args)
    sensor_hub = pipeline["sensor_hub"]
    sensor_hub.start()

    model_label = args.model if not pipeline["prompt_classes"] else f"{args.model} ({','.join(pipeline['prompt_classes'])})"
    if pipeline.get("fdct_depth_completer") is not None:
        model_label = f"{model_label} + FDCT depth({args.fdct_cameras})"
    window_name = "cam0_grasp_target_follow"
    cam1_window_name = "cam1_view"
    depth_window_name = "cam0_depth"

    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()
    previous_hand_approach = False

    controller = None
    shared_state = FollowSharedState(args)
    target_logger = TaskTargetLogger(shared_state, args.target_log_dir, args.fps)
    metadata_recorder = HandoverMetadataRecorder()
    target_logger.start()
    control_thread = None
    active_task_epoch = None

    if args.enable_follow:
        controller = init_rtde(args)
        move_robot_to_home_pose(controller, args)
        shared_state.set_fixed_pose_from_robot(controller)
        task_ready_timestamp = metadata_recorder.mark_task_ready(shared_state)
        print(f"[INFO] Metadata task start timestamp={task_ready_timestamp}")

        control_thread = threading.Thread(
            target=robot_control_loop,
            args=(controller, shared_state, args),
            daemon=True,
        )
        control_thread.start()

    try:
        while True:
            current_time = time.time()
            loop_perf = time.perf_counter()
            with shared_state.lock:
                current_task_epoch = int(shared_state.task_epoch)
                motion_triggered = bool(shared_state.motion_triggered)
            if current_task_epoch != active_task_epoch:
                pipeline["hand_relative_fallback"].reset()
                active_task_epoch = current_task_epoch

            snapshot = sensor_hub.read_next_pair()
            object_frame_cam0, object_frame_cam1 = apply_fdct_depth_to_object_frames(snapshot, pipeline, args)

            object_cam0 = pipeline["object_worker_cam0"].process_frame(object_frame_cam0, frame_id=snapshot.pair_index)
            object_cam1 = pipeline["object_worker_cam1"].process_frame(object_frame_cam1, frame_id=snapshot.pair_index)
            hand_cam0 = pipeline["hand_worker_cam0"].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            hand_cam1 = pipeline["hand_worker_cam1"].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
            selected_hand = pipeline["hand_selector"].process_states(hand_cam0, hand_cam1)
            merged_object = pipeline["object_merger"].process_states(
                object_cam0,
                object_cam1,
                hand_approach_detected=previous_hand_approach,
            )
            shape_fitting_state = pipeline["shape_fitting_tracker"].process(merged_object)
            metadata_recorder.update_geometry(shape_fitting_state, now_perf=loop_perf)
            fitted_merged_object = build_fitted_merged_object(merged_object, shape_fitting_state)
            fusion_state = pipeline["fusion"].process_states(
                fitted_merged_object,
                selected_hand,
                now_timestamp=current_time,
            )
            grasp_target = pipeline["grasp_planner"].process_states(
                fitted_merged_object,
                selected_hand,
                fusion_state,
            )
            previous_hand_approach = bool(
                fusion_state.hand_approach_detected or fusion_state.hand_approach_latched
            )

            measured_object_point_base = choose_point(
                fusion_state.filtered_object_centroid_base,
                fitted_merged_object.centroid_base,
            )
            measured_grasp_point_base = grasp_target.target_position_base if grasp_target.valid else None
            hand_relative_fallback_state = pipeline["hand_relative_fallback"].process(
                measured_object_position_base=measured_object_point_base,
                measured_grasp_position_base=measured_grasp_point_base,
                selected_hand=selected_hand,
                fusion_state=fusion_state,
                motion_triggered=motion_triggered,
                now_timestamp=current_time,
            )

            object_point_base = measured_object_point_base
            grasp_point_base = measured_grasp_point_base
            measurement_source = "measured"
            if measured_object_point_base is None and hand_relative_fallback_state.valid:
                object_point_base = hand_relative_fallback_state.object_position_base
                grasp_point_base = hand_relative_fallback_state.grasp_position_base
                measurement_source = "hand_fallback"

            object_pixel = project_base_point_to_cam0(
                object_point_base,
                snapshot.cam0.intrinsics,
                pipeline["t_cam0_base"],
                snapshot.cam0.color_image.shape[1],
                snapshot.cam0.color_image.shape[0],
            )

            if object_point_base is not None:
                eef_xyz_mm = None
                if controller is not None:
                    state = controller.read_robot_state(now_timestamp=time.time())
                    if state.actual_tcp_pose_base is not None:
                        eef_xyz_mm = meters_to_mm(state.actual_tcp_pose_base[:3])

                shared_state.update_target(
                    grasp_point_base,
                    object_point_base,
                    object_pixel,
                    eef_xyz_mm=eef_xyz_mm,
                    measurement_source=measurement_source,
                )
                if controller is not None and shared_state.should_start_pregrasp(controller):
                    print("[INFO] DIRECT GRASP trigger")
                    shared_state.stop_follow()
                    if stop_follow_for_handoff(controller, shared_state, args.follow_handoff_timeout_s):
                        grasp_ok = execute_gripper_close(
                            controller,
                            timeout_s=args.gripper_close_timeout_s,
                            verbose=True,
                            metadata_recorder=metadata_recorder,
                        )
                        print(f"[INFO] grasp_ok = {grasp_ok}")
                        if grasp_ok:
                            save_grasp_offset(controller, shared_state)
                            execute_return_and_place(
                                controller,
                                shared_state,
                                args,
                                metadata_recorder=metadata_recorder,
                            )
            else:
                shared_state.clear_target(reset_prediction=False, reset_arm=False)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now

            annotated = render_cam0_perception_debug(
                snapshot,
                pipeline,
                fitted_merged_object,
                merged_object,
                shape_fitting_state,
                selected_hand,
                fusion_state,
                grasp_target,
                object_point_base,
                grasp_point_base,
                hand_relative_fallback_state,
                shared_state,
                model_label,
                smoothed_fps,
            )
            cam1_preview = render_camera_mask_preview(
                snapshot,
                pipeline,
                pipeline["object_worker_cam1"],
                selected_hand,
                fusion_state,
                grasp_target,
                grasp_point_base,
                camera_label="cam1",
            )
            cv.imshow(window_name, annotated)
            cv.imshow(cam1_window_name, cam1_preview)
            if args.show_depth:
                cv.imshow(depth_window_name, render_depth(snapshot.cam0.depth_image_m, args.depth_max_m))

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("f"):
                shared_state.clear_follow_pause()
                shared_state.toggle_follow()
            elif key == ord("r"):
                try:
                    reset_system_to_start_state(
                        controller,
                        shared_state,
                        args,
                        target_logger=target_logger,
                        metadata_recorder=metadata_recorder,
                    )
                except Exception as exc:
                    print(f"[WARN] Reset failed: {exc}")
            elif key == ord("s"):
                shared_state.request_follow_pause()
                shared_state.stop_follow()
                shared_state.wait_for_follow_idle(args.follow_handoff_timeout_s)
                safe_stop_rtde(controller)

    finally:
        shared_state.stop_event.set()
        if control_thread is not None:
            control_thread.join(timeout=1.0)
        target_logger.stop()
        safe_stop_rtde(controller)
        disconnect_rtde(controller)
        try:
            pipeline["hand_worker_cam0"].close()
        finally:
            try:
                pipeline["hand_worker_cam1"].close()
            finally:
                sensor_hub.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
