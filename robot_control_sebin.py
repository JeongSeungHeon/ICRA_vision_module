# grasp 까지
#python object_pt_extraction/follow_mode_final.py   --prompt "wine glass" glass  --select-mode highest_score  


import argparse
import time
import sys
import pickle
import threading
from pathlib import Path
from collections import deque

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np
from xarm.wrapper import XArmAPI

from object_pt_extraction.segmentation_engine import (
    SegmentationEngine,
    format_instance_summary,
    parse_prompt_classes,
    select_instances,
)
from utils.realsense_stream import RealSenseCamera, list_realsense_serials


# =========================
# Constants
# =========================
WORKSPACE = {
    "x": (200.0, 650.0),   # mm
    "y": (-120.0, 210.0),  # mm
    "z": (50.0, 700.0),    # mm
}

BASE_POSE = {
    "x": 397.0,
    "y": 8.0,
    "z": 135.0,
    "roll": -128.0,
    "pitch": 90.0,
    "yaw": -128.0,
}

# object motion trigger
REFERENCE_LOCK_COUNT = 8
MOTION_TRIGGER_MM = 15.0

# control loop
CONTROL_HZ =30.0
MAX_XY_SPEED_MM_S = 80
MAX_STEP_MM = MAX_XY_SPEED_MM_S / CONTROL_HZ
MAX_Z_SPEED_MM_S = 80
MAX_STEP_Z_MM = MAX_Z_SPEED_MM_S / CONTROL_HZ

# EEF target offset from detected object center (robot frame)
EEF_X_OFFSET_MM = -80.0
EEF_Y_OFFSET_MM = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOE segmentation on RealSense and follow object with xArm in a separate control loop."
    )
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="Model name or local weights path.")
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=None,
        help="Text prompt classes for YOLOE, e.g. --prompt person bus or --prompt person,bus",
    )
    parser.add_argument("--serial", default=None, help="RealSense serial number. Defaults to the first detected camera.")
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

    # 3D point options
    parser.add_argument(
        "--calib-pkl",
        default="/home/sebin/handover_2026_ICRA/calibration/cameras_robot.pckl",
        help="Path to camera-to-robot 4x4 transform pickle.",
    )
    parser.add_argument(
        "--point-mode",
        choices=["median", "mean", "centroid_depth"],
        default="median",
        help="Representative 3D point extraction mode.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05, help="Minimum valid depth.")
    parser.add_argument("--max-valid-depth-m", type=float, default=2.0, help="Maximum valid depth.")
    parser.add_argument("--ema-alpha", type=float, default=0.2, help="EMA smoothing factor for 3D point.")

    # xArm follow options
    parser.add_argument("--enable-follow", action="store_true", help="Enable xArm follow mode.")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.218", help="XArm IP address")
    parser.add_argument("--min-valid-count", type=int, default=3, help="Min consecutive valid detections before follow.")
    parser.add_argument("--target-timeout-s", type=float, default=0.5, help="Stop following if target is stale.")
    parser.add_argument("--workspace-x", nargs=2, type=float, default=list(WORKSPACE["x"]), help="Workspace X limits in mm.")
    parser.add_argument("--workspace-y", nargs=2, type=float, default=list(WORKSPACE["y"]), help="Workspace Y limits in mm.")
    parser.add_argument("--workspace-z", nargs=2, type=float, default=list(WORKSPACE["z"]), help="Workspace Z limits in mm.")
    parser.add_argument("--move-to-base", action="store_true", help="Move robot to current preset base pose before follow.")
    parser.add_argument("--open-gripper", action="store_true", help="Open gripper during init.")
    parser.add_argument("--verbose-robot", action="store_true", help="Print detailed robot command logs.")
    parser.add_argument("--follow-z",action="store_true",help="Enable z-axis follow. If omitted, z stays fixed at the current robot z.")
    return parser.parse_args()


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


def overlay_status(frame, serial, model_name, fps, infer_ms, summary, follow_lines=None):
    lines = [
        f"model: {model_name}",
        # f"serial: {serial}",
        #f"fps: {fps:.1f}  infer: {infer_ms:.1f} ms",
        #summary,
        #"ESC / q: quit | f: toggle follow | s: stop follow",
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
        C = pickle.load(f)

    C = np.asarray(C, dtype=np.float64)
    if C.shape != (4, 4):
        raise RuntimeError(f"Invalid transform shape: {C.shape}, expected (4, 4)")
    print(f"[INFO] Loaded calibration: {path}")
    return C


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


def camera_to_robot_point(camera_xyz, C_camera_to_robot):
    if C_camera_to_robot is None or camera_xyz is None:
        return None
    p_cam = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2], 1.0], dtype=np.float64).reshape(4, 1)
    p_robot = (C_camera_to_robot @ p_cam).reshape(-1)
    return p_robot[:3].astype(np.float32)

def draw_point_overlay(frame, point_info, robot_xyz=None, home_pixel=None):
    if point_info is None: return
    u, v = point_info["pixel"]
    cam_xyz = point_info["camera_xyz"]

    # 1. 포인트는 그대로 찍어줌 (시각적 확인용)
    cv.circle(frame, (u, v), 5, (0, 255, 255), -1, cv.LINE_AA)

    # 2. 텍스트 위치를 화면 왼쪽 상단(20, 40)으로 고정
    text_x, text_y = 10, 440
    
    lines = [f"cam xyz: [{cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f}] m"]
    if robot_xyz is not None:
        lines.append(f"robot xyz: [{robot_xyz[0]:.3f}, {robot_xyz[1]:.3f}, {robot_xyz[2]:.3f}] m")

    for i, text in enumerate(lines):
        org = (text_x, text_y + i * 25)
        # 가독성을 위한 검정색 외곽선 + 노란색 글자
        cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)
    
    if home_pixel is not None:
        hu, hv = home_pixel
        cv.circle(frame, (hu, hv), 5, (0, 0, 255), -1, cv.LINE_AA)   # 빨간 점
        cv.putText(frame, "HOME", (hu + 10, hv - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, "HOME", (hu + 10, hv - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv.LINE_AA)

def clamp_value(v, low, high):
    return max(low, min(high, v))


def clamp_pose_mm(x_mm, y_mm, z_mm, args):
    x_mm = clamp_value(x_mm, args.workspace_x[0], args.workspace_x[1])
    y_mm = clamp_value(y_mm, args.workspace_y[0], args.workspace_y[1])
    z_mm = clamp_value(z_mm, args.workspace_z[0], args.workspace_z[1])
    return x_mm, y_mm, z_mm


def init_xarm(ip, args):
    print(f"[INFO] Connecting to XArm7 at {ip} ...")
    arm = XArmAPI(ip, is_radian=False)
    arm.connect()

    if arm.has_error:
        print(f"[INFO] Clearing robot error: {arm.error_code}")
        arm.clean_error()
        time.sleep(0.5)

    if arm.has_warn:
        print(f"[INFO] Clearing robot warning: {arm.warn_code}")
        arm.clean_warn()
        time.sleep(0.1)

    arm.motion_enable(enable=True)
    time.sleep(0.2)
    arm.set_mode(1)
    time.sleep(0.2)
    arm.set_state(0)
    time.sleep(0.5)
    arm.set_state(0)
    time.sleep(0.5)

    print(f"[DEBUG] robot state after init: state={arm.state}, mode={arm.mode}, has_error={arm.has_error}, has_warn={arm.has_warn}")

    ok, pos = arm.get_position()
    if ok == 0:
        print(
            f"[INFO] Current pose: "
            f"x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}, "
            f"roll={pos[3]:.1f}, pitch={pos[4]:.1f}, yaw={pos[5]:.1f}"
        )

    if args.move_to_base:
        print("[INFO] move-to-base requested, but this threaded version uses current pose as fixed pose baseline.")

    if args.open_gripper:
        try:
            code, ret = arm.robotiq_set_position(0, speed=255, force=50, wait=True)
            print(f"[INFO] robotiq_set_position(open) => code={code}, ret={ret}")
        except Exception as e:
            print(f"[WARN] Failed to open gripper: {e}")

    ret = arm.core.set_modbus_timeout(100)
    print("set modbus timeout ret:", ret[0] if isinstance(ret, (list, tuple)) else ret)

    ret = arm.core.set_modbus_baudrate(115200)
    print("set modbus baudrate ret:", ret[0] if isinstance(ret, (list, tuple)) else ret)
    activate_gripper(arm)
    return arm


def safe_stop_xarm(arm):
    if arm is None:
        return
    try:
        arm.set_state(4)
    except Exception:
        pass


def disconnect_xarm(arm):
    if arm is None:
        return
    try:
        arm.disconnect()
    except Exception:
        pass


def send_servo_cartesian(arm, pose6, verbose=False):
    code = arm.set_servo_cartesian(pose6, is_radian=False)
    if verbose:
        print(f"[ROBOT] set_servo_cartesian(pose6, is_radian=False) => {code}")
    return code


class FollowSharedState:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()

        self.follow_enabled = args.enable_follow
        self.latest_target_xyz_mm = None
        self.latest_target_t = 0.0
        self.valid_detection_streak = 0

        self.reference_object_xy_mm = None
        self.reference_locked = False
        self.motion_triggered = False
        self.reference_streak = 0

        self.fixed_z_mm = None
        self.fixed_rpy_deg = None

        self.stop_event = threading.Event()

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
        self.task_state = "FOLLOW"   # FOLLOW, PREGRASP, GRASPED, RETURNING, PLACING, DONE
        

    def set_fixed_pose_from_robot(self, arm):
        ok, p = arm.get_position()
        if ok != 0:
            raise RuntimeError(f"Failed to get current robot pose for fixed pose. code={ok}")
        with self.lock:
            self.fixed_z_mm = float(p[2])
            self.fixed_rpy_deg = [float(p[3]), float(p[4]), float(p[5])]
        print(
            f"[INFO] Fixed pose set from current robot pose: "
            f"z={p[2]:.2f}, roll={p[3]:.2f}, pitch={p[4]:.2f}, yaw={p[5]:.2f}"
        )

    def toggle_follow(self):
        with self.lock:
            self.follow_enabled = not self.follow_enabled
            print(f"[INFO] follow_enabled = {self.follow_enabled}")

    def stop_follow(self):
        with self.lock:
            self.follow_enabled = False
            print("[INFO] Robot follow stopped.")

    def clear_target(self):
        with self.lock:
            self.latest_target_xyz_mm = None
            self.valid_detection_streak = 0
            self.reference_streak = 0

    def update_target(self, robot_xyz_m, pixel_xy):
        if robot_xyz_m is None:
            self.clear_target()
            return

        robot_xyz_mm = np.asarray(robot_xyz_m, dtype=np.float32) * 1000.0
        object_xyz_mm = robot_xyz_mm[:3].copy()
        object_xy_mm = object_xyz_mm[:2].copy()
        
        self.try_lock_home_pose(object_xyz_mm, pixel_xy)
        self.update_stop_state(object_xyz_mm)

        # EEF target offset from object
        target_xyz_mm = object_xyz_mm.copy()
        target_xyz_mm[0] += EEF_X_OFFSET_MM
        target_xyz_mm[1] += EEF_Y_OFFSET_MM
        #target_xyz_mm[2] += self.args.eef_z_offset_mm

        with self.lock:
            self.latest_object_xyz_mm = object_xyz_mm.copy()
            self.valid_detection_streak += 1

            if not self.reference_locked:
                self.reference_streak += 1
                if self.reference_streak >= REFERENCE_LOCK_COUNT:
                    self.reference_object_xy_mm = object_xy_mm.copy()
                    self.reference_locked = True
                    self.motion_triggered = False
                    print(f"[INFO] Reference object position locked: {self.reference_object_xy_mm}")
                return

            move_dist_mm = float(np.linalg.norm(object_xy_mm - self.reference_object_xy_mm))
            if not self.motion_triggered and move_dist_mm >= MOTION_TRIGGER_MM:
                self.motion_triggered = True
                print(f"[INFO] Object motion detected: {move_dist_mm:.2f} mm -> follow start")

            self.latest_target_xyz_mm = target_xyz_mm
            self.latest_target_t = time.perf_counter()

    def get_snapshot(self):
        with self.lock:
            return {
                "follow_enabled": self.follow_enabled,
                "latest_target_xyz_mm": None if self.latest_target_xyz_mm is None else self.latest_target_xyz_mm.copy(),
                "latest_target_t": self.latest_target_t,
                "valid_detection_streak": self.valid_detection_streak,
                "motion_triggered": self.motion_triggered,
                "fixed_z_mm": self.fixed_z_mm,
                "fixed_rpy_deg": None if self.fixed_rpy_deg is None else list(self.fixed_rpy_deg),
                "latest_object_xyz_mm": None if self.latest_object_xyz_mm is None else self.latest_object_xyz_mm.copy(),
                "object_stopped": self.object_stopped,
                "pregrasp_started": self.pregrasp_started,
            }

    def get_follow_status_text(self):
        with self.lock:
            mode = "ON" if self.follow_enabled else "OFF"
            streak = self.valid_detection_streak
            trig = "ON" if self.motion_triggered else "WAIT"
            stopped = "YES" if self.object_stopped else "NO"
            home = "LOCKED" if self.home_object_locked else "SEARCH"
            target = "NONE" if self.latest_target_xyz_mm is None else \
                f"[{self.latest_target_xyz_mm[0]:.1f}, {self.latest_target_xyz_mm[1]:.1f}, {self.latest_target_xyz_mm[2]:.1f}]"

        return [
        f"follow: {mode} | trigger: {trig} | stopped: {stopped}",
        f"home: {home} | streak: {streak}",
        f"target: {target}",
        ]

    def try_lock_home_pose(self, object_xyz_mm, pixel_xy):
        """
        물체가 책상 위에 가만히 있을 때 최근 K프레임 평균으로 home pose를 저장한다.
        최근 버퍼의 xyz 변화폭이 작으면 안정적이라고 판단하고 lock한다.
        """
        if self.home_object_locked:
            return

        self.home_pose_buffer.append(object_xyz_mm.copy())
        self.home_pixel_buffer.append(np.array(pixel_xy, dtype=np.int32))

        if len(self.home_pose_buffer) < self.home_pose_buffer.maxlen:
            return

        buf = np.stack(self.home_pose_buffer, axis=0)   # [K, 3]
        xyz_range = buf.max(axis=0) - buf.min(axis=0)   # [dx, dy, dz]

        stable_xy = (xyz_range[0] < 5.0) and (xyz_range[1] < 5.0)
        stable_z = (xyz_range[2] < 8.0)

        if stable_xy and stable_z:
            self.home_object_xyz_mm = buf.mean(axis=0)

            pix_buf = np.stack(self.home_pixel_buffer, axis=0)   # [K, 2]
            home_pix = np.median(pix_buf, axis=0).astype(np.int32)
            self.home_object_pixel = (int(home_pix[0]), int(home_pix[1]))

            self.home_object_locked = True
            print(f"[INFO] Home object position locked: {self.home_object_xyz_mm}")
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

        stable_xy = (xyz_range[0] < 8.0) and (xyz_range[1] < 8.0)
        stable_z = (xyz_range[2] < 10.0)

        if stable_xy and stable_z:
            if not self.object_stopped:
                print(f"[INFO] Object STOPPED detected. xyz_range={xyz_range}")
            self.object_stopped = True
        else:
            self.object_stopped = False
    
    def should_start_pregrasp(self, arm, xy_thresh_mm=30.0):
        with self.lock:
            if not self.object_stopped or self.pregrasp_started:
                return False
            target_xyz = None if self.latest_target_xyz_mm is None else self.latest_target_xyz_mm.copy()

        if target_xyz is None:
            return False

        ok, cur_pose = arm.get_position()
        if ok != 0:
            return False

        cur_x, cur_y = cur_pose[0], cur_pose[1]
        target_x, target_y = target_xyz[0], target_xyz[1]

        dxy = float(np.linalg.norm(np.array([cur_x - target_x, cur_y - target_y])))

        if dxy < xy_thresh_mm:
            with self.lock:
                self.pregrasp_started = True
            print(f"[INFO] PREGRASP condition met. dxy={dxy:.1f}")
            return True
        print(f"[DEBUG] pregrasp check dxy={dxy:.1f}, thresh={xy_thresh_mm}")
        return False

def robot_control_loop(arm, shared_state, args):
    interval = 1.0 / CONTROL_HZ
    last_sent_pose = None
    ref_target_xyz_mm = None

    while not shared_state.stop_event.is_set():
        start_t = time.time()
        snap = shared_state.get_snapshot()

        if not snap["follow_enabled"]:
            time.sleep(max(0.0, interval - (time.time() - start_t)))
            continue

        if snap["latest_target_xyz_mm"] is None:
            time.sleep(max(0.0, interval - (time.time() - start_t)))
            continue

        if snap["valid_detection_streak"] < args.min_valid_count:
            time.sleep(max(0.0, interval - (time.time() - start_t)))
            continue

        if not snap["motion_triggered"]:
            time.sleep(max(0.0, interval - (time.time() - start_t)))
            continue

        if (time.perf_counter() - snap["latest_target_t"]) > args.target_timeout_s:
            time.sleep(max(0.0, interval - (time.time() - start_t)))
            continue

        target_xyz_mm = snap["latest_target_xyz_mm"]
        fixed_z_mm = snap["fixed_z_mm"]
        fixed_rpy_deg = snap["fixed_rpy_deg"]

        if fixed_z_mm is None or fixed_rpy_deg is None:
            time.sleep(max(0.0, interval - (time.time() - start_t)))
            continue

        # 아주 중요: 내부 reference는 현재 로봇 위치에서 시작
        if ref_target_xyz_mm is None:
            ok, p = arm.get_position()
            if ok != 0:
                time.sleep(max(0.0, interval - (time.time() - start_t)))
                continue
            ref_target_xyz_mm = np.array([p[0], p[1], p[2]], dtype=np.float32)
            print(f"[INFO] ref_target initialized from current EEF xyz: {ref_target_xyz_mm}")
            
        # 내부 reference가 실제 target을 아주 천천히 따라감
        ref_err_xyz = target_xyz_mm - ref_target_xyz_mm

        ref_step_xyz = np.zeros(3, dtype=np.float32)
        ref_step_xyz[0:2] = np.clip(ref_err_xyz[0:2], -MAX_STEP_MM, MAX_STEP_MM)

        if args.follow_z:
            ref_step_xyz[2] = np.clip(ref_err_xyz[2], -MAX_STEP_Z_MM, MAX_STEP_Z_MM)
        else:
            ref_target_xyz_mm[2] = fixed_z_mm

        ref_target_xyz_mm = ref_target_xyz_mm + ref_step_xyz

        cmd_z = float(ref_target_xyz_mm[2]) if args.follow_z else float(fixed_z_mm)

        pose6 = [
            float(ref_target_xyz_mm[0]),
            float(ref_target_xyz_mm[1]),
            cmd_z,
            float(fixed_rpy_deg[0]),
            float(fixed_rpy_deg[1]),
            float(fixed_rpy_deg[2]),
        ]

        pose6[0], pose6[1], pose6[2] = clamp_pose_mm(pose6[0], pose6[1], pose6[2], args)

        # 이전 명령과 거의 같으면 skip
        new_pose = np.array(pose6, dtype=np.float32)
        if last_sent_pose is not None:
            pos_delta = np.linalg.norm(new_pose[:3] - last_sent_pose[:3])
            ang_delta = np.linalg.norm(new_pose[3:] - last_sent_pose[3:])
            if pos_delta < 0.2 and ang_delta < 0.5:
                time.sleep(max(0.0, interval - (time.time() - start_t)))
                continue

        code = send_servo_cartesian(arm, pose6, verbose=False)
        if code == 0:
            last_sent_pose = new_pose
            if args.verbose_robot:
                print(f"[ROBOT] target={target_xyz_mm}, ref={ref_target_xyz_mm}, pose6={pose6}")
        else:
            print(f"[WARN] set_servo_cartesian failed: code={code}")

        elapsed = time.time() - start_t
        time.sleep(max(0.0, interval - elapsed))

def execute_pregrasp_x_only(arm, shared_state, args):
    snap = shared_state.get_snapshot()
    obj_xyz = snap["latest_object_xyz_mm"]
    fixed_rpy_deg = snap["fixed_rpy_deg"]

    if obj_xyz is None or fixed_rpy_deg is None:
        print("[WARN] No object pose available for pregrasp.")
        return False

    # 현재 pose 읽기
    ok, cur_pose = arm.get_position()
    if ok != 0:
        print(f"[WARN] Failed to read current robot pose. code={ok}")
        return False

    cur_x, cur_y, cur_z = cur_pose[:3]

    # object 실제 center 기준
    obj_x, obj_y, obj_z = obj_xyz

    # x축만 접근: y,z는 현재값 유지
    # 방향은 좌표계에 맞게 조정 필요
    PREGRASP_X_OFFSET_MM =10.0

    target_x = obj_x + PREGRASP_X_OFFSET_MM
    target_y = cur_y
    target_z = cur_z

    target_x, target_y, target_z = clamp_pose_mm(target_x, target_y, target_z, args)

    print(f"[INFO] PREGRASP start")
    print(f"[INFO] current pose: x={cur_x:.1f}, y={cur_y:.1f}, z={cur_z:.1f}")
    print(f"[INFO] object xyz:  x={obj_x:.1f}, y={obj_y:.1f}, z={obj_z:.1f}")
    print(f"[INFO] pregrasp target: x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f}")

    # follow 끄고 position mode로 전환
    shared_state.stop_follow()

    arm.set_mode(0)
    time.sleep(0.2)
    arm.set_state(0)
    time.sleep(0.2)

    code = arm.set_position(
        x=target_x,
        y=target_y,
        z=target_z,
        roll=float(fixed_rpy_deg[0]),
        pitch=float(fixed_rpy_deg[1]),
        yaw=float(fixed_rpy_deg[2]),
        speed=50,
        wait=True,
    )

    if code != 0:
        print(f"[WARN] Pregrasp move failed. code={code}")
        return False

    print("[INFO] PREGRASP reached")
    return True

SLAVE_ID = 0x09

CLOSE_SPEED = 0x40
CLOSE_FORCE = 0x30
CLOSE_POS = 0xFF


def fc16_write_3regs_from_03E8(arm: XArmAPI, b0, b1, b2, b3, b4, b5):
    frame = [
        SLAVE_ID,
        0x10,
        0x03, 0xE8,
        0x00, 0x03,
        0x06,
        b0 & 0xFF, b1 & 0xFF, b2 & 0xFF,
        b3 & 0xFF, b4 & 0xFF, b5 & 0xFF,
    ]
    ret = arm.core.tgpio_set_modbus(frame, len(frame))
    code = ret[0] if isinstance(ret, (list, tuple)) and len(ret) > 0 else ret
    return code, ret


def fc04_read_input_regs_from_07D0(arm: XArmAPI, reg_count=2):
    frame = [
        SLAVE_ID,
        0x04,
        0x07, 0xD0,
        0x00, reg_count & 0xFF,
    ]
    ret = arm.core.tgpio_set_modbus(frame, len(frame))
    code = ret[0] if isinstance(ret, (list, tuple)) and len(ret) > 0 else ret
    return code, ret


def parse_gripper_status(ret):
    if ret is None or len(ret) < 9:
        return None

    data = ret[5:9]
    byte0, byte1, byte2, byte3 = data

    gOBJ = (byte0 >> 6) & 0x03
    gSTA = (byte0 >> 4) & 0x03
    gGTO = (byte0 >> 3) & 0x01
    gACT = (byte0 >> 0) & 0x01

    return {
        "gOBJ": gOBJ,
        "gSTA": gSTA,
        "gGTO": gGTO,
        "gACT": gACT,
        "raw_data": data,
    }


def gripper_goto_position(arm: XArmAPI, pos, speed=0x80, force=0x80):
    action = 0x09
    code, ret = fc16_write_3regs_from_03E8(
        arm, action, 0x00, 0x00, pos, speed, force
    )
    return code, ret

def activate_gripper(arm: XArmAPI):
    # clear rACT
    code, _ = fc16_write_3regs_from_03E8(arm, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
    print("[clear rACT] code:", code)
    time.sleep(0.2)

    # set rACT
    code, _ = fc16_write_3regs_from_03E8(arm, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00)
    print("[set rACT] code:", code)
    time.sleep(2.0)

def execute_gripper_close(
    arm: XArmAPI,
    close_pos: int = CLOSE_POS,
    speed: int = CLOSE_SPEED,
    force: int = CLOSE_FORCE,
    timeout_s: float = 5.0,
    poll_dt: float = 0.02,
    verbose: bool = True,
):
    """
    닫기 명령 후 polling 하다가 gOBJ == 2 (object detected) 나오면 바로 성공 종료.
    그 외에는 성공으로 보지 않음.
    """
    if verbose:
        print(f"[INFO] GRIPPER CLOSE start: pos=0x{close_pos:02X}, speed=0x{speed:02X}, force=0x{force:02X}")

    code, ret = gripper_goto_position(arm, close_pos, speed=speed, force=force)
    if verbose:
        print(f"[INFO] goto close ret: code={code}, ret={ret}")

    if code != 0:
        return False

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        code, ret = fc04_read_input_regs_from_07D0(arm, reg_count=2)
        if code != 0:
            time.sleep(poll_dt)
            continue

        status = parse_gripper_status(ret)
        if status is None:
            time.sleep(poll_dt)
            continue

        if verbose:
            print(f"[GRIPPER] gOBJ={status['gOBJ']}, gSTA={status['gSTA']}, raw={status['raw_data']}")

        # object detected while closing
        if status["gOBJ"] == 2:
            if verbose:
                print("[INFO] Object detected while closing.")
            return True

        time.sleep(poll_dt)

    if verbose:
        print("[WARN] Object not detected during close.")
    return False

def save_grasp_offset(arm, shared_state):
    snap = shared_state.get_snapshot()
    obj_xyz = snap["latest_object_xyz_mm"]
    if obj_xyz is None:
        print("[WARN] Cannot save grasp offset: no latest object xyz.")
        return False

    ok, cur_pose = arm.get_position()
    if ok != 0:
        print(f"[WARN] Cannot read EEF pose for grasp offset. code={ok}")
        return False

    eef_xyz = np.array(cur_pose[:3], dtype=np.float32)
    grasp_offset_xyz = obj_xyz - eef_xyz

    with shared_state.lock:
        shared_state.grasp_offset_xyz_mm = grasp_offset_xyz
        shared_state.grasp_closed = True
        shared_state.task_state = "GRASPED"

    print(f"[INFO] grasp_offset_xyz_mm saved: {grasp_offset_xyz}")
    return True

def compute_place_target(shared_state):
    with shared_state.lock:
        home_xyz = None if shared_state.home_object_xyz_mm is None else shared_state.home_object_xyz_mm.copy()
        grasp_offset = None if shared_state.grasp_offset_xyz_mm is None else shared_state.grasp_offset_xyz_mm.copy()

    if home_xyz is None or grasp_offset is None:
        return None

    target_eef_xyz = home_xyz - grasp_offset
    return target_eef_xyz

def execute_return_and_place(arm, shared_state, args):
    target_eef_xyz = compute_place_target(shared_state)
    if target_eef_xyz is None:
        print("[WARN] Cannot compute place target.")
        return False

    snap = shared_state.get_snapshot()
    fixed_rpy_deg = snap["fixed_rpy_deg"]
    if fixed_rpy_deg is None:
        print("[WARN] No fixed RPY.")
        return False

    target_x, target_y, target_z = target_eef_xyz
    target_x, target_y, target_z = clamp_pose_mm(target_x, target_y, target_z, args)

    HOVER_Z_OFFSET_MM = 30.0
    DESCEND_EXTRA_MM = 5.0   # 너무 많이 내리지 않기

    hover_z = target_z + HOVER_Z_OFFSET_MM
    hover_x, hover_y, hover_z = clamp_pose_mm(target_x, target_y, hover_z, args)

    place_z = target_z + DESCEND_EXTRA_MM
    place_x, place_y, place_z = clamp_pose_mm(target_x, target_y, place_z, args)

    print(f"[INFO] RETURN hover target: ({hover_x:.1f}, {hover_y:.1f}, {hover_z:.1f})")
    print(f"[INFO] PLACE target: ({place_x:.1f}, {place_y:.1f}, {place_z:.1f})")

    arm.set_mode(0)
    time.sleep(0.2)
    arm.set_state(0)
    time.sleep(0.2)

    # 1) hover
    code = arm.set_position(
        x=hover_x, y=hover_y, z=hover_z,
        roll=float(fixed_rpy_deg[0]),
        pitch=float(fixed_rpy_deg[1]),
        yaw=float(fixed_rpy_deg[2]),
        speed=60,
        wait=True,
    )
    if code != 0:
        print(f"[WARN] Hover move failed. code={code}")
        return False

    # 2) descend a little
    code = arm.set_position(
        x=place_x, y=place_y, z=place_z,
        roll=float(fixed_rpy_deg[0]),
        pitch=float(fixed_rpy_deg[1]),
        yaw=float(fixed_rpy_deg[2]),
        speed=60,
        wait=True,
    )
    if code != 0:
        print(f"[WARN] Place descend failed. code={code}")
        return False

    # 3) open gripper
    try:
        code, ret = arm.robotiq_set_position(0, speed=255, force=50, wait=True)
        print(f"[INFO] gripper open: code={code}, ret={ret}")
    except Exception as e:
        print(f"[WARN] Gripper open failed: {e}")
        return False

    # 4) lift more
    safe_lift_z = hover_z + 40.0
    safe_x, safe_y, safe_lift_z = clamp_pose_mm(hover_x, hover_y, safe_lift_z, args)

    arm.set_position(
        x=safe_x, y=safe_y, z=safe_lift_z,
        roll=float(fixed_rpy_deg[0]),
        pitch=float(fixed_rpy_deg[1]),
        yaw=float(fixed_rpy_deg[2]),
        speed=60,
        wait=True,
    )
    if code != 0:
        print(f"[WARN] Retreat failed. code={code}")
        return False

    # 5) back off in x
    backoff_x = safe_x - 100.0   # 방향은 좌표계 맞게 조정
    backoff_x, backoff_y, backoff_z = clamp_pose_mm(backoff_x, safe_y, safe_lift_z, args)

    arm.set_position(
        x=backoff_x, y=backoff_y, z=backoff_z,
        roll=float(fixed_rpy_deg[0]),
        pitch=float(fixed_rpy_deg[1]),
        yaw=float(fixed_rpy_deg[2]),
        speed=70,
        wait=True,
    )
    if code != 0:
        print(f"[WARN] Back off failed. code={code}")
        return False
    
    # 6) go base
    arm.set_position(
        x=BASE_POSE["x"],
        y=BASE_POSE["y"],
        z=BASE_POSE["z"],
        roll=BASE_POSE["roll"],
        pitch=BASE_POSE["pitch"],
        yaw=BASE_POSE["yaw"],
        speed=70,
        wait=True,
    )

    with shared_state.lock:
        shared_state.task_state = "DONE"

    print("[INFO] RETURN + PLACE done")
    return True

def main():
    args = parse_args()
    serial = pick_serial(args.serial)
    prompt_classes = parse_prompt_classes(args.prompt)

    segmentation_engine = SegmentationEngine(
        model_name=args.model,
        prompt_classes=prompt_classes,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        classes=args.classes,
        half=args.half,
        retina_masks=True,
    )

    camera = RealSenseCamera(serial=serial, width=args.width, height=args.height, fps=args.fps)
    C_camera_to_robot = load_camera_to_robot_transform(args.calib_pkl)

    window_name = "realsense_yoloe_seg"
    depth_window_name = "realsense_depth"

    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()
    smoothed_camera_xyz = None

    arm = None
    shared_state = FollowSharedState(args)
    control_thread = None

    if args.enable_follow:
        arm = init_xarm(args.robot_ip, args)
        shared_state.set_fixed_pose_from_robot(arm)

        control_thread = threading.Thread(
            target=robot_control_loop,
            args=(arm, shared_state, args),
            daemon=True,
        )
        control_thread.start()

    try:
        while True:
            frame_bundle = camera.read()
            fx, fy, cx, cy = get_color_intrinsics(camera, frame_bundle)

            segmentation_result = segmentation_engine.predict(frame_bundle.color_image)
            annotated = segmentation_engine.render(segmentation_result)
            selected_instances = select_instances(
                segmentation_result.instances,
                mode=args.select_mode,
                class_names=args.select_class,
            )
            summary = format_instance_summary(selected_instances)

            point_info = None
            robot_xyz = None

            if selected_instances:
                instance = selected_instances[0]
                point_info = compute_object_3d_point(
                    instance=instance,
                    depth_image_m=frame_bundle.depth_image_m,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    point_mode=args.point_mode,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_valid_depth_m,
                )

                if point_info is not None:
                    smoothed_camera_xyz = smooth_point(
                        point_info["camera_xyz"],
                        smoothed_camera_xyz,
                        alpha=args.ema_alpha,
                    )
                    point_info["camera_xyz"] = smoothed_camera_xyz
                    robot_xyz = camera_to_robot_point(smoothed_camera_xyz, C_camera_to_robot)
                    home_pixel = shared_state.home_object_pixel if shared_state.home_object_locked else None
                    draw_point_overlay(annotated, point_info, robot_xyz, home_pixel=home_pixel)
                    
                    if robot_xyz is not None:
                        shared_state.update_target(robot_xyz, point_info["pixel"])
                        if arm is not None and shared_state.should_start_pregrasp(arm):
                            print("[INFO] STOPPED + CLOSE_ENOUGH -> PREGRASP transition")
                            ok = execute_pregrasp_x_only(arm, shared_state, args)
                            if ok:
                                grasp_ok = execute_gripper_close(arm, verbose=True)
                                print(f"[INFO] grasp_ok = {grasp_ok}")
                                if grasp_ok:
                                    save_grasp_offset(arm, shared_state)
                                    execute_return_and_place(arm, shared_state, args)
                    else:
                        shared_state.clear_target()
                else:
                    shared_state.clear_target()
            else:
                shared_state.clear_target()

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now

            overlay_status(
                annotated,
                serial=frame_bundle.serial,
                model_name=args.model if not prompt_classes else f"{args.model} ({','.join(prompt_classes)})",
                fps=smoothed_fps,
                infer_ms=segmentation_result.infer_ms,
                summary=summary,
                follow_lines=shared_state.get_follow_status_text(),
            )

            cv.imshow(window_name, annotated)
            if args.show_depth:
                cv.imshow(depth_window_name, render_depth(frame_bundle.depth_image_m, args.depth_max_m))

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("f"):
                shared_state.toggle_follow()
            elif key == ord("s"):
                safe_stop_xarm(arm)
                shared_state.stop_follow()

    finally:
        shared_state.stop_event.set()
        safe_stop_xarm(arm)
        disconnect_xarm(arm)
        camera.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()