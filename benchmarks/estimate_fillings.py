import argparse
import time
import sys
import pickle
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from object_pt_extraction.segmentation_engine import (
    SegmentationEngine,
    format_instance_summary,
    parse_prompt_classes,
    select_instances,
)
from utils.realsense_stream import RealSenseCamera, list_realsense_serials

from collections import deque

height_hist_euclid = deque(maxlen=15)
height_hist_axis = deque(maxlen=15)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOE segmentation on RealSense and measure only rice occupied height."
    )
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="Model name or local weights path.")
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=None,
        help="Text prompt classes for YOLOE, e.g. --prompt cup or --prompt cup,drinking glass",
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
        default="highest_score",
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
        "--calib-pkl",
        default="/home/sebin/handover_2026_ICRA/calibration/cameras_robot.pckl",
        help="Path to camera-to-robot 4x4 transform pickle.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05, help="Minimum valid depth.")
    parser.add_argument("--max-valid-depth-m", type=float, default=2.0, help="Maximum valid depth.")
    parser.add_argument("--ema-alpha", type=float, default=1.0, help="EMA smoothing factor for 3D point.")
    parser.add_argument("--depth-patch-half", type=int, default=4, help="Patch half-size for depth median.")

    # rice-top estimation
    parser.add_argument("--roi-shrink-x", type=float, default=0.18, help="How much to shrink cup mask horizontally.")
    parser.add_argument("--roi-shrink-top", type=float, default=0.08, help="How much to crop top of cup ROI.")
    parser.add_argument("--roi-shrink-bottom", type=float, default=0.05, help="How much to crop bottom of cup ROI.")
    parser.add_argument("--profile-k", type=int, default=3, help="Number of vertical profile zones.")
    parser.add_argument("--rice-top-bias-px", type=int, default=5, help="Move rice_top_center upward by this many pixels.")

    # bottom-center estimation
    parser.add_argument("--bottom-center-offset-px", type=int, default=5,
                        help="Use a row slightly above raw bottom for bottom center.")
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


def overlay_status(frame, model_name, fps, infer_ms, summary, extra_lines=None):
    lines = [
        f"model: {model_name}",
        f"fps: {fps:.1f} | infer: {infer_ms:.1f} ms",
        summary,
        "ESC / q: quit",
    ]
    if extra_lines:
        lines.extend(extra_lines)

    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 24)
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

    raise RuntimeError("Could not find color intrinsics in RealSenseCamera/frame_bundle.")


def get_instance_box_xyxy(instance):
    candidate_attrs = ["box_xyxy", "bbox_xyxy", "box", "bbox", "xyxy"]

    for attr in candidate_attrs:
        if hasattr(instance, attr):
            box = getattr(instance, attr)
            if box is None:
                continue
            box = np.asarray(box).reshape(-1)
            if box.size >= 4:
                x1, y1, x2, y2 = box[:4]
                return float(x1), float(y1), float(x2), float(y2)

    if hasattr(instance, "mask") and instance.mask is not None:
        ys, xs = np.where(instance.mask.astype(bool))
        if len(xs) > 0 and len(ys) > 0:
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    raise AttributeError("Could not find bbox in SegmentationInstance.")


def compute_object_3d_point(instance, depth_image_m, fx, fy, cx, cy, min_depth_m, max_depth_m, patch_half=4):
    x1, y1, x2, y2 = get_instance_box_xyxy(instance)

    h, w = depth_image_m.shape[:2]
    u = int(round((x1 + x2) * 0.5))
    v = int(round((y1 + y2) * 0.5))
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))

    x0 = max(0, u - patch_half)
    x1p = min(w, u + patch_half + 1)
    y0 = max(0, v - patch_half)
    y1p = min(h, v + patch_half + 1)

    patch = depth_image_m[y0:y1p, x0:x1p]
    valid = patch[
        np.isfinite(patch)
        & (patch > min_depth_m)
        & (patch < max_depth_m)
    ]

    if valid.size == 0:
        return None

    z = float(np.median(valid))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return {
        "pixel": (u, v),
        "camera_xyz": np.array([x, y, z], dtype=np.float32),
        "num_points": int(valid.size),
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


def draw_point_overlay(frame, point_info, robot_xyz=None):
    if point_info is None:
        return

    u, v = point_info["pixel"]
    cam_xyz = point_info["camera_xyz"]

    cv.circle(frame, (u, v), 5, (0, 255, 255), -1, cv.LINE_AA)

    text_x, text_y = 300, 40
    lines = [f"cam xyz: [{cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f}] m"]
    if robot_xyz is not None:
        lines.append(f"robot xyz: [{robot_xyz[0]:.3f}, {robot_xyz[1]:.3f}, {robot_xyz[2]:.3f}] m")

    for i, text in enumerate(lines):
        org = (text_x, text_y + i * 25)
        cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)


def render_masks_only(image_bgr, instances, alpha=0.35):
    output = image_bgr.copy()
    if not instances:
        return output

    overlay = image_bgr.copy()
    for instance in instances:
        if not hasattr(instance, "mask") or instance.mask is None:
            continue
        mask = instance.mask.astype(bool)
        if mask.shape[:2] != output.shape[:2]:
            continue
        overlay[mask] = (0, 255, 255)

    output = cv.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)
    return output


def build_inner_roi_mask_from_cup_mask(cup_mask, shrink_ratio_x=0.18, shrink_ratio_y_top=0.08, shrink_ratio_y_bottom=0.05):
    if cup_mask is None:
        return None

    ys, xs = np.where(cup_mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min + 1

    y0 = int(round(y_min + shrink_ratio_y_top * h))
    y1 = int(round(y_max - shrink_ratio_y_bottom * h))

    inner_mask = np.zeros_like(cup_mask, dtype=np.uint8)

    for y in range(y0, y1 + 1):
        row_x = np.where(cup_mask[y].astype(bool))[0]
        if len(row_x) < 4:
            continue

        x_left = row_x.min()
        x_right = row_x.max()
        width = x_right - x_left + 1

        dx = int(round(shrink_ratio_x * width))
        xl = x_left + dx
        xr = x_right - dx

        if xr > xl:
            inner_mask[y, xl:xr + 1] = 1

    return inner_mask.astype(bool)


def compute_row_features(image_bgr, roi_mask):
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    grad_x = cv.Sobel(gray_f, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray_f, cv.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    lap = cv.Laplacian(gray_f, cv.CV_32F, ksize=3)

    h, _ = gray.shape[:2]
    row_indices = []
    feat_list = []

    for y in range(h):
        xs = np.where(roi_mask[y])[0]
        if len(xs) < 8:
            continue

        vals = gray_f[y, xs]
        grads = grad_mag[y, xs]
        laps = lap[y, xs]

        feat = [
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.mean(grads)),
            float(np.std(grads)),
            float(np.var(laps)),
            float(y) / max(h - 1, 1),
        ]
        row_indices.append(y)
        feat_list.append(feat)

    if len(feat_list) == 0:
        return None, None

    return np.asarray(row_indices, dtype=np.int32), np.asarray(feat_list, dtype=np.float32)


def normalize_features(feats):
    mu = np.mean(feats, axis=0, keepdims=True)
    sigma = np.std(feats, axis=0, keepdims=True) + 1e-6
    return (feats - mu) / sigma


def segment_vertical_profile(row_indices, feats, k=3):
    if row_indices is None or feats is None or len(row_indices) < max(9, k * 3):
        return None

    feats_n = normalize_features(feats).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    compactness, labels, centers = cv.kmeans(
        feats_n, k, None, criteria, 10, cv.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(-1)

    cluster_info = []
    for c in range(k):
        ys_c = row_indices[labels == c]
        if len(ys_c) == 0:
            continue
        cluster_info.append((c, float(np.mean(ys_c))))

    if len(cluster_info) < k:
        return None

    cluster_info = sorted(cluster_info, key=lambda x: x[1])

    zone_labels = np.full_like(labels, fill_value=-1)
    for zone_id, (cluster_id, _) in enumerate(cluster_info):
        zone_labels[labels == cluster_id] = zone_id

    return {
        "row_indices": row_indices,
        "raw_labels": labels,
        "zone_labels": zone_labels,
        "cluster_info": cluster_info,
        "centers": centers,
        "compactness": compactness,
    }


def estimate_rice_surface_y_center(profile_result):
    if profile_result is None:
        return None

    row_indices = profile_result["row_indices"]
    zone_labels = profile_result["zone_labels"]

    bottom_zone_id = int(np.max(zone_labels))
    rice_rows = row_indices[zone_labels == bottom_zone_id]
    if len(rice_rows) == 0:
        return None

    return int(np.min(rice_rows))


def get_bottom_center_y(cup_mask, bottom_margin_px=5):
    ys, xs = np.where(cup_mask.astype(bool))
    if len(ys) == 0:
        return None
    y_bottom_raw = int(np.max(ys))
    h, _ = cup_mask.shape[:2]
    return int(np.clip(y_bottom_raw - bottom_margin_px, 0, h - 1))


def get_row_center_x(mask, y):
    xs = np.where(mask[y].astype(bool))[0]
    if len(xs) == 0:
        return None
    return int(np.median(xs))


def pixel_to_camera_point(u, v, depth_image_m, fx, fy, cx, cy,
                          min_depth_m=0.05, max_depth_m=2.0, patch_half=4):
    if depth_image_m is None:
        return None

    h, w = depth_image_m.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))

    x0 = max(0, u - patch_half)
    x1 = min(w, u + patch_half + 1)
    y0 = max(0, v - patch_half)
    y1 = min(h, v + patch_half + 1)

    patch = depth_image_m[y0:y1, x0:x1]
    valid = patch[
        np.isfinite(patch)
        & (patch > min_depth_m)
        & (patch < max_depth_m)
    ]
    if valid.size == 0:
        return None

    z = float(np.median(valid))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def pixel_to_robot_point(u, v, depth_image_m, fx, fy, cx, cy, C_camera_to_robot,
                         min_depth_m=0.05, max_depth_m=2.0, patch_half=4):
    cam_pt = pixel_to_camera_point(
        u, v, depth_image_m, fx, fy, cx, cy,
        min_depth_m=min_depth_m, max_depth_m=max_depth_m, patch_half=patch_half
    )
    if cam_pt is None:
        return None
    return camera_to_robot_point(cam_pt, C_camera_to_robot)


def estimate_rice_height_metrics(
    cup_mask,
    rice_top_y_center,
    y_bottom_center,
    depth_image_m,
    fx, fy, cx, cy,
    C_camera_to_robot,
    min_depth_m=0.05,
    max_depth_m=2.0,
    patch_half=4,
):
    x_rice_top = get_row_center_x(cup_mask, rice_top_y_center)
    x_bottom_center = get_row_center_x(cup_mask, y_bottom_center)

    if x_rice_top is None or x_bottom_center is None:
        return None

    p_rice_top = pixel_to_robot_point(
        x_rice_top, rice_top_y_center,
        depth_image_m, fx, fy, cx, cy, C_camera_to_robot,
        min_depth_m=min_depth_m, max_depth_m=max_depth_m, patch_half=patch_half,
    )
    p_bottom_center = pixel_to_robot_point(
        x_bottom_center, y_bottom_center,
        depth_image_m, fx, fy, cx, cy, C_camera_to_robot,
        min_depth_m=min_depth_m, max_depth_m=max_depth_m, patch_half=patch_half,
    )

    if p_rice_top is None or p_bottom_center is None:
        return None

    diff_fill = p_bottom_center - p_rice_top

    fill_h_mm_axis = float(abs(diff_fill[2]) * 1000.0)
    fill_h_mm_euclid = float(np.linalg.norm(diff_fill) * 1000.0)
    fill_h_px = int(y_bottom_center - rice_top_y_center)

    return {
        "rice_top_pixel": (int(x_rice_top), int(rice_top_y_center)),
        "bottom_center_pixel": (int(x_bottom_center), int(y_bottom_center)),
        "fill_h_px": fill_h_px,
        "fill_h_mm_axis": fill_h_mm_axis,
        "fill_h_mm_euclid": fill_h_mm_euclid,
    }


def draw_result(frame, cup_mask, rice_top_y_center=None, metrics=None):
    vis = frame.copy()

    if rice_top_y_center is not None:
        x_rice = get_row_center_x(cup_mask, rice_top_y_center)
        if x_rice is not None:
            cv.circle(vis, (x_rice, rice_top_y_center), 7, (0, 0, 255), -1, cv.LINE_AA)
            cv.putText(vis, "rice_top_center", (x_rice + 8, rice_top_y_center - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)

    if metrics is not None:
        x_bot, y_bot = metrics["bottom_center_pixel"]
        x_rice, y_rice = metrics["rice_top_pixel"]

        cv.circle(vis, (x_bot, y_bot), 7, (0, 165, 255), -1, cv.LINE_AA)
        cv.putText(vis, "bottom_center", (x_bot + 8, y_bot - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv.LINE_AA)

        cv.line(vis, (x_rice, y_rice), (x_bot, y_bot), (255, 255, 255), 2, cv.LINE_AA)

    return vis

def extract_bowl_mask_from_glass_mask(mask, width_ratio_thresh=0.45, min_run_len=20):
    """
    전체 glass mask에서 stem/base를 최대한 제외하고 bowl 영역만 남긴다.
    아이디어:
    - 각 row의 mask width 계산
    - 전체 최대 폭의 일정 비율 이상인 row만 bowl 후보
    - 위쪽에서부터 이어지는 가장 큰 contiguous run을 bowl로 사용
    """
    mask = mask.astype(bool)
    h, w = mask.shape[:2]

    row_widths = np.zeros(h, dtype=np.int32)
    for y in range(h):
        xs = np.where(mask[y])[0]
        if len(xs) > 0:
            row_widths[y] = xs.max() - xs.min() + 1

    max_width = row_widths.max()
    if max_width <= 0:
        return None

    valid_rows = row_widths >= int(round(max_width * width_ratio_thresh))

    # contiguous runs 찾기
    runs = []
    start = None
    for y in range(h):
        if valid_rows[y] and start is None:
            start = y
        elif not valid_rows[y] and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, h - 1))

    # 충분히 긴 run만 남기고, 가장 위쪽 run을 bowl로 간주
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_run_len]
    if not runs:
        return mask

    bowl_y0, bowl_y1 = runs[0]

    bowl_mask = np.zeros_like(mask, dtype=bool)
    bowl_mask[bowl_y0:bowl_y1 + 1] = mask[bowl_y0:bowl_y1 + 1]
    return bowl_mask

def get_adaptive_container_mask(container_mask):
    container_mask = container_mask.astype(bool)

    ys, xs = np.where(container_mask)
    if len(ys) == 0:
        return container_mask, "raw"

    y_min, y_max = ys.min(), ys.max()

    row_widths = compute_row_widths(container_mask)
    local_widths = row_widths[y_min:y_max + 1]

    max_width = local_widths.max()
    if max_width <= 0:
        return container_mask, "raw"

    thin_thresh = 0.35 * max_width
    thin_rows = local_widths < thin_thresh
    max_thin_run = longest_true_run(thin_rows)

    if max_thin_run >= 12:
        bowl_mask = extract_bowl_mask_from_glass_mask(container_mask)
        if bowl_mask is not None:
            return bowl_mask, "bowl_only"

    return container_mask, "raw"
    
def compute_row_widths(mask):
    h, w = mask.shape[:2]
    row_widths = np.zeros(h, dtype=np.int32)

    for y in range(h):
        xs = np.where(mask[y])[0]
        if len(xs) > 0:
            row_widths[y] = xs.max() - xs.min() + 1

    return row_widths

def longest_true_run(arr):
    best = 0
    cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

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

    try:
        while True:
            frame_bundle = camera.read()
            fx, fy, cx, cy = get_color_intrinsics(camera, frame_bundle)

            segmentation_result = segmentation_engine.predict(frame_bundle.color_image)

            selected_instances = select_instances(
                segmentation_result.instances,
                mode=args.select_mode,
                class_names=args.select_class,
            )
            summary = format_instance_summary(selected_instances)

            annotated = render_masks_only(frame_bundle.color_image, selected_instances, alpha=0.35)

            extra_lines = []
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
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_valid_depth_m,
                    patch_half=args.depth_patch_half,
                )

                if point_info is not None:
                    smoothed_camera_xyz = smooth_point(
                        point_info["camera_xyz"],
                        smoothed_camera_xyz,
                        alpha=args.ema_alpha,
                    )
                    point_info["camera_xyz"] = smoothed_camera_xyz
                    robot_xyz = camera_to_robot_point(smoothed_camera_xyz, C_camera_to_robot)
                    draw_point_overlay(annotated, point_info, robot_xyz)

                if hasattr(instance, "mask") and instance.mask is not None:
                    # cup_mask = instance.mask.astype(bool)

                    # y_bottom_center = get_bottom_center_y(
                    #     cup_mask,
                    #     bottom_margin_px=args.bottom_center_offset_px,
                    # )

                    # inner_roi_mask = build_inner_roi_mask_from_cup_mask(
                    #     cup_mask,
                    #     shrink_ratio_x=args.roi_shrink_x,
                    #     shrink_ratio_y_top=args.roi_shrink_top,
                    #     shrink_ratio_y_bottom=args.roi_shrink_bottom,
                    # )
                    cup_mask = instance.mask.astype(bool)
                    usable_mask, mask_mode = get_adaptive_container_mask(cup_mask)

                    y_bottom_center = get_bottom_center_y(
                        usable_mask,
                        bottom_margin_px=args.bottom_center_offset_px,
                    )

                    inner_roi_mask = build_inner_roi_mask_from_cup_mask(
                        usable_mask,
                        shrink_ratio_x=args.roi_shrink_x,
                        shrink_ratio_y_top=args.roi_shrink_top,
                        shrink_ratio_y_bottom=args.roi_shrink_bottom,
                    )

                    rice_profile_result = None
                    rice_top_y_center = None
                    metrics = None

                    if inner_roi_mask is not None:
                        row_indices, feats = compute_row_features(frame_bundle.color_image, inner_roi_mask)
                        if row_indices is not None:
                            rice_profile_result = segment_vertical_profile(row_indices, feats, k=args.profile_k)
                            rice_top_y_center = estimate_rice_surface_y_center(rice_profile_result)
                            if rice_top_y_center is not None:
                                rice_top_y_center = max(rice_top_y_center - args.rice_top_bias_px, 0)

                    if y_bottom_center is not None and rice_top_y_center is not None:
                        # metrics = estimate_rice_height_metrics(
                        #     cup_mask=cup_mask,
                        #     rice_top_y_center=rice_top_y_center,
                        #     y_bottom_center=y_bottom_center,
                        #     depth_image_m=frame_bundle.depth_image_m,
                        #     fx=fx, fy=fy, cx=cx, cy=cy,
                        #     C_camera_to_robot=C_camera_to_robot,
                        #     min_depth_m=args.min_depth_m,
                        #     max_depth_m=args.max_valid_depth_m,
                        #     patch_half=args.depth_patch_half,
                        # )
                        metrics = estimate_rice_height_metrics(
                            cup_mask=usable_mask,
                            rice_top_y_center=rice_top_y_center,
                            y_bottom_center=y_bottom_center,
                            depth_image_m=frame_bundle.depth_image_m,
                            fx=fx, fy=fy, cx=cx, cy=cy,
                            C_camera_to_robot=C_camera_to_robot,
                            min_depth_m=args.min_depth_m,
                            max_depth_m=args.max_valid_depth_m,
                            patch_half=args.depth_patch_half,
                        )

                    annotated = draw_result(
                        annotated,
                        usable_mask,
                        rice_top_y_center=rice_top_y_center,
                        metrics=metrics,
                    )

                    if y_bottom_center is not None:
                        extra_lines.append(f"bottom_center={y_bottom_center}")
                    if rice_top_y_center is not None:
                        extra_lines.append(f"rice_top_center={rice_top_y_center}")

                    if metrics is not None:
                        extra_lines.append(
                            f"rice_height_px={metrics['fill_h_px']}"
                        )
                        extra_lines.append(
                            f"rice_h_axis={metrics['fill_h_mm_axis']:.1f} mm"
                        )
                        extra_lines.append(
                            f"rice_h_euclid={metrics['fill_h_mm_euclid']:.1f} mm"
                        )
                        extra_lines.append(f"mask_mode={mask_mode}")
                        height_hist_axis.append(metrics["fill_h_mm_axis"])
                        height_hist_euclid.append(metrics["fill_h_mm_euclid"])

                        stable_h_euclid = float(np.percentile(height_hist_euclid, 100))
                        stable_h_axis = float(np.percentile(height_hist_axis, 100))

                        extra_lines.append(f"stable_euclid_p100={stable_h_euclid:.1f} mm")
                        extra_lines.append(f"stable_axis_p100={stable_h_axis:.1f} mm")
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now

            overlay_status(
                annotated,
                model_name=args.model if not prompt_classes else f"{args.model} ({','.join(prompt_classes)})",
                fps=smoothed_fps,
                infer_ms=segmentation_result.infer_ms,
                summary=summary,
                extra_lines=extra_lines,
            )

            cv.imshow(window_name, annotated)
            if args.show_depth:
                cv.imshow(depth_window_name, render_depth(frame_bundle.depth_image_m, args.depth_max_m))

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        camera.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()