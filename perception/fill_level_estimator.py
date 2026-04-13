"""Fill-height estimation helpers reused by the handover metadata pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import cv2 as cv
except Exception:  # pragma: no cover - optional during unit tests
    cv = None
import numpy as np
try:
    import yaml
except Exception:  # pragma: no cover - optional during unit tests
    yaml = None

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
CUP_FILL_HEIGHT_BIAS_MM = 20.0


def _require_cv2():
    if cv is None:
        raise RuntimeError("OpenCV is required for fill-level estimation.")


@dataclass(frozen=True)
class FillLevelEstimate:
    valid: bool
    fill_height_mm: float | None
    mask_mode: str
    rice_top_y_center: int | None
    bottom_center_y: int | None
    usable_mask: np.ndarray | None = None
    rice_top_pixel: tuple[int, int] | None = None
    bottom_center_pixel: tuple[int, int] | None = None


def build_inner_roi_mask_from_cup_mask(cup_mask, shrink_ratio_x=0.18, shrink_ratio_y_top=0.08, shrink_ratio_y_bottom=0.05):
    if cup_mask is None:
        return None

    ys, xs = np.where(cup_mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    height = y_max - y_min + 1

    y0 = int(round(y_min + shrink_ratio_y_top * height))
    y1 = int(round(y_max - shrink_ratio_y_bottom * height))

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
    _require_cv2()
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    grad_x = cv.Sobel(gray_f, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray_f, cv.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    lap = cv.Laplacian(gray_f, cv.CV_32F, ksize=3)

    height, _ = gray.shape[:2]
    row_indices = []
    feat_list = []

    for y in range(height):
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
            float(y) / max(height - 1, 1),
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
    _require_cv2()
    if row_indices is None or feats is None or len(row_indices) < max(9, k * 3):
        return None

    feats_n = normalize_features(feats).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, centers = cv.kmeans(feats_n, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    cluster_info = []
    for cluster_id in range(k):
        ys_c = row_indices[labels == cluster_id]
        if len(ys_c) == 0:
            continue
        cluster_info.append((cluster_id, float(np.mean(ys_c))))

    if len(cluster_info) < k:
        return None

    cluster_info = sorted(cluster_info, key=lambda item: item[1])
    zone_labels = np.full_like(labels, fill_value=-1)
    for zone_id, (cluster_id, _) in enumerate(cluster_info):
        zone_labels[labels == cluster_id] = zone_id

    return {
        "row_indices": row_indices,
        "zone_labels": zone_labels,
        "cluster_info": cluster_info,
        "centers": centers,
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
    ys, _ = np.where(cup_mask.astype(bool))
    if len(ys) == 0:
        return None
    y_bottom_raw = int(np.max(ys))
    height, _ = cup_mask.shape[:2]
    return int(np.clip(y_bottom_raw - bottom_margin_px, 0, height - 1))


def get_row_center_x(mask, y):
    xs = np.where(mask[y].astype(bool))[0]
    if len(xs) == 0:
        return None
    return int(np.median(xs))


def pixel_to_camera_point(u, v, depth_image_m, fx, fy, cx, cy, min_depth_m=0.05, max_depth_m=2.0, patch_half=4):
    if depth_image_m is None:
        return None

    height, width = depth_image_m.shape[:2]
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))

    x0 = max(0, u - patch_half)
    x1 = min(width, u + patch_half + 1)
    y0 = max(0, v - patch_half)
    y1 = min(height, v + patch_half + 1)

    patch = depth_image_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > min_depth_m) & (patch < max_depth_m)]
    if valid.size == 0:
        return None

    z = float(np.median(valid))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def camera_to_base_point(camera_xyz, camera_to_base):
    if camera_to_base is None or camera_xyz is None:
        return None
    p_cam = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2], 1.0], dtype=np.float64).reshape(4, 1)
    p_base = (np.asarray(camera_to_base, dtype=np.float64).reshape((4, 4)) @ p_cam).reshape(-1)
    return p_base[:3].astype(np.float32)


def pixel_to_base_point(u, v, depth_image_m, fx, fy, cx, cy, camera_to_base, min_depth_m=0.05, max_depth_m=2.0, patch_half=4):
    cam_pt = pixel_to_camera_point(
        u,
        v,
        depth_image_m,
        fx,
        fy,
        cx,
        cy,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        patch_half=patch_half,
    )
    if cam_pt is None:
        return None
    return camera_to_base_point(cam_pt, camera_to_base)


def estimate_fill_height_metrics(
    cup_mask,
    rice_top_y_center,
    y_bottom_center,
    depth_image_m,
    fx,
    fy,
    cx,
    cy,
    camera_to_base,
    min_depth_m=0.05,
    max_depth_m=2.0,
    patch_half=4,
):
    x_rice_top = get_row_center_x(cup_mask, rice_top_y_center)
    x_bottom_center = get_row_center_x(cup_mask, y_bottom_center)
    if x_rice_top is None or x_bottom_center is None:
        return None

    p_rice_top = pixel_to_base_point(
        x_rice_top,
        rice_top_y_center,
        depth_image_m,
        fx,
        fy,
        cx,
        cy,
        camera_to_base,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        patch_half=patch_half,
    )
    p_bottom_center = pixel_to_base_point(
        x_bottom_center,
        y_bottom_center,
        depth_image_m,
        fx,
        fy,
        cx,
        cy,
        camera_to_base,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        patch_half=patch_half,
    )
    if p_rice_top is None or p_bottom_center is None:
        return None

    diff_fill = p_bottom_center - p_rice_top
    return {
        "fill_h_mm_axis": float(abs(diff_fill[2]) * 1000.0),
        "fill_h_mm_euclid": float(np.linalg.norm(diff_fill) * 1000.0),
    }


def compute_row_widths(mask):
    height, _ = mask.shape[:2]
    row_widths = np.zeros(height, dtype=np.int32)
    for y in range(height):
        xs = np.where(mask[y])[0]
        if len(xs) > 0:
            row_widths[y] = xs.max() - xs.min() + 1
    return row_widths


def longest_true_run(arr):
    best = 0
    cur = 0
    for value in arr:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def extract_bowl_mask_from_glass_mask(mask, width_ratio_thresh=0.45, min_run_len=20):
    mask = mask.astype(bool)
    height, _ = mask.shape[:2]

    row_widths = np.zeros(height, dtype=np.int32)
    for y in range(height):
        xs = np.where(mask[y])[0]
        if len(xs) > 0:
            row_widths[y] = xs.max() - xs.min() + 1

    max_width = row_widths.max()
    if max_width <= 0:
        return None

    valid_rows = row_widths >= int(round(max_width * width_ratio_thresh))

    runs = []
    start = None
    for y in range(height):
        if valid_rows[y] and start is None:
            start = y
        elif not valid_rows[y] and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, height - 1))

    runs = [run for run in runs if (run[1] - run[0] + 1) >= min_run_len]
    if not runs:
        return mask

    bowl_y0, bowl_y1 = runs[0]
    bowl_mask = np.zeros_like(mask, dtype=bool)
    bowl_mask[bowl_y0:bowl_y1 + 1] = mask[bowl_y0:bowl_y1 + 1]
    return bowl_mask


def get_adaptive_container_mask(container_mask):
    container_mask = container_mask.astype(bool)

    ys, _ = np.where(container_mask)
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


class FillLevelEstimator:
    def __init__(
        self,
        *,
        roi_shrink_x: float = 0.18,
        roi_shrink_top: float = 0.08,
        roi_shrink_bottom: float = 0.05,
        profile_k: int = 3,
        rice_top_bias_px: int = 5,
        bottom_center_offset_px: int = 5,
        min_depth_m: float = 0.05,
        max_valid_depth_m: float = 2.0,
        depth_patch_half: int = 4,
    ) -> None:
        self.roi_shrink_x = float(roi_shrink_x)
        self.roi_shrink_top = float(roi_shrink_top)
        self.roi_shrink_bottom = float(roi_shrink_bottom)
        self.profile_k = int(profile_k)
        self.rice_top_bias_px = int(rice_top_bias_px)
        self.bottom_center_offset_px = int(bottom_center_offset_px)
        self.min_depth_m = float(min_depth_m)
        self.max_valid_depth_m = float(max_valid_depth_m)
        self.depth_patch_half = int(depth_patch_half)

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "FillLevelEstimator":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load FillLevelEstimator from config.")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        fill_cfg = config.get("perception", {}).get("fill_level_estimation", {})
        return cls(
            roi_shrink_x=float(fill_cfg.get("roi_shrink_x", 0.18)),
            roi_shrink_top=float(fill_cfg.get("roi_shrink_top", 0.08)),
            roi_shrink_bottom=float(fill_cfg.get("roi_shrink_bottom", 0.05)),
            profile_k=int(fill_cfg.get("profile_k", 3)),
            rice_top_bias_px=int(fill_cfg.get("rice_top_bias_px", 5)),
            bottom_center_offset_px=int(fill_cfg.get("bottom_center_offset_px", 5)),
            min_depth_m=float(fill_cfg.get("min_depth_m", 0.05)),
            max_valid_depth_m=float(fill_cfg.get("max_valid_depth_m", 2.0)),
            depth_patch_half=int(fill_cfg.get("depth_patch_half", 4)),
        )

    def estimate_fill_level_from_cam0(
        self,
        *,
        color_image_bgr: np.ndarray,
        depth_image_m: np.ndarray,
        intrinsics: dict[str, Any],
        container_mask: np.ndarray | None,
        camera_to_base: np.ndarray,
        label: str | None = None,
    ) -> FillLevelEstimate:
        _require_cv2()

        if container_mask is None:
            return FillLevelEstimate(False, None, "raw", None, None)

        mask_bool = np.asarray(container_mask, dtype=bool)
        if not np.any(mask_bool):
            return FillLevelEstimate(False, None, "raw", None, None)

        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

        usable_mask, mask_mode = get_adaptive_container_mask(mask_bool)
        y_bottom_center = get_bottom_center_y(usable_mask, bottom_margin_px=self.bottom_center_offset_px)
        inner_roi_mask = build_inner_roi_mask_from_cup_mask(
            usable_mask,
            shrink_ratio_x=self.roi_shrink_x,
            shrink_ratio_y_top=self.roi_shrink_top,
            shrink_ratio_y_bottom=self.roi_shrink_bottom,
        )
        if inner_roi_mask is None:
            return FillLevelEstimate(False, None, mask_mode, None, y_bottom_center, usable_mask=usable_mask)

        row_indices, feats = compute_row_features(color_image_bgr, inner_roi_mask)
        if row_indices is None:
            return FillLevelEstimate(False, None, mask_mode, None, y_bottom_center, usable_mask=usable_mask)

        profile_result = segment_vertical_profile(row_indices, feats, k=self.profile_k)
        rice_top_y_center = estimate_rice_surface_y_center(profile_result)
        if rice_top_y_center is not None:
            rice_top_y_center = max(int(rice_top_y_center) - self.rice_top_bias_px, 0)

        if y_bottom_center is None or rice_top_y_center is None:
            return FillLevelEstimate(False, None, mask_mode, rice_top_y_center, y_bottom_center, usable_mask=usable_mask)

        metrics = estimate_fill_height_metrics(
            usable_mask,
            rice_top_y_center,
            y_bottom_center,
            depth_image_m,
            fx,
            fy,
            cx,
            cy,
            camera_to_base,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_valid_depth_m,
            patch_half=self.depth_patch_half,
        )
        if metrics is None:
            return FillLevelEstimate(False, None, mask_mode, rice_top_y_center, y_bottom_center, usable_mask=usable_mask)

        rice_top_x = get_row_center_x(usable_mask, rice_top_y_center)
        bottom_center_x = get_row_center_x(usable_mask, y_bottom_center)
        fill_height_mm = float(metrics["fill_h_mm_axis"])
        normalized_label = "" if label is None else str(label).strip().lower()
        if normalized_label == "cup":
            fill_height_mm += float(CUP_FILL_HEIGHT_BIAS_MM)

        return FillLevelEstimate(
            valid=True,
            fill_height_mm=fill_height_mm,
            mask_mode=mask_mode,
            rice_top_y_center=rice_top_y_center,
            bottom_center_y=y_bottom_center,
            usable_mask=usable_mask,
            rice_top_pixel=None if rice_top_x is None else (int(rice_top_x), int(rice_top_y_center)),
            bottom_center_pixel=None if bottom_center_x is None else (int(bottom_center_x), int(y_bottom_center)),
        )


def _parse_demo_args():
    parser = argparse.ArgumentParser(description="Run fill-level estimator demo with RealSense + segmentation visualization.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to handover YAML config.")
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="Model name or local weights path.")
    parser.add_argument("--prompt", nargs="*", default=None, help="Text prompt classes for YOLOE.")
    parser.add_argument("--serial", default=None, help="RealSense serial number. Defaults to the first detected camera.")
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RealSense stream FPS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--device", default=None, help="Ultralytics device string.")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional class id filter.")
    parser.add_argument("--select-mode", choices=["all_instances", "highest_score", "class_filter"], default="highest_score")
    parser.add_argument("--select-class", nargs="*", default=None, help="Optional class-name filter used with selection.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on supported devices.")
    parser.add_argument("--show-depth", action="store_true", help="Show a second depth preview window.")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="Upper bound for depth visualization.")
    return parser.parse_args()


def _pick_serial(serial):
    from utils.realsense_stream import list_realsense_serials

    if serial:
        return serial
    serials = list_realsense_serials()
    if not serials:
        raise RuntimeError("No RealSense devices detected.")
    return serials[0]


def _render_depth(depth_image_m, max_depth_m):
    max_depth_m = max(max_depth_m, 1e-6)
    clipped = np.clip(depth_image_m, 0.0, max_depth_m)
    scaled = (255.0 * clipped / max_depth_m).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)


def _render_mask_overlay(image_bgr, mask, alpha=0.35):
    output = image_bgr.copy()
    if mask is None:
        return output
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return output
    overlay = image_bgr.copy()
    overlay[mask_bool] = (0, 255, 255)
    return cv.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)


def _draw_fill_estimate_overlay(image_bgr, fill_estimate: FillLevelEstimate):
    vis = image_bgr.copy()
    if fill_estimate.usable_mask is not None:
        vis = _render_mask_overlay(vis, fill_estimate.usable_mask, alpha=0.25)

    if fill_estimate.rice_top_pixel is not None:
        x_rice, y_rice = fill_estimate.rice_top_pixel
        cv.circle(vis, (x_rice, y_rice), 7, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(vis, "rice_top", (x_rice + 8, y_rice - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)

    if fill_estimate.bottom_center_pixel is not None:
        x_bot, y_bot = fill_estimate.bottom_center_pixel
        cv.circle(vis, (x_bot, y_bot), 7, (0, 165, 255), -1, cv.LINE_AA)
        cv.putText(vis, "bottom", (x_bot + 8, y_bot - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv.LINE_AA)

    if fill_estimate.rice_top_pixel is not None and fill_estimate.bottom_center_pixel is not None:
        cv.line(vis, fill_estimate.rice_top_pixel, fill_estimate.bottom_center_pixel, (255, 255, 255), 2, cv.LINE_AA)

    return vis


def _draw_status_lines(frame, lines):
    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 24)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


def run_demo():
    _require_cv2()

    from calibration.extrinsics import load_transform_chain
    from object_pt_extraction.segmentation_engine import (
        SegmentationEngine,
        format_instance_summary,
        parse_prompt_classes,
        select_instances,
    )
    from utils.realsense_stream import RealSenseCamera

    args = _parse_demo_args()
    serial = _pick_serial(args.serial)
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
    estimator = FillLevelEstimator.from_config(args.config)
    transform_chain = load_transform_chain(args.config)
    camera = RealSenseCamera(serial=serial, width=args.width, height=args.height, fps=args.fps)

    window_name = "fill_level_estimator_demo"
    depth_window_name = "fill_level_estimator_depth"
    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()

    try:
        while True:
            frame_bundle = camera.read()
            segmentation_result = segmentation_engine.predict(frame_bundle.color_image)
            selected_instances = select_instances(
                segmentation_result.instances,
                mode=args.select_mode,
                class_names=args.select_class,
            )
            summary = format_instance_summary(selected_instances)

            estimate = FillLevelEstimate(False, None, "raw", None, None)
            annotated = frame_bundle.color_image.copy()

            if selected_instances:
                instance = selected_instances[0]
                estimate = estimator.estimate_fill_level_from_cam0(
                    color_image_bgr=frame_bundle.color_image,
                    depth_image_m=frame_bundle.depth_image_m,
                    intrinsics=frame_bundle.intrinsics,
                    container_mask=instance.mask,
                    camera_to_base=transform_chain.t_base_cam0,
                    label=instance.class_name,
                )
                annotated = _draw_fill_estimate_overlay(annotated, estimate)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now

            model_text = args.model if not prompt_classes else f"{args.model} ({','.join(prompt_classes)})"
            lines = [
                f"model: {model_text}",
                f"fps: {smoothed_fps:.1f} | infer: {segmentation_result.infer_ms:.1f} ms",
                summary,
                f"fill_valid={bool(estimate.valid)} mask_mode={estimate.mask_mode}",
                "fill_h_mm=-" if estimate.fill_height_mm is None else f"fill_h_mm={float(estimate.fill_height_mm):.1f}",
                "ESC / q: quit",
            ]
            _draw_status_lines(annotated, lines)

            cv.imshow(window_name, annotated)
            if args.show_depth:
                cv.imshow(depth_window_name, _render_depth(frame_bundle.depth_image_m, args.depth_max_m))

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.stop()
        cv.destroyAllWindows()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "FillLevelEstimate",
    "FillLevelEstimator",
    "build_inner_roi_mask_from_cup_mask",
    "compute_row_features",
    "segment_vertical_profile",
    "estimate_rice_surface_y_center",
    "extract_bowl_mask_from_glass_mask",
    "get_adaptive_container_mask",
]


if __name__ == "__main__":
    run_demo()
