#!/usr/bin/env python3
"""Live debug viewer for single-ZED object cloud, hand pose, and grasp target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2 as cv
import numpy as np
import yaml

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("mediapipe is required for tools/visualize_zed_target_debug.py") from exc

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("open3d is required for tools/visualize_zed_target_debug.py") from exc

from calibration.extrinsics import TransformChain, load_dat_transform, load_pickle_transform
from object_pt_extraction.segmentation_engine import SegmentationEngine, select_instances
from perception.grasp_target import GraspTargetPlanner
from perception.hand_worker import HandWorkerCam0
from perception.object_worker import ObjectWorkerCam0
from system.shared_state import GraspTargetState, HandState, MergedObjectState, ObjectState, SelectedHandState
from system.single_sensor_hub import SingleSensorHub

HAND_CONNECTIONS = tuple(tuple(pair) for pair in mp.solutions.hands.HAND_CONNECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ZED cam0-only target generation in verification frame.")
    parser.add_argument("--config", default="configs/handover_zed_single.yaml", help="Path to the ZED verification config file.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run until closed.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Open3D point size.")
    parser.add_argument("--normal-length", type=float, default=0.08, help="Length of the palm normal debug line in meters.")
    parser.add_argument("--print-every", type=int, default=10, help="Print one status line every N frames.")
    parser.add_argument("--show-2d", action="store_true", help="Show the RGB debug window with object mask and hand keypoints.")
    parser.add_argument("--show-depth", action="store_true", help="Show a depth colormap debug window.")
    parser.add_argument("--show-seg-input", action="store_true", help="Show the segmentation preprocessed image.")
    parser.add_argument("--show-contrast-debug", action="store_true", help="Show original vs preprocessed foreground/background contrast analysis.")
    parser.add_argument("--show-seg-compare", action="store_true", help="Run YOLO twice per frame and compare raw vs preprocessed segmentation.")
    parser.add_argument("--depth-max-m", type=float, default=None, help="Upper bound for depth visualization in meters.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_identity_transform_chain(source_name: str) -> TransformChain:
    identity = np.eye(4, dtype=np.float32)
    return TransformChain(
        t_base_cam0=identity.copy(),
        t_cam0_cam1=identity.copy(),
        t_base_cam1=identity.copy(),
        source_config=source_name,
    )


def load_single_camera_transform_chain(config_path: str | Path, config: dict) -> TransformChain:
    calibration_cfg = config.get("calibration", {})
    if bool(calibration_cfg.get("use_identity_base_frame", True)):
        return build_identity_transform_chain(f"{config_path}#identity")

    transform_file = calibration_cfg.get("cam0_to_base_file")
    if not transform_file:
        return build_identity_transform_chain(f"{config_path}#identity_fallback")

    transform_format = str(calibration_cfg.get("cam0_to_base_format", "pickle_cam0_to_robot")).strip().lower()
    if transform_format in {"pickle", "pickle_cam0_to_robot"}:
        t_base_cam0 = load_pickle_transform(transform_file)
    elif transform_format in {"dat", "dat_cam0_to_robot"}:
        t_base_cam0 = load_dat_transform(
            transform_file,
            translation_unit=calibration_cfg.get("cam0_to_base_translation_unit", "m"),
        )
    else:
        raise ValueError(f"Unsupported cam0_to_base_format: {transform_format}")

    identity = np.eye(4, dtype=np.float32)
    return TransformChain(
        t_base_cam0=np.asarray(t_base_cam0, dtype=np.float32),
        t_cam0_cam1=identity.copy(),
        t_base_cam1=np.asarray(t_base_cam0, dtype=np.float32),
        source_config=str(config_path),
    )


def build_object_worker(config: dict, transform_chain: TransformChain) -> ObjectWorkerCam0:
    object_cfg = config.get("perception", {}).get("object", {})
    segmentation_cfg = object_cfg.get("segmentation", {})
    segmentation_engine = SegmentationEngine(
        model_name=segmentation_cfg.get("model_name", "yoloe-26l-seg.pt"),
        prompt_classes=segmentation_cfg.get("prompt_classes", ["cup"]),
        imgsz=int(segmentation_cfg.get("imgsz", 640)),
        conf=float(segmentation_cfg.get("conf", 0.25)),
        iou=float(segmentation_cfg.get("iou", 0.45)),
        max_det=int(segmentation_cfg.get("max_det", 100)),
        device=segmentation_cfg.get("device"),
        classes=segmentation_cfg.get("classes"),
        half=bool(segmentation_cfg.get("half", False)),
        retina_masks=bool(segmentation_cfg.get("retina_masks", True)),
        preprocess_config=dict(segmentation_cfg.get("preprocess", {}) or {}),
    )
    return ObjectWorkerCam0(
        camera_id=0,
        segmentation_engine=segmentation_engine,
        transform_chain=transform_chain,
        config=config,
    )


def build_hand_worker(config: dict, transform_chain: TransformChain) -> HandWorkerCam0:
    return HandWorkerCam0(camera_id=0, transform_chain=transform_chain, config=config)


def make_point_cloud(points_xyz: np.ndarray, color_rgb: Iterable[float]) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    colors = np.tile(np.asarray(list(color_rgb), dtype=np.float64).reshape(1, 3), (len(points_xyz), 1))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def make_sphere(center_xyz: np.ndarray, radius_m: float, color_rgb: Iterable[float]) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius_m)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(np.asarray(list(color_rgb), dtype=np.float64))
    sphere.translate(center_xyz.astype(np.float64))
    return sphere


def make_line_set(points_xyz: np.ndarray, lines_ij: list[list[int]], color_rgb: Iterable[float]) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines_ij, dtype=np.int32))
    colors = np.tile(np.asarray(list(color_rgb), dtype=np.float64).reshape(1, 3), (len(lines_ij), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def format_vec3(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return "None"
    return f"({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})"


def build_hand_geometries(points_3d_base: np.ndarray, valid_mask: np.ndarray, color_rgb: tuple[float, float, float]) -> list[o3d.geometry.Geometry]:
    valid_indices = [index for index, is_valid in enumerate(valid_mask.tolist()) if bool(is_valid)]
    if not valid_indices:
        return []

    valid_points = np.asarray(points_3d_base[valid_indices], dtype=np.float32).reshape((-1, 3))
    if len(valid_points) == 0:
        return []

    index_map = {original_index: new_index for new_index, original_index in enumerate(valid_indices)}
    valid_lines: list[list[int]] = []
    for start_index, end_index in HAND_CONNECTIONS:
        if start_index in index_map and end_index in index_map:
            valid_lines.append([index_map[start_index], index_map[end_index]])

    geometries: list[o3d.geometry.Geometry] = [make_point_cloud(valid_points, color_rgb)]
    if valid_lines:
        geometries.append(make_line_set(valid_points, valid_lines, color_rgb))
    return geometries


def build_selected_palm_geometries(selected_hand: SelectedHandState, normal_length_m: float) -> list[o3d.geometry.Geometry]:
    if not selected_hand.valid or selected_hand.palm_center_base is None:
        return []
    center = np.asarray(selected_hand.palm_center_base, dtype=np.float32)
    geometries: list[o3d.geometry.Geometry] = [make_sphere(center, radius_m=0.012, color_rgb=(0.1, 0.85, 0.95))]

    if selected_hand.palm_normal_base is not None:
        normal = np.asarray(selected_hand.palm_normal_base, dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-6:
            normal = normal / norm
            line_points = np.stack([center, center + normal * float(normal_length_m)], axis=0)
            geometries.append(make_line_set(line_points, [[0, 1]], (0.1, 0.85, 0.95)))
    return geometries


def build_scene(
    object_state: ObjectState,
    hand_state: HandState,
    hand_debug: object | None,
    selected_hand: SelectedHandState,
    grasp_target: GraspTargetState,
    normal_length_m: float,
) -> list[o3d.geometry.Geometry]:
    geometries: list[o3d.geometry.Geometry] = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)]

    if object_state.valid and object_state.points_base:
        points_base = np.asarray(object_state.points_base, dtype=np.float32).reshape((-1, 3))
        geometries.append(make_point_cloud(points_base, (0.72, 0.72, 0.72)))

    if object_state.centroid_base is not None:
        centroid = np.asarray(object_state.centroid_base, dtype=np.float32)
        geometries.append(make_sphere(centroid, radius_m=0.011, color_rgb=(0.95, 0.25, 0.25)))

    if hand_state.valid and hand_debug is not None:
        points_3d_base = getattr(hand_debug, "points_3d_base", None)
        valid_mask = getattr(hand_debug, "valid_mask", None)
        if points_3d_base is not None and valid_mask is not None:
            geometries.extend(build_hand_geometries(np.asarray(points_3d_base), np.asarray(valid_mask), (0.25, 0.45, 1.0)))

    geometries.extend(build_selected_palm_geometries(selected_hand, normal_length_m=normal_length_m))

    if grasp_target.valid and grasp_target.target_position_base is not None:
        target = np.asarray(grasp_target.target_position_base, dtype=np.float32)
        geometries.append(make_sphere(target, radius_m=0.013, color_rgb=(0.1, 0.95, 0.2)))
        if object_state.centroid_base is not None:
            centroid = np.asarray(object_state.centroid_base, dtype=np.float32)
            geometries.append(make_line_set(np.stack([centroid, target], axis=0), [[0, 1]], (0.1, 0.95, 0.2)))

    return geometries


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray | None, color_bgr: tuple[int, int, int]) -> np.ndarray:
    output = image_bgr.copy()
    if mask is None:
        return output
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != output.shape[:2]:
        return output
    overlay = np.zeros_like(output, dtype=np.uint8)
    overlay[mask_bool] = np.asarray(color_bgr, dtype=np.uint8)
    return cv.addWeighted(output, 1.0, overlay, 0.35, 0.0)


def draw_hand_overlay(image_bgr: np.ndarray, hand_debug: object | None, color_bgr: tuple[int, int, int]) -> np.ndarray:
    output = image_bgr.copy()
    if hand_debug is None:
        return output
    keypoints_2d = getattr(hand_debug, "keypoints_2d", None)
    valid_mask = getattr(hand_debug, "valid_mask", None)
    if keypoints_2d is None or valid_mask is None:
        return output

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.int32).reshape((-1, 2))
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape((-1,))
    for start_index, end_index in HAND_CONNECTIONS:
        if start_index >= len(valid_mask) or end_index >= len(valid_mask):
            continue
        if not (bool(valid_mask[start_index]) and bool(valid_mask[end_index])):
            continue
        p0 = tuple(int(v) for v in keypoints_2d[start_index])
        p1 = tuple(int(v) for v in keypoints_2d[end_index])
        cv.line(output, p0, p1, color_bgr, 2, cv.LINE_AA)

    for point_index, is_valid in enumerate(valid_mask.tolist()):
        if point_index >= len(keypoints_2d):
            continue
        point = tuple(int(v) for v in keypoints_2d[point_index])
        point_color = color_bgr if bool(is_valid) else (64, 64, 64)
        cv.circle(output, point, 3, point_color, -1, cv.LINE_AA)
    return output


def draw_text_block(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image_bgr.copy()
    y = 24
    for line in lines:
        cv.putText(output, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv.LINE_AA)
        cv.putText(output, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv.LINE_AA)
        y += 22
    return output


def render_depth(depth_image_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    clipped = np.nan_to_num(np.clip(depth_image_m, 0.0, max(max_depth_m, 1e-6)), nan=0.0, posinf=0.0, neginf=0.0)
    scaled = (255.0 * clipped / max(max_depth_m, 1e-6)).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)


def build_camera_debug_image(
    frame_bgr: np.ndarray,
    object_state: ObjectState,
    object_debug: object | None,
    hand_state: HandState,
    hand_debug: object | None,
    selected_hand: SelectedHandState,
    grasp_target: GraspTargetState,
) -> np.ndarray:
    output = overlay_mask(frame_bgr, getattr(object_debug, "combined_mask", None), (0, 180, 255))
    output = draw_hand_overlay(output, hand_debug, (255, 128, 0))

    text_lines = [
        f"zed cam0: object_detected={object_state.object_detected} valid={object_state.valid}",
        f"obj label={object_state.label} points={object_state.point_count} conf={object_state.confidence:.2f}",
        f"hand_detected={hand_state.hand_detected} valid={hand_state.valid} handedness={hand_state.handedness}",
        f"selected_cam={selected_hand.selected_camera}",
        f"object_centroid={format_vec3(object_state.centroid_base)}",
        f"target={format_vec3(grasp_target.target_position_base if grasp_target.valid else None)}",
    ]
    return draw_text_block(output, text_lines)


def build_segmentation_input_debug_image(
    image_bgr: np.ndarray | None,
    preprocess_cfg: dict | None,
) -> np.ndarray | None:
    if image_bgr is None:
        return None

    output = np.asarray(image_bgr, dtype=np.uint8).copy()
    cfg = dict(preprocess_cfg or {})
    clahe_cfg = dict(cfg.get("clahe", {}) or {})
    text_lines = [
        f"seg preprocess enabled={bool(cfg.get('enabled', False))}",
        f"alpha={float(cfg.get('contrast_alpha', 1.0)):.2f} beta={float(cfg.get('brightness_beta', 0.0)):.1f}",
        f"gamma={float(cfg.get('gamma', 1.0)):.2f}",
        f"clahe={bool(clahe_cfg.get('enabled', False))} clip={float(clahe_cfg.get('clip_limit', 2.0)):.2f}",
    ]
    return draw_text_block(output, text_lines)


def combine_instance_masks(instances: list, image_shape: tuple[int, ...]) -> np.ndarray:
    combined_mask = np.zeros(image_shape[:2], dtype=bool)
    for instance in instances:
        combined_mask |= np.asarray(instance.mask, dtype=bool)
    return combined_mask


def format_selection_summary(selected_instances: list) -> str:
    if not selected_instances:
        return "none"
    primary = max(selected_instances, key=lambda instance: float(instance.score))
    return f"{primary.class_name}:{float(primary.score):.2f} count={len(selected_instances)}"


def build_segmentation_compare_debug_image(
    original_bgr: np.ndarray,
    segmentation_engine: SegmentationEngine,
    selection_mode: str,
    selection_class_names: list[str] | None,
) -> np.ndarray | None:
    raw_preprocess_cfg = {
        "enabled": False,
        "contrast_alpha": 1.0,
        "brightness_beta": 0.0,
        "gamma": 1.0,
        "clahe": {"enabled": False},
    }
    raw_result = segmentation_engine.predict(
        original_bgr,
        preprocess_config=raw_preprocess_cfg,
        store_debug_frame=False,
    )
    prep_result = segmentation_engine.predict(
        original_bgr,
        preprocess_config=segmentation_engine.preprocess_config,
        store_debug_frame=False,
    )

    raw_selected = select_instances(
        raw_result.instances,
        mode=selection_mode,
        class_names=selection_class_names or None,
    )
    prep_selected = select_instances(
        prep_result.instances,
        mode=selection_mode,
        class_names=selection_class_names or None,
    )

    raw_mask = combine_instance_masks(raw_selected, original_bgr.shape)
    prep_mask = combine_instance_masks(prep_selected, original_bgr.shape)

    raw_view = overlay_mask(original_bgr, raw_mask, (60, 180, 255))
    prep_input = prep_result.preprocessed_image if prep_result.preprocessed_image is not None else original_bgr
    prep_view = overlay_mask(prep_input, prep_mask, (0, 255, 120))
    panel = cv.hconcat([raw_view, prep_view])

    raw_mask_area = int(np.count_nonzero(raw_mask))
    prep_mask_area = int(np.count_nonzero(prep_mask))
    area_gain = prep_mask_area - raw_mask_area
    ratio = prep_mask_area / max(raw_mask_area, 1)
    lines = [
        "left=raw segmentation, right=preprocessed segmentation",
        f"raw: {format_selection_summary(raw_selected)} infer={raw_result.infer_ms:.1f}ms area={raw_mask_area}",
        f"prep: {format_selection_summary(prep_selected)} infer={prep_result.infer_ms:.1f}ms area={prep_mask_area}",
        f"area gain={area_gain:+d} ratio={ratio:.2f}x",
    ]
    return draw_text_block(panel, lines)


def build_background_ring(mask: np.ndarray, ring_radius_px: int = 15) -> np.ndarray:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    kernel_size = max(int(ring_radius_px) * 2 + 1, 3)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv.dilate(mask_uint8, kernel, iterations=1)
    ring = (dilated > 0) & ~(mask_uint8 > 0)
    return ring


def compute_region_stats(image_bgr: np.ndarray, region_mask: np.ndarray) -> dict | None:
    mask_bool = np.asarray(region_mask, dtype=bool)
    if image_bgr is None or not np.any(mask_bool):
        return None

    lab_image = cv.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv.COLOR_BGR2LAB)
    l_channel = lab_image[:, :, 0].astype(np.float32)
    values = l_channel[mask_bool]
    if values.size == 0:
        return None

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def build_contrast_metrics(original_bgr: np.ndarray, preprocessed_bgr: np.ndarray, mask: np.ndarray | None) -> dict | None:
    if mask is None:
        return None
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != original_bgr.shape[:2] or not np.any(mask_bool):
        return None

    ring_mask = build_background_ring(mask_bool)
    if not np.any(ring_mask):
        ring_mask = ~mask_bool
    if not np.any(ring_mask):
        return None

    orig_fg = compute_region_stats(original_bgr, mask_bool)
    orig_bg = compute_region_stats(original_bgr, ring_mask)
    prep_fg = compute_region_stats(preprocessed_bgr, mask_bool)
    prep_bg = compute_region_stats(preprocessed_bgr, ring_mask)
    if any(stat is None for stat in (orig_fg, orig_bg, prep_fg, prep_bg)):
        return None

    orig_delta = abs(float(orig_fg["mean"]) - float(orig_bg["mean"]))
    prep_delta = abs(float(prep_fg["mean"]) - float(prep_bg["mean"]))
    gain = prep_delta - orig_delta
    ratio = prep_delta / max(orig_delta, 1e-6)
    return {
        "mask": mask_bool,
        "ring_mask": ring_mask,
        "orig_fg": orig_fg,
        "orig_bg": orig_bg,
        "prep_fg": prep_fg,
        "prep_bg": prep_bg,
        "orig_delta": orig_delta,
        "prep_delta": prep_delta,
        "gain": gain,
        "ratio": ratio,
    }


def annotate_mask_regions(image_bgr: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    output = np.asarray(image_bgr, dtype=np.uint8).copy()
    fg_bool = np.asarray(fg_mask, dtype=bool)
    bg_bool = np.asarray(bg_mask, dtype=bool)
    if fg_bool.shape[:2] == output.shape[:2]:
        fg_overlay = np.zeros_like(output, dtype=np.uint8)
        fg_overlay[fg_bool] = np.array([0, 220, 255], dtype=np.uint8)
        output = cv.addWeighted(output, 1.0, fg_overlay, 0.30, 0.0)
    if bg_bool.shape[:2] == output.shape[:2]:
        bg_overlay = np.zeros_like(output, dtype=np.uint8)
        bg_overlay[bg_bool] = np.array([255, 140, 0], dtype=np.uint8)
        output = cv.addWeighted(output, 1.0, bg_overlay, 0.18, 0.0)
    return output


def build_contrast_debug_image(
    original_bgr: np.ndarray,
    preprocessed_bgr: np.ndarray | None,
    combined_mask: np.ndarray | None,
) -> np.ndarray | None:
    if preprocessed_bgr is None or combined_mask is None:
        return None

    metrics = build_contrast_metrics(original_bgr, preprocessed_bgr, combined_mask)
    if metrics is None:
        return None

    original_view = annotate_mask_regions(original_bgr, metrics["mask"], metrics["ring_mask"])
    preprocessed_view = annotate_mask_regions(preprocessed_bgr, metrics["mask"], metrics["ring_mask"])
    panel = cv.hconcat([original_view, preprocessed_view])

    lines = [
        "fg=segmented region, bg=local ring background",
        f"orig delta(L)={metrics['orig_delta']:.2f} prep delta(L)={metrics['prep_delta']:.2f}",
        f"gain={metrics['gain']:+.2f} ratio={metrics['ratio']:.2f}x",
        f"orig fg={metrics['orig_fg']['mean']:.1f} bg={metrics['orig_bg']['mean']:.1f}",
        f"prep fg={metrics['prep_fg']['mean']:.1f} bg={metrics['prep_bg']['mean']:.1f}",
    ]
    return draw_text_block(panel, lines)


def to_cam0_only_merged_state(object_state: ObjectState) -> MergedObjectState:
    return MergedObjectState(
        frame_id_cam0=int(object_state.frame_id),
        frame_id_cam1=-1,
        object_detected=bool(object_state.object_detected),
        label=object_state.label,
        confidence=float(object_state.confidence),
        centroid_base=object_state.centroid_base,
        initial_centroid_base=None,
        merged_point_count=int(object_state.point_count),
        merged_points_base=list(object_state.points_base),
        object_lifted=False,
        lift_height_delta_m=0.0,
        timestamp=float(object_state.timestamp),
        valid=bool(object_state.valid),
    )


def to_cam0_selected_hand(hand_state: HandState) -> SelectedHandState:
    return SelectedHandState(
        frame_id=int(hand_state.frame_id),
        selected_camera=0 if hand_state.valid else None,
        handedness=hand_state.handedness if hand_state.valid else None,
        confidence=float(hand_state.confidence),
        palm_center_base=hand_state.palm_center_base,
        palm_normal_base=hand_state.palm_normal_base,
        wrist_base=hand_state.wrist_base,
        hand_velocity_base=hand_state.hand_velocity_base,
        timestamp=float(hand_state.timestamp),
        valid=bool(hand_state.valid),
    )


def print_status(frame_index: int, object_state: ObjectState, hand_state: HandState, grasp_target: GraspTargetState) -> None:
    grasp_label = "invalid"
    if grasp_target.valid and grasp_target.target_position_base is not None:
        position = tuple(round(value, 3) for value in grasp_target.target_position_base)
        grasp_label = (
            f"valid pos={position} clearance={grasp_target.hand_height_clearance_m:.3f}m "
            f"d_centroid={grasp_target.distance_to_centroid_m:.3f}m"
        )

    print(
        f"[frame {frame_index:05d}] "
        f"zed_object(det={object_state.object_detected},pts={object_state.point_count},conf={object_state.confidence:.2f}) "
        f"zed_hand(det={hand_state.hand_detected},valid={hand_state.valid},h={hand_state.handedness}) "
        f"centroid={format_vec3(object_state.centroid_base)} "
        f"target={format_vec3(grasp_target.target_position_base if grasp_target.valid else None)} "
        f"grasp={grasp_label}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    sensor_hub = SingleSensorHub.from_config(args.config)
    transform_chain = load_single_camera_transform_chain(args.config, config)
    object_worker = build_object_worker(config, transform_chain)
    hand_worker = build_hand_worker(config, transform_chain)
    grasp_planner = GraspTargetPlanner(config=config)

    debug_cfg = config.get("debug", {})
    depth_max_m = float(args.depth_max_m if args.depth_max_m is not None else debug_cfg.get("depth_visualization_max_m", 1.5))

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="ZED Target Debug", width=1440, height=960)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
    render_option.point_size = float(args.point_size)
    render_option.line_width = 2.0

    frame_index = 0

    try:
        sensor_hub.start()
        while True:
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            snapshot = sensor_hub.read()
            object_state = object_worker.process_frame(snapshot.cam0, frame_id=snapshot.frame_id)
            hand_state = hand_worker.process_frame(snapshot.cam0, frame_id=snapshot.frame_id)
            merged_like = to_cam0_only_merged_state(object_state)
            selected_hand = to_cam0_selected_hand(hand_state)
            grasp_target = grasp_planner.process_states(merged_like, selected_hand, None)

            visualizer.clear_geometries()
            for geometry in build_scene(
                object_state=object_state,
                hand_state=hand_state,
                hand_debug=hand_worker.last_debug,
                selected_hand=selected_hand,
                grasp_target=grasp_target,
                normal_length_m=float(args.normal_length),
            ):
                visualizer.add_geometry(geometry, reset_bounding_box=(frame_index == 0))
            if not visualizer.poll_events():
                break
            visualizer.update_renderer()

            if args.show_2d:
                cam0_view = build_camera_debug_image(
                    snapshot.cam0.color_image,
                    object_state,
                    object_worker.last_debug,
                    hand_state,
                    hand_worker.last_debug,
                    selected_hand,
                    grasp_target,
                )
                cv.imshow("zed cam0 target debug", cam0_view)

            if args.show_seg_input:
                seg_input_view = build_segmentation_input_debug_image(
                    getattr(object_worker.segmentation_engine, "last_preprocessed_frame", None),
                    getattr(object_worker.segmentation_engine, "preprocess_config", None),
                )
                if seg_input_view is not None:
                    cv.imshow("zed segmentation input", seg_input_view)

            if args.show_contrast_debug:
                contrast_view = build_contrast_debug_image(
                    snapshot.cam0.color_image,
                    getattr(object_worker.segmentation_engine, "last_preprocessed_frame", None),
                    getattr(object_worker.last_debug, "combined_mask", None),
                )
                if contrast_view is not None:
                    cv.imshow("zed contrast debug", contrast_view)

            if args.show_seg_compare:
                seg_compare_view = build_segmentation_compare_debug_image(
                    snapshot.cam0.color_image,
                    object_worker.segmentation_engine,
                    object_worker.selection_mode,
                    object_worker.selection_class_names,
                )
                if seg_compare_view is not None:
                    cv.imshow("zed segmentation compare", seg_compare_view)

            if args.show_depth:
                cv.imshow("zed cam0 depth", render_depth(snapshot.cam0.depth_image_m, depth_max_m))

            if args.show_2d or args.show_depth or args.show_seg_input or args.show_contrast_debug or args.show_seg_compare:
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1
            if args.print_every > 0 and (frame_index == 1 or frame_index % args.print_every == 0):
                print_status(frame_index, object_state, hand_state, grasp_target)
    finally:
        cv.destroyAllWindows()
        visualizer.destroy_window()
        sensor_hub.stop()
        hand_worker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
