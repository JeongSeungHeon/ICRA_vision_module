#!/usr/bin/env python3
"""Live debug viewer for cam0-only object cloud, hand pose, and grasp target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("mediapipe is required for tools/visualize_cam0_target_debug.py") from exc

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("open3d is required for tools/visualize_cam0_target_debug.py") from exc

from perception.grasp_target import GraspTargetPlanner
from perception.hand_worker import HandWorkerCam0
from perception.object_worker import ObjectWorkerCam0
from system.dual_sensor_hub import DualSensorHub
from system.shared_state import GraspTargetState, HandState, MergedObjectState, ObjectState, SelectedHandState

HAND_CONNECTIONS = tuple(tuple(pair) for pair in mp.solutions.hands.HAND_CONNECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cam0-only target generation in robot base frame.")
    parser.add_argument("--config", default="configs/handover.yaml", help="Path to the handover config file.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run until closed.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Open3D point size.")
    parser.add_argument("--normal-length", type=float, default=0.08, help="Length of the palm normal debug line in meters.")
    parser.add_argument("--print-every", type=int, default=10, help="Print one status line every N frames.")
    parser.add_argument("--show-2d", action="store_true", help="Show the cam0 2D debug window with mask and hand keypoints.")
    return parser.parse_args()


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


def build_cam0_only_scene(
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
        points_3d_base = getattr(hand_debug, 'points_3d_base', None)
        valid_mask = getattr(hand_debug, 'valid_mask', None)
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
    keypoints_2d = getattr(hand_debug, 'keypoints_2d', None)
    valid_mask = getattr(hand_debug, 'valid_mask', None)
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


def build_camera_debug_image(
    frame_bgr: np.ndarray,
    object_state: ObjectState,
    object_debug: object | None,
    hand_state: HandState,
    hand_debug: object | None,
    selected_hand: SelectedHandState,
    grasp_target: GraspTargetState,
) -> np.ndarray:
    output = overlay_mask(frame_bgr, getattr(object_debug, 'combined_mask', None), (0, 180, 255))
    output = draw_hand_overlay(output, hand_debug, (255, 128, 0))

    text_lines = [
        f"cam0 only: object_detected={object_state.object_detected} valid={object_state.valid}",
        f"obj points={object_state.point_count} conf={object_state.confidence:.2f}",
        f"hand_detected={hand_state.hand_detected} valid={hand_state.valid} handedness={hand_state.handedness}",
        f"selected_cam={selected_hand.selected_camera}",
        f"object_centroid_base={format_vec3(object_state.centroid_base)}",
        f"target_base={format_vec3(grasp_target.target_position_base if grasp_target.valid else None)}",
    ]
    return draw_text_block(output, text_lines)


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


def print_status(
    frame_index: int,
    object_state: ObjectState,
    hand_state: HandState,
    grasp_target: GraspTargetState,
) -> None:
    grasp_label = 'invalid'
    if grasp_target.valid and grasp_target.target_position_base is not None:
        position = tuple(round(value, 3) for value in grasp_target.target_position_base)
        grasp_label = (
            f"valid pos={position} clearance={grasp_target.hand_height_clearance_m:.3f}m "
            f"d_centroid={grasp_target.distance_to_centroid_m:.3f}m"
        )

    print(
        f"[frame {frame_index:05d}] "
        f"cam0_object(det={object_state.object_detected},pts={object_state.point_count},conf={object_state.confidence:.2f}) "
        f"cam0_hand(det={hand_state.hand_detected},valid={hand_state.valid},h={hand_state.handedness}) "
        f"centroid_base={format_vec3(object_state.centroid_base)} "
        f"target_base={format_vec3(grasp_target.target_position_base if grasp_target.valid else None)} "
        f"grasp={grasp_label}",
        flush=True,
    )


def main() -> int:
    args = parse_args()

    sensor_hub = DualSensorHub.from_config(args.config)
    object_worker_cam0 = ObjectWorkerCam0.from_config(args.config)
    hand_worker_cam0 = HandWorkerCam0.from_config(args.config)
    grasp_planner = GraspTargetPlanner.from_config(args.config)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='Cam0 Target Debug', width=1440, height=960)
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

            snapshot = sensor_hub.read_next_pair()
            object_state = object_worker_cam0.process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            hand_state = hand_worker_cam0.process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            merged_like = to_cam0_only_merged_state(object_state)
            selected_hand = to_cam0_selected_hand(hand_state)
            grasp_target = grasp_planner.process_states(merged_like, selected_hand, None)

            visualizer.clear_geometries()
            for geometry in build_cam0_only_scene(
                object_state=object_state,
                hand_state=hand_state,
                hand_debug=hand_worker_cam0.last_debug,
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
                    object_worker_cam0.last_debug,
                    hand_state,
                    hand_worker_cam0.last_debug,
                    selected_hand,
                    grasp_target,
                )
                cv.imshow('cam0 target debug', cam0_view)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

            frame_index += 1
            if args.print_every > 0 and (frame_index == 1 or frame_index % args.print_every == 0):
                print_status(frame_index, object_state, hand_state, grasp_target)
    finally:
        cv.destroyAllWindows()
        visualizer.destroy_window()
        sensor_hub.stop()
        hand_worker_cam0.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
