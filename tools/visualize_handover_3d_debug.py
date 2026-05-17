#!/usr/bin/env python3
"""Playback viewer for saved handover 3D debug recordings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("open3d is required for tools/visualize_handover_3d_debug.py") from exc

from utils.debug_3d_recorder import load_debug_3d_npz


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved handover 3D debug .npz recording.")
    parser.add_argument("recording", help="Path to output/debug_3d/*.npz")
    parser.add_argument("--point-size", type=float, default=3.0, help="Open3D point size.")
    parser.add_argument("--normal-length", type=float, default=0.08, help="Palm normal line length in meters.")
    parser.add_argument("--template-axis-length", type=float, default=0.06, help="Template local x/y/z axis line length in meters.")
    parser.add_argument("--play-hz", type=float, default=10.0, help="Playback frame rate.")
    parser.add_argument("--dry-run", action="store_true", help="Load the recording and build frame 0 geometries, then exit.")
    return parser.parse_args()


def _as_points(value: object) -> np.ndarray:
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(value, dtype=np.float32).reshape((-1, 3))
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def _is_valid_vec(value: object, length: int = 3) -> bool:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    return len(vec) >= int(length) and bool(np.isfinite(vec[:length]).all())


def _valid_template_axes(data: dict[str, np.ndarray], idx: int) -> bool:
    if "template_axes_base" not in data or "template_centroid_base" not in data:
        return False
    if not _is_valid_vec(data["template_centroid_base"][idx], 3):
        return False
    axes = np.asarray(data["template_axes_base"][idx], dtype=np.float32)
    return axes.shape == (3, 3) and bool(np.isfinite(axes).all())


def _string_at(data: dict[str, np.ndarray], key: str, index: int) -> str:
    if key not in data:
        return ""
    value = data[key][index]
    return "" if value is None else str(value)


def make_point_cloud(points_xyz: np.ndarray, color_rgb: Iterable[float]) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    colors = np.tile(np.asarray(list(color_rgb), dtype=np.float64).reshape(1, 3), (len(points_xyz), 1))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def make_sphere(center_xyz: np.ndarray, radius_m: float, color_rgb: Iterable[float]) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius_m))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(np.asarray(list(color_rgb), dtype=np.float64))
    sphere.translate(np.asarray(center_xyz, dtype=np.float64).reshape(3))
    return sphere


def make_line_set(points_xyz: np.ndarray, lines_ij: list[list[int]], color_rgb: Iterable[float]) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines_ij, dtype=np.int32))
    colors = np.tile(np.asarray(list(color_rgb), dtype=np.float64).reshape(1, 3), (len(lines_ij), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def build_template_axis_geometries(
    data: dict[str, np.ndarray],
    idx: int,
    *,
    axis_length_m: float,
) -> list[o3d.geometry.Geometry]:
    if not _valid_template_axes(data, idx):
        return []
    origin = np.asarray(data["template_centroid_base"][idx], dtype=np.float32).reshape(-1)[:3]
    axes = np.asarray(data["template_axes_base"][idx], dtype=np.float32).reshape((3, 3))
    colors = [(1.0, 0.0, 0.0), (0.0, 0.86, 0.0), (0.0, 0.47, 1.0)]
    geometries: list[o3d.geometry.Geometry] = []
    length = max(float(axis_length_m), 0.0)
    for axis_index, color in enumerate(colors):
        axis = axes[axis_index]
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 1e-9:
            return []
        axis = axis / norm
        points = np.stack([origin, origin + axis * length], axis=0)
        geometries.append(make_line_set(points, [[0, 1]], color))
    return geometries


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    x, y, z = axis
    skew = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def make_pose_frame(pose_xyz_rotvec: np.ndarray, *, size: float) -> o3d.geometry.TriangleMesh:
    pose = np.asarray(pose_xyz_rotvec, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotvec_to_matrix(pose[3:6])
    transform[:3, 3] = pose[:3]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size))
    frame.transform(transform)
    return frame


def build_hand_geometries(
    points_base: np.ndarray,
    valid_mask: np.ndarray,
    *,
    color_rgb: tuple[float, float, float],
    selected: bool,
) -> list[o3d.geometry.Geometry]:
    points = np.asarray(points_base, dtype=np.float32).reshape((-1, 3))
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_indices = [idx for idx in range(min(len(points), len(mask))) if mask[idx] and np.isfinite(points[idx]).all()]
    if not valid_indices:
        return []

    valid_points = points[valid_indices]
    index_map = {original_index: new_index for new_index, original_index in enumerate(valid_indices)}
    valid_lines: list[list[int]] = []
    for start_index, end_index in HAND_CONNECTIONS:
        if start_index in index_map and end_index in index_map:
            valid_lines.append([index_map[start_index], index_map[end_index]])

    hand_color = (1.0, 0.95, 0.1) if selected else color_rgb
    geometries: list[o3d.geometry.Geometry] = [make_point_cloud(valid_points, hand_color)]
    if valid_lines:
        geometries.append(make_line_set(valid_points, valid_lines, hand_color))
    return geometries


def build_frame_geometries(
    data: dict[str, np.ndarray],
    frame_index: int,
    *,
    normal_length_m: float = 0.08,
    template_axis_length_m: float = 0.06,
) -> list[o3d.geometry.Geometry]:
    idx = int(frame_index)
    geometries: list[o3d.geometry.Geometry] = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)]

    object_points = _as_points(data["object_points_base"][idx])
    if len(object_points) > 0:
        color = (0.78, 0.78, 0.78) if bool(data["object_valid"][idx]) else (0.85, 0.25, 0.22)
        geometries.append(make_point_cloud(object_points, color))

    template_points = _as_points(data["template_points_base"][idx])
    if len(template_points) > 0:
        geometries.append(make_point_cloud(template_points, (0.1, 0.8, 0.95)))

    selected_camera = int(data["selected_hand_camera"][idx])
    geometries.extend(
        build_hand_geometries(
            data["cam0_hand_points_base"][idx],
            data["cam0_hand_valid_mask"][idx],
            color_rgb=(0.25, 0.45, 1.0),
            selected=selected_camera == 0,
        )
    )
    geometries.extend(
        build_hand_geometries(
            data["cam1_hand_points_base"][idx],
            data["cam1_hand_valid_mask"][idx],
            color_rgb=(1.0, 0.3, 0.75),
            selected=selected_camera == 1,
        )
    )

    for key, radius, color in (
        ("object_point_base", 0.012, (1.0, 0.55, 0.05)),
        ("grasp_point_base", 0.014, (0.1, 0.95, 0.2)),
        ("selected_palm_center_base", 0.012, (1.0, 0.95, 0.1)),
        ("selected_wrist_base", 0.009, (0.7, 0.7, 1.0)),
        ("template_centroid_base", 0.009, (0.0, 0.9, 1.0)),
    ):
        if key in data and _is_valid_vec(data[key][idx], 3):
            geometries.append(make_sphere(np.asarray(data[key][idx], dtype=np.float32)[:3], radius, color))

    palm_center = data["selected_palm_center_base"][idx]
    palm_normal = data["selected_palm_normal_base"][idx]
    if _is_valid_vec(palm_center, 3) and _is_valid_vec(palm_normal, 3):
        center = np.asarray(palm_center, dtype=np.float32)[:3]
        normal = np.asarray(palm_normal, dtype=np.float32)[:3]
        norm = float(np.linalg.norm(normal))
        if norm > 1e-6:
            normal = normal / norm
            normal_points = np.stack([center, center + normal * float(normal_length_m)], axis=0)
            geometries.append(make_line_set(normal_points, [[0, 1]], (1.0, 0.95, 0.1)))

    eef_pose = data["eef_pose_base"][idx]
    if _is_valid_vec(eef_pose, 6):
        geometries.append(make_pose_frame(np.asarray(eef_pose, dtype=np.float32), size=0.055))

    geometries.extend(build_template_axis_geometries(data, idx, axis_length_m=template_axis_length_m))

    return geometries


def print_frame_status(data: dict[str, np.ndarray], frame_index: int) -> None:
    idx = int(frame_index)
    count = int(data["frame_count"])
    record_clock = _string_at(data, "record_clock_text", idx)
    if not record_clock and "record_elapsed_s" in data:
        elapsed_s = float(data["record_elapsed_s"][idx])
        if np.isfinite(elapsed_s):
            record_clock = f"{elapsed_s:.3f}s"
    if not record_clock:
        record_clock = f"unix={float(data['timestamp_unix_s'][idx]):.3f}"
    label = _string_at(data, "object_label", idx) or "-"
    template_id = _string_at(data, "template_id", idx) or "-"
    reason = _string_at(data, "shape_fitting_reason", idx) or "-"
    source = _string_at(data, "measurement_source", idx) or "-"

    def fmt(key: str, length: int = 3) -> str:
        value = data[key][idx]
        if not _is_valid_vec(value, length):
            return "None"
        vec = np.asarray(value, dtype=np.float32).reshape(-1)[:length]
        return "(" + ", ".join(f"{float(v):.3f}" for v in vec) + ")"

    print(
        f"[FRAME {idx + 1}/{count}] t={record_clock} label={label} template={template_id} "
        f"fit={reason} src={source} object={fmt('object_point_base')} "
        f"grasp={fmt('grasp_point_base')} eef={fmt('eef_pose_base', 6)}",
        flush=True,
    )


class PlaybackState:
    def __init__(self, frame_count: int, play_hz: float) -> None:
        self.frame_count = max(int(frame_count), 1)
        self.index = 0
        self.playing = False
        self.frame_period_s = 1.0 / max(float(play_hz), 1e-6)
        self.last_advance_time = 0.0
        self.changed = True

    def step(self, delta: int) -> None:
        self.index = int(np.clip(self.index + int(delta), 0, self.frame_count - 1))
        self.changed = True

    def toggle_playing(self) -> None:
        self.playing = not self.playing
        self.last_advance_time = time.perf_counter()

    def advance_if_due(self) -> None:
        if not self.playing:
            return
        now = time.perf_counter()
        if now - self.last_advance_time < self.frame_period_s:
            return
        self.last_advance_time = now
        self.index = (self.index + 1) % self.frame_count
        self.changed = True


def main() -> int:
    args = parse_args()
    data = load_debug_3d_npz(args.recording)
    frame_count = int(data["frame_count"])
    if frame_count <= 0:
        raise RuntimeError("Recording contains no frames.")

    if args.dry_run:
        geometries = build_frame_geometries(
            data,
            0,
            normal_length_m=args.normal_length,
            template_axis_length_m=args.template_axis_length,
        )
        print(f"[INFO] Loaded {frame_count} frames and built {len(geometries)} geometries for frame 0.")
        return 0

    state = PlaybackState(frame_count, args.play_hz)
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="Handover 3D Debug Playback", width=1440, height=960)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
    render_option.point_size = float(args.point_size)
    render_option.line_width = 2.0

    def request_close(vis: o3d.visualization.Visualizer) -> bool:
        vis.close()
        return False

    visualizer.register_key_callback(ord(" "), lambda vis: (state.toggle_playing() or False))
    visualizer.register_key_callback(262, lambda vis: (state.step(1) or False))
    visualizer.register_key_callback(263, lambda vis: (state.step(-1) or False))
    visualizer.register_key_callback(265, lambda vis: (state.step(10) or False))
    visualizer.register_key_callback(264, lambda vis: (state.step(-10) or False))
    visualizer.register_key_callback(ord("Q"), request_close)
    visualizer.register_key_callback(256, request_close)

    try:
        while True:
            state.advance_if_due()
            if state.changed:
                visualizer.clear_geometries()
                for geometry in build_frame_geometries(
                    data,
                    state.index,
                    normal_length_m=args.normal_length,
                    template_axis_length_m=args.template_axis_length,
                ):
                    visualizer.add_geometry(geometry, reset_bounding_box=(state.index == 0))
                print_frame_status(data, state.index)
                state.changed = False
            if not visualizer.poll_events():
                break
            visualizer.update_renderer()
            time.sleep(0.005)
    finally:
        visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
