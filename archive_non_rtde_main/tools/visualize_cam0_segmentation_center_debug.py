#!/usr/bin/env python3
"""Visualize the cam0 YOLO segmentation center transformed into robot base coordinates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("open3d is required for tools/visualize_cam0_segmentation_center_debug.py") from exc

from object_pt_extraction.segmentation_engine import select_instances
from perception.object_worker import ObjectWorkerCam0
from system.dual_sensor_hub import DualSensorHub
from utils.depth_lifter import deproject_pixel_to_point, sample_depth_at_keypoint


WINDOW_NAME_2D = "cam0 segmentation center debug"
WINDOW_NAME_3D = "cam0 segmentation center in robot base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the cam0 YOLO segmentation center transformed into robot base coordinates."
    )
    parser.add_argument("--config", default="configs/handover.yaml", help="Path to the handover config file.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run until closed.")
    parser.add_argument("--print-every", type=int, default=10, help="Print one status line every N frames.")
    parser.add_argument("--patch-radius", type=int, default=2, help="Depth sampling patch radius in pixels.")
    parser.add_argument("--show-2d", action="store_true", help="Show the 2D debug window.")
    return parser.parse_args()



def compute_segmentation_center(mask: np.ndarray, bbox_xyxy: np.ndarray | None) -> tuple[int, int] | None:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        return None

    ys, xs = np.nonzero(mask_bool)
    if len(xs) > 0:
        u = int(round(float(np.mean(xs))))
        v = int(round(float(np.mean(ys))))
        return u, v

    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None

    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    return int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.5))



def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    output = image_bgr.copy()
    if mask is None:
        return output
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != output.shape[:2]:
        return output
    overlay = np.zeros_like(output, dtype=np.uint8)
    overlay[mask_bool] = np.array([0, 180, 255], dtype=np.uint8)
    return cv.addWeighted(output, 1.0, overlay, 0.35, 0.0)



def draw_text_block(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image_bgr.copy()
    y = 24
    for line in lines:
        cv.putText(output, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv.LINE_AA)
        cv.putText(output, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv.LINE_AA)
        y += 22
    return output



def format_vec2(values: tuple[int, int] | None) -> str:
    if values is None:
        return "None"
    return f"({values[0]}, {values[1]})"



def format_vec3(values: np.ndarray | None) -> str:
    if values is None:
        return "None"
    arr = np.asarray(values, dtype=np.float32).reshape(3)
    return f"({arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f})"



def make_base_scene(point_base: np.ndarray | None) -> list[o3d.geometry.Geometry]:
    geometries: list[o3d.geometry.Geometry] = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)]
    if point_base is None:
        return geometries

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(np.array([0.95, 0.2, 0.2], dtype=np.float64))
    sphere.translate(np.asarray(point_base, dtype=np.float64).reshape(3))
    geometries.append(sphere)
    return geometries



def print_status(
    frame_index: int,
    selected_count: int,
    class_name: str | None,
    center_uv: tuple[int, int] | None,
    depth_m: float | None,
    point_cam0: np.ndarray | None,
    point_base: np.ndarray | None,
) -> None:
    depth_label = "None" if depth_m is None or not np.isfinite(depth_m) else f"{depth_m:.3f}"
    print(
        f"[frame {frame_index:05d}] "
        f"instances={selected_count} "
        f"class={class_name} "
        f"center_uv={format_vec2(center_uv)} "
        f"depth={depth_label} "
        f"cam0={format_vec3(point_cam0)} "
        f"base={format_vec3(point_base)}"
    )



def main() -> int:
    args = parse_args()

    object_worker = ObjectWorkerCam0.from_config(args.config)
    sensor_hub = DualSensorHub.from_config(args.config)

    if args.show_2d:
        cv.namedWindow(WINDOW_NAME_2D, cv.WINDOW_NORMAL)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=WINDOW_NAME_3D, width=1280, height=800)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.02, 0.02, 0.02], dtype=np.float64)
    render_option.point_size = 5.0

    frame_index = 0
    with sensor_hub:
        while True:
            snapshot = sensor_hub.read_next_pair()
            frame_bundle = snapshot.cam0
            segmentation_result = object_worker.segmentation_engine.predict(frame_bundle.color_image)
            selected_instances = select_instances(
                segmentation_result.instances,
                mode=object_worker.selection_mode,
                class_names=object_worker.selection_class_names or None,
            )

            primary = None
            if selected_instances:
                primary = max(selected_instances, key=lambda instance: float(instance.score))

            center_uv: tuple[int, int] | None = None
            depth_m: float | None = None
            point_cam0: np.ndarray | None = None
            point_base: np.ndarray | None = None
            class_name: str | None = None
            combined_mask = None

            if primary is not None:
                class_name = str(primary.class_name)
                combined_mask = np.asarray(primary.mask, dtype=bool)
                center_uv = compute_segmentation_center(combined_mask, primary.bbox)
                if center_uv is not None:
                    depth_value, _ = sample_depth_at_keypoint(
                        frame_bundle.depth_image_m,
                        center_uv[0],
                        center_uv[1],
                        patch_radius=args.patch_radius,
                        min_depth_m=object_worker.min_depth_m,
                        max_depth_m=object_worker.max_depth_m,
                    )
                    if np.isfinite(depth_value):
                        depth_m = float(depth_value)
                        point_cam0 = deproject_pixel_to_point(
                            center_uv[0],
                            center_uv[1],
                            depth_m,
                            frame_bundle.intrinsics,
                        )
                        point_base = object_worker.transform_chain.transform_point_camera_to_base(0, point_cam0)

            geometries = make_base_scene(point_base)
            vis.clear_geometries()
            for geometry in geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
            vis.poll_events()
            vis.update_renderer()

            if args.show_2d:
                debug_image = overlay_mask(frame_bundle.color_image, combined_mask)
                if center_uv is not None:
                    cv.circle(debug_image, center_uv, 6, (0, 0, 255), -1, cv.LINE_AA)
                text_lines = [
                    f"cam0 segmentation center debug",
                    f"class={class_name} instances={len(selected_instances)} infer_ms={segmentation_result.infer_ms:.1f}",
                    f"center_uv={format_vec2(center_uv)} depth_m={('None' if depth_m is None else f'{depth_m:.3f}')}",
                    f"point_cam0={format_vec3(point_cam0)}",
                    f"point_base={format_vec3(point_base)}",
                ]
                debug_image = draw_text_block(debug_image, text_lines)
                cv.imshow(WINDOW_NAME_2D, debug_image)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

            if args.print_every > 0 and frame_index % args.print_every == 0:
                print_status(
                    frame_index=frame_index,
                    selected_count=len(selected_instances),
                    class_name=class_name,
                    center_uv=center_uv,
                    depth_m=depth_m,
                    point_cam0=point_cam0,
                    point_base=point_base,
                )

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

    vis.destroy_window()
    if args.show_2d:
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
