#!/usr/bin/env python3
"""Live 2D debug viewer for dual-camera perception and FoundationPose tracking."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

import cv2 as cv
import numpy as np

from robot_control_rtde_v2 import (
    apply_config_defaults,
    apply_fdct_depth_to_object_frames,
    build_dual_perception_pipeline,
    load_yaml_config,
    parse_prompt_classes,
    render_camera_mask_preview,
    render_cam0_perception_debug,
    render_depth,
)


class VisualizerSharedState:
    """Minimal render-state shim for perception-only visualization."""

    def __init__(self) -> None:
        self.home_object_pixel = None
        self.home_object_locked = False

    @staticmethod
    def get_follow_status_text() -> list[str]:
        return ["mode=perception_only", "robot=disabled"]


def format_vec3(values) -> str:
    if values is None:
        return "None"
    vec = np.asarray(values, dtype=np.float32).reshape(3)
    return f"({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize dual RealSense segmentation, merged object cloud, and FoundationPose tracking without robot actions."
    )
    parser.add_argument("--config", default="configs/handover.yaml", help="Path to the handover config file.")
    parser.add_argument("--model", default="yoloe-26x-seg.pt", help="Model name or local weights path.")
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=None,
        help="Text prompt classes for YOLOE, e.g. --prompt cup wine_glass",
    )
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RealSense stream FPS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. cpu, 0, cuda:0.")
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
    parser.add_argument("--show-depth", action="store_true", help="Show the cam0 depth window.")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="Upper bound for depth visualization.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run until closed.")
    parser.add_argument("--print-every", type=int, default=10, help="Print one status line every N frames.")
    parser.add_argument("--profile-every", type=int, default=10, help="Print one timing breakdown every N frames. 0 disables profiling logs.")
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
    parser.add_argument("--fdct-checkpoint", default=None, help="Path to FDCT checkpoint.")
    parser.add_argument("--fdct-device", default=None, help="Torch device for FDCT: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--fdct-debug-stats", action="store_true", help="Print FDCT raw/completed depth stats.")
    return parser.parse_args()


def build_model_label(args: argparse.Namespace, pipeline: dict) -> str:
    prompt_classes = parse_prompt_classes(args.prompt)
    model_label = args.model if not prompt_classes else f"{args.model} ({','.join(prompt_classes)})"
    if pipeline.get("fdct_depth_completer") is not None:
        model_label = f"{model_label} + FDCT depth({args.fdct_cameras})"
    return model_label


def prepare_args_for_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Backfill fields expected by robot_control_rtde_v2.apply_config_defaults()."""
    defaults = {
        "robot_ip": None,
        "control_hz": None,
        "follow_z": None,
        "workspace_x": None,
        "workspace_y": None,
        "workspace_z": None,
        "calib_pkl": None,
    }
    for field_name, default_value in defaults.items():
        if not hasattr(args, field_name):
            setattr(args, field_name, default_value)
    return args


def print_status(
    frame_index: int,
    pose_tracking,
    merged_object,
    selected_hand,
    grasp_target,
) -> None:
    print(
        f"[frame {frame_index:05d}] "
        f"pose_mode={pose_tracking.mode} "
        f"backend={pose_tracking.backend_available} "
        f"pose_valid={pose_tracking.valid} "
        f"label={pose_tracking.label} "
        f"tpl={pose_tracking.template_id} "
        f"conf={pose_tracking.tracking_confidence:.2f} "
        f"mask_iou={pose_tracking.mask_iou:.2f} "
        f"depth_inlier={pose_tracking.depth_inlier_ratio:.2f} "
        f"reinit={pose_tracking.reinit_reason} "
        f"merged_pts={merged_object.merged_point_count} "
        f"hand_cam={selected_hand.selected_camera} "
        f"grasp_valid={grasp_target.valid}",
        flush=True,
    )


def print_profile(
    frame_index: int,
    stage_ms: dict[str, float],
    pose_tracking,
    pose_debug,
    smoothed_fps: float,
) -> None:
    print(
        f"[profile {frame_index:05d}] "
        f"read={stage_ms.get('read_pair', 0.0):.1f}ms "
        f"fdct={stage_ms.get('fdct', 0.0):.1f}ms "
        f"obj0={stage_ms.get('object_cam0', 0.0):.1f}ms "
        f"obj1={stage_ms.get('object_cam1', 0.0):.1f}ms "
        f"hand0={stage_ms.get('hand_cam0', 0.0):.1f}ms "
        f"hand1={stage_ms.get('hand_cam1', 0.0):.1f}ms "
        f"merge={stage_ms.get('merge', 0.0):.1f}ms "
        f"pose_total={stage_ms.get('pose_update', 0.0):.1f}ms "
        f"pose_scale={getattr(pose_debug, 'scale_estimation_ms', 0.0):.1f}ms "
        f"pose_{getattr(pose_debug, 'backend_stage', 'none')}={getattr(pose_debug, 'backend_call_ms', 0.0):.1f}ms "
        f"pose_health={getattr(pose_debug, 'health_check_ms', 0.0):.1f}ms "
        f"fusion={stage_ms.get('fusion', 0.0):.1f}ms "
        f"grasp={stage_ms.get('grasp', 0.0):.1f}ms "
        f"render={stage_ms.get('render', 0.0):.1f}ms "
        f"total={stage_ms.get('total', 0.0):.1f}ms "
        f"fps={smoothed_fps:.2f} "
        f"mode={pose_tracking.mode} "
        f"valid={pose_tracking.valid}",
        flush=True,
    )


def draw_text_block(image_bgr: np.ndarray, lines: list[str], origin=(12, 28), line_step: int = 24) -> np.ndarray:
    output = image_bgr.copy()
    x0, y0 = origin
    for line_index, text_line in enumerate(lines):
        point = (int(x0), int(y0 + line_index * line_step))
        cv.putText(output, text_line, point, cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(output, text_line, point, cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA)
    return output


def project_cam_points(points_cam: np.ndarray, intrinsics: dict[str, float], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_cam, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    z = points[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    points = points[valid]
    z = z[valid]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    u = np.round((points[:, 0] * fx / z) + cx).astype(np.int32)
    v = np.round((points[:, 1] * fy / z) + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    if not np.any(in_bounds):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    return np.stack([u[in_bounds], v[in_bounds]], axis=1), z[in_bounds].astype(np.float32)


def draw_pose_axes(
    image_bgr: np.ndarray,
    pose_cam0,
    intrinsics: dict[str, float],
    axis_length_m: float,
) -> np.ndarray:
    if pose_cam0 is None:
        return image_bgr
    pose = np.asarray(pose_cam0, dtype=np.float32).reshape(4, 4)
    axis_length_m = max(float(axis_length_m), 0.02)
    axis_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float32,
    )
    rotation = pose[:3, :3]
    translation = pose[:3, 3].reshape(1, 3)
    points_cam = axis_points @ rotation.T + translation
    pixels, _ = project_cam_points(points_cam, intrinsics, image_bgr.shape[1], image_bgr.shape[0])
    if len(pixels) != 4:
        return image_bgr
    origin = tuple(int(v) for v in pixels[0])
    axes = [
        (tuple(int(v) for v in pixels[1]), (0, 0, 255), "x"),
        (tuple(int(v) for v in pixels[2]), (0, 255, 0), "y"),
        (tuple(int(v) for v in pixels[3]), (255, 0, 0), "z"),
    ]
    output = image_bgr.copy()
    for endpoint, color_bgr, label in axes:
        cv.line(output, origin, endpoint, color_bgr, 2, cv.LINE_AA)
        cv.circle(output, endpoint, 4, color_bgr, -1, cv.LINE_AA)
        cv.putText(output, label, (endpoint[0] + 6, endpoint[1] - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(output, label, (endpoint[0] + 6, endpoint[1] - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv.LINE_AA)
    cv.circle(output, origin, 5, (255, 255, 255), -1, cv.LINE_AA)
    return output


def build_tracking_visualization(
    snapshot,
    pose_tracking,
    pose_debug,
) -> np.ndarray:
    image_bgr = np.asarray(snapshot.cam0.color_image).copy()
    rendered_mask = getattr(pose_debug, "rendered_mask", None)
    if rendered_mask is not None and np.any(rendered_mask):
        rendered_mask_u8 = np.asarray(rendered_mask, dtype=np.uint8)
        mask_overlay = np.zeros_like(image_bgr, dtype=np.uint8)
        mask_overlay[rendered_mask_u8.astype(bool)] = np.array([255, 96, 0], dtype=np.uint8)
        image_bgr = cv.addWeighted(image_bgr, 0.78, mask_overlay, 0.22, 0.0)
        contours, _ = cv.findContours(rendered_mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(image_bgr, contours, -1, (0, 200, 255), 2, cv.LINE_AA)

    projected_pixels = getattr(pose_debug, "projected_pixels", None)
    if projected_pixels is not None and len(projected_pixels) > 0:
        overlay = image_bgr.copy()
        overlay[projected_pixels[:, 1], projected_pixels[:, 0]] = np.array([255, 64, 255], dtype=np.uint8)
        image_bgr = cv.addWeighted(image_bgr, 0.78, overlay, 0.22, 0.0)
        for u, v in projected_pixels[:: max(1, len(projected_pixels) // 150)].tolist():
            cv.circle(image_bgr, (int(u), int(v)), 2, (255, 64, 255), -1, cv.LINE_AA)

    axis_length_m = 0.05
    if pose_tracking.scale_xyz is not None:
        axis_length_m = max(float(max(pose_tracking.scale_xyz)), 0.05)
    image_bgr = draw_pose_axes(
        image_bgr,
        pose_tracking.pose_cam0,
        snapshot.cam0.intrinsics,
        axis_length_m=axis_length_m,
    )

    lines = [
        "FoundationPose Tracking View",
        f"backend={pose_tracking.backend_available} valid={pose_tracking.valid}",
        f"mode={pose_tracking.mode} conf={pose_tracking.tracking_confidence:.2f}",
        f"label={pose_tracking.label} tpl={pose_tracking.template_id}",
        f"scale={format_vec3(pose_tracking.scale_xyz)}",
        f"mask_iou={pose_tracking.mask_iou:.2f} depth_inlier={pose_tracking.depth_inlier_ratio:.2f}",
        f"reinit_reason={pose_tracking.reinit_reason}",
    ]
    if not pose_tracking.backend_available:
        lines.append("FP inactive: backend unavailable")
    elif not pose_tracking.valid:
        lines.append("FP not yet tracking: waiting register/track")
    else:
        lines.append("FP active: projected mesh + pose axes")
    return draw_text_block(image_bgr, lines)


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    args = prepare_args_for_config_defaults(args)
    args = apply_config_defaults(args, config)

    pipeline = build_dual_perception_pipeline(args)
    sensor_hub = pipeline["sensor_hub"]
    sensor_hub.start()

    shared_state = VisualizerSharedState()
    window_name = "cam0_pose_tracking"
    cam1_window_name = "cam1_perception"
    tracking_window_name = "cam0_template_tracking"
    depth_window_name = "cam0_depth"

    previous_hand_approach = False
    frame_index = 0
    last_loop_time = time.perf_counter()
    smoothed_fps = 0.0
    model_label = build_model_label(args, pipeline)

    try:
        while True:
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            current_time = time.time()
            stage_ms: dict[str, float] = {}
            loop_start = time.perf_counter()

            stage_start = time.perf_counter()
            snapshot = sensor_hub.read_next_pair()
            stage_ms["read_pair"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            object_frame_cam0, object_frame_cam1 = apply_fdct_depth_to_object_frames(snapshot, pipeline, args)
            stage_ms["fdct"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            object_cam0 = pipeline["object_worker_cam0"].process_frame(object_frame_cam0, frame_id=snapshot.pair_index)
            stage_ms["object_cam0"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            object_cam1 = pipeline["object_worker_cam1"].process_frame(object_frame_cam1, frame_id=snapshot.pair_index)
            stage_ms["object_cam1"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            hand_cam0 = pipeline["hand_worker_cam0"].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            stage_ms["hand_cam0"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            hand_cam1 = pipeline["hand_worker_cam1"].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
            stage_ms["hand_cam1"] = (time.perf_counter() - stage_start) * 1000.0

            selected_hand = pipeline["hand_selector"].process_states(hand_cam0, hand_cam1)

            stage_start = time.perf_counter()
            merged_object = pipeline["object_merger"].process_states(
                object_cam0,
                object_cam1,
                hand_approach_detected=previous_hand_approach,
            )
            stage_ms["merge"] = (time.perf_counter() - stage_start) * 1000.0

            pose_mask_cam0 = None
            object_debug_cam0 = getattr(pipeline["object_worker_cam0"], "last_debug", None)
            if object_debug_cam0 is not None:
                pose_mask_cam0 = getattr(object_debug_cam0, "combined_mask", None)
            pipeline["pose_tracker"].set_anchor_observation(object_frame_cam0, pose_mask_cam0)

            stage_start = time.perf_counter()
            pose_tracking = pipeline["pose_tracker"].update(
                snapshot,
                object_cam0,
                object_cam1,
                merged_object,
            )
            stage_ms["pose_update"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            fusion_state = pipeline["fusion"].process_states(
                merged_object,
                selected_hand,
                now_timestamp=current_time,
            )
            stage_ms["fusion"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            grasp_target = pipeline["grasp_planner"].process_states(
                merged_object,
                selected_hand,
                fusion_state,
            )
            stage_ms["grasp"] = (time.perf_counter() - stage_start) * 1000.0
            previous_hand_approach = bool(
                fusion_state.hand_approach_detected or fusion_state.hand_approach_latched
            )

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now

            stage_start = time.perf_counter()
            cam0_view = render_cam0_perception_debug(
                snapshot,
                pipeline,
                merged_object,
                selected_hand,
                fusion_state,
                pose_tracking,
                grasp_target,
                shared_state,
                model_label,
                smoothed_fps,
            )
            cam1_view = render_camera_mask_preview(
                snapshot,
                pipeline,
                pipeline["object_worker_cam1"],
                selected_hand,
                fusion_state,
                pose_tracking,
                grasp_target,
                camera_label="cam1",
            )
            tracking_view = build_tracking_visualization(
                snapshot,
                pose_tracking,
                getattr(pipeline["pose_tracker"], "last_debug", None),
            )
            stage_ms["render"] = (time.perf_counter() - stage_start) * 1000.0
            stage_ms["total"] = (time.perf_counter() - loop_start) * 1000.0

            cv.imshow(window_name, cam0_view)
            cv.imshow(cam1_window_name, cam1_view)
            cv.imshow(tracking_window_name, tracking_view)
            if args.show_depth:
                cv.imshow(depth_window_name, render_depth(snapshot.cam0.depth_image_m, args.depth_max_m))

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_index += 1
            if args.print_every > 0 and (frame_index == 1 or frame_index % args.print_every == 0):
                print_status(
                    frame_index,
                    pose_tracking,
                    merged_object,
                    selected_hand,
                    grasp_target,
                )
            if args.profile_every > 0 and (frame_index == 1 or frame_index % args.profile_every == 0):
                print_profile(
                    frame_index,
                    stage_ms,
                    pose_tracking,
                    getattr(pipeline["pose_tracker"], "last_debug", None),
                    smoothed_fps,
                )
    finally:
        cv.destroyAllWindows()
        sensor_hub.stop()
        pipeline["hand_worker_cam0"].close()
        pipeline["hand_worker_cam1"].close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
