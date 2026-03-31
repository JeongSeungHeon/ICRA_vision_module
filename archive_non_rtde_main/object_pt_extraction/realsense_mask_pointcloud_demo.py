import argparse
import time
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2 as cv

from object_pt_extraction.pointcloud_utils import (
    build_point_cloud_from_instances,
    colorize_selected_mask,
    render_depth,
    render_mask_preview,
    render_point_cloud_preview,
    save_pointcloud_snapshot,
)
from object_pt_extraction.segmentation_engine import (
    SegmentationEngine,
    format_instance_summary,
    parse_prompt_classes,
    select_instances,
)
from utils.realsense_stream import RealSenseCamera, list_realsense_serials


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a point cloud from a prompt-selected segmentation mask on a live RealSense stream."
    )
    parser.add_argument("--model", default="yoloe-26m-seg.pt", help="Model name or local weights path.")
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=None,
        help="Text prompt classes for YOLOE, e.g. --prompt bottle or --prompt person,bus",
    )
    parser.add_argument("--serial", default=None, help="RealSense serial number. Defaults to the first detected camera.")
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RealSense stream FPS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. cpu, 0.")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional class id filter.")
    parser.add_argument(
        "--select-mode",
        choices=["all_instances", "highest_score", "class_filter"],
        default="highest_score",
        help="Instance selection policy for downstream point-cloud generation.",
    )
    parser.add_argument(
        "--select-class",
        nargs="*",
        default=None,
        help="Optional class-name filter applied before point-cloud generation.",
    )
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on supported devices.")
    parser.add_argument("--stride", type=int, default=2, help="Sample every Nth mask pixel before deprojection.")
    parser.add_argument("--max-points", type=int, default=20000, help="Maximum number of point-cloud samples.")
    parser.add_argument("--min-depth-m", type=float, default=0.1, help="Minimum valid depth for point cloud generation.")
    parser.add_argument("--max-depth-m", type=float, default=1.5, help="Maximum valid depth for point cloud generation.")
    parser.add_argument("--show-depth", action="store_true", help="Show a second depth preview window.")
    parser.add_argument("--show-mask", action="store_true", help="Show the selected binary mask in a separate window.")
    parser.add_argument(
        "--save-dir",
        default="outputs/pointcloud_demo",
        help="Directory used when saving point-cloud snapshots with the `s` key.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Automatically save every N frames. Use 0 to disable auto-save.",
    )
    return parser.parse_args()


def pick_serial(serial):
    if serial:
        return serial

    serials = list_realsense_serials()
    if not serials:
        raise RuntimeError("No RealSense devices detected.")
    return serials[0]


def overlay_status(frame, serial, model_label, fps, infer_ms, selection_summary, point_count, save_dir):
    lines = [
        f"model: {model_label}",
        f"serial: {serial}",
        f"fps: {fps:.1f}  infer: {infer_ms:.1f} ms",
        selection_summary,
        f"points: {point_count}",
        f"s: save snapshot -> {save_dir}",
        "ESC / q: quit",
    ]

    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 24)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


def save_snapshot(save_dir, frame_index, frame_bundle, color_overlay, combined_mask, points_xyz, colors_rgb):
    serial_label = frame_bundle.serial or "cam0"
    timestamp_label = int(round(frame_bundle.timestamp_ms))
    base_name = f"{serial_label}_{frame_index:06d}_{timestamp_label}"
    return save_pointcloud_snapshot(
        save_dir=save_dir,
        base_name=base_name,
        color_overlay=color_overlay,
        combined_mask=combined_mask,
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
    )


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
    window_name = "realsense_mask_pointcloud"
    pointcloud_window_name = "realsense_pointcloud_preview"
    depth_window_name = "realsense_depth"
    mask_window_name = "realsense_selected_mask"

    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()
    frame_index = 0

    try:
        while True:
            frame_index += 1
            frame_bundle = camera.read()

            segmentation_result = segmentation_engine.predict(frame_bundle.color_image)
            selected_instances = select_instances(
                segmentation_result.instances,
                mode=args.select_mode,
                class_names=args.select_class,
            )
            combined_mask, points_xyz, _, _, colors_rgb = build_point_cloud_from_instances(
                frame_bundle.color_image,
                frame_bundle.depth_image_m,
                frame_bundle.intrinsics,
                selected_instances,
                stride=args.stride,
                max_points=args.max_points,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
            )

            annotated = segmentation_engine.render(segmentation_result)
            annotated = colorize_selected_mask(annotated, combined_mask)
            pointcloud_preview = render_point_cloud_preview(points_xyz, colors_rgb)

            selection_summary = format_instance_summary(selected_instances)
            model_label = args.model if not prompt_classes else f"{args.model} ({','.join(prompt_classes)})"

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now

            overlay_status(
                annotated,
                serial=frame_bundle.serial,
                model_label=model_label,
                fps=smoothed_fps,
                infer_ms=segmentation_result.infer_ms,
                selection_summary=selection_summary,
                point_count=len(points_xyz),
                save_dir=args.save_dir,
            )

            cv.imshow(window_name, annotated)
            cv.imshow(pointcloud_window_name, pointcloud_preview)
            if args.show_depth:
                cv.imshow(depth_window_name, render_depth(frame_bundle.depth_image_m, args.max_depth_m))
            if args.show_mask:
                cv.imshow(mask_window_name, render_mask_preview(combined_mask))

            should_auto_save = args.save_every > 0 and frame_index % args.save_every == 0

            key = cv.waitKey(1) & 0xFF
            if key == ord("s") or should_auto_save:
                save_snapshot(
                    args.save_dir,
                    frame_index,
                    frame_bundle,
                    annotated,
                    combined_mask,
                    points_xyz,
                    colors_rgb,
                )
            if key in (27, ord("q")):
                break
    finally:
        camera.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
