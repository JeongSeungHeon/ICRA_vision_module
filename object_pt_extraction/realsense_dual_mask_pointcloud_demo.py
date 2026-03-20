import argparse
from pathlib import Path
import time
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from object_pt_extraction.dual_realsense_manager import DualRealSenseManager
from object_pt_extraction.extrinsics import convert_points_between_cameras, load_camera_extrinsic
from object_pt_extraction.pointcloud_utils import (
    build_point_cloud_from_instances,
    colorize_selected_mask,
    render_projected_point_cloud_overlay,
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


def parse_args():
    # 듀얼 RealSense 입력, YOLOE 추론, extrinsic 정렬 설정을 한 번에 받는다.
    parser = argparse.ArgumentParser(
        description="Run dual RealSense local point-cloud generation with nearest-timestamp pairing."
    )
    parser.add_argument("--model", default="yoloe-26m-seg.pt", help="Model name or local weights path.")
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=None,
        help="Text prompt classes for YOLOE, e.g. --prompt bottle or --prompt person,bus",
    )
    parser.add_argument("--serials", nargs="*", default=None, help="Two RealSense serials. Defaults to the first two detected devices.")
    parser.add_argument(
        "--camera-ids",
        nargs=2,
        type=int,
        default=[0, 1],
        help="Calibration camera ids that correspond to the two serials in order.",
    )
    parser.add_argument("--target-camera-id", type=int, default=0, help="Target calibration camera id used for aligned preview.")
    parser.add_argument("--extrinsics-dir", default="camera_parameters", help="Directory containing rot_trans_c*.dat files.")
    parser.add_argument(
        "--translation-unit",
        choices=["mm", "m"],
        default="mm",
        help="Translation unit stored in the extrinsic files.",
    )
    parser.add_argument("--anchor-index", type=int, default=0, help="Anchor camera index for nearest-timestamp pairing.")
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
    parser.add_argument("--select-class", nargs="*", default=None, help="Optional class-name filter applied before point-cloud generation.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on supported devices.")
    parser.add_argument("--stride", type=int, default=2, help="Sample every Nth mask pixel before deprojection.")
    parser.add_argument("--max-points", type=int, default=20000, help="Maximum number of point-cloud samples.")
    parser.add_argument("--min-depth-m", type=float, default=0.1, help="Minimum valid depth for point cloud generation.")
    parser.add_argument("--max-depth-m", type=float, default=1.5, help="Maximum valid depth for point cloud generation.")
    parser.add_argument("--show-depth", action="store_true", help="Show separate depth windows for both cameras.")
    parser.add_argument("--show-mask", action="store_true", help="Show separate binary mask windows for both cameras.")
    parser.add_argument("--save-dir", default="outputs/dual_pointcloud_demo", help="Directory used when saving paired snapshots with the `s` key.")
    parser.add_argument("--save-every", type=int, default=0, help="Automatically save every N paired frames. Use 0 to disable auto-save.")
    return parser.parse_args()


def _make_model_label(model_name, prompt_classes):
    # 프롬프트가 있을 때는 창 상단에서 모델과 타깃 클래스를 함께 보이게 한다.
    if not prompt_classes:
        return model_name
    return f"{model_name} ({','.join(prompt_classes)})"


def _stack_previews(previews):
    # 좌우 비교가 쉽도록 preview 높이를 맞춘 뒤 가로로 붙인다.
    resized_previews = []
    max_height = max(preview.shape[0] for preview in previews)
    for preview in previews:
        if preview.shape[0] == max_height:
            resized_previews.append(preview)
            continue
        scale = max_height / preview.shape[0]
        resized_previews.append(cv.resize(preview, (int(round(preview.shape[1] * scale)), max_height)))
    return cv.hconcat(resized_previews)


def _overlay_camera_status(frame, serial, timestamp_ms, infer_ms, summary, point_count, camera_label):
    # 각 카메라 창 위에 serial, timestamp, 추론 시간, local cloud 점 개수를 표시한다.
    lines = [
        camera_label,
        f"serial: {serial}",
        f"timestamp: {timestamp_ms:.1f} ms",
        f"infer: {infer_ms:.1f} ms",
        summary,
        f"local points: {point_count}",
    ]
    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 24)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


def _overlay_pair_status(frame, model_label, fps, timestamp_delta_ms, save_dir):
    # 두 카메라 pair 수준의 상태값은 합쳐진 상단 preview에 따로 표기한다.
    lines = [
        f"model: {model_label}",
        f"pair delta: {timestamp_delta_ms:+.1f} ms",
        f"loop fps: {fps:.1f}",
        f"s: save paired snapshot -> {save_dir}",
        "ESC / q: quit",
    ]
    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 26)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv.LINE_AA)


def _process_camera_frame(frame_bundle, segmentation_engine, args, calibration_camera_id):
    # 카메라 한 대 기준으로 segmentation -> instance 선택 -> local point cloud 생성까지 처리한다.
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
    summary = format_instance_summary(selected_instances)
    pointcloud_preview = render_point_cloud_preview(points_xyz, colors_rgb)
    return {
        "frame_bundle": frame_bundle,
        "calibration_camera_id": int(calibration_camera_id),
        "segmentation_result": segmentation_result,
        "selected_instances": selected_instances,
        "combined_mask": combined_mask,
        "points_xyz": points_xyz,
        "colors_rgb": colors_rgb,
        "annotated": annotated,
        "summary": summary,
        "pointcloud_preview": pointcloud_preview,
    }


def _uniform_colors(point_count, rgb_color):
    # 정렬 preview에서는 원본 RGB 대신 카메라별 고정색으로 겹침 정도를 쉽게 본다.
    if point_count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    return np.tile(np.asarray(rgb_color, dtype=np.uint8).reshape(1, 3), (point_count, 1))


def _build_serial_to_camera_id(serials, camera_ids):
    # 실제 연결된 serial 순서를 calibration의 c0/c1 id와 매핑한다.
    if len(serials) != len(camera_ids):
        raise ValueError("serials and camera_ids must have the same length")
    return {serial: int(camera_id) for serial, camera_id in zip(serials, camera_ids)}


def _align_processed_frames(processed_frames, extrinsics_by_camera_id, target_camera_id):
    # 각 카메라 local cloud를 target camera 좌표계로 옮긴다.
    # target camera 자체는 그대로 두고, 나머지 카메라 cloud만 extrinsic으로 변환한다.
    aligned_frames = []
    for processed_frame in processed_frames:
        calibration_camera_id = processed_frame["calibration_camera_id"]
        source_extrinsic = extrinsics_by_camera_id[calibration_camera_id]
        target_extrinsic = extrinsics_by_camera_id[target_camera_id]

        if calibration_camera_id == target_camera_id:
            transformed_points = np.asarray(processed_frame["points_xyz"], dtype=np.float32)
        else:
            transformed_points = convert_points_between_cameras(
                processed_frame["points_xyz"],
                source_extrinsic=source_extrinsic,
                target_extrinsic=target_extrinsic,
            )

        aligned_frame = dict(processed_frame)
        aligned_frame["aligned_points_xyz"] = transformed_points.astype(np.float32)
        aligned_frames.append(aligned_frame)
    return aligned_frames


def _make_alignment_preview(aligned_frames, target_camera_id):
    # 정렬된 cloud를 target camera 실제 이미지 평면으로 다시 투영해서 겹침 정도를 본다.
    target_frame = None
    for aligned_frame in aligned_frames:
        if aligned_frame["calibration_camera_id"] == target_camera_id:
            target_frame = aligned_frame
            break

    if target_frame is None:
        raise RuntimeError(f"Target camera id {target_camera_id} is missing from aligned_frames")

    projection_colors_bgr = [
        (80, 80, 255),
        (0, 255, 255),
    ]
    labels = [
        f"cam{aligned_frames[0]['calibration_camera_id']} -> cam{target_camera_id}",
        f"cam{aligned_frames[1]['calibration_camera_id']} -> cam{target_camera_id}",
    ]
    return render_projected_point_cloud_overlay(
        base_frame_bgr=target_frame["annotated"],
        point_clouds=[aligned_frames[0]["aligned_points_xyz"], aligned_frames[1]["aligned_points_xyz"]],
        intrinsics=target_frame["frame_bundle"].intrinsics,
        point_colors_bgr=projection_colors_bgr,
        labels=labels,
        point_radius=1,
        overlay_alpha=0.78,
    )


def _save_dual_snapshot(save_dir, pair_index, processed_frames, aligned_frames, timestamp_delta_ms, target_camera_id):
    # 디버깅을 위해 local cloud와 target camera 기준 transformed cloud를 모두 저장한다.
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_prefix = f"pair_{pair_index:06d}_dt_{int(round(timestamp_delta_ms)):+d}ms"

    for camera_index, processed_frame in enumerate(processed_frames):
        frame_bundle = processed_frame["frame_bundle"]
        serial_label = frame_bundle.serial or f"cam{camera_index}"
        timestamp_label = int(round(frame_bundle.timestamp_ms))
        base_name = f"{pair_prefix}_{serial_label}_{timestamp_label}"
        save_pointcloud_snapshot(
            save_dir=save_dir,
            base_name=base_name,
            color_overlay=processed_frame["annotated"],
            combined_mask=processed_frame["combined_mask"],
            points_xyz=processed_frame["points_xyz"],
            colors_rgb=processed_frame["colors_rgb"],
        )

    for aligned_frame in aligned_frames:
        calibration_camera_id = aligned_frame["calibration_camera_id"]
        if calibration_camera_id == target_camera_id:
            continue

        frame_bundle = aligned_frame["frame_bundle"]
        serial_label = frame_bundle.serial or f"cam{calibration_camera_id}"
        timestamp_label = int(round(frame_bundle.timestamp_ms))
        base_name = f"{pair_prefix}_{serial_label}_{timestamp_label}_in_cam{target_camera_id}"
        save_pointcloud_snapshot(
            save_dir=save_dir,
            base_name=base_name,
            color_overlay=aligned_frame["annotated"],
            combined_mask=aligned_frame["combined_mask"],
            points_xyz=aligned_frame["aligned_points_xyz"],
            colors_rgb=aligned_frame["colors_rgb"],
        )


def main():
    args = parse_args()
    prompt_classes = parse_prompt_classes(args.prompt)
    # YOLOE는 두 카메라 프레임 모두 같은 엔진 인스턴스로 순차 추론한다.
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
    # pair manager가 anchor 카메라 기준으로 최근접 timestamp 프레임을 묶어준다.
    camera_manager = DualRealSenseManager(
        serials=args.serials,
        width=args.width,
        height=args.height,
        fps=args.fps,
        anchor_index=args.anchor_index,
    )
    # serial 순서와 calibration camera id(c0/c1)를 연결해야 extrinsic을 올바르게 적용할 수 있다.
    serial_to_camera_id = _build_serial_to_camera_id(camera_manager.serials, args.camera_ids)
    extrinsics_by_camera_id = {
        camera_id: load_camera_extrinsic(
            camera_id,
            savefolder=args.extrinsics_dir,
            translation_unit=args.translation_unit,
        )
        for camera_id in args.camera_ids
    }
    if args.target_camera_id not in extrinsics_by_camera_id:
        raise ValueError("target_camera_id must be included in --camera-ids")

    pair_index = 0
    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()
    model_label = _make_model_label(args.model, prompt_classes)

    try:
        while True:
            pair_index += 1
            paired_frames = camera_manager.read_paired_frames()

            # 각 카메라에서 독립적으로 local cloud를 만든다.
            processed_anchor = _process_camera_frame(
                paired_frames.anchor_frame,
                segmentation_engine,
                args,
                calibration_camera_id=serial_to_camera_id[paired_frames.anchor_serial],
            )
            processed_paired = _process_camera_frame(
                paired_frames.paired_frame,
                segmentation_engine,
                args,
                calibration_camera_id=serial_to_camera_id[paired_frames.paired_serial],
            )
            processed_frames = [processed_anchor, processed_paired]
            # local cloud를 target camera 좌표계로 바꿔 정렬 상태를 확인한다.
            aligned_frames = _align_processed_frames(
                processed_frames,
                extrinsics_by_camera_id=extrinsics_by_camera_id,
                target_camera_id=args.target_camera_id,
            )

            for camera_index, processed_frame in enumerate(processed_frames):
                _overlay_camera_status(
                    processed_frame["annotated"],
                    serial=processed_frame["frame_bundle"].serial,
                    timestamp_ms=processed_frame["frame_bundle"].timestamp_ms,
                    infer_ms=processed_frame["segmentation_result"].infer_ms,
                    summary=processed_frame["summary"],
                    point_count=len(processed_frame["points_xyz"]),
                    camera_label=f"cam{camera_index} local cloud",
                )

            color_preview = _stack_previews([processed_anchor["annotated"], processed_paired["annotated"]])
            pointcloud_preview = _stack_previews([processed_anchor["pointcloud_preview"], processed_paired["pointcloud_preview"]])
            # 이 창은 두 cloud가 같은 target camera frame에 들어왔을 때 얼마나 겹치는지 보는 용도다.
            aligned_preview = _make_alignment_preview(aligned_frames, target_camera_id=args.target_camera_id)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now

            _overlay_pair_status(
                color_preview,
                model_label=model_label,
                fps=smoothed_fps,
                timestamp_delta_ms=paired_frames.timestamp_delta_ms,
                save_dir=args.save_dir,
            )

            cv.imshow("dual_realsense_masks", color_preview)
            cv.imshow("dual_realsense_local_pointclouds", pointcloud_preview)
            cv.imshow(f"dual_realsense_aligned_cam{args.target_camera_id}", aligned_preview)

            if args.show_depth:
                cv.imshow(
                    f"depth_{processed_anchor['frame_bundle'].serial}",
                    render_depth(processed_anchor["frame_bundle"].depth_image_m, args.max_depth_m),
                )
                cv.imshow(
                    f"depth_{processed_paired['frame_bundle'].serial}",
                    render_depth(processed_paired["frame_bundle"].depth_image_m, args.max_depth_m),
                )
            if args.show_mask:
                cv.imshow(
                    f"mask_{processed_anchor['frame_bundle'].serial}",
                    render_mask_preview(processed_anchor["combined_mask"]),
                )
                cv.imshow(
                    f"mask_{processed_paired['frame_bundle'].serial}",
                    render_mask_preview(processed_paired["combined_mask"]),
                )

            should_auto_save = args.save_every > 0 and pair_index % args.save_every == 0
            key = cv.waitKey(1) & 0xFF
            if key == ord("s") or should_auto_save:
                _save_dual_snapshot(
                    args.save_dir,
                    pair_index,
                    processed_frames,
                    aligned_frames,
                    paired_frames.timestamp_delta_ms,
                    target_camera_id=args.target_camera_id,
                )
            if key in (27, ord("q")):
                break
    finally:
        camera_manager.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
