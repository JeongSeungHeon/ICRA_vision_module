import argparse
import time
import sys
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


def parse_args():
    # 실시간 테스트에 필요한 추론/카메라 옵션을 커맨드라인에서 조정할 수 있게 한다.
    parser = argparse.ArgumentParser(
        description="Run a YOLOE segmentation model on a live RealSense color stream."
    )
    parser.add_argument("--model", default="yoloe-26m-seg.pt", help="Model name or local weights path.")
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
    return parser.parse_args()


def pick_serial(serial):
    # 사용자가 serial을 주지 않으면 연결된 첫 번째 RealSense 카메라를 사용한다.
    if serial:
        return serial

    serials = list_realsense_serials()
    if not serials:
        raise RuntimeError("No RealSense devices detected.")
    return serials[0]


def render_depth(depth_image_m, max_depth_m):
    # 깊이 영상은 확인용 보조 창이므로 지정 거리까지만 잘라 컬러맵으로 변환한다.
    max_depth_m = max(max_depth_m, 1e-6)
    clipped = np.clip(depth_image_m, 0.0, max_depth_m)
    scaled = (255.0 * clipped / max_depth_m).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)


def overlay_status(frame, serial, model_name, fps, infer_ms, summary):
    # 모델명, 카메라 serial, FPS, 추론 시간 등을 프레임 위에 직접 덮어쓴다.
    lines = [
        f"model: {model_name}",
        f"serial: {serial}",
        f"fps: {fps:.1f}  infer: {infer_ms:.1f} ms",
        summary,
        "ESC / q: quit",
    ]

    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 26)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv.LINE_AA)


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

    # 기존 프로젝트의 RealSense 래퍼를 그대로 재사용해 컬러/깊이 프레임을 읽는다.
    camera = RealSenseCamera(serial=serial, width=args.width, height=args.height, fps=args.fps)
    window_name = "realsense_yoloe_seg"
    depth_window_name = "realsense_depth"

    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()

    try:
        while True:
            # RealSense에서 정렬된 color/depth 프레임 한 쌍을 가져온다.
            frame_bundle = camera.read()

            # 컬러 프레임 하나를 segmentation 엔진에 넣어 구조화된 결과를 얻는다.
            segmentation_result = segmentation_engine.predict(frame_bundle.color_image)
            annotated = segmentation_engine.render(segmentation_result)
            selected_instances = select_instances(
                segmentation_result.instances,
                mode=args.select_mode,
                class_names=args.select_class,
            )
            summary = format_instance_summary(selected_instances)

            # 루프 주기 기반 FPS를 계산하고 EMA로 살짝 평활화해 표시를 안정화한다.
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
            )

            cv.imshow(window_name, annotated)
            if args.show_depth:
                # 필요할 때만 깊이 프리뷰 창을 추가로 띄워 거리 감각을 같이 본다.
                cv.imshow(depth_window_name, render_depth(frame_bundle.depth_image_m, args.depth_max_m))

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        # 예외가 나더라도 RealSense 파이프라인과 OpenCV 창은 정리한다.
        camera.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
