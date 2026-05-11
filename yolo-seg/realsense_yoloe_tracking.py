from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

try:
    import cv2 as cv
except ImportError:
    cv = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_MODEL_CANDIDATES = [
    Path.cwd() / "yoloe-26l-seg.pt",
    Path.home() / "yolo-seg/yoloe-26l-seg.pt",
    Path.home() / "ICRA_vision_module/yoloe-26l-seg.pt",
    Path.home() / "Downloads/yolo-seg/yoloe-26l-seg.pt",
    Path("/home/ur5/yolo-seg/yoloe-26l-seg.pt"),
    Path("/home/ur5/ICRA_vision_module/yoloe-26l-seg.pt"),
    Path("/home/ur5/ICRA_vision_module _codex/yoloe-26l-seg.pt"),
    Path("/home/ur5/Fast-FoundationStereo/yoloe-26l-seg.pt"),
    Path("/home/ur5/rayst3r/yoloe-26l-seg.pt"),
]

DEFAULT_TEXT_ENCODER_CANDIDATES = [
    Path.cwd() / "mobileclip2_b.ts",
    Path.home() / "yolo-seg/mobileclip2_b.ts",
    Path.home() / "ICRA_vision_module/mobileclip2_b.ts",
    Path.home() / "Downloads/yolo-seg/mobileclip2_b.ts",
    Path("/home/ur5/yolo-seg/mobileclip2_b.ts"),
    Path("/home/ur5/ICRA_vision_module/mobileclip2_b.ts"),
    Path("/home/ur5/ICRA_vision_module _codex/mobileclip2_b.ts"),
    Path("/home/ur5/ICRA_vision_module/archive_non_rtde_main/mobileclip2_b.ts"),
    Path("/home/ur5/Fast-FoundationStereo/mobileclip2_b.ts"),
    Path("/home/ur5/rayst3r/mobileclip2_b.ts"),
]


@dataclass
class FrameBundle:
    color_image: np.ndarray
    depth_image_m: np.ndarray
    intrinsics: dict
    timestamp_ms: float
    serial: str


@dataclass
class TrackState:
    track_id: int
    class_name: str
    center: tuple[int, int]
    bbox: np.ndarray
    misses: int = 0


def require_cv2():
    if cv is None:
        raise RuntimeError(
            "opencv-python is required. Activate the correct environment or install `opencv-python`."
        )


def require_realsense():
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is required. Activate the correct environment or install `pyrealsense2`."
        )


def require_ultralytics():
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is required. Activate the correct environment or install `ultralytics`."
        )


def parse_prompt_classes(prompt_args: Optional[Iterable[str]]) -> list[str]:
    if not prompt_args:
        return []

    prompt_classes: list[str] = []
    for raw_item in prompt_args:
        for class_name in raw_item.split(","):
            normalized = class_name.strip()
            if normalized:
                prompt_classes.append(normalized)
    return prompt_classes


def resolve_model_path(model_arg: Optional[str]) -> Union[Path, str]:
    if model_arg:
        candidate = Path(model_arg).expanduser()
        return candidate.resolve() if candidate.exists() else model_arg

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    return "yoloe-26l-seg.pt"


def list_realsense_serials() -> list[str]:
    require_realsense()
    context = rs.context()
    return [
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    ]


class RealSenseCamera:
    def __init__(
        self,
        serial: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 15,
    ) -> None:
        require_realsense()

        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)

        if serial:
            self.config.enable_device(serial)

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        self.serial = self.device.get_info(rs.camera_info.serial_number)
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = float(self.depth_sensor.get_depth_scale())

        for _ in range(max(1, warmup_frames)):
            self.pipeline.wait_for_frames()

    def read(self) -> FrameBundle:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError(f"Failed to read aligned frames from RealSense {self.serial}.")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_image_m = depth_image * self.depth_scale
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        return FrameBundle(
            color_image=color_image,
            depth_image_m=depth_image_m,
            intrinsics={
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.ppx),
                "cy": float(intrinsics.ppy),
            },
            timestamp_ms=float(frames.get_timestamp()),
            serial=self.serial,
        )

    def stop(self) -> None:
        self.pipeline.stop()


def pick_serial(serial: Optional[str]) -> str:
    if serial:
        return serial

    serials = list_realsense_serials()
    if not serials:
        raise RuntimeError("No RealSense devices detected.")
    return serials[0]


def build_yoloe_model(model_name: Union[str, Path]):
    require_ultralytics()

    try:
        from ultralytics import YOLOE

        return YOLOE(str(model_name))
    except ImportError:
        return YOLO(str(model_name))
    except Exception:
        return YOLO(str(model_name))


def configure_yoloe_prompt(model, prompt_classes: list[str]) -> None:
    if not prompt_classes:
        return

    if hasattr(model, "get_text_pe") and hasattr(model, "set_classes"):
        try:
            text_pe = model.get_text_pe(prompt_classes)
            model.set_classes(prompt_classes, text_pe)
            return
        except Exception:
            pass

    if hasattr(model, "set_classes"):
        model.set_classes(prompt_classes)
        return

    raise RuntimeError(
        "The loaded Ultralytics model does not support text-prompt class configuration."
    )


def ensure_local_text_encoder_asset() -> Optional[Path]:
    filename = "mobileclip2_b.ts"
    cwd_asset = Path.cwd() / filename
    if cwd_asset.exists():
        return cwd_asset

    if cwd_asset.is_symlink():
        cwd_asset.unlink()

    for candidate in DEFAULT_TEXT_ENCODER_CANDIDATES:
        if not candidate.exists():
            continue
        if candidate.resolve() == cwd_asset:
            return cwd_asset

        try:
            cwd_asset.symlink_to(candidate)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(candidate, cwd_asset)
        return cwd_asset if cwd_asset.exists() else candidate

    return None


def get_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def render_depth(depth_image_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    require_cv2()
    max_depth_m = max(max_depth_m, 1e-6)
    clipped = np.clip(depth_image_m, 0.0, max_depth_m)
    scaled = (255.0 * clipped / max_depth_m).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)


def stable_color(track_id: int) -> tuple[int, int, int]:
    color_seed = int(track_id) * 0x45D9F3B
    return (
        64 + (color_seed & 0x7F),
        64 + ((color_seed >> 8) & 0x7F),
        64 + ((color_seed >> 16) & 0x7F),
    )


def estimate_depth_m(mask: np.ndarray, depth_image_m: np.ndarray) -> Optional[float]:
    if mask.shape != depth_image_m.shape:
        return None

    valid_depths = depth_image_m[mask]
    valid_depths = valid_depths[np.isfinite(valid_depths) & (valid_depths > 0.0)]
    if valid_depths.size == 0:
        return None

    lower = np.percentile(valid_depths, 15.0)
    upper = np.percentile(valid_depths, 85.0)
    trimmed = valid_depths[(valid_depths >= lower) & (valid_depths <= upper)]
    if trimmed.size == 0:
        trimmed = valid_depths
    return float(np.median(trimmed))


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(np.float32)
    bx1, by1, bx2, by2 = box_b.astype(np.float32)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0.0 else 0.0


def compute_mask_center(mask: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        x1, y1, x2, y2 = bbox_xyxy.astype(int)
        return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

    return int(np.mean(xs)), int(np.mean(ys))


def draw_text_block(frame: np.ndarray, lines: list[str]) -> None:
    require_cv2()
    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 24)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv.LINE_AA)


def extract_track_rows(result, depth_image_m: np.ndarray) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None or result.masks.data is None:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = result.masks.data.detach().cpu().numpy() > 0.5

    track_ids = None
    if getattr(result.boxes, "id", None) is not None:
        track_ids = result.boxes.id.detach().cpu().numpy().astype(int)

    rows: list[dict] = []
    count = min(len(boxes_xyxy), len(scores), len(class_ids), len(masks))
    for index in range(count):
        class_id = int(class_ids[index])
        class_name = get_class_name(result.names, class_id)
        mask = masks[index]
        center = compute_mask_center(mask, boxes_xyxy[index])
        depth_m = estimate_depth_m(mask, depth_image_m)
        rows.append(
            {
                "track_id": int(track_ids[index]) if track_ids is not None else None,
                "class_name": str(class_name),
                "score": float(scores[index]),
                "bbox": boxes_xyxy[index].astype(np.float32),
                "mask": mask,
                "center": center,
                "depth_m": depth_m,
            }
        )
    return rows


def summarize_tracks(rows: list[dict]) -> str:
    if not rows:
        return "tracks: 0"

    by_class: dict[str, int] = {}
    tracked = 0
    for row in rows:
        by_class[row["class_name"]] = by_class.get(row["class_name"], 0) + 1
        if row["track_id"] is not None:
            tracked += 1

    class_summary = ", ".join(f"{name}:{count}" for name, count in sorted(by_class.items()))
    return f"tracks: {len(rows)} ({tracked} with id) | {class_summary}"


class SimpleTracker:
    def __init__(self, max_distance: float = 80.0, max_missing: int = 15) -> None:
        self.max_distance = float(max_distance)
        self.max_missing = int(max_missing)
        self.next_track_id = 1
        self.tracks: dict[int, TrackState] = {}

    def update(self, rows: list[dict]) -> list[dict]:
        for track in self.tracks.values():
            track.misses += 1

        assignments: list[tuple[float, int, int]] = []
        for det_index, row in enumerate(rows):
            center = np.array(row["center"], dtype=np.float32)
            bbox = row["bbox"]
            for track_id, track in self.tracks.items():
                if track.class_name != row["class_name"]:
                    continue

                track_center = np.array(track.center, dtype=np.float32)
                distance = float(np.linalg.norm(center - track_center))
                iou = bbox_iou(bbox, track.bbox)
                if distance <= self.max_distance or iou >= 0.1:
                    cost = distance - 25.0 * iou
                    assignments.append((cost, det_index, track_id))

        matched_detections = set()
        matched_tracks = set()
        for _, det_index, track_id in sorted(assignments, key=lambda item: item[0]):
            if det_index in matched_detections or track_id in matched_tracks:
                continue

            row = rows[det_index]
            track = self.tracks[track_id]
            track.center = row["center"]
            track.bbox = row["bbox"]
            track.misses = 0
            row["track_id"] = track_id
            matched_detections.add(det_index)
            matched_tracks.add(track_id)

        for det_index, row in enumerate(rows):
            if det_index in matched_detections:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                class_name=row["class_name"],
                center=row["center"],
                bbox=row["bbox"],
                misses=0,
            )
            row["track_id"] = track_id

        expired_ids = [
            track_id for track_id, track in self.tracks.items() if track.misses > self.max_missing
        ]
        for track_id in expired_ids:
            self.tracks.pop(track_id, None)

        return rows


def draw_track_annotations(
    frame: np.ndarray,
    rows: list[dict],
    track_history: dict[int, deque[tuple[int, int]]],
    history_length: int,
) -> None:
    require_cv2()
    for row in rows:
        center = row["center"]
        track_id = row["track_id"]
        depth_text = f"{row['depth_m']:.3f}m" if row["depth_m"] is not None else "depth n/a"
        label = (
            f"id {track_id} | {row['class_name']} {row['score']:.2f} | {depth_text}"
            if track_id is not None
            else f"{row['class_name']} {row['score']:.2f} | {depth_text}"
        )

        color = stable_color(track_id if track_id is not None else hash(row["class_name"]) % 2048)
        cv.circle(frame, center, 4, color, -1, lineType=cv.LINE_AA)
        cv.putText(
            frame,
            label,
            (center[0] + 6, max(20, center[1] - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            label,
            (center[0] + 6, max(20, center[1] - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )

        if track_id is None:
            continue

        history = track_history.setdefault(track_id, deque(maxlen=history_length))
        history.append(center)
        if len(history) < 2:
            continue

        history_points = np.array(history, dtype=np.int32).reshape(-1, 1, 2)
        cv.polylines(frame, [history_points], isClosed=False, color=color, thickness=2, lineType=cv.LINE_AA)


def render_masked_frame(color_image: np.ndarray, rows: list[dict]) -> np.ndarray:
    masked_frame = np.zeros_like(color_image)
    if not rows:
        return masked_frame

    combined_mask = np.zeros(color_image.shape[:2], dtype=bool)
    for row in rows:
        mask = row.get("mask")
        if mask is None or mask.shape != combined_mask.shape:
            continue
        combined_mask |= mask

    masked_frame[combined_mask] = color_image[combined_mask]
    return masked_frame


def save_masked_frame_png(output_dir: Path, color_image: np.ndarray, rows: list[dict]) -> Path:
    require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)

    masked_frame = render_masked_frame(color_image, rows)
    now = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    milliseconds = int((now % 1.0) * 1000.0)
    output_path = output_dir / f"mask_{timestamp}_{milliseconds:03d}.png"

    if not cv.imwrite(str(output_path), masked_frame):
        raise RuntimeError(f"Failed to save mask image to {output_path}.")
    return output_path


def create_video_writer(output_path: Path, frame_size: tuple[int, int], fps: float):
    require_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}.")
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOE segmentation + tracking on a live RealSense stream."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLOE weights path. Defaults to the first discovered local yoloe-26l-seg.pt.",
    )
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=None,
        help="Optional YOLOE open-vocabulary prompt, e.g. --prompt cup or --prompt cup,bottle.",
    )
    parser.add_argument("--serial", default=None, help="RealSense serial number. Defaults to the first detected camera.")
    parser.add_argument("--width", type=int, default=640, help="RealSense color/depth width.")
    parser.add_argument("--height", type=int, default=480, help="RealSense color/depth height.")
    parser.add_argument("--fps", type=int, default=30, help="RealSense FPS.")
    parser.add_argument("--warmup-frames", type=int, default=15, help="Frames to discard before live inference starts.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Max detections per frame.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional numeric class filter.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference when supported.")
    parser.add_argument(
        "--tracker-backend",
        choices=["custom", "ultralytics"],
        default="custom",
        help="Tracking backend. `custom` works offline without extra packages.",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--max-distance", type=float, default=80.0, help="Custom tracker max centroid distance.")
    parser.add_argument("--max-missing", type=int, default=15, help="Custom tracker tolerated missed frames.")
    parser.add_argument("--history", type=int, default=30, help="Per-track trail length.")
    parser.add_argument("--show-depth", action="store_true", help="Show an aligned depth preview window.")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="Maximum depth for visualization.")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save a segmentation-masked RGB stream to MP4.",
    )
    parser.add_argument(
        "--output",
        default="outputs/realsense_yoloe_tracking.mp4",
        help="Output path for --save-video.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="outputs/masks",
        help="Directory where `s` key mask PNG snapshots are saved.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List RealSense serials and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Load the model, print the resolved config, and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    require_cv2()
    resolved_model = resolve_model_path(args.model)
    prompt_classes = parse_prompt_classes(args.prompt)

    if prompt_classes:
        local_text_encoder = ensure_local_text_encoder_asset()
        if local_text_encoder is not None:
            os.environ.setdefault("ULTRALYTICS_TEXT_ENCODER", str(local_text_encoder))

    if args.list_cameras:
        serials = list_realsense_serials()
        if serials:
            print("\n".join(serials))
        else:
            print("No RealSense devices detected.")
        return

    model = build_yoloe_model(resolved_model)
    try:
        configure_yoloe_prompt(model, prompt_classes)
    except Exception as exc:
        if prompt_classes:
            raise RuntimeError(
                "Failed to configure YOLOE text prompts. If you are offline, make sure "
                "`mobileclip2_b.ts` is present locally or run without `--prompt`."
            ) from exc
        raise

    if args.dry_run:
        print(f"model={resolved_model}")
        print(f"prompt={prompt_classes or '[]'}")
        print(f"tracker_backend={args.tracker_backend}")
        print(f"tracker={args.tracker}")
        return

    serial = pick_serial(args.serial)
    camera = RealSenseCamera(
        serial=serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
    )
    window_name = "realsense_yoloe_tracking"
    depth_window_name = "realsense_depth"
    track_history: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.history))
    simple_tracker = SimpleTracker(max_distance=args.max_distance, max_missing=args.max_missing)
    writer = None

    if args.save_video:
        writer = create_video_writer(Path(args.output), (args.width, args.height), float(args.fps))

    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()

    try:
        while True:
            frame_bundle = camera.read()

            infer_start = time.perf_counter()
            if args.tracker_backend == "ultralytics":
                try:
                    result = model.track(
                        source=frame_bundle.color_image,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        max_det=args.max_det,
                        device=args.device,
                        classes=args.classes,
                        half=args.half,
                        retina_masks=True,
                        persist=True,
                        tracker=args.tracker,
                        verbose=False,
                    )[0]
                except ModuleNotFoundError as exc:
                    if exc.name == "lap":
                        raise RuntimeError(
                            "Ultralytics tracking backend requires `lap`. Use the default "
                            "`--tracker-backend custom` or install `lap>=0.5.12`."
                        ) from exc
                    raise
            else:
                result = model.predict(
                    source=frame_bundle.color_image,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=args.device,
                    classes=args.classes,
                    half=args.half,
                    retina_masks=True,
                    verbose=False,
                )[0]
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            annotated = result.plot()
            rows = extract_track_rows(result, frame_bundle.depth_image_m)
            if args.tracker_backend == "custom":
                rows = simple_tracker.update(rows)
            draw_track_annotations(annotated, rows, track_history, history_length=args.history)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else (0.9 * smoothed_fps + 0.1 * instant_fps)
            last_loop_time = now

            header_lines = [
                f"model: {resolved_model}",
                f"serial: {frame_bundle.serial}",
                f"tracker: {args.tracker_backend}",
                f"fps: {smoothed_fps:.1f} | infer: {infer_ms:.1f} ms",
                summarize_tracks(rows),
                "keys: s save mask | q / ESC quit",
            ]
            if prompt_classes:
                header_lines.insert(1, f"prompt: {', '.join(prompt_classes)}")

            draw_text_block(annotated, header_lines)
            cv.imshow(window_name, annotated)

            if writer is not None:
                writer.write(render_masked_frame(frame_bundle.color_image, rows))

            if args.show_depth:
                cv.imshow(depth_window_name, render_depth(frame_bundle.depth_image_m, args.depth_max_m))

            key = cv.waitKey(1) & 0xFF
            if key == ord("s"):
                saved_path = save_masked_frame_png(
                    Path(args.snapshot_dir),
                    frame_bundle.color_image,
                    rows,
                )
                print(f"Saved mask image: {saved_path}")
                continue
            if key in (27, ord("q")):
                break
    finally:
        camera.stop()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
