"""Offline replay of saved 3D debug RGB/depth recordings.

This script re-runs the perception side of robot_control_rtde_fitting_final.py
from raw frames stored by Debug3DRecorder. It intentionally does not start
RealSense cameras or send any RTDE robot commands.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from utils.debug_3d_recorder import Debug3DRecorder, load_debug_3d_npz
from utils.realsense_stream import FrameBundle


DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
REQUIRED_RAW_KEYS = (
    "cam0_color_image",
    "cam0_depth_image_m",
    "cam0_intrinsics",
    "cam1_color_image",
    "cam1_depth_image_m",
    "cam1_intrinsics",
)


@dataclass(frozen=True)
class OfflineReplayResult:
    input_path: Path
    output_path: Path
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the live perception pipeline on raw color/depth frames saved "
            "inside Debug3DRecorder .npz files."
        )
    )
    parser.add_argument("input", help="Input .npz file or directory containing .npz recordings.")
    parser.add_argument("--output-dir", default="output/data_inference", help="Directory for replayed 3D debug .npz files.")
    parser.add_argument("--prefix", default="offline_inference", help="Output filename prefix.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config.")

    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="Model name or local weights path.")
    parser.add_argument("--prompt", nargs="*", default=None, help="YOLOE text prompt classes.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional class id filter.")
    parser.add_argument(
        "--select-mode",
        choices=["all_instances", "highest_score", "class_filter"],
        default="all_instances",
        help="Instance selection policy for object point cloud generation.",
    )
    parser.add_argument("--select-class", nargs="*", default=None, help="Optional class-name filter.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference when supported.")

    parser.add_argument("--fps", type=int, default=30, help="Timestamp fallback FPS for recordings missing timestamps.")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame index to process.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive source frame index stop.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth source frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of selected frames to process.")
    parser.add_argument("--save-image", action="store_true", help="Copy raw color/depth frames into the output .npz.")
    parser.add_argument("--debug-3d-max-object-points", type=int, default=8000)
    parser.add_argument("--debug-3d-max-template-points", type=int, default=8000)
    parser.add_argument(
        "--no-debug-3d-template-axes",
        dest="debug_3d_template_axes",
        action="store_false",
        help="Do not store fitted template local x/y/z axes in replayed 3D debug recordings.",
    )
    parser.set_defaults(debug_3d_template_axes=True)
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failed recordings when input is a directory.")
    return parser.parse_args()


def find_input_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Input file must be a .npz recording: {path}")
        return [path]
    if path.is_dir():
        files = sorted(child for child in path.glob("*.npz") if child.is_file())
        if not files:
            raise ValueError(f"No .npz recordings found in directory: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def validate_raw_recording(data: dict[str, Any], input_path: str | Path) -> int:
    missing = [key for key in REQUIRED_RAW_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"{input_path} does not contain raw saved frames. "
            "Record with --3d-debug --save-image first. Missing keys: "
            + ", ".join(missing)
        )

    frame_count = int(np.asarray(data.get("frame_count", len(data["cam0_color_image"]))).reshape(()))
    for key in REQUIRED_RAW_KEYS:
        if len(data[key]) != frame_count:
            raise ValueError(
                f"{input_path} has inconsistent frame count for {key}: "
                f"expected {frame_count}, got {len(data[key])}"
            )
    return frame_count


def select_frame_indices(
    frame_count: int,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[int]:
    start = max(int(start_frame), 0)
    stop = int(frame_count) if end_frame is None else min(max(int(end_frame), 0), int(frame_count))
    step = max(int(stride), 1)
    if start >= stop:
        return []
    indices = list(range(start, stop, step))
    if max_frames is not None:
        indices = indices[: max(int(max_frames), 0)]
    return indices


def intrinsics_vector_to_dict(value: Any) -> dict[str, float]:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(vector) < 4:
        raise ValueError(f"Intrinsics vector must contain fx, fy, cx, cy; got shape {vector.shape}")
    return {
        "fx": float(vector[0]),
        "fy": float(vector[1]),
        "cx": float(vector[2]),
        "cy": float(vector[3]),
    }


def optional_string_array_value(data: dict[str, Any], key: str, index: int, default: str = "") -> str:
    values = data.get(key)
    if values is None or len(values) <= index:
        return default
    value = values[index]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value)
    return default if text == "nan" else text


def optional_float_array_value(data: dict[str, Any], key: str, index: int, default: float = np.nan) -> float:
    values = data.get(key)
    if values is None or len(values) <= index:
        return float(default)
    try:
        return float(values[index])
    except Exception:
        return float(default)


def optional_vec(data: dict[str, Any], key: str, index: int, length: int) -> tuple[float, ...] | None:
    values = data.get(key)
    if values is None or len(values) <= index:
        return None
    vector = np.asarray(values[index], dtype=np.float32).reshape(-1)
    if len(vector) < length:
        return None
    vector = vector[:length]
    if not np.isfinite(vector).all():
        return None
    return tuple(float(v) for v in vector)


def build_offline_snapshot(data: dict[str, Any], source_index: int, replay_index: int, fps: int) -> SimpleNamespace:
    cam0 = FrameBundle(
        color_image=np.asarray(data["cam0_color_image"][source_index], dtype=np.uint8),
        depth_image_m=np.asarray(data["cam0_depth_image_m"][source_index], dtype=np.float32),
        intrinsics=intrinsics_vector_to_dict(data["cam0_intrinsics"][source_index]),
        timestamp_ms=optional_float_array_value(data, "cam0_timestamp_ms", source_index, replay_index * 1000.0 / max(fps, 1)),
        serial=optional_string_array_value(data, "cam0_serial", source_index, "offline_cam0"),
    )
    cam1 = FrameBundle(
        color_image=np.asarray(data["cam1_color_image"][source_index], dtype=np.uint8),
        depth_image_m=np.asarray(data["cam1_depth_image_m"][source_index], dtype=np.float32),
        intrinsics=intrinsics_vector_to_dict(data["cam1_intrinsics"][source_index]),
        timestamp_ms=optional_float_array_value(data, "cam1_timestamp_ms", source_index, replay_index * 1000.0 / max(fps, 1)),
        serial=optional_string_array_value(data, "cam1_serial", source_index, "offline_cam1"),
    )
    return SimpleNamespace(
        pair_index=int(source_index),
        cam0=cam0,
        cam1=cam1,
        timestamp_delta_ms=optional_float_array_value(data, "timestamp_delta_ms", source_index, 0.0),
        within_sync_tolerance=bool(data.get("within_sync_tolerance", np.ones(len(data["cam0_color_image"]), dtype=bool))[source_index]),
    )


def infer_input_size(data: dict[str, Any]) -> tuple[int, int]:
    first_color = np.asarray(data["cam0_color_image"][0])
    if first_color.ndim < 2:
        raise ValueError(f"cam0_color_image has invalid shape: {first_color.shape}")
    height, width = first_color.shape[:2]
    return int(width), int(height)


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


def _load_live_module():
    _ensure_repo_root_on_path()
    import robot_control_rtde_fitting_final as live

    return live


def prepare_runtime_args(args: argparse.Namespace, data: dict[str, Any]) -> argparse.Namespace:
    runtime_args = copy.copy(args)
    width, height = infer_input_size(data)
    runtime_args.width = width
    runtime_args.height = height

    # Robot/control fields are not used for offline inference, but the live
    # config-default helper expects them to exist.
    runtime_args.robot_ip = None
    runtime_args.control_hz = None
    runtime_args.follow_z = None
    runtime_args.workspace_x = None
    runtime_args.workspace_y = None
    runtime_args.workspace_z = None
    runtime_args.enable_follow = False
    runtime_args.move_to_base = False
    runtime_args.open_gripper = False
    runtime_args.verbose_robot = False
    runtime_args.min_valid_count = 3
    runtime_args.target_timeout_s = 0.5
    runtime_args.position_tolerance_m = 0.01
    runtime_args.move_timeout_s = 10.0
    runtime_args.gripper_close_timeout_s = 2.0
    runtime_args.gripper_release_dwell_s = 0.5
    runtime_args.follow_handoff_timeout_s = 5.0
    runtime_args.enable_target_prediction = True
    runtime_args.prediction_max_horizon_s = 0.25
    runtime_args.prediction_process_noise_mm_s2 = 800.0
    runtime_args.prediction_measurement_noise_mm = 25.0
    runtime_args.prediction_max_xy_speed_mm_s = 200.0
    runtime_args.prediction_reinit_jump_mm = 120.0
    runtime_args.show_depth = False
    runtime_args.depth_max_m = 1.5
    runtime_args.debug_3d = True
    runtime_args.debug_3d_dir = str(args.output_dir)
    runtime_args.disable_debug_3d_recording = False
    runtime_args.profile_runtime = False
    runtime_args.profile_dir = "output/runtime_profile"
    runtime_args.profile_sample_interval_s = 1.0
    runtime_args.profile_print_every_s = 0.0
    return runtime_args


def close_pipeline(pipeline: dict[str, Any]) -> None:
    for key in ("hand_worker_cam0", "hand_worker_cam1"):
        worker = pipeline.get(key)
        if worker is not None and hasattr(worker, "close"):
            try:
                worker.close()
            except Exception:
                pass
    sensor_hub = pipeline.get("sensor_hub")
    if sensor_hub is not None and hasattr(sensor_hub, "stop"):
        try:
            sensor_hub.stop()
        except Exception:
            pass


def run_replay(input_path: str | Path, args: argparse.Namespace) -> OfflineReplayResult:
    input_path = Path(input_path)
    data = load_debug_3d_npz(input_path)
    frame_count = validate_raw_recording(data, input_path)
    frame_indices = select_frame_indices(
        frame_count,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    if not frame_indices:
        raise ValueError(f"No frames selected from {input_path}")

    live = _load_live_module()
    runtime_args = prepare_runtime_args(args, data)
    config = live.load_yaml_config(runtime_args.config)
    runtime_args = live.apply_config_defaults(runtime_args, config)
    pipeline = live.build_dual_perception_pipeline(runtime_args)
    recorder = Debug3DRecorder(
        output_dir=args.output_dir,
        enabled=True,
        save_images=bool(args.save_image),
        max_object_points=args.debug_3d_max_object_points,
        max_template_points=args.debug_3d_max_template_points,
        record_template_axes=bool(args.debug_3d_template_axes),
    )

    previous_hand_approach = False
    replay_start_perf = time.perf_counter()
    try:
        for replay_index, source_index in enumerate(frame_indices):
            current_time = optional_float_array_value(data, "timestamp_unix_s", source_index, time.time())
            loop_perf = optional_float_array_value(data, "timestamp_perf_s", source_index, time.perf_counter())
            record_elapsed_s = optional_float_array_value(
                data,
                "record_elapsed_s",
                source_index,
                time.perf_counter() - replay_start_perf,
            )
            if not np.isfinite(record_elapsed_s):
                record_elapsed_s = time.perf_counter() - replay_start_perf
            task_epoch = int(optional_float_array_value(data, "task_epoch", source_index, 0.0))
            snapshot = build_offline_snapshot(data, source_index, replay_index, runtime_args.fps)

            object_cam0 = pipeline["object_worker_cam0"].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            object_cam1 = pipeline["object_worker_cam1"].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
            hand_cam0 = pipeline["hand_worker_cam0"].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            hand_cam1 = pipeline["hand_worker_cam1"].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)

            selected_hand = pipeline["hand_selector"].process_states(hand_cam0, hand_cam1)
            merged_object = pipeline["object_merger"].process_states(
                object_cam0,
                object_cam1,
                hand_approach_detected=previous_hand_approach,
            )
            shape_fitting_state = pipeline["shape_fitting_tracker"].process(merged_object)
            object_debug_cam0 = getattr(pipeline["object_worker_cam0"], "last_debug", None)
            cam0_mask = None if object_debug_cam0 is None else getattr(object_debug_cam0, "combined_mask", None)
            pipeline["fill_level_estimator"].estimate_fill_level_from_cam0(
                color_image_bgr=snapshot.cam0.color_image,
                depth_image_m=snapshot.cam0.depth_image_m,
                intrinsics=snapshot.cam0.intrinsics,
                container_mask=cam0_mask,
                camera_to_base=pipeline["transform_chain"].t_base_cam0,
                label=object_cam0.label,
            )
            fitted_merged_object = live.build_fitted_merged_object(merged_object, shape_fitting_state)
            fusion_state = pipeline["fusion"].process_states(
                fitted_merged_object,
                selected_hand,
                now_timestamp=current_time,
            )
            grasp_target = pipeline["grasp_planner"].process_states(
                fitted_merged_object,
                selected_hand,
                fusion_state,
            )
            previous_hand_approach = bool(
                fusion_state.hand_approach_detected or fusion_state.hand_approach_latched
            )

            measured_object_point_base = live.choose_point(
                fusion_state.filtered_object_centroid_base,
                fitted_merged_object.centroid_base,
            )
            measured_grasp_point_base = grasp_target.target_position_base if grasp_target.valid else None
            measured_grasp_point_base = live.offset_point_base_mm(
                measured_grasp_point_base,
                y_mm=live.GRASP_POINT_Y_OFFSET_MM,
            )

            fallback_state = pipeline["hand_relative_fallback"].process(
                measured_object_position_base=measured_object_point_base,
                measured_grasp_position_base=measured_grasp_point_base,
                selected_hand=selected_hand,
                fusion_state=fusion_state,
                motion_triggered=False,
                now_timestamp=current_time,
                frame_id=snapshot.pair_index,
                record_elapsed_s=record_elapsed_s,
            )

            object_point_base = measured_object_point_base
            grasp_point_base = measured_grasp_point_base
            measurement_source = "measured"
            if measured_object_point_base is None and fallback_state.valid:
                object_point_base = fallback_state.object_position_base
                grasp_point_base = fallback_state.grasp_position_base
                measurement_source = "hand_fallback"

            live.append_debug_3d_frame(
                recorder,
                snapshot=snapshot,
                current_time=current_time,
                loop_perf=loop_perf,
                record_elapsed_s=record_elapsed_s,
                current_task_epoch=task_epoch,
                pipeline=pipeline,
                selected_hand=selected_hand,
                merged_object=merged_object,
                shape_fitting_state=shape_fitting_state,
                object_point_base=object_point_base,
                grasp_point_base=grasp_point_base,
                eef_pose_base=optional_vec(data, "eef_pose_base", source_index, 6),
                measurement_source=measurement_source,
            )

            if replay_index == 0 or (replay_index + 1) % 10 == 0 or replay_index == len(frame_indices) - 1:
                print(
                    f"[INFO] {input_path.name}: processed {replay_index + 1}/{len(frame_indices)} "
                    f"(source_frame={source_index})",
                    flush=True,
                )

        output_path = recorder.save(prefix=f"{args.prefix}_{input_path.stem}")
        return OfflineReplayResult(
            input_path=input_path,
            output_path=output_path,
            frame_count=len(frame_indices),
        )
    finally:
        recorder.close()
        close_pipeline(pipeline)


def run_all(input_files: Iterable[Path], args: argparse.Namespace) -> list[OfflineReplayResult]:
    results: list[OfflineReplayResult] = []
    failures: list[tuple[Path, Exception]] = []
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for input_path in input_files:
        print(f"[INFO] Replaying {input_path}", flush=True)
        try:
            results.append(run_replay(input_path, args))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((input_path, exc))
            print(f"[WARN] Failed to replay {input_path}: {exc}", flush=True)

    if failures:
        print(f"[WARN] {len(failures)} recording(s) failed.", flush=True)
    return results


def main() -> int:
    args = parse_args()
    input_files = find_input_files(args.input)
    results = run_all(input_files, args)
    for result in results:
        print(
            f"[INFO] Saved offline inference: {result.output_path} "
            f"({result.frame_count} frame(s) from {result.input_path.name})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
