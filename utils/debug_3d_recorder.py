"""3D debug recording for handover failure analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 7
IMAGE_ARRAY_KEYS = (
    "cam0_color_image",
    "cam0_depth_image_m",
    "cam1_color_image",
    "cam1_depth_image_m",
)


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _none_if_missing(value: Any) -> Any:
    return None if value is None else value


def format_record_clock(elapsed_s: float | None) -> str:
    if elapsed_s is None or not np.isfinite(float(elapsed_s)):
        return ""
    elapsed_s = max(float(elapsed_s), 0.0)
    minutes = int(elapsed_s // 60.0)
    seconds = int(elapsed_s % 60.0)
    tenths = int((elapsed_s - int(elapsed_s)) * 10.0)
    return f"REC {minutes:02d}:{seconds:02d}.{tenths:d}"


def _vec(value: Any, length: int, *, dtype=np.float32) -> np.ndarray:
    output = np.full((int(length),), np.nan, dtype=dtype)
    if value is None:
        return output
    array = np.array(value, dtype=dtype, copy=True).reshape(-1)
    count = min(len(array), int(length))
    if count > 0:
        output[:count] = array[:count]
    return output


def _matrix(value: Any, shape: tuple[int, int], *, dtype=np.float32) -> np.ndarray:
    output = np.full(shape, np.nan, dtype=dtype)
    if value is None:
        return output
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape == shape:
        output[...] = array
    return output


def _points(value: Any, *, max_points: int) -> np.ndarray:
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    points = np.array(value, dtype=np.float32, copy=True).reshape((-1, 3))
    if max_points > 0 and len(points) > int(max_points):
        sample_indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
        points = points[sample_indices]
    return np.ascontiguousarray(points, dtype=np.float32).copy()


def _hand_debug_points(hand_debug: Any) -> np.ndarray:
    if hand_debug is None:
        return np.full((21, 3), np.nan, dtype=np.float32)
    points = getattr(hand_debug, "points_3d_base", None)
    output = np.full((21, 3), np.nan, dtype=np.float32)
    if points is None:
        return output
    points = np.array(points, dtype=np.float32, copy=True).reshape((-1, 3))
    count = min(len(points), 21)
    output[:count] = points[:count]
    return output


def _hand_debug_mask(hand_debug: Any) -> np.ndarray:
    if hand_debug is None:
        return np.zeros((21,), dtype=bool)
    mask = getattr(hand_debug, "valid_mask", None)
    output = np.zeros((21,), dtype=bool)
    if mask is None:
        return output
    mask = np.array(mask, dtype=bool, copy=True).reshape(-1)
    count = min(len(mask), 21)
    output[:count] = mask[:count]
    return output


def _hand_debug_rotation(hand_debug: Any) -> np.ndarray:
    if hand_debug is None:
        return np.full((3, 3), np.nan, dtype=np.float32)
    palm_pose = getattr(hand_debug, "palm_pose_base", None)
    if not isinstance(palm_pose, dict):
        return np.full((3, 3), np.nan, dtype=np.float32)
    return _matrix(palm_pose.get("rotation_matrix"), (3, 3), dtype=np.float32)


def _intrinsics_vector(frame_bundle: Any) -> np.ndarray:
    intrinsics = getattr(frame_bundle, "intrinsics", None) or {}
    return np.asarray(
        [
            float(intrinsics.get("fx", np.nan)),
            float(intrinsics.get("fy", np.nan)),
            float(intrinsics.get("cx", np.nan)),
            float(intrinsics.get("cy", np.nan)),
        ],
        dtype=np.float32,
    )


def _hand_debug_valid_count(hand_debug: Any) -> int:
    if hand_debug is None:
        return 0
    return int(np.count_nonzero(_hand_debug_mask(hand_debug)))


def _hand_debug_detected(hand_debug: Any) -> bool:
    if hand_debug is None:
        return False
    palm_pose = getattr(hand_debug, "palm_pose_base", None)
    if isinstance(palm_pose, dict):
        return bool(palm_pose.get("valid", False))
    return bool(_hand_debug_valid_count(hand_debug) > 0)


def _hand_debug_confidence(hand_debug: Any) -> float:
    if hand_debug is None:
        return np.nan
    palm_pose = getattr(hand_debug, "palm_pose_base", None)
    if isinstance(palm_pose, dict):
        return float(palm_pose.get("quality", np.nan))
    return float(getattr(hand_debug, "confidence", np.nan))


def _hand_debug_reason(hand_debug: Any) -> str:
    if hand_debug is None:
        return "no_debug"
    palm_pose = getattr(hand_debug, "palm_pose_base", None)
    if isinstance(palm_pose, dict) and bool(palm_pose.get("valid", False)):
        return "palm_pose_valid"
    valid_count = _hand_debug_valid_count(hand_debug)
    if valid_count <= 0:
        return "no_valid_depth_landmarks"
    return f"palm_pose_invalid_valid_landmarks_{valid_count}"


def _selector_field(selector_debug: Any, field_name: str, default: Any = "") -> Any:
    if selector_debug is None:
        return default
    return getattr(selector_debug, field_name, default)


def _safe_load_array(path: Path, *, dtype: Any) -> np.ndarray:
    return np.load(path).astype(dtype, copy=False)


def _optional_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        output = float(value)
    except Exception:
        return np.nan
    return output if np.isfinite(output) else np.nan


def _normalize_tactile_snapshot(tactile_snapshot: Any, *, frame_timestamp_perf_s: float) -> dict[str, Any]:
    if tactile_snapshot is None:
        return {
            "tactile_enabled": False,
            "tactile_num_mags": 0,
            "tactile_values": np.empty((0, 3), dtype=np.float32),
            "tactile_mag_norms": np.empty((0,), dtype=np.float32),
            "tactile_total_norm": np.nan,
            "tactile_release_ref_norm": np.nan,
            "tactile_release_delta_norm": np.nan,
            "tactile_status": "disabled",
            "tactile_error": "",
            "tactile_timestamp_perf_s": np.nan,
            "tactile_timestamp_unix_s": np.nan,
            "tactile_frame_delta_ms": np.nan,
        }

    values = np.asarray(tactile_snapshot.get("values", []), dtype=np.float32).reshape(-1)
    num_mags = int(tactile_snapshot.get("num_mags", 0) or 0)
    if num_mags <= 0 and values.size > 0:
        num_mags = max(values.size // 3, 1)

    expected_values = max(num_mags, 0) * 3
    if expected_values > 0:
        if values.size < expected_values:
            padded = np.full((expected_values,), np.nan, dtype=np.float32)
            padded[: values.size] = values
            values = padded
        values = values[:expected_values].reshape((num_mags, 3)).copy()
    else:
        values = np.empty((0, 3), dtype=np.float32)

    if len(values) > 0:
        mag_norms = np.linalg.norm(values, axis=1).astype(np.float32, copy=False)
    else:
        mag_norms = np.empty((0,), dtype=np.float32)

    total_norm = _optional_float(tactile_snapshot.get("total_norm"))
    if not np.isfinite(total_norm) and len(values) > 0:
        total_norm = float(np.linalg.norm(values.reshape(-1)))

    tactile_timestamp_perf_s = _optional_float(
        tactile_snapshot.get("timestamp_perf_s", tactile_snapshot.get("sample_perf_s"))
    )
    tactile_timestamp_unix_s = _optional_float(
        tactile_snapshot.get("timestamp_unix_s", tactile_snapshot.get("sample_unix_s"))
    )
    frame_delta_ms = np.nan
    if np.isfinite(tactile_timestamp_perf_s) and np.isfinite(float(frame_timestamp_perf_s)):
        frame_delta_ms = (float(tactile_timestamp_perf_s) - float(frame_timestamp_perf_s)) * 1000.0

    error = tactile_snapshot.get("error")
    return {
        "tactile_enabled": True,
        "tactile_num_mags": int(num_mags),
        "tactile_values": np.ascontiguousarray(values, dtype=np.float32),
        "tactile_mag_norms": np.ascontiguousarray(mag_norms, dtype=np.float32),
        "tactile_total_norm": float(total_norm),
        "tactile_release_ref_norm": _optional_float(tactile_snapshot.get("release_ref_norm")),
        "tactile_release_delta_norm": _optional_float(tactile_snapshot.get("release_delta_norm")),
        "tactile_status": str(tactile_snapshot.get("status", "")),
        "tactile_error": "" if error is None else str(error),
        "tactile_timestamp_perf_s": float(tactile_timestamp_perf_s),
        "tactile_timestamp_unix_s": float(tactile_timestamp_unix_s),
        "tactile_frame_delta_ms": float(frame_delta_ms),
    }


@dataclass
class Debug3DRecorder:
    """Recorder that saves one compressed 3D debug session on demand."""

    output_dir: str | Path = "output/debug_3d"
    enabled: bool = True
    save_images: bool = False
    max_object_points: int = 8000
    max_template_points: int = 8000
    record_template_axes: bool = True
    frames: list[dict[str, Any]] = field(default_factory=list)
    session_index: int = 0
    _image_spool_dir: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.save_images and self.enabled:
            self._reset_image_spool()

    def clear(self) -> None:
        self.frames.clear()
        self._discard_image_spool()
        self.session_index += 1
        if self.save_images and self.enabled:
            self._reset_image_spool()

    def close(self) -> None:
        self.frames.clear()
        self._discard_image_spool()

    def _discard_image_spool(self) -> None:
        if self._image_spool_dir is not None and self._image_spool_dir.exists():
            shutil.rmtree(self._image_spool_dir, ignore_errors=True)
        self._image_spool_dir = None

    def _reset_image_spool(self) -> None:
        self._discard_image_spool()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._image_spool_dir = Path(
            tempfile.mkdtemp(prefix=f".debug_3d_images_session{self.session_index:03d}_", dir=str(output_dir))
        )

    def append_frame(
        self,
        *,
        frame_index: int,
        timestamp_unix_s: float,
        timestamp_perf_s: float,
        record_elapsed_s: float | None,
        task_epoch: int,
        selected_hand: Any,
        hand_debug_cam0: Any,
        hand_debug_cam1: Any,
        raw_merged_object: Any,
        shape_fitting_state: Any,
        object_point_base: Any,
        grasp_point_base: Any,
        eef_pose_base: Any,
        measurement_source: str,
        hand_selector_debug: Any = None,
        tactile_snapshot: Any = None,
        snapshot: Any = None,
    ) -> None:
        if not self.enabled:
            return

        raw_points = _points(
            getattr(raw_merged_object, "merged_points_base", None),
            max_points=self.max_object_points,
        )
        fitted_points = _points(
            getattr(shape_fitting_state, "fitted_points_base", None),
            max_points=self.max_template_points,
        )
        tactile_fields = _normalize_tactile_snapshot(
            tactile_snapshot,
            frame_timestamp_perf_s=float(timestamp_perf_s),
        )

        frame = {
            "frame_index": int(frame_index),
            "timestamp_unix_s": float(timestamp_unix_s),
            "timestamp_perf_s": float(timestamp_perf_s),
            "record_elapsed_s": np.nan if record_elapsed_s is None else float(record_elapsed_s),
            "record_clock_text": format_record_clock(record_elapsed_s),
            "task_epoch": int(task_epoch),
            "measurement_source": str(measurement_source or "none"),
            "object_label": _none_if_missing(getattr(raw_merged_object, "label", None)),
            "object_valid": bool(getattr(raw_merged_object, "valid", False)),
            "object_point_count": int(getattr(raw_merged_object, "merged_point_count", len(raw_points))),
            "object_points_base": raw_points,
            "object_centroid_base": _vec(getattr(raw_merged_object, "centroid_base", None), 3),
            "fitted_label": _none_if_missing(getattr(shape_fitting_state, "label", None)),
            "template_id": _none_if_missing(getattr(shape_fitting_state, "template_id", None)),
            "shape_fitting_valid": bool(getattr(shape_fitting_state, "valid", False)),
            "shape_fitting_initialized": bool(getattr(shape_fitting_state, "initialized", False)),
            "shape_fitting_reason": str(getattr(shape_fitting_state, "reason", "none")),
            "shape_fitting_scale": np.nan
            if getattr(shape_fitting_state, "scale", None) is None
            else float(getattr(shape_fitting_state, "scale")),
            "shape_fitting_scale_x": _vec(getattr(shape_fitting_state, "scale_xyz", None), 3)[0],
            "shape_fitting_scale_y": _vec(getattr(shape_fitting_state, "scale_xyz", None), 3)[1],
            "shape_fitting_scale_z": _vec(getattr(shape_fitting_state, "scale_xyz", None), 3)[2],
            "shape_fitting_scale_mode": str(getattr(shape_fitting_state, "scale_mode", "uniform")),
            "shape_fitting_silhouette_enabled": bool(getattr(shape_fitting_state, "silhouette_enabled", False)),
            "shape_fitting_silhouette_reason": str(getattr(shape_fitting_state, "silhouette_reason", "disabled")),
            "shape_fitting_silhouette_candidate_count": int(
                getattr(shape_fitting_state, "silhouette_candidate_count", 0)
            ),
            "shape_fitting_silhouette_valid_camera_count": int(
                getattr(shape_fitting_state, "silhouette_valid_camera_count", 0)
            ),
            "shape_fitting_silhouette_best_iou_cam0": np.nan
            if getattr(shape_fitting_state, "silhouette_best_iou_cam0", None) is None
            else float(getattr(shape_fitting_state, "silhouette_best_iou_cam0")),
            "shape_fitting_silhouette_best_iou_cam1": np.nan
            if getattr(shape_fitting_state, "silhouette_best_iou_cam1", None) is None
            else float(getattr(shape_fitting_state, "silhouette_best_iou_cam1")),
            "shape_fitting_silhouette_loss": np.nan
            if getattr(shape_fitting_state, "silhouette_loss", None) is None
            else float(getattr(shape_fitting_state, "silhouette_loss")),
            "shape_fitting_silhouette_outside_loss": np.nan
            if getattr(shape_fitting_state, "silhouette_outside_loss", None) is None
            else float(getattr(shape_fitting_state, "silhouette_outside_loss")),
            "shape_fitting_silhouette_robust_3d_loss": np.nan
            if getattr(shape_fitting_state, "robust_3d_loss", None) is None
            else float(getattr(shape_fitting_state, "robust_3d_loss")),
            "shape_fitting_silhouette_scale_prior_loss": np.nan
            if getattr(shape_fitting_state, "scale_prior_loss", None) is None
            else float(getattr(shape_fitting_state, "scale_prior_loss")),
            "shape_fitting_silhouette_temporal_scale_loss": np.nan
            if getattr(shape_fitting_state, "temporal_scale_loss", None) is None
            else float(getattr(shape_fitting_state, "temporal_scale_loss")),
            "shape_fitting_pre_rerank_scale": np.nan
            if getattr(shape_fitting_state, "pre_rerank_scale", None) is None
            else float(getattr(shape_fitting_state, "pre_rerank_scale")),
            "shape_fitting_post_rerank_scale": np.nan
            if getattr(shape_fitting_state, "post_rerank_scale", None) is None
            else float(getattr(shape_fitting_state, "post_rerank_scale")),
            "shape_fitting_pre_rerank_scale_xyz": _vec(getattr(shape_fitting_state, "pre_rerank_scale_xyz", None), 3),
            "shape_fitting_post_rerank_scale_xyz": _vec(getattr(shape_fitting_state, "post_rerank_scale_xyz", None), 3),
            "shape_fitting_pre_rerank_template_extent": _vec(
                getattr(shape_fitting_state, "pre_rerank_template_extent", None), 3
            ),
            "shape_fitting_post_rerank_template_extent": _vec(
                getattr(shape_fitting_state, "post_rerank_template_extent", None), 3
            ),
            "shape_fitting_rerank_changed_candidate": bool(
                getattr(shape_fitting_state, "rerank_changed_candidate", False)
            ),
            "template_points_base": fitted_points,
            "template_centroid_base": _vec(getattr(shape_fitting_state, "centroid_base", None), 3),
            "cam0_hand_points_base": _hand_debug_points(hand_debug_cam0),
            "cam0_hand_valid_mask": _hand_debug_mask(hand_debug_cam0),
            "cam0_palm_rotation_base": _hand_debug_rotation(hand_debug_cam0),
            "cam0_hand_detected": _hand_debug_detected(hand_debug_cam0),
            "cam0_hand_valid_count": _hand_debug_valid_count(hand_debug_cam0),
            "cam0_hand_confidence": _hand_debug_confidence(hand_debug_cam0),
            "cam0_hand_reason": _hand_debug_reason(hand_debug_cam0),
            "cam1_hand_points_base": _hand_debug_points(hand_debug_cam1),
            "cam1_hand_valid_mask": _hand_debug_mask(hand_debug_cam1),
            "cam1_palm_rotation_base": _hand_debug_rotation(hand_debug_cam1),
            "cam1_hand_detected": _hand_debug_detected(hand_debug_cam1),
            "cam1_hand_valid_count": _hand_debug_valid_count(hand_debug_cam1),
            "cam1_hand_confidence": _hand_debug_confidence(hand_debug_cam1),
            "cam1_hand_reason": _hand_debug_reason(hand_debug_cam1),
            "selected_hand_valid": bool(getattr(selected_hand, "valid", False)),
            "selected_hand_camera": -1
            if getattr(selected_hand, "selected_camera", None) is None
            else int(getattr(selected_hand, "selected_camera")),
            "selected_handedness": _none_if_missing(getattr(selected_hand, "handedness", None)),
            "selected_palm_center_base": _vec(getattr(selected_hand, "palm_center_base", None), 3),
            #"selected_palm_normal_base": _vec(getattr(selected_hand, "palm_normal_base", None), 3),
            "selected_wrist_base": _vec(getattr(selected_hand, "wrist_base", None), 3),
            "selected_hand_confidence": float(getattr(selected_hand, "confidence", np.nan)),
            "hand_selection_reason": str(_selector_field(hand_selector_debug, "selection_reason", "")),
            "hand_selector_cam0_reject_reason": str(_selector_field(hand_selector_debug, "cam0_reject_reason", "")),
            "hand_selector_cam1_reject_reason": str(_selector_field(hand_selector_debug, "cam1_reject_reason", "")),
            "hand_selector_chosen_camera": -1
            if _selector_field(hand_selector_debug, "chosen_camera", None) is None
            else int(_selector_field(hand_selector_debug, "chosen_camera")),
            "eef_pose_base": _vec(eef_pose_base, 6),
            "object_point_base": _vec(object_point_base, 3),
            "grasp_point_base": _vec(grasp_point_base, 3),
            **tactile_fields,
        }
        if self.record_template_axes:
            frame["template_axes_base"] = _matrix(getattr(shape_fitting_state, "template_axes_base", None), (3, 3))
        self.frames.append(frame)
        if self.save_images:
            if snapshot is None:
                raise ValueError("snapshot is required when save_images=True")
            self._append_snapshot_images(self.frames[-1], snapshot)

    def _append_snapshot_images(self, frame: dict[str, Any], snapshot: Any) -> None:
        if self._image_spool_dir is None:
            self._reset_image_spool()
        assert self._image_spool_dir is not None
        spool_index = len(self.frames) - 1
        for camera_name in ("cam0", "cam1"):
            frame_bundle = getattr(snapshot, camera_name)
            color = np.array(frame_bundle.color_image, dtype=np.uint8, copy=True)
            depth = np.array(frame_bundle.depth_image_m, dtype=np.float32, copy=True)
            color_path = self._image_spool_dir / f"frame{spool_index:06d}_{camera_name}_color.npy"
            depth_path = self._image_spool_dir / f"frame{spool_index:06d}_{camera_name}_depth.npy"
            np.save(color_path, np.ascontiguousarray(color))
            np.save(depth_path, np.ascontiguousarray(depth))
            frame[f"{camera_name}_color_image"] = color_path
            frame[f"{camera_name}_depth_image_m"] = depth_path
            frame[f"{camera_name}_intrinsics"] = _intrinsics_vector(frame_bundle)
            frame[f"{camera_name}_timestamp_ms"] = float(getattr(frame_bundle, "timestamp_ms", np.nan))
            frame[f"{camera_name}_serial"] = _none_if_missing(getattr(frame_bundle, "serial", None))
        frame["timestamp_delta_ms"] = float(getattr(snapshot, "timestamp_delta_ms", np.nan))
        frame["within_sync_tolerance"] = bool(getattr(snapshot, "within_sync_tolerance", False))

    def save(self, *, prefix: str = "handover_3d_debug") -> Path:
        if not self.enabled:
            raise RuntimeError("3D debug recording is disabled.")
        if not self.frames:
            raise RuntimeError("No 3D debug frames are buffered.")

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{prefix}_{_timestamp_for_filename()}_session{self.session_index:03d}.npz"
        payload = self._build_payload(path)
        np.savez_compressed(path, **payload)
        return path

    def _build_payload(self, path: Path) -> dict[str, Any]:
        frames = self.frames
        variable_cloud_keys = ("object_points_base", "template_points_base")
        string_keys = (
            "measurement_source",
            "object_label",
            "fitted_label",
            "template_id",
            "shape_fitting_reason",
            "shape_fitting_scale_mode",
            "shape_fitting_silhouette_reason",
            "hand_selection_reason",
            "hand_selector_cam0_reject_reason",
            "hand_selector_cam1_reject_reason",
            "cam0_hand_reason",
            "cam1_hand_reason",
            "selected_handedness",
            "record_clock_text",
            "cam0_serial",
            "cam1_serial",
            "tactile_status",
            "tactile_error",
        )
        max_tactile_mags = 0
        if "tactile_num_mags" in frames[0]:
            max_tactile_mags = max(int(frame.get("tactile_num_mags", 0)) for frame in frames)
        payload: dict[str, Any] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "saved_path": np.asarray(str(path)),
            "created_at": np.asarray(datetime.now().isoformat(timespec="seconds")),
            "frame_count": np.asarray(len(frames), dtype=np.int32),
            "max_object_points": np.asarray(int(self.max_object_points), dtype=np.int32),
            "max_template_points": np.asarray(int(self.max_template_points), dtype=np.int32),
            "max_tactile_mags": np.asarray(int(max_tactile_mags), dtype=np.int32),
        }

        keys = sorted(frames[0].keys())
        for key in keys:
            values = [frame[key] for frame in frames]
            if key in variable_cloud_keys:
                payload[f"{key}_offsets"] = _build_offsets(values)
                payload[f"{key}_flat"] = _flatten_point_frames(values)
            elif key in IMAGE_ARRAY_KEYS:
                payload[key] = _stack_image_values(values, key)
            elif key == "tactile_values":
                payload[key] = _stack_tactile_values(values)
            elif key == "tactile_mag_norms":
                payload[key] = _stack_tactile_mag_norms(values)
            elif key in string_keys:
                payload[key] = np.asarray(["" if value is None else str(value) for value in values])
            else:
                payload[key] = np.asarray(values)
        return payload


def _build_offsets(point_frames: list[np.ndarray]) -> np.ndarray:
    offsets = np.zeros((len(point_frames) + 1,), dtype=np.int64)
    cursor = 0
    for index, points in enumerate(point_frames):
        cursor += len(np.asarray(points, dtype=np.float32).reshape((-1, 3)))
        offsets[index + 1] = cursor
    return offsets


def _flatten_point_frames(point_frames: list[np.ndarray]) -> np.ndarray:
    if not point_frames:
        return np.empty((0, 3), dtype=np.float32)
    normalized = [np.asarray(points, dtype=np.float32).reshape((-1, 3)) for points in point_frames]
    if not normalized:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(normalized, axis=0).astype(np.float32, copy=False)


def _stack_image_values(values: list[Any], key: str) -> np.ndarray:
    dtype = np.uint8 if key.endswith("_color_image") else np.float32
    arrays: list[np.ndarray] = []
    for value in values:
        if isinstance(value, Path):
            arrays.append(_safe_load_array(value, dtype=dtype))
        else:
            arrays.append(np.array(value, dtype=dtype, copy=True))
    return np.stack(arrays, axis=0)


def _stack_tactile_values(values: list[Any]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32).reshape((-1, 3)) for value in values]
    max_mags = max((len(array) for array in arrays), default=0)
    output = np.full((len(arrays), max_mags, 3), np.nan, dtype=np.float32)
    for index, array in enumerate(arrays):
        if len(array) > 0:
            output[index, : len(array), :] = array
    return output


def _stack_tactile_mag_norms(values: list[Any]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    max_mags = max((len(array) for array in arrays), default=0)
    output = np.full((len(arrays), max_mags), np.nan, dtype=np.float32)
    for index, array in enumerate(arrays):
        if len(array) > 0:
            output[index, : len(array)] = array
    return output


def _reconstruct_point_frames(flat_points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    flat_points = np.asarray(flat_points, dtype=np.float32).reshape((-1, 3))
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    frames = np.empty((max(len(offsets) - 1, 0),), dtype=object)
    for index in range(len(frames)):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        frames[index] = flat_points[start:end].copy()
    return frames


def load_debug_3d_npz(path: str | Path) -> dict[str, Any]:
    """Load a 3D debug recording with object arrays enabled."""

    with np.load(Path(path), allow_pickle=True) as data:
        loaded = {key: data[key] for key in data.files}

    for key in ("object_points_base", "template_points_base"):
        flat_key = f"{key}_flat"
        offsets_key = f"{key}_offsets"
        if flat_key in loaded and offsets_key in loaded:
            loaded[key] = _reconstruct_point_frames(loaded[flat_key], loaded[offsets_key])
    return loaded
