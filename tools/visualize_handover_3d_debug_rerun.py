#!/usr/bin/env python3
"""Rerun viewer/exporter for saved handover 3D debug recordings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

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

MARKERS = (
    ("object_point_base", "/world/object/target_point", 0.012, (255, 140, 13)),
    ("grasp_point_base", "/world/grasp/point", 0.014, (26, 242, 51)),
    ("selected_palm_center_base", "/world/selected/palm_center", 0.012, (255, 242, 26)),
    ("selected_wrist_base", "/world/selected/wrist", 0.009, (179, 179, 255)),
    ("template_centroid_base", "/world/template/centroid", 0.009, (0, 230, 255)),
)

EMPTY_POINTS = np.empty((0, 3), dtype=np.float32)
EMPTY_LINE_STRIPS: list[np.ndarray] = []
TEMPLATE_AXIS_COLORS = [(255, 0, 0), (0, 220, 0), (0, 120, 255)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved handover 3D debug .npz recording in Rerun.")
    parser.add_argument("recording", help="Path to output/debug_3d/*.npz")
    parser.add_argument("--normal-length", type=float, default=0.08, help="Palm normal line length in meters.")
    parser.add_argument("--template-axis-length", type=float, default=0.06, help="Template local x/y/z axis line length in meters.")
    parser.add_argument("--depth-max-m", type=float, default=1.7, help="Maximum depth range shown in Rerun depth views.")
    parser.add_argument("--dry-run", action="store_true", help="Load and validate Rerun frame conversion without importing Rerun.")
    parser.add_argument("--connect", action="store_true", help="Connect to an already running Rerun viewer instead of spawning one.")
    parser.add_argument("--save-rrd", default=None, help="Save the Rerun recording to this .rrd path instead of opening a viewer.")
    parser.add_argument("--app-id", default="handover_3d_debug", help="Rerun application id.")
    parser.add_argument(
        "--no-point-coordinate-labels",
        action="store_true",
        help="Do not attach per-point base-frame xyz labels for Rerun selection/inspection.",
    )
    parser.add_argument(
        "--point-coordinate-precision",
        type=int,
        default=4,
        help="Decimal places used for per-point base-frame xyz labels.",
    )
    return parser.parse_args()


def _as_points(value: object) -> np.ndarray:
    if value is None:
        return EMPTY_POINTS
    points = np.asarray(value, dtype=np.float32).reshape((-1, 3))
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def _is_valid_vec(value: object, length: int = 3) -> bool:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    return len(vec) >= int(length) and bool(np.isfinite(vec[:length]).all())


def _valid_template_axes(data: dict[str, Any], idx: int) -> bool:
    if "template_axes_base" not in data or "template_centroid_base" not in data:
        return False
    if not _is_valid_vec(data["template_centroid_base"][idx], 3):
        return False
    axes = np.asarray(data["template_axes_base"][idx], dtype=np.float32)
    return axes.shape == (3, 3) and bool(np.isfinite(axes).all())


def _first_valid_template_axes_index(data: dict[str, Any]) -> int | None:
    if "template_axes_base" not in data:
        return None
    for idx in range(int(data["frame_count"])):
        if _valid_template_axes(data, idx):
            return idx
    return None


def _string_at(data: dict[str, Any], key: str, index: int) -> str:
    if key not in data:
        return ""
    value = data[key][index]
    return "" if value is None else str(value)


def _finite_float_at(data: dict[str, Any], key: str, index: int) -> float | None:
    if key not in data:
        return None
    value = float(data[key][index])
    return value if np.isfinite(value) else None


def has_tactile_stream(data: dict[str, Any]) -> bool:
    return "tactile_valid" in data and "tactile_values" in data and "tactile_total_norm" in data


def tactile_status_text(data: dict[str, Any], frame_index: int) -> str:
    idx = int(frame_index)
    if not has_tactile_stream(data):
        return "No tactile stream in this recording."

    valid = bool(data["tactile_valid"][idx])
    num_mags = int(data["tactile_num_mags"][idx]) if "tactile_num_mags" in data else 0
    total_norm = _finite_float_at(data, "tactile_total_norm", idx)
    ref_norm = _finite_float_at(data, "tactile_release_ref_norm", idx)
    delta_norm = _finite_float_at(data, "tactile_release_delta_norm", idx)
    status = _string_at(data, "tactile_status", idx) or "-"
    error = _string_at(data, "tactile_error", idx)
    parts = [
        f"[TACTILE {idx + 1}/{int(data['frame_count'])}]",
        f"valid={valid}",
        f"num_mags={num_mags}",
        f"norm={total_norm:.3f}" if total_norm is not None else "norm=None",
        f"ref={ref_norm:.3f}" if ref_norm is not None else "ref=None",
        f"delta={delta_norm:.3f}" if delta_norm is not None else "delta=None",
        f"status={status}",
    ]
    if error:
        parts.append(f"error={error}")
    return " ".join(parts)


def _rgb(color: tuple[int, int, int] | tuple[float, float, float]) -> tuple[int, int, int]:
    values = np.asarray(color, dtype=np.float32).reshape(3)
    if float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    clipped = np.clip(values, 0, 255).astype(np.uint8)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def _color_array(count: int, color: tuple[int, int, int] | tuple[float, float, float]) -> np.ndarray:
    return np.tile(np.asarray(_rgb(color), dtype=np.uint8).reshape(1, 3), (max(int(count), 0), 1))


def point_coordinate_labels(points: np.ndarray, prefix: str, precision: int = 4) -> list[str]:
    points = _as_points(points)
    if len(points) == 0:
        return []

    decimals = max(int(precision), 0)
    index_width = max(4, len(str(len(points) - 1)))
    labels: list[str] = []
    for point_index, (x, y, z) in enumerate(points):
        labels.append(
            f"{prefix}[{point_index:0{index_width}d}] "
            f"base=({float(x):.{decimals}f}, {float(y):.{decimals}f}, {float(z):.{decimals}f})m"
        )
    return labels


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


def frame_status_text(data: dict[str, Any], frame_index: int) -> str:
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
    hand_selection = _string_at(data, "hand_selection_reason", idx) or "-"

    def fmt(key: str, length: int = 3) -> str:
        value = data[key][idx]
        if not _is_valid_vec(value, length):
            return "None"
        vec = np.asarray(value, dtype=np.float32).reshape(-1)[:length]
        return "(" + ", ".join(f"{float(v):.3f}" for v in vec) + ")"

    return (
        f"[FRAME {idx + 1}/{count}] t={record_clock} label={label} template={template_id} "
        f"fit={reason} src={source} hand_select={hand_selection} object={fmt('object_point_base')} "
        f"grasp={fmt('grasp_point_base')} eef={fmt('eef_pose_base', 6)} "
        f"tactile={_finite_float_at(data, 'tactile_total_norm', idx) if has_tactile_stream(data) else 'none'}"
    )


def sync_status_text(data: dict[str, Any], frame_index: int) -> str:
    idx = int(frame_index)
    if not has_image_streams(data):
        return "No synchronized RGB/depth images in this recording."

    delta_ms = _finite_float_at(data, "timestamp_delta_ms", idx)
    within_sync = bool(data["within_sync_tolerance"][idx]) if "within_sync_tolerance" in data else False

    parts = [
        f"[SYNC {idx + 1}/{int(data['frame_count'])}]",
        f"delta_ms={delta_ms:.3f}" if delta_ms is not None else "delta_ms=None",
        f"within_tolerance={within_sync}",
    ]
    for camera_id in (0, 1):
        prefix = f"cam{camera_id}"
        timestamp_ms = _finite_float_at(data, f"{prefix}_timestamp_ms", idx)
        serial = _string_at(data, f"{prefix}_serial", idx) or "-"
        parts.append(f"{prefix}_serial={serial}")
        parts.append(f"{prefix}_timestamp_ms={timestamp_ms:.3f}" if timestamp_ms is not None else f"{prefix}_timestamp_ms=None")
    return " ".join(parts)


def has_camera_images(data: dict[str, Any], camera_id: int) -> bool:
    prefix = f"cam{camera_id}"
    return f"{prefix}_color_image" in data and f"{prefix}_depth_image_m" in data


def has_image_streams(data: dict[str, Any]) -> bool:
    return has_camera_images(data, 0) and has_camera_images(data, 1)


def _point_counts(data: dict[str, Any], key: str) -> np.ndarray:
    if key in data:
        return np.asarray([len(_as_points(frame)) for frame in data[key]], dtype=np.int64)
    flat_key = f"{key}_flat"
    offsets_key = f"{key}_offsets"
    if flat_key in data and offsets_key in data:
        return np.diff(np.asarray(data[offsets_key], dtype=np.int64))
    return np.zeros((int(data["frame_count"]),), dtype=np.int64)


def _first_positive_index(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values) > 0)
    return None if len(indices) == 0 else int(indices[0])


def _hand_valid_counts(data: dict[str, Any], camera_id: int) -> np.ndarray:
    count_key = f"cam{camera_id}_hand_valid_count"
    if count_key in data:
        return np.asarray(data[count_key], dtype=np.int64)
    mask_key = f"cam{camera_id}_hand_valid_mask"
    if mask_key in data:
        return np.asarray(data[mask_key], dtype=bool).sum(axis=1).astype(np.int64)
    return np.zeros((int(data["frame_count"]),), dtype=np.int64)


def _image_means(data: dict[str, Any], key: str, indices: list[int]) -> list[float]:
    if key not in data:
        return []
    return [float(np.nanmean(np.asarray(data[key][index]))) for index in indices]


def recording_summary_text(data: dict[str, Any]) -> str:
    frame_count = int(data["frame_count"])
    object_counts = _point_counts(data, "object_points_base")
    template_counts = _point_counts(data, "template_points_base")
    cam0_hand_counts = _hand_valid_counts(data, 0)
    cam1_hand_counts = _hand_valid_counts(data, 1)
    selected_hand = np.asarray(data.get("selected_hand_valid", np.zeros(frame_count, dtype=bool)), dtype=bool)
    lines = [
        f"frames={frame_count}",
        f"object_cloud_frames={int((object_counts > 0).sum())}/{frame_count} first={_first_positive_index(object_counts)} max_points={int(object_counts.max()) if len(object_counts) else 0}",
        f"template_cloud_frames={int((template_counts > 0).sum())}/{frame_count} first={_first_positive_index(template_counts)} max_points={int(template_counts.max()) if len(template_counts) else 0}",
        f"cam0_hand_valid_frames={int((cam0_hand_counts > 0).sum())}/{frame_count} first={_first_positive_index(cam0_hand_counts)} max_landmarks={int(cam0_hand_counts.max()) if len(cam0_hand_counts) else 0}",
        f"cam1_hand_valid_frames={int((cam1_hand_counts > 0).sum())}/{frame_count} first={_first_positive_index(cam1_hand_counts)} max_landmarks={int(cam1_hand_counts.max()) if len(cam1_hand_counts) else 0}",
        f"selected_hand_valid_frames={int(selected_hand.sum())}/{frame_count}",
    ]
    axes_first = _first_valid_template_axes_index(data)
    axes_key_status = "present" if "template_axes_base" in data else "missing"
    lines.append(f"template_local_axes={axes_key_status} first_valid={axes_first}")
    if has_tactile_stream(data):
        tactile_valid = np.asarray(data["tactile_valid"], dtype=bool)
        tactile_norms = np.asarray(data["tactile_total_norm"], dtype=np.float32)
        finite_norms = tactile_norms[np.isfinite(tactile_norms)]
        num_mags = np.asarray(data.get("tactile_num_mags", np.zeros(frame_count, dtype=np.int32)), dtype=np.int32)
        norm_text = (
            f"norm_min={float(finite_norms.min()):.3f} norm_max={float(finite_norms.max()):.3f}"
            if len(finite_norms) > 0
            else "norm_min=None norm_max=None"
        )
        lines.append(
            f"tactile_valid_frames={int(tactile_valid.sum())}/{frame_count} "
            f"first={_first_positive_index(tactile_valid.astype(np.int32))} "
            f"max_mags={int(num_mags.max()) if len(num_mags) else 0} {norm_text}"
        )
    else:
        lines.append("tactile_stream=missing")
    if "hand_selector_cam0_reject_reason" in data:
        lines.append(f"frame0_hand_reject_cam0={_string_at(data, 'hand_selector_cam0_reject_reason', 0) or '-'}")
        lines.append(f"frame0_hand_reject_cam1={_string_at(data, 'hand_selector_cam1_reject_reason', 0) or '-'}")
    if has_image_streams(data):
        indices = sorted(set([0, min(frame_count - 1, 1), frame_count // 2, frame_count - 1]))
        for key in ("cam0_color_image", "cam1_color_image", "cam0_depth_image_m", "cam1_depth_image_m"):
            means = _image_means(data, key, indices)
            unique = len(set(round(value, 4) for value in means))
            lines.append(
                f"{key}: shape={tuple(np.asarray(data[key]).shape)} "
                f"sample_indices={indices} sample_means={[round(value, 5) for value in means]} unique_means={unique}"
            )
    return "\n".join(lines)


def _camera_image_shapes(data: dict[str, Any], camera_id: int, idx: int) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    prefix = f"cam{camera_id}"
    color_shape = tuple(int(value) for value in np.asarray(data[f"{prefix}_color_image"][idx]).shape) if f"{prefix}_color_image" in data else None
    depth_shape = tuple(int(value) for value in np.asarray(data[f"{prefix}_depth_image_m"][idx]).shape) if f"{prefix}_depth_image_m" in data else None
    return color_shape, depth_shape


def _valid_color_image_rgb(data: dict[str, Any], camera_id: int, idx: int) -> np.ndarray | None:
    key = f"cam{camera_id}_color_image"
    if key not in data:
        return None
    image = np.asarray(data[key][idx])
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image[..., ::-1].copy()


def _valid_depth_image_m(data: dict[str, Any], camera_id: int, idx: int) -> np.ndarray | None:
    key = f"cam{camera_id}_depth_image_m"
    if key not in data:
        return None
    depth = np.asarray(data[key][idx], dtype=np.float32)
    if depth.ndim != 2:
        return None
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _valid_intrinsics(data: dict[str, Any], camera_id: int, idx: int) -> tuple[float, float, float, float] | None:
    key = f"cam{camera_id}_intrinsics"
    if key not in data:
        return None
    values = np.asarray(data[key][idx], dtype=np.float32).reshape(-1)
    if len(values) < 4 or not np.isfinite(values[:4]).all():
        return None
    fx, fy, cx, cy = (float(value) for value in values[:4])
    return fx, fy, cx, cy


def build_hand_payload(
    points_base: np.ndarray,
    valid_mask: np.ndarray,
    *,
    selected: bool,
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int, int]]:
    points = np.asarray(points_base, dtype=np.float32).reshape((-1, 3))
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_indices = [idx for idx in range(min(len(points), len(mask))) if mask[idx] and np.isfinite(points[idx]).all()]
    if not valid_indices:
        return EMPTY_POINTS, [], (255, 242, 26) if selected else (255, 255, 255)

    valid_points = points[valid_indices]
    index_map = {original_index: new_index for new_index, original_index in enumerate(valid_indices)}
    strips: list[np.ndarray] = []
    for start_index, end_index in HAND_CONNECTIONS:
        if start_index in index_map and end_index in index_map:
            strips.append(valid_points[[index_map[start_index], index_map[end_index]]])

    return valid_points, strips, (255, 242, 26) if selected else (255, 255, 255)


def count_frame_entities(data: dict[str, Any], frame_index: int) -> int:
    idx = int(frame_index)
    count = 1  # frame status text
    count += int(len(_as_points(data["object_points_base"][idx])) > 0)
    count += int(len(_as_points(data["template_points_base"][idx])) > 0)
    for key, _, _, _ in MARKERS:
        count += int(key in data and _is_valid_vec(data[key][idx], 3))
    for camera_id in (0, 1):
        points, strips, _ = build_hand_payload(
            data[f"cam{camera_id}_hand_points_base"][idx],
            data[f"cam{camera_id}_hand_valid_mask"][idx],
            selected=int(data["selected_hand_camera"][idx]) == camera_id,
        )
        count += int(len(points) > 0)
        count += int(len(strips) > 0)
    count += int(_valid_palm_normal(data, idx))
    count += int(_is_valid_vec(data["eef_pose_base"][idx], 6))
    count += int(_valid_template_axes(data, idx))
    if has_image_streams(data):
        count += 1  # sync status text
        for camera_id in (0, 1):
            color = _valid_color_image_rgb(data, camera_id, idx)
            depth = _valid_depth_image_m(data, camera_id, idx)
            count += int(color is not None)
            count += int(depth is not None)
    if has_tactile_stream(data):
        count += 1  # tactile status text
        count += int(_finite_float_at(data, "tactile_total_norm", idx) is not None)
        count += int(_finite_float_at(data, "tactile_release_ref_norm", idx) is not None)
        count += int(_finite_float_at(data, "tactile_release_delta_norm", idx) is not None)
        values = np.asarray(data["tactile_values"][idx], dtype=np.float32).reshape((-1, 3))
        count += int(np.isfinite(values).any()) * int(np.isfinite(values).all(axis=1).sum()) * 4
    return count


def _valid_palm_normal(data: dict[str, Any], idx: int) -> bool:
    if not (_is_valid_vec(data["selected_palm_center_base"][idx], 3) and _is_valid_vec(data["selected_palm_normal_base"][idx], 3)):
        return False
    normal = np.asarray(data["selected_palm_normal_base"][idx], dtype=np.float32).reshape(-1)[:3]
    return bool(float(np.linalg.norm(normal)) > 1e-6)


def _import_rerun() -> Any:
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "rerun-sdk is required for this viewer. Install it with: pip install rerun-sdk"
        ) from exc
    return rr


def init_rerun(rr: Any, args: argparse.Namespace) -> None:
    rr.init(args.app_id)
    if args.save_rrd:
        Path(args.save_rrd).expanduser().parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(args.save_rrd))
    elif args.connect:
        if hasattr(rr, "connect_grpc"):
            rr.connect_grpc()
        elif hasattr(rr, "connect"):
            rr.connect()
        else:
            raise RuntimeError("This rerun-sdk version does not expose connect_grpc() or connect().")
    elif hasattr(rr, "spawn"):
        rr.spawn()


def send_blueprint(rr: Any, *, include_images: bool, include_tactile: bool = False) -> None:
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return

    try:
        views: list[Any] = [rrb.Spatial3DView(origin="/world", name="Handover 3D Debug")]
        if include_images and hasattr(rrb, "Spatial2DView"):
            camera_views = [
                rrb.Spatial2DView(origin="/cameras/cam0/rgb", name="Cam0 RGB"),
                rrb.Spatial2DView(origin="/cameras/cam0/depth", name="Cam0 Depth"),
                rrb.Spatial2DView(origin="/cameras/cam1/rgb", name="Cam1 RGB"),
                rrb.Spatial2DView(origin="/cameras/cam1/depth", name="Cam1 Depth"),
            ]
            if hasattr(rrb, "Grid"):
                views.append(rrb.Grid(*camera_views))
            else:
                views.extend(camera_views)
        # if hasattr(rrb, "TextDocumentView"):
        #     views.append(rrb.TextDocumentView(origin="/status", name="Frame Status"))
        if include_tactile and hasattr(rrb, "TimeSeriesView"):
            views.append(rrb.TimeSeriesView(origin="/tactile", name="Tactile"))
        if hasattr(rrb, "Horizontal") and len(views) > 1:
            blueprint = rrb.Blueprint(rrb.Horizontal(*views), collapse_panels=True)
        else:
            blueprint = rrb.Blueprint(*views, collapse_panels=True)
        rr.send_blueprint(blueprint)
    except Exception as exc:  # pragma: no cover - best effort viewer layout
        print(f"[WARN] Could not send Rerun blueprint: {exc}", file=sys.stderr)


def set_frame_time(rr: Any, data: dict[str, Any], idx: int) -> None:
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=int(idx))
    else:
        rr.set_time_sequence("frame", int(idx))

    if "record_elapsed_s" not in data:
        return
    elapsed_s = float(data["record_elapsed_s"][idx])
    if not np.isfinite(elapsed_s):
        return
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("record_elapsed_s", elapsed_s)
        return
    if hasattr(rr, "set_time"):
        try:
            rr.set_time("record_elapsed_s", duration=elapsed_s)
        except TypeError:
            pass


def log_points(
    rr: Any,
    entity: str,
    points: np.ndarray,
    color: tuple[int, int, int] | tuple[float, float, float],
    *,
    radius: float | None = None,
    labels: list[str] | None = None,
    show_labels: bool | None = None,
) -> None:
    points = _as_points(points)
    if len(points) == 0:
        log_clear(rr, entity)
        return
    kwargs: dict[str, Any] = {"colors": _color_array(len(points), color)}
    if radius is not None:
        kwargs["radii"] = np.full((len(points),), float(radius), dtype=np.float32)
    if labels is not None:
        labels = list(labels)
        if len(labels) != len(points):
            raise ValueError(f"Point label count mismatch for {entity}: labels={len(labels)} points={len(points)}")
        kwargs["labels"] = labels
        if show_labels is not None:
            kwargs["show_labels"] = bool(show_labels)
    rr.log(entity, rr.Points3D(points, **kwargs))


def log_line_strips(
    rr: Any,
    entity: str,
    strips: list[np.ndarray],
    color: tuple[int, int, int] | tuple[float, float, float],
    *,
    radius: float | None = None,
) -> None:
    normalized = [np.asarray(strip, dtype=np.float32).reshape((-1, 3)) for strip in strips if len(strip) > 0]
    if not normalized:
        log_clear(rr, entity)
        return
    kwargs: dict[str, Any] = {"colors": _rgb(color)}
    if radius is not None:
        kwargs["radii"] = float(radius)
    rr.log(entity, rr.LineStrips3D(normalized, **kwargs))


def log_clear(rr: Any, entity: str) -> None:
    if hasattr(rr, "Clear"):
        rr.log(entity, rr.Clear(recursive=False))
    elif hasattr(rr, "log_cleared"):
        rr.log_cleared(entity, recursive=False)


def log_pinhole_if_available(
    rr: Any,
    entity: str,
    intrinsics: tuple[float, float, float, float] | None,
    *,
    width: int,
    height: int,
) -> None:
    if intrinsics is None or not hasattr(rr, "Pinhole"):
        return
    fx, fy, cx, cy = intrinsics
    rr.log(
        entity,
        rr.Pinhole(
            width=int(width),
            height=int(height),
            focal_length=[float(fx), float(fy)],
            principal_point=[float(cx), float(cy)],
        ),
    )


def log_camera_images(rr: Any, data: dict[str, Any], idx: int, camera_id: int, *, depth_max_m: float) -> None:
    prefix = f"cam{camera_id}"
    rgb_entity = f"/cameras/{prefix}/rgb"
    depth_entity = f"/cameras/{prefix}/depth"
    intrinsics = _valid_intrinsics(data, camera_id, idx)

    color_rgb = _valid_color_image_rgb(data, camera_id, idx)
    if color_rgb is None:
        log_clear(rr, rgb_entity)
    else:
        height, width = color_rgb.shape[:2]
        log_pinhole_if_available(rr, rgb_entity, intrinsics, width=width, height=height)
        rr.log(rgb_entity, rr.Image(color_rgb))

    depth_m = _valid_depth_image_m(data, camera_id, idx)
    if depth_m is None:
        log_clear(rr, depth_entity)
    else:
        height, width = depth_m.shape[:2]
        log_pinhole_if_available(rr, depth_entity, intrinsics, width=width, height=height)
        depth_max_m = max(float(depth_max_m), 1e-6)
        try:
            rr.log(depth_entity, rr.DepthImage(depth_m, meter=1.0, depth_range=[0.0, depth_max_m]))
        except TypeError:
            rr.log(depth_entity, rr.DepthImage(depth_m, meter=1.0))


def log_hand(
    rr: Any,
    data: dict[str, Any],
    idx: int,
    camera_id: int,
    *,
    include_coordinate_labels: bool,
    point_coordinate_precision: int,
) -> None:
    selected = int(data["selected_hand_camera"][idx]) == int(camera_id)
    default_color = (64, 115, 255) if camera_id == 0 else (255, 77, 191)
    points, strips, selected_color = build_hand_payload(
        data[f"cam{camera_id}_hand_points_base"][idx],
        data[f"cam{camera_id}_hand_valid_mask"][idx],
        selected=selected,
    )
    color = selected_color if selected else default_color
    prefix = f"/world/hands/cam{camera_id}"
    labels = point_coordinate_labels(points, f"cam{camera_id}_hand", point_coordinate_precision) if include_coordinate_labels else None
    log_points(rr, f"{prefix}/keypoints", points, color, radius=0.006, labels=labels, show_labels=False)
    log_line_strips(rr, f"{prefix}/skeleton", strips, color, radius=0.003)


def log_marker(
    rr: Any,
    data: dict[str, Any],
    idx: int,
    key: str,
    entity: str,
    radius: float,
    color: tuple[int, int, int],
) -> None:
    if key in data and _is_valid_vec(data[key][idx], 3):
        point = np.asarray(data[key][idx], dtype=np.float32).reshape(-1)[:3].reshape(1, 3)
    else:
        point = EMPTY_POINTS
    log_points(rr, entity, point, color, radius=radius)


def log_palm_normal(rr: Any, data: dict[str, Any], idx: int, normal_length_m: float) -> None:
    strips: list[np.ndarray] = []
    if _valid_palm_normal(data, idx):
        center = np.asarray(data["selected_palm_center_base"][idx], dtype=np.float32).reshape(-1)[:3]
        normal = np.asarray(data["selected_palm_normal_base"][idx], dtype=np.float32).reshape(-1)[:3]
        normal = normal / float(np.linalg.norm(normal))
        strips = [np.stack([center, center + normal * float(normal_length_m)], axis=0)]
    log_line_strips(rr, "/world/selected/palm_normal", strips, (255, 242, 26), radius=0.004)


def log_eef_axes(rr: Any, pose_xyz_rotvec: np.ndarray) -> None:
    if not _is_valid_vec(pose_xyz_rotvec, 6):
        log_line_strips(rr, "/world/eef/axes", EMPTY_LINE_STRIPS, (255, 255, 255), radius=0.004)
        return
    pose = np.asarray(pose_xyz_rotvec, dtype=np.float32).reshape(-1)[:6]
    origin = pose[:3]
    rotation = rotvec_to_matrix(pose[3:6])
    axis_length = 0.055
    strips = [np.stack([origin, origin + rotation[:, axis_index] * axis_length], axis=0) for axis_index in range(3)]
    colors = [(255, 0, 0), (0, 220, 0), (0, 120, 255)]
    try:
        rr.log("/world/eef/axes", rr.LineStrips3D(strips, colors=colors, radii=0.004))
    except TypeError:
        log_line_strips(rr, "/world/eef/axes", strips, (255, 255, 255), radius=0.004)


def log_template_axes(rr: Any, data: dict[str, Any], idx: int, axis_length_m: float) -> None:
    entity = "/world/template/local_axes"
    if not _valid_template_axes(data, idx):
        log_line_strips(rr, entity, EMPTY_LINE_STRIPS, (255, 255, 255), radius=0.004)
        return

    origin = np.asarray(data["template_centroid_base"][idx], dtype=np.float32).reshape(-1)[:3]
    axes = np.asarray(data["template_axes_base"][idx], dtype=np.float32).reshape((3, 3))
    strips: list[np.ndarray] = []
    length = max(float(axis_length_m), 0.0)
    for axis_index in range(3):
        axis = axes[axis_index]
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 1e-9:
            log_line_strips(rr, entity, EMPTY_LINE_STRIPS, (255, 255, 255), radius=0.004)
            return
        axis = axis / norm
        strips.append(np.stack([origin, origin + axis * length], axis=0))
    try:
        rr.log(entity, rr.LineStrips3D(strips, colors=TEMPLATE_AXIS_COLORS, radii=0.004))
    except TypeError:
        log_line_strips(rr, entity, strips, (255, 255, 255), radius=0.004)


def log_text(rr: Any, entity: str, text: str) -> None:
    if hasattr(rr, "TextDocument"):
        rr.log(entity, rr.TextDocument(text))
    elif hasattr(rr, "TextLog"):
        rr.log(entity, rr.TextLog(text))


def log_status(rr: Any, text: str) -> None:
    log_text(rr, "/status/frame", text)


def _scalar_payload(rr: Any, value: float):
    scalar_cls = getattr(rr, "Scalars", None)
    if scalar_cls is None:
        scalar_cls = getattr(rr, "Scalar")
    return scalar_cls(float(value))


def log_scalar_if_finite(rr: Any, entity: str, value: float | None) -> None:
    if value is None or not np.isfinite(float(value)):
        log_clear(rr, entity)
        return
    rr.log(entity, _scalar_payload(rr, float(value)))


def log_tactile(rr: Any, data: dict[str, Any], idx: int) -> None:
    if not has_tactile_stream(data):
        log_text(rr, "/status/tactile", "No tactile stream in this recording.")
        return

    log_scalar_if_finite(rr, "/tactile/total_norm", _finite_float_at(data, "tactile_total_norm", idx))
    log_scalar_if_finite(rr, "/tactile/release_ref_norm", _finite_float_at(data, "tactile_release_ref_norm", idx))
    log_scalar_if_finite(rr, "/tactile/release_delta_norm", _finite_float_at(data, "tactile_release_delta_norm", idx))
    log_text(rr, "/status/tactile", tactile_status_text(data, idx))

    values = np.asarray(data["tactile_values"][idx], dtype=np.float32).reshape((-1, 3))
    norms = np.asarray(data.get("tactile_mag_norms", np.full((int(data["frame_count"]), len(values)), np.nan))[idx], dtype=np.float32).reshape(-1)
    for mag_idx, mag_values in enumerate(values):
        bx, by, bz = (float(value) for value in mag_values)
        norm = float(norms[mag_idx]) if mag_idx < len(norms) else float(np.linalg.norm(mag_values))
        log_scalar_if_finite(rr, f"/tactile/mag_{mag_idx}/x", bx)
        log_scalar_if_finite(rr, f"/tactile/mag_{mag_idx}/y", by)
        log_scalar_if_finite(rr, f"/tactile/mag_{mag_idx}/z", bz)
        log_scalar_if_finite(rr, f"/tactile/mag_{mag_idx}/norm", norm)


def log_frame(
    rr: Any,
    data: dict[str, Any],
    idx: int,
    *,
    normal_length_m: float,
    template_axis_length_m: float,
    depth_max_m: float,
    include_coordinate_labels: bool,
    point_coordinate_precision: int,
) -> None:
    set_frame_time(rr, data, idx)

    object_points = _as_points(data["object_points_base"][idx])
    object_color = (199, 199, 199) if bool(data["object_valid"][idx]) else (217, 64, 56)
    object_labels = point_coordinate_labels(object_points, "object", point_coordinate_precision) if include_coordinate_labels else None
    log_points(rr, "/world/object/cloud", object_points, object_color, radius=0.0025, labels=object_labels, show_labels=False)

    template_points = _as_points(data["template_points_base"][idx])
    template_labels = point_coordinate_labels(template_points, "template", point_coordinate_precision) if include_coordinate_labels else None
    log_points(rr, "/world/template/cloud", template_points, (26, 204, 242), radius=0.0025, labels=template_labels, show_labels=False)

    log_hand(
        rr,
        data,
        idx,
        0,
        include_coordinate_labels=include_coordinate_labels,
        point_coordinate_precision=point_coordinate_precision,
    )
    log_hand(
        rr,
        data,
        idx,
        1,
        include_coordinate_labels=include_coordinate_labels,
        point_coordinate_precision=point_coordinate_precision,
    )

    for key, entity, radius, color in MARKERS:
        log_marker(rr, data, idx, key, entity, radius, color)

    log_palm_normal(rr, data, idx, normal_length_m)
    log_eef_axes(rr, data["eef_pose_base"][idx])
    log_template_axes(rr, data, idx, template_axis_length_m)
    log_tactile(rr, data, idx)
    log_status(rr, frame_status_text(data, idx) + "\n\n" + tactile_status_text(data, idx) + "\n\n" + recording_summary_text(data))
    if has_image_streams(data):
        for camera_id in (0, 1):
            log_camera_images(rr, data, idx, camera_id, depth_max_m=depth_max_m)
        log_text(rr, "/status/sync", sync_status_text(data, idx))
    else:
        log_clear(rr, "/status/sync")


def dry_run(data: dict[str, Any], *, normal_length_m: float, template_axis_length_m: float, depth_max_m: float) -> None:
    print("[INFO] Debug recording summary:")
    print(recording_summary_text(data))
    frame_count = int(data["frame_count"])
    first_entities = count_frame_entities(data, 0)
    last_entities = count_frame_entities(data, frame_count - 1)
    object_points = len(_as_points(data["object_points_base"][0]))
    template_points = len(_as_points(data["template_points_base"][0]))
    print(
        f"[INFO] Loaded {frame_count} frames. "
        f"Frame 1 converts to {first_entities} Rerun entities "
        f"({object_points} object points, {template_points} template points)."
    )
    if frame_count > 1:
        print(f"[INFO] Last frame converts to {last_entities} Rerun entities.")
    if _valid_palm_normal(data, 0):
        print(f"[INFO] Palm normal line length: {float(normal_length_m):.3f} m.")
    axes_first = _first_valid_template_axes_index(data)
    if axes_first is None:
        print("[INFO] Template local axes: not recorded or no valid axes.")
    else:
        print(f"[INFO] Template local axes first valid frame: {axes_first}; axis length: {float(template_axis_length_m):.3f} m.")
    if has_image_streams(data):
        print(f"[INFO] RGB/depth image streams detected. Depth display range: 0.000-{max(float(depth_max_m), 1e-6):.3f} m.")
        for camera_id in (0, 1):
            color_shape, depth_shape = _camera_image_shapes(data, camera_id, 0)
            intrinsics = _valid_intrinsics(data, camera_id, 0)
            print(f"[INFO] cam{camera_id} color={color_shape} depth={depth_shape} intrinsics={intrinsics}")
        print(sync_status_text(data, 0))
    else:
        print("[INFO] No synchronized RGB/depth image streams found in this recording.")
    print(tactile_status_text(data, 0))
    print(frame_status_text(data, 0))


def main() -> int:
    args = parse_args()
    if args.connect and args.save_rrd:
        raise RuntimeError("--connect and --save-rrd are mutually exclusive.")

    data = load_debug_3d_npz(args.recording)
    frame_count = int(data["frame_count"])
    if frame_count <= 0:
        raise RuntimeError("Recording contains no frames.")

    if args.dry_run:
        dry_run(data, normal_length_m=args.normal_length, template_axis_length_m=args.template_axis_length, depth_max_m=args.depth_max_m)
        return 0

    rr = _import_rerun()
    init_rerun(rr, args)
    send_blueprint(rr, include_images=has_image_streams(data), include_tactile=has_tactile_stream(data))

    for idx in range(frame_count):
        log_frame(
            rr,
            data,
            idx,
            normal_length_m=args.normal_length,
            template_axis_length_m=args.template_axis_length,
            depth_max_m=args.depth_max_m,
            include_coordinate_labels=not args.no_point_coordinate_labels,
            point_coordinate_precision=args.point_coordinate_precision,
        )
        print(frame_status_text(data, idx), flush=True)

    if args.save_rrd:
        print(f"[INFO] Saved Rerun recording to {args.save_rrd}")
    else:
        print("[INFO] Logged all frames to Rerun. Use the frame timeline in the viewer to scrub/play.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
