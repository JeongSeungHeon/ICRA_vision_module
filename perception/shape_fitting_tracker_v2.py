"""Template-based point-cloud fitting with frozen orientation and translation-only ICP."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover - optional dependency
    o3d = None
    _OPEN3D_IMPORT_ERROR = exc
else:  # pragma: no cover - optional dependency
    _OPEN3D_IMPORT_ERROR = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None

from object_pt_extraction.pointcloud_utils import voxel_downsample_point_cloud
from system.shared_state import HEIGHT_AXIS_X, HEIGHT_AXIS_Y, HEIGHT_AXIS_Z, MergedObjectState

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
REPO_ROOT = Path(__file__).resolve().parents[1]
_HEIGHT_AXIS_TO_INDEX = {
    HEIGHT_AXIS_X: 0,
    HEIGHT_AXIS_Y: 1,
    HEIGHT_AXIS_Z: 2,
    "x": 0,
    "y": 1,
    "z": 2,
}
SCALE_MODE_UNIFORM = "uniform"
SCALE_MODE_AXIS_XYZ = "axis_xyz"
VALID_SCALE_MODES = {SCALE_MODE_UNIFORM, SCALE_MODE_AXIS_XYZ}
TEMPLATE_PROFILE_BINS = 40
TEMPLATE_PROFILE_SMOOTHING_BINS = 5
TOP_WIDTH_FRACTION_FOR_REFERENCE = 0.15
BOWL_WIDTH_RATIO_THRESHOLD = 0.60
CONSECUTIVE_BINS_REQUIRED = 2


def _resolve_path(path_like: str | Path, *, base_dir: str | Path | None = None) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    search_roots = [Path.cwd(), REPO_ROOT]
    if base_dir is not None:
        base_dir_path = Path(base_dir).expanduser().resolve()
        search_roots.extend([base_dir_path, base_dir_path.parent])

    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def _limit_point_count(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if max_points <= 0 or len(points) <= max_points:
        return points
    sample_indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[sample_indices]


def _points_to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape((-1, 3)))
    return pcd


def _compute_robust_extent(points: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return np.zeros((3,), dtype=np.float64)
    lower = np.percentile(points, low_q, axis=0)
    upper = np.percentile(points, high_q, axis=0)
    return (upper - lower).astype(np.float64)


def _build_oriented_bbox(points: np.ndarray) -> o3d.geometry.OrientedBoundingBox | None:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return None

    if len(points) < 4:
        extent = np.maximum(_compute_robust_extent(points, 0.0, 100.0), 1e-6)
        return o3d.geometry.OrientedBoundingBox(np.mean(points, axis=0), np.eye(3), extent)

    pcd = _points_to_point_cloud(points)
    try:
        return pcd.get_oriented_bounding_box()
    except RuntimeError:
        extent = np.maximum(_compute_robust_extent(points, 0.0, 100.0), 1e-6)
        return o3d.geometry.OrientedBoundingBox(np.mean(points, axis=0), np.eye(3), extent)


def _compute_oriented_robust_extent(
    points: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray | None,
    low_q: float,
    high_q: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return np.zeros((3,), dtype=np.float64)

    rotation = np.asarray(rotation, dtype=np.float64).reshape((3, 3))
    center = np.mean(points, axis=0) if center is None else np.asarray(center, dtype=np.float64).reshape(3)
    points_local = (points - center.reshape(1, 3)) @ rotation
    lower = np.percentile(points_local, low_q, axis=0)
    upper = np.percentile(points_local, high_q, axis=0)
    return (upper - lower).astype(np.float64)


def _estimate_uniform_scale(source_extent: np.ndarray, target_extent: np.ndarray, min_scale: float, max_scale: float) -> float:
    source_extent = np.sort(np.asarray(source_extent, dtype=np.float64).reshape(3))
    target_extent = np.sort(np.asarray(target_extent, dtype=np.float64).reshape(3))
    valid_axes = np.logical_and(source_extent > 1e-6, target_extent > 1e-6)
    if not np.any(valid_axes):
        return 1.0

    raw_axis_scales = target_extent[valid_axes] / source_extent[valid_axes]
    uniform_scale = float(np.median(raw_axis_scales))
    return float(np.clip(uniform_scale, min_scale, max_scale))


def _estimate_axis_scale(
    source_extent: np.ndarray,
    target_extent: np.ndarray,
    min_scale: float,
    max_scale: float,
    *,
    fallback_scale: float = 1.0,
) -> np.ndarray:
    source_extent = np.asarray(source_extent, dtype=np.float64).reshape(3)
    target_extent = np.asarray(target_extent, dtype=np.float64).reshape(3)
    fallback_scale = float(np.clip(float(fallback_scale), min_scale, max_scale))
    scale_xyz = np.full((3,), fallback_scale, dtype=np.float64)

    finite_source = np.isfinite(source_extent)
    finite_target = np.isfinite(target_extent)
    source_order = np.argsort(np.where(finite_source, source_extent, np.inf))
    target_order = np.argsort(np.where(finite_target, target_extent, np.inf))

    for source_axis, target_axis in zip(source_order, target_order):
        source_value = float(source_extent[source_axis])
        target_value = float(target_extent[target_axis])
        if source_value > 1e-6 and target_value > 1e-6 and np.isfinite(source_value) and np.isfinite(target_value):
            scale_xyz[int(source_axis)] = target_value / source_value

    return np.clip(scale_xyz, min_scale, max_scale).astype(np.float64)


def _apply_similarity_pose(
    points: np.ndarray,
    rotation: np.ndarray,
    target_center: np.ndarray,
    *,
    uniform_scale: float | None = None,
    scale_xyz: np.ndarray | tuple[float, float, float] | None = None,
    scale_basis: np.ndarray | None = None,
    scale_center: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return points

    if scale_xyz is None:
        scale_value = 1.0 if uniform_scale is None else float(uniform_scale)
        centroid = np.mean(points, axis=0, keepdims=True)
        scaled = (points - centroid) * scale_value
    else:
        scale_arr = np.asarray(scale_xyz, dtype=np.float64).reshape(3)
        basis = np.eye(3, dtype=np.float64) if scale_basis is None else np.asarray(scale_basis, dtype=np.float64).reshape((3, 3))
        center = np.mean(points, axis=0) if scale_center is None else np.asarray(scale_center, dtype=np.float64).reshape(3)
        points_local = (points - center.reshape(1, 3)) @ basis
        scaled = (points_local * scale_arr.reshape(1, 3)) @ basis.T
        scaled = scaled - np.mean(scaled, axis=0, keepdims=True)
    rotated = scaled @ np.asarray(rotation, dtype=np.float64).reshape((3, 3)).T
    return rotated + np.asarray(target_center, dtype=np.float64).reshape((1, 3))


def _z_axis_rotation_matrix(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(float(angle_deg))
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return np.asarray(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return points

    transform = np.asarray(transform, dtype=np.float64).reshape((4, 4))
    homogeneous = np.ones((len(points), 4), dtype=np.float64)
    homogeneous[:, :3] = points
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def _filter_points_by_local_height(
    points: np.ndarray,
    *,
    height_axis_index: int,
    remove_top_fraction: float,
    min_points_after_crop: int,
    min_height_extent_m: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | None]]:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    debug = {
        "crop_enabled": False,
        "cropped_point_count": int(len(points)),
        "crop_threshold_z": None,
        "height_extent_m": 0.0,
    }
    if len(points) == 0:
        return points, debug

    axis_index = int(height_axis_index)
    if axis_index < 0 or axis_index >= points.shape[1]:
        return points, debug

    remove_fraction = float(remove_top_fraction)
    if remove_fraction <= 0.0 or remove_fraction >= 1.0:
        return points, debug

    axis_values = points[:, axis_index]
    axis_min = float(np.min(axis_values))
    axis_max = float(np.max(axis_values))
    height_extent = axis_max - axis_min
    debug["height_extent_m"] = height_extent
    if height_extent < float(min_height_extent_m):
        return points, debug

    crop_threshold = axis_min + (1.0 - remove_fraction) * height_extent
    filtered_points = points[axis_values <= crop_threshold]
    if len(filtered_points) < int(min_points_after_crop):
        return points, debug

    debug["crop_enabled"] = True
    debug["cropped_point_count"] = int(len(filtered_points))
    debug["crop_threshold_z"] = crop_threshold
    return filtered_points, debug


def _subsample_points_for_icp(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if max_points <= 0 or len(points) <= max_points:
        return points
    sample_indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[sample_indices]


def _query_nearest_neighbors_batched(query_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query_points = np.asarray(query_points, dtype=np.float64).reshape((-1, 3))
    target_points = np.asarray(target_points, dtype=np.float64).reshape((-1, 3))
    if len(query_points) == 0 or len(target_points) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)

    if cKDTree is not None:
        tree = cKDTree(target_points)
        distances, indices = tree.query(query_points, k=1)
        return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)

    target_tree = o3d.geometry.KDTreeFlann(_points_to_point_cloud(target_points))
    indices = np.empty((len(query_points),), dtype=np.int64)
    distances = np.empty((len(query_points),), dtype=np.float64)
    for point_index, point in enumerate(query_points):
        _, nearest_indices, distance2 = target_tree.search_knn_vector_3d(point, 1)
        if nearest_indices:
            indices[point_index] = int(nearest_indices[0])
            distances[point_index] = float(np.sqrt(distance2[0]))
        else:
            indices[point_index] = 0
            distances[point_index] = float("inf")
    return distances, indices


def _estimate_xy_max_diameter(points_xy: np.ndarray, max_points: int = 128) -> float | None:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2))
    if len(points_xy) == 0:
        return None
    if len(points_xy) > max_points:
        sample_indices = np.linspace(0, len(points_xy) - 1, max_points, dtype=np.int32)
        points_xy = points_xy[sample_indices]
    if len(points_xy) == 1:
        return 0.0
    deltas = points_xy[:, None, :] - points_xy[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    return float(np.max(distances))


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or window_size <= 1:
        return values.copy()
    pad = max(int(window_size) // 2, 0)
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones((int(window_size),), dtype=np.float64) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def _estimate_template_bowl_height_fraction(
    canonical_points: np.ndarray,
    *,
    profile_bins: int = TEMPLATE_PROFILE_BINS,
    smoothing_bins: int = TEMPLATE_PROFILE_SMOOTHING_BINS,
    top_width_fraction: float = TOP_WIDTH_FRACTION_FOR_REFERENCE,
    bowl_width_ratio_threshold: float = BOWL_WIDTH_RATIO_THRESHOLD,
    consecutive_bins_required: int = CONSECUTIVE_BINS_REQUIRED,
) -> float | None:
    points = np.asarray(canonical_points, dtype=np.float32).reshape((-1, 3))
    if len(points) < 8:
        return None

    z_values = points[:, 2]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_extent = z_max - z_min
    if not np.isfinite(z_extent) or z_extent <= 1e-9:
        return None

    profile_bins = max(int(profile_bins), 4)
    edges = np.linspace(z_min, z_max, profile_bins + 1, dtype=np.float64)
    widths = np.full((profile_bins,), np.nan, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for bin_index in range(profile_bins):
        if bin_index == profile_bins - 1:
            mask = np.logical_and(z_values >= edges[bin_index], z_values <= edges[bin_index + 1])
        else:
            mask = np.logical_and(z_values >= edges[bin_index], z_values < edges[bin_index + 1])
        diameter = _estimate_xy_max_diameter(points[mask, :2])
        if diameter is not None:
            widths[bin_index] = diameter

    finite_mask = np.isfinite(widths)
    if np.count_nonzero(finite_mask) < max(consecutive_bins_required, 3):
        return None

    valid_indices = np.flatnonzero(finite_mask)
    widths = np.interp(np.arange(profile_bins), valid_indices, widths[finite_mask])
    smoothed_widths = _moving_average(widths, max(int(smoothing_bins), 1))

    top_bin_count = max(int(np.ceil(profile_bins * float(top_width_fraction))), 1)
    top_reference_width = float(np.median(smoothed_widths[-top_bin_count:]))
    if not np.isfinite(top_reference_width) or top_reference_width <= 1e-9:
        return None

    width_threshold = float(bowl_width_ratio_threshold) * top_reference_width
    consecutive = max(int(consecutive_bins_required), 1)
    above_threshold = smoothed_widths >= width_threshold
    bowl_start_index = None
    for start_index in range(0, profile_bins - consecutive + 1):
        if np.all(above_threshold[start_index : start_index + consecutive]):
            bowl_start_index = start_index
            break
    if bowl_start_index is None:
        return None

    bowl_start_z = float(centers[bowl_start_index])
    bowl_height_fraction = (z_max - bowl_start_z) / z_extent
    if not np.isfinite(bowl_height_fraction):
        return None
    return float(np.clip(bowl_height_fraction, 0.0, 1.0))


@dataclass(frozen=True)
class ShapeTemplateModel:
    label: str
    template_id: str
    asset_path: Path
    unit_scale_m: float
    scale_mode: str
    canonical_points: np.ndarray
    source_extent_xyz: np.ndarray
    bowl_height_fraction: float | None
    z_rotation_enabled: bool
    z_rotation_min_deg: float
    z_rotation_max_deg: float
    z_rotation_step_deg: float


@dataclass
class ShapeFittingState:
    valid: bool
    label: str | None
    template_id: str | None
    fitted_points_base: np.ndarray
    centroid_base: tuple[float, float, float] | None
    scale: float | None
    scale_xyz: tuple[float, float, float] | None
    scale_mode: str
    bowl_height_fraction: float | None
    initialized: bool
    reason: str


@dataclass
class ShapeFittingDebug:
    label: str | None
    template_id: str | None
    raw_point_count: int
    cluster_point_count: int
    output_point_count: int
    scale_buffer_count: int
    initialized: bool
    scale: float | None
    scale_xyz: tuple[float, float, float] | None
    scale_mode: str
    reason: str
    tracking_mode: str
    icp_time_ms: float | None
    icp_fps: float | None
    icp_fitness: float | None
    icp_rmse: float | None
    icp_translation_m: float | None
    icp_source_points: int
    icp_target_points: int
    icp_iterations_used: int
    z_rotation_deg: float | None


class ShapeFittingTracker:
    """Fits a canonical template cloud to the merged object cloud using translation-only ICP."""

    def __init__(self, config: dict[str, Any], *, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        if o3d is None:
            raise RuntimeError(
                "open3d is required for ShapeFittingTracker. "
                f"Import failed with: {_OPEN3D_IMPORT_ERROR}"
            )

        self._config_path = _resolve_path(config_path)
        perception_cfg = config.get("perception", {})
        fitting_cfg = perception_cfg.get("shape_fitting", {})
        pose_tracking_cfg = perception_cfg.get("pose_tracking", {})

        self.enabled = bool(fitting_cfg.get("enabled", True))
        template_cfg = fitting_cfg.get("template_library") or pose_tracking_cfg.get("template_library") or {}
        if not template_cfg:
            raise ValueError("No template library configured for perception.shape_fitting.")

        cluster_cfg = fitting_cfg.get("cluster", {})
        scale_cfg = fitting_cfg.get("scale_init", {})
        tracking_cfg = fitting_cfg.get("tracking", {})
        downsample_cfg = fitting_cfg.get("downsample", {})
        icp_cfg = fitting_cfg.get("icp", {})
        crop_cfg = icp_cfg.get("crop", {})

        self.dbscan_eps_m = float(cluster_cfg.get("dbscan_eps_m", 0.02))
        self.dbscan_min_points = int(cluster_cfg.get("dbscan_min_points", 10))
        self.max_cluster_jump_m = float(cluster_cfg.get("max_cluster_jump_m", 0.08))

        self.scale_init_valid_frames = max(1, int(scale_cfg.get("stable_frames", 8)))
        self.scale_low_q = float(scale_cfg.get("percentile_low", 5.0))
        self.scale_high_q = float(scale_cfg.get("percentile_high", 95.0))
        self.min_scale = float(scale_cfg.get("min_scale", 0.5))
        self.max_scale = float(scale_cfg.get("max_scale", 1.8))
        self._default_scale_mode = self._normalize_scale_mode(scale_cfg.get("mode", SCALE_MODE_UNIFORM))

        self.output_voxel_size_m = float(downsample_cfg.get("voxel_size_m", 0.004))
        self.output_max_points = int(downsample_cfg.get("max_points", 6000))
        self.min_cluster_extent_m = float(tracking_cfg.get("min_cluster_extent_m", 1e-5))

        self.icp_max_points = int(icp_cfg.get("max_points", 2000))
        self.icp_distance_threshold_m = float(icp_cfg.get("distance_threshold_m", 0.09))
        self.icp_max_iterations = int(icp_cfg.get("max_iterations", 12))
        self.icp_min_fitness = float(icp_cfg.get("min_fitness", 0.02))
        self.icp_translation_tolerance_m = float(icp_cfg.get("translation_tolerance_m", 1e-5))
        self.icp_max_centroid_jump_m = float(icp_cfg.get("max_centroid_jump_m", self.max_cluster_jump_m))

        self.icp_crop_enabled = bool(crop_cfg.get("enabled", True))
        self.icp_crop_height_axis_index = int(crop_cfg.get("height_axis_index", 2))
        self.icp_crop_target_top_fraction = crop_cfg.get("target_top_fraction", 0.10)
        self.icp_crop_source_top_fraction = crop_cfg.get("source_top_fraction", None)
        self.icp_crop_min_points_after_crop = int(crop_cfg.get("min_points_after_crop", 80))
        self.icp_crop_min_height_extent_m = float(crop_cfg.get("min_height_extent_m", 0.01))

        self._templates = self._load_template_library(template_cfg)
        self._active_template: ShapeTemplateModel | None = None
        self._tracked_cluster_centroid: np.ndarray | None = None
        self._extent_buffer: list[np.ndarray] = []
        self._frozen_scale: float | None = None
        self._frozen_scale_xyz: np.ndarray | None = None
        self._frozen_scale_mode = SCALE_MODE_UNIFORM
        self._frozen_scale_basis: np.ndarray | None = None
        self._frozen_scale_center: np.ndarray | None = None
        self._frozen_rotation = np.eye(3, dtype=np.float64)
        self._frozen_z_rotation_deg: float | None = None
        self._current_template_points = np.empty((0, 3), dtype=np.float32)
        self._initialized = False

        self.last_state = ShapeFittingState(
            valid=False,
            label=None,
            template_id=None,
            fitted_points_base=np.empty((0, 3), dtype=np.float32),
            centroid_base=None,
            scale=None,
            scale_xyz=None,
            scale_mode=SCALE_MODE_UNIFORM,
            bowl_height_fraction=None,
            initialized=False,
            reason="uninitialized",
        )
        self.last_debug = ShapeFittingDebug(
            label=None,
            template_id=None,
            raw_point_count=0,
            cluster_point_count=0,
            output_point_count=0,
            scale_buffer_count=0,
            initialized=False,
            scale=None,
            scale_xyz=None,
            scale_mode=SCALE_MODE_UNIFORM,
            reason="uninitialized",
            tracking_mode="uninitialized",
            icp_time_ms=None,
            icp_fps=None,
            icp_fitness=None,
            icp_rmse=None,
            icp_translation_m=None,
            icp_source_points=0,
            icp_target_points=0,
            icp_iterations_used=0,
            z_rotation_deg=None,
        )

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ShapeFittingTracker":
        resolved_config_path = _resolve_path(config_path)
        with open(resolved_config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config, config_path=resolved_config_path)

    def reset(self) -> None:
        self._tracked_cluster_centroid = None
        self._extent_buffer = []
        self._frozen_scale = None
        self._frozen_scale_xyz = None
        self._frozen_scale_mode = SCALE_MODE_UNIFORM
        self._frozen_scale_basis = None
        self._frozen_scale_center = None
        self._frozen_rotation = np.eye(3, dtype=np.float64)
        self._frozen_z_rotation_deg = None
        self._current_template_points = np.empty((0, 3), dtype=np.float32)
        self._initialized = False

    def process(self, merged_object: MergedObjectState) -> ShapeFittingState:
        if not self.enabled:
            return self._make_state(valid=False, template=None, fitted_points=None, centroid=None, reason="disabled")

        raw_points = np.asarray(getattr(merged_object, "merged_points_base", []), dtype=np.float32).reshape((-1, 3))
        label = getattr(merged_object, "label", None)
        template = self._resolve_template(label)
        if template is None:
            self._set_debug(
                label=label,
                template_id=None,
                raw_point_count=len(raw_points),
                cluster_point_count=0,
                output_point_count=0,
                reason="template_not_found",
                tracking_mode="uninitialized",
            )
            return self._make_state(valid=False, template=None, fitted_points=None, centroid=None, reason="template_not_found")

        if self._active_template is None or self._active_template.template_id != template.template_id:
            self._active_template = template
            self.reset()

        if not merged_object.valid or len(raw_points) == 0:
            self._set_debug(
                label=template.label,
                template_id=template.template_id,
                raw_point_count=len(raw_points),
                cluster_point_count=0,
                output_point_count=0,
                reason="no_merged_object",
                tracking_mode="hold",
            )
            return self._make_state(valid=False, template=template, fitted_points=None, centroid=None, reason="no_merged_object")

        filtered_points, cluster_centroid = self._filter_single_cluster(raw_points, self._tracked_cluster_centroid)
        if len(filtered_points) == 0 or cluster_centroid is None:
            self._tracked_cluster_centroid = None
            self._set_debug(
                label=template.label,
                template_id=template.template_id,
                raw_point_count=len(raw_points),
                cluster_point_count=0,
                output_point_count=0,
                reason="no_valid_cluster",
                tracking_mode="hold",
            )
            return self._make_state(valid=False, template=template, fitted_points=None, centroid=None, reason="no_valid_cluster")

        self._tracked_cluster_centroid = cluster_centroid.astype(np.float32)
        filtered_points = np.asarray(filtered_points, dtype=np.float64).reshape((-1, 3))

        if not self._initialized:
            target_obb = _build_oriented_bbox(filtered_points)
            if target_obb is not None:
                robust_extent = _compute_oriented_robust_extent(
                    filtered_points,
                    target_obb.R,
                    target_obb.center,
                    self.scale_low_q,
                    self.scale_high_q,
                )
                if np.all(robust_extent > self.min_cluster_extent_m):
                    self._extent_buffer.append(robust_extent)

            if len(self._extent_buffer) < self.scale_init_valid_frames:
                reason = f"initializing_scale_{len(self._extent_buffer)}/{self.scale_init_valid_frames}"
                self._set_debug(
                    label=template.label,
                    template_id=template.template_id,
                    raw_point_count=len(raw_points),
                    cluster_point_count=len(filtered_points),
                    output_point_count=0,
                    reason=reason,
                    tracking_mode="init_pending",
                )
                return self._make_state(valid=False, template=template, fitted_points=None, centroid=None, reason=reason)

            initialized_points = self._initialize_template(template, filtered_points)
            output_points = self._downsample_output(initialized_points)
            output_centroid = None if len(output_points) == 0 else np.mean(output_points, axis=0)
            state = self._make_state(
                valid=len(output_points) > 0,
                template=template,
                fitted_points=output_points,
                centroid=output_centroid,
                reason="ok" if len(output_points) > 0 else "empty_fitted_template",
            )
            self._set_debug(
                label=template.label,
                template_id=template.template_id,
                raw_point_count=len(raw_points),
                cluster_point_count=len(filtered_points),
                output_point_count=len(output_points),
                reason=state.reason,
                tracking_mode="init",
            )
            return state

        assert self._frozen_scale is not None
        assert self._frozen_scale_xyz is not None
        current_template_points = np.asarray(self._current_template_points, dtype=np.float64).reshape((-1, 3))
        current_centroid = np.mean(current_template_points, axis=0)
        scaled_points = _apply_similarity_pose(
            template.canonical_points,
            rotation=self._frozen_rotation,
            target_center=current_centroid,
            scale_xyz=self._frozen_scale_xyz,
            scale_basis=self._frozen_scale_basis,
            scale_center=self._frozen_scale_center,
        )

        icp_start = time.perf_counter()
        icp_transform, icp_fitness, icp_rmse, icp_translation_m, icp_source_points, icp_target_points, icp_iterations_used = self._run_translation_only_icp(
            scaled_points,
            filtered_points,
            height_axis_index=self._height_axis_index_for_object(merged_object),
        )
        icp_time_ms = (time.perf_counter() - icp_start) * 1000.0

        proposed_points = _transform_points(scaled_points, icp_transform)
        current_output = self._downsample_output(current_template_points)
        proposed_output = self._downsample_output(proposed_points)
        centroid_jump = float(
            np.linalg.norm(np.mean(proposed_points, axis=0) - np.mean(current_template_points, axis=0))
        )
        quality_ok = (
            np.isfinite(icp_fitness) and icp_fitness >= self.icp_min_fitness
        ) or (
            np.isfinite(icp_rmse) and icp_rmse <= (2.5 * self.icp_distance_threshold_m)
        )

        if centroid_jump <= self.icp_max_centroid_jump_m and (quality_ok or icp_translation_m > 0.0):
            self._current_template_points = proposed_points.astype(np.float32)
            output_points = proposed_output
            output_centroid = None if len(output_points) == 0 else np.mean(output_points, axis=0)
            reason = "ok" if len(output_points) > 0 else "empty_fitted_template"
            tracking_mode = "translation_icp"
        else:
            output_points = current_output
            output_centroid = None if len(output_points) == 0 else np.mean(output_points, axis=0)
            reason = f"hold_fitness_{icp_fitness:.3f}_jump_{centroid_jump:.3f}"
            tracking_mode = "hold"

        state = self._make_state(
            valid=len(output_points) > 0,
            template=template,
            fitted_points=output_points,
            centroid=output_centroid,
            reason=reason,
        )
        self._set_debug(
            label=template.label,
            template_id=template.template_id,
            raw_point_count=len(raw_points),
            cluster_point_count=len(filtered_points),
            output_point_count=len(output_points),
            reason=state.reason,
            tracking_mode=tracking_mode,
            icp_time_ms=icp_time_ms,
            icp_fitness=icp_fitness,
            icp_rmse=icp_rmse,
            icp_translation_m=icp_translation_m,
            icp_source_points=icp_source_points,
            icp_target_points=icp_target_points,
            icp_iterations_used=icp_iterations_used,
        )
        return state

    def _initialize_template(self, template: ShapeTemplateModel, target_points: np.ndarray) -> np.ndarray:
        median_target_extent = np.median(np.asarray(self._extent_buffer, dtype=np.float64), axis=0)
        template_obb = _build_oriented_bbox(template.canonical_points)
        scale_mode = self._scale_mode_for_template(template)
        if template_obb is None:
            uniform_scale = 1.0
            scale_xyz = np.ones((3,), dtype=np.float64)
            scale_basis = None
            scale_center = None
        else:
            template_extent = _compute_oriented_robust_extent(
                template.canonical_points,
                template_obb.R,
                template_obb.center,
                0.0,
                100.0,
            )
            uniform_scale = _estimate_uniform_scale(template_extent, median_target_extent, self.min_scale, self.max_scale)
            if scale_mode == SCALE_MODE_AXIS_XYZ:
                scale_xyz = _estimate_axis_scale(
                    template_extent,
                    median_target_extent,
                    self.min_scale,
                    self.max_scale,
                    fallback_scale=uniform_scale,
                )
                scale_basis = np.asarray(template_obb.R, dtype=np.float64).reshape((3, 3))
                scale_center = np.asarray(template_obb.center, dtype=np.float64).reshape(3)
            else:
                scale_xyz = np.full((3,), uniform_scale, dtype=np.float64)
                scale_basis = None
                scale_center = None

        initial_center = np.mean(target_points, axis=0)
        candidate_degrees = self._z_rotation_candidate_degrees(template)
        best_score: tuple[float, float, float] | None = None
        best_rotation = np.eye(3, dtype=np.float64)
        best_z_rotation_deg = 0.0
        best_initialized_points = np.empty((0, 3), dtype=np.float64)
        best_icp_time_ms = 0.0
        best_icp_fitness = 0.0
        best_icp_rmse = float("inf")
        best_icp_translation_m = 0.0
        best_icp_source_points = 0
        best_icp_target_points = 0
        best_icp_iterations_used = 0

        for z_rotation_deg in candidate_degrees:
            candidate_rotation = _z_axis_rotation_matrix(z_rotation_deg)
            candidate_points = _apply_similarity_pose(
                template.canonical_points,
                rotation=candidate_rotation,
                target_center=initial_center,
                scale_xyz=scale_xyz,
                scale_basis=scale_basis,
                scale_center=scale_center,
            )

            icp_start = time.perf_counter()
            (
                icp_transform,
                icp_fitness,
                icp_rmse,
                icp_translation_m,
                icp_source_points,
                icp_target_points,
                icp_iterations_used,
            ) = self._run_translation_only_icp(
                candidate_points,
                target_points,
                height_axis_index=self.icp_crop_height_axis_index,
            )
            icp_time_ms = (time.perf_counter() - icp_start) * 1000.0
            initialized_points = _transform_points(candidate_points, icp_transform)
            finite_rmse = icp_rmse if np.isfinite(icp_rmse) else float("inf")
            score = (float(icp_fitness), -float(finite_rmse), -float(icp_translation_m))
            if best_score is None or score > best_score:
                best_score = score
                best_rotation = candidate_rotation
                best_z_rotation_deg = float(z_rotation_deg)
                best_initialized_points = initialized_points
                best_icp_time_ms = icp_time_ms
                best_icp_fitness = icp_fitness
                best_icp_rmse = icp_rmse
                best_icp_translation_m = icp_translation_m
                best_icp_source_points = icp_source_points
                best_icp_target_points = icp_target_points
                best_icp_iterations_used = icp_iterations_used

        self._frozen_scale = float(np.median(scale_xyz))
        self._frozen_scale_xyz = np.asarray(scale_xyz, dtype=np.float64).reshape(3)
        self._frozen_scale_mode = scale_mode
        self._frozen_scale_basis = None if scale_basis is None else np.asarray(scale_basis, dtype=np.float64).reshape((3, 3))
        self._frozen_scale_center = None if scale_center is None else np.asarray(scale_center, dtype=np.float64).reshape(3)
        self._frozen_rotation = best_rotation
        self._frozen_z_rotation_deg = best_z_rotation_deg if template.z_rotation_enabled else None
        self._current_template_points = best_initialized_points.astype(np.float32)
        self._initialized = True
        self._set_debug(
            label=template.label,
            template_id=template.template_id,
            raw_point_count=0,
            cluster_point_count=len(target_points),
            output_point_count=len(best_initialized_points),
            reason="ok",
            tracking_mode="init",
            icp_time_ms=best_icp_time_ms,
            icp_fitness=best_icp_fitness,
            icp_rmse=best_icp_rmse,
            icp_translation_m=best_icp_translation_m,
            icp_source_points=best_icp_source_points,
            icp_target_points=best_icp_target_points,
            icp_iterations_used=best_icp_iterations_used,
            z_rotation_deg=self._frozen_z_rotation_deg,
        )
        return best_initialized_points.astype(np.float32)

    def _z_rotation_candidate_degrees(self, template: ShapeTemplateModel) -> list[float]:
        if not template.z_rotation_enabled:
            return [0.0]

        min_deg = float(template.z_rotation_min_deg)
        max_deg = float(template.z_rotation_max_deg)
        if not np.isfinite(min_deg) or not np.isfinite(max_deg):
            return [0.0]
        if min_deg > max_deg:
            min_deg, max_deg = max_deg, min_deg

        step_deg = abs(float(template.z_rotation_step_deg))
        if not np.isfinite(step_deg) or step_deg <= 0.0:
            step_deg = 5.0

        candidates = list(np.arange(min_deg, max_deg + (0.5 * step_deg), step_deg, dtype=np.float64))
        candidates = [float(np.clip(candidate, min_deg, max_deg)) for candidate in candidates]
        candidates.append(max_deg)
        if min_deg <= 0.0 <= max_deg and not any(abs(candidate) <= 1e-9 for candidate in candidates):
            candidates.append(0.0)
        return sorted(set(round(candidate, 9) for candidate in candidates))

    def _run_translation_only_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        *,
        height_axis_index: int,
    ) -> tuple[np.ndarray, float, float, float, int, int, int]:
        source_fit_points = np.asarray(source_points, dtype=np.float64).reshape((-1, 3)).copy()
        target_fit_points = np.asarray(target_points, dtype=np.float64).reshape((-1, 3)).copy()
        if len(source_fit_points) == 0 or len(target_fit_points) == 0:
            return np.eye(4, dtype=np.float64), 0.0, float("inf"), 0.0, 0, 0, 0

        if self.icp_crop_enabled:
            if self.icp_crop_source_top_fraction is not None:
                source_fit_points, _ = _filter_points_by_local_height(
                    source_fit_points,
                    height_axis_index=height_axis_index,
                    remove_top_fraction=float(self.icp_crop_source_top_fraction),
                    min_points_after_crop=self.icp_crop_min_points_after_crop,
                    min_height_extent_m=self.icp_crop_min_height_extent_m,
                )
            if self.icp_crop_target_top_fraction is not None:
                target_fit_points, _ = _filter_points_by_local_height(
                    target_fit_points,
                    height_axis_index=height_axis_index,
                    remove_top_fraction=float(self.icp_crop_target_top_fraction),
                    min_points_after_crop=self.icp_crop_min_points_after_crop,
                    min_height_extent_m=self.icp_crop_min_height_extent_m,
                )

        source_fit_points = _subsample_points_for_icp(source_fit_points, self.icp_max_points)
        target_fit_points = _subsample_points_for_icp(target_fit_points, self.icp_max_points)
        if len(source_fit_points) == 0 or len(target_fit_points) == 0:
            return np.eye(4, dtype=np.float64), 0.0, float("inf"), 0.0, len(source_fit_points), len(target_fit_points), 0

        total_transform = np.eye(4, dtype=np.float64)
        total_transform[:3, 3] = np.mean(target_fit_points, axis=0) - np.mean(source_fit_points, axis=0)
        transformed_source = source_fit_points + total_transform[:3, 3].reshape(1, 3)

        last_fitness = 0.0
        last_rmse = float("inf")
        iterations_used = 0

        for iteration_index in range(self.icp_max_iterations):
            nearest_distances, nearest_indices = _query_nearest_neighbors_batched(transformed_source, target_fit_points)
            if len(nearest_indices) == 0:
                return (
                    total_transform,
                    0.0,
                    float("inf"),
                    float(np.linalg.norm(total_transform[:3, 3])),
                    len(source_fit_points),
                    len(target_fit_points),
                    iterations_used,
                )

            nearest_targets = target_fit_points[nearest_indices]
            translation_delta = np.mean(nearest_targets - transformed_source, axis=0)

            delta_transform = np.eye(4, dtype=np.float64)
            delta_transform[:3, 3] = translation_delta
            total_transform = delta_transform @ total_transform
            transformed_source = transformed_source + translation_delta.reshape(1, 3)

            inlier_mask = nearest_distances <= self.icp_distance_threshold_m
            last_fitness = float(np.count_nonzero(inlier_mask) / max(1, len(nearest_distances)))
            if np.any(inlier_mask):
                last_rmse = float(np.sqrt(np.mean(np.square(nearest_distances[inlier_mask]))))
            else:
                last_rmse = float(np.sqrt(np.mean(np.square(nearest_distances))))
            iterations_used = iteration_index + 1
            if np.linalg.norm(translation_delta) <= self.icp_translation_tolerance_m:
                break

        translation_distance_m = float(np.linalg.norm(total_transform[:3, 3]))
        return (
            total_transform,
            last_fitness,
            last_rmse,
            translation_distance_m,
            len(source_fit_points),
            len(target_fit_points),
            iterations_used,
        )

    def _downsample_output(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        dummy_colors = np.zeros((len(points), 3), dtype=np.uint8)
        if self.output_voxel_size_m > 0.0:
            points, _ = voxel_downsample_point_cloud(points, dummy_colors, voxel_size_m=self.output_voxel_size_m)
        points = _limit_point_count(points, self.output_max_points)
        return np.asarray(points, dtype=np.float32)

    def _filter_single_cluster(
        self,
        points: np.ndarray,
        prev_centroid: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32), None

        pcd = _points_to_point_cloud(points)
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.dbscan_eps_m,
                min_points=self.dbscan_min_points,
                print_progress=False,
            )
        )
        valid_cluster_ids = np.unique(labels[labels >= 0])
        if len(valid_cluster_ids) == 0:
            return np.empty((0, 3), dtype=np.float32), None

        cluster_points = {cluster_id: points[labels == cluster_id] for cluster_id in valid_cluster_ids}
        cluster_counts = {cluster_id: len(cluster_pts) for cluster_id, cluster_pts in cluster_points.items()}
        cluster_centroids = {
            cluster_id: np.mean(cluster_pts, axis=0)
            for cluster_id, cluster_pts in cluster_points.items()
        }

        selected_cluster_id = max(cluster_counts, key=cluster_counts.get)
        if prev_centroid is not None:
            prev = np.asarray(prev_centroid, dtype=np.float64).reshape(3)
            closest_cluster_id = min(
                cluster_centroids,
                key=lambda cluster_id: np.linalg.norm(cluster_centroids[cluster_id] - prev),
            )
            closest_distance = float(np.linalg.norm(cluster_centroids[closest_cluster_id] - prev))
            if closest_distance <= self.max_cluster_jump_m:
                selected_cluster_id = closest_cluster_id

        selected_points = np.asarray(cluster_points[selected_cluster_id], dtype=np.float32)
        selected_centroid = np.asarray(cluster_centroids[selected_cluster_id], dtype=np.float32)
        return selected_points, selected_centroid

    def _height_axis_index_for_object(self, merged_object: MergedObjectState) -> int:
        axis_name = getattr(merged_object, "height_axis_name", None)
        return int(_HEIGHT_AXIS_TO_INDEX.get(str(axis_name).lower(), self.icp_crop_height_axis_index))

    @staticmethod
    def _normalize_scale_mode(value: Any) -> str:
        mode = str(value or SCALE_MODE_UNIFORM).strip().lower()
        return mode if mode in VALID_SCALE_MODES else SCALE_MODE_UNIFORM

    def _scale_mode_for_template(self, template: ShapeTemplateModel | None) -> str:
        if template is not None:
            return self._normalize_scale_mode(template.scale_mode)
        return self._normalize_scale_mode(getattr(self, "_default_scale_mode", SCALE_MODE_UNIFORM))

    def _debug_scale_mode(self) -> str:
        if self._initialized:
            return self._normalize_scale_mode(self._frozen_scale_mode)
        return self._scale_mode_for_template(self._active_template)

    def _resolve_template(self, label: str | None) -> ShapeTemplateModel | None:
        if not label:
            return None
        return self._templates.get(str(label))

    def _load_template_library(self, template_cfg: dict[str, Any]) -> dict[str, ShapeTemplateModel]:
        templates: dict[str, ShapeTemplateModel] = {}
        for label, entry in template_cfg.items():
            asset_path = _resolve_path(entry["asset_path"], base_dir=self._config_path.parent)
            canonical_points = np.load(asset_path).astype(np.float32) * float(entry.get("unit_scale_m", 1.0))
            normalized_label = str(label).strip().lower()
            bowl_height_fraction = None
            if normalized_label == "wine glass":
                bowl_height_fraction = _estimate_template_bowl_height_fraction(canonical_points)
            z_rotation_cfg = entry.get("z_rotation", {}) or {}
            if isinstance(z_rotation_cfg, bool):
                z_rotation_cfg = {"enabled": z_rotation_cfg}
            z_rotation_enabled = bool(z_rotation_cfg.get("enabled", False))
            z_rotation_min_deg = float(z_rotation_cfg.get("min_deg", 0.0))
            z_rotation_max_deg = float(z_rotation_cfg.get("max_deg", 0.0))
            z_rotation_step_deg = float(z_rotation_cfg.get("step_deg", 5.0))
            default_scale_mode = getattr(self, "_default_scale_mode", SCALE_MODE_UNIFORM)
            scale_mode = ShapeFittingTracker._normalize_scale_mode(entry.get("scale_mode", default_scale_mode))
            templates[str(label)] = ShapeTemplateModel(
                label=str(label),
                template_id=str(entry.get("template_id", label)),
                asset_path=asset_path,
                unit_scale_m=float(entry.get("unit_scale_m", 1.0)),
                scale_mode=scale_mode,
                canonical_points=canonical_points,
                source_extent_xyz=_compute_robust_extent(canonical_points, 0.0, 100.0),
                bowl_height_fraction=bowl_height_fraction,
                z_rotation_enabled=z_rotation_enabled,
                z_rotation_min_deg=z_rotation_min_deg,
                z_rotation_max_deg=z_rotation_max_deg,
                z_rotation_step_deg=z_rotation_step_deg,
            )
        return templates

    def _set_debug(
        self,
        *,
        label: str | None,
        template_id: str | None,
        raw_point_count: int,
        cluster_point_count: int,
        output_point_count: int,
        reason: str,
        tracking_mode: str,
        icp_time_ms: float | None = None,
        icp_fitness: float | None = None,
        icp_rmse: float | None = None,
        icp_translation_m: float | None = None,
        icp_source_points: int = 0,
        icp_target_points: int = 0,
        icp_iterations_used: int = 0,
        z_rotation_deg: float | None = None,
    ) -> None:
        icp_fps = None
        if icp_time_ms is not None and np.isfinite(icp_time_ms) and icp_time_ms > 0.0:
            icp_fps = 1000.0 / float(icp_time_ms)
        if z_rotation_deg is None and self._active_template is not None and self._active_template.z_rotation_enabled:
            z_rotation_deg = self._frozen_z_rotation_deg
        scale_xyz = None if self._frozen_scale_xyz is None else tuple(float(v) for v in self._frozen_scale_xyz)
        self.last_debug = ShapeFittingDebug(
            label=label,
            template_id=template_id,
            raw_point_count=int(raw_point_count),
            cluster_point_count=int(cluster_point_count),
            output_point_count=int(output_point_count),
            scale_buffer_count=len(self._extent_buffer),
            initialized=bool(self._initialized),
            scale=None if self._frozen_scale is None else float(self._frozen_scale),
            scale_xyz=scale_xyz,
            scale_mode=self._debug_scale_mode(),
            reason=str(reason),
            tracking_mode=str(tracking_mode),
            icp_time_ms=None if icp_time_ms is None else float(icp_time_ms),
            icp_fps=icp_fps,
            icp_fitness=None if icp_fitness is None else float(icp_fitness),
            icp_rmse=None if icp_rmse is None else float(icp_rmse),
            icp_translation_m=None if icp_translation_m is None else float(icp_translation_m),
            icp_source_points=int(icp_source_points),
            icp_target_points=int(icp_target_points),
            icp_iterations_used=int(icp_iterations_used),
            z_rotation_deg=None if z_rotation_deg is None else float(z_rotation_deg),
        )

    def _make_state(
        self,
        *,
        valid: bool,
        template: ShapeTemplateModel | None,
        fitted_points: np.ndarray | None,
        centroid: np.ndarray | tuple[float, float, float] | None,
        reason: str,
    ) -> ShapeFittingState:
        fitted_points_arr = (
            np.empty((0, 3), dtype=np.float32)
            if fitted_points is None
            else np.asarray(fitted_points, dtype=np.float32).reshape((-1, 3))
        )
        centroid_arr = None if centroid is None else np.asarray(centroid, dtype=np.float32).reshape(3)
        centroid_tuple = None if centroid_arr is None else tuple(float(v) for v in centroid_arr)
        scale_xyz = None if self._frozen_scale_xyz is None else tuple(float(v) for v in self._frozen_scale_xyz)
        state = ShapeFittingState(
            valid=bool(valid),
            label=None if template is None else template.label,
            template_id=None if template is None else template.template_id,
            fitted_points_base=fitted_points_arr,
            centroid_base=centroid_tuple,
            scale=None if self._frozen_scale is None else float(self._frozen_scale),
            scale_xyz=scale_xyz,
            scale_mode=self._scale_mode_for_template(template),
            bowl_height_fraction=None if template is None else template.bowl_height_fraction,
            initialized=bool(self._initialized),
            reason=str(reason),
        )
        self.last_state = state
        return state


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ShapeFittingDebug",
    "ShapeFittingState",
    "ShapeFittingTracker",
]
