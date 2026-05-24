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
from perception.silhouette_constraint import (
    SilhouetteObservation,
    SilhouetteScore,
    compute_silhouette_score,
)
from system.shared_state import HEIGHT_AXIS_X, HEIGHT_AXIS_Y, HEIGHT_AXIS_Z, MergedObjectState

# 설정 파일과 템플릿 자산 경로를 안정적으로 찾기 위한 기준 경로입니다.
DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
REPO_ROOT = Path(__file__).resolve().parents[1]

# shared_state에서 전달되는 높이 축 이름을 numpy 좌표 인덱스로 변환합니다.
_HEIGHT_AXIS_TO_INDEX = {
    HEIGHT_AXIS_X: 0,
    HEIGHT_AXIS_Y: 1,
    HEIGHT_AXIS_Z: 2,
    "x": 0,
    "y": 1,
    "z": 2,
}
# 템플릿 스케일링 모드: 전체 균일 배율 또는 템플릿 로컬 축별 배율입니다.
SCALE_MODE_UNIFORM = "uniform"
SCALE_MODE_AXIS_XYZ = "axis_xyz"
VALID_SCALE_MODES = {SCALE_MODE_UNIFORM, SCALE_MODE_AXIS_XYZ}

# 와인잔처럼 bowl 영역이 중요한 템플릿의 높이 비율을 추정할 때 쓰는 프로파일 파라미터입니다.
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


def _x_axis_rotation_matrix(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(float(angle_deg))
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle],
            [0.0, sin_angle, cos_angle],
        ],
        dtype=np.float64,
    )


def _y_axis_rotation_matrix(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(float(angle_deg))
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return np.asarray(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
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
    """설정 파일에서 로드한 하나의 canonical point-cloud 템플릿 정보."""

    # 템플릿 식별 및 원본 자산 정보입니다.
    label: str
    template_id: str
    asset_path: Path
    unit_scale_m: float
    scale_mode: str

    # canonical_points는 템플릿의 기준 좌표계 점군이며, source_extent_xyz는 초기 스케일 추정 기준입니다.
    canonical_points: np.ndarray
    source_extent_xyz: np.ndarray
    bowl_height_fraction: float | None

    # 초기 정렬 시 z축 회전 후보를 탐색할지와 탐색 범위를 정의합니다.
    z_rotation_enabled: bool
    z_rotation_min_deg: float
    z_rotation_max_deg: float
    z_rotation_step_deg: float
    z_rotation_coarse_to_fine_enabled: bool
    z_rotation_coarse_step_deg: float
    z_rotation_refine_radius_deg: float
    z_rotation_refine_step_deg: float
    axis_rotation_enabled: bool
    axis_roll_candidates_deg: tuple[float, ...]
    axis_pitch_candidates_deg: tuple[float, ...]


@dataclass
class ShapeFittingState:
    """외부 모듈에 전달되는 shape fitting 결과 상태."""

    # 현재 프레임에서 유효한 템플릿 fitting 결과가 있는지와 어떤 템플릿인지 나타냅니다.
    valid: bool
    label: str | None
    template_id: str | None

    # fitted_points_base는 로봇 base 좌표계에 정렬된 템플릿 점군입니다.
    fitted_points_base: np.ndarray
    centroid_base: tuple[float, float, float] | None

    # scale은 대표 배율, scale_xyz는 축별 배율이며 scale_mode가 해석 방식을 결정합니다.
    scale: float | None
    scale_xyz: tuple[float, float, float] | None
    scale_mode: str

    # template_axes_base는 canonical 템플릿 축이 base 좌표계에서 향하는 단위 벡터입니다.
    template_axes_base: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None
    z_rotation_deg: float | None
    roll_rotation_deg: float | None
    pitch_rotation_deg: float | None
    bowl_height_fraction: float | None
    initialized: bool
    reason: str

    # silhouette_* 필드는 2D 마스크 기반 scale 재평가가 켜졌을 때의 품질/디버그 지표입니다.
    silhouette_enabled: bool = False
    silhouette_reason: str = "disabled"
    silhouette_candidate_count: int = 0
    silhouette_valid_camera_count: int = 0
    silhouette_best_iou_cam0: float | None = None
    silhouette_best_iou_cam1: float | None = None
    silhouette_loss: float | None = None
    silhouette_outside_loss: float | None = None
    robust_3d_loss: float | None = None
    scale_prior_loss: float | None = None
    temporal_scale_loss: float | None = None
    pre_rerank_scale: float | None = None
    post_rerank_scale: float | None = None
    pre_rerank_scale_xyz: tuple[float, float, float] | None = None
    post_rerank_scale_xyz: tuple[float, float, float] | None = None
    pre_rerank_template_extent: tuple[float, float, float] | None = None
    post_rerank_template_extent: tuple[float, float, float] | None = None
    rerank_changed_candidate: bool = False


@dataclass
class ShapeFittingDebug:
    """실시간 튜닝과 로그 확인을 위한 내부 디버그 지표."""

    # 입력 점군, 선택된 클러스터, 최종 출력 점군의 크기를 추적합니다.
    label: str | None
    template_id: str | None
    raw_point_count: int
    cluster_point_count: int
    output_point_count: int
    scale_buffer_count: int

    # 현재 초기화/추적 상태와 고정된 스케일 값을 보여줍니다.
    initialized: bool
    scale: float | None
    scale_xyz: tuple[float, float, float] | None
    scale_mode: str
    reason: str
    tracking_mode: str

    # translation-only ICP의 수행 시간과 정합 품질 지표입니다.
    icp_time_ms: float | None
    icp_fps: float | None
    icp_fitness: float | None
    icp_rmse: float | None
    icp_translation_m: float | None
    icp_source_points: int
    icp_target_points: int
    icp_iterations_used: int
    z_rotation_deg: float | None
    roll_rotation_deg: float | None
    pitch_rotation_deg: float | None
    rotation_search_mode: str = "exhaustive"
    rotation_coarse_candidate_count: int = 0
    rotation_refine_candidate_count: int = 0
    rotation_candidate_count: int = 0

    # ShapeFittingState와 동일한 silhouette 디버그 값을 debug 객체에도 복사합니다.
    silhouette_enabled: bool = False
    silhouette_reason: str = "disabled"
    silhouette_candidate_count: int = 0
    silhouette_valid_camera_count: int = 0
    silhouette_best_iou_cam0: float | None = None
    silhouette_best_iou_cam1: float | None = None
    silhouette_loss: float | None = None
    silhouette_outside_loss: float | None = None
    robust_3d_loss: float | None = None
    scale_prior_loss: float | None = None
    temporal_scale_loss: float | None = None
    pre_rerank_scale: float | None = None
    post_rerank_scale: float | None = None
    pre_rerank_scale_xyz: tuple[float, float, float] | None = None
    post_rerank_scale_xyz: tuple[float, float, float] | None = None
    pre_rerank_template_extent: tuple[float, float, float] | None = None
    post_rerank_template_extent: tuple[float, float, float] | None = None
    rerank_changed_candidate: bool = False


@dataclass(frozen=True)
class _SilhouetteCandidateScore:
    """silhouette 재랭킹에서 비교할 scale 후보 하나의 손실 값 묶음."""

    index: int
    scale_xyz: np.ndarray
    points_base: np.ndarray
    total_loss: float
    robust_3d_loss: float
    silhouette_score: SilhouetteScore
    scale_prior_loss: float
    temporal_scale_loss: float
    is_growth_candidate: bool


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
        silhouette_cfg = fitting_cfg.get("silhouette_constraint", {}) or {}

        # DBSCAN으로 merged point cloud에서 실제 추적할 단일 물체 클러스터를 고릅니다.
        self.dbscan_eps_m = float(cluster_cfg.get("dbscan_eps_m", 0.02))
        self.dbscan_min_points = int(cluster_cfg.get("dbscan_min_points", 10))
        self.max_cluster_jump_m = float(cluster_cfg.get("max_cluster_jump_m", 0.08))

        # 초기 몇 프레임의 robust extent를 모아 템플릿 스케일을 한 번 고정합니다.
        self.scale_init_valid_frames = max(1, int(scale_cfg.get("stable_frames", 8)))
        self.scale_low_q = float(scale_cfg.get("percentile_low", 5.0))
        self.scale_high_q = float(scale_cfg.get("percentile_high", 95.0))
        self.min_scale = float(scale_cfg.get("min_scale", 0.5))
        self.max_scale = float(scale_cfg.get("max_scale", 1.8))
        self._default_scale_mode = self._normalize_scale_mode(scale_cfg.get("mode", SCALE_MODE_UNIFORM))

        self.output_voxel_size_m = float(downsample_cfg.get("voxel_size_m", 0.004))
        self.output_max_points = int(downsample_cfg.get("max_points", 6000))
        self.min_cluster_extent_m = float(tracking_cfg.get("min_cluster_extent_m", 1e-5))

        # 추적 단계에서는 회전/스케일을 고정하고 translation-only ICP로 위치만 갱신합니다.
        self.icp_max_points = int(icp_cfg.get("max_points", 2000))
        self.icp_distance_threshold_m = float(icp_cfg.get("distance_threshold_m", 0.09))
        self.icp_max_iterations = int(icp_cfg.get("max_iterations", 12))
        self.icp_min_fitness = float(icp_cfg.get("min_fitness", 0.02))
        self.icp_translation_tolerance_m = float(icp_cfg.get("translation_tolerance_m", 1e-5))
        self.icp_max_centroid_jump_m = float(icp_cfg.get("max_centroid_jump_m", self.max_cluster_jump_m))

        # 손/가림 영역이 섞이기 쉬운 위쪽 점들을 ICP 입력에서 선택적으로 제거합니다.
        self.icp_crop_enabled = bool(crop_cfg.get("enabled", True))
        self.icp_crop_height_axis_index = int(crop_cfg.get("height_axis_index", 2))
        self.icp_crop_target_top_fraction = crop_cfg.get("target_top_fraction", 0.10)
        self.icp_crop_source_top_fraction = crop_cfg.get("source_top_fraction", None)
        self.icp_crop_min_points_after_crop = int(crop_cfg.get("min_points_after_crop", 80))
        self.icp_crop_min_height_extent_m = float(crop_cfg.get("min_height_extent_m", 0.01))

        # 2D segmentation silhouette와 3D 거리 손실을 함께 사용해 scale 후보를 재평가합니다.
        self.silhouette_enabled = bool(silhouette_cfg.get("enabled", False))
        default_silhouette_labels = ["cup", "wine glass", "glass", "bottle"]
        self.silhouette_apply_to_all = bool(silhouette_cfg.get("apply_to_all", False))
        self.silhouette_apply_to_labels = {
            str(label).strip().lower()
            for label in silhouette_cfg.get("apply_to_labels", default_silhouette_labels)
            if str(label).strip()
        }
        self.silhouette_lambda_3d = float(silhouette_cfg.get("lambda_3d", 1.0))
        self.silhouette_lambda_iou = float(silhouette_cfg.get("lambda_iou", 0.75))
        self.silhouette_lambda_outside = float(silhouette_cfg.get("lambda_outside", 1.25))
        self.silhouette_lambda_scale_prior = float(silhouette_cfg.get("lambda_scale_prior", 0.25))
        self.silhouette_lambda_temporal = float(silhouette_cfg.get("lambda_temporal", 0.4))
        self.silhouette_point_radius_px = int(silhouette_cfg.get("point_radius_px", 2))
        self.silhouette_close_kernel_px = int(silhouette_cfg.get("close_kernel_px", 5))
        self.silhouette_dilate_px = int(silhouette_cfg.get("dilate_px", 1))
        self.silhouette_mask_erode_px = int(silhouette_cfg.get("mask_erode_px", 0))
        self.silhouette_min_rendered_pixels = int(silhouette_cfg.get("min_rendered_pixels", 30))
        self.silhouette_min_segmentation_pixels = int(silhouette_cfg.get("min_segmentation_pixels", 50))
        self.silhouette_distance_trunc_px = float(silhouette_cfg.get("distance_trunc_px", 25.0))
        self.silhouette_robust_3d_trunc_m = float(silhouette_cfg.get("robust_3d_trunc_m", 0.03))
        self.silhouette_observed_to_template_weight = float(silhouette_cfg.get("observed_to_template_weight", 0.25))
        self.silhouette_growth_penalty = float(silhouette_cfg.get("growth_penalty", 2.0))
        self.silhouette_max_growth_factor = float(silhouette_cfg.get("max_growth_factor", 1.05))
        self.silhouette_shrink_factors = [
            float(value) for value in silhouette_cfg.get("shrink_factors", [0.75, 0.85, 0.95])
        ]
        self.silhouette_uniform_growth_factors = [
            float(value) for value in silhouette_cfg.get("uniform_growth_factors", [1.0, 1.05])
        ]
        self.silhouette_max_projection_points = int(silhouette_cfg.get("max_projection_points", 1500))
        self.silhouette_robust_3d_max_points = int(silhouette_cfg.get("robust_3d_max_points", 2000))
        self.silhouette_debug = bool(silhouette_cfg.get("debug", False))

        # 템플릿 라이브러리와 프레임 간 유지되는 추적 상태입니다.
        self._templates = self._load_template_library(template_cfg)
        self._active_template: ShapeTemplateModel | None = None
        self._tracked_cluster_centroid: np.ndarray | None = None

        # 초기화 전에는 extent_buffer에 안정적인 물체 크기 샘플을 누적합니다.
        self._extent_buffer: list[np.ndarray] = []

        # 초기화 후에는 scale/rotation을 고정하고 이후 프레임에서는 translation만 업데이트합니다.
        self._frozen_scale: float | None = None
        self._frozen_scale_xyz: np.ndarray | None = None
        self._frozen_scale_mode = SCALE_MODE_UNIFORM
        self._frozen_scale_basis: np.ndarray | None = None
        self._frozen_scale_center: np.ndarray | None = None
        self._frozen_rotation = np.eye(3, dtype=np.float64)
        self._frozen_z_rotation_deg: float | None = None
        self._frozen_roll_rotation_deg: float | None = None
        self._frozen_pitch_rotation_deg: float | None = None

        # 현재 base 좌표계에 놓인 템플릿 점군입니다. 최종 출력은 여기서 downsample됩니다.
        self._current_template_points = np.empty((0, 3), dtype=np.float32)

        # silhouette 재랭킹의 직전 scale과 상세 지표를 저장해 temporal penalty와 debug에 사용합니다.
        self._last_silhouette_scale_xyz: np.ndarray | None = None
        self._last_silhouette_debug = self._make_empty_silhouette_debug(
            enabled=self.silhouette_enabled,
            reason="disabled" if not self.silhouette_enabled else "not_evaluated",
        )
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
            template_axes_base=None,
            z_rotation_deg=None,
            roll_rotation_deg=None,
            pitch_rotation_deg=None,
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
            roll_rotation_deg=None,
            pitch_rotation_deg=None,
            rotation_search_mode="uninitialized",
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
        self._frozen_roll_rotation_deg = None
        self._frozen_pitch_rotation_deg = None
        self._current_template_points = np.empty((0, 3), dtype=np.float32)
        self._last_silhouette_scale_xyz = None
        self._last_silhouette_debug = self._make_empty_silhouette_debug(
            enabled=self.silhouette_enabled,
            reason="disabled" if not self.silhouette_enabled else "reset",
        )
        self._initialized = False

    def process(
        self,
        merged_object: MergedObjectState,
        *,
        silhouette_observations: list[SilhouetteObservation] | tuple[SilhouetteObservation, ...] | None = None,
        freeze_silhouette_scale: bool = False,
    ) -> ShapeFittingState:
        # process()는 한 프레임의 merged object를 받아 템플릿 점군을 base 좌표계에 맞춘 상태로 갱신합니다.
        self._last_silhouette_debug = self._make_empty_silhouette_debug(
            enabled=self.silhouette_enabled,
            reason="disabled" if not self.silhouette_enabled else "not_evaluated",
        )
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

        # 여러 클러스터가 들어오면 이전 centroid와 가까운 클러스터를 우선해 추적 대상이 튀지 않게 합니다.
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
            # 초기화 구간에서는 여러 프레임의 물체 크기를 모은 뒤 템플릿 스케일/회전을 결정합니다.
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
            return self._maybe_rerank_with_silhouette(
                state,
                template=template,
                observed_points=filtered_points,
                silhouette_observations=silhouette_observations,
                freeze_silhouette_scale=freeze_silhouette_scale,
            )

        assert self._frozen_scale is not None
        assert self._frozen_scale_xyz is not None
        # 초기화 이후에는 고정된 scale/rotation으로 템플릿을 복원하고 translation ICP만 수행합니다.
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

        # ICP 결과가 갑자기 멀리 튀거나 품질이 낮으면 이전 템플릿 위치를 유지합니다.
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
        return self._maybe_rerank_with_silhouette(
            state,
            template=template,
            observed_points=filtered_points,
            silhouette_observations=silhouette_observations,
            freeze_silhouette_scale=freeze_silhouette_scale,
        )

    def _maybe_rerank_with_silhouette(
        self,
        state: ShapeFittingState,
        *,
        template: ShapeTemplateModel,
        observed_points: np.ndarray,
        silhouette_observations: list[SilhouetteObservation] | tuple[SilhouetteObservation, ...] | None,
        freeze_silhouette_scale: bool = False,
    ) -> ShapeFittingState:
        # silhouette constraint는 3D ICP가 만든 결과를 2D 마스크 투영 품질로 한 번 더 고르는 후처리입니다.
        if not self.silhouette_enabled:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(enabled=False, reason="disabled")
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        if not self._silhouette_applies_to_template(template):
            self._last_silhouette_debug = self._make_empty_silhouette_debug(enabled=True, reason="label_not_enabled")
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        if freeze_silhouette_scale:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(
                enabled=True,
                reason="scale_frozen_home_locked",
            )
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        if not state.valid or not self._initialized:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(enabled=True, reason="invalid_fit")
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        if not silhouette_observations:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(enabled=True, reason="no_masks")
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        if self._frozen_scale_xyz is None or len(self._current_template_points) == 0:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(enabled=True, reason="missing_template_state")
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        observed_points = np.asarray(observed_points, dtype=np.float64).reshape((-1, 3))
        if len(observed_points) == 0:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(enabled=True, reason="no_3d_points")
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        pre_scale_xyz = np.asarray(self._frozen_scale_xyz, dtype=np.float64).reshape(3)
        current_template_points = np.asarray(self._current_template_points, dtype=np.float64).reshape((-1, 3))
        current_centroid = np.mean(current_template_points, axis=0)
        pre_extent = _compute_robust_extent(current_template_points, 0.0, 100.0)
        scale_mode = self._scale_mode_for_template(template)

        # 현재 scale을 기준으로 축별 shrink/growth 후보를 만들고 3D+2D 손실로 최적 후보를 고릅니다.
        candidate_scales = self._build_silhouette_scale_candidates(pre_scale_xyz, scale_mode=scale_mode)
        if not candidate_scales:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(enabled=True, reason="no_candidates")
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        candidate_scores: list[_SilhouetteCandidateScore] = []
        for candidate_index, candidate_scale_xyz in enumerate(candidate_scales):
            candidate_points = _apply_similarity_pose(
                template.canonical_points,
                rotation=self._frozen_rotation,
                target_center=current_centroid,
                scale_xyz=candidate_scale_xyz,
                scale_basis=self._frozen_scale_basis,
                scale_center=self._frozen_scale_center,
            )
            robust_3d_loss = self._compute_robust_3d_loss(candidate_points, observed_points)
            silhouette_score = compute_silhouette_score(
                candidate_points,
                silhouette_observations,
                point_radius_px=self.silhouette_point_radius_px,
                close_kernel_px=self.silhouette_close_kernel_px,
                dilate_px=self.silhouette_dilate_px,
                mask_erode_px=self.silhouette_mask_erode_px,
                min_rendered_pixels=self.silhouette_min_rendered_pixels,
                min_segmentation_pixels=self.silhouette_min_segmentation_pixels,
                distance_trunc_px=self.silhouette_distance_trunc_px,
                max_projection_points=self.silhouette_max_projection_points,
            )
            if not silhouette_score.valid:
                continue
            scale_prior_loss = self._compute_scale_prior_loss(candidate_scale_xyz, pre_scale_xyz)
            temporal_scale_loss = self._compute_temporal_scale_loss(candidate_scale_xyz)
            total_loss = (
                self.silhouette_lambda_3d * robust_3d_loss
                + self.silhouette_lambda_iou * silhouette_score.loss_iou
                + self.silhouette_lambda_outside * silhouette_score.outside_loss
                + self.silhouette_lambda_scale_prior * scale_prior_loss
                + self.silhouette_lambda_temporal * temporal_scale_loss
            )
            ratios = candidate_scale_xyz / np.maximum(pre_scale_xyz, 1e-9)
            candidate_scores.append(
                _SilhouetteCandidateScore(
                    index=candidate_index,
                    scale_xyz=np.asarray(candidate_scale_xyz, dtype=np.float64).reshape(3),
                    points_base=np.asarray(candidate_points, dtype=np.float64).reshape((-1, 3)),
                    total_loss=float(total_loss),
                    robust_3d_loss=float(robust_3d_loss),
                    silhouette_score=silhouette_score,
                    scale_prior_loss=float(scale_prior_loss),
                    temporal_scale_loss=float(temporal_scale_loss),
                    is_growth_candidate=bool(np.any(ratios > 1.000001)),
                )
            )

        if not candidate_scores:
            self._last_silhouette_debug = self._make_empty_silhouette_debug(
                enabled=True,
                reason="no_valid_silhouette",
                candidate_count=len(candidate_scales),
            )
            self._apply_silhouette_fields_to_state(state)
            self._apply_silhouette_fields_to_debug()
            return state

        baseline_score = candidate_scores[0]
        eligible_scores = [
            score
            for score in candidate_scores
            if self._silhouette_growth_candidate_allowed(score, baseline_score)
        ]
        if not eligible_scores:
            eligible_scores = [baseline_score]
        best_score = min(eligible_scores, key=lambda score: score.total_loss)

        # scale이 바뀐 경우 frozen scale과 현재 템플릿 점군을 즉시 갱신해 다음 프레임 기준으로 사용합니다.
        changed = bool(not np.allclose(best_score.scale_xyz, pre_scale_xyz, rtol=1e-4, atol=1e-6))
        post_scale_xyz = best_score.scale_xyz
        post_points = best_score.points_base
        post_extent = _compute_robust_extent(post_points, 0.0, 100.0)
        self._last_silhouette_scale_xyz = np.asarray(post_scale_xyz, dtype=np.float64).reshape(3).copy()
        self._last_silhouette_debug = self._make_silhouette_debug_from_score(
            best_score,
            enabled=True,
            reason="changed" if changed else "kept",
            candidate_count=len(candidate_scales),
            pre_scale_xyz=pre_scale_xyz,
            post_scale_xyz=post_scale_xyz,
            pre_extent=pre_extent,
            post_extent=post_extent,
            changed=changed,
        )

        if changed:
            self._frozen_scale_xyz = np.asarray(post_scale_xyz, dtype=np.float64).reshape(3)
            self._frozen_scale = float(np.median(self._frozen_scale_xyz))
            self._current_template_points = np.asarray(post_points, dtype=np.float32).reshape((-1, 3))
            output_points = self._downsample_output(self._current_template_points)
            output_centroid = None if len(output_points) == 0 else np.mean(output_points, axis=0)
            state = self._make_state(
                valid=len(output_points) > 0,
                template=template,
                fitted_points=output_points,
                centroid=output_centroid,
                reason=state.reason,
            )
        else:
            self._apply_silhouette_fields_to_state(state)
            self.last_state = state

        self._apply_silhouette_fields_to_debug()
        return state

    def _silhouette_applies_to_template(self, template: ShapeTemplateModel | None) -> bool:
        if template is None:
            return False
        if self.silhouette_apply_to_all:
            return True
        return str(template.label).strip().lower() in self.silhouette_apply_to_labels

    def _build_silhouette_scale_candidates(self, scale_xyz: np.ndarray, *, scale_mode: str) -> list[np.ndarray]:
        base_scale = np.asarray(scale_xyz, dtype=np.float64).reshape(3)
        max_growth = max(float(self.silhouette_max_growth_factor), 1.0)
        min_scale_xyz = np.full((3,), self.min_scale, dtype=np.float64)
        max_scale_xyz = np.minimum(
            np.full((3,), self.max_scale, dtype=np.float64),
            base_scale * max_growth,
        )

        def clipped(candidate: np.ndarray) -> np.ndarray:
            return np.clip(np.asarray(candidate, dtype=np.float64).reshape(3), min_scale_xyz, max_scale_xyz)

        candidates: list[np.ndarray] = [clipped(base_scale)]
        shrink_factors = [factor for factor in self.silhouette_shrink_factors if np.isfinite(factor) and factor > 0.0]
        growth_factors = [
            min(max(float(factor), 1.0), max_growth)
            for factor in self.silhouette_uniform_growth_factors
            if np.isfinite(float(factor)) and float(factor) > 0.0
        ]

        if scale_mode == SCALE_MODE_UNIFORM:
            for factor in [*shrink_factors, 1.0, *growth_factors]:
                candidates.append(clipped(base_scale * float(factor)))
        else:
            for factor in shrink_factors:
                candidates.append(clipped(base_scale * float(factor)))
            for axis_index in range(3):
                for factor in shrink_factors:
                    candidate = base_scale.copy()
                    candidate[axis_index] *= float(factor)
                    candidates.append(clipped(candidate))
            for factor in growth_factors:
                if factor > 1.0:
                    candidates.append(clipped(base_scale * float(factor)))

        return self._deduplicate_scale_candidates(candidates)

    @staticmethod
    def _deduplicate_scale_candidates(candidates: list[np.ndarray]) -> list[np.ndarray]:
        unique_candidates: list[np.ndarray] = []
        seen: set[tuple[float, float, float]] = set()
        for candidate in candidates:
            candidate = np.asarray(candidate, dtype=np.float64).reshape(3)
            key = tuple(float(round(value, 6)) for value in candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    def _compute_robust_3d_loss(self, candidate_points: np.ndarray, observed_points: np.ndarray) -> float:
        candidate = np.asarray(candidate_points, dtype=np.float64).reshape((-1, 3))
        observed = np.asarray(observed_points, dtype=np.float64).reshape((-1, 3))
        if len(candidate) == 0 or len(observed) == 0:
            return 1.0

        candidate = _subsample_points_for_icp(candidate, self.silhouette_robust_3d_max_points)
        observed = _subsample_points_for_icp(observed, self.silhouette_robust_3d_max_points)
        trunc = max(float(self.silhouette_robust_3d_trunc_m), 1e-6)
        template_to_cloud_distances, _ = _query_nearest_neighbors_batched(candidate, observed)
        cloud_to_template_distances, _ = _query_nearest_neighbors_batched(observed, candidate)

        if len(template_to_cloud_distances) == 0 or len(cloud_to_template_distances) == 0:
            return 1.0

        template_loss = float(np.mean(np.minimum(template_to_cloud_distances, trunc)) / trunc)
        observed_loss = float(np.mean(np.minimum(cloud_to_template_distances, trunc)) / trunc)
        return float(template_loss + self.silhouette_observed_to_template_weight * observed_loss)

    def _compute_scale_prior_loss(self, candidate_scale_xyz: np.ndarray, baseline_scale_xyz: np.ndarray) -> float:
        candidate = np.asarray(candidate_scale_xyz, dtype=np.float64).reshape(3)
        baseline = np.asarray(baseline_scale_xyz, dtype=np.float64).reshape(3)
        ratios = candidate / np.maximum(baseline, 1e-9)
        growth = np.maximum(ratios - 1.0, 0.0)
        shrink = np.maximum(1.0 - ratios, 0.0)
        boundary_low = np.maximum(float(self.min_scale) - candidate, 0.0)
        boundary_high = np.maximum(candidate - float(self.max_scale), 0.0)
        return float(
            self.silhouette_growth_penalty * np.mean(np.square(growth))
            + 0.10 * np.mean(np.square(shrink))
            + np.mean(np.square(boundary_low + boundary_high))
        )

    def _compute_temporal_scale_loss(self, candidate_scale_xyz: np.ndarray) -> float:
        if self._last_silhouette_scale_xyz is None:
            return 0.0
        candidate = np.asarray(candidate_scale_xyz, dtype=np.float64).reshape(3)
        previous = np.asarray(self._last_silhouette_scale_xyz, dtype=np.float64).reshape(3)
        ratios = candidate / np.maximum(previous, 1e-9)
        growth = np.maximum(ratios - 1.0, 0.0)
        shrink = np.maximum(1.0 - ratios, 0.0)
        return float(self.silhouette_growth_penalty * np.mean(np.square(growth)) + 0.05 * np.mean(np.square(shrink)))

    @staticmethod
    def _silhouette_growth_candidate_allowed(
        score: _SilhouetteCandidateScore,
        baseline_score: _SilhouetteCandidateScore,
    ) -> bool:
        if not score.is_growth_candidate:
            return True
        iou_improved = score.silhouette_score.mean_iou > baseline_score.silhouette_score.mean_iou + 1e-4
        robust_not_worse = score.robust_3d_loss <= baseline_score.robust_3d_loss + 0.05
        return bool(iou_improved and robust_not_worse)

    @staticmethod
    def _float_or_none(value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    def _make_empty_silhouette_debug(
        self,
        *,
        enabled: bool,
        reason: str,
        candidate_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "silhouette_enabled": bool(enabled),
            "silhouette_reason": str(reason),
            "silhouette_candidate_count": int(candidate_count),
            "silhouette_valid_camera_count": 0,
            "silhouette_best_iou_cam0": None,
            "silhouette_best_iou_cam1": None,
            "silhouette_loss": None,
            "silhouette_outside_loss": None,
            "robust_3d_loss": None,
            "scale_prior_loss": None,
            "temporal_scale_loss": None,
            "pre_rerank_scale": None,
            "post_rerank_scale": None,
            "pre_rerank_scale_xyz": None,
            "post_rerank_scale_xyz": None,
            "pre_rerank_template_extent": None,
            "post_rerank_template_extent": None,
            "rerank_changed_candidate": False,
        }

    def _make_silhouette_debug_from_score(
        self,
        score: _SilhouetteCandidateScore,
        *,
        enabled: bool,
        reason: str,
        candidate_count: int,
        pre_scale_xyz: np.ndarray,
        post_scale_xyz: np.ndarray,
        pre_extent: np.ndarray,
        post_extent: np.ndarray,
        changed: bool,
    ) -> dict[str, Any]:
        pre_scale_xyz = np.asarray(pre_scale_xyz, dtype=np.float64).reshape(3)
        post_scale_xyz = np.asarray(post_scale_xyz, dtype=np.float64).reshape(3)
        pre_extent = np.asarray(pre_extent, dtype=np.float64).reshape(3)
        post_extent = np.asarray(post_extent, dtype=np.float64).reshape(3)
        return {
            "silhouette_enabled": bool(enabled),
            "silhouette_reason": str(reason),
            "silhouette_candidate_count": int(candidate_count),
            "silhouette_valid_camera_count": int(score.silhouette_score.valid_camera_count),
            "silhouette_best_iou_cam0": self._float_or_none(score.silhouette_score.per_camera_iou.get(0)),
            "silhouette_best_iou_cam1": self._float_or_none(score.silhouette_score.per_camera_iou.get(1)),
            "silhouette_loss": float(score.silhouette_score.loss_iou),
            "silhouette_outside_loss": float(score.silhouette_score.outside_loss),
            "robust_3d_loss": float(score.robust_3d_loss),
            "scale_prior_loss": float(score.scale_prior_loss),
            "temporal_scale_loss": float(score.temporal_scale_loss),
            "pre_rerank_scale": float(np.median(pre_scale_xyz)),
            "post_rerank_scale": float(np.median(post_scale_xyz)),
            "pre_rerank_scale_xyz": tuple(float(value) for value in pre_scale_xyz),
            "post_rerank_scale_xyz": tuple(float(value) for value in post_scale_xyz),
            "pre_rerank_template_extent": tuple(float(value) for value in pre_extent),
            "post_rerank_template_extent": tuple(float(value) for value in post_extent),
            "rerank_changed_candidate": bool(changed),
        }

    def _apply_silhouette_fields_to_state(self, state: ShapeFittingState) -> None:
        for key, value in self._last_silhouette_debug.items():
            if hasattr(state, key):
                setattr(state, key, value)

    def _apply_silhouette_fields_to_debug(self) -> None:
        debug = getattr(self, "last_debug", None)
        if debug is None:
            return
        for key, value in self._last_silhouette_debug.items():
            if hasattr(debug, key):
                setattr(debug, key, value)

    def _initialize_template(self, template: ShapeTemplateModel, target_points: np.ndarray) -> np.ndarray:
        # 초기화는 누적된 target extent의 median으로 scale을 잡고, 가능한 z 회전 후보 중 ICP 점수가 가장 좋은 것을 선택합니다.
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
        axis_candidate_degrees = self._axis_rotation_candidate_degrees(template)
        rotation_search_mode = (
            "coarse_to_fine"
            if template.z_rotation_enabled and template.z_rotation_coarse_to_fine_enabled
            else "exhaustive"
        )
        rotation_coarse_candidate_count = 0
        rotation_refine_candidate_count = 0
        rotation_candidate_count = 0
        best_score: tuple[float, float, float] | None = None
        best_rotation = np.eye(3, dtype=np.float64)
        best_z_rotation_deg = 0.0
        best_roll_rotation_deg = 0.0
        best_pitch_rotation_deg = 0.0
        best_initialized_points = np.empty((0, 3), dtype=np.float64)
        best_icp_time_ms = 0.0
        best_icp_fitness = 0.0
        best_icp_rmse = float("inf")
        best_icp_translation_m = 0.0
        best_icp_source_points = 0
        best_icp_target_points = 0
        best_icp_iterations_used = 0

        def evaluate_rotation_candidate(
            roll_rotation_deg: float,
            pitch_rotation_deg: float,
            z_rotation_deg: float,
            axis_rotation: np.ndarray,
        ) -> tuple[
            tuple[float, float, float],
            np.ndarray,
            float,
            float,
            float,
            np.ndarray,
            float,
            float,
            float,
            float,
            int,
            int,
            int,
        ]:
            candidate_rotation = _z_axis_rotation_matrix(z_rotation_deg) @ axis_rotation
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
            return (
                score,
                candidate_rotation,
                float(z_rotation_deg),
                float(roll_rotation_deg),
                float(pitch_rotation_deg),
                initialized_points,
                icp_time_ms,
                icp_fitness,
                icp_rmse,
                icp_translation_m,
                icp_source_points,
                icp_target_points,
                icp_iterations_used,
            )

        def update_best(
            result: tuple[
                tuple[float, float, float],
                np.ndarray,
                float,
                float,
                float,
                np.ndarray,
                float,
                float,
                float,
                float,
                int,
                int,
                int,
            ],
        ) -> None:
            nonlocal best_score
            nonlocal best_rotation
            nonlocal best_z_rotation_deg
            nonlocal best_roll_rotation_deg
            nonlocal best_pitch_rotation_deg
            nonlocal best_initialized_points
            nonlocal best_icp_time_ms
            nonlocal best_icp_fitness
            nonlocal best_icp_rmse
            nonlocal best_icp_translation_m
            nonlocal best_icp_source_points
            nonlocal best_icp_target_points
            nonlocal best_icp_iterations_used

            score = result[0]
            if best_score is not None and score <= best_score:
                return
            best_score = score
            best_rotation = result[1]
            best_z_rotation_deg = result[2]
            best_roll_rotation_deg = result[3]
            best_pitch_rotation_deg = result[4]
            best_initialized_points = result[5]
            best_icp_time_ms = result[6]
            best_icp_fitness = result[7]
            best_icp_rmse = result[8]
            best_icp_translation_m = result[9]
            best_icp_source_points = result[10]
            best_icp_target_points = result[11]
            best_icp_iterations_used = result[12]

        # roll/pitch 90도 축 전환 후보를 먼저 적용하고, 그 결과를 기존 yaw 후보로 회전합니다.
        for roll_rotation_deg, pitch_rotation_deg in axis_candidate_degrees:
            roll_rotation = _x_axis_rotation_matrix(roll_rotation_deg)
            pitch_rotation = _y_axis_rotation_matrix(pitch_rotation_deg)
            axis_rotation = pitch_rotation @ roll_rotation
            if rotation_search_mode == "coarse_to_fine":
                coarse_best: tuple[
                    tuple[float, float, float],
                    np.ndarray,
                    float,
                    float,
                    float,
                    np.ndarray,
                    float,
                    float,
                    float,
                    float,
                    int,
                    int,
                    int,
                ] | None = None
                coarse_seen: set[float] = set()
                for z_rotation_deg in self._z_rotation_coarse_candidate_degrees(template):
                    result = evaluate_rotation_candidate(
                        roll_rotation_deg,
                        pitch_rotation_deg,
                        z_rotation_deg,
                        axis_rotation,
                    )
                    rotation_coarse_candidate_count += 1
                    rotation_candidate_count += 1
                    coarse_seen.add(float(round(z_rotation_deg, 9)))
                    if coarse_best is None or result[0] > coarse_best[0]:
                        coarse_best = result
                    update_best(result)

                if coarse_best is None:
                    continue

                for z_rotation_deg in self._z_rotation_refine_candidate_degrees(template, coarse_best[2]):
                    z_key = float(round(z_rotation_deg, 9))
                    if z_key in coarse_seen:
                        continue
                    result = evaluate_rotation_candidate(
                        roll_rotation_deg,
                        pitch_rotation_deg,
                        z_rotation_deg,
                        axis_rotation,
                    )
                    rotation_refine_candidate_count += 1
                    rotation_candidate_count += 1
                    update_best(result)
            else:
                for z_rotation_deg in candidate_degrees:
                    result = evaluate_rotation_candidate(
                        roll_rotation_deg,
                        pitch_rotation_deg,
                        z_rotation_deg,
                        axis_rotation,
                    )
                    rotation_candidate_count += 1
                    update_best(result)

        # 선택된 scale/rotation은 이후 추적 중 고정되어 물체 자세가 흔들리는 것을 줄입니다.
        self._frozen_scale = float(np.median(scale_xyz))
        self._frozen_scale_xyz = np.asarray(scale_xyz, dtype=np.float64).reshape(3)
        self._frozen_scale_mode = scale_mode
        self._frozen_scale_basis = None if scale_basis is None else np.asarray(scale_basis, dtype=np.float64).reshape((3, 3))
        self._frozen_scale_center = None if scale_center is None else np.asarray(scale_center, dtype=np.float64).reshape(3)
        self._frozen_rotation = best_rotation
        self._frozen_z_rotation_deg = best_z_rotation_deg if template.z_rotation_enabled else None
        self._frozen_roll_rotation_deg = best_roll_rotation_deg if template.axis_rotation_enabled else None
        self._frozen_pitch_rotation_deg = best_pitch_rotation_deg if template.axis_rotation_enabled else None
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
            roll_rotation_deg=self._frozen_roll_rotation_deg,
            pitch_rotation_deg=self._frozen_pitch_rotation_deg,
            rotation_search_mode=rotation_search_mode,
            rotation_coarse_candidate_count=rotation_coarse_candidate_count,
            rotation_refine_candidate_count=rotation_refine_candidate_count,
            rotation_candidate_count=rotation_candidate_count,
        )
        return best_initialized_points.astype(np.float32)

    def _z_rotation_candidate_degrees(self, template: ShapeTemplateModel) -> list[float]:
        if not template.z_rotation_enabled:
            return [0.0]

        return self._build_z_rotation_candidate_degrees(
            template.z_rotation_min_deg,
            template.z_rotation_max_deg,
            template.z_rotation_step_deg,
        )

    @staticmethod
    def _build_z_rotation_candidate_degrees(min_deg: float, max_deg: float, step_deg: float) -> list[float]:
        min_deg = float(min_deg)
        max_deg = float(max_deg)
        if not np.isfinite(min_deg) or not np.isfinite(max_deg):
            return [0.0]
        if min_deg > max_deg:
            min_deg, max_deg = max_deg, min_deg

        step_deg = abs(float(step_deg))
        if not np.isfinite(step_deg) or step_deg <= 0.0:
            step_deg = 5.0

        candidates = list(np.arange(min_deg, max_deg + (0.5 * step_deg), step_deg, dtype=np.float64))
        candidates = [float(np.clip(candidate, min_deg, max_deg)) for candidate in candidates]
        candidates.append(max_deg)
        if min_deg <= 0.0 <= max_deg and not any(abs(candidate) <= 1e-9 for candidate in candidates):
            candidates.append(0.0)
        return sorted(set(round(candidate, 9) for candidate in candidates))

    def _z_rotation_coarse_candidate_degrees(self, template: ShapeTemplateModel) -> list[float]:
        if not template.z_rotation_enabled:
            return [0.0]
        return self._build_z_rotation_candidate_degrees(
            template.z_rotation_min_deg,
            template.z_rotation_max_deg,
            template.z_rotation_coarse_step_deg,
        )

    def _z_rotation_refine_candidate_degrees(self, template: ShapeTemplateModel, center_deg: float) -> list[float]:
        if not template.z_rotation_enabled:
            return [0.0]

        min_deg = float(template.z_rotation_min_deg)
        max_deg = float(template.z_rotation_max_deg)
        if min_deg > max_deg:
            min_deg, max_deg = max_deg, min_deg

        radius_deg = abs(float(template.z_rotation_refine_radius_deg))
        if not np.isfinite(radius_deg):
            radius_deg = 10.0
        lower_deg = max(min_deg, float(center_deg) - radius_deg)
        upper_deg = min(max_deg, float(center_deg) + radius_deg)
        return self._build_z_rotation_candidate_degrees(
            lower_deg,
            upper_deg,
            template.z_rotation_refine_step_deg,
        )

    @staticmethod
    def _axis_rotation_candidate_degrees(template: ShapeTemplateModel) -> list[tuple[float, float]]:
        if not template.axis_rotation_enabled:
            return [(0.0, 0.0)]

        candidates: list[tuple[float, float]] = [(0.0, 0.0)]
        for roll_deg in template.axis_roll_candidates_deg:
            roll_deg = float(roll_deg)
            if np.isfinite(roll_deg) and abs(roll_deg) > 1e-9:
                candidates.append((roll_deg, 0.0))
        for pitch_deg in template.axis_pitch_candidates_deg:
            pitch_deg = float(pitch_deg)
            if np.isfinite(pitch_deg) and abs(pitch_deg) > 1e-9:
                candidates.append((0.0, pitch_deg))

        unique_candidates: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for roll_deg, pitch_deg in candidates:
            key = (float(round(roll_deg, 9)), float(round(pitch_deg, 9)))
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(key)
        return unique_candidates

    def _run_translation_only_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        *,
        height_axis_index: int,
    ) -> tuple[np.ndarray, float, float, float, int, int, int]:
        # 회전과 스케일을 바꾸지 않고, 최근접점 평균 오차로 translation만 반복 갱신합니다.
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

        # fitness는 거리 threshold 안에 들어온 대응점 비율, rmse는 해당 대응점들의 평균 거리입니다.
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
        # DBSCAN 결과 중 기본적으로 가장 큰 클러스터를 쓰되, 이전 centroid 근처 클러스터가 있으면 그쪽을 유지합니다.
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

    def _template_axes_base(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
        if not self._initialized:
            return None

        # canonical 축을 scale basis와 frozen rotation을 거쳐 base 좌표계 방향 벡터로 변환합니다.
        basis = (
            np.eye(3, dtype=np.float64)
            if self._frozen_scale_basis is None
            else np.asarray(self._frozen_scale_basis, dtype=np.float64).reshape((3, 3))
        )
        scale = (
            np.ones((3,), dtype=np.float64)
            if self._frozen_scale_xyz is None
            else np.asarray(self._frozen_scale_xyz, dtype=np.float64).reshape(3)
        )
        rotation = np.asarray(self._frozen_rotation, dtype=np.float64).reshape((3, 3))
        canonical_axes = np.eye(3, dtype=np.float64)
        axes_base: list[tuple[float, float, float]] = []
        for axis in canonical_axes:
            axis_base = ((axis @ basis) * scale) @ basis.T @ rotation.T
            norm = float(np.linalg.norm(axis_base))
            if not np.isfinite(norm) or norm <= 1e-9:
                return None
            axis_base = axis_base / norm
            axes_base.append(tuple(float(v) for v in axis_base))
        return tuple(axes_base)  # type: ignore[return-value]

    def _resolve_template(self, label: str | None) -> ShapeTemplateModel | None:
        if not label:
            return None
        return self._templates.get(str(label))

    def _load_template_library(self, template_cfg: dict[str, Any]) -> dict[str, ShapeTemplateModel]:
        # YAML의 template_library 항목을 실제 numpy 점군과 메타데이터로 변환합니다.
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
            coarse_to_fine_cfg = z_rotation_cfg.get("coarse_to_fine", {}) or {}
            if isinstance(coarse_to_fine_cfg, bool):
                coarse_to_fine_cfg = {"enabled": coarse_to_fine_cfg}
            z_rotation_coarse_to_fine_enabled = bool(coarse_to_fine_cfg.get("enabled", False))
            z_rotation_coarse_step_deg = float(coarse_to_fine_cfg.get("coarse_step_deg", 15.0))
            z_rotation_refine_radius_deg = float(coarse_to_fine_cfg.get("refine_radius_deg", 10.0))
            z_rotation_refine_step_deg = float(coarse_to_fine_cfg.get("refine_step_deg", z_rotation_step_deg))
            axis_rotation_cfg = entry.get("axis_rotation_candidates", {}) or {}
            if isinstance(axis_rotation_cfg, bool):
                axis_rotation_cfg = {"enabled": axis_rotation_cfg}
            axis_rotation_enabled = bool(axis_rotation_cfg.get("enabled", False))
            axis_roll_candidates_deg = tuple(
                float(value) for value in axis_rotation_cfg.get("roll_deg", [90.0, -90.0])
            )
            axis_pitch_candidates_deg = tuple(
                float(value) for value in axis_rotation_cfg.get("pitch_deg", [90.0, -90.0])
            )
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
                z_rotation_coarse_to_fine_enabled=z_rotation_coarse_to_fine_enabled,
                z_rotation_coarse_step_deg=z_rotation_coarse_step_deg,
                z_rotation_refine_radius_deg=z_rotation_refine_radius_deg,
                z_rotation_refine_step_deg=z_rotation_refine_step_deg,
                axis_rotation_enabled=axis_rotation_enabled,
                axis_roll_candidates_deg=axis_roll_candidates_deg,
                axis_pitch_candidates_deg=axis_pitch_candidates_deg,
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
        roll_rotation_deg: float | None = None,
        pitch_rotation_deg: float | None = None,
        rotation_search_mode: str = "exhaustive",
        rotation_coarse_candidate_count: int = 0,
        rotation_refine_candidate_count: int = 0,
        rotation_candidate_count: int = 0,
    ) -> None:
        # last_debug는 UI/로그에서 현재 tracking 품질을 바로 읽기 위한 스냅샷입니다.
        icp_fps = None
        if icp_time_ms is not None and np.isfinite(icp_time_ms) and icp_time_ms > 0.0:
            icp_fps = 1000.0 / float(icp_time_ms)
        if z_rotation_deg is None and self._active_template is not None and self._active_template.z_rotation_enabled:
            z_rotation_deg = self._frozen_z_rotation_deg
        if roll_rotation_deg is None and self._active_template is not None and self._active_template.axis_rotation_enabled:
            roll_rotation_deg = self._frozen_roll_rotation_deg
        if pitch_rotation_deg is None and self._active_template is not None and self._active_template.axis_rotation_enabled:
            pitch_rotation_deg = self._frozen_pitch_rotation_deg
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
            roll_rotation_deg=None if roll_rotation_deg is None else float(roll_rotation_deg),
            pitch_rotation_deg=None if pitch_rotation_deg is None else float(pitch_rotation_deg),
            rotation_search_mode=str(rotation_search_mode),
            rotation_coarse_candidate_count=int(rotation_coarse_candidate_count),
            rotation_refine_candidate_count=int(rotation_refine_candidate_count),
            rotation_candidate_count=int(rotation_candidate_count),
        )
        self._apply_silhouette_fields_to_debug()

    def _make_state(
        self,
        *,
        valid: bool,
        template: ShapeTemplateModel | None,
        fitted_points: np.ndarray | None,
        centroid: np.ndarray | tuple[float, float, float] | None,
        reason: str,
    ) -> ShapeFittingState:
        # last_state는 다른 모듈이 소비하는 표준 출력이므로 numpy 타입과 tuple 타입을 여기서 정리합니다.
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
            template_axes_base=self._template_axes_base() if valid and template is not None else None,
            z_rotation_deg=None if self._frozen_z_rotation_deg is None else float(self._frozen_z_rotation_deg),
            roll_rotation_deg=None if self._frozen_roll_rotation_deg is None else float(self._frozen_roll_rotation_deg),
            pitch_rotation_deg=None if self._frozen_pitch_rotation_deg is None else float(self._frozen_pitch_rotation_deg),
            bowl_height_fraction=None if template is None else template.bowl_height_fraction,
            initialized=bool(self._initialized),
            reason=str(reason),
        )
        self._apply_silhouette_fields_to_state(state)
        self.last_state = state
        return state


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ShapeFittingDebug",
    "ShapeFittingState",
    "ShapeFittingTracker",
    "SilhouetteObservation",
]
