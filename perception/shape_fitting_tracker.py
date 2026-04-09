"""Template-based point-cloud fitting for the RTDE perception pipeline."""

from __future__ import annotations

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

from object_pt_extraction.pointcloud_utils import voxel_downsample_point_cloud
from system.shared_state import MergedObjectState

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _robust_extent(points: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return np.zeros((3,), dtype=np.float32)
    lower = np.percentile(points, low_q, axis=0)
    upper = np.percentile(points, high_q, axis=0)
    return (upper - lower).astype(np.float32)


def _limit_point_count(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if max_points <= 0 or len(points) <= max_points:
        return points
    sample_indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[sample_indices]


@dataclass(frozen=True)
class ShapeTemplateModel:
    label: str
    template_id: str
    asset_path: Path
    unit_scale_m: float
    canonical_points: np.ndarray
    centered_points: np.ndarray
    source_extent_xyz: np.ndarray


@dataclass
class ShapeFittingState:
    valid: bool
    label: str | None
    template_id: str | None
    fitted_points_base: np.ndarray
    centroid_base: tuple[float, float, float] | None
    scale: float | None
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
    reason: str


class ShapeFittingTracker:
    """Fits a canonical template cloud to the merged object cloud."""

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

        self.dbscan_eps_m = float(cluster_cfg.get("dbscan_eps_m", 0.02))
        self.dbscan_min_points = int(cluster_cfg.get("dbscan_min_points", 10))
        self.max_cluster_jump_m = float(cluster_cfg.get("max_cluster_jump_m", 0.08))

        self.scale_init_valid_frames = max(1, int(scale_cfg.get("stable_frames", 8)))
        self.scale_low_q = float(scale_cfg.get("percentile_low", 5.0))
        self.scale_high_q = float(scale_cfg.get("percentile_high", 95.0))
        self.min_scale = float(scale_cfg.get("min_scale", 0.5))
        self.max_scale = float(scale_cfg.get("max_scale", 1.8))

        self.output_voxel_size_m = float(downsample_cfg.get("voxel_size_m", 0.004))
        self.output_max_points = int(downsample_cfg.get("max_points", 6000))
        self.min_cluster_extent_m = float(tracking_cfg.get("min_cluster_extent_m", 1e-5))

        self._templates = self._load_template_library(template_cfg)
        self._active_template: ShapeTemplateModel | None = None
        self._tracked_cluster_centroid: np.ndarray | None = None
        self._extent_buffer: list[np.ndarray] = []
        self._scaled_template_centered: np.ndarray | None = None
        self._frozen_scale: float | None = None
        self._initialized = False

        self.last_state = ShapeFittingState(
            valid=False,
            label=None,
            template_id=None,
            fitted_points_base=np.empty((0, 3), dtype=np.float32),
            centroid_base=None,
            scale=None,
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
            reason="uninitialized",
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
        self._scaled_template_centered = None
        self._frozen_scale = None
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
            )
            return self._make_state(valid=False, template=template, fitted_points=None, centroid=None, reason="no_merged_object")

        filtered_points, cluster_centroid = self._filter_single_cluster(raw_points, self._tracked_cluster_centroid)
        if len(filtered_points) == 0 or cluster_centroid is None:
            self._set_debug(
                label=template.label,
                template_id=template.template_id,
                raw_point_count=len(raw_points),
                cluster_point_count=0,
                output_point_count=0,
                reason="no_valid_cluster",
            )
            return self._make_state(valid=False, template=template, fitted_points=None, centroid=None, reason="no_valid_cluster")

        self._tracked_cluster_centroid = cluster_centroid.astype(np.float32)
        target_centroid = np.mean(filtered_points, axis=0).astype(np.float32)

        if not self._initialized:
            robust_extent = _robust_extent(filtered_points, self.scale_low_q, self.scale_high_q)
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
                )
                return self._make_state(valid=False, template=template, fitted_points=None, centroid=None, reason=reason)
            self._initialize_template(template)

        assert self._scaled_template_centered is not None
        fitted_points = self._scaled_template_centered + target_centroid.reshape(1, 3)
        fitted_points = self._downsample_output(fitted_points)

        state = self._make_state(
            valid=len(fitted_points) > 0,
            template=template,
            fitted_points=fitted_points,
            centroid=target_centroid,
            reason="ok" if len(fitted_points) > 0 else "empty_fitted_template",
        )
        self._set_debug(
            label=template.label,
            template_id=template.template_id,
            raw_point_count=len(raw_points),
            cluster_point_count=len(filtered_points),
            output_point_count=len(fitted_points),
            reason=state.reason,
        )
        return state

    def _initialize_template(self, template: ShapeTemplateModel) -> None:
        median_extent = np.median(np.asarray(self._extent_buffer, dtype=np.float32), axis=0)
        valid_axes = template.source_extent_xyz > 1e-6
        if not np.any(valid_axes):
            self._scaled_template_centered = template.centered_points.astype(np.float32)
            self._frozen_scale = 1.0
        else:
            raw_axis_scales = median_extent[valid_axes] / template.source_extent_xyz[valid_axes]
            uniform_scale = float(np.median(raw_axis_scales))
            uniform_scale = float(np.clip(uniform_scale, self.min_scale, self.max_scale))
            self._scaled_template_centered = (template.centered_points * uniform_scale).astype(np.float32)
            self._frozen_scale = uniform_scale
        self._initialized = True

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

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
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

    def _resolve_template(self, label: str | None) -> ShapeTemplateModel | None:
        if not label:
            return None
        return self._templates.get(str(label))

    def _load_template_library(self, template_cfg: dict[str, Any]) -> dict[str, ShapeTemplateModel]:
        templates: dict[str, ShapeTemplateModel] = {}
        for label, entry in template_cfg.items():
            asset_path = _resolve_path(entry["asset_path"], base_dir=self._config_path.parent)
            canonical_points = np.load(asset_path).astype(np.float32) * float(entry.get("unit_scale_m", 1.0))
            centroid = np.mean(canonical_points, axis=0, keepdims=True)
            centered_points = canonical_points - centroid
            templates[str(label)] = ShapeTemplateModel(
                label=str(label),
                template_id=str(entry.get("template_id", label)),
                asset_path=asset_path,
                unit_scale_m=float(entry.get("unit_scale_m", 1.0)),
                canonical_points=canonical_points,
                centered_points=centered_points.astype(np.float32),
                source_extent_xyz=_robust_extent(canonical_points, 0.0, 100.0),
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
    ) -> None:
        self.last_debug = ShapeFittingDebug(
            label=label,
            template_id=template_id,
            raw_point_count=int(raw_point_count),
            cluster_point_count=int(cluster_point_count),
            output_point_count=int(output_point_count),
            scale_buffer_count=len(self._extent_buffer),
            initialized=bool(self._initialized),
            scale=None if self._frozen_scale is None else float(self._frozen_scale),
            reason=str(reason),
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
        state = ShapeFittingState(
            valid=bool(valid),
            label=None if template is None else template.label,
            template_id=None if template is None else template.template_id,
            fitted_points_base=fitted_points_arr,
            centroid_base=centroid_tuple,
            scale=None if self._frozen_scale is None else float(self._frozen_scale),
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
