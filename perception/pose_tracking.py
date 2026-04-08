"""Scale-adaptive FoundationPose tracking built on the dual-camera pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

try:
    import cv2 as cv
except Exception:  # pragma: no cover - optional during unit tests
    cv = None
import numpy as np
try:
    import yaml
except Exception:  # pragma: no cover - optional during unit tests
    yaml = None

from calibration.extrinsics import TransformChain, load_transform_chain
from system.dual_sensor_hub import DualFrameSnapshot
from system.shared_state import (
    POSE_TRACKING_REGISTERING,
    POSE_TRACKING_REINIT_PENDING,
    POSE_TRACKING_TRACKING,
    POSE_TRACKING_UNINITIALIZED,
    POSE_TRACKING_WAITING_STABLE_SCALE,
    PoseTrackingState,
)
from utils.realsense_stream import FrameBundle

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONPOSE_DIR = REPO_ROOT / "FoundationPose"
HEIGHT_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


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


def _matrix_to_tuple(matrix: np.ndarray | None) -> tuple[tuple[float, float, float, float], ...] | None:
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    return tuple(tuple(float(v) for v in row) for row in array)


def _compose_intrinsics_matrix(intrinsics: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size != 3:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or not np.isfinite(norm):
        return None
    return vector / norm


def _orthogonal_basis_from_height(height_axis: np.ndarray) -> np.ndarray:
    height = _normalize(height_axis)
    assert height is not None

    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(reference, height))) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    radial_x = np.cross(height, reference)
    radial_x = _normalize(radial_x)
    assert radial_x is not None
    radial_y = np.cross(height, radial_x)
    radial_y = _normalize(radial_y)
    assert radial_y is not None
    return np.stack([radial_x, radial_y, height], axis=1).astype(np.float32)


def _robust_span(values: np.ndarray, low_q: float, high_q: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, high_q) - np.percentile(values, low_q))


def _draw_disk(mask: np.ndarray, u: int, v: int, radius: int) -> None:
    height, width = mask.shape[:2]
    if cv is not None:
        cv.circle(mask, (int(u), int(v)), int(radius), 1, -1, cv.LINE_AA)
        return

    radius = max(int(radius), 0)
    x0 = max(int(u) - radius, 0)
    x1 = min(int(u) + radius + 1, width)
    y0 = max(int(v) - radius, 0)
    y1 = min(int(v) + radius + 1, height)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - int(u)) ** 2 + (yy - int(v)) ** 2 <= radius ** 2
    mask[y0:y1, x0:x1][disk] = 1


@dataclass
class TemplateSpec:
    label: str
    template_id: str
    asset_path: Path
    unit_scale_m: float
    mesh_sample_count: int


@dataclass
class TemplateModel:
    spec: TemplateSpec
    mesh: Any
    canonical_points: np.ndarray
    centered_points: np.ndarray
    radial_extent_m: float
    height_extent_m: float


@dataclass
class StableScaleEstimate:
    template_id: str
    label: str
    scale_xyz: np.ndarray
    radial_scale: float
    height_scale: float
    point_count: int


@dataclass
class PoseTrackingDebug:
    anchor_mask: np.ndarray | None = None
    projected_pixels: np.ndarray | None = None
    rendered_mask: np.ndarray | None = None
    mask_iou: float = 0.0
    depth_inlier_ratio: float = 0.0
    low_confidence_streak: int = 0
    stable_scale_ready: bool = False
    backend_stage: str | None = None
    scale_estimation_ms: float = 0.0
    backend_call_ms: float = 0.0
    health_check_ms: float = 0.0
    total_update_ms: float = 0.0


@dataclass
class _FallbackVisual:
    vertex_colors: np.ndarray


class _FallbackMesh:
    """Minimal mesh-like container used when trimesh is unavailable."""

    def __init__(self, vertices: np.ndarray, vertex_colors: np.ndarray | None = None) -> None:
        vertices = np.asarray(vertices, dtype=np.float32).reshape((-1, 3))
        if vertex_colors is None:
            vertex_colors = np.tile(np.array([160, 160, 160], dtype=np.uint8), (len(vertices), 1))
        self.vertices = vertices
        self.visual = _FallbackVisual(np.asarray(vertex_colors, dtype=np.uint8).reshape((-1, 3)))

    def copy(self) -> "_FallbackMesh":
        return _FallbackMesh(self.vertices.copy(), self.visual.vertex_colors.copy())

    def sample(self, count: int) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.empty((0, 3), dtype=np.float32)
        if count <= len(self.vertices):
            return self.vertices[:count].astype(np.float32, copy=True)
        repeats = int(np.ceil(float(count) / float(len(self.vertices))))
        tiled = np.tile(self.vertices, (repeats, 1))
        return tiled[:count].astype(np.float32, copy=True)


class TemplateLibrary:
    """Maps YOLO labels to canonical template assets."""

    def __init__(self, models: dict[str, TemplateModel]) -> None:
        self._models = dict(models)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        config: dict[str, Any] | None = None,
    ) -> "TemplateLibrary":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load pose tracking templates from config.")
        config_path = _resolve_path(config_path)
        if config is None:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        tracking_cfg = config.get("perception", {}).get("pose_tracking", {})
        library_cfg = tracking_cfg.get("template_library", {})
        mesh_sample_count = int(tracking_cfg.get("surface_sample_count", 1500))
        models: dict[str, TemplateModel] = {}
        for label, entry in library_cfg.items():
            asset_path = _resolve_path(entry["asset_path"], base_dir=config_path.parent)
            spec = TemplateSpec(
                label=str(label),
                template_id=str(entry.get("template_id", label)),
                asset_path=asset_path,
                unit_scale_m=float(entry.get("unit_scale_m", 1.0)),
                mesh_sample_count=int(entry.get("mesh_sample_count", mesh_sample_count)),
            )
            models[str(label)] = cls._load_template_model(spec)
        return cls(models=models)

    @staticmethod
    def _load_template_model(spec: TemplateSpec) -> TemplateModel:
        trimesh = None
        try:
            import trimesh
        except Exception:
            trimesh = None

        suffix = spec.asset_path.suffix.lower()
        if suffix == ".npy":
            raw_points = np.load(spec.asset_path).astype(np.float32)
            canonical_points = raw_points * float(spec.unit_scale_m)
            if trimesh is not None:
                point_cloud = trimesh.points.PointCloud(canonical_points)
                try:
                    mesh = point_cloud.convex_hull
                except Exception:
                    extents = np.maximum(np.ptp(canonical_points, axis=0), 1e-3)
                    mesh = trimesh.creation.box(extents=extents)
                    mesh.apply_translation(np.mean(canonical_points, axis=0))
            else:
                mesh = _FallbackMesh(canonical_points)
        else:
            if trimesh is None:
                raise RuntimeError(
                    f"trimesh is required to load mesh asset `{spec.asset_path}`. "
                    "Install `trimesh` or switch the template asset to `.npy`."
                )
            mesh = trimesh.load(spec.asset_path, process=False)
            mesh = mesh.copy()
            mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * float(spec.unit_scale_m)
            if spec.mesh_sample_count > 0:
                canonical_points = mesh.sample(spec.mesh_sample_count).astype(np.float32)
            else:
                canonical_points = np.asarray(mesh.vertices, dtype=np.float32).copy()

        if getattr(mesh.visual, "vertex_colors", None) is None:
            mesh.visual.vertex_colors = np.tile(np.array([160, 160, 160], dtype=np.uint8), (len(mesh.vertices), 1))

        centered_points = canonical_points - np.mean(canonical_points, axis=0, keepdims=True)
        radial_extent_m = _robust_span(np.linalg.norm(centered_points[:, :2], axis=1), 5.0, 95.0) * 2.0
        height_extent_m = _robust_span(centered_points[:, 2], 5.0, 95.0)
        radial_extent_m = max(float(radial_extent_m), 1e-3)
        height_extent_m = max(float(height_extent_m), 1e-3)
        return TemplateModel(
            spec=spec,
            mesh=mesh,
            canonical_points=canonical_points.astype(np.float32),
            centered_points=centered_points.astype(np.float32),
            radial_extent_m=radial_extent_m,
            height_extent_m=height_extent_m,
        )

    def resolve(self, label: str | None) -> TemplateModel | None:
        if not label:
            return None
        return self._models.get(str(label))


class ScaleEstimator:
    """Radial-height scale estimation from merged object point clouds."""

    def __init__(self, config: dict[str, Any]) -> None:
        tracking_cfg = config.get("perception", {}).get("pose_tracking", {})
        scale_cfg = tracking_cfg.get("scale_estimation", {})
        frame_cfg = config.get("frames", {}).get("height_axis", {})

        self.min_points = int(scale_cfg.get("min_points", 500))
        self.stable_frames = max(1, int(scale_cfg.get("stable_frames", 5)))
        self.low_q = float(scale_cfg.get("percentile_low", 5.0))
        self.high_q = float(scale_cfg.get("percentile_high", 95.0))
        self.min_scale = float(scale_cfg.get("min_scale", 0.25))
        self.max_scale = float(scale_cfg.get("max_scale", 4.0))
        axis_name = str(frame_cfg.get("name", "z")).strip().lower()
        axis_index = HEIGHT_AXIS_TO_INDEX.get(axis_name, 2)
        height_axis = np.zeros((3,), dtype=np.float32)
        height_axis[axis_index] = 1.0
        self.height_axis_base = height_axis
        self._history_by_template_id: dict[str, deque[np.ndarray]] = {}

    def reset_template(self, template_id: str) -> None:
        self._history_by_template_id.pop(str(template_id), None)

    def estimate(self, template: TemplateModel, merged_points_base: np.ndarray) -> StableScaleEstimate | None:
        points = np.asarray(merged_points_base, dtype=np.float32).reshape((-1, 3))
        if len(points) < self.min_points:
            return None

        centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        candidate_axes = np.asarray(vh.T, dtype=np.float32)

        best_axis = candidate_axes[:, 0]
        best_alignment = -1.0
        for axis_idx in range(candidate_axes.shape[1]):
            axis = candidate_axes[:, axis_idx]
            alignment = abs(float(np.dot(axis, self.height_axis_base)))
            if alignment > best_alignment:
                best_alignment = alignment
                best_axis = axis

        if float(np.dot(best_axis, self.height_axis_base)) < 0.0:
            best_axis = -best_axis

        basis = _orthogonal_basis_from_height(best_axis)
        local_points = centered @ basis
        radial_distances = np.linalg.norm(local_points[:, :2], axis=1)
        radial_extent_m = _robust_span(radial_distances, self.low_q, self.high_q) * 2.0
        height_extent_m = _robust_span(local_points[:, 2], self.low_q, self.high_q)

        radial_scale = np.clip(radial_extent_m / template.radial_extent_m, self.min_scale, self.max_scale)
        height_scale = np.clip(height_extent_m / template.height_extent_m, self.min_scale, self.max_scale)
        scale_xyz = np.asarray([radial_scale, radial_scale, height_scale], dtype=np.float32)

        history = self._history_by_template_id.setdefault(
            template.spec.template_id,
            deque(maxlen=self.stable_frames),
        )
        history.append(scale_xyz.copy())
        if len(history) < self.stable_frames:
            return None

        stable_scale = np.median(np.stack(history, axis=0), axis=0).astype(np.float32)
        return StableScaleEstimate(
            template_id=template.spec.template_id,
            label=template.spec.label,
            scale_xyz=stable_scale,
            radial_scale=float(stable_scale[0]),
            height_scale=float(stable_scale[2]),
            point_count=int(len(points)),
        )


class LazyFoundationPoseBackend:
    """Lazy wrapper around FoundationPose to keep unit tests lightweight."""

    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = Path(debug_dir)
        self.available = False
        self.error_message: str | None = None
        self._initialized = False
        self._estimator = None
        self._trimesh = None
        self._debug_counter = 0

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self.available
        self._initialized = True

        try:  # pragma: no cover - exercised in integration environments
            if str(FOUNDATIONPOSE_DIR) not in sys.path:
                sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

            import trimesh
            from Utils import dr, set_seed
            from estimater import FoundationPose
            from learning.training.predict_pose_refine import PoseRefinePredictor
            from learning.training.predict_score import ScorePredictor

            set_seed(0)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()
            mesh = trimesh.creation.box(extents=np.asarray([0.05, 0.05, 0.05], dtype=np.float32))
            mesh.visual.vertex_colors = np.tile(np.array([160, 160, 160], dtype=np.uint8), (len(mesh.vertices), 1))
            self._estimator = FoundationPose(
                model_pts=np.asarray(mesh.vertices).copy(),
                model_normals=np.asarray(mesh.vertex_normals).copy(),
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                glctx=glctx,
                debug=0,
                debug_dir=str(self.debug_dir),
            )
            self._trimesh = trimesh
            self.available = True
        except Exception as exc:
            self.error_message = str(exc)
            self.available = False
            print(f"[WARN] FoundationPose backend unavailable: {self.error_message}", flush=True)
        return self.available

    def reset_object(self, mesh: Any) -> None:
        if not self._ensure_initialized():
            raise RuntimeError(self.error_message or "FoundationPose backend is unavailable.")
        assert self._estimator is not None
        mesh = mesh.copy()
        self._estimator.reset_object(
            model_pts=np.asarray(mesh.vertices).copy(),
            model_normals=np.asarray(mesh.vertex_normals).copy(),
            symmetry_tfs=None,
            mesh=mesh,
        )

    def register(self, frame: FrameBundle, object_mask: np.ndarray, iteration: int) -> np.ndarray:
        if not self._ensure_initialized():
            raise RuntimeError(self.error_message or "FoundationPose backend is unavailable.")
        assert self._estimator is not None
        rgb_bgr = np.ascontiguousarray(frame.color_image[:, :, ::-1], dtype=np.uint8)
        depth_m = np.ascontiguousarray(frame.depth_image_m, dtype=np.float32)
        mask_u8 = np.ascontiguousarray(object_mask, dtype=np.uint8)
        pose = self._estimator.register(
            K=_compose_intrinsics_matrix(frame.intrinsics),
            rgb=rgb_bgr,
            depth=depth_m,
            ob_mask=mask_u8,
            iteration=int(iteration),
        )
        return np.asarray(pose, dtype=np.float32).reshape(4, 4)

    def track(self, frame: FrameBundle, iteration: int) -> np.ndarray:
        if not self._ensure_initialized():
            raise RuntimeError(self.error_message or "FoundationPose backend is unavailable.")
        assert self._estimator is not None
        rgb_bgr = np.ascontiguousarray(frame.color_image[:, :, ::-1], dtype=np.uint8)
        depth_m = np.ascontiguousarray(frame.depth_image_m, dtype=np.float32)
        pose = self._estimator.track_one(
            rgb=rgb_bgr,
            depth=depth_m,
            K=_compose_intrinsics_matrix(frame.intrinsics),
            iteration=int(iteration),
        )
        return np.asarray(pose, dtype=np.float32).reshape(4, 4)


class FoundationPoseTracker:
    """State machine for scale-aware registration and tracking."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        template_library: TemplateLibrary,
        scale_estimator: ScaleEstimator,
        transform_chain: TransformChain,
        backend: Any | None = None,
    ) -> None:
        self.config = config
        tracking_cfg = config.get("perception", {}).get("pose_tracking", {})
        registration_cfg = tracking_cfg.get("registration", {})
        track_cfg = tracking_cfg.get("tracking", {})
        health_cfg = tracking_cfg.get("health", {})
        reinit_cfg = tracking_cfg.get("reinit", {})

        anchor_camera = str(tracking_cfg.get("anchor_camera", "cam0")).strip().lower()
        if anchor_camera != "cam0":
            raise ValueError("FoundationPoseTracker currently supports only anchor_camera=cam0.")
        self.anchor_camera_id = 0
        self.enabled = bool(tracking_cfg.get("enabled", False))
        self.surface_sample_count = max(32, int(tracking_cfg.get("surface_sample_count", 1500)))
        self.registration_iterations = int(registration_cfg.get("refine_iter", 5))
        self.track_iterations = int(track_cfg.get("refine_iter", 2))
        self.mask_iou_threshold = float(health_cfg.get("mask_iou_threshold", 0.15))
        self.depth_inlier_threshold = float(health_cfg.get("depth_inlier_threshold", 0.30))
        self.depth_tolerance_m = float(health_cfg.get("depth_tolerance_m", 0.03))
        self.low_conf_frames = max(1, int(reinit_cfg.get("low_conf_frames", 5)))
        self.min_merged_points_for_tracking = max(1, int(reinit_cfg.get("min_merged_points", 300)))

        self.template_library = template_library
        self.scale_estimator = scale_estimator
        self.transform_chain = transform_chain
        self.backend = backend if backend is not None else LazyFoundationPoseBackend(
            debug_dir=_resolve_path(
                tracking_cfg.get("debug_dir", "logs/foundationpose_debug"),
                base_dir=DEFAULT_CONFIG_PATH.parent,
            )
        )

        self._mode = POSE_TRACKING_UNINITIALIZED
        self._reinit_reason: str | None = None
        self._template: TemplateModel | None = None
        self._scale_xyz: np.ndarray | None = None
        self._scaled_mesh = None
        self._scaled_mesh_points = np.empty((0, 3), dtype=np.float32)
        self._pose_cam0: np.ndarray | None = None
        self._pose_base: np.ndarray | None = None
        self._low_confidence_streak = 0
        self._anchor_frame_override: FrameBundle | None = None
        self._anchor_mask_override: np.ndarray | None = None
        self.last_debug = PoseTrackingDebug()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        backend: Any | None = None,
    ) -> "FoundationPoseTracker":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load pose tracking configuration.")
        config_path = _resolve_path(config_path)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(
            config=config,
            template_library=TemplateLibrary.from_config(config_path=config_path, config=config),
            scale_estimator=ScaleEstimator(config),
            transform_chain=load_transform_chain(config_path),
            backend=backend,
        )

    def set_anchor_observation(self, frame_bundle: FrameBundle, object_mask: np.ndarray | None) -> None:
        self._anchor_frame_override = frame_bundle
        self._anchor_mask_override = None if object_mask is None else np.asarray(object_mask, dtype=bool).copy()

    def update(
        self,
        snapshot: DualFrameSnapshot,
        object_cam0: Any,
        object_cam1: Any,
        merged_object: Any,
    ) -> PoseTrackingState:
        del object_cam1
        update_start = time.perf_counter()
        scale_ms = 0.0
        backend_ms = 0.0
        health_ms = 0.0
        backend_stage: str | None = None

        if not self.enabled:
            return self._build_state(
                label=getattr(object_cam0, "label", None),
                template_id=None,
                object_detected=bool(getattr(object_cam0, "object_detected", False)),
                tracking_confidence=0.0,
                mask_iou=0.0,
                depth_inlier_ratio=0.0,
                backend_available=bool(getattr(self.backend, "available", False)),
            )

        frame = self._anchor_frame_override if self._anchor_frame_override is not None else snapshot.cam0
        object_mask = self._anchor_mask_override
        self._anchor_frame_override = None
        self._anchor_mask_override = None

        label = getattr(object_cam0, "label", None)
        object_detected = bool(getattr(object_cam0, "object_detected", False))
        merged_point_count = int(getattr(merged_object, "merged_point_count", 0))
        merged_points = np.asarray(getattr(merged_object, "merged_points_base", []), dtype=np.float32).reshape((-1, 3))
        template = self.template_library.resolve(label)

        if template is None:
            self._reset_tracking(mode=POSE_TRACKING_UNINITIALIZED, reason="template_missing")
            return self._build_state(
                label=label,
                template_id=None,
                object_detected=object_detected,
                tracking_confidence=0.0,
                mask_iou=0.0,
                depth_inlier_ratio=0.0,
                backend_available=bool(getattr(self.backend, "available", False)),
            )

        label_changed = self._template is not None and self._template.spec.label != template.spec.label
        if label_changed:
            self._reset_tracking(mode=POSE_TRACKING_REINIT_PENDING, reason="label_changed")

        self._template = template

        if object_mask is None or not np.any(object_mask):
            self._reset_tracking(mode=POSE_TRACKING_REINIT_PENDING, reason="mask_missing")
            return self._build_state(
                label=label,
                template_id=template.spec.template_id,
                object_detected=object_detected,
                tracking_confidence=0.0,
                mask_iou=0.0,
                depth_inlier_ratio=0.0,
                backend_available=bool(getattr(self.backend, "available", False)),
            )

        if merged_point_count < self.min_merged_points_for_tracking:
            self._reset_tracking(mode=POSE_TRACKING_REINIT_PENDING, reason="merged_points_insufficient")
            return self._build_state(
                label=label,
                template_id=template.spec.template_id,
                object_detected=object_detected,
                tracking_confidence=0.0,
                mask_iou=0.0,
                depth_inlier_ratio=0.0,
                backend_available=bool(getattr(self.backend, "available", False)),
            )

        scale_start = time.perf_counter()
        scale_estimate = self.scale_estimator.estimate(template, merged_points)
        scale_ms = (time.perf_counter() - scale_start) * 1000.0
        self.last_debug = PoseTrackingDebug(
            anchor_mask=np.asarray(object_mask, dtype=bool),
            stable_scale_ready=scale_estimate is not None,
            scale_estimation_ms=scale_ms,
        )
        if self._scale_xyz is None and scale_estimate is None:
            self._mode = POSE_TRACKING_WAITING_STABLE_SCALE
            return self._build_state(
                label=label,
                template_id=template.spec.template_id,
                object_detected=object_detected,
                tracking_confidence=0.0,
                mask_iou=0.0,
                depth_inlier_ratio=0.0,
                backend_available=bool(getattr(self.backend, "available", False)),
            )

        if scale_estimate is not None and (
            self._scale_xyz is None
            or self._mode in {POSE_TRACKING_UNINITIALIZED, POSE_TRACKING_REINIT_PENDING, POSE_TRACKING_WAITING_STABLE_SCALE}
        ):
            try:
                self._apply_scale_estimate(template, scale_estimate.scale_xyz)
            except Exception as exc:
                self._reset_tracking(mode=POSE_TRACKING_REINIT_PENDING, reason=f"backend_error:{exc}")
                return self._build_state(
                    label=label,
                    template_id=template.spec.template_id,
                    object_detected=object_detected,
                    tracking_confidence=0.0,
                    mask_iou=0.0,
                    depth_inlier_ratio=0.0,
                    backend_available=False,
                )

        try:
            backend_start = time.perf_counter()
            if self._mode != POSE_TRACKING_TRACKING or self._pose_cam0 is None:
                self._mode = POSE_TRACKING_REGISTERING
                backend_stage = "register"
                pose_cam0 = self.backend.register(frame, object_mask, self.registration_iterations)
            else:
                backend_stage = "track"
                pose_cam0 = self.backend.track(frame, self.track_iterations)
            backend_ms = (time.perf_counter() - backend_start) * 1000.0
        except Exception as exc:
            backend_ms = (time.perf_counter() - backend_start) * 1000.0 if "backend_start" in locals() else 0.0
            self.last_debug = PoseTrackingDebug(
                anchor_mask=np.asarray(object_mask, dtype=bool),
                stable_scale_ready=scale_estimate is not None,
                backend_stage=backend_stage,
                scale_estimation_ms=scale_ms,
                backend_call_ms=backend_ms,
                total_update_ms=(time.perf_counter() - update_start) * 1000.0,
            )
            self._reset_tracking(mode=POSE_TRACKING_REINIT_PENDING, reason=f"backend_error:{exc}")
            return self._build_state(
                label=label,
                template_id=template.spec.template_id,
                object_detected=object_detected,
                tracking_confidence=0.0,
                mask_iou=0.0,
                depth_inlier_ratio=0.0,
                backend_available=False,
            )

        self._pose_cam0 = np.asarray(pose_cam0, dtype=np.float32).reshape(4, 4)
        self._pose_base = np.asarray(self.transform_chain.t_base_cam0 @ self._pose_cam0, dtype=np.float32)
        health_start = time.perf_counter()
        metrics = self._compute_tracking_health(frame, np.asarray(object_mask, dtype=bool), object_detected)
        health_ms = (time.perf_counter() - health_start) * 1000.0
        low_confidence = (
            not object_detected
            or metrics["mask_iou"] < self.mask_iou_threshold
            or metrics["depth_inlier_ratio"] < self.depth_inlier_threshold
        )
        if low_confidence:
            self._low_confidence_streak += 1
        else:
            self._low_confidence_streak = 0
            self._mode = POSE_TRACKING_TRACKING
            self._reinit_reason = None

        self.last_debug = PoseTrackingDebug(
            anchor_mask=np.asarray(object_mask, dtype=bool),
            projected_pixels=metrics["projected_pixels"],
            rendered_mask=metrics["rendered_mask"],
            mask_iou=float(metrics["mask_iou"]),
            depth_inlier_ratio=float(metrics["depth_inlier_ratio"]),
            low_confidence_streak=int(self._low_confidence_streak),
            stable_scale_ready=scale_estimate is not None,
            backend_stage=backend_stage,
            scale_estimation_ms=scale_ms,
            backend_call_ms=backend_ms,
            health_check_ms=health_ms,
            total_update_ms=(time.perf_counter() - update_start) * 1000.0,
        )

        if self._low_confidence_streak >= self.low_conf_frames:
            self._reset_tracking(mode=POSE_TRACKING_REINIT_PENDING, reason="low_confidence")
            self.scale_estimator.reset_template(template.spec.template_id)

        return self._build_state(
            label=label,
            template_id=template.spec.template_id,
            object_detected=object_detected,
            tracking_confidence=float(metrics["tracking_confidence"]),
            mask_iou=float(metrics["mask_iou"]),
            depth_inlier_ratio=float(metrics["depth_inlier_ratio"]),
            backend_available=bool(getattr(self.backend, "available", True)),
        )

    def _apply_scale_estimate(self, template: TemplateModel, scale_xyz: np.ndarray) -> None:
        scale_xyz = np.asarray(scale_xyz, dtype=np.float32).reshape(3)
        self._scale_xyz = scale_xyz
        mesh = template.mesh.copy()
        mesh.vertices = self._scale_points_about_centroid(np.asarray(mesh.vertices, dtype=np.float32), scale_xyz)
        self._scaled_mesh = mesh
        self._scaled_mesh_points = self._build_surface_sample_points(mesh)
        self.backend.reset_object(mesh)
        self._pose_cam0 = None
        self._pose_base = None
        self._low_confidence_streak = 0
        self._mode = POSE_TRACKING_REGISTERING
        self._reinit_reason = None

    def _reset_tracking(self, *, mode: str, reason: str | None) -> None:
        if self._template is not None:
            self.scale_estimator.reset_template(self._template.spec.template_id)
        self._mode = mode
        self._reinit_reason = reason
        self._pose_cam0 = None
        self._pose_base = None
        self._scale_xyz = None
        self._scaled_mesh = None
        self._scaled_mesh_points = np.empty((0, 3), dtype=np.float32)
        self._low_confidence_streak = 0

    def _build_state(
        self,
        *,
        label: str | None,
        template_id: str | None,
        object_detected: bool,
        tracking_confidence: float,
        mask_iou: float,
        depth_inlier_ratio: float,
        backend_available: bool,
    ) -> PoseTrackingState:
        return PoseTrackingState(
            anchor_camera_id=self.anchor_camera_id,
            object_detected=bool(object_detected),
            label=label,
            template_id=template_id,
            scale_xyz=None if self._scale_xyz is None else tuple(float(v) for v in self._scale_xyz),
            pose_cam0=_matrix_to_tuple(self._pose_cam0),
            pose_base=_matrix_to_tuple(self._pose_base),
            tracking_confidence=float(tracking_confidence),
            mask_iou=float(mask_iou),
            depth_inlier_ratio=float(depth_inlier_ratio),
            mode=self._mode,
            reinit_reason=self._reinit_reason,
            backend_available=bool(backend_available),
            valid=bool(self._pose_cam0 is not None),
        )

    def _compute_tracking_health(
        self,
        frame: FrameBundle,
        anchor_mask: np.ndarray,
        object_detected: bool,
    ) -> dict[str, Any]:
        if self._pose_cam0 is None or len(self._scaled_mesh_points) == 0:
            return {
                "tracking_confidence": 0.0,
                "mask_iou": 0.0,
                "depth_inlier_ratio": 0.0,
                "projected_pixels": np.empty((0, 2), dtype=np.int32),
                "rendered_mask": np.zeros(anchor_mask.shape, dtype=bool),
            }

        points_cam = self._transform_points(self._scaled_mesh_points, self._pose_cam0)
        projected_pixels, valid_depths = self._project_camera_points(points_cam, frame.intrinsics, anchor_mask.shape[1], anchor_mask.shape[0])
        rendered_mask = np.zeros(anchor_mask.shape, dtype=np.uint8)
        if len(projected_pixels) > 0:
            for u, v in projected_pixels.tolist():
                _draw_disk(rendered_mask, int(u), int(v), radius=2)
        rendered_mask_bool = rendered_mask.astype(bool)
        union = np.logical_or(rendered_mask_bool, anchor_mask)
        if np.any(union):
            mask_iou = float(np.count_nonzero(np.logical_and(rendered_mask_bool, anchor_mask))) / float(np.count_nonzero(union))
        else:
            mask_iou = 0.0

        depth_inlier_ratio = 0.0
        if len(projected_pixels) > 0:
            sampled_depth = frame.depth_image_m[projected_pixels[:, 1], projected_pixels[:, 0]]
            valid = np.isfinite(sampled_depth) & (sampled_depth > 1e-4)
            if np.any(valid):
                depth_residual = np.abs(sampled_depth[valid] - valid_depths[valid])
                depth_inlier_ratio = float(np.count_nonzero(depth_residual <= self.depth_tolerance_m)) / float(np.count_nonzero(valid))

        detection_score = 1.0 if object_detected else 0.0
        tracking_confidence = float(np.clip(0.2 * detection_score + 0.4 * mask_iou + 0.4 * depth_inlier_ratio, 0.0, 1.0))
        return {
            "tracking_confidence": tracking_confidence,
            "mask_iou": float(mask_iou),
            "depth_inlier_ratio": float(depth_inlier_ratio),
            "projected_pixels": projected_pixels,
            "rendered_mask": rendered_mask_bool,
        }

    def _build_surface_sample_points(self, mesh: Any) -> np.ndarray:
        try:
            sampled = mesh.sample(self.surface_sample_count).astype(np.float32)
            if len(sampled) > 0:
                return sampled
        except Exception:
            pass
        return np.asarray(mesh.vertices, dtype=np.float32).copy()

    @staticmethod
    def _scale_points_about_centroid(points_xyz: np.ndarray, scale_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
        scale = np.asarray(scale_xyz, dtype=np.float32).reshape(1, 3)
        centroid = np.mean(points, axis=0, keepdims=True)
        return (points - centroid) * scale + centroid

    @staticmethod
    def _transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
        rotation = np.asarray(transform[:3, :3], dtype=np.float32)
        translation = np.asarray(transform[:3, 3], dtype=np.float32).reshape(1, 3)
        return (points @ rotation.T + translation).astype(np.float32)

    @staticmethod
    def _project_camera_points(
        points_cam: np.ndarray,
        intrinsics: dict[str, float],
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_cam, dtype=np.float32).reshape((-1, 3))
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
        z = points[:, 2]
        valid = np.isfinite(z) & (z > 1e-6)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
        points = points[valid]
        z = z[valid]
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        u = np.round((points[:, 0] * fx / z) + cx).astype(np.int32)
        v = np.round((points[:, 1] * fy / z) + cy).astype(np.int32)
        in_bounds = (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
        if not np.any(in_bounds):
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
        return np.stack([u[in_bounds], v[in_bounds]], axis=1), z[in_bounds].astype(np.float32)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "TemplateSpec",
    "TemplateModel",
    "StableScaleEstimate",
    "PoseTrackingDebug",
    "TemplateLibrary",
    "ScaleEstimator",
    "LazyFoundationPoseBackend",
    "FoundationPoseTracker",
]
