"""Reusable per-camera object perception worker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibration.extrinsics import TransformChain, load_transform_chain
from object_pt_extraction.pointcloud_utils import (
    build_point_cloud_from_instances,
    remove_radius_outliers,
    remove_statistical_outliers,
    summarize_point_cloud,
    voxel_downsample_point_cloud,
)
from object_pt_extraction.segmentation_engine import SegmentationEngine, select_instances
from system.shared_state import ObjectState
from utils.realsense_stream import FrameBundle

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
DEFAULT_CLASS_LOCK_FRAMES = 15
CAMERA_HANDEDNESS_FLIP = {
    0: True,
    1: True,
}
REFERENCE_CAMERA_BY_CORRECTED_HAND = {
    "left": 0,
    "right": 1,
}


@dataclass
class ObjectWorkerDebug:
    camera_id: int
    selected_instance_count: int
    infer_ms: float
    all_class_names: tuple[str, ...]
    selected_class_names: tuple[str, ...]
    combined_mask: np.ndarray
    points_camera: np.ndarray
    points_base: np.ndarray
    colors_rgb: np.ndarray
    locked_label: str | None = None
    label_history: tuple[str, ...] = ()


class TemporalClassLocker:
    """Lock a segmented class once the same label is stable for several frames."""

    def __init__(self, stable_frames: int = DEFAULT_CLASS_LOCK_FRAMES) -> None:
        self.stable_frames = max(int(stable_frames), 1)
        self.history: deque[str] = deque(maxlen=self.stable_frames)
        self.locked_label: str | None = None

    def update(self, label: str | None, segmented: bool) -> str | None:
        if self.locked_label is not None:
            return self.locked_label

        if segmented and label:
            normalized_label = str(label)
            self.history.append(normalized_label)
            if len(self.history) == self.stable_frames and len(set(self.history)) == 1:
                self.locked_label = normalized_label
                return self.locked_label

        return label

    def reset(self) -> None:
        self.history.clear()
        self.locked_label = None


def _normalize_handedness(handedness: str | None) -> str | None:
    if handedness is None:
        return None
    normalized = str(handedness).strip().lower()
    if normalized in {"left", "right"}:
        return normalized
    return normalized or None


def _format_handedness_for_log(handedness: str | None) -> str | None:
    normalized = _normalize_handedness(handedness)
    if normalized == "left":
        return "Left"
    if normalized == "right":
        return "Right"
    return normalized


def correct_handedness(raw_handedness: str | None, camera_id: int | None) -> str | None:
    normalized = _normalize_handedness(raw_handedness)
    if normalized not in {"left", "right"} or camera_id is None:
        return normalized
    if CAMERA_HANDEDNESS_FLIP.get(int(camera_id), False):
        return "left" if normalized == "right" else "right"
    return normalized


class HandednessAwareObjectClassLock:
    """Lock one canonical object class from the hand-occlusion-favored camera."""

    def __init__(self) -> None:
        self.locked_class: str | None = None
        self.locked_from_camera: int | None = None
        self._last_missing_match_cameras: set[int] = set()

    def reset(self) -> None:
        if self.locked_class is not None:
            print("[class-lock] reset locked class", flush=True)
        self.locked_class = None
        self.locked_from_camera = None
        self._last_missing_match_cameras.clear()

    def process_states(
        self,
        *,
        selected_hand: Any,
        hand_cam0: Any,
        hand_cam1: Any,
        object_cam0: ObjectState,
        object_cam1: ObjectState,
        object_worker_cam0: Any,
        object_worker_cam1: Any,
    ) -> tuple[ObjectState, ObjectState]:
        hand_camera, raw_handedness = self._resolve_hand_for_lock(selected_hand, hand_cam0, hand_cam1)
        corrected_handedness = correct_handedness(raw_handedness, hand_camera)

        if self.locked_class is None:
            reference_camera = self._reference_camera_for_handedness(corrected_handedness)
            if reference_camera is not None:
                reference_object = object_cam0 if int(reference_camera) == 0 else object_cam1
                if bool(getattr(reference_object, "valid", False)) and getattr(reference_object, "label", None):
                    self.locked_class = str(reference_object.label)
                    self.locked_from_camera = int(reference_camera)
                    print(
                        "[class-lock] "
                        f"raw_handedness={_format_handedness_for_log(raw_handedness)} "
                        f"corrected_handedness={_format_handedness_for_log(corrected_handedness)} "
                        f"selected_ref_cam=cam{int(reference_camera)} "
                        f"locked_class={self.locked_class}",
                        flush=True,
                    )

        if self.locked_class is not None:
            object_cam0 = object_worker_cam0.reselect_last_frame_by_label(self.locked_class)
            object_cam1 = object_worker_cam1.reselect_last_frame_by_label(self.locked_class)
            self._log_missing_matches_once(object_cam0, object_cam1)

        return object_cam0, object_cam1

    @staticmethod
    def _reference_camera_for_handedness(handedness: str | None) -> int | None:
        return REFERENCE_CAMERA_BY_CORRECTED_HAND.get(str(handedness))

    @staticmethod
    def _resolve_hand_for_lock(selected_hand: Any, hand_cam0: Any, hand_cam1: Any) -> tuple[int | None, str | None]:
        selected_camera = getattr(selected_hand, "selected_camera", None)
        if selected_camera is not None and bool(getattr(selected_hand, "valid", False)):
            return int(selected_camera), getattr(selected_hand, "handedness", None)

        hand_state = None
        if selected_camera is not None:
            hand_state = hand_cam0 if int(selected_camera) == 0 else hand_cam1
        if hand_state is not None and bool(getattr(hand_state, "valid", False)):
            return int(selected_camera), getattr(hand_state, "handedness", None)

        valid_hand_states = [
            state
            for state in (hand_cam0, hand_cam1)
            if bool(getattr(state, "valid", False))
        ]
        if valid_hand_states:
            best_state = max(valid_hand_states, key=lambda state: float(getattr(state, "confidence", 0.0)))
            return int(getattr(best_state, "camera_id")), getattr(best_state, "handedness", None)

        if selected_camera is not None:
            return int(selected_camera), getattr(selected_hand, "handedness", None)
        return None, getattr(selected_hand, "handedness", None)

    def _log_missing_matches_once(self, object_cam0: ObjectState, object_cam1: ObjectState) -> None:
        states = {0: object_cam0, 1: object_cam1}
        missing_cameras = {
            camera_id
            for camera_id, state in states.items()
            if not bool(getattr(state, "valid", False))
        }
        for camera_id in sorted(missing_cameras - self._last_missing_match_cameras):
            other_camera = 1 - camera_id
            if not bool(getattr(states[other_camera], "valid", False)):
                continue
            print(
                "[class-lock] "
                f"cam{camera_id} has no usable instance matching locked_class={self.locked_class}; "
                f"using cam{other_camera} only",
                flush=True,
            )
        self._last_missing_match_cameras = missing_cameras


class ObjectWorker:
    def __init__(
        self,
        camera_id: int,
        segmentation_engine: SegmentationEngine,
        transform_chain: TransformChain,
        config: dict[str, Any],
    ) -> None:
        self.camera_id = int(camera_id)
        self.segmentation_engine = segmentation_engine
        self.transform_chain = transform_chain
        self.config = config

        object_cfg = config.get("perception", {}).get("object", {})
        point_cfg = object_cfg.get("point_cloud", {})
        segmentation_cfg = object_cfg.get("segmentation", {})

        self.target_label = object_cfg.get("target_label")
        self.confidence_threshold = float(object_cfg.get("confidence_threshold", 0.5))
        self.selection_mode = segmentation_cfg.get("selection_mode", "highest_score")
        self.selection_class_names = list(segmentation_cfg.get("selection_class_names", []))
        self.prefer_wine_glass_over_cup = bool(segmentation_cfg.get("prefer_wine_glass_over_cup", False))
        self.stride = int(point_cfg.get("stride", 2))
        self.max_points = int(point_cfg.get("max_points", 20000))
        self.min_depth_m = float(point_cfg.get("min_depth_m", 0.10))
        self.max_depth_m = float(point_cfg.get("max_depth_m", 1.50))
        self.voxel_size_m = float(point_cfg.get("per_camera_voxel_size_m", 0.005))
        self.outlier_method = str(point_cfg.get("outlier_method", "statistical")).strip().lower()
        self.outlier_nb_neighbors = int(point_cfg.get("outlier_nb_neighbors", 20))
        self.outlier_std_ratio = float(point_cfg.get("outlier_std_ratio", 2.0))
        self.outlier_radius_m = float(point_cfg.get("outlier_radius_m", 0.01))
        self.outlier_min_neighbors = int(point_cfg.get("outlier_min_neighbors", 8))
        self.min_points_per_camera = int(point_cfg.get("min_points_per_camera", 300))
        # Kept for compatibility with older tests/debug fields; task-level
        # HandednessAwareObjectClassLock now owns class stability.
        self.class_locker = TemporalClassLocker(DEFAULT_CLASS_LOCK_FRAMES)
        self.last_debug: ObjectWorkerDebug | None = None
        self._last_frame_bundle: FrameBundle | None = None
        self._last_raw_instances: list[Any] = []
        self._last_infer_ms: float = 0.0
        self._last_frame_id: int = -1

    @classmethod
    def from_config(cls, camera_id: int, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ObjectWorker":
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        object_cfg = config.get("perception", {}).get("object", {})
        segmentation_cfg = object_cfg.get("segmentation", {})
        model_name = segmentation_cfg.get("model_name", "yoloe-26l-seg.pt")
        model_path = Path(str(model_name))
        repo_relative_model_path = config_path.parent.parent / model_path
        if not model_path.is_absolute() and repo_relative_model_path.exists():
            model_name = str(repo_relative_model_path)
        segmentation_engine = SegmentationEngine(
            model_name=model_name,
            prompt_classes=segmentation_cfg.get("prompt_classes", ["cup"]),
            imgsz=int(segmentation_cfg.get("imgsz", 640)),
            conf=float(segmentation_cfg.get("conf", 0.25)),
            iou=float(segmentation_cfg.get("iou", 0.45)),
            max_det=int(segmentation_cfg.get("max_det", 100)),
            device=segmentation_cfg.get("device"),
            classes=segmentation_cfg.get("classes"),
            half=bool(segmentation_cfg.get("half", False)),
            retina_masks=bool(segmentation_cfg.get("retina_masks", True)),
            preprocess_config=dict(segmentation_cfg.get("preprocess", {}) or {}),
        )
        transform_chain = load_transform_chain(config_path)
        return cls(
            camera_id=camera_id,
            segmentation_engine=segmentation_engine,
            transform_chain=transform_chain,
            config=config,
        )

    def process_frame(
        self,
        frame_bundle: FrameBundle,
        frame_id: int = -1,
        class_name_filter: str | None = None,
    ) -> ObjectState:
        segmentation_result = self.segmentation_engine.predict(frame_bundle.color_image)
        self._last_frame_bundle = frame_bundle
        self._last_raw_instances = list(segmentation_result.instances)
        self._last_infer_ms = float(segmentation_result.infer_ms)
        self._last_frame_id = int(frame_id)
        return self._build_state_from_instances(
            frame_bundle=frame_bundle,
            frame_id=frame_id,
            raw_instances=segmentation_result.instances,
            infer_ms=float(segmentation_result.infer_ms),
            class_name_filter=class_name_filter,
        )

    def reselect_last_frame_by_label(self, label: str) -> ObjectState:
        if self._last_frame_bundle is None:
            return ObjectState(camera_id=self.camera_id, frame_id=self._last_frame_id, label=str(label), valid=False)
        return self._build_state_from_instances(
            frame_bundle=self._last_frame_bundle,
            frame_id=self._last_frame_id,
            raw_instances=self._last_raw_instances,
            infer_ms=self._last_infer_ms,
            class_name_filter=str(label),
        )

    def _build_state_from_instances(
        self,
        *,
        frame_bundle: FrameBundle,
        frame_id: int,
        raw_instances: list[Any],
        infer_ms: float,
        class_name_filter: str | None = None,
    ) -> ObjectState:
        class_names = [str(class_name_filter)] if class_name_filter else (self.selection_class_names or None)
        selected_instances = select_instances(
            raw_instances,
            mode=self.selection_mode,
            class_names=class_names,
            prefer_wine_glass_over_cup=bool(self.prefer_wine_glass_over_cup and class_name_filter is None),
        )
        combined_mask, points_camera, _, _, colors_rgb = build_point_cloud_from_instances(
            frame_bundle.color_image,
            frame_bundle.depth_image_m,
            frame_bundle.intrinsics,
            selected_instances,
            stride=self.stride,
            max_points=self.max_points,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )

        points_base = self.transform_chain.transform_points_camera_to_base(self.camera_id, points_camera)
        points_base, colors_rgb = self._postprocess_points(points_base, colors_rgb)
        summary = summarize_point_cloud(points_base)
        label, confidence = self._extract_primary_detection(selected_instances)
        object_detected = (
            len(selected_instances) > 0
            and len(points_base) >= self.min_points_per_camera
            and confidence >= self.confidence_threshold
        )
        if class_name_filter is not None:
            label = str(class_name_filter) if label is None else label
        centroid_base = None
        if object_detected and summary["point_count"] > 0:
            centroid_base = tuple(float(value) for value in summary["centroid_xyz"])

        state = ObjectState(
            camera_id=self.camera_id,
            frame_id=int(frame_id),
            object_detected=bool(object_detected),
            label=label,
            confidence=float(confidence),
            centroid_base=centroid_base,
            point_count=int(summary["point_count"]),
            points_base=[tuple(float(v) for v in point) for point in points_base],
            timestamp=float(frame_bundle.timestamp_ms) / 1000.0,
            valid=bool(object_detected),
        )
        self.last_debug = ObjectWorkerDebug(
            camera_id=self.camera_id,
            selected_instance_count=len(selected_instances),
            infer_ms=float(infer_ms),
            all_class_names=tuple(str(instance.class_name) for instance in raw_instances),
            selected_class_names=tuple(str(instance.class_name) for instance in selected_instances),
            combined_mask=combined_mask,
            points_camera=np.asarray(points_camera, dtype=np.float32),
            points_base=np.asarray(points_base, dtype=np.float32),
            colors_rgb=np.asarray(colors_rgb, dtype=np.uint8),
            locked_label=self.class_locker.locked_label,
            label_history=tuple(self.class_locker.history),
        )
        return state

    def process_latest(self, sensor_hub: Any) -> ObjectState:
        frame_bundle = sensor_hub.get_latest_cam0() if self.camera_id == 0 else sensor_hub.get_latest_cam1()
        sensor_state = sensor_hub.get_latest_sensor_state(self.camera_id)
        return self.process_frame(frame_bundle, frame_id=sensor_state.frame_id)

    def reset(self) -> None:
        self.class_locker.reset()
        self.last_debug = None

    def _postprocess_points(self, points_base: np.ndarray, colors_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        processed_points = np.asarray(points_base, dtype=np.float32).reshape((-1, 3))
        processed_colors = np.asarray(colors_rgb, dtype=np.uint8).reshape((-1, 3))
        if len(processed_points) == 0:
            return processed_points, processed_colors

        if self.voxel_size_m > 0.0:
            processed_points, processed_colors = voxel_downsample_point_cloud(
                processed_points,
                processed_colors,
                voxel_size_m=self.voxel_size_m,
            )

        if self.outlier_method == "statistical":
            processed_points, processed_colors = remove_statistical_outliers(
                processed_points,
                processed_colors,
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio,
            )
        elif self.outlier_method == "radius":
            processed_points, processed_colors = remove_radius_outliers(
                processed_points,
                processed_colors,
                radius_m=self.outlier_radius_m,
                min_neighbors=self.outlier_min_neighbors,
            )
        elif self.outlier_method not in {"none", "off"}:
            raise ValueError(f"Unsupported outlier method: {self.outlier_method}")

        return processed_points.astype(np.float32), processed_colors.astype(np.uint8)

    @staticmethod
    def _extract_primary_detection(selected_instances: list[Any]) -> tuple[str | None, float]:
        if not selected_instances:
            return None, 0.0
        primary = max(selected_instances, key=lambda instance: float(instance.score))
        return str(primary.class_name), float(primary.score)


class ObjectWorkerCam0(ObjectWorker):
    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ObjectWorkerCam0":
        worker = super().from_config(camera_id=0, config_path=config_path)
        return cls(worker.camera_id, worker.segmentation_engine, worker.transform_chain, worker.config)


class ObjectWorkerCam1(ObjectWorker):
    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ObjectWorkerCam1":
        worker = super().from_config(camera_id=1, config_path=config_path)
        return cls(worker.camera_id, worker.segmentation_engine, worker.transform_chain, worker.config)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ObjectWorkerDebug",
    "TemporalClassLocker",
    "HandednessAwareObjectClassLock",
    "correct_handedness",
    "ObjectWorker",
    "ObjectWorkerCam0",
    "ObjectWorkerCam1",
]
