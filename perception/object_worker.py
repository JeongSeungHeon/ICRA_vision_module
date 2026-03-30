"""Reusable per-camera object perception worker."""

from __future__ import annotations

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


@dataclass
class ObjectWorkerDebug:
    camera_id: int
    selected_instance_count: int
    infer_ms: float
    combined_mask: np.ndarray
    points_camera: np.ndarray
    points_base: np.ndarray
    colors_rgb: np.ndarray


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
        self.last_debug: ObjectWorkerDebug | None = None

    @classmethod
    def from_config(cls, camera_id: int, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ObjectWorker":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        object_cfg = config.get("perception", {}).get("object", {})
        segmentation_cfg = object_cfg.get("segmentation", {})
        segmentation_engine = SegmentationEngine(
            model_name=segmentation_cfg.get("model_name", "yoloe-26l-seg.pt"),
            prompt_classes=segmentation_cfg.get("prompt_classes", ["cup"]),
            imgsz=int(segmentation_cfg.get("imgsz", 640)),
            conf=float(segmentation_cfg.get("conf", 0.25)),
            iou=float(segmentation_cfg.get("iou", 0.45)),
            max_det=int(segmentation_cfg.get("max_det", 100)),
            device=segmentation_cfg.get("device"),
            classes=segmentation_cfg.get("classes"),
            half=bool(segmentation_cfg.get("half", False)),
            retina_masks=bool(segmentation_cfg.get("retina_masks", True)),
        )
        transform_chain = load_transform_chain(config_path)
        return cls(
            camera_id=camera_id,
            segmentation_engine=segmentation_engine,
            transform_chain=transform_chain,
            config=config,
        )

    def process_frame(self, frame_bundle: FrameBundle, frame_id: int = -1) -> ObjectState:
        segmentation_result = self.segmentation_engine.predict(frame_bundle.color_image)
        selected_instances = select_instances(
            segmentation_result.instances,
            mode=self.selection_mode,
            class_names=self.selection_class_names or None,
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
            infer_ms=float(segmentation_result.infer_ms),
            combined_mask=combined_mask,
            points_camera=np.asarray(points_camera, dtype=np.float32),
            points_base=np.asarray(points_base, dtype=np.float32),
            colors_rgb=np.asarray(colors_rgb, dtype=np.uint8),
        )
        return state

    def process_latest(self, sensor_hub: Any) -> ObjectState:
        frame_bundle = sensor_hub.get_latest_cam0() if self.camera_id == 0 else sensor_hub.get_latest_cam1()
        sensor_state = sensor_hub.get_latest_sensor_state(self.camera_id)
        return self.process_frame(frame_bundle, frame_id=sensor_state.frame_id)

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
    "ObjectWorker",
    "ObjectWorkerCam0",
    "ObjectWorkerCam1",
]
