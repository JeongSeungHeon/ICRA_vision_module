"""Merged object point-cloud processing for the dual-camera system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from object_pt_extraction.pointcloud_utils import merge_point_clouds
from system.shared_state import HEIGHT_AXIS_Y, MergedObjectState, ObjectState, Vec3

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
HEIGHT_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
HEIGHT_DIRECTION_TO_SIGN = {"up": 1.0, "positive": 1.0, "+": 1.0, "down": -1.0, "negative": -1.0, "-": -1.0}


@dataclass
class ObjectMergerDebug:
    contributing_camera_ids: tuple[int, ...]
    contributing_point_counts: tuple[int, ...]
    merge_stats: dict[str, Any]
    stable_observation_frames: int
    initial_centroid_locked: bool
    hand_approach_required: bool
    hand_approach_detected: bool


class ObjectMerger:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        object_cfg = config.get("perception", {}).get("object", {})
        point_cfg = object_cfg.get("point_cloud", {})
        activation_cfg = config.get("perception", {}).get("activation", {})
        lift_cfg = activation_cfg.get("lift_detection", {})
        frame_cfg = config.get("frames", {}).get("height_axis", {})

        self.merged_voxel_size_m = float(object_cfg.get("merged_voxel_size_m", 0.005))
        self.outlier_method = str(point_cfg.get("outlier_method", "statistical")).strip().lower()
        self.outlier_nb_neighbors = int(point_cfg.get("outlier_nb_neighbors", 20))
        self.outlier_std_ratio = float(point_cfg.get("outlier_std_ratio", 2.0))
        self.outlier_radius_m = float(point_cfg.get("outlier_radius_m", 0.01))
        self.outlier_min_neighbors = int(point_cfg.get("outlier_min_neighbors", 8))
        self.stable_frames_required = int(activation_cfg.get("stable_frames_required", 5))
        self.lift_threshold_m = float(lift_cfg.get("threshold_m", 0.05))
        self.require_hand_approach_first = bool(lift_cfg.get("require_hand_approach_first", True))
        self.height_axis_name = str(
            lift_cfg.get("axis_name")
            or frame_cfg.get("name")
            or HEIGHT_AXIS_Y
        ).strip().lower()
        if self.height_axis_name not in HEIGHT_AXIS_TO_INDEX:
            raise ValueError(f"Unsupported height axis: {self.height_axis_name}")
        self.height_axis_index = HEIGHT_AXIS_TO_INDEX[self.height_axis_name]
        self.height_axis_direction = str(lift_cfg.get("positive_direction") or frame_cfg.get("positive_direction") or "up").strip().lower()
        if self.height_axis_direction not in HEIGHT_DIRECTION_TO_SIGN:
            raise ValueError(f"Unsupported height axis direction: {self.height_axis_direction}")
        self.height_axis_sign = float(HEIGHT_DIRECTION_TO_SIGN[self.height_axis_direction])

        self._initial_centroid_base: Vec3 | None = None
        self._stable_observation_frames = 0
        self._last_valid_merged_state: MergedObjectState | None = None
        self.last_debug: ObjectMergerDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ObjectMerger":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    @property
    def initial_centroid_base(self) -> Vec3 | None:
        return self._initial_centroid_base

    def reset_initial_centroid(self) -> None:
        self._initial_centroid_base = None
        self._stable_observation_frames = 0

    def process_states(
        self,
        object_cam0: ObjectState,
        object_cam1: ObjectState,
        *,
        hand_approach_detected: bool = False,
    ) -> MergedObjectState:
        valid_states = [state for state in (object_cam0, object_cam1) if self._is_state_usable(state)]
        if not valid_states:
            self._stable_observation_frames = 0
            merged_state = MergedObjectState(
                frame_id_cam0=int(object_cam0.frame_id),
                frame_id_cam1=int(object_cam1.frame_id),
                object_detected=False,
                label=None,
                confidence=0.0,
                centroid_base=None,
                initial_centroid_base=self._initial_centroid_base,
                merged_point_count=0,
                merged_points_base=[],
                object_lifted=False,
                lift_height_delta_m=0.0,
                height_axis_name=self.height_axis_name,
                timestamp=max(float(object_cam0.timestamp), float(object_cam1.timestamp)),
                valid=False,
            )
            self.last_debug = ObjectMergerDebug(
                contributing_camera_ids=tuple(),
                contributing_point_counts=tuple(),
                merge_stats={},
                stable_observation_frames=self._stable_observation_frames,
                initial_centroid_locked=self._initial_centroid_base is not None,
                hand_approach_required=self.require_hand_approach_first,
                hand_approach_detected=bool(hand_approach_detected),
            )
            return merged_state

        point_clouds = []
        colors_rgb_list = []
        contributing_camera_ids: list[int] = []
        contributing_point_counts: list[int] = []
        for state in valid_states:
            points = np.asarray(state.points_base, dtype=np.float32).reshape((-1, 3))
            if len(points) == 0:
                continue
            point_clouds.append(points)
            colors_rgb_list.append(np.zeros((len(points), 3), dtype=np.uint8))
            contributing_camera_ids.append(int(state.camera_id))
            contributing_point_counts.append(int(len(points)))

        if not point_clouds:
            return self.process_states(
                object_cam0.copy_with(valid=False, object_detected=False, points_base=[]),
                object_cam1.copy_with(valid=False, object_detected=False, points_base=[]),
                hand_approach_detected=hand_approach_detected,
            )

        merged_points, _, merge_stats = merge_point_clouds(
            point_clouds,
            colors_rgb_list,
            voxel_size_m=self.merged_voxel_size_m,
            outlier_method=self.outlier_method,
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio,
            radius_m=self.outlier_radius_m,
            min_neighbors=self.outlier_min_neighbors,
        )
        merged_summary = merge_stats["filtered_summary"]
        merged_centroid = merged_summary["centroid_xyz"] if merged_summary["point_count"] > 0 else None

        if merged_centroid is not None:
            self._stable_observation_frames += 1
            if (
                self._initial_centroid_base is None
                and self._stable_observation_frames >= self.stable_frames_required
            ):
                self._initial_centroid_base = tuple(float(v) for v in merged_centroid)
        else:
            self._stable_observation_frames = 0

        lift_height_delta_m = self._compute_lift_delta(merged_centroid)
        lift_allowed = bool(hand_approach_detected) or not self.require_hand_approach_first
        object_lifted = bool(
            merged_centroid is not None
            and self._initial_centroid_base is not None
            and lift_allowed
            and lift_height_delta_m >= self.lift_threshold_m
        )

        best_state = max(valid_states, key=lambda state: float(state.confidence))
        merged_state = MergedObjectState(
            frame_id_cam0=int(object_cam0.frame_id),
            frame_id_cam1=int(object_cam1.frame_id),
            object_detected=merged_summary["point_count"] > 0,
            label=best_state.label,
            confidence=max(float(state.confidence) for state in valid_states),
            centroid_base=None if merged_centroid is None else tuple(float(v) for v in merged_centroid),
            initial_centroid_base=self._initial_centroid_base,
            merged_point_count=int(merged_summary["point_count"]),
            merged_points_base=[tuple(float(v) for v in point) for point in merged_points],
            object_lifted=object_lifted,
            lift_height_delta_m=float(lift_height_delta_m),
            height_axis_name=self.height_axis_name,
            timestamp=max(float(state.timestamp) for state in valid_states),
            valid=merged_summary["point_count"] > 0,
        )
        self._last_valid_merged_state = merged_state if merged_state.valid else self._last_valid_merged_state
        self.last_debug = ObjectMergerDebug(
            contributing_camera_ids=tuple(contributing_camera_ids),
            contributing_point_counts=tuple(contributing_point_counts),
            merge_stats=merge_stats,
            stable_observation_frames=self._stable_observation_frames,
            initial_centroid_locked=self._initial_centroid_base is not None,
            hand_approach_required=self.require_hand_approach_first,
            hand_approach_detected=bool(hand_approach_detected),
        )
        return merged_state

    def _compute_lift_delta(self, merged_centroid: np.ndarray | None) -> float:
        if merged_centroid is None or self._initial_centroid_base is None:
            return 0.0
        current_height = float(np.asarray(merged_centroid, dtype=np.float32)[self.height_axis_index])
        initial_height = float(np.asarray(self._initial_centroid_base, dtype=np.float32)[self.height_axis_index])
        raw_delta = current_height - initial_height
        return self.height_axis_sign * raw_delta

    @staticmethod
    def _is_state_usable(state: ObjectState) -> bool:
        return bool(state.valid and state.object_detected and state.point_count > 0 and state.points_base)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HEIGHT_AXIS_TO_INDEX",
    "HEIGHT_DIRECTION_TO_SIGN",
    "ObjectMergerDebug",
    "ObjectMerger",
]
