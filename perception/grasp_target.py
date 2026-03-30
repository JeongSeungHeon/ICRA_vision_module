"""Grasp target computation from merged object geometry and selected hand state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from system.shared_state import FusionState, GraspTargetState, MergedObjectState, SelectedHandState

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
HEIGHT_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class GraspPlannerDebug:
    raw_candidate_count: int
    search_radius_candidate_count: int
    clearance_candidate_count: int
    used_filtered_object_centroid: bool
    used_filtered_hand_center: bool
    chosen_distance_to_centroid_m: float | None
    chosen_hand_height_clearance_m: float | None
    selection_reason: str


class GraspTargetPlanner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        grasp_cfg = config.get("grasp", {})
        frame_cfg = config.get("frames", {}).get("height_axis", {})

        self.fixed_orientation_format = str(grasp_cfg.get("fixed_orientation_format", "rotvec"))
        self.fixed_orientation_base = tuple(float(v) for v in grasp_cfg.get("fixed_orientation_base", [0.0, 0.0, 0.0]))
        self.candidate_search_radius_from_centroid_m = float(
            grasp_cfg.get("candidate_search_radius_from_centroid_m", 0.05)
        )
        self.min_hand_height_clearance_m = float(grasp_cfg.get("min_hand_height_clearance_m", 0.03))
        self.no_candidate_action = str(grasp_cfg.get("no_candidate_action", "hold"))
        self.height_axis_name = str(frame_cfg.get("name", "z")).strip().lower()
        if self.height_axis_name not in HEIGHT_AXIS_TO_INDEX:
            raise ValueError(f"Unsupported height axis: {self.height_axis_name}")
        self.height_axis_index = HEIGHT_AXIS_TO_INDEX[self.height_axis_name]
        self.last_debug: GraspPlannerDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "GraspTargetPlanner":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def process_states(
        self,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        fusion_state: FusionState | None = None,
    ) -> GraspTargetState:
        points = np.asarray(merged_object.merged_points_base, dtype=np.float32).reshape((-1, 3))
        if len(points) == 0 or not merged_object.valid:
            return self._invalid_state(merged_object, selected_hand, reason="no_merged_object")
        if not selected_hand.valid or selected_hand.palm_center_base is None:
            return self._invalid_state(merged_object, selected_hand, reason="no_selected_hand")

        object_centroid, used_filtered_object_centroid = self._resolve_object_centroid(merged_object, fusion_state)
        hand_center, used_filtered_hand_center = self._resolve_hand_center(selected_hand, fusion_state)
        if object_centroid is None:
            return self._invalid_state(merged_object, selected_hand, reason="no_object_centroid")
        if hand_center is None:
            return self._invalid_state(merged_object, selected_hand, reason="no_hand_center")

        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        raw_candidate_count = int(len(points))
        if raw_candidate_count == 0:
            return self._invalid_state(merged_object, selected_hand, reason="no_finite_candidates")

        centroid_distances = np.linalg.norm(points - object_centroid.reshape(1, 3), axis=1)
        within_radius_mask = centroid_distances <= self.candidate_search_radius_from_centroid_m
        if np.any(within_radius_mask):
            candidate_points = points[within_radius_mask]
            candidate_distances = centroid_distances[within_radius_mask]
            selection_reason = "selected_within_search_radius"
        else:
            candidate_points = points
            candidate_distances = centroid_distances
            selection_reason = "fallback_to_all_candidates"
        search_radius_candidate_count = int(len(candidate_points))

        hand_axis_values = np.abs(candidate_points[:, self.height_axis_index] - hand_center[self.height_axis_index])
        clearance_mask = hand_axis_values >= self.min_hand_height_clearance_m
        clearance_candidate_count = int(np.count_nonzero(clearance_mask))
        if clearance_candidate_count == 0:
            return self._invalid_state(
                merged_object,
                selected_hand,
                reason="no_candidate_after_clearance_filter",
                raw_candidate_count=raw_candidate_count,
                search_radius_candidate_count=search_radius_candidate_count,
                clearance_candidate_count=0,
                used_filtered_object_centroid=used_filtered_object_centroid,
                used_filtered_hand_center=used_filtered_hand_center,
            )

        valid_points = candidate_points[clearance_mask]
        valid_distances = candidate_distances[clearance_mask]
        valid_clearances = hand_axis_values[clearance_mask]
        best_index = int(np.argmin(valid_distances))
        best_point = valid_points[best_index]
        best_distance = float(valid_distances[best_index])
        best_clearance = float(valid_clearances[best_index])

        state = GraspTargetState(
            target_position_base=tuple(float(v) for v in best_point),
            fixed_orientation_base=self.fixed_orientation_base,
            hand_height_clearance_m=best_clearance,
            distance_to_centroid_m=best_distance,
            source_point_count=raw_candidate_count,
            timestamp=max(float(merged_object.timestamp), float(selected_hand.timestamp)),
            valid=True,
        )
        self.last_debug = GraspPlannerDebug(
            raw_candidate_count=raw_candidate_count,
            search_radius_candidate_count=search_radius_candidate_count,
            clearance_candidate_count=clearance_candidate_count,
            used_filtered_object_centroid=used_filtered_object_centroid,
            used_filtered_hand_center=used_filtered_hand_center,
            chosen_distance_to_centroid_m=best_distance,
            chosen_hand_height_clearance_m=best_clearance,
            selection_reason=selection_reason,
        )
        return state

    def _resolve_object_centroid(
        self,
        merged_object: MergedObjectState,
        fusion_state: FusionState | None,
    ) -> tuple[np.ndarray | None, bool]:
        if fusion_state is not None and fusion_state.valid and fusion_state.object_fresh and fusion_state.filtered_object_centroid_base is not None:
            return np.asarray(fusion_state.filtered_object_centroid_base, dtype=np.float32).reshape(3), True
        if merged_object.centroid_base is not None:
            return np.asarray(merged_object.centroid_base, dtype=np.float32).reshape(3), False
        return None, False

    def _resolve_hand_center(
        self,
        selected_hand: SelectedHandState,
        fusion_state: FusionState | None,
    ) -> tuple[np.ndarray | None, bool]:
        if fusion_state is not None and fusion_state.valid and fusion_state.hand_fresh and fusion_state.filtered_hand_center_base is not None:
            return np.asarray(fusion_state.filtered_hand_center_base, dtype=np.float32).reshape(3), True
        if selected_hand.palm_center_base is not None:
            return np.asarray(selected_hand.palm_center_base, dtype=np.float32).reshape(3), False
        return None, False

    def _invalid_state(
        self,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        *,
        reason: str,
        raw_candidate_count: int = 0,
        search_radius_candidate_count: int = 0,
        clearance_candidate_count: int = 0,
        used_filtered_object_centroid: bool = False,
        used_filtered_hand_center: bool = False,
    ) -> GraspTargetState:
        self.last_debug = GraspPlannerDebug(
            raw_candidate_count=raw_candidate_count,
            search_radius_candidate_count=search_radius_candidate_count,
            clearance_candidate_count=clearance_candidate_count,
            used_filtered_object_centroid=used_filtered_object_centroid,
            used_filtered_hand_center=used_filtered_hand_center,
            chosen_distance_to_centroid_m=None,
            chosen_hand_height_clearance_m=None,
            selection_reason=reason,
        )
        return GraspTargetState(
            target_position_base=None,
            fixed_orientation_base=self.fixed_orientation_base,
            hand_height_clearance_m=None,
            distance_to_centroid_m=None,
            source_point_count=int(merged_object.merged_point_count),
            timestamp=max(float(merged_object.timestamp), float(selected_hand.timestamp)),
            valid=False,
        )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HEIGHT_AXIS_TO_INDEX",
    "GraspPlannerDebug",
    "GraspTargetPlanner",
]
