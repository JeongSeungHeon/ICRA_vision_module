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
    used_temporal_hold: bool = False
    selected_candidate_index: int = -1
    target_xy_locked: bool = False
    dropout_hold_active: bool = False


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
        selection_cfg = grasp_cfg.get("target_selection", {})
        self.temporal_selection_enabled = bool(selection_cfg.get("enabled", True))
        self.switch_margin = float(selection_cfg.get("switch_margin", 0.01))
        self.hold_frames = max(0, int(selection_cfg.get("hold_frames", 6)))
        self.dropout_hold_frames = max(0, int(selection_cfg.get("dropout_hold_frames", 8)))
        self.score_previous_distance_weight = float(selection_cfg.get("score_previous_distance_weight", 0.10))
        self.height_axis_name = str(frame_cfg.get("name", "z")).strip().lower()
        if self.height_axis_name not in HEIGHT_AXIS_TO_INDEX:
            raise ValueError(f"Unsupported height axis: {self.height_axis_name}")
        self.height_axis_index = HEIGHT_AXIS_TO_INDEX[self.height_axis_name]
        self.last_debug: GraspPlannerDebug | None = None
        self._previous_candidate_index: int = -1
        self._previous_candidate_point: np.ndarray | None = None
        self._previous_grasp_point: np.ndarray | None = None
        self._hold_counter: int = 0
        self._dropout_hold_counter: int = 0

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
            return self._hold_or_invalid_state(
                merged_object,
                selected_hand,
                object_centroid=object_centroid,
                used_filtered_object_centroid=used_filtered_object_centroid,
                used_filtered_hand_center=used_filtered_hand_center,
                reason="no_finite_candidates",
                raw_candidate_count=0,
                search_radius_candidate_count=0,
                clearance_candidate_count=0,
            )

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
            return self._hold_or_invalid_state(
                merged_object,
                selected_hand,
                object_centroid=object_centroid,
                used_filtered_object_centroid=used_filtered_object_centroid,
                used_filtered_hand_center=used_filtered_hand_center,
                reason="no_candidate_after_clearance_filter",
                raw_candidate_count=raw_candidate_count,
                search_radius_candidate_count=search_radius_candidate_count,
                clearance_candidate_count=0,
            )

        valid_points = candidate_points[clearance_mask]
        valid_distances = candidate_distances[clearance_mask]
        valid_clearances = hand_axis_values[clearance_mask]
        valid_indices = np.nonzero(clearance_mask)[0]
        scores = -valid_distances.astype(np.float32)
        if self.temporal_selection_enabled and self._previous_candidate_point is not None:
            previous_distances = np.linalg.norm(valid_points - self._previous_candidate_point.reshape(1, 3), axis=1)
            scores = scores - (self.score_previous_distance_weight * previous_distances.astype(np.float32))

        best_local_index = int(np.argmax(scores))
        selected_local_index = best_local_index
        used_temporal_hold = False
        if self.temporal_selection_enabled and 0 <= self._previous_candidate_index < len(candidate_points):
            previous_local_candidates = np.where(valid_indices == self._previous_candidate_index)[0]
            if len(previous_local_candidates) > 0:
                previous_local_index = int(previous_local_candidates[0])
                previous_score = float(scores[previous_local_index])
                best_score = float(scores[best_local_index])
                if best_local_index == previous_local_index:
                    self._hold_counter = 0
                elif best_score <= (previous_score + self.switch_margin) and self._hold_counter < self.hold_frames:
                    selected_local_index = previous_local_index
                    used_temporal_hold = True
                    self._hold_counter += 1
                else:
                    self._hold_counter = 0
            else:
                self._hold_counter = 0
        else:
            self._hold_counter = 0

        selected_global_index = int(valid_indices[selected_local_index])
        selected_candidate_point = valid_points[selected_local_index]
        selected_distance = float(valid_distances[selected_local_index])
        selected_clearance = float(valid_clearances[selected_local_index])
        grasp_point = np.asarray(
            [object_centroid[0], object_centroid[1], selected_candidate_point[2]],
            dtype=np.float32,
        )

        state = GraspTargetState(
            target_position_base=tuple(float(v) for v in grasp_point),
            fixed_orientation_base=self.fixed_orientation_base,
            hand_height_clearance_m=selected_clearance,
            distance_to_centroid_m=selected_distance,
            source_point_count=raw_candidate_count,
            selected_candidate_index=selected_global_index,
            used_temporal_hold=used_temporal_hold,
            xy_locked_to_centroid=True,
            dropout_hold_active=False,
            timestamp=max(float(merged_object.timestamp), float(selected_hand.timestamp)),
            valid=True,
        )
        self._previous_candidate_index = selected_global_index
        self._previous_candidate_point = np.asarray(selected_candidate_point, dtype=np.float32).reshape(3)
        self._previous_grasp_point = grasp_point.reshape(3)
        self._dropout_hold_counter = 0
        self.last_debug = GraspPlannerDebug(
            raw_candidate_count=raw_candidate_count,
            search_radius_candidate_count=search_radius_candidate_count,
            clearance_candidate_count=clearance_candidate_count,
            used_filtered_object_centroid=used_filtered_object_centroid,
            used_filtered_hand_center=used_filtered_hand_center,
            chosen_distance_to_centroid_m=selected_distance,
            chosen_hand_height_clearance_m=selected_clearance,
            selection_reason=selection_reason,
            used_temporal_hold=used_temporal_hold,
            selected_candidate_index=selected_global_index,
            target_xy_locked=True,
            dropout_hold_active=False,
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
            selected_candidate_index=-1,
            used_temporal_hold=False,
            xy_locked_to_centroid=False,
            dropout_hold_active=False,
            timestamp=max(float(merged_object.timestamp), float(selected_hand.timestamp)),
            valid=False,
        )

    def _hold_or_invalid_state(
        self,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        *,
        object_centroid: np.ndarray | None,
        used_filtered_object_centroid: bool,
        used_filtered_hand_center: bool,
        reason: str,
        raw_candidate_count: int,
        search_radius_candidate_count: int,
        clearance_candidate_count: int,
    ) -> GraspTargetState:
        if (
            self.temporal_selection_enabled
            and self._previous_grasp_point is not None
            and self._dropout_hold_counter < self.dropout_hold_frames
        ):
            self._dropout_hold_counter += 1
            held_point = self._previous_grasp_point.copy()
            if object_centroid is not None:
                held_point[0] = float(object_centroid[0])
                held_point[1] = float(object_centroid[1])
            self._previous_grasp_point = held_point.copy()
            self.last_debug = GraspPlannerDebug(
                raw_candidate_count=raw_candidate_count,
                search_radius_candidate_count=search_radius_candidate_count,
                clearance_candidate_count=clearance_candidate_count,
                used_filtered_object_centroid=used_filtered_object_centroid,
                used_filtered_hand_center=used_filtered_hand_center,
                chosen_distance_to_centroid_m=None,
                chosen_hand_height_clearance_m=None,
                selection_reason=f"hold_{reason}",
                used_temporal_hold=True,
                selected_candidate_index=self._previous_candidate_index,
                target_xy_locked=object_centroid is not None,
                dropout_hold_active=True,
            )
            return GraspTargetState(
                target_position_base=tuple(float(v) for v in held_point),
                fixed_orientation_base=self.fixed_orientation_base,
                hand_height_clearance_m=None,
                distance_to_centroid_m=None,
                source_point_count=int(merged_object.merged_point_count),
                selected_candidate_index=int(self._previous_candidate_index),
                used_temporal_hold=True,
                xy_locked_to_centroid=bool(object_centroid is not None),
                dropout_hold_active=True,
                timestamp=max(float(merged_object.timestamp), float(selected_hand.timestamp)),
                valid=True,
            )

        self._hold_counter = 0
        self._dropout_hold_counter = 0
        self._previous_candidate_index = -1
        self._previous_candidate_point = None
        self._previous_grasp_point = None
        return self._invalid_state(
            merged_object,
            selected_hand,
            reason=reason,
            raw_candidate_count=raw_candidate_count,
            search_radius_candidate_count=search_radius_candidate_count,
            clearance_candidate_count=clearance_candidate_count,
            used_filtered_object_centroid=used_filtered_object_centroid,
            used_filtered_hand_center=used_filtered_hand_center,
        )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HEIGHT_AXIS_TO_INDEX",
    "GraspPlannerDebug",
    "GraspTargetPlanner",
]
