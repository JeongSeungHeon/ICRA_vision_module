"""Grasp target computation from merged object geometry and selected hand state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from system.shared_state import FusionState, GraspTargetState, MergedObjectState, SelectedHandState

# Grasp target planner가 기본으로 읽는 handover/grasp 설정 파일.
DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")

# 설정 파일의 height_axis 이름을 numpy point 배열의 축 인덱스로 변환한다.
HEIGHT_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class GraspPlannerDebug:
    """최근 grasp target 선택 과정에서 필터링/hold가 어떻게 적용됐는지 기록한다."""

    # merged object point cloud에서 NaN/Inf를 제거한 뒤 남은 원본 후보 수.
    raw_candidate_count: int

    # object centroid 주변 탐색 반경 필터를 통과한 후보 수.
    search_radius_candidate_count: int

    # 손과의 높이축 간격 필터를 통과해 실제 선택 가능한 후보 수.
    clearance_candidate_count: int

    # centroid/hand center가 fusion filter 결과에서 왔는지 여부.
    used_filtered_object_centroid: bool
    used_filtered_hand_center: bool

    # 최종 선택 후보의 centroid 거리와 손 높이축 clearance. 실패/hold 시 None일 수 있다.
    chosen_distance_to_centroid_m: float | None
    chosen_hand_height_clearance_m: float | None

    # 선택/실패/hold가 발생한 이유를 문자열로 남겨 downstream 로그에서 확인한다.
    selection_reason: str

    # 후보가 급격히 바뀌지 않도록 이전 후보를 유지했는지 여부.
    used_temporal_hold: bool = False

    # 최종 선택된 후보의 candidate_points 기준 인덱스. 선택 실패 시 -1이다.
    selected_candidate_index: int = -1

    # target의 x/y가 object centroid에 고정됐는지 여부.
    target_xy_locked: bool = False

    # 후보가 순간적으로 사라졌을 때 이전 grasp point를 임시 유지했는지 여부.
    dropout_hold_active: bool = False

    # 최종 후보가 나온 point cloud source. template/raw_object/hold_*/none 중 하나다.
    candidate_source: str = "none"


@dataclass
class _CandidateSelection:
    """Candidate selection result for one point-cloud source."""

    valid: bool
    source: str
    reason: str
    raw_candidate_count: int
    search_radius_candidate_count: int
    clearance_candidate_count: int
    selected_candidate_index: int = -1
    selected_candidate_point: np.ndarray | None = None
    selected_distance_to_centroid_m: float | None = None
    selected_hand_height_clearance_m: float | None = None
    used_temporal_hold: bool = False
    next_hold_counter: int = 0


class GraspTargetPlanner:
    """Object point cloud와 선택된 손 위치로부터 robot base 기준 grasp target을 계산한다."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        grasp_cfg = config.get("grasp", {})
        frame_cfg = config.get("frames", {}).get("height_axis", {})

        # Robot controller에 넘길 고정 orientation. 현재 모듈은 위치만 계산하고 자세는 설정값을 사용한다.
        self.fixed_orientation_format = str(grasp_cfg.get("fixed_orientation_format", "rotvec"))
        self.fixed_orientation_base = tuple(float(v) for v in grasp_cfg.get("fixed_orientation_base", [0.0, 0.0, 0.0]))

        # Object centroid 근처의 표면점만 grasp 후보로 우선 고려하는 반경.
        self.candidate_search_radius_from_centroid_m = float(
            grasp_cfg.get("candidate_search_radius_from_centroid_m", 0.05)
        )

        # 선택된 손과 너무 가까운 높이축 후보를 제거해 hand/object 충돌 가능성을 낮춘다.
        self.min_hand_height_clearance_m = float(grasp_cfg.get("min_hand_height_clearance_m", 0.03))

        # 후보가 없을 때의 정책값. 현재 구현에서는 hold 가능 시 유지하고, 아니면 invalid로 전환한다.
        self.no_candidate_action = str(grasp_cfg.get("no_candidate_action", "hold"))

        # Template 후보가 없을 때 raw object cloud의 중간 z band에서 후보를 재탐색한다.
        object_fallback_cfg = grasp_cfg.get("object_cloud_fallback", {})
        self.object_cloud_fallback_enabled = bool(object_fallback_cfg.get("enabled", True))
        self.object_cloud_fallback_z_crop_margin_m = max(
            float(object_fallback_cfg.get("z_crop_margin_m", 0.01)),
            0.0,
        )

        # Temporal selection은 프레임 간 target 튐을 줄이기 위한 hysteresis/hold 설정이다.
        selection_cfg = grasp_cfg.get("target_selection", {})
        self.temporal_selection_enabled = bool(selection_cfg.get("enabled", True))
        self.switch_margin = float(selection_cfg.get("switch_margin", 0.01))
        self.hold_frames = max(0, int(selection_cfg.get("hold_frames", 6)))
        self.dropout_hold_frames = max(0, int(selection_cfg.get("dropout_hold_frames", 8)))
        self.score_previous_distance_weight = float(selection_cfg.get("score_previous_distance_weight", 0.9))

        # height_axis는 hand clearance를 계산할 축이다. 보통 base frame의 z축을 사용한다.
        self.height_axis_name = str(frame_cfg.get("name", "z")).strip().lower()
        if self.height_axis_name not in HEIGHT_AXIS_TO_INDEX:
            raise ValueError(f"Unsupported height axis: {self.height_axis_name}")
        self.height_axis_index = HEIGHT_AXIS_TO_INDEX[self.height_axis_name]

        # 아래 상태값들은 이전 프레임과 현재 프레임의 선택을 비교하기 위한 내부 메모리다.
        self.last_debug: GraspPlannerDebug | None = None
        self._previous_candidate_index: int = -1
        self._previous_candidate_point: np.ndarray | None = None
        self._previous_candidate_source: str | None = None
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
        *,
        fallback_object: MergedObjectState | None = None,
    ) -> GraspTargetState:
        """한 프레임의 object/hand 상태를 받아 최종 grasp target 상태로 변환한다."""

        # Object나 hand가 유효하지 않으면 target을 만들 수 없으므로 invalid 상태를 반환한다.
        if not merged_object.valid:
            return self._invalid_state(merged_object, selected_hand, reason="no_merged_object")
        if not selected_hand.valid or selected_hand.palm_center_base is None:
            return self._invalid_state(merged_object, selected_hand, reason="no_selected_hand")

        # FusionState가 신선하면 필터링된 centroid/hand center를 우선 사용하고, 없으면 raw state로 fallback한다.
        object_centroid, used_filtered_object_centroid = self._resolve_object_centroid(merged_object, fusion_state)
        hand_center, used_filtered_hand_center = self._resolve_hand_center(selected_hand, fusion_state)
        if object_centroid is None:
            return self._invalid_state(merged_object, selected_hand, reason="no_object_centroid")
        if hand_center is None:
            return self._invalid_state(merged_object, selected_hand, reason="no_hand_center")

        template_points = np.asarray(merged_object.merged_points_base, dtype=np.float32).reshape((-1, 3))
        primary_selection = self._select_candidate_from_points(
            template_points,
            object_centroid=object_centroid,
            hand_center=hand_center,
            source="template",
            prefiltered=False,
        )
        if primary_selection.valid:
            return self._valid_state_from_selection(
                primary_selection,
                merged_object,
                selected_hand,
                object_centroid=object_centroid,
                used_filtered_object_centroid=used_filtered_object_centroid,
                used_filtered_hand_center=used_filtered_hand_center,
            )

        fallback_selection = self._select_raw_object_fallback_candidate(
            fallback_object,
            object_centroid=object_centroid,
            hand_center=hand_center,
        )
        if fallback_selection is not None and fallback_selection.valid:
            return self._valid_state_from_selection(
                fallback_selection,
                merged_object,
                selected_hand,
                object_centroid=object_centroid,
                used_filtered_object_centroid=used_filtered_object_centroid,
                used_filtered_hand_center=used_filtered_hand_center,
            )

        failure_selection = fallback_selection if fallback_selection is not None else primary_selection
        return self._hold_or_invalid_state(
            merged_object,
            selected_hand,
            object_centroid=object_centroid,
            used_filtered_object_centroid=used_filtered_object_centroid,
            used_filtered_hand_center=used_filtered_hand_center,
            reason=failure_selection.reason,
            raw_candidate_count=failure_selection.raw_candidate_count,
            search_radius_candidate_count=failure_selection.search_radius_candidate_count,
            clearance_candidate_count=failure_selection.clearance_candidate_count,
        )

    def _select_candidate_from_points(
        self,
        points: np.ndarray,
        *,
        object_centroid: np.ndarray,
        hand_center: np.ndarray,
        source: str,
        prefiltered: bool,
    ) -> _CandidateSelection:
        """한 point-cloud source에서 기존 기준을 적용해 grasp z 후보를 고른다."""

        points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        if not prefiltered:
            finite_mask = np.isfinite(points).all(axis=1)
            points = points[finite_mask]
        raw_candidate_count = int(len(points))
        if raw_candidate_count == 0:
            return _CandidateSelection(
                valid=False,
                source=source,
                reason=f"{source}_no_finite_candidates",
                raw_candidate_count=0,
                search_radius_candidate_count=0,
                clearance_candidate_count=0,
                next_hold_counter=0,
            )

        centroid_distances = np.linalg.norm(points - object_centroid.reshape(1, 3), axis=1)
        within_radius_mask = centroid_distances <= self.candidate_search_radius_from_centroid_m
        if np.any(within_radius_mask):
            candidate_points = points[within_radius_mask]
            candidate_distances = centroid_distances[within_radius_mask]
            selection_reason = f"{source}_selected_within_search_radius"
        else:
            candidate_points = points
            candidate_distances = centroid_distances
            selection_reason = f"{source}_fallback_to_all_candidates"
        search_radius_candidate_count = int(len(candidate_points))

        hand_axis_values = np.abs(candidate_points[:, self.height_axis_index] - hand_center[self.height_axis_index])
        clearance_mask = hand_axis_values >= self.min_hand_height_clearance_m
        clearance_candidate_count = int(np.count_nonzero(clearance_mask))
        if clearance_candidate_count == 0:
            return _CandidateSelection(
                valid=False,
                source=source,
                reason=f"{source}_no_candidate_after_clearance_filter",
                raw_candidate_count=raw_candidate_count,
                search_radius_candidate_count=search_radius_candidate_count,
                clearance_candidate_count=0,
                next_hold_counter=0,
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
        next_hold_counter = 0
        if (
            self.temporal_selection_enabled
            and self._previous_candidate_source == source
            and 0 <= self._previous_candidate_index < len(candidate_points)
        ):
            previous_local_candidates = np.where(valid_indices == self._previous_candidate_index)[0]
            if len(previous_local_candidates) > 0:
                previous_local_index = int(previous_local_candidates[0])
                previous_score = float(scores[previous_local_index])
                best_score = float(scores[best_local_index])
                if best_local_index == previous_local_index:
                    next_hold_counter = 0
                elif best_score <= (previous_score + self.switch_margin) and self._hold_counter < self.hold_frames:
                    selected_local_index = previous_local_index
                    used_temporal_hold = True
                    next_hold_counter = self._hold_counter + 1
                else:
                    next_hold_counter = 0

        selected_global_index = int(valid_indices[selected_local_index])
        return _CandidateSelection(
            valid=True,
            source=source,
            reason=selection_reason,
            raw_candidate_count=raw_candidate_count,
            search_radius_candidate_count=search_radius_candidate_count,
            clearance_candidate_count=clearance_candidate_count,
            selected_candidate_index=selected_global_index,
            selected_candidate_point=np.asarray(valid_points[selected_local_index], dtype=np.float32).reshape(3),
            selected_distance_to_centroid_m=float(valid_distances[selected_local_index]),
            selected_hand_height_clearance_m=float(valid_clearances[selected_local_index]),
            used_temporal_hold=used_temporal_hold,
            next_hold_counter=next_hold_counter,
        )

    def _select_raw_object_fallback_candidate(
        self,
        fallback_object: MergedObjectState | None,
        *,
        object_centroid: np.ndarray,
        hand_center: np.ndarray,
    ) -> _CandidateSelection | None:
        """Template 후보 실패 시 raw object cloud의 중간 z band에서 후보를 찾는다."""

        if not self.object_cloud_fallback_enabled:
            return None
        if fallback_object is None or not fallback_object.valid:
            return _CandidateSelection(
                valid=False,
                source="raw_object",
                reason="raw_object_unavailable",
                raw_candidate_count=0,
                search_radius_candidate_count=0,
                clearance_candidate_count=0,
            )

        points = np.asarray(fallback_object.merged_points_base, dtype=np.float32).reshape((-1, 3))
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        finite_count = int(len(points))
        if finite_count == 0:
            return _CandidateSelection(
                valid=False,
                source="raw_object",
                reason="raw_object_no_finite_candidates",
                raw_candidate_count=0,
                search_radius_candidate_count=0,
                clearance_candidate_count=0,
            )

        z_values = points[:, 2]
        z_min = float(np.min(z_values))
        z_max = float(np.max(z_values))
        margin = float(self.object_cloud_fallback_z_crop_margin_m)
        z_mask = np.logical_and(z_values >= z_min + margin, z_values <= z_max - margin)
        cropped_points = points[z_mask]
        if len(cropped_points) == 0:
            return _CandidateSelection(
                valid=False,
                source="raw_object",
                reason="raw_object_empty_after_z_crop",
                raw_candidate_count=finite_count,
                search_radius_candidate_count=0,
                clearance_candidate_count=0,
            )

        return self._select_candidate_from_points(
            cropped_points,
            object_centroid=object_centroid,
            hand_center=hand_center,
            source="raw_object",
            prefiltered=True,
        )

    def _valid_state_from_selection(
        self,
        selection: _CandidateSelection,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        *,
        object_centroid: np.ndarray,
        used_filtered_object_centroid: bool,
        used_filtered_hand_center: bool,
    ) -> GraspTargetState:
        """선택된 후보를 GraspTargetState와 debug 상태로 변환한다."""

        selected_candidate_point = np.asarray(selection.selected_candidate_point, dtype=np.float32).reshape(3)
        selected_distance = float(selection.selected_distance_to_centroid_m)
        selected_clearance = float(selection.selected_hand_height_clearance_m)

        # 최종 target은 x/y를 object centroid에 고정하고, z는 선택된 표면 후보의 높이를 사용한다.
        grasp_point = np.asarray(
            [object_centroid[0], object_centroid[1], selected_candidate_point[2]],
            dtype=np.float32,
        )

        state = GraspTargetState(
            target_position_base=tuple(float(v) for v in grasp_point),
            fixed_orientation_base=self.fixed_orientation_base,
            hand_height_clearance_m=selected_clearance,
            distance_to_centroid_m=selected_distance,
            source_point_count=selection.raw_candidate_count,
            selected_candidate_index=selection.selected_candidate_index,
            used_temporal_hold=selection.used_temporal_hold,
            xy_locked_to_centroid=True,
            dropout_hold_active=False,
            timestamp=max(float(merged_object.timestamp), float(selected_hand.timestamp)),
            valid=True,
        )

        # 다음 프레임에서 temporal hold/dropout hold 판단에 사용할 선택 이력을 갱신한다.
        self._previous_candidate_index = selection.selected_candidate_index
        self._previous_candidate_point = np.asarray(selected_candidate_point, dtype=np.float32).reshape(3)
        self._previous_candidate_source = selection.source
        self._previous_grasp_point = grasp_point.reshape(3)
        self._hold_counter = int(selection.next_hold_counter)
        self._dropout_hold_counter = 0
        self.last_debug = GraspPlannerDebug(
            raw_candidate_count=selection.raw_candidate_count,
            search_radius_candidate_count=selection.search_radius_candidate_count,
            clearance_candidate_count=selection.clearance_candidate_count,
            used_filtered_object_centroid=used_filtered_object_centroid,
            used_filtered_hand_center=used_filtered_hand_center,
            chosen_distance_to_centroid_m=selected_distance,
            chosen_hand_height_clearance_m=selected_clearance,
            selection_reason=selection.reason,
            used_temporal_hold=selection.used_temporal_hold,
            selected_candidate_index=selection.selected_candidate_index,
            target_xy_locked=True,
            dropout_hold_active=False,
            candidate_source=selection.source,
        )
        return state

    def _resolve_object_centroid(
        self,
        merged_object: MergedObjectState,
        fusion_state: FusionState | None,
    ) -> tuple[np.ndarray | None, bool]:
        """필터링된 object centroid를 우선 사용하고, 없으면 merged object centroid로 대체한다."""

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
        """필터링된 hand center를 우선 사용하고, 없으면 selected hand palm center로 대체한다."""

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
        """Target 산출이 불가능할 때 invalid GraspTargetState와 debug reason을 만든다."""

        self.last_debug = GraspPlannerDebug(
            raw_candidate_count=raw_candidate_count,
            search_radius_candidate_count=search_radius_candidate_count,
            clearance_candidate_count=clearance_candidate_count,
            used_filtered_object_centroid=used_filtered_object_centroid,
            used_filtered_hand_center=used_filtered_hand_center,
            chosen_distance_to_centroid_m=None,
            chosen_hand_height_clearance_m=None,
            selection_reason=reason,
            candidate_source="none",
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
        """후보가 사라진 경우 이전 target을 잠깐 유지하거나 invalid 상태로 전환한다."""

        if (
            self.temporal_selection_enabled
            and self._previous_grasp_point is not None
            and self._dropout_hold_counter < self.dropout_hold_frames
        ):
            self._dropout_hold_counter += 1
            held_point = self._previous_grasp_point.copy()
            if object_centroid is not None:
                # Object centroid는 계속 갱신해 x/y tracking은 유지하고, grasp height만 이전 값을 쓴다.
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
                candidate_source=f"hold_{self._previous_candidate_source or 'unknown'}",
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

        # 허용된 dropout hold 기간도 지나면 이전 선택 이력을 비우고 invalid 상태를 반환한다.
        self._hold_counter = 0
        self._dropout_hold_counter = 0
        self._previous_candidate_index = -1
        self._previous_candidate_point = None
        self._previous_candidate_source = None
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
