"""Object-distance hand selection with conservative handover locking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from system.shared_state import HandCandidateState, HandState, ObjectState, SelectedHandState, Vec3

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class HandSelectorDebug:
    candidate_cameras: tuple[int, ...]
    candidate_handedness: tuple[str, ...]
    cam0_reject_reason: str
    cam1_reject_reason: str
    chosen_camera: int | None
    current_camera: int | None
    pending_camera: int | None
    pending_frames: int
    hysteresis_frames: int
    locked_camera: int | None
    lock_stable_frames: int
    lock_after_stable_frames: int
    selection_reason: str
    chosen_candidate_id: str | None = None
    current_candidate_id: str | None = None
    pending_candidate_id: str | None = None
    object_center_base: Vec3 | None = None
    chosen_distance_m: float | None = None
    current_distance_m: float | None = None


class HandSelector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        legacy_cfg = self.config.get("perception", {}).get("hand", {}).get("handedness_selection", {})
        selection_cfg = self.config.get("perception", {}).get("hand_selection", {})

        self.active_hand_distance_threshold_m = float(
            selection_cfg.get("active_hand_distance_threshold_m", 0.15)
        )
        self.switch_margin_m = float(selection_cfg.get("switch_margin_m", 0.05))
        self.switch_confirm_frames = max(
            int(selection_cfg.get("switch_confirm_frames", legacy_cfg.get("hysteresis_frames", 5))),
            1,
        )
        self.lost_timeout_s = max(float(selection_cfg.get("lost_timeout_s", 0.5)), 0.0)
        self.lock_active_hand_during_task = bool(selection_cfg.get("lock_active_hand_during_task", True))

        self._current_candidate_id: str | None = None
        self._current_candidate: HandCandidateState | None = None
        self._pending_candidate_id: str | None = None
        self._pending_frames = 0
        self._locked_candidate_id: str | None = None
        self._lock_stable_frames = 0
        self.last_debug: HandSelectorDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "HandSelector":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._current_candidate_id = None
        self._current_candidate = None
        self._pending_candidate_id = None
        self._pending_frames = 0
        self._locked_candidate_id = None
        self._lock_stable_frames = 0
        self.last_debug = None

    def process_states(
        self,
        hand_cam0: HandState,
        hand_cam1: HandState,
        *,
        object_center_base: Vec3 | None = None,
    ) -> SelectedHandState:
        candidates, reject_reasons = self._collect_candidates(hand_cam0, hand_cam1)
        distances = self._compute_distances(candidates, object_center_base)
        now_timestamp = self._resolve_now_timestamp(hand_cam0, hand_cam1, candidates)
        selection_reason = "no_valid_candidate"

        if object_center_base is None:
            selected, selection_reason = self._hold_or_invalidate(
                hand_cam0,
                hand_cam1,
                candidates,
                now_timestamp,
                "no_object_center",
            )
            self._set_debug(candidates, reject_reasons, selected, object_center_base, distances, selection_reason)
            return selected

        eligible_ids = [
            candidate_id
            for candidate_id, distance in distances.items()
            if distance <= self.active_hand_distance_threshold_m
        ]
        best_candidate_id = min(eligible_ids, key=lambda candidate_id: distances[candidate_id], default=None)

        if self._current_candidate_id is None:
            if best_candidate_id is None:
                selected = self._build_invalid_state(hand_cam0, hand_cam1)
                selection_reason = "no_candidate_within_threshold"
            else:
                self._select_candidate(candidates[best_candidate_id])
                selected = self._build_selected_state(self._current_candidate)
                selection_reason = "select_within_object_threshold"
            self._set_debug(candidates, reject_reasons, selected, object_center_base, distances, selection_reason)
            return selected

        current_candidate = candidates.get(self._current_candidate_id)
        if current_candidate is None:
            selected, selection_reason = self._handle_current_lost(
                hand_cam0,
                hand_cam1,
                candidates,
                now_timestamp,
                best_candidate_id,
            )
            self._set_debug(candidates, reject_reasons, selected, object_center_base, distances, selection_reason)
            return selected

        self._current_candidate = current_candidate
        current_distance = distances.get(self._current_candidate_id)
        challenger_id = self._find_switch_challenger(best_candidate_id, current_distance, distances)
        if challenger_id is not None:
            if self._pending_candidate_id == challenger_id:
                self._pending_frames += 1
            else:
                self._pending_candidate_id = challenger_id
                self._pending_frames = 1
            selection_reason = "pending_switch_closer_to_object"
            if self._pending_frames >= self.switch_confirm_frames:
                self._select_candidate(candidates[challenger_id])
                selection_reason = "switch_after_confirmed_closer_to_object"
        else:
            self._pending_candidate_id = None
            self._pending_frames = 0
            selection_reason = "keep_locked_active_hand"

        selected = self._build_selected_state(self._current_candidate)
        self._set_debug(candidates, reject_reasons, selected, object_center_base, distances, selection_reason)
        return selected

    def _handle_current_lost(
        self,
        hand_cam0: HandState,
        hand_cam1: HandState,
        candidates: dict[str, HandCandidateState],
        now_timestamp: float,
        best_candidate_id: str | None,
    ) -> tuple[SelectedHandState, str]:
        if self._can_hold_current(now_timestamp):
            self._pending_candidate_id = None
            self._pending_frames = 0
            return self._build_selected_state(self._current_candidate), "hold_active_hand_lost_timeout"

        self._clear_current()
        if best_candidate_id is not None and best_candidate_id in candidates:
            self._select_candidate(candidates[best_candidate_id])
            return self._build_selected_state(self._current_candidate), "select_after_lost_timeout"
        return self._build_invalid_state(hand_cam0, hand_cam1), "active_hand_lost_timeout_no_replacement"

    def _hold_or_invalidate(
        self,
        hand_cam0: HandState,
        hand_cam1: HandState,
        candidates: dict[str, HandCandidateState],
        now_timestamp: float,
        reason_prefix: str,
    ) -> tuple[SelectedHandState, str]:
        if self._current_candidate_id in candidates:
            self._current_candidate = candidates[self._current_candidate_id]
            return self._build_selected_state(self._current_candidate), f"{reason_prefix}_keep_current"
        if self._can_hold_current(now_timestamp):
            return self._build_selected_state(self._current_candidate), f"{reason_prefix}_hold_current"
        self._clear_current()
        return self._build_invalid_state(hand_cam0, hand_cam1), reason_prefix

    def _find_switch_challenger(
        self,
        best_candidate_id: str | None,
        current_distance: float | None,
        distances: dict[str, float],
    ) -> str | None:
        if best_candidate_id is None or best_candidate_id == self._current_candidate_id:
            return None
        if current_distance is None:
            return best_candidate_id
        challenger_distance = distances.get(best_candidate_id)
        if challenger_distance is None:
            return None
        if challenger_distance <= current_distance - self.switch_margin_m:
            return best_candidate_id
        return None

    def _select_candidate(self, candidate: HandCandidateState) -> None:
        self._current_candidate_id = self._candidate_key(candidate)
        self._current_candidate = candidate
        self._pending_candidate_id = None
        self._pending_frames = 0
        if self.lock_active_hand_during_task:
            self._locked_candidate_id = self._current_candidate_id
            self._lock_stable_frames += 1

    def _clear_current(self) -> None:
        self._current_candidate_id = None
        self._current_candidate = None
        self._pending_candidate_id = None
        self._pending_frames = 0
        self._locked_candidate_id = None
        self._lock_stable_frames = 0

    def _can_hold_current(self, now_timestamp: float) -> bool:
        if self._current_candidate is None:
            return False
        age_s = max(float(now_timestamp) - float(self._current_candidate.timestamp), 0.0)
        return age_s <= self.lost_timeout_s

    def _set_debug(
        self,
        candidates: dict[str, HandCandidateState],
        reject_reasons: dict[int, str],
        selected: SelectedHandState,
        object_center_base: Vec3 | None,
        distances: dict[str, float],
        selection_reason: str,
    ) -> None:
        candidate_items = sorted(candidates.items(), key=lambda item: (int(item[1].camera_id), int(item[1].candidate_index)))
        chosen_id = selected.selected_candidate_id
        self.last_debug = HandSelectorDebug(
            candidate_cameras=tuple(int(candidate.camera_id) for _, candidate in candidate_items),
            candidate_handedness=tuple(str(candidate.handedness) for _, candidate in candidate_items),
            cam0_reject_reason=str(reject_reasons.get(0, "")),
            cam1_reject_reason=str(reject_reasons.get(1, "")),
            chosen_camera=selected.selected_camera,
            current_camera=None if self._current_candidate is None else int(self._current_candidate.camera_id),
            pending_camera=None if self._pending_candidate_id is None else int(candidates[self._pending_candidate_id].camera_id) if self._pending_candidate_id in candidates else None,
            pending_frames=self._pending_frames,
            hysteresis_frames=self.switch_confirm_frames,
            locked_camera=None if self._current_candidate is None else int(self._current_candidate.camera_id),
            lock_stable_frames=self._lock_stable_frames,
            lock_after_stable_frames=1,
            selection_reason=selection_reason,
            chosen_candidate_id=chosen_id,
            current_candidate_id=self._current_candidate_id,
            pending_candidate_id=self._pending_candidate_id,
            object_center_base=object_center_base,
            chosen_distance_m=None if chosen_id is None else distances.get(chosen_id),
            current_distance_m=None if self._current_candidate_id is None else distances.get(self._current_candidate_id),
        )

    @staticmethod
    def _collect_candidates(
        hand_cam0: HandState,
        hand_cam1: HandState,
    ) -> tuple[dict[str, HandCandidateState], dict[int, str]]:
        candidates: dict[str, HandCandidateState] = {}
        reject_reasons: dict[int, str] = {}
        for state in (hand_cam0, hand_cam1):
            state_candidates = list(getattr(state, "hand_candidates", []) or [])
            if not state_candidates and HandSelector._is_valid_scalar_hand_state(state):
                state_candidates = [HandSelector._candidate_from_scalar_state(state)]

            valid_for_camera = 0
            for candidate in state_candidates:
                reason = HandSelector._candidate_reject_reason(candidate)
                if reason == "":
                    candidates[HandSelector._candidate_key(candidate)] = candidate
                    valid_for_camera += 1
            if valid_for_camera == 0:
                reject_reasons[int(state.camera_id)] = HandSelector._camera_reject_reason(state, state_candidates)
        return candidates, reject_reasons

    @staticmethod
    def _camera_reject_reason(state: HandState, state_candidates: list[HandCandidateState]) -> str:
        if state_candidates:
            return "no_valid_candidate"
        if not bool(state.hand_detected):
            return "hand_not_detected"
        if state.palm_center_base is None:
            return "missing_palm_center"
        if state.palm_normal_base is None:
            return "missing_palm_normal"
        if not bool(state.valid):
            return "state_invalid"
        return "no_valid_candidate"

    @staticmethod
    def _candidate_reject_reason(candidate: HandCandidateState | None) -> str:
        if candidate is None:
            return "missing_candidate"
        if not bool(candidate.hand_detected):
            return "hand_not_detected"
        if candidate.palm_center_base is None:
            return "missing_palm_center"
        if candidate.palm_normal_base is None:
            return "missing_palm_normal"
        if not bool(candidate.valid):
            return "state_invalid"
        return ""

    @staticmethod
    def _is_valid_scalar_hand_state(state: HandState | None) -> bool:
        if state is None:
            return False
        return bool(
            state.valid
            and state.hand_detected
            and state.palm_center_base is not None
            and state.palm_normal_base is not None
        )

    @staticmethod
    def _candidate_from_scalar_state(state: HandState) -> HandCandidateState:
        return HandCandidateState(
            camera_id=int(state.camera_id),
            frame_id=int(state.frame_id),
            candidate_index=0,
            candidate_id=f"cam{int(state.camera_id)}:legacy",
            hand_detected=bool(state.hand_detected),
            handedness=str(state.handedness),
            confidence=float(state.confidence),
            palm_center_base=state.palm_center_base,
            palm_normal_base=state.palm_normal_base,
            wrist_base=state.wrist_base,
            hand_velocity_base=state.hand_velocity_base,
            timestamp=float(state.timestamp),
            valid=bool(state.valid),
        )

    @staticmethod
    def _candidate_key(candidate: HandCandidateState) -> str:
        if candidate.candidate_id:
            return str(candidate.candidate_id)
        return f"cam{int(candidate.camera_id)}:hand{int(candidate.candidate_index)}"

    @staticmethod
    def _compute_distances(
        candidates: dict[str, HandCandidateState],
        object_center_base: Vec3 | None,
    ) -> dict[str, float]:
        if object_center_base is None:
            return {}
        object_center = np.asarray(object_center_base, dtype=np.float32).reshape(3)
        distances: dict[str, float] = {}
        for candidate_id, candidate in candidates.items():
            if candidate.palm_center_base is None:
                continue
            palm_center = np.asarray(candidate.palm_center_base, dtype=np.float32).reshape(3)
            if np.isfinite(palm_center).all() and np.isfinite(object_center).all():
                distances[candidate_id] = float(np.linalg.norm(palm_center - object_center))
        return distances

    @staticmethod
    def _resolve_now_timestamp(
        hand_cam0: HandState,
        hand_cam1: HandState,
        candidates: dict[str, HandCandidateState],
    ) -> float:
        timestamps = [float(hand_cam0.timestamp), float(hand_cam1.timestamp)]
        timestamps.extend(float(candidate.timestamp) for candidate in candidates.values())
        return max(timestamps)

    @staticmethod
    def _build_invalid_state(hand_cam0: HandState, hand_cam1: HandState) -> SelectedHandState:
        return SelectedHandState(
            frame_id=max(int(hand_cam0.frame_id), int(hand_cam1.frame_id)),
            selected_camera=None,
            selected_candidate_index=None,
            selected_candidate_id=None,
            handedness=None,
            confidence=0.0,
            palm_center_base=None,
            palm_normal_base=None,
            wrist_base=None,
            hand_velocity_base=None,
            timestamp=max(float(hand_cam0.timestamp), float(hand_cam1.timestamp)),
            valid=False,
        )

    @staticmethod
    def _build_selected_state(chosen_candidate: HandCandidateState) -> SelectedHandState:
        return SelectedHandState(
            frame_id=int(chosen_candidate.frame_id),
            selected_camera=int(chosen_candidate.camera_id),
            selected_candidate_index=int(chosen_candidate.candidate_index),
            selected_candidate_id=HandSelector._candidate_key(chosen_candidate),
            handedness=str(chosen_candidate.handedness),
            confidence=float(chosen_candidate.confidence),
            palm_center_base=chosen_candidate.palm_center_base,
            palm_normal_base=chosen_candidate.palm_normal_base,
            wrist_base=chosen_candidate.wrist_base,
            hand_velocity_base=chosen_candidate.hand_velocity_base,
            timestamp=float(chosen_candidate.timestamp),
            valid=True,
        )


def object_center_for_hand_selection(object_cam0: ObjectState, object_cam1: ObjectState) -> Vec3 | None:
    usable: list[tuple[np.ndarray, float]] = []
    for state in (object_cam0, object_cam1):
        centroid = getattr(state, "centroid_base", None)
        if not bool(getattr(state, "valid", False)) or centroid is None:
            continue
        center = np.asarray(centroid, dtype=np.float32).reshape(3)
        if not np.isfinite(center).all():
            continue
        weight = max(float(getattr(state, "point_count", 0) or 0), 1.0)
        usable.append((center, weight))
    if not usable:
        return None
    total_weight = sum(weight for _, weight in usable)
    weighted_center = sum(center * weight for center, weight in usable) / total_weight
    return tuple(float(value) for value in weighted_center)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HandSelectorDebug",
    "HandSelector",
    "object_center_for_hand_selection",
]
