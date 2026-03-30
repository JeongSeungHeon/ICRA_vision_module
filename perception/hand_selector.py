"""Handedness-based hand source selection with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from system.shared_state import (
    HANDEDNESS_LEFT,
    HANDEDNESS_RIGHT,
    HandState,
    SelectedHandState,
)

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class HandSelectorDebug:
    candidate_cameras: tuple[int, ...]
    candidate_handedness: tuple[str, ...]
    chosen_camera: int | None
    current_camera: int | None
    pending_camera: int | None
    pending_frames: int
    hysteresis_frames: int
    selection_reason: str


class HandSelector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        selection_cfg = config.get("perception", {}).get("hand", {}).get("handedness_selection", {})
        self.right_hand_camera = int(selection_cfg.get("right_hand_camera", 0))
        self.left_hand_camera = int(selection_cfg.get("left_hand_camera", 1))
        self.hysteresis_frames = max(int(selection_cfg.get("hysteresis_frames", 5)), 1)
        self.dropout_hold_frames = max(int(selection_cfg.get("dropout_hold_frames", 12)), 0)
        self.confidence_drop_margin = float(selection_cfg.get("confidence_drop_margin", 0.15))

        self._current_camera: int | None = None
        self._pending_camera: int | None = None
        self._pending_frames = 0
        self._current_state: HandState | None = None
        self._missing_current_frames = 0
        self.last_debug: HandSelectorDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "HandSelector":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._current_camera = None
        self._pending_camera = None
        self._pending_frames = 0
        self._current_state = None
        self._missing_current_frames = 0

    def process_states(self, hand_cam0: HandState, hand_cam1: HandState) -> SelectedHandState:
        states_by_camera = {
            int(hand_cam0.camera_id): hand_cam0,
            int(hand_cam1.camera_id): hand_cam1,
        }
        candidates = self._collect_candidates(states_by_camera)
        selection_reason = "no_valid_candidate"

        if not candidates:
            if self._can_hold_current_selection():
                self._missing_current_frames += 1
                selection_reason = "hold_last_selected_dropout"
                selected = self._build_selected_state(self._current_state)
            else:
                self.reset()
                selected = self._build_invalid_state(hand_cam0, hand_cam1)
            self.last_debug = HandSelectorDebug(
                candidate_cameras=tuple(),
                candidate_handedness=tuple(),
                chosen_camera=selected.selected_camera,
                current_camera=self._current_camera,
                pending_camera=self._pending_camera,
                pending_frames=self._pending_frames,
                hysteresis_frames=self.hysteresis_frames,
                selection_reason=selection_reason,
            )
            return selected

        best_camera = self._choose_best_candidate(candidates)
        if self._current_camera is None:
            self._current_camera = best_camera
            self._pending_camera = None
            self._pending_frames = 0
            self._current_state = candidates[self._current_camera]
            self._missing_current_frames = 0
            selection_reason = "select_best_available"
        elif self._current_camera not in candidates:
            if self._can_hold_current_selection():
                self._missing_current_frames += 1
                selection_reason = "hold_current_during_dropout"
                selected = self._build_selected_state(self._current_state)
                self.last_debug = HandSelectorDebug(
                    candidate_cameras=tuple(sorted(candidates.keys())),
                    candidate_handedness=tuple(str(candidates[key].handedness) for key in sorted(candidates.keys())),
                    chosen_camera=selected.selected_camera,
                    current_camera=self._current_camera,
                    pending_camera=self._pending_camera,
                    pending_frames=self._pending_frames,
                    hysteresis_frames=self.hysteresis_frames,
                    selection_reason=selection_reason,
                )
                return selected
            self._current_camera = best_camera
            self._pending_camera = None
            self._pending_frames = 0
            self._current_state = candidates[self._current_camera]
            self._missing_current_frames = 0
            selection_reason = "switch_after_dropout_timeout"
        elif best_camera == self._current_camera:
            self._pending_camera = None
            self._pending_frames = 0
            self._current_state = candidates[self._current_camera]
            self._missing_current_frames = 0
            selection_reason = "keep_current_best"
        else:
            current_conf = float(candidates[self._current_camera].confidence)
            challenger_conf = float(candidates[best_camera].confidence)
            if challenger_conf <= current_conf + self.confidence_drop_margin:
                selection_reason = "keep_current_with_margin"
                self._pending_camera = None
                self._pending_frames = 0
                self._current_state = candidates[self._current_camera]
                self._missing_current_frames = 0
            else:
                if self._pending_camera == best_camera:
                    self._pending_frames += 1
                else:
                    self._pending_camera = best_camera
                    self._pending_frames = 1
                selection_reason = "pending_switch"
                if self._pending_frames >= self.hysteresis_frames:
                    self._current_camera = best_camera
                    self._pending_camera = None
                    self._pending_frames = 0
                    self._current_state = candidates[self._current_camera]
                    self._missing_current_frames = 0
                    selection_reason = "switch_after_hysteresis"
                else:
                    self._current_state = candidates[self._current_camera]
                    self._missing_current_frames = 0

        chosen_state = self._current_state if self._current_state is not None else candidates[self._current_camera]
        selected = self._build_selected_state(chosen_state)
        self.last_debug = HandSelectorDebug(
            candidate_cameras=tuple(sorted(candidates.keys())),
            candidate_handedness=tuple(str(candidates[key].handedness) for key in sorted(candidates.keys())),
            chosen_camera=int(chosen_state.camera_id),
            current_camera=self._current_camera,
            pending_camera=self._pending_camera,
            pending_frames=self._pending_frames,
            hysteresis_frames=self.hysteresis_frames,
            selection_reason=selection_reason,
        )
        return selected

    def _can_hold_current_selection(self) -> bool:
        return bool(
            self._current_state is not None
            and self._current_camera is not None
            and self._missing_current_frames < self.dropout_hold_frames
        )

    def _collect_candidates(self, states_by_camera: dict[int, HandState]) -> dict[int, HandState]:
        candidates: dict[int, HandState] = {}
        right_state = states_by_camera.get(self.right_hand_camera)
        left_state = states_by_camera.get(self.left_hand_camera)

        if self._is_matching_candidate(right_state, expected_handedness=HANDEDNESS_RIGHT):
            candidates[int(right_state.camera_id)] = right_state
        if self._is_matching_candidate(left_state, expected_handedness=HANDEDNESS_LEFT):
            candidates[int(left_state.camera_id)] = left_state
        return candidates

    def _choose_best_candidate(self, candidates: dict[int, HandState]) -> int:
        if self._current_camera in candidates:
            return int(self._current_camera)
        return int(max(candidates.items(), key=lambda item: float(item[1].confidence))[0])

    @staticmethod
    def _is_matching_candidate(state: HandState | None, *, expected_handedness: str) -> bool:
        if state is None:
            return False
        return bool(
            state.valid
            and state.hand_detected
            and state.handedness == expected_handedness
            and state.palm_center_base is not None
            and state.palm_normal_base is not None
        )

    @staticmethod
    def _build_invalid_state(hand_cam0: HandState, hand_cam1: HandState) -> SelectedHandState:
        return SelectedHandState(
            frame_id=max(int(hand_cam0.frame_id), int(hand_cam1.frame_id)),
            selected_camera=None,
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
    def _build_selected_state(chosen_state: HandState) -> SelectedHandState:
        return SelectedHandState(
            frame_id=int(chosen_state.frame_id),
            selected_camera=int(chosen_state.camera_id),
            handedness=str(chosen_state.handedness),
            confidence=float(chosen_state.confidence),
            palm_center_base=chosen_state.palm_center_base,
            palm_normal_base=chosen_state.palm_normal_base,
            wrist_base=chosen_state.wrist_base,
            hand_velocity_base=chosen_state.hand_velocity_base,
            timestamp=float(chosen_state.timestamp),
            valid=True,
        )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HandSelectorDebug",
    "HandSelector",
]
