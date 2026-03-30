"""Temporal filtering and event detection for perception outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from system.shared_state import (
    FusionState,
    HEIGHT_AXIS_Y,
    MergedObjectState,
    SelectedHandState,
)

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
HEIGHT_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
HEIGHT_DIRECTION_TO_SIGN = {"up": 1.0, "positive": 1.0, "+": 1.0, "down": -1.0, "negative": -1.0, "-": -1.0}


@dataclass
class FusionDebug:
    raw_object_centroid_base: tuple[float, float, float] | None
    raw_hand_center_base: tuple[float, float, float] | None
    raw_hand_normal_base: tuple[float, float, float] | None
    object_age_sec: float | None
    hand_age_sec: float | None
    object_jump_rejected: bool
    hand_jump_rejected: bool
    approach_frames: int
    activation_frames: int
    hand_approach_latched: bool


class PerceptionFusion:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        sync_cfg = config.get("perception", {}).get("synchronization", {})
        fusion_cfg = config.get("perception", {}).get("fusion", {})
        activation_cfg = config.get("perception", {}).get("activation", {})
        lift_cfg = activation_cfg.get("lift_detection", {})
        frame_cfg = config.get("frames", {}).get("height_axis", {})

        self.stale_timeout_sec = float(sync_cfg.get("stale_timeout_sec", 0.30))
        self.object_centroid_alpha = float(fusion_cfg.get("object_centroid_alpha", 0.35))
        self.hand_center_alpha = float(fusion_cfg.get("hand_center_alpha", 0.35))
        self.hand_normal_alpha = float(fusion_cfg.get("hand_normal_alpha", 0.50))
        self.max_position_jump_m = float(fusion_cfg.get("max_position_jump_m", 0.20))
        self.stable_frames_required = int(
            fusion_cfg.get("stable_frames_required", activation_cfg.get("stable_frames_required", 5))
        )
        self.hand_approach_distance_threshold_m = float(
            activation_cfg.get("hand_approach_distance_threshold_m", 0.10)
        )
        self.lift_threshold_m = float(lift_cfg.get("threshold_m", 0.05))
        self.require_hand_approach_first = bool(lift_cfg.get("require_hand_approach_first", True))
        self.height_axis_name = str(
            lift_cfg.get("axis_name") or frame_cfg.get("name") or HEIGHT_AXIS_Y
        ).strip().lower()
        if self.height_axis_name not in HEIGHT_AXIS_TO_INDEX:
            raise ValueError(f"Unsupported height axis: {self.height_axis_name}")
        self.height_axis_index = HEIGHT_AXIS_TO_INDEX[self.height_axis_name]
        self.height_axis_direction = str(lift_cfg.get("positive_direction") or frame_cfg.get("positive_direction") or "up").strip().lower()
        if self.height_axis_direction not in HEIGHT_DIRECTION_TO_SIGN:
            raise ValueError(f"Unsupported height axis direction: {self.height_axis_direction}")
        self.height_axis_sign = float(HEIGHT_DIRECTION_TO_SIGN[self.height_axis_direction])

        self._filtered_object_centroid: np.ndarray | None = None
        self._filtered_hand_center: np.ndarray | None = None
        self._filtered_hand_normal: np.ndarray | None = None
        self._approach_frames = 0
        self._activation_frames = 0
        self._hand_approach_latched = False
        self.last_debug: FusionDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "PerceptionFusion":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._filtered_object_centroid = None
        self._filtered_hand_center = None
        self._filtered_hand_normal = None
        self._approach_frames = 0
        self._activation_frames = 0
        self._hand_approach_latched = False

    def process_states(
        self,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        *,
        now_timestamp: float | None = None,
    ) -> FusionState:
        current_time = float(now_timestamp) if now_timestamp is not None else max(
            float(merged_object.timestamp),
            float(selected_hand.timestamp),
        )

        object_age_sec = None
        hand_age_sec = None
        object_fresh = False
        hand_fresh = False
        object_jump_rejected = False
        hand_jump_rejected = False

        raw_object_centroid = self._to_array(merged_object.centroid_base)
        raw_hand_center = self._to_array(selected_hand.palm_center_base)
        raw_hand_normal = self._to_array(selected_hand.palm_normal_base)

        if merged_object.valid and raw_object_centroid is not None:
            object_age_sec = max(current_time - float(merged_object.timestamp), 0.0)
            object_fresh = object_age_sec <= self.stale_timeout_sec
        if selected_hand.valid and raw_hand_center is not None and raw_hand_normal is not None:
            hand_age_sec = max(current_time - float(selected_hand.timestamp), 0.0)
            hand_fresh = hand_age_sec <= self.stale_timeout_sec

        if object_fresh:
            self._filtered_object_centroid, object_jump_rejected = self._smooth_position(
                self._filtered_object_centroid,
                raw_object_centroid,
                alpha=self.object_centroid_alpha,
            )
            if object_jump_rejected:
                object_fresh = False
        else:
            self._filtered_object_centroid = None

        if hand_fresh:
            self._filtered_hand_center, hand_jump_rejected = self._smooth_position(
                self._filtered_hand_center,
                raw_hand_center,
                alpha=self.hand_center_alpha,
            )
            if hand_jump_rejected:
                hand_fresh = False
            self._filtered_hand_normal = self._smooth_normal(
                self._filtered_hand_normal,
                raw_hand_normal,
                alpha=self.hand_normal_alpha,
            ) if hand_fresh else self._filtered_hand_normal
        else:
            self._filtered_hand_center = None
            self._filtered_hand_normal = None

        filtered_object_centroid = self._filtered_object_centroid if object_fresh else None
        filtered_hand_center = self._filtered_hand_center if hand_fresh else None
        filtered_hand_normal = self._filtered_hand_normal if hand_fresh else None

        hand_object_distance_m = None
        approach_condition = False
        if filtered_object_centroid is not None and filtered_hand_center is not None:
            hand_object_distance_m = float(np.linalg.norm(filtered_hand_center - filtered_object_centroid))
            approach_condition = hand_object_distance_m <= self.hand_approach_distance_threshold_m

        if approach_condition:
            self._approach_frames += 1
        else:
            self._approach_frames = 0
        hand_approach_detected = self._approach_frames >= self.stable_frames_required
        if hand_approach_detected:
            self._hand_approach_latched = True

        lift_height_delta_m = self._compute_lift_delta(
            filtered_object_centroid=filtered_object_centroid,
            initial_centroid_base=merged_object.initial_centroid_base,
        )
        lift_gate = self._hand_approach_latched or not self.require_hand_approach_first
        object_lifted = bool(
            object_fresh
            and self._filtered_object_centroid is not None
            and merged_object.initial_centroid_base is not None
            and lift_gate
            and lift_height_delta_m >= self.lift_threshold_m
        )

        if hand_approach_detected and object_lifted:
            self._activation_frames += 1
        else:
            self._activation_frames = 0
        robot_activation_ready = self._activation_frames >= self.stable_frames_required

        fusion_state = FusionState(
            filtered_object_centroid_base=self._to_tuple(filtered_object_centroid),
            filtered_hand_center_base=self._to_tuple(filtered_hand_center),
            filtered_hand_normal_base=self._to_tuple(filtered_hand_normal),
            hand_object_distance_m=hand_object_distance_m,
            lift_height_delta_m=float(lift_height_delta_m),
            hand_approach_detected=bool(hand_approach_detected),
            hand_approach_latched=bool(self._hand_approach_latched),
            object_lifted=bool(object_lifted),
            robot_activation_ready=bool(robot_activation_ready),
            object_fresh=bool(object_fresh),
            hand_fresh=bool(hand_fresh),
            stable_event_frames=int(self._activation_frames),
            height_axis_name=self.height_axis_name,
            timestamp=current_time,
            valid=bool(object_fresh or hand_fresh),
        )
        self.last_debug = FusionDebug(
            raw_object_centroid_base=self._to_tuple(raw_object_centroid),
            raw_hand_center_base=self._to_tuple(raw_hand_center),
            raw_hand_normal_base=self._to_tuple(raw_hand_normal),
            object_age_sec=object_age_sec,
            hand_age_sec=hand_age_sec,
            object_jump_rejected=bool(object_jump_rejected),
            hand_jump_rejected=bool(hand_jump_rejected),
            approach_frames=int(self._approach_frames),
            activation_frames=int(self._activation_frames),
            hand_approach_latched=bool(self._hand_approach_latched),
        )
        return fusion_state

    def _compute_lift_delta(
        self,
        *,
        filtered_object_centroid: np.ndarray | None,
        initial_centroid_base: tuple[float, float, float] | None,
    ) -> float:
        if filtered_object_centroid is None or initial_centroid_base is None:
            return 0.0
        initial = np.asarray(initial_centroid_base, dtype=np.float32)
        raw_delta = float(filtered_object_centroid[self.height_axis_index] - initial[self.height_axis_index])
        return self.height_axis_sign * raw_delta

    def _smooth_position(
        self,
        previous: np.ndarray | None,
        current: np.ndarray,
        *,
        alpha: float,
    ) -> tuple[np.ndarray, bool]:
        if previous is None:
            return current.astype(np.float32), False
        if float(np.linalg.norm(current - previous)) > self.max_position_jump_m:
            return previous.astype(np.float32), True
        filtered = (1.0 - alpha) * previous + alpha * current
        return filtered.astype(np.float32), False

    @staticmethod
    def _smooth_normal(
        previous: np.ndarray | None,
        current: np.ndarray,
        *,
        alpha: float,
    ) -> np.ndarray | None:
        current_normalized = PerceptionFusion._normalize(current)
        if current_normalized is None:
            return None
        if previous is None:
            return current_normalized
        filtered = (1.0 - alpha) * previous + alpha * current_normalized
        return PerceptionFusion._normalize(filtered)

    @staticmethod
    def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
        if vector is None:
            return None
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32)

    @staticmethod
    def _to_array(vector: tuple[float, float, float] | None) -> np.ndarray | None:
        if vector is None:
            return None
        array = np.asarray(vector, dtype=np.float32).reshape(3)
        if not np.isfinite(array).all():
            return None
        return array

    @staticmethod
    def _to_tuple(vector: np.ndarray | None) -> tuple[float, float, float] | None:
        if vector is None:
            return None
        return tuple(float(value) for value in np.asarray(vector, dtype=np.float32).reshape(3))


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HEIGHT_AXIS_TO_INDEX",
    "HEIGHT_DIRECTION_TO_SIGN",
    "FusionDebug",
    "PerceptionFusion",
]
