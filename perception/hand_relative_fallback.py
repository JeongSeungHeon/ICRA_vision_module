"""Hand-relative fallback target reconstruction for segmentation dropouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from system.shared_state import FusionState, HandRelativeFallbackState, SelectedHandState

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class HandRelativeFallbackDebug:
    anchor_locked: bool
    lock_streak: int
    used_filtered_hand_center: bool
    dropout_age_s: float | None
    reason: str


class HandRelativeFallbackTracker:
    def __init__(self, config: dict[str, Any]) -> None:
        grasp_cfg = config.get("grasp", {})
        fallback_cfg = grasp_cfg.get("hand_relative_fallback", {})

        self.enabled = bool(fallback_cfg.get("enabled", True))
        self.lock_frames = max(1, int(fallback_cfg.get("lock_frames", 5)))
        self.max_dropout_sec = max(float(fallback_cfg.get("max_dropout_sec", 1.0)), 0.0)
        self.require_hand_approach = bool(fallback_cfg.get("require_hand_approach", True))
        self.require_motion_triggered = bool(fallback_cfg.get("require_motion_triggered", True))

        self._anchor_locked = False
        self._lock_streak = 0
        self._object_offset_base: np.ndarray | None = None
        self._grasp_offset_base: np.ndarray | None = None
        self._anchor_hand_position_base: np.ndarray | None = None
        self._last_measured_timestamp: float | None = None
        self.last_debug = HandRelativeFallbackDebug(
            anchor_locked=False,
            lock_streak=0,
            used_filtered_hand_center=False,
            dropout_age_s=None,
            reason="uninitialized",
        )

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "HandRelativeFallbackTracker":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._anchor_locked = False
        self._lock_streak = 0
        self._object_offset_base = None
        self._grasp_offset_base = None
        self._anchor_hand_position_base = None
        self._last_measured_timestamp = None
        self.last_debug = HandRelativeFallbackDebug(
            anchor_locked=False,
            lock_streak=0,
            used_filtered_hand_center=False,
            dropout_age_s=None,
            reason="reset",
        )

    def process(
        self,
        *,
        measured_object_position_base: tuple[float, float, float] | None,
        measured_grasp_position_base: tuple[float, float, float] | None,
        selected_hand: SelectedHandState,
        fusion_state: FusionState | None,
        motion_triggered: bool,
        now_timestamp: float | None = None,
    ) -> HandRelativeFallbackState:
        current_time = (
            float(now_timestamp)
            if now_timestamp is not None
            else max(
                float(getattr(selected_hand, "timestamp", 0.0)),
                float(getattr(fusion_state, "timestamp", 0.0)) if fusion_state is not None else 0.0,
            )
        )

        if not self.enabled:
            return self._invalid_state(current_time, reason="disabled")

        hand_center, used_filtered_hand_center = self._resolve_hand_center(selected_hand, fusion_state)
        measured_object = self._to_array(measured_object_position_base)
        measured_grasp = self._to_array(measured_grasp_position_base)
        measured_available = hand_center is not None and measured_object is not None and measured_grasp is not None
        hand_approach_ok = bool(
            not self.require_hand_approach
            or (
                fusion_state is not None
                and bool(
                    getattr(fusion_state, "hand_approach_detected", False)
                    or getattr(fusion_state, "hand_approach_latched", False)
                )
            )
        )

        if measured_available:
            self._last_measured_timestamp = current_time
            if not self._anchor_locked and hand_approach_ok:
                self._lock_streak += 1
                if self._lock_streak >= self.lock_frames:
                    self._anchor_locked = True
                    self._object_offset_base = measured_object - hand_center
                    self._grasp_offset_base = measured_grasp - hand_center
                    self._anchor_hand_position_base = hand_center.copy()
            elif not self._anchor_locked:
                self._lock_streak = 0
            self.last_debug = HandRelativeFallbackDebug(
                anchor_locked=self._anchor_locked,
                lock_streak=self._lock_streak,
                used_filtered_hand_center=used_filtered_hand_center,
                dropout_age_s=0.0,
                reason="measured_available" if hand_approach_ok else "hand_approach_required",
            )
            return HandRelativeFallbackState(
                object_position_base=None,
                grasp_position_base=None,
                anchor_hand_position_base=None if self._anchor_hand_position_base is None else self._to_tuple(self._anchor_hand_position_base),
                dropout_age_s=0.0,
                reason="measured_available",
                timestamp=current_time,
                valid=False,
            )

        self._lock_streak = 0
        if hand_center is None:
            return self._invalid_state(current_time, reason="no_hand_center", used_filtered_hand_center=used_filtered_hand_center)
        if not self._anchor_locked or self._object_offset_base is None or self._grasp_offset_base is None:
            return self._invalid_state(current_time, reason="anchor_not_locked", used_filtered_hand_center=used_filtered_hand_center)
        if self.require_motion_triggered and not motion_triggered:
            return self._invalid_state(current_time, reason="motion_not_triggered", used_filtered_hand_center=used_filtered_hand_center)
        if self._last_measured_timestamp is None:
            return self._invalid_state(current_time, reason="no_measured_history", used_filtered_hand_center=used_filtered_hand_center)

        dropout_age_s = max(current_time - float(self._last_measured_timestamp), 0.0)
        if dropout_age_s > self.max_dropout_sec:
            return self._invalid_state(
                current_time,
                reason="dropout_timeout",
                used_filtered_hand_center=used_filtered_hand_center,
                dropout_age_s=dropout_age_s,
            )

        object_position = hand_center + self._object_offset_base
        grasp_position = hand_center + self._grasp_offset_base
        self.last_debug = HandRelativeFallbackDebug(
            anchor_locked=True,
            lock_streak=self._lock_streak,
            used_filtered_hand_center=used_filtered_hand_center,
            dropout_age_s=dropout_age_s,
            reason="fallback_active",
        )
        return HandRelativeFallbackState(
            object_position_base=self._to_tuple(object_position),
            grasp_position_base=self._to_tuple(grasp_position),
            anchor_hand_position_base=self._to_tuple(self._anchor_hand_position_base),
            dropout_age_s=float(dropout_age_s),
            reason="fallback_active",
            timestamp=current_time,
            valid=True,
        )

    def _invalid_state(
        self,
        timestamp: float,
        *,
        reason: str,
        used_filtered_hand_center: bool = False,
        dropout_age_s: float | None = None,
    ) -> HandRelativeFallbackState:
        self.last_debug = HandRelativeFallbackDebug(
            anchor_locked=self._anchor_locked,
            lock_streak=self._lock_streak,
            used_filtered_hand_center=used_filtered_hand_center,
            dropout_age_s=dropout_age_s,
            reason=reason,
        )
        return HandRelativeFallbackState(
            object_position_base=None,
            grasp_position_base=None,
            anchor_hand_position_base=None if self._anchor_hand_position_base is None else self._to_tuple(self._anchor_hand_position_base),
            dropout_age_s=dropout_age_s,
            reason=reason,
            timestamp=float(timestamp),
            valid=False,
        )

    @staticmethod
    def _to_array(values: tuple[float, float, float] | None) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(array)):
            return None
        return array

    @staticmethod
    def _to_tuple(values: np.ndarray | None) -> tuple[float, float, float] | None:
        if values is None:
            return None
        return tuple(float(v) for v in np.asarray(values, dtype=np.float32).reshape(3))

    @staticmethod
    def _resolve_hand_center(
        selected_hand: SelectedHandState,
        fusion_state: FusionState | None,
    ) -> tuple[np.ndarray | None, bool]:
        if (
            fusion_state is not None
            and fusion_state.valid
            and fusion_state.hand_fresh
            and fusion_state.filtered_hand_center_base is not None
        ):
            return np.asarray(fusion_state.filtered_hand_center_base, dtype=np.float32).reshape(3), True
        if selected_hand.valid and selected_hand.palm_center_base is not None:
            return np.asarray(selected_hand.palm_center_base, dtype=np.float32).reshape(3), False
        return None, False


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HandRelativeFallbackDebug",
    "HandRelativeFallbackTracker",
]
