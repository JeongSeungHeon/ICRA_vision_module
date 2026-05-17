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
    frame_id: int | None = None
    anchor_frame_id: int | None = None
    last_measured_frame_id: int | None = None
    record_elapsed_s: float | None = None
    anchor_record_elapsed_s: float | None = None
    last_measured_record_elapsed_s: float | None = None
    hand_center_base: tuple[float, float, float] | None = None
    anchor_hand_position_base: tuple[float, float, float] | None = None
    measured_object_base: tuple[float, float, float] | None = None
    measured_grasp_base: tuple[float, float, float] | None = None
    object_offset_base: tuple[float, float, float] | None = None
    grasp_offset_base: tuple[float, float, float] | None = None
    fallback_object_base: tuple[float, float, float] | None = None
    fallback_grasp_base: tuple[float, float, float] | None = None


class HandRelativeFallbackTracker:
    def __init__(self, config: dict[str, Any]) -> None:
        grasp_cfg = config.get("grasp", {})
        fallback_cfg = grasp_cfg.get("hand_relative_fallback", {})

        self.enabled = bool(fallback_cfg.get("enabled", True))
        self.lock_frames = max(1, int(fallback_cfg.get("lock_frames", 5)))
        self.max_dropout_sec = max(float(fallback_cfg.get("max_dropout_sec", 1.0)), 0.0)
        self.require_hand_approach = bool(fallback_cfg.get("require_hand_approach", True))
        self.require_motion_triggered = bool(fallback_cfg.get("require_motion_triggered", True))
        self.debug_log = bool(fallback_cfg.get("debug_log", False))
        self.log_lock_progress = bool(fallback_cfg.get("log_lock_progress", True))
        self.log_fallback_every_frames = max(1, int(fallback_cfg.get("log_fallback_every_frames", 1)))

        self._anchor_locked = False
        self._lock_streak = 0
        self._object_offset_base: np.ndarray | None = None
        self._grasp_offset_base: np.ndarray | None = None
        self._anchor_hand_position_base: np.ndarray | None = None
        self._last_measured_timestamp: float | None = None
        self._anchor_frame_id: int | None = None
        self._last_measured_frame_id: int | None = None
        self._anchor_record_elapsed_s: float | None = None
        self._last_measured_record_elapsed_s: float | None = None
        self._fallback_log_counter = 0
        self._last_anchor_wait_reason: str | None = None
        self._last_anchor_wait_log_key: tuple[str, int] | None = None
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
        self._anchor_frame_id = None
        self._last_measured_frame_id = None
        self._anchor_record_elapsed_s = None
        self._last_measured_record_elapsed_s = None
        self._fallback_log_counter = 0
        self._last_anchor_wait_reason = None
        self._last_anchor_wait_log_key = None
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
        frame_id: int | None = None,
        record_elapsed_s: float | None = None,
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
            return self._invalid_state(current_time, reason="disabled", frame_id=frame_id)

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
        motion_trigger_ok = bool(not self.require_motion_triggered or motion_triggered)

        if measured_available:
            self._last_measured_timestamp = current_time
            self._last_measured_frame_id = frame_id
            self._last_measured_record_elapsed_s = record_elapsed_s
            if not self._anchor_locked and hand_approach_ok and motion_trigger_ok:
                self._lock_streak += 1
                if self._lock_streak >= self.lock_frames:
                    self._anchor_locked = True
                    self._object_offset_base = measured_object - hand_center
                    self._grasp_offset_base = measured_grasp - hand_center
                    self._anchor_hand_position_base = hand_center.copy()
                    self._anchor_frame_id = frame_id
                    self._anchor_record_elapsed_s = record_elapsed_s
                    self._fallback_log_counter = 0
                    self._last_anchor_wait_reason = None
                    self._last_anchor_wait_log_key = None
                    self._log(
                        "ANCHOR_LOCKED "
                        f"clock={self._format_record_clock(record_elapsed_s)} "
                        f"lock={self._lock_streak}/{self.lock_frames} "
                        f"hand={self._format_vec(hand_center)} "
                        f"object={self._format_vec(measured_object)} "
                        f"grasp={self._format_vec(measured_grasp)} "
                        f"object_offset={self._format_vec(self._object_offset_base)} "
                        f"grasp_offset={self._format_vec(self._grasp_offset_base)}"
                    )
                elif self.log_lock_progress:
                    self._log_anchor_wait(
                        "measured_available",
                        record_elapsed_s=record_elapsed_s,
                        hand_center=hand_center,
                        measured_object=measured_object,
                        measured_grasp=measured_grasp,
                        hand_approach_ok=hand_approach_ok,
                        motion_trigger_ok=motion_trigger_ok,
                    )
            elif not self._anchor_locked:
                self._lock_streak = 0
            reason = "measured_available"
            if not hand_approach_ok:
                reason = "hand_approach_required"
            elif not motion_trigger_ok:
                reason = "motion_trigger_required"
            if not self._anchor_locked and reason != "measured_available":
                self._log_anchor_wait(
                    reason,
                    record_elapsed_s=record_elapsed_s,
                    hand_center=hand_center,
                    measured_object=measured_object,
                    measured_grasp=measured_grasp,
                    hand_approach_ok=hand_approach_ok,
                    motion_trigger_ok=motion_trigger_ok,
                )
            self.last_debug = HandRelativeFallbackDebug(
                anchor_locked=self._anchor_locked,
                lock_streak=self._lock_streak,
                used_filtered_hand_center=used_filtered_hand_center,
                dropout_age_s=0.0,
                reason=reason,
                frame_id=frame_id,
                anchor_frame_id=self._anchor_frame_id,
                last_measured_frame_id=self._last_measured_frame_id,
                record_elapsed_s=record_elapsed_s,
                anchor_record_elapsed_s=self._anchor_record_elapsed_s,
                last_measured_record_elapsed_s=self._last_measured_record_elapsed_s,
                hand_center_base=self._to_tuple(hand_center),
                measured_object_base=self._to_tuple(measured_object),
                measured_grasp_base=self._to_tuple(measured_grasp),
                object_offset_base=self._to_tuple(self._object_offset_base),
                grasp_offset_base=self._to_tuple(self._grasp_offset_base),
            )
            return HandRelativeFallbackState(
                object_position_base=None,
                grasp_position_base=None,
                anchor_hand_position_base=None if self._anchor_hand_position_base is None else self._to_tuple(self._anchor_hand_position_base),
                dropout_age_s=0.0,
                reason=reason,
                timestamp=current_time,
                valid=False,
            )

        self._lock_streak = 0
        if hand_center is None:
            if not self._anchor_locked:
                self._log_anchor_wait(
                    "no_hand_center",
                    record_elapsed_s=record_elapsed_s,
                    hand_center=hand_center,
                    measured_object=measured_object,
                    measured_grasp=measured_grasp,
                    hand_approach_ok=hand_approach_ok,
                    motion_trigger_ok=motion_trigger_ok,
                )
            return self._invalid_state(current_time, reason="no_hand_center", used_filtered_hand_center=used_filtered_hand_center, frame_id=frame_id)
        if not self._anchor_locked or self._object_offset_base is None or self._grasp_offset_base is None:
            if measured_object is None:
                reason = "no_measured_object"
            elif measured_grasp is None:
                reason = "no_measured_grasp"
            else:
                reason = "anchor_not_locked"
            self._log_anchor_wait(
                reason,
                record_elapsed_s=record_elapsed_s,
                hand_center=hand_center,
                measured_object=measured_object,
                measured_grasp=measured_grasp,
                hand_approach_ok=hand_approach_ok,
                motion_trigger_ok=motion_trigger_ok,
            )
            return self._invalid_state(current_time, reason=reason, used_filtered_hand_center=used_filtered_hand_center, frame_id=frame_id)
        if self.require_motion_triggered and not motion_triggered:
            return self._invalid_state(current_time, reason="motion_not_triggered", used_filtered_hand_center=used_filtered_hand_center, frame_id=frame_id)
        if self._last_measured_timestamp is None:
            return self._invalid_state(current_time, reason="no_measured_history", used_filtered_hand_center=used_filtered_hand_center, frame_id=frame_id)

        dropout_age_s = max(current_time - float(self._last_measured_timestamp), 0.0)
        if dropout_age_s > self.max_dropout_sec:
            return self._invalid_state(
                current_time,
                reason="dropout_timeout",
                used_filtered_hand_center=used_filtered_hand_center,
                dropout_age_s=dropout_age_s,
                frame_id=frame_id,
            )

        object_position = hand_center + self._object_offset_base
        grasp_position = hand_center + self._grasp_offset_base
        self._fallback_log_counter += 1
        if (self._fallback_log_counter - 1) % self.log_fallback_every_frames == 0:
            self._log(
                "ACTIVE "
                f"clock={self._format_record_clock(record_elapsed_s)} "
                f"anchor_clock={self._format_record_clock(self._anchor_record_elapsed_s)} "
                f"last_measured_clock={self._format_record_clock(self._last_measured_record_elapsed_s)} "
                f"dropout_age={dropout_age_s:.3f}s "
                f"hand={self._format_vec(hand_center)} "
                f"grasp={self._format_vec(grasp_position)} "
                f"object={self._format_vec(object_position)} "
                f"grasp_offset={self._format_vec(self._grasp_offset_base)}"
            )
        self.last_debug = HandRelativeFallbackDebug(
            anchor_locked=True,
            lock_streak=self._lock_streak,
            used_filtered_hand_center=used_filtered_hand_center,
            dropout_age_s=dropout_age_s,
            reason="fallback_active",
            frame_id=frame_id,
            anchor_frame_id=self._anchor_frame_id,
            last_measured_frame_id=self._last_measured_frame_id,
            record_elapsed_s=record_elapsed_s,
            anchor_record_elapsed_s=self._anchor_record_elapsed_s,
            last_measured_record_elapsed_s=self._last_measured_record_elapsed_s,
            hand_center_base=self._to_tuple(hand_center),
            object_offset_base=self._to_tuple(self._object_offset_base),
            grasp_offset_base=self._to_tuple(self._grasp_offset_base),
            fallback_object_base=self._to_tuple(object_position),
            fallback_grasp_base=self._to_tuple(grasp_position),
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
        frame_id: int | None = None,
    ) -> HandRelativeFallbackState:
        self.last_debug = HandRelativeFallbackDebug(
            anchor_locked=self._anchor_locked,
            lock_streak=self._lock_streak,
            used_filtered_hand_center=used_filtered_hand_center,
            dropout_age_s=dropout_age_s,
            reason=reason,
            frame_id=frame_id,
            anchor_frame_id=self._anchor_frame_id,
            last_measured_frame_id=self._last_measured_frame_id,
            anchor_record_elapsed_s=self._anchor_record_elapsed_s,
            last_measured_record_elapsed_s=self._last_measured_record_elapsed_s,
            anchor_hand_position_base=self._to_tuple(self._anchor_hand_position_base),
            object_offset_base=self._to_tuple(self._object_offset_base),
            grasp_offset_base=self._to_tuple(self._grasp_offset_base),
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
    def _format_frame(frame_id: int | None) -> str:
        return "-" if frame_id is None else str(int(frame_id))

    @staticmethod
    def _format_record_clock(elapsed_s: float | None) -> str:
        if elapsed_s is None:
            return "REC --:--.-"
        elapsed_s = max(float(elapsed_s), 0.0)
        minutes = int(elapsed_s // 60.0)
        seconds = int(elapsed_s % 60.0)
        tenths = int((elapsed_s - int(elapsed_s)) * 10.0)
        return f"REC {minutes:02d}:{seconds:02d}.{tenths:d}"

    @staticmethod
    def _format_vec(values: np.ndarray | tuple[float, float, float] | None) -> str:
        if values is None:
            return "None"
        array = np.asarray(values, dtype=np.float32).reshape(3)
        return f"mm=({array[0] * 1000.0:.1f},{array[1] * 1000.0:.1f},{array[2] * 1000.0:.1f})"

    def _log(self, message: str) -> None:
        if self.debug_log:
            print(f"[HAND_FALLBACK] {message}", flush=True)

    def _log_anchor_wait(
        self,
        reason: str,
        *,
        record_elapsed_s: float | None,
        hand_center: np.ndarray | None,
        measured_object: np.ndarray | None,
        measured_grasp: np.ndarray | None,
        hand_approach_ok: bool,
        motion_trigger_ok: bool,
    ) -> None:
        if not self.debug_log or self._anchor_locked:
            return

        reason_changed = reason != self._last_anchor_wait_reason
        if reason_changed:
            self._last_anchor_wait_reason = reason

        log_key = (reason, int(self._lock_streak))
        progress_changed = log_key != self._last_anchor_wait_log_key
        should_log = reason_changed or progress_changed
        if not should_log:
            return
        self._last_anchor_wait_log_key = log_key

        self._log(
            "ANCHOR_WAIT "
            f"clock={self._format_record_clock(record_elapsed_s)} "
            f"reason={reason} "
            f"lock={self._lock_streak}/{self.lock_frames} "
            f"hand_approach={hand_approach_ok} "
            f"motion_triggered={motion_trigger_ok} "
            f"hand={self._format_vec(hand_center)} "
            f"object={self._format_vec(measured_object)} "
            f"grasp={self._format_vec(measured_grasp)}"
        )

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
