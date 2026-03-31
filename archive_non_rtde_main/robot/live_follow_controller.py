"""Live follow-servo target policy inspired by robot_control_sebin.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from system.shared_state import (
    FusionState,
    GraspTargetState,
    HEIGHT_AXIS_X,
    HEIGHT_AXIS_Y,
    HEIGHT_AXIS_Z,
    LiveFollowState,
    RobotState,
    TASK_IDLE,
)

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
_AXIS_TO_INDEX = {HEIGHT_AXIS_X: 0, HEIGHT_AXIS_Y: 1, HEIGHT_AXIS_Z: 2}


@dataclass
class LiveFollowControllerDebug:
    raw_target_position_base: tuple[float, float, float] | None
    approach_target_position_base: tuple[float, float, float] | None
    servo_target_position_base: tuple[float, float, float] | None
    valid_streak_count: int
    target_age_sec: float | None
    target_is_fresh: bool
    timed_out: bool
    workspace_clamped: bool
    reason: str


class LiveFollowController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        robot_cfg = config.get("robot", {})
        live_cfg = robot_cfg.get("live_follow", {})
        safety_cfg = config.get("safety", {})
        grasp_cfg = config.get("grasp", {})

        self.control_mode = str(robot_cfg.get("control_mode", "classic_state_machine"))
        self.enabled = bool(live_cfg.get("enabled", False))
        self.control_hz = float(live_cfg.get("control_hz", 30.0))
        self.min_valid_count = int(live_cfg.get("min_valid_count", 3))
        self.target_timeout_sec = float(live_cfg.get("target_timeout_sec", 0.5))
        self.follow_z = bool(live_cfg.get("follow_z", True))
        self.approach_axis = str(live_cfg.get("approach_axis", "x")).strip().lower()
        if self.approach_axis not in _AXIS_TO_INDEX:
            raise ValueError(f"Unsupported live-follow approach axis: {self.approach_axis}")
        self.approach_axis_index = _AXIS_TO_INDEX[self.approach_axis]
        self.approach_offset_m = float(live_cfg.get("approach_offset_m", -0.08))
        self.max_xy_step_m = float(live_cfg.get("max_xy_step_m", 0.0027))
        self.max_z_step_m = float(live_cfg.get("max_z_step_m", 0.0027))
        self.workspace_clamp_enabled = bool(live_cfg.get("workspace_clamp_enabled", True))
        self.hold_last_target_until_timeout = bool(live_cfg.get("hold_last_target_until_timeout", True))
        self.default_fixed_orientation_base = tuple(
            float(value) for value in grasp_cfg.get("fixed_orientation_base", [0.0, 0.0, 0.0])
        )

        workspace_cfg = safety_cfg.get("workspace_bounds_m", {})
        self.workspace_bounds_m = {
            axis: tuple(float(v) for v in workspace_cfg.get(axis, [-np.inf, np.inf]))
            for axis in ("x", "y", "z")
        }

        self._valid_streak_count = 0
        self._last_seen_target_at: float | None = None
        self._last_raw_target_position_base: tuple[float, float, float] | None = None
        self._last_output_target_position_base: tuple[float, float, float] | None = None
        self.last_debug: LiveFollowControllerDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "LiveFollowController":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._valid_streak_count = 0
        self._last_seen_target_at = None
        self._last_raw_target_position_base = None
        self._last_output_target_position_base = None
        self.last_debug = None

    def process_states(
        self,
        grasp_target: GraspTargetState,
        fusion_state: FusionState,
        robot_state: RobotState,
        *,
        source_mode: str = TASK_IDLE,
        now_timestamp: float | None = None,
    ) -> LiveFollowState:
        current_time = float(now_timestamp) if now_timestamp is not None else max(
            float(grasp_target.timestamp),
            float(fusion_state.timestamp),
            float(robot_state.timestamp),
        )

        raw_target = None
        approach_target = None
        servo_target = None
        target_is_fresh = False
        timed_out = False
        workspace_clamped = False
        reason = "disabled" if not self.enabled else "waiting_for_target"

        if grasp_target.valid and grasp_target.target_position_base is not None:
            raw_target_np = self._to_vec3(grasp_target.target_position_base)
            raw_target = tuple(float(v) for v in raw_target_np)
            self._last_raw_target_position_base = raw_target
            self._last_seen_target_at = current_time
            self._valid_streak_count += 1
            target_is_fresh = True
            reason = "fresh_target"
        else:
            self._valid_streak_count = 0

        target_age_sec = None
        if self._last_seen_target_at is not None:
            target_age_sec = max(current_time - self._last_seen_target_at, 0.0)
            timed_out = target_age_sec > self.target_timeout_sec

        candidate_raw = raw_target
        if candidate_raw is None and self.hold_last_target_until_timeout and not timed_out:
            candidate_raw = self._last_raw_target_position_base
            if candidate_raw is not None:
                reason = "holding_last_target"

        if candidate_raw is not None and self.enabled:
            approach_target_np = self._apply_approach_offset(self._to_vec3(candidate_raw))
            if not self.follow_z:
                approach_target_np = self._freeze_z_to_current_robot(approach_target_np, robot_state)
            approach_target = tuple(float(v) for v in approach_target_np)

            servo_target_np = self._limit_step_from_robot(approach_target_np, robot_state)
            servo_target_np, workspace_clamped = self._clamp_workspace(servo_target_np)
            servo_target = tuple(float(v) for v in servo_target_np)
            self._last_output_target_position_base = servo_target

            if self._valid_streak_count < self.min_valid_count and raw_target is not None:
                reason = "warming_up_valid_streak"
            elif workspace_clamped:
                reason = "workspace_clamped"
            else:
                reason = "servo_target_ready"
        elif timed_out:
            reason = "target_timed_out"

        state_valid = bool(
            self.enabled
            and servo_target is not None
            and not timed_out
            and self._valid_streak_count >= self.min_valid_count
        )

        state = LiveFollowState(
            follow_enabled=bool(self.enabled),
            source_mode=str(source_mode),
            reason=reason,
            raw_target_position_base=raw_target,
            approach_target_position_base=approach_target,
            servo_target_position_base=servo_target,
            fixed_orientation_base=self.default_fixed_orientation_base,
            target_is_fresh=bool(target_is_fresh),
            timed_out=bool(timed_out),
            workspace_clamped=bool(workspace_clamped),
            follow_z_enabled=bool(self.follow_z),
            valid_streak_count=int(self._valid_streak_count),
            timestamp=current_time,
            valid=state_valid,
        )

        self.last_debug = LiveFollowControllerDebug(
            raw_target_position_base=raw_target,
            approach_target_position_base=approach_target,
            servo_target_position_base=servo_target,
            valid_streak_count=int(self._valid_streak_count),
            target_age_sec=target_age_sec,
            target_is_fresh=bool(target_is_fresh),
            timed_out=bool(timed_out),
            workspace_clamped=bool(workspace_clamped),
            reason=reason,
        )
        return state

    def _apply_approach_offset(self, target_position_base: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(target_position_base, dtype=np.float32).reshape(3).copy()
        adjusted[self.approach_axis_index] += float(self.approach_offset_m)
        return adjusted

    def _freeze_z_to_current_robot(self, target_position_base: np.ndarray, robot_state: RobotState) -> np.ndarray:
        adjusted = np.asarray(target_position_base, dtype=np.float32).reshape(3).copy()
        if robot_state.actual_tcp_pose_base is not None:
            adjusted[2] = float(robot_state.actual_tcp_pose_base[2])
        elif self._last_output_target_position_base is not None:
            adjusted[2] = float(self._last_output_target_position_base[2])
        return adjusted

    def _limit_step_from_robot(self, target_position_base: np.ndarray, robot_state: RobotState) -> np.ndarray:
        target = np.asarray(target_position_base, dtype=np.float32).reshape(3).copy()

        # Determine the reference position we step *from*.
        # Preference order:
        #   1. Actual TCP pose from RTDE (most accurate)
        #   2. Last servo output target (safe continuity)
        #   3. Unknown — return target unchanged but log implicitly via streak
        if robot_state is not None and robot_state.actual_tcp_pose_base is not None:
            current = np.asarray(robot_state.actual_tcp_pose_base[:3], dtype=np.float32).reshape(3)
        elif self._last_output_target_position_base is not None:
            current = np.asarray(self._last_output_target_position_base, dtype=np.float32).reshape(3)
        else:
            # No reference at all — clamp target to the raw target itself;
            # this avoids a huge initial step if the robot hasn't reported
            # its position yet.
            return target

        delta = target - current
        delta[0] = float(np.clip(delta[0], -self.max_xy_step_m, self.max_xy_step_m))
        delta[1] = float(np.clip(delta[1], -self.max_xy_step_m, self.max_xy_step_m))
        delta[2] = float(np.clip(delta[2], -self.max_z_step_m, self.max_z_step_m))
        return (current + delta).astype(np.float32)

    def _clamp_workspace(self, target_position_base: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.workspace_clamp_enabled:
            return np.asarray(target_position_base, dtype=np.float32).reshape(3), False
        clamped = np.asarray(target_position_base, dtype=np.float32).reshape(3).copy()
        before = clamped.copy()
        clamped[0] = float(np.clip(clamped[0], *self.workspace_bounds_m["x"]))
        clamped[1] = float(np.clip(clamped[1], *self.workspace_bounds_m["y"]))
        clamped[2] = float(np.clip(clamped[2], *self.workspace_bounds_m["z"]))
        return clamped, bool(np.linalg.norm(clamped - before) > 1e-9)

    @staticmethod
    def _to_vec3(values: tuple[float, float, float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape(3)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "LiveFollowController",
    "LiveFollowControllerDebug",
]
