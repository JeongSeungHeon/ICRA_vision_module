"""High-level task manager for the dual-camera receive-and-place system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import yaml

from system.shared_state import (
    GRIPPER_CLOSE,
    GRIPPER_HOLD,
    GRIPPER_OPEN,
    HEIGHT_AXIS_Y,
    MergedObjectState,
    RobotCommandState,
    RobotState,
    SelectedHandState,
    FusionState,
    GraspTargetState,
    LiveFollowState,
    TASK_ACTIVATE_ROBOT,
    TASK_APPROACH_GRASP_POINT,
    TASK_FAIL_SAFE,
    TASK_FOLLOW_SERVO,
    TASK_IDLE,
    TASK_MOVE_TO_INITIAL_PLACE,
    TASK_OBSERVE_INITIAL_OBJECT,
    TASK_RECEIVE_OBJECT,
    TASK_RELEASE,
    TASK_RETREAT,
    TASK_VERIFY_GRASP,
    TASK_WAIT_FOR_HAND_APPROACH,
    TASK_WAIT_FOR_OBJECT_LIFT,
    TaskState,
    ROBOT_CMD_HOLD,
    ROBOT_CMD_MOVE_TO_POSITION,
    ROBOT_CMD_SERVO_TO_POSITION,
    ROBOT_CMD_STOP,
)

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
HEIGHT_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class TaskManagerDebug:
    previous_mode: str
    current_mode: str
    transition_reason: str
    state_age_sec: float
    placement_ready: bool
    at_target: bool
    fail_safe_triggered: bool
    memorized_initial_centroid: Tuple[float, float, float] | None
    retreat_target_position_base: Tuple[float, float, float] | None


class TaskManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        frame_cfg = config.get("frames", {}).get("height_axis", {})
        placement_cfg = config.get("placement", {})
        task_cfg = config.get("task_manager", {})
        grasp_cfg = config.get("grasp", {})
        robot_cfg = config.get("robot", {})

        self.control_mode = str(robot_cfg.get("control_mode", "classic_state_machine")).strip().lower()
        self.follow_mode_enabled = self.control_mode == "live_follow_servo"

        self.height_axis_name = str(frame_cfg.get("name", HEIGHT_AXIS_Y)).strip().lower()
        if self.height_axis_name not in HEIGHT_AXIS_TO_INDEX:
            raise ValueError("Unsupported height axis: %s" % self.height_axis_name)
        self.height_axis_index = HEIGHT_AXIS_TO_INDEX[self.height_axis_name]

        self.use_initial_object_centroid = bool(placement_cfg.get("use_initial_object_centroid", True))
        self.release_height_offset_m = float(placement_cfg.get("release_height_offset_m", 0.0))
        self.position_tolerance_m = float(placement_cfg.get("position_tolerance_m", 0.01))
        self.default_fixed_orientation_base = tuple(
            float(value) for value in grasp_cfg.get("fixed_orientation_base", [0.0, 0.0, 0.0])
        )

        self.approach_grasp_timeout_sec = float(task_cfg.get("approach_grasp_timeout_sec", 8.0))
        self.receive_close_dwell_sec = float(task_cfg.get("receive_close_dwell_sec", 0.40))
        self.verify_grasp_timeout_sec = float(task_cfg.get("verify_grasp_timeout_sec", 2.0))
        self.verify_grasp_force_only_fallback = bool(task_cfg.get("verify_grasp_force_only_fallback", False))
        self.move_to_place_timeout_sec = float(task_cfg.get("move_to_place_timeout_sec", 8.0))
        self.release_dwell_sec = float(task_cfg.get("release_dwell_sec", 0.50))
        self.retreat_distance_m = float(task_cfg.get("retreat_distance_m", 0.08))
        self.retreat_timeout_sec = float(task_cfg.get("retreat_timeout_sec", 4.0))
        self.missing_grasp_target_timeout_sec = float(task_cfg.get("missing_grasp_target_timeout_sec", 0.50))
        # After this many seconds in FAIL_SAFE the task manager resets to IDLE
        # so the operator can retry without restarting the process.
        self.fail_safe_auto_recover_sec = float(task_cfg.get("fail_safe_auto_recover_sec", 5.0))

        self._mode = TASK_IDLE
        self._mode_entered_at = 0.0
        self._placement_position_base: Tuple[float, float, float] | None = None
        self._memorized_initial_centroid: Tuple[float, float, float] | None = None
        self._retreat_target_position_base: Tuple[float, float, float] | None = None
        self._latched_grasp_target_position_base: Tuple[float, float, float] | None = None
        self._latched_grasp_fixed_orientation_base: Tuple[float, ...] | None = None
        self._missing_grasp_target_since: float | None = None
        self._last_transition_reason = "initialize"
        self.last_debug: TaskManagerDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "TaskManager":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._mode = TASK_IDLE
        self._mode_entered_at = 0.0
        self._placement_position_base = None
        self._memorized_initial_centroid = None
        self._retreat_target_position_base = None
        self._latched_grasp_target_position_base = None
        self._latched_grasp_fixed_orientation_base = None
        self._missing_grasp_target_since = None
        self._last_transition_reason = "reset"

    def process_states(
        self,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        grasp_target: GraspTargetState,
        robot_state: RobotState,
        fusion_state: FusionState | None = None,
        live_follow_state: LiveFollowState | None = None,
        *,
        safety_ok: bool = True,
        now_timestamp: float | None = None,
    ) -> tuple[TaskState, RobotCommandState]:
        current_time = float(now_timestamp) if now_timestamp is not None else max(
            float(merged_object.timestamp),
            float(selected_hand.timestamp),
            float(grasp_target.timestamp),
            float(robot_state.timestamp),
        )
        if self._mode_entered_at <= 0.0:
            self._mode_entered_at = current_time

        self._refresh_placement_target(merged_object)
        self._update_latched_grasp_target(grasp_target)
        current_mode = self._mode
        state_age_sec = max(current_time - self._mode_entered_at, 0.0)

        hand_approach_detected = bool(
            fusion_state is not None and (fusion_state.hand_approach_detected or fusion_state.hand_approach_latched)
        )
        object_lifted = bool(fusion_state is not None and fusion_state.object_lifted)
        grasp_verified = bool(robot_state.grasp_verified_force_current)
        at_grasp_target = self._robot_at_position(robot_state, grasp_target.target_position_base)
        at_place_target = self._robot_at_position(robot_state, self._placement_position_base)
        at_retreat_target = self._robot_at_position(robot_state, self._retreat_target_position_base)

        fail_safe_reason = self._check_fail_safe(current_mode, state_age_sec, robot_state, safety_ok)
        if fail_safe_reason is not None and current_mode != TASK_FAIL_SAFE:
            self._transition(TASK_FAIL_SAFE, fail_safe_reason, current_time)
            current_mode = self._mode
            state_age_sec = 0.0

        if current_mode == TASK_IDLE:
            if merged_object.valid:
                self._transition(TASK_OBSERVE_INITIAL_OBJECT, "merged_object_valid", current_time)
        elif current_mode == TASK_OBSERVE_INITIAL_OBJECT:
            if not merged_object.valid:
                self._transition(TASK_IDLE, "merged_object_lost", current_time)
            elif self._placement_position_base is not None:
                self._transition(TASK_WAIT_FOR_HAND_APPROACH, "initial_centroid_memorized", current_time)
        elif current_mode == TASK_WAIT_FOR_HAND_APPROACH:
            if not merged_object.valid:
                self._transition(TASK_OBSERVE_INITIAL_OBJECT, "merged_object_lost", current_time)
            elif hand_approach_detected:
                self._transition(TASK_WAIT_FOR_OBJECT_LIFT, "hand_approach_detected", current_time)
        elif current_mode == TASK_WAIT_FOR_OBJECT_LIFT:
            if not merged_object.valid:
                self._transition(TASK_OBSERVE_INITIAL_OBJECT, "merged_object_lost", current_time)
            elif object_lifted:
                self._transition(TASK_ACTIVATE_ROBOT, "object_lifted", current_time)
            elif not hand_approach_detected:
                self._transition(TASK_WAIT_FOR_HAND_APPROACH, "hand_approach_lost", current_time)
        elif current_mode == TASK_ACTIVATE_ROBOT:
            follow_ready = bool(live_follow_state is not None and live_follow_state.valid)
            grasp_ready = self._latched_grasp_target_position_base is not None
            if self.follow_mode_enabled and follow_ready:
                self._missing_grasp_target_since = None
                self._transition(TASK_FOLLOW_SERVO, "live_follow_ready", current_time)
            elif (not self.follow_mode_enabled) and grasp_ready:
                self._missing_grasp_target_since = None
                self._transition(TASK_APPROACH_GRASP_POINT, "valid_grasp_target", current_time)
        elif current_mode == TASK_FOLLOW_SERVO:
            if live_follow_state is not None and live_follow_state.valid and live_follow_state.servo_target_position_base is not None:
                self._missing_grasp_target_since = None
            else:
                if self._missing_grasp_target_since is None:
                    self._missing_grasp_target_since = current_time
                elif (current_time - self._missing_grasp_target_since) >= self.missing_grasp_target_timeout_sec:
                    self._transition(TASK_FAIL_SAFE, "live_follow_target_missing", current_time)
            follow_target = None if live_follow_state is None else live_follow_state.servo_target_position_base
            if follow_target is not None and self._robot_at_position(robot_state, follow_target):
                self._transition(TASK_RECEIVE_OBJECT, "robot_reached_follow_target", current_time)
        elif current_mode == TASK_APPROACH_GRASP_POINT:
            if grasp_target.valid and grasp_target.target_position_base is not None:
                self._missing_grasp_target_since = None
            elif self._latched_grasp_target_position_base is None:
                if self._missing_grasp_target_since is None:
                    self._missing_grasp_target_since = current_time
                elif (current_time - self._missing_grasp_target_since) >= self.missing_grasp_target_timeout_sec:
                    self._transition(TASK_FAIL_SAFE, "grasp_target_missing", current_time)
            elif self._missing_grasp_target_since is None:
                self._missing_grasp_target_since = current_time
            elif (current_time - self._missing_grasp_target_since) >= self.missing_grasp_target_timeout_sec:
                self._missing_grasp_target_since = current_time
            if self._latched_grasp_target_position_base is not None and at_grasp_target:
                self._transition(TASK_RECEIVE_OBJECT, "robot_reached_grasp_target", current_time)
        elif current_mode == TASK_RECEIVE_OBJECT:
            if robot_state.gripper_state == GRIPPER_CLOSE or state_age_sec >= self.receive_close_dwell_sec:
                self._transition(TASK_VERIFY_GRASP, "gripper_closed_for_receive", current_time)
        elif current_mode == TASK_VERIFY_GRASP:
            if grasp_verified:
                self._transition(TASK_MOVE_TO_INITIAL_PLACE, "grasp_verified_force_current", current_time)
            elif state_age_sec >= self.verify_grasp_timeout_sec:
                if self.verify_grasp_force_only_fallback:
                    # Force/current verification timed out but fallback is
                    # enabled: assume the gripper has the object and proceed.
                    self._transition(TASK_MOVE_TO_INITIAL_PLACE, "grasp_verify_timeout_force_fallback", current_time)
                else:
                    self._transition(TASK_FAIL_SAFE, "grasp_verification_timeout", current_time)
        elif current_mode == TASK_MOVE_TO_INITIAL_PLACE:
            if self._placement_position_base is None:
                self._transition(TASK_FAIL_SAFE, "placement_target_missing", current_time)
            elif at_place_target:
                self._transition(TASK_RELEASE, "robot_reached_placement_target", current_time)
        elif current_mode == TASK_RELEASE:
            if state_age_sec >= self.release_dwell_sec or robot_state.gripper_state == GRIPPER_OPEN:
                self._retreat_target_position_base = self._build_retreat_target()
                self._transition(TASK_RETREAT, "release_complete", current_time)
        elif current_mode == TASK_RETREAT:
            if at_retreat_target or state_age_sec >= self.retreat_timeout_sec:
                self._clear_cycle_memory()
                self._transition(TASK_IDLE, "retreat_complete", current_time)
        elif current_mode == TASK_FAIL_SAFE:
            # Auto-recover to IDLE after a dwell so the operator can retry
            # without killing the process.  Cycle memory is cleared first.
            if state_age_sec >= self.fail_safe_auto_recover_sec:
                self._clear_cycle_memory()
                self._transition(TASK_IDLE, "fail_safe_auto_recover", current_time)

        task_state = self._build_task_state(
            merged_object=merged_object,
            selected_hand=selected_hand,
            grasp_target=grasp_target,
            robot_state=robot_state,
            fusion_state=fusion_state,
            live_follow_state=live_follow_state,
            safety_ok=safety_ok,
            current_time=current_time,
        )
        robot_command = self._build_robot_command(task_state=task_state, live_follow_state=live_follow_state)

        self.last_debug = TaskManagerDebug(
            previous_mode=current_mode,
            current_mode=self._mode,
            transition_reason=self._last_transition_reason,
            state_age_sec=max(current_time - self._mode_entered_at, 0.0),
            placement_ready=self._placement_position_base is not None,
            at_target=bool(
                self._robot_at_position(robot_state, task_state.target_position_base)
                if task_state.target_position_base is not None else False
            ),
            fail_safe_triggered=self._mode == TASK_FAIL_SAFE,
            memorized_initial_centroid=self._memorized_initial_centroid,
            retreat_target_position_base=self._retreat_target_position_base,
        )
        return task_state, robot_command

    def _refresh_placement_target(self, merged_object: MergedObjectState) -> None:
        initial_centroid = merged_object.initial_centroid_base
        if not self.use_initial_object_centroid or initial_centroid is None:
            return
        placement = np.asarray(initial_centroid, dtype=np.float32).reshape(3)
        placement[self.height_axis_index] += float(self.release_height_offset_m)
        placement_tuple = tuple(float(value) for value in placement)
        self._memorized_initial_centroid = tuple(float(value) for value in initial_centroid)
        if self._placement_position_base is None:
            self._placement_position_base = placement_tuple

    def _build_task_state(
        self,
        *,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        grasp_target: GraspTargetState,
        robot_state: RobotState,
        fusion_state: FusionState | None,
        live_follow_state: LiveFollowState | None,
        safety_ok: bool,
        current_time: float,
    ) -> TaskState:
        target_position = None
        gripper_cmd = GRIPPER_HOLD

        if self._mode == TASK_FOLLOW_SERVO:
            if live_follow_state is not None:
                target_position = live_follow_state.servo_target_position_base
            gripper_cmd = GRIPPER_OPEN
        elif self._mode == TASK_APPROACH_GRASP_POINT:
            target_position = self._resolve_grasp_target_position(grasp_target)
            gripper_cmd = GRIPPER_OPEN
        elif self._mode == TASK_RECEIVE_OBJECT:
            target_position = self._resolve_grasp_target_position(grasp_target)
            gripper_cmd = GRIPPER_CLOSE
        elif self._mode == TASK_VERIFY_GRASP:
            target_position = self._resolve_grasp_target_position(grasp_target)
            gripper_cmd = GRIPPER_CLOSE
        elif self._mode == TASK_MOVE_TO_INITIAL_PLACE:
            target_position = self._placement_position_base
            gripper_cmd = GRIPPER_CLOSE
        elif self._mode == TASK_RELEASE:
            target_position = self._placement_position_base
            gripper_cmd = GRIPPER_OPEN
        elif self._mode == TASK_RETREAT:
            target_position = self._retreat_target_position_base
            gripper_cmd = GRIPPER_OPEN
        elif self._mode in {TASK_WAIT_FOR_HAND_APPROACH, TASK_WAIT_FOR_OBJECT_LIFT, TASK_ACTIVATE_ROBOT}:
            gripper_cmd = GRIPPER_OPEN
        elif self._mode == TASK_FAIL_SAFE:
            target_position = self._current_robot_position(robot_state)
            gripper_cmd = GRIPPER_HOLD

        fixed_orientation = self._resolve_grasp_fixed_orientation(grasp_target)
        return TaskState(
            mode=self._mode,
            active_reason=self._last_transition_reason,
            target_position_base=target_position,
            fixed_orientation_base=fixed_orientation,
            placement_position_base=self._placement_position_base,
            gripper_cmd=gripper_cmd,
            selected_hand_camera=selected_hand.selected_camera,
            object_lifted=bool(fusion_state.object_lifted) if fusion_state is not None else bool(merged_object.object_lifted),
            hand_approach_detected=bool(
                fusion_state.hand_approach_detected or fusion_state.hand_approach_latched
            ) if fusion_state is not None else False,
            grasp_verified=bool(robot_state.grasp_verified_force_current),
            safety_ok=bool(safety_ok),
            timestamp=current_time,
            valid=True,
        )

    def _build_robot_command(self, task_state: TaskState, live_follow_state: LiveFollowState | None = None) -> RobotCommandState:
        command_type = ROBOT_CMD_HOLD
        stop_requested = False

        if task_state.mode == TASK_FOLLOW_SERVO:
            command_type = ROBOT_CMD_SERVO_TO_POSITION
        elif task_state.mode in {TASK_APPROACH_GRASP_POINT, TASK_MOVE_TO_INITIAL_PLACE, TASK_RETREAT}:
            command_type = ROBOT_CMD_MOVE_TO_POSITION
        elif task_state.mode == TASK_FAIL_SAFE:
            command_type = ROBOT_CMD_STOP
            stop_requested = True

        fixed_orientation_base = task_state.fixed_orientation_base
        if task_state.mode == TASK_FOLLOW_SERVO and live_follow_state is not None and live_follow_state.fixed_orientation_base is not None:
            fixed_orientation_base = live_follow_state.fixed_orientation_base

        return RobotCommandState(
            command_type=command_type,
            target_position_base=task_state.target_position_base,
            fixed_orientation_base=fixed_orientation_base,
            gripper_action=task_state.gripper_cmd,
            source_mode=task_state.mode,
            stop_requested=stop_requested,
            timestamp=task_state.timestamp,
            valid=True,
        )

    def _check_fail_safe(
        self,
        mode: str,
        state_age_sec: float,
        robot_state: RobotState,
        safety_ok: bool,
    ) -> str | None:
        if not safety_ok:
            return "safety_not_ok"
        if robot_state.last_error:
            return "robot_error"
        if mode in {TASK_APPROACH_GRASP_POINT, TASK_FOLLOW_SERVO} and state_age_sec >= self.approach_grasp_timeout_sec:
            return "approach_grasp_timeout"
        if mode == TASK_MOVE_TO_INITIAL_PLACE and state_age_sec >= self.move_to_place_timeout_sec:
            return "move_to_place_timeout"
        return None

    def _transition(self, new_mode: str, reason: str, current_time: float) -> None:
        if self._mode == new_mode:
            self._last_transition_reason = reason
            return
        self._mode = new_mode
        self._mode_entered_at = current_time
        self._last_transition_reason = reason

    def _build_retreat_target(self) -> Tuple[float, float, float] | None:
        reference = self._placement_position_base
        if reference is None:
            return None
        retreat = np.asarray(reference, dtype=np.float32).reshape(3)
        retreat[self.height_axis_index] += float(self.retreat_distance_m)
        return tuple(float(value) for value in retreat)

    def _clear_cycle_memory(self) -> None:
        self._placement_position_base = None
        self._memorized_initial_centroid = None
        self._retreat_target_position_base = None
        self._latched_grasp_target_position_base = None
        self._latched_grasp_fixed_orientation_base = None
        self._missing_grasp_target_since = None

    def _update_latched_grasp_target(self, grasp_target: GraspTargetState) -> None:
        if not grasp_target.valid or grasp_target.target_position_base is None:
            return
        self._latched_grasp_target_position_base = tuple(float(value) for value in grasp_target.target_position_base)
        if grasp_target.fixed_orientation_base is not None:
            self._latched_grasp_fixed_orientation_base = tuple(float(value) for value in grasp_target.fixed_orientation_base)

    def _resolve_grasp_target_position(self, grasp_target: GraspTargetState) -> Tuple[float, float, float] | None:
        if grasp_target.valid and grasp_target.target_position_base is not None:
            return tuple(float(value) for value in grasp_target.target_position_base)
        return self._latched_grasp_target_position_base

    def _resolve_grasp_fixed_orientation(self, grasp_target: GraspTargetState) -> Tuple[float, ...] | None:
        if grasp_target.fixed_orientation_base is not None:
            return tuple(float(value) for value in grasp_target.fixed_orientation_base)
        if self._latched_grasp_fixed_orientation_base is not None:
            return self._latched_grasp_fixed_orientation_base
        return self.default_fixed_orientation_base

    def _robot_at_position(
        self,
        robot_state: RobotState,
        target_position_base: Tuple[float, float, float] | None,
    ) -> bool:
        if target_position_base is None:
            return False
        current_position = self._current_robot_position(robot_state)
        if current_position is None:
            return False
        return float(np.linalg.norm(np.asarray(current_position) - np.asarray(target_position_base))) <= self.position_tolerance_m

    @staticmethod
    def _current_robot_position(robot_state: RobotState) -> Tuple[float, float, float] | None:
        if robot_state.actual_tcp_pose_base is None:
            return None
        pose = tuple(float(value) for value in robot_state.actual_tcp_pose_base)
        return pose[:3]


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HEIGHT_AXIS_TO_INDEX",
    "TaskManagerDebug",
    "TaskManager",
]
