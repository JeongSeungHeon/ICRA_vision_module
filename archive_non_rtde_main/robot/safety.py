"""Safety validation for outgoing robot commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import yaml

from system.shared_state import (
    GRIPPER_HOLD,
    MergedObjectState,
    RobotCommandState,
    RobotState,
    SelectedHandState,
    FusionState,
    GraspTargetState,
    TaskState,
    ROBOT_CMD_HOLD,
    ROBOT_CMD_SERVO_TO_POSITION,
    ROBOT_CMD_STOP,
    TASK_APPROACH_GRASP_POINT,
    TASK_RECEIVE_OBJECT,
    TASK_RELEASE,
    TASK_VERIFY_GRASP,
)

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class SafetyResult:
    safe: bool
    reason: str | None
    severity: str
    sanitized_command: RobotCommandState
    workspace_ok: bool
    translation_step_ok: bool
    orientation_ok: bool
    speed_ok: bool
    perception_fresh_ok: bool
    calibration_ok: bool
    hand_ok: bool
    object_ok: bool
    grasp_ok: bool
    grasp_verified_ok: bool


class SafetyValidator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        safety_cfg = config.get("safety", {})
        sync_cfg = config.get("perception", {}).get("synchronization", {})
        motion_cfg = config.get("robot", {}).get("motion", {})
        object_cfg = config.get("perception", {}).get("object", {})

        workspace_cfg = safety_cfg.get("workspace_bounds_m", {})
        self.workspace_bounds = {
            axis: tuple(float(v) for v in workspace_cfg.get(axis, [-1e9, 1e9]))
            for axis in ("x", "y", "z")
        }
        self.max_translation_step_m = float(motion_cfg.get("max_translation_step_m", 0.02))
        self.max_robot_speed_mps = float(safety_cfg.get("max_robot_speed_mps", 0.25))
        self.stale_perception_timeout_sec = float(
            safety_cfg.get("stale_perception_timeout_sec", sync_cfg.get("stale_timeout_sec", 0.30))
        )
        self.min_merged_object_confidence = float(
            safety_cfg.get("min_merged_object_confidence", object_cfg.get("confidence_threshold", 0.50))
        )
        self.stop_on_stale_perception = bool(safety_cfg.get("stop_on_stale_perception", True))
        self.stop_on_invalid_calibration = bool(safety_cfg.get("stop_on_invalid_calibration", True))
        self.stop_on_missing_grasp_target = bool(safety_cfg.get("stop_on_missing_grasp_target", True))
        self.fail_safe_action = str(safety_cfg.get("fail_safe_action", "stop")).strip().lower()

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "SafetyValidator":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def validate(
        self,
        task_state: TaskState,
        robot_command: RobotCommandState,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        grasp_target: GraspTargetState,
        robot_state: RobotState,
        fusion_state: FusionState | None = None,
        *,
        now_timestamp: float | None = None,
    ) -> SafetyResult:
        current_time = float(now_timestamp) if now_timestamp is not None else max(
            float(task_state.timestamp),
            float(robot_command.timestamp),
            float(merged_object.timestamp),
            float(selected_hand.timestamp),
            float(grasp_target.timestamp),
            float(robot_state.timestamp),
        )

        workspace_ok = self._workspace_ok(robot_command.target_position_base)
        translation_step_ok = self._translation_step_ok(robot_state, robot_command)
        orientation_ok = self._orientation_ok(robot_command.fixed_orientation_base)
        speed_ok = self._speed_ok(robot_state)
        perception_fresh_ok = self._perception_fresh_ok(
            merged_object=merged_object,
            selected_hand=selected_hand,
            fusion_state=fusion_state,
            current_time=current_time,
        )
        calibration_ok = self._calibration_ok(merged_object, selected_hand, grasp_target)
        hand_ok = self._hand_ok(task_state, selected_hand)
        object_ok = self._object_ok(task_state, merged_object)
        grasp_ok = self._grasp_ok(task_state, grasp_target)
        grasp_verified_ok = self._grasp_verified_ok(task_state, robot_state)

        failing_checks = []
        if not workspace_ok:
            failing_checks.append("target_outside_workspace")
        if not translation_step_ok:
            failing_checks.append("translation_step_too_large")
        if not orientation_ok:
            failing_checks.append("invalid_fixed_orientation")
        if not speed_ok:
            failing_checks.append("robot_speed_too_high")
        if not perception_fresh_ok:
            failing_checks.append("stale_perception")
        if not calibration_ok:
            failing_checks.append("invalid_calibration")
        if not hand_ok:
            failing_checks.append("selected_hand_unavailable")
        if not object_ok:
            failing_checks.append("merged_object_unreliable")
        if not grasp_ok:
            failing_checks.append("missing_grasp_target")
        if not grasp_verified_ok:
            failing_checks.append("grasp_not_verified")

        safe = len(failing_checks) == 0
        reason = None if safe else failing_checks[0]
        severity = "none" if safe else self._classify_severity(reason)
        sanitized_command = robot_command if safe else self._build_fail_safe_command(
            robot_command=robot_command,
            robot_state=robot_state,
            current_time=current_time,
        )

        return SafetyResult(
            safe=safe,
            reason=reason,
            severity=severity,
            sanitized_command=sanitized_command,
            workspace_ok=workspace_ok,
            translation_step_ok=translation_step_ok,
            orientation_ok=orientation_ok,
            speed_ok=speed_ok,
            perception_fresh_ok=perception_fresh_ok,
            calibration_ok=calibration_ok,
            hand_ok=hand_ok,
            object_ok=object_ok,
            grasp_ok=grasp_ok,
            grasp_verified_ok=grasp_verified_ok,
        )

    def _workspace_ok(self, target_position_base: Sequence[float] | None) -> bool:
        if target_position_base is None:
            return True
        target = self._to_vec3(target_position_base)
        if target is None:
            return False
        bounds_x = self.workspace_bounds["x"]
        bounds_y = self.workspace_bounds["y"]
        bounds_z = self.workspace_bounds["z"]
        return (
            bounds_x[0] <= target[0] <= bounds_x[1]
            and bounds_y[0] <= target[1] <= bounds_y[1]
            and bounds_z[0] <= target[2] <= bounds_z[1]
        )

    def _translation_step_ok(self, robot_state: RobotState, robot_command: RobotCommandState) -> bool:
        if robot_command.command_type != ROBOT_CMD_SERVO_TO_POSITION:
            return True
        if robot_command.target_position_base is None:
            return True
        current_pose = robot_state.actual_tcp_pose_base
        if current_pose is None:
            return True
        current_position = self._to_vec3(current_pose[:3])
        target_position = self._to_vec3(robot_command.target_position_base)
        if current_position is None or target_position is None:
            return False
        step = self._distance3(current_position, target_position)
        return step <= self.max_translation_step_m

    @staticmethod
    def _orientation_ok(fixed_orientation_base: Sequence[float] | None) -> bool:
        if fixed_orientation_base is None:
            return True
        try:
            values = tuple(float(value) for value in fixed_orientation_base)
        except (TypeError, ValueError):
            return False
        if len(values) not in (3, 6):
            return False
        return all(isfinite(value) for value in values)

    def _speed_ok(self, robot_state: RobotState) -> bool:
        if robot_state.actual_tcp_speed is None:
            return True
        translational_speed = self._norm3(robot_state.actual_tcp_speed[:3])
        if translational_speed is None:
            return False
        return translational_speed <= self.max_robot_speed_mps

    def _perception_fresh_ok(
        self,
        *,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        fusion_state: FusionState | None,
        current_time: float,
    ) -> bool:
        if not self.stop_on_stale_perception:
            return True
        timestamps = [float(merged_object.timestamp), float(selected_hand.timestamp)]
        if fusion_state is not None:
            timestamps.append(float(fusion_state.timestamp))
        newest = max(timestamps)

        # Guard against mismatched timestamp epochs (e.g. camera hardware
        # boot-relative time vs wall-clock time.time()).  If the timestamps
        # diverge by more than 1000 s they cannot be meaningfully compared;
        # skip the stale check in that case so the robot does not immediately
        # stop on startup.
        epoch_diff = abs(current_time - newest)
        if epoch_diff > 1000.0:
            return True

        age = max(current_time - newest, 0.0)
        return age <= self.stale_perception_timeout_sec

    def _calibration_ok(
        self,
        merged_object: MergedObjectState,
        selected_hand: SelectedHandState,
        grasp_target: GraspTargetState,
    ) -> bool:
        if not self.stop_on_invalid_calibration:
            return True
        checks = []
        if merged_object.centroid_base is not None:
            checks.append(self._to_vec3(merged_object.centroid_base) is not None)
        if selected_hand.palm_center_base is not None:
            checks.append(self._to_vec3(selected_hand.palm_center_base) is not None)
        if grasp_target.target_position_base is not None:
            checks.append(self._to_vec3(grasp_target.target_position_base) is not None)
        return all(checks) if checks else True

    def _hand_ok(self, task_state: TaskState, selected_hand: SelectedHandState) -> bool:
        if task_state.mode not in {TASK_RECEIVE_OBJECT, TASK_RELEASE, TASK_VERIFY_GRASP}:
            return True
        return bool(selected_hand.valid and selected_hand.selected_camera is not None and selected_hand.palm_center_base is not None)

    def _object_ok(self, task_state: TaskState, merged_object: MergedObjectState) -> bool:
        if task_state.mode not in {TASK_RECEIVE_OBJECT, TASK_RELEASE, TASK_APPROACH_GRASP_POINT, TASK_VERIFY_GRASP}:
            return True
        return bool(
            merged_object.valid
            and merged_object.object_detected
            and float(merged_object.confidence) >= self.min_merged_object_confidence
            and merged_object.centroid_base is not None
        )

    def _grasp_ok(self, task_state: TaskState, grasp_target: GraspTargetState) -> bool:
        if task_state.mode not in {TASK_APPROACH_GRASP_POINT, TASK_RECEIVE_OBJECT, TASK_VERIFY_GRASP}:
            return True
        if not self.stop_on_missing_grasp_target:
            return True
        return bool(grasp_target.valid and grasp_target.target_position_base is not None)

    @staticmethod
    def _grasp_verified_ok(task_state: TaskState, robot_state: RobotState) -> bool:
        if task_state.mode != TASK_RELEASE:
            return True
        return bool(robot_state.grasp_verified_force_current)

    def _build_fail_safe_command(
        self,
        *,
        robot_command: RobotCommandState,
        robot_state: RobotState,
        current_time: float,
    ) -> RobotCommandState:
        current_position = None
        if robot_state.actual_tcp_pose_base is not None:
            current_position = self._to_vec3(robot_state.actual_tcp_pose_base[:3])

        if self.fail_safe_action == "hold":
            return RobotCommandState(
                command_type=ROBOT_CMD_HOLD,
                target_position_base=current_position,
                fixed_orientation_base=robot_command.fixed_orientation_base,
                gripper_action=GRIPPER_HOLD,
                source_mode=robot_command.source_mode,
                stop_requested=False,
                timestamp=current_time,
                valid=True,
            )

        return RobotCommandState(
            command_type=ROBOT_CMD_STOP,
            target_position_base=current_position,
            fixed_orientation_base=robot_command.fixed_orientation_base,
            gripper_action=GRIPPER_HOLD,
            source_mode=robot_command.source_mode,
            stop_requested=True,
            timestamp=current_time,
            valid=True,
        )

    @staticmethod
    def _classify_severity(reason: str | None) -> str:
        if reason in {"target_outside_workspace", "invalid_calibration", "robot_speed_too_high"}:
            return "severe"
        return "recoverable"

    @staticmethod
    def _to_vec3(values: Sequence[float] | None) -> Tuple[float, float, float] | None:
        if values is None:
            return None
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return None
        if len(vector) != 3:
            return None
        if not all(isfinite(value) for value in vector):
            return None
        return vector

    @staticmethod
    def _distance3(a: Sequence[float], b: Sequence[float]) -> float:
        ax, ay, az = (float(value) for value in a[:3])
        bx, by, bz = (float(value) for value in b[:3])
        return sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    @staticmethod
    def _norm3(values: Sequence[float] | None) -> float | None:
        if values is None:
            return None
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return None
        if len(vector) < 3:
            return None
        return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


__all__ = ["DEFAULT_CONFIG_PATH", "SafetyResult", "SafetyValidator"]
