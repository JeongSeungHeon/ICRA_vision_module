"""RTDE-based UR controller with mock fallback and watchdog support."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import threading
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml

from system.shared_state import (
    GRIPPER_CLOSE,
    GRIPPER_HOLD,
    GRIPPER_OPEN,
    ROBOT_CMD_HOLD,
    ROBOT_CMD_MOVE_TO_POSITION,
    ROBOT_CMD_SERVO_TO_POSITION,
    ROBOT_CMD_STOP,
    RobotCommandState,
    RobotState,
)

try:  # pragma: no cover - optional runtime dependency
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:  # pragma: no cover - optional runtime dependency
    RTDEControlInterface = None
    RTDEReceiveInterface = None

try:  # pragma: no cover - optional runtime dependency
    from rtde_io import RTDEIOInterface
except ImportError:  # pragma: no cover - optional runtime dependency
    RTDEIOInterface = None

from robot.robotiq_gripper_controller import RobotiqGripperController

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class RtdeControllerDebug:
    using_mock: bool
    last_command_type: str | None
    last_target_pose_base: Tuple[float, float, float, float, float, float] | None
    watchdog_triggered: bool
    gripper_output_applied: bool
    grasp_force_ok: bool
    grasp_current_ok: bool
    last_error: str | None


class RtdeController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        robot_cfg = config.get("robot", {})
        rtde_cfg = robot_cfg.get("rtde", {})
        motion_cfg = robot_cfg.get("motion", {})
        gripper_cfg = robot_cfg.get("gripper", {})
        grasp_verify_cfg = robot_cfg.get("grasp_verification", {})
        grasp_cfg = config.get("grasp", {})
        frame_map_cfg = robot_cfg.get("frame_mapping", {})

        self.robot_ip = str(rtde_cfg.get("robot_ip", "192.168.0.2"))
        self.dashboard_port = int(rtde_cfg.get("dashboard_port", 29999))
        self.control_hz = float(rtde_cfg.get("control_hz", 125.0))
        self.watchdog_timeout_sec = float(rtde_cfg.get("watchdog_timeout_sec", 0.20))
        self.use_mock_fallback = bool(rtde_cfg.get("use_mock_fallback", True))
        self.force_mock = bool(rtde_cfg.get("force_mock", False))
        self.move_asynchronous = bool(rtde_cfg.get("move_asynchronous", True))
        self.move_acceleration = float(rtde_cfg.get("move_acceleration", 0.20))
        self.servo_velocity = float(rtde_cfg.get("servo_velocity", motion_cfg.get("approach_speed_mps", 0.05)))
        self.servo_acceleration = float(rtde_cfg.get("servo_acceleration", 0.20))
        self.servo_lookahead_time = float(rtde_cfg.get("servo_lookahead_time", 0.10))
        self.servo_gain = float(rtde_cfg.get("servo_gain", 300.0))
        self.stop_acceleration = float(rtde_cfg.get("stop_acceleration", 1.0))
        self.move_speed_mps = float(motion_cfg.get("approach_speed_mps", 0.05))
        self.retreat_speed_mps = float(motion_cfg.get("retreat_speed_mps", 0.08))

        self.open_command = str(gripper_cfg.get("open_command", GRIPPER_OPEN))
        self.close_command = str(gripper_cfg.get("close_command", GRIPPER_CLOSE))
        self.gripper_control_mode = str(gripper_cfg.get("control_mode", "socket")).strip().lower()
        self.gripper_output_domain = str(gripper_cfg.get("output_domain", "standard"))
        self.open_digital_output = gripper_cfg.get("open_digital_output")
        self.close_digital_output = gripper_cfg.get("close_digital_output")
        self.output_high_when_active = bool(gripper_cfg.get("output_high_when_active", True))
        self.gripper_debug = bool(gripper_cfg.get("debug", True))
        self.gripper_socket_port = int(gripper_cfg.get("socket_port", 63352))
        self.gripper_socket_timeout_sec = float(gripper_cfg.get("socket_timeout_sec", 2.0))
        self.gripper_socket_settle_time = float(gripper_cfg.get("socket_settle_time", 0.15))
        self.gripper_activate_on_connect = bool(gripper_cfg.get("activate_on_connect", True))
        self.gripper_open_speed = int(gripper_cfg.get("open_speed", 255))
        self.gripper_open_force = int(gripper_cfg.get("open_force", 255))
        self.gripper_close_speed = int(gripper_cfg.get("close_speed", 255))
        self.gripper_close_force = int(gripper_cfg.get("close_force", 255))
        self.gripper_position_complete_threshold = int(gripper_cfg.get("position_complete_threshold", 200))

        self.fixed_orientation_format = str(grasp_cfg.get("fixed_orientation_format", "rotvec")).strip().lower()

        self.use_tcp_force = bool(grasp_verify_cfg.get("use_tcp_force", True))
        self.use_joint_current = bool(grasp_verify_cfg.get("use_joint_current", True))
        self.min_tcp_force_norm_n = float(grasp_verify_cfg.get("min_tcp_force_norm_n", 8.0))
        self.min_mean_joint_current_a = float(grasp_verify_cfg.get("min_mean_joint_current_a", 0.15))
        self.require_both_signals = bool(grasp_verify_cfg.get("require_both_signals", False))
        self.frame_mapping_enabled = bool(frame_map_cfg.get("enabled", True))
        self.position_signs = self._normalize_position_signs(frame_map_cfg.get("position_signs", [-1.0, -1.0, 1.0]))

        # Home pose ---------------------------------------------------------------
        home_cfg = config.get("home_pose", {})
        self.home_pose_enabled = bool(home_cfg.get("enabled", False))
        _home_pos_raw = home_cfg.get("position_m", None)
        self.home_pose_position_m: Optional[Tuple[float, float, float]] = (
            tuple(float(v) for v in _home_pos_raw[:3])
            if _home_pos_raw is not None and len(_home_pos_raw) >= 3
            else None
        )
        self.home_pose_speed_mps = float(home_cfg.get("move_speed_mps", 0.05))
        self.home_pose_acceleration = float(home_cfg.get("move_acceleration", 0.10))
        # Orientation for home move reuses the grasp RPY.
        self.home_pose_orientation: Tuple[float, ...] = tuple(
            float(v) for v in grasp_cfg.get("fixed_orientation_base", [0.0, 0.0, 0.0])
        )

        self._rtde_control = None
        self._rtde_receive = None
        self._rtde_io = None
        self._robotiq_gripper = None
        self._using_mock = False
        self._rtde_lock = threading.RLock()
        self._connected = False
        self._last_error = None
        self._connection_notice = None
        self._last_command_timestamp = 0.0
        self._last_command_type = None
        self._last_safe_pose_base = None
        self._last_gripper_state = None

        # Async moveL tracking ---------------------------------------------------
        # When move_asynchronous=True the UR controller keeps executing a moveL
        # motion across many step() calls.  Re-issuing moveL while the robot is
        # still moving causes the dreaded
        # "another thread is already controlling the robot" RuntimeError.
        # We track the pending target and only re-issue when:
        #   a) the target has changed by more than _MOVE_REISSUE_THRESHOLD_M, or
        #   b) the robot has already reached the previous target (isSteady).
        self._pending_move_target: Optional[Tuple[float, ...]] = None
        self._move_in_progress: bool = False
        # Minimum positional change (metres) to re-issue a new moveL.
        self._MOVE_REISSUE_THRESHOLD_M: float = 0.005

        self._mock_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._mock_speed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._mock_force = (0.0, 0.0, 0.0)
        self._mock_joint_positions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._mock_joint_currents = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.last_debug = RtdeControllerDebug(
            using_mock=False,
            last_command_type=None,
            last_target_pose_base=None,
            watchdog_triggered=False,
            gripper_output_applied=False,
            grasp_force_ok=False,
            grasp_current_ok=False,
            last_error=None,
        )

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "RtdeController":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    @property
    def is_connected(self) -> bool:
        return bool(self._connected)

    @property
    def using_mock(self) -> bool:
        return bool(self._using_mock)

    def connect(self) -> "RtdeController":
        if self._connected:
            return self

        if self.force_mock:
            self._enter_mock_mode("force_mock_enabled")
            return self

        if RTDEControlInterface is None or RTDEReceiveInterface is None:
            if not self.use_mock_fallback:
                raise RuntimeError("ur_rtde is not installed and mock fallback is disabled.")
            self._enter_mock_mode("ur_rtde_not_available")
            return self

        try:
            self._last_error = None
            self._rtde_control = RTDEControlInterface(self.robot_ip)
            self._rtde_receive = RTDEReceiveInterface(self.robot_ip)
            if RTDEIOInterface is not None:
                try:
                    self._rtde_io = RTDEIOInterface(self.robot_ip)
                except Exception as exc:
                    self._rtde_io = None
                    self._last_error = f"Failed to create RTDEIOInterface: {exc}"
                    if self.gripper_debug:
                        print(f"[rtde_controller] {self._last_error}", flush=True)
            else:
                self._rtde_io = None
                if self.gripper_debug:
                    print("[rtde_controller] RTDEIOInterface is unavailable; gripper digital outputs cannot be driven.", flush=True)

            if self.gripper_control_mode in {"socket", "robotiq", "robotiq_socket", "daemon"}:
                try:
                    self._robotiq_gripper = RobotiqGripperController(
                        self.robot_ip,
                        port=self.gripper_socket_port,
                        settle_time=self.gripper_socket_settle_time,
                        socket_timeout=self.gripper_socket_timeout_sec,
                        verbose=self.gripper_debug,
                        activate_on_connect=self.gripper_activate_on_connect,
                    )
                    self._robotiq_gripper.connect()
                except Exception as exc:
                    self._robotiq_gripper = None
                    self._last_error = f"Failed to connect Robotiq gripper socket controller: {exc}"
                    if self.gripper_debug:
                        print(f"[rtde_controller] {self._last_error}", flush=True)

            self._connected = True
            self._using_mock = False
            self._connection_notice = None
        except Exception as exc:  # pragma: no cover - hardware dependent
            if not self.use_mock_fallback:
                raise
            self._enter_mock_mode(str(exc))
        return self

    def disconnect(self) -> None:
        if self._using_mock:
            self._connected = False
            return

        try:
            if self._rtde_control is not None:
                self._safe_stop_motion()
                stop_script = getattr(self._rtde_control, "stopScript", None)
                if callable(stop_script):
                    stop_script()
        finally:
            if self._robotiq_gripper is not None:
                try:
                    self._robotiq_gripper.disconnect()
                except Exception:
                    pass
            self._rtde_control = None
            self._rtde_receive = None
            self._rtde_io = None
            self._robotiq_gripper = None
            self._connected = False

    def close(self) -> None:
        self.disconnect()

    def open_gripper_blocking(self) -> bool:
        with self._rtde_lock:
            self.connect()
            if self._using_mock:
                self._last_gripper_state = GRIPPER_OPEN
                return False
            if self.gripper_control_mode in {"socket", "robotiq", "robotiq_socket", "daemon"}:
                if self._robotiq_gripper is None:
                    return False
                self._robotiq_gripper.open(
                    speed=self.gripper_open_speed,
                    force=self.gripper_open_force,
                    wait=True,
                )
                self._last_gripper_state = GRIPPER_OPEN
                return True
            return self._apply_gripper_action(GRIPPER_OPEN)

    def start_gripper_close(self) -> bool:
        with self._rtde_lock:
            self.connect()
            if self._using_mock:
                self._last_gripper_state = GRIPPER_CLOSE
                return False
            if self.gripper_control_mode in {"socket", "robotiq", "robotiq_socket", "daemon"}:
                if self._robotiq_gripper is None:
                    return False
                self._robotiq_gripper.close(
                    speed=self.gripper_close_speed,
                    force=self.gripper_close_force,
                    wait=False,
                )
                self._last_gripper_state = GRIPPER_CLOSE
                return True
            return self._apply_gripper_action(GRIPPER_CLOSE)

    def stop_gripper_motion(self) -> bool:
        with self._rtde_lock:
            self.connect()
            if self._using_mock:
                return False
            if self.gripper_control_mode in {"socket", "robotiq", "robotiq_socket", "daemon"}:
                if self._robotiq_gripper is None:
                    return False
                return bool(self._robotiq_gripper.stop())
            return False

    def get_gripper_close_state(self) -> dict[str, object] | None:
        with self._rtde_lock:
            self.connect()
            if self._using_mock:
                return None
            if self.gripper_control_mode in {"socket", "robotiq", "robotiq_socket", "daemon"}:
                if self._robotiq_gripper is None:
                    return None
                return dict(self._robotiq_gripper.get_motion_state())
            return None

    def get_gripper_diagnostic_state(self) -> dict[str, object] | None:
        with self._rtde_lock:
            self.connect()
            if self._using_mock:
                return {
                    "mode": "mock",
                    "last_gripper_state": self._last_gripper_state,
                }
            if self.gripper_control_mode in {"socket", "robotiq", "robotiq_socket", "daemon"}:
                if self._robotiq_gripper is None:
                    return {
                        "mode": self.gripper_control_mode,
                        "connected": False,
                        "error": "Robotiq socket controller is not connected",
                    }
                state = dict(self._robotiq_gripper.get_diagnostic_state())
                state.update(
                    {
                        "mode": self.gripper_control_mode,
                        "connected": bool(self._robotiq_gripper.is_connected),
                        "last_gripper_state": self._last_gripper_state,
                        "open_speed": int(self.gripper_open_speed),
                        "open_force": int(self.gripper_open_force),
                        "close_speed": int(self.gripper_close_speed),
                        "close_force": int(self.gripper_close_force),
                    }
                )
                return state
            return {
                "mode": self.gripper_control_mode,
                "last_gripper_state": self._last_gripper_state,
                "error": "Detailed diagnostics are only available for Robotiq socket mode",
            }

    # ------------------------------------------------------------------
    # Home pose
    # ------------------------------------------------------------------

    def move_home(self) -> bool:
        """Move the robot to the configured home pose using a blocking moveL.

        The home pose position is specified in the UR RTDE robot-base frame
        (the same coordinates shown on the teach pendant), so frame_mapping
        position_signs are NOT applied here.  Only the orientation is converted
        from RPY to rotvec as required by the RTDE API.

        Blocks until the motion completes.  Returns True on success, False if
        home pose is disabled or not configured.
        """
        if not self.home_pose_enabled or self.home_pose_position_m is None:
            return False

        # Resolve orientation (RPY → rotvec for RTDE).
        orientation = self._resolve_home_orientation()
        if orientation is None:
            return False

        # Use RTDE-frame coordinates directly — no position_signs flip.
        home_xyz = tuple(float(v) for v in self.home_pose_position_m[:3])
        target = list(home_xyz + orientation)

        print(
            "[rtde_controller] Moving to home pose (RTDE frame): "
            "xyz=(%.3f, %.3f, %.3f)  rotvec=(%.3f, %.3f, %.3f)"
            % (*home_xyz, *orientation),
            flush=True,
        )

        if self._using_mock:
            self._mock_pose = tuple(target)
            print("[rtde_controller] Home pose reached (mock).", flush=True)
            return True

        self.connect()
        move_l = getattr(self._rtde_control, "moveL", None)
        if not callable(move_l):
            return False

        try:
            # Blocking move (async=False) — runner.start() waits for completion.
            move_l(target, self.home_pose_speed_mps, self.home_pose_acceleration, False)
            self._pending_move_target = None
            self._move_in_progress = False
            print("[rtde_controller] Home pose reached.", flush=True)
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[rtde_controller] move_home failed: {exc}", flush=True)
            return False

    def move_to_joint_positions(
        self,
        joints_rad: Sequence[float],
        *,
        speed_rad_s: float = 0.5,
        acceleration_rad_s2: float = 0.5,
        async_move: bool = True,
    ) -> bool:
        """Move the robot to an absolute joint target using RTDE moveJ."""
        target = self._to_joint_tuple(joints_rad)
        if target is None:
            return False

        with self._rtde_lock:
            self.connect()

            self._move_in_progress = False
            self._pending_move_target = None

            if self._using_mock:
                self._mock_joint_positions = target
                print(
                    "[rtde_controller] Joint target reached (mock): "
                    "q=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)"
                    % target,
                    flush=True,
                )
                return True

            move_j = getattr(self._rtde_control, "moveJ", None)
            if not callable(move_j):
                self._last_error = "RTDE control interface does not expose moveJ"
                return False

            try:
                move_j(list(target), float(speed_rad_s), float(acceleration_rad_s2), bool(async_move))
                self._last_error = None
                return True
            except TypeError:
                try:
                    move_j(list(target), float(speed_rad_s), float(acceleration_rad_s2))
                    self._last_error = None
                    return True
                except Exception as exc:
                    self._last_error = str(exc)
                    print(f"[rtde_controller] move_to_joint_positions failed: {exc}", flush=True)
                    return False
            except Exception as exc:
                self._last_error = str(exc)
                print(f"[rtde_controller] move_to_joint_positions failed: {exc}", flush=True)
                return False

    def stop_joint_motion(self) -> None:
        """Stop an active joint move when supported by the RTDE interface."""
        with self._rtde_lock:
            if self._using_mock:
                return
            stop_j = getattr(self._rtde_control, "stopJ", None)
            if callable(stop_j):
                try:
                    stop_j(self.stop_acceleration)
                    return
                except TypeError:
                    try:
                        stop_j()
                        return
                    except Exception:
                        pass
                except Exception:
                    pass
            self._safe_stop_motion()

    def _resolve_home_orientation(self) -> Optional[Tuple[float, float, float]]:
        """Convert home_pose_orientation (RPY or rotvec) to RTDE rotvec 3-tuple."""
        if self.fixed_orientation_format == "rpy":
            return self._rpy_to_rotvec(self.home_pose_orientation[:3])
        return tuple(float(v) for v in self.home_pose_orientation[:3])

    def step(
        self,
        robot_command: RobotCommandState,
        *,
        now_timestamp: Optional[float] = None,
        loop_dt: Optional[float] = None,
    ) -> RobotState:
        import time as _time
        with self._rtde_lock:
            self.connect()
            current_time = float(now_timestamp) if now_timestamp is not None else _time.time()
            # Resolve the actual loop period: prefer the measured loop_dt from
            # the caller (runner), fall back to the configured control rate.
            effective_loop_dt = float(loop_dt) if loop_dt is not None and loop_dt > 0.0 else (1.0 / max(self.control_hz, 1.0))

            watchdog_triggered = self._is_command_stale(robot_command, current_time)
            gripper_output_applied = False
            if watchdog_triggered:
                self._safe_stop_motion()
            else:
                gripper_output_applied = self._apply_gripper_action(robot_command.gripper_action)
                self._execute_motion_command(robot_command, effective_loop_dt)
                self._last_command_timestamp = current_time
                self._last_command_type = robot_command.command_type

            robot_state = self.read_robot_state(now_timestamp=current_time)
            force_ok, current_ok, grasp_verified = self._compute_grasp_flags(robot_state)
            robot_state.grasp_verified_force_current = grasp_verified
            robot_state.timestamp = current_time
            robot_state.valid = True

            self.last_debug = RtdeControllerDebug(
                using_mock=self._using_mock,
                last_command_type=robot_command.command_type,
                last_target_pose_base=self._compose_target_pose(
                    robot_command.target_position_base,
                    robot_command.fixed_orientation_base,
                ) if robot_command.target_position_base is not None else None,
                watchdog_triggered=watchdog_triggered,
                gripper_output_applied=gripper_output_applied,
                grasp_force_ok=force_ok,
                grasp_current_ok=current_ok,
                last_error=self._connection_notice or self._last_error,
            )
            return robot_state

    def read_robot_state(self, *, now_timestamp: Optional[float] = None) -> RobotState:
        with self._rtde_lock:
            self.connect()
            current_time = float(now_timestamp) if now_timestamp is not None else 0.0

            if self._using_mock:
                tcp_force_norm = self._norm3(self._mock_force)
                mean_joint_current = self._mean_abs(self._mock_joint_currents)
                force_ok, current_ok, grasp_verified = self._compute_grasp_flags_from_values(
                    tcp_force_norm_n=tcp_force_norm,
                    mean_joint_current_a=mean_joint_current,
                )
                return RobotState(
                    is_connected=self._connected,
                    robot_mode="mock",
                    actual_tcp_pose_base=self._map_incoming_pose6(self._mock_pose),
                    actual_tcp_speed=self._map_incoming_pose6(self._mock_speed),
                    actual_tcp_force_base=self._map_incoming_vec3(self._mock_force),
                    tcp_force_norm_n=tcp_force_norm,
                    joint_positions=self._mock_joint_positions,
                    joint_currents_a=self._mock_joint_currents,
                    mean_joint_current_a=mean_joint_current,
                    gripper_state=self._last_gripper_state,
                    grasp_verified_force_current=grasp_verified,
                    last_error=None,
                    timestamp=current_time,
                    valid=True,
                )

            tcp_pose = self._map_incoming_pose6(
                self._to_pose6(self._call_first_available(self._rtde_receive, ["getActualTCPPose"]))
            )
            tcp_speed = self._map_incoming_pose6(
                self._to_pose6(self._call_first_available(self._rtde_receive, ["getActualTCPSpeed"]))
            )
            tcp_force_wrench = self._to_float_sequence(
                self._call_first_available(self._rtde_receive, ["getActualTCPForce"]),
                expected_length=None,
            )
            joint_currents = self._to_float_sequence(
                self._call_first_available(
                    self._rtde_receive,
                    ["getActualCurrent", "getJointCurrents", "getActualJointCurrents"],
                ),
                expected_length=None,
            )
            actual_q = self._to_float_sequence(
                self._call_first_available(self._rtde_receive, ["getActualQ"]),
                expected_length=None,
            )

            tcp_force_base = None
            tcp_force_norm = None
            if tcp_force_wrench:
                force_xyz = tuple(float(value) for value in tcp_force_wrench[:3])
                tcp_force_base = self._map_incoming_vec3(force_xyz)
                tcp_force_norm = self._norm3(force_xyz)

            mean_joint_current = self._mean_abs(joint_currents)
            force_ok, current_ok, grasp_verified = self._compute_grasp_flags_from_values(
                tcp_force_norm_n=tcp_force_norm,
                mean_joint_current_a=mean_joint_current,
            )
            return RobotState(
                is_connected=self._connected,
                robot_mode="rtde",
                actual_tcp_pose_base=tcp_pose,
                actual_tcp_speed=tcp_speed,
                actual_tcp_force_base=tcp_force_base,
                tcp_force_norm_n=tcp_force_norm,
                joint_positions=self._to_joint_tuple(actual_q),
                joint_currents_a=self._to_joint_tuple(joint_currents),
                mean_joint_current_a=mean_joint_current,
                gripper_state=self._last_gripper_state,
                grasp_verified_force_current=grasp_verified,
                last_error=self._last_error,
                timestamp=current_time,
                valid=True,
            )

    def set_mock_feedback(
        self,
        *,
        tcp_force_vector: Optional[Sequence[float]] = None,
        joint_currents_a: Optional[Sequence[float]] = None,
    ) -> None:
        if tcp_force_vector is not None:
            self._mock_force = self._to_vec3(tcp_force_vector) or self._mock_force
        if joint_currents_a is not None:
            self._mock_joint_currents = self._to_joint_tuple(joint_currents_a) or self._mock_joint_currents

    def _enter_mock_mode(self, reason: str) -> None:
        self._rtde_control = None
        self._rtde_receive = None
        self._rtde_io = None
        self._using_mock = True
        self._connected = True
        self._connection_notice = reason
        self._last_error = None

    def _is_command_stale(self, robot_command: RobotCommandState, current_time: float) -> bool:
        if not robot_command.valid:
            return True
        age = max(current_time - float(robot_command.timestamp), 0.0)
        return age > self.watchdog_timeout_sec

    def _execute_motion_command(self, robot_command: RobotCommandState, loop_dt: float = 1.0 / 30.0) -> None:
        if robot_command.stop_requested or robot_command.command_type == ROBOT_CMD_STOP:
            self._safe_stop_motion()
            return

        if robot_command.command_type == ROBOT_CMD_HOLD:
            if self._using_mock:
                self._mock_speed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        target_pose = self._compose_target_pose(
            robot_command.target_position_base,
            robot_command.fixed_orientation_base,
        )
        if target_pose is None:
            return
        self._last_safe_pose_base = target_pose

        if self._using_mock:
            self._mock_pose = target_pose
            self._mock_speed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        if robot_command.command_type == ROBOT_CMD_MOVE_TO_POSITION:
            self._move_to_pose(target_pose)
        elif robot_command.command_type == ROBOT_CMD_SERVO_TO_POSITION:
            self._servo_to_pose(target_pose, loop_dt=loop_dt)

    def _move_to_pose(self, target_pose: Tuple[float, float, float, float, float, float]) -> None:
        """Issue an async moveL to target_pose, guarding against the UR
        'another thread is already controlling the robot' error.

        Strategy
        --------
        1. If we are already moving toward the *same* target (within threshold)
           AND the robot is not yet steady, skip this call — the ongoing motion
           will get there on its own.
        2. If the target changed significantly, stop the current motion first,
           then issue a fresh moveL.
        3. On any 'another thread' exception, stop then retry once.
        """
        import time as _time

        move_l = getattr(self._rtde_control, "moveL", None)
        stop_l = getattr(self._rtde_control, "stopL", None)
        if not callable(move_l):
            raise RuntimeError("RTDE control interface does not expose moveL")

        # --- Check if we are still chasing the same target ---
        if self._pending_move_target is not None and self._move_in_progress:
            delta = self._norm3([
                target_pose[i] - self._pending_move_target[i] for i in range(3)
            ]) or 0.0
            if delta < self._MOVE_REISSUE_THRESHOLD_M:
                # Same target — check whether the robot has finished.
                try:
                    is_steady = getattr(self._rtde_control, "isSteady", None)
                    if callable(is_steady) and not is_steady():
                        return  # Still moving, nothing to do.
                    # Robot reached the target or isSteady not available.
                    self._move_in_progress = False
                except Exception:
                    return  # Assume still moving on error.

        # --- New target or previous move finished: stop first if needed ---
        if self._move_in_progress and callable(stop_l):
            try:
                stop_l(self.stop_acceleration)
                _time.sleep(0.02)  # Brief settle so UR releases the thread.
            except Exception:
                pass
            self._move_in_progress = False

        # --- Issue the new moveL (async=True) ---
        def _do_move() -> None:
            try:
                move_l(list(target_pose), self.move_speed_mps, self.move_acceleration, True)
            except TypeError:
                move_l(list(target_pose), self.move_speed_mps, self.move_acceleration)

        try:
            _do_move()
            self._pending_move_target = target_pose
            self._move_in_progress = True
        except Exception as exc:
            err_str = str(exc).lower()
            if "another thread" in err_str or "already controlling" in err_str:
                # UR firmware still has the thread lock — stop and retry once.
                if callable(stop_l):
                    try:
                        stop_l(self.stop_acceleration)
                    except Exception:
                        pass
                _time.sleep(0.05)
                try:
                    _do_move()
                    self._pending_move_target = target_pose
                    self._move_in_progress = True
                    self._last_error = None
                except Exception as exc2:
                    self._last_error = str(exc2)
                    self._move_in_progress = False
            else:
                self._last_error = str(exc)
                self._move_in_progress = False

    def _servo_to_pose(self, target_pose: Tuple[float, float, float, float, float, float], *, loop_dt: float = 1.0 / 30.0) -> None:
        """Send a servoL command to the UR controller.

        The ``loop_dt`` argument MUST reflect the actual wall-clock period
        between consecutive servoL calls.  Passing the wrong value causes
        the UR controller to mis-estimate the expected arrival rate and
        triggers a protective stop.  We accept it from the caller (runner)
        which measures the real loop period each step.
        """
        servo_l = getattr(self._rtde_control, "servoL", None)
        if not callable(servo_l):
            self._move_to_pose(target_pose)
            return
        # Clamp dt to a reasonable range: UR requires 0.002 <= dt <= 0.2
        dt = float(np.clip(loop_dt, 0.002, 0.2))
        servo_l(
            list(target_pose),
            self.servo_velocity,
            self.servo_acceleration,
            dt,
            self.servo_lookahead_time,
            self.servo_gain,
        )

    def _safe_stop_motion(self) -> None:
        if self._using_mock:
            self._mock_speed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._move_in_progress = False
            self._pending_move_target = None
            return

        # Clear pending move state regardless of success
        self._move_in_progress = False
        self._pending_move_target = None

        if self._last_command_type == ROBOT_CMD_SERVO_TO_POSITION:
            servo_stop = getattr(self._rtde_control, "servoStop", None)
            if callable(servo_stop):
                try:
                    servo_stop()
                    self._last_command_type = ROBOT_CMD_HOLD
                    return
                except TypeError:
                    try:
                        servo_stop(self.stop_acceleration)
                        self._last_command_type = ROBOT_CMD_HOLD
                        return
                    except Exception:
                        pass
                except Exception:
                    pass

        stop_l = getattr(self._rtde_control, "stopL", None)
        if callable(stop_l):
            try:
                stop_l(self.stop_acceleration)
                self._last_command_type = ROBOT_CMD_HOLD
                return
            except TypeError:
                stop_l()
                self._last_command_type = ROBOT_CMD_HOLD
                return
            except Exception:
                pass

        speed_stop = getattr(self._rtde_control, "speedStop", None)
        if callable(speed_stop):
            try:
                speed_stop()
            finally:
                self._last_command_type = ROBOT_CMD_HOLD

    def _apply_gripper_action(self, gripper_action: Optional[str]) -> bool:
        if gripper_action in (None, GRIPPER_HOLD):
            return False

        normalized = self._normalize_gripper_action(gripper_action)
        self._last_gripper_state = normalized
        if self._using_mock:
            if self.gripper_debug:
                print(f"[rtde_controller] mock gripper action={normalized}", flush=True)
            return False

        if self.gripper_control_mode in {"socket", "robotiq", "robotiq_socket", "daemon"}:
            if self._robotiq_gripper is None:
                if self.gripper_debug:
                    print(
                        "[rtde_controller] gripper action requested but Robotiq socket controller is not connected.",
                        flush=True,
                    )
                return False
            try:
                if self.gripper_debug:
                    print(
                        "[rtde_controller] gripper action=%s mode=%s port=%s"
                        % (normalized, self.gripper_control_mode, self.gripper_socket_port),
                        flush=True,
                    )
                if normalized == GRIPPER_OPEN:
                    self._robotiq_gripper.open(speed=self.gripper_open_speed, force=self.gripper_open_force)
                elif normalized == GRIPPER_CLOSE:
                    self._robotiq_gripper.close(speed=self.gripper_close_speed, force=self.gripper_close_force)
                else:
                    return False
                return True
            except Exception as exc:
                self._last_error = f"Failed Robotiq socket gripper action {normalized}: {exc}"
                if self.gripper_debug:
                    print(f"[rtde_controller] {self._last_error}", flush=True)
                return False

        if self._rtde_io is None:
            if self.gripper_debug:
                print(
                    "[rtde_controller] gripper action requested but RTDE IO is not available. "
                    "If your Robotiq uses URCap/RS485, switch robot.gripper.control_mode to socket.",
                    flush=True,
                )
            return False

        if self.gripper_debug:
            print(
                "[rtde_controller] gripper action=%s domain=%s open_do=%s close_do=%s high_when_active=%s"
                % (
                    normalized,
                    self.gripper_output_domain,
                    self.open_digital_output,
                    self.close_digital_output,
                    self.output_high_when_active,
                ),
                flush=True,
            )

        applied = False
        if normalized == GRIPPER_OPEN:
            applied |= self._set_configured_output(self.open_digital_output, True)
            applied |= self._set_configured_output(self.close_digital_output, False)
        elif normalized == GRIPPER_CLOSE:
            applied |= self._set_configured_output(self.close_digital_output, True)
            applied |= self._set_configured_output(self.open_digital_output, False)

        if self.gripper_debug and not applied:
            print(
                "[rtde_controller] no gripper outputs were applied. Check robot.gripper config and whether this gripper actually uses digital outputs.",
                flush=True,
            )
        return applied

    def _set_configured_output(self, output_index: Any, active: bool) -> bool:
        if output_index is None:
            if self.gripper_debug:
                print("[rtde_controller] skip gripper output because output index is None", flush=True)
            return False
        value = bool(active) if self.output_high_when_active else not bool(active)
        output_index = int(output_index)
        domain = self.gripper_output_domain.strip().lower()
        if domain == "tool":
            setter = getattr(self._rtde_io, "setToolDigitalOut", None)
        elif domain == "configurable":
            setter = getattr(self._rtde_io, "setConfigurableDigitalOut", None)
        else:
            setter = getattr(self._rtde_io, "setStandardDigitalOut", None)
        if not callable(setter):
            if self.gripper_debug:
                print(f"[rtde_controller] no setter available for gripper output domain={domain}", flush=True)
            return False
        try:
            setter(output_index, value)
            if self.gripper_debug:
                print(
                    f"[rtde_controller] set {domain} digital output {output_index} -> {int(value)}",
                    flush=True,
                )
            return True
        except Exception as exc:
            self._last_error = f"Failed gripper output domain={domain} index={output_index} value={int(value)}: {exc}"
            if self.gripper_debug:
                print(f"[rtde_controller] {self._last_error}", flush=True)
            return False

    def _normalize_gripper_action(self, gripper_action: str) -> str:
        if gripper_action == self.open_command:
            return GRIPPER_OPEN
        if gripper_action == self.close_command:
            return GRIPPER_CLOSE
        return str(gripper_action)

    def _compute_grasp_flags(self, robot_state: RobotState) -> Tuple[bool, bool, bool]:
        return self._compute_grasp_flags_from_values(
            tcp_force_norm_n=robot_state.tcp_force_norm_n,
            mean_joint_current_a=robot_state.mean_joint_current_a,
        )

    def _compute_grasp_flags_from_values(
        self,
        *,
        tcp_force_norm_n: Optional[float],
        mean_joint_current_a: Optional[float],
    ) -> Tuple[bool, bool, bool]:
        force_ok = (not self.use_tcp_force) or (
            tcp_force_norm_n is not None and float(tcp_force_norm_n) >= self.min_tcp_force_norm_n
        )
        current_ok = (not self.use_joint_current) or (
            mean_joint_current_a is not None and float(mean_joint_current_a) >= self.min_mean_joint_current_a
        )

        if self.use_tcp_force and self.use_joint_current:
            verified = (force_ok and current_ok) if self.require_both_signals else (force_ok or current_ok)
        elif self.use_tcp_force:
            verified = force_ok
        elif self.use_joint_current:
            verified = current_ok
        else:
            verified = False
        return bool(force_ok), bool(current_ok), bool(verified)

    def _compose_target_pose(
        self,
        target_position_base: Optional[Sequence[float]],
        fixed_orientation_base: Optional[Sequence[float]],
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        if target_position_base is None:
            return None

        position = self._to_vec3(target_position_base)
        if position is None:
            return None
        position = self._map_outgoing_vec3(position)

        orientation = self._resolve_orientation(fixed_orientation_base)
        if orientation is None:
            return None
        return position + orientation

    def _resolve_orientation(
        self,
        fixed_orientation_base: Optional[Sequence[float]],
    ) -> Optional[Tuple[float, float, float]]:
        sequence = self._to_float_sequence(fixed_orientation_base, expected_length=None)
        if sequence is None:
            if self._last_safe_pose_base is not None:
                return self._last_safe_pose_base[3:6]
            return (0.0, 0.0, 0.0)
        if len(sequence) == 3:
            if self.fixed_orientation_format == "rpy":
                return self._rpy_to_rotvec(sequence)
            return tuple(float(value) for value in sequence)
        if len(sequence) == 6:
            trailing = tuple(float(value) for value in sequence[3:6])
            if self.fixed_orientation_format == "rpy":
                return self._rpy_to_rotvec(trailing)
            return trailing
        return None

    @staticmethod
    def _rpy_to_rotvec(rpy: Sequence[float]) -> Tuple[float, float, float]:
        roll, pitch, yaw = [float(value) for value in rpy]
        cx, sx = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw), np.sin(yaw)

        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        rotation = rot_z @ rot_y @ rot_x

        trace = float(np.trace(rotation))
        cos_theta = max(min((trace - 1.0) * 0.5, 1.0), -1.0)
        theta = float(np.arccos(cos_theta))
        if theta < 1e-9:
            return (0.0, 0.0, 0.0)

        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            diag = np.diag(rotation)
            axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0))
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-9:
                return (0.0, 0.0, 0.0)
        axis = axis / axis_norm
        rotvec = axis * theta
        return (float(rotvec[0]), float(rotvec[1]), float(rotvec[2]))

    @staticmethod
    def _call_first_available(interface: Any, method_names: Sequence[str]) -> Any:
        if interface is None:
            return None
        for method_name in method_names:
            method = getattr(interface, method_name, None)
            if callable(method):
                try:
                    return method()
                except Exception:
                    continue
        return None

    @staticmethod
    def _to_float_sequence(values: Any, expected_length: Optional[int]) -> Optional[Tuple[float, ...]]:
        if values is None:
            return None
        if isinstance(values, (str, bytes)):
            return None
        try:
            sequence = tuple(float(value) for value in values)
        except TypeError:
            return None
        if expected_length is not None and len(sequence) != expected_length:
            return None
        return sequence


    @staticmethod
    def _normalize_position_signs(values: Sequence[float]) -> Tuple[float, float, float]:
        if values is None:
            return (-1.0, -1.0, 1.0)
        try:
            signs = tuple(float(value) for value in values)
        except TypeError:
            return (-1.0, -1.0, 1.0)
        if len(signs) != 3:
            return (-1.0, -1.0, 1.0)
        normalized = []
        for value in signs:
            normalized.append(-1.0 if value < 0.0 else 1.0)
        return tuple(normalized)

    def _map_incoming_vec3(self, values: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
        vector = self._to_vec3(values)
        if vector is None:
            return None
        if not self.frame_mapping_enabled:
            return vector
        return tuple(float(sign * value) for sign, value in zip(self.position_signs, vector))

    def _map_outgoing_vec3(self, values: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
        vector = self._to_vec3(values)
        if vector is None:
            return None
        if not self.frame_mapping_enabled:
            return vector
        return tuple(float(sign * value) for sign, value in zip(self.position_signs, vector))

    def _map_incoming_pose6(self, values: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
        pose = self._to_pose6(values)
        if pose is None:
            return None
        if not self.frame_mapping_enabled:
            return pose
        xyz = tuple(float(sign * value) for sign, value in zip(self.position_signs, pose[:3]))
        return xyz + tuple(float(value) for value in pose[3:6])

    @classmethod
    def _to_vec3(cls, values: Sequence[float]) -> Optional[Tuple[float, float, float]]:
        sequence = cls._to_float_sequence(values, expected_length=3)
        if sequence is None:
            return None
        return tuple(float(value) for value in sequence)

    @classmethod
    def _to_pose6(cls, values: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
        sequence = cls._to_float_sequence(values, expected_length=6)
        if sequence is None:
            return None
        return tuple(float(value) for value in sequence)

    @classmethod
    def _to_joint_tuple(cls, values: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
        sequence = cls._to_float_sequence(values, expected_length=6)
        if sequence is None:
            return None
        return tuple(float(value) for value in sequence)

    @staticmethod
    def _norm3(values: Sequence[float]) -> Optional[float]:
        if values is None:
            return None
        values = tuple(float(value) for value in values)
        if len(values) < 3:
            return None
        return sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])

    @staticmethod
    def _mean_abs(values: Optional[Iterable[float]]) -> Optional[float]:
        if values is None:
            return None
        values = tuple(float(value) for value in values)
        if not values:
            return None
        return sum(abs(value) for value in values) / float(len(values))


__all__ = ["DEFAULT_CONFIG_PATH", "RtdeControllerDebug", "RtdeController"]
