"""Low-level robot action helpers shared by handover entrypoints.

The large live script still exposes compatibility aliases for existing tests, but
new code should route robot execution through :mod:`robot.robot_worker`.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from robot.rtde_controller import RtdeController
from system.shared_state import (
    GRIPPER_OPEN,
    ROBOT_CMD_HOLD,
    ROBOT_CMD_STOP,
    RobotCommandState,
)


DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
HOME_JOINTS_DEG = [0.0, -135.0, 135.0, 0.0, 90.0, 0.0]
HOME_JOINT_TOLERANCE_DEG = 1.0
HOME_JOINT_SPEED_RAD_S = 0.5
HOME_JOINT_ACCELERATION_RAD_S2 = 0.5


def get_home_joints_rad() -> tuple[float, float, float, float, float, float]:
    return tuple(float(np.deg2rad(value)) for value in HOME_JOINTS_DEG)


def make_robot_command(
    command_type: str,
    *,
    target_position_base=None,
    fixed_orientation_base=None,
    gripper_action=None,
    source_mode: str = "manual",
) -> RobotCommandState:
    return RobotCommandState(
        command_type=command_type,
        target_position_base=target_position_base,
        fixed_orientation_base=fixed_orientation_base,
        gripper_action=gripper_action,
        source_mode=source_mode,
        stop_requested=(command_type == ROBOT_CMD_STOP),
        timestamp=time.time(),
        valid=True,
    )


def send_robot_command(
    controller: Any,
    command_type: str,
    *,
    target_position_base=None,
    fixed_orientation_base=None,
    gripper_action=None,
    source_mode: str = "manual",
):
    command = make_robot_command(
        command_type,
        target_position_base=target_position_base,
        fixed_orientation_base=fixed_orientation_base,
        gripper_action=gripper_action,
        source_mode=source_mode,
    )
    return controller.step(command, now_timestamp=command.timestamp)


def init_rtde(args: Any):
    print(f"[INFO] Connecting to UR5 RTDE using config {args.config} ...")
    controller = RtdeController.from_config(args.config)
    if getattr(args, "robot_ip", None):
        controller.robot_ip = args.robot_ip
    controller.fixed_orientation_format = "rotvec"
    controller.connect()

    state = controller.read_robot_state(now_timestamp=time.time())
    if not state.is_connected:
        raise RuntimeError(f"Failed to connect RTDE controller. last_error={state.last_error}")
    if state.actual_tcp_pose_base is None:
        raise RuntimeError("RTDE connected but actual_tcp_pose_base is unavailable.")

    if getattr(args, "move_to_base", False):
        print("[INFO] --move-to-base is ignored; HOME is defined by HOME_JOINTS_DEG.")

    startup_open_ok = False
    if hasattr(controller, "open_gripper_blocking"):
        try:
            startup_open_ok = bool(controller.open_gripper_blocking())
        except Exception as exc:
            print(f"[WARN] Direct startup gripper open failed: {exc}")
    if not startup_open_ok:
        send_robot_command(controller, ROBOT_CMD_HOLD, gripper_action=GRIPPER_OPEN, source_mode="startup_open")
    time.sleep(0.2)
    return controller


def safe_stop_rtde(controller: Any) -> None:
    if controller is None:
        return
    try:
        send_robot_command(controller, ROBOT_CMD_STOP, source_mode="manual_stop")
    except Exception:
        pass


def disconnect_rtde(controller: Any) -> None:
    if controller is None:
        return
    try:
        controller.close()
    except Exception:
        pass


def wait_until_target_reached(
    controller: Any,
    target_position_base,
    *,
    timeout_s: float,
    tolerance_m: float,
    poll_dt: float = 0.05,
    cancel_event=None,
) -> bool:
    deadline = time.time() + float(timeout_s)
    target = np.asarray(target_position_base, dtype=np.float32).reshape(3)
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_rtde(controller)
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        pose = state.actual_tcp_pose_base
        if pose is not None:
            current = np.asarray(pose[:3], dtype=np.float32).reshape(3)
            if float(np.linalg.norm(current - target)) <= float(tolerance_m):
                return True
        time.sleep(float(poll_dt))
    return False


def wait_until_joint_target_reached(
    controller: Any,
    target_joints_rad,
    *,
    timeout_s: float,
    tolerance_rad: float,
    poll_dt: float = 0.05,
    cancel_event=None,
) -> bool:
    deadline = time.time() + float(timeout_s)
    target = np.asarray(target_joints_rad, dtype=np.float64).reshape(6)
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            if hasattr(controller, "stop_joint_motion"):
                controller.stop_joint_motion()
            else:
                safe_stop_rtde(controller)
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        joints = getattr(state, "joint_positions", None)
        if joints is not None:
            current = np.asarray(joints, dtype=np.float64).reshape(6)
            if np.all(np.isfinite(current)) and float(np.max(np.abs(current - target))) <= float(tolerance_rad):
                return True
        time.sleep(float(poll_dt))
    return False


def move_robot_to_home_pose(controller: Any, args: Any, cancel_event=None) -> None:
    home_joints_rad = get_home_joints_rad()
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("HOME move cancelled before start.")
    if not hasattr(controller, "move_to_joint_positions"):
        raise RuntimeError("RTDE controller does not support joint HOME moves.")
    started = bool(
        controller.move_to_joint_positions(
            home_joints_rad,
            speed_rad_s=HOME_JOINT_SPEED_RAD_S,
            acceleration_rad_s2=HOME_JOINT_ACCELERATION_RAD_S2,
            async_move=True,
        )
    )
    if not started:
        raise RuntimeError("Failed to start HOME joint move.")
    ok = wait_until_joint_target_reached(
        controller,
        home_joints_rad,
        timeout_s=args.move_timeout_s,
        tolerance_rad=float(np.deg2rad(HOME_JOINT_TOLERANCE_DEG)),
        cancel_event=cancel_event,
    )
    if not ok:
        if hasattr(controller, "stop_joint_motion"):
            controller.stop_joint_motion()
        else:
            safe_stop_rtde(controller)
        raise RuntimeError("Failed to reach HOME joints during startup.")


def _live_action(name: str):
    """Load legacy live-script helpers lazily for compatibility imports."""
    import importlib

    live = importlib.import_module("robot_control_rtde_fitting_final")
    return getattr(live, name)


def move_robot_and_wait(*args, **kwargs):
    return _live_action("move_robot_and_wait")(*args, **kwargs)


def execute_gripper_close(*args, **kwargs):
    return _live_action("execute_gripper_close")(*args, **kwargs)


def execute_gripper_open(*args, **kwargs):
    return _live_action("execute_gripper_open")(*args, **kwargs)


def capture_tactile_release_reference(*args, **kwargs):
    return _live_action("capture_tactile_release_reference")(*args, **kwargs)


def execute_tactile_release_descent(*args, **kwargs):
    return _live_action("execute_tactile_release_descent")(*args, **kwargs)


def reset_tactile_baseline_after_open(*args, **kwargs):
    return _live_action("reset_tactile_baseline_after_open")(*args, **kwargs)


def save_grasp_offset(*args, **kwargs):
    return _live_action("save_grasp_offset")(*args, **kwargs)


def compute_place_target(*args, **kwargs):
    return _live_action("compute_place_target")(*args, **kwargs)


def execute_return_and_place(*args, **kwargs):
    return _live_action("execute_return_and_place")(*args, **kwargs)


def configure_gripper_position_threshold_from_geometry(*args, **kwargs):
    return _live_action("configure_gripper_position_threshold_from_geometry")(*args, **kwargs)


def reset_gripper_position_threshold_to_config_default(*args, **kwargs):
    return _live_action("reset_gripper_position_threshold_to_config_default")(*args, **kwargs)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HOME_JOINTS_DEG",
    "disconnect_rtde",
    "get_home_joints_rad",
    "init_rtde",
    "make_robot_command",
    "capture_tactile_release_reference",
    "compute_place_target",
    "configure_gripper_position_threshold_from_geometry",
    "move_robot_to_home_pose",
    "move_robot_and_wait",
    "execute_gripper_close",
    "execute_gripper_open",
    "execute_return_and_place",
    "execute_tactile_release_descent",
    "safe_stop_rtde",
    "save_grasp_offset",
    "send_robot_command",
    "reset_gripper_position_threshold_to_config_default",
    "reset_tactile_baseline_after_open",
    "wait_until_joint_target_reached",
    "wait_until_target_reached",
]
