#!/usr/bin/env python3
"""Small UR5 + Robotiq connection smoke test.

This script intentionally lives outside the main runtime. It reads the same
handover YAML settings, connects to the UR5 with RTDE, opens/closes the
Robotiq gripper, and optionally nudges the TCP along the UR base +X axis.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on selected virtualenv
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "handover.yaml"
DEFAULT_ROBOT_IP = "192.168.56.101"
RTDE_PORT = 30004
DASHBOARD_PORT = 29999

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:  # pragma: no cover - depends on robot PC environment
    RTDEControlInterface = None
    RTDEReceiveInterface = None

from robot.robotiq_gripper_controller import RobotiqGripperController


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"[WARN] Config not found: {path}")
        return {}
    if yaml is None:
        print("[WARN] PyYAML is not installed. Using built-in defaults and CLI overrides only.")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def nested_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_robot_ip(config: dict[str, Any], cli_robot_ip: str | None) -> str:
    if cli_robot_ip:
        return cli_robot_ip
    config_ip = nested_get(config, "robot", "rtde", "robot_ip")
    if config_ip:
        return str(config_ip)
    return DEFAULT_ROBOT_IP


def confirm(prompt: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("[SKIP] Interactive confirmation is unavailable. Re-run with --yes to allow motion.")
        return False
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def format_pose(pose: list[float] | tuple[float, ...] | None) -> str:
    if pose is None:
        return "None"
    if len(pose) < 6:
        return str(pose)
    return (
        "x={:.4f}, y={:.4f}, z={:.4f}, "
        "rx={:.4f}, ry={:.4f}, rz={:.4f}"
    ).format(*pose[:6])


def pose_position_error_m(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)) ** 0.5


def describe_robot_mode(value: object) -> str:
    return {
        7: "RUNNING",
    }.get(value, "unknown")


def describe_safety_mode(value: object) -> str:
    return {
        1: "NORMAL",
    }.get(value, "unknown")


def describe_runtime_state(value: object) -> str:
    return {
        0: "STOPPING",
        1: "STOPPED",
        2: "PLAYING",
        3: "PAUSING",
        4: "PAUSED",
        5: "RESUMING",
    }.get(value, "unknown")


def check_tcp_port(host: str, port: int, timeout_s: float, *, label: str = "TCP") -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            print(f"[OK] {label} port reachable: {host}:{port}")
            return True
    except OSError as exc:
        print(f"[FAIL] {label} port check failed: {host}:{port} ({exc})")
        return False


def print_remote_control_hint() -> None:
    print(
        "[HINT] Check UR teach pendant: robot is powered on, brakes are released, "
        "and Remote Control mode is enabled. Also verify the PC and UR5 are on "
        "the same subnet."
    )


def print_rtde_register_hint() -> None:
    print(
        "[HINT] RTDE receive is reachable, but RTDE control could not reserve "
        "the robot input registers. On the UR teach pendant, disable unused "
        "EtherNet/IP, PROFINET, or MODBUS fieldbus/adapters, then restart the "
        "program/robot if needed. Also close any other Python process that may "
        "already be using RTDE control."
    )


def print_motion_rejected_hint() -> None:
    print(
        "[HINT] moveL returned False, so the robot rejected the motion command. "
        "Because RTDE read works but Dashboard 29999 times out and runtime_state "
        "is likely STOPPED, check that the UR is in Remote Control mode and that "
        "remote script/program execution is allowed. If needed, switch Local -> "
        "Remote on the pendant, release brakes, and make sure no other UR program "
        "or RTDE client is holding control."
    )


def connect_rtde(robot_ip: str, *, timeout_s: float, require_control: bool):
    if RTDEReceiveInterface is None or (require_control and RTDEControlInterface is None):
        raise RuntimeError(
            "ur_rtde is not installed in this Python environment "
            "(missing rtde_control / rtde_receive)."
        )
    if not check_tcp_port(robot_ip, RTDE_PORT, timeout_s, label="RTDE"):
        print_remote_control_hint()
        raise RuntimeError("UR RTDE port is not reachable.")

    check_tcp_port(robot_ip, DASHBOARD_PORT, timeout_s, label="Dashboard")
    print(f"[RTDE] Connecting to UR5 at {robot_ip} ...")
    rtde_receive = RTDEReceiveInterface(robot_ip)
    rtde_control = None
    if require_control:
        try:
            rtde_control = RTDEControlInterface(robot_ip)
        except RuntimeError as exc:
            if "input registers are already in use" in str(exc):
                print_rtde_register_hint()
            raise
    print("[RTDE] Connected." if require_control else "[RTDE] Receive-only connection OK.")
    return rtde_control, rtde_receive


def print_robot_state(rtde_receive) -> list[float]:
    pose = list(rtde_receive.getActualTCPPose())
    joints = list(rtde_receive.getActualQ())
    print(f"[RTDE] Actual TCP pose: {format_pose(pose)}")
    print("[RTDE] Actual joints(rad): " + ", ".join(f"{value:.4f}" for value in joints))

    diagnostics = [
        ("Robot mode", "getRobotMode", describe_robot_mode),
        ("Safety mode", "getSafetyMode", describe_safety_mode),
        ("Runtime state", "getRuntimeState", describe_runtime_state),
        ("Program running", "isProgramRunning", None),
        ("Protective stopped", "isProtectiveStopped", None),
        ("Emergency stopped", "isEmergencyStopped", None),
    ]
    for label, method_name, describer in diagnostics:
        method = getattr(rtde_receive, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
            description = f" ({describer(value)})" if callable(describer) else ""
            print(f"[RTDE] {label}: {value}{description}")
        except Exception as exc:
            print(f"[WARN] Could not read {label}: {exc}")
    return pose


def stop_rtde(rtde_control) -> None:
    if rtde_control is None:
        return
    for method_name in ("servoStop", "stopL", "speedStop"):
        method = getattr(rtde_control, method_name, None)
        if not callable(method):
            continue
        try:
            if method_name == "stopL":
                method(0.5)
            else:
                method()
        except TypeError:
            try:
                method()
            except Exception:
                pass
        except Exception:
            pass


def disconnect_rtde(rtde_control) -> None:
    if rtde_control is None:
        return
    stop_rtde(rtde_control)
    stop_script = getattr(rtde_control, "stopScript", None)
    if callable(stop_script):
        try:
            stop_script()
        except Exception:
            pass


def run_move_l_and_wait(
    rtde_control,
    rtde_receive,
    target_pose: list[float],
    *,
    speed_mps: float,
    acceleration: float,
    timeout_s: float,
    label: str,
) -> bool:
    print(f"[MOVE] Sending {label} moveL command...", flush=True)
    try:
        result = rtde_control.moveL(target_pose, float(speed_mps), float(acceleration), True)
        
        print(f"[MOVE] moveL returned: {result}", flush=True)
        if result is False:
            print_motion_rejected_hint()
            return False
    except Exception as exc:
        print(f"[ERROR] moveL command failed: {exc}", flush=True)
        return False

    start_time = time.monotonic()
    last_report_time = 0.0
    while True:
        current_pose = list(rtde_receive.getActualTCPPose())
        error_m = pose_position_error_m(current_pose, target_pose)
        elapsed_s = time.monotonic() - start_time

        if elapsed_s - last_report_time >= 0.5:
            print(
                f"[MOVE] Waiting {label}: pos_error={error_m * 1000.0:.1f} mm, "
                f"pose={format_pose(current_pose)}",
                flush=True,
            )
            last_report_time = elapsed_s

        if error_m <= 0.003:
            print(f"[MOVE] {label} reached: {format_pose(current_pose)}", flush=True)
            return True

        if elapsed_s >= timeout_s:
            print(
                f"[FAIL] {label} move timed out after {timeout_s:.1f}s. "
                f"Last pos_error={error_m * 1000.0:.1f} mm",
                flush=True,
            )
            stop_rtde(rtde_control)
            return False

        time.sleep(0.05)


def move_x_smoke_test(
    rtde_control,
    rtde_receive,
    *,
    delta_m: float,
    speed_mps: float,
    acceleration: float,
    move_timeout_s: float,
    return_home: bool,
    assume_yes: bool,
) -> None:
    if abs(delta_m) <= 0.0:
        print("[MOVE] Skipped because X delta is 0.")
        return

    start_pose = list(rtde_receive.getActualTCPPose())
    target_pose = list(start_pose)
    target_pose[0] += float(delta_m)

    print(f"[MOVE] Start pose : {format_pose(start_pose)}")
    print(f"[MOVE] Target pose: {format_pose(target_pose)}")
    print(f"[MOVE] X delta   : {delta_m * 1000.0:.1f} mm in UR base frame")

    if not confirm("Move the UR5 TCP along X now?", assume_yes=assume_yes):
        print("[MOVE] Skipped by user.")
        return

    reached = run_move_l_and_wait(
        rtde_control,
        rtde_receive,
        target_pose,
        speed_mps=speed_mps,
        acceleration=acceleration,
        timeout_s=move_timeout_s,
        label="+X",
    )
    if not reached:
        print_robot_state(rtde_receive)
        return

    if not return_home:
        print("[MOVE] Return move disabled (--no-return).")
        return

    if not confirm("Return the UR5 TCP to the start pose?", assume_yes=assume_yes):
        print("[MOVE] Return skipped by user.")
        return

    returned = run_move_l_and_wait(
        rtde_control,
        rtde_receive,
        start_pose,
        speed_mps=speed_mps,
        acceleration=acceleration,
        timeout_s=move_timeout_s,
        label="return",
    )
    if not returned:
        print_robot_state(rtde_receive)


def gripper_smoke_test(config: dict[str, Any], robot_ip: str, *, assume_yes: bool) -> None:
    gripper_cfg = nested_get(config, "robot", "gripper", default={}) or {}
    port = int(gripper_cfg.get("socket_port", 63352))
    timeout_s = float(gripper_cfg.get("socket_timeout_sec", 2.0))
    settle_s = float(gripper_cfg.get("socket_settle_time", 0.15))
    activate_on_connect = bool(gripper_cfg.get("activate_on_connect", True))
    open_speed = int(gripper_cfg.get("open_speed", 250))
    open_force = int(gripper_cfg.get("open_force", 255))
    close_speed = int(gripper_cfg.get("close_speed", 255))
    close_force = int(gripper_cfg.get("close_force", 255))

    print(f"[GRIPPER] Checking socket {robot_ip}:{port} ...")
    if not check_tcp_port(robot_ip, port, timeout_s, label="Gripper"):
        return

    if not confirm("Run gripper open-close-open test?", assume_yes=assume_yes):
        print("[GRIPPER] Skipped by user.")
        return

    controller = RobotiqGripperController(
        robot_ip,
        port=port,
        settle_time=settle_s,
        socket_timeout=timeout_s,
        verbose=True,
        activate_on_connect=activate_on_connect,
    )

    try:
        controller.connect()
        print(f"[GRIPPER] Diagnostic after connect: {controller.get_diagnostic_state()}")
        controller.open(speed=open_speed, force=open_force, wait=True)
        print(f"[GRIPPER] After open : {controller.get_motion_state()}")
        time.sleep(0.3)
        controller.close(speed=close_speed, force=close_force, wait=True)
        print(f"[GRIPPER] After close: {controller.get_motion_state()}")
        time.sleep(0.3)
        controller.open(speed=open_speed, force=open_force, wait=True)
        print(f"[GRIPPER] After reopen: {controller.get_motion_state()}")
    finally:
        controller.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test UR5 RTDE connection, Robotiq gripper socket, and a small +X move."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to handover YAML.")
    parser.add_argument("--robot-ip", default=None, help="Override robot IP from YAML.")
    parser.add_argument("--skip-robot", action="store_true", help="Skip RTDE connection and X move.")
    parser.add_argument("--skip-gripper", action="store_true", help="Skip Robotiq open/close test.")
    parser.add_argument("--skip-move", action="store_true", help="Only connect/read robot state; do not move X.")
    parser.add_argument("--x-mm", type=float, default=20.0, help="X-axis test move in millimetres.")
    parser.add_argument("--speed", type=float, default=0.03, help="moveL speed in m/s.")
    parser.add_argument("--acceleration", type=float, default=0.10, help="moveL acceleration.")
    parser.add_argument("--connect-timeout", type=float, default=2.0, help="TCP port check timeout in seconds.")
    parser.add_argument("--move-timeout", type=float, default=5.0, help="Timeout for each moveL command in seconds.")
    parser.add_argument("--no-return", action="store_true", help="Do not move back to the start pose.")
    parser.add_argument("--yes", action="store_true", help="Run physical actions without interactive prompts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    robot_ip = resolve_robot_ip(config, args.robot_ip)

    print(f"[CONFIG] Config path: {args.config}")
    print(f"[CONFIG] Robot IP   : {robot_ip}")

    rtde_control = None
    try:
        if not args.skip_robot:
            rtde_control, rtde_receive = connect_rtde(
                robot_ip,
                timeout_s=float(args.connect_timeout),
                require_control=not args.skip_move,
            )
            print_robot_state(rtde_receive)
            if not args.skip_move:
                if rtde_control is None:
                    raise RuntimeError("RTDE control is not connected, so the X move cannot run.")
                move_x_smoke_test(
                    rtde_control,
                    rtde_receive,
                    delta_m=float(args.x_mm) / 1000.0,
                    speed_mps=float(args.speed),
                    acceleration=float(args.acceleration),
                    move_timeout_s=float(args.move_timeout),
                    return_home=not args.no_return,
                    assume_yes=bool(args.yes),
                )

        if not args.skip_gripper:
            gripper_smoke_test(config, robot_ip, assume_yes=bool(args.yes))

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
        stop_rtde(rtde_control)
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}")
        stop_rtde(rtde_control)
        return 1
    finally:
        disconnect_rtde(rtde_control)

    print("[DONE] Smoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
