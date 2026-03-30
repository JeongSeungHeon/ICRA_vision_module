"""High-level Robotiq gripper controller using the UR daemon socket interface."""

from __future__ import annotations

import time

from robot.robotiq_gripper import RobotiqGripper


class RobotiqGripperController:
    def __init__(
        self,
        robot_ip: str,
        *,
        port: int = 63352,
        settle_time: float = 0.15,
        socket_timeout: float = 2.0,
        verbose: bool = False,
        activate_on_connect: bool = True,
    ) -> None:
        self.robot_ip = robot_ip
        self.port = int(port)
        self.settle_time = float(settle_time)
        self.socket_timeout = float(socket_timeout)
        self.verbose = bool(verbose)
        self.activate_on_connect = bool(activate_on_connect)
        self._is_closed = False
        self._is_connected = False
        self.gripper = RobotiqGripper()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def connect(self) -> None:
        self.gripper.connect(self.robot_ip, self.port, socket_timeout=self.socket_timeout)
        self._is_connected = True
        if self.verbose:
            print(f"[Gripper] Connected to {self.robot_ip}:{self.port}", flush=True)
        if self.activate_on_connect:
            if self.verbose:
                print("[Gripper] Activating gripper...", flush=True)
            self.gripper.activate(auto_calibrate=False)
            time.sleep(1.0)
            if self.verbose:
                print("[Gripper] Activation complete.", flush=True)

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._is_connected = False
        self.gripper.disconnect()
        if self.verbose:
            print(f"[Gripper] Disconnected from {self.robot_ip}:{self.port}", flush=True)

    def set_closed(self, is_closed: bool, *, speed: int = 255, force: int = 255, wait: bool = True) -> bool:
        if not self._is_connected:
            raise RuntimeError("Gripper controller is not connected.")
        if is_closed:
            pos = self.gripper.get_closed_position()
        else:
            pos = self.gripper.get_open_position()

        if wait:
            self.gripper.move_and_wait_for_pos(pos, speed, force)
        else:
            ok, _ = self.gripper.move(pos, speed, force)
            if not ok:
                raise RuntimeError("Failed to send gripper move command.")

        self._is_closed = bool(is_closed)
        if wait and self.settle_time > 0:
            time.sleep(self.settle_time)
        if self.verbose:
            state = "CLOSED" if self._is_closed else "OPEN"
            suffix = "" if wait else " (async)"
            print(f"[Gripper] {state}{suffix}", flush=True)
        return self._is_closed

    def stop(self) -> bool:
        if not self._is_connected:
            raise RuntimeError("Gripper controller is not connected.")
        stopped = self.gripper.stop()
        if self.verbose:
            print("[Gripper] STOP", flush=True)
        return bool(stopped)

    def get_motion_state(self) -> dict[str, object]:
        if not self._is_connected:
            raise RuntimeError("Gripper controller is not connected.")
        position = int(self.gripper.get_current_position())
        closed_position = int(self.gripper.get_closed_position())
        object_status = self.gripper.get_object_status()
        return {
            "position": position,
            "closed_position": closed_position,
            "object_status": object_status.name,
            "fully_closed": position >= max(closed_position - 1, 0),
        }

    def open(self, *, speed: int = 255, force: int = 255, wait: bool = True) -> bool:
        return self.set_closed(False, speed=speed, force=force, wait=wait)

    def close(self, *, speed: int = 255, force: int = 255, wait: bool = True) -> bool:
        return self.set_closed(True, speed=speed, force=force, wait=wait)
