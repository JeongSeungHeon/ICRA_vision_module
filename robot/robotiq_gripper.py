"""Socket-based Robotiq gripper client adapted for UR daemon control."""

from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import Union


class RobotiqGripper:
    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    ADR = "ADR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"
    STA = "STA"
    PRE = "PRE"
    OBJ = "OBJ"
    FLT = "FLT"

    ENCODING = "UTF-8"

    class GripperStatus(Enum):
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(Enum):
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self) -> None:
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        if self.socket is None:
            return
        try:
            self.socket.close()
        finally:
            self.socket = None

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]) -> bool:
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {value}"
        cmd += "\n"
        with self.command_lock:
            assert self.socket is not None
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return data == b"ack"

    def _set_var(self, variable: str, value: Union[int, float]) -> bool:
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str) -> int:
        with self.command_lock:
            assert self.socket is not None
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data!r} for variable {variable}")
        return int(value_str)

    def _reset(self) -> None:
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while self._get_var(self.ACT) != 0 or self._get_var(self.STA) != 0:
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)

    def activate(self, auto_calibrate: bool = False) -> None:
        if not self.is_active():
            self._reset()
            while self._get_var(self.ACT) != 0 or self._get_var(self.STA) != 0:
                time.sleep(0.01)
            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            while self._get_var(self.ACT) != 1 or self._get_var(self.STA) != 3:
                time.sleep(0.01)
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self) -> bool:
        return self.GripperStatus(self._get_var(self.STA)) == self.GripperStatus.ACTIVE

    def get_open_position(self) -> int:
        return self._min_position

    def get_closed_position(self) -> int:
        return self._max_position

    def get_current_position(self) -> int:
        return self._get_var(self.POS)

    def get_object_status(self) -> "RobotiqGripper.ObjectStatus":
        return self.ObjectStatus(self._get_var(self.OBJ))

    def stop(self) -> bool:
        current_position = self.get_current_position()
        ok, _ = self.move(current_position, 0, 0)
        return bool(ok)

    def auto_calibrate(self) -> None:
        position, status = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {status}")
        position, status = self.move_and_wait_for_pos(self.get_closed_position(), 64, 1)
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {status}")
        self._max_position = position
        position, status = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {status}")
        self._min_position = position

    def move(self, position: int, speed: int, force: int) -> tuple[bool, int]:
        def clip_val(min_val: int, val: int, max_val: int) -> int:
            return max(min_val, min(int(val), max_val))

        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)
        var_dict = OrderedDict([
            (self.POS, clip_pos),
            (self.SPE, clip_spe),
            (self.FOR, clip_for),
            (self.GTO, 1),
        ])
        return self._set_vars(var_dict), clip_pos

    def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> tuple[int, "RobotiqGripper.ObjectStatus"]:
        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move.")
        while self._get_var(self.PRE) != cmd_pos:
            time.sleep(0.001)
        cur_obj = self._get_var(self.OBJ)
        while self.ObjectStatus(cur_obj) == self.ObjectStatus.MOVING:
            cur_obj = self._get_var(self.OBJ)
        final_pos = self._get_var(self.POS)
        return final_pos, self.ObjectStatus(cur_obj)
