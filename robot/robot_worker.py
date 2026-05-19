"""Single-owner robot worker for UR handover actions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from system.shared_state import (
    GRIPPER_HOLD,
    ROBOT_CMD_SERVO_TO_POSITION,
)


class RobotRequestType(str, Enum):
    INIT_ROBOT = "INIT_ROBOT"
    START_FOLLOW = "START_FOLLOW"
    STOP_FOLLOW = "STOP_FOLLOW"
    START_GRASP_PLACE = "START_GRASP_PLACE"
    RESET_HOME = "RESET_HOME"
    SAVE_AND_STOP = "SAVE_AND_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SHUTDOWN = "SHUTDOWN"


class RobotWorkerState(str, Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    FOLLOWING = "FOLLOWING"
    GRASPING = "GRASPING"
    RETURNING = "RETURNING"
    PLACING = "PLACING"
    RESETTING = "RESETTING"
    DONE = "DONE"
    ERROR = "ERROR"
    STOPPING = "STOPPING"


@dataclass(frozen=True)
class RobotRequest:
    type: RobotRequestType | str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def request_type(self) -> RobotRequestType:
        return self.type if isinstance(self.type, RobotRequestType) else RobotRequestType(str(self.type))


@dataclass(frozen=True)
class RobotWorkerStatus:
    state: RobotWorkerState = RobotWorkerState.IDLE
    last_error: str | None = None
    is_connected: bool = False
    using_mock: bool = False
    active_request: RobotRequestType | None = None
    last_robot_pose: tuple[float, float, float, float, float, float] | None = None
    last_robot_state: Any | None = None
    last_command_type: str | None = None
    grasp_ok: bool | None = None
    task_done: bool = False
    reset_done: bool = False


@dataclass
class RobotActionCallbacks:
    init_rtde: Callable[..., Any]
    move_robot_to_home_pose: Callable[..., Any]
    send_robot_command: Callable[..., Any]
    safe_stop_rtde: Callable[..., Any]
    disconnect_rtde: Callable[..., Any]
    execute_gripper_close: Callable[..., bool]
    execute_gripper_open: Callable[..., bool]
    execute_return_and_place: Callable[..., bool]
    save_grasp_offset: Callable[..., bool]
    configure_gripper_position_threshold_from_geometry: Callable[..., Any]
    reset_gripper_position_threshold_to_config_default: Callable[..., Any]


class RobotWorker:
    """Own all RTDE/gripper commands from one background thread."""

    def __init__(
        self,
        *,
        args: Any,
        shared_state: Any,
        actions: RobotActionCallbacks,
        name: str = "robot-worker",
    ) -> None:
        self.args = args
        self.shared_state = shared_state
        self.actions = actions
        self.name = name
        self.controller = None

        self._requests: queue.Queue[RobotRequest] = queue.Queue()
        self._queued_types: set[RobotRequestType] = set()
        self._queued_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status = RobotWorkerStatus()
        self._cancel_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._last_sent_pose_mm: np.ndarray | None = None
        self._ref_target_xyz_mm: np.ndarray | None = None
        self._follow_was_active = False

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, request: RobotRequest | RobotRequestType | str, payload: dict[str, Any] | None = None) -> bool:
        if not isinstance(request, RobotRequest):
            request = RobotRequest(request, {} if payload is None else dict(payload))
        request_type = request.request_type()

        if request_type in {
            RobotRequestType.RESET_HOME,
            RobotRequestType.SAVE_AND_STOP,
            RobotRequestType.EMERGENCY_STOP,
            RobotRequestType.SHUTDOWN,
        }:
            self._cancel_event.set()

        if not self._can_accept(request_type):
            return False

        with self._queued_lock:
            if request_type in self._queued_types and request_type not in {
                RobotRequestType.EMERGENCY_STOP,
                RobotRequestType.SHUTDOWN,
            }:
                return False
            self._queued_types.add(request_type)
        self._requests.put(request)
        return True

    def get_status(self) -> RobotWorkerStatus:
        with self._status_lock:
            return replace(self._status)

    def wait_for_state(
        self,
        states: set[RobotWorkerState] | tuple[RobotWorkerState, ...] | list[RobotWorkerState],
        timeout_s: float,
    ) -> RobotWorkerStatus:
        wanted = set(states)
        deadline = time.time() + max(float(timeout_s), 0.0)
        while True:
            status = self.get_status()
            if status.state in wanted:
                return status
            if time.time() >= deadline:
                return status
            time.sleep(0.01)

    def acknowledge_reset_done(self) -> None:
        self._set_status(reset_done=False)

    def acknowledge_task_done(self) -> None:
        self._set_status(task_done=False)

    def _can_accept(self, request_type: RobotRequestType) -> bool:
        status = self.get_status()
        if request_type in {RobotRequestType.EMERGENCY_STOP, RobotRequestType.SHUTDOWN, RobotRequestType.RESET_HOME}:
            return True
        if request_type == RobotRequestType.INIT_ROBOT:
            return self.controller is None and status.state in {RobotWorkerState.IDLE, RobotWorkerState.ERROR}
        if request_type == RobotRequestType.START_GRASP_PLACE:
            return status.state in {RobotWorkerState.IDLE, RobotWorkerState.FOLLOWING, RobotWorkerState.DONE}
        if request_type == RobotRequestType.START_FOLLOW:
            return self.controller is not None and status.state in {
                RobotWorkerState.IDLE,
                RobotWorkerState.DONE,
                RobotWorkerState.FOLLOWING,
            }
        if request_type in {RobotRequestType.STOP_FOLLOW, RobotRequestType.SAVE_AND_STOP}:
            return True
        return True

    def _run(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                request = self._requests.get(timeout=self._follow_interval())
            except queue.Empty:
                self._tick_follow()
                self._refresh_robot_status()
                continue

            request_type = request.request_type()
            with self._queued_lock:
                self._queued_types.discard(request_type)
            try:
                self._handle_request(request)
            except Exception as exc:
                self._set_status(state=RobotWorkerState.ERROR, last_error=str(exc), active_request=None)
                print(f"[WARN] RobotWorker request {request_type.value} failed: {exc}", flush=True)

        self._safe_shutdown_controller()

    def _handle_request(self, request: RobotRequest) -> None:
        request_type = request.request_type()
        if request_type == RobotRequestType.INIT_ROBOT:
            self._handle_init(request)
        elif request_type == RobotRequestType.START_FOLLOW:
            self._cancel_event.clear()
            self.shared_state.clear_follow_pause()
            with self.shared_state.lock:
                self.shared_state.follow_enabled = True
            self._set_status(state=RobotWorkerState.FOLLOWING, active_request=None, task_done=False, reset_done=False)
        elif request_type == RobotRequestType.STOP_FOLLOW:
            self._stop_following("stop_follow")
            self._set_status(state=RobotWorkerState.IDLE, active_request=None)
        elif request_type == RobotRequestType.SAVE_AND_STOP:
            self._stop_following("save_and_stop")
            self._set_status(state=RobotWorkerState.IDLE, active_request=None)
        elif request_type == RobotRequestType.START_GRASP_PLACE:
            self._handle_grasp_place(request)
        elif request_type == RobotRequestType.RESET_HOME:
            self._handle_reset(request)
        elif request_type == RobotRequestType.EMERGENCY_STOP:
            self._handle_emergency_stop()
        elif request_type == RobotRequestType.SHUTDOWN:
            self._handle_shutdown()

    def _handle_init(self, request: RobotRequest) -> None:
        self._cancel_event.clear()
        self._set_status(state=RobotWorkerState.INITIALIZING, active_request=RobotRequestType.INIT_ROBOT, reset_done=False)
        self.controller = self.actions.init_rtde(self.args)
        self.actions.move_robot_to_home_pose(self.controller, self.args, cancel_event=self._cancel_event)
        self.shared_state.set_fixed_pose_from_robot(self.controller)
        self._refresh_robot_status()
        self._set_status(state=RobotWorkerState.IDLE, active_request=None)

    def _handle_grasp_place(self, request: RobotRequest) -> None:
        if self.controller is None:
            self._set_status(state=RobotWorkerState.ERROR, last_error="Robot is not initialized", active_request=None)
            return

        payload = dict(request.payload or {})
        self._cancel_event.clear()
        self._stop_following("grasp_place_start")
        with self.shared_state.lock:
            self.shared_state.pregrasp_started = True
            self.shared_state.follow_enabled = False

        tactile_manager = payload.get("tactile_manager")
        metadata_recorder = payload.get("metadata_recorder")
        if tactile_manager is not None and bool(getattr(tactile_manager, "enabled", False)):
            self.actions.reset_gripper_position_threshold_to_config_default(self.controller)
        else:
            self.actions.configure_gripper_position_threshold_from_geometry(
                self.controller,
                payload.get("fitted_points_base"),
                payload.get("grasp_point_base"),
                object_label=payload.get("object_label"),
                template_axes_base=payload.get("template_axes_base"),
            )

        self._set_status(
            state=RobotWorkerState.GRASPING,
            active_request=RobotRequestType.START_GRASP_PLACE,
            grasp_ok=None,
            task_done=False,
        )
        grasp_ok = self.actions.execute_gripper_close(
            self.controller,
            timeout_s=self.args.gripper_close_timeout_s,
            verbose=True,
            metadata_recorder=metadata_recorder,
            tactile_manager=tactile_manager,
            tactile_contact_threshold=getattr(self.args, "tactile_contact_norm_threshold", None),
            tactile_extra_grasp_pos=getattr(self.args, "tactile_extra_grasp_pos", None),
            cancel_event=self._cancel_event,
        )
        self._set_status(grasp_ok=bool(grasp_ok))
        if not grasp_ok or self._cancel_event.is_set():
            self._stop_following("grasp_cancel_or_fail")
            self._set_status(state=RobotWorkerState.DONE, active_request=None, task_done=True)
            return

        self.actions.save_grasp_offset(self.controller, self.shared_state)
        self._set_status(state=RobotWorkerState.RETURNING)
        place_ok = self.actions.execute_return_and_place(
            self.controller,
            self.shared_state,
            self.args,
            metadata_recorder=metadata_recorder,
            tactile_manager=tactile_manager,
            cancel_event=self._cancel_event,
        )
        if not place_ok and not self._cancel_event.is_set():
            self._set_status(state=RobotWorkerState.ERROR, last_error="Return/place failed", active_request=None, task_done=True)
            return
        self._set_status(state=RobotWorkerState.DONE, active_request=None, task_done=True)

    def _handle_reset(self, request: RobotRequest) -> None:
        if self.controller is None:
            self._set_status(state=RobotWorkerState.IDLE, active_request=None, reset_done=True)
            return
        self._set_status(state=RobotWorkerState.RESETTING, active_request=RobotRequestType.RESET_HOME, reset_done=False)
        self._cancel_event.set()
        self._stop_following("reset")
        self._cancel_event.clear()
        self.actions.execute_gripper_open(
            self.controller,
            dwell_s=getattr(self.args, "gripper_release_dwell_s", 0.0),
            cancel_event=self._cancel_event,
        )
        self.actions.move_robot_to_home_pose(self.controller, self.args, cancel_event=self._cancel_event)
        self.shared_state.reset_for_restart(follow_enabled=bool(getattr(self.args, "enable_follow", False)))
        self.shared_state.set_fixed_pose_from_robot(self.controller)
        if bool(getattr(self.args, "enable_follow", False)):
            state = RobotWorkerState.FOLLOWING
        else:
            state = RobotWorkerState.IDLE
        self._set_status(state=state, active_request=None, reset_done=True, task_done=False)

    def _handle_emergency_stop(self) -> None:
        self._set_status(state=RobotWorkerState.STOPPING, active_request=RobotRequestType.EMERGENCY_STOP)
        self._cancel_event.set()
        self._stop_following("emergency_stop")
        self._set_status(state=RobotWorkerState.IDLE, active_request=None)

    def _handle_shutdown(self) -> None:
        self._set_status(state=RobotWorkerState.STOPPING, active_request=RobotRequestType.SHUTDOWN)
        self._cancel_event.set()
        self._stop_following("shutdown")
        self._shutdown_event.set()

    def _stop_following(self, source_mode: str) -> None:
        try:
            self.shared_state.request_follow_pause()
            self.shared_state.stop_follow()
        except Exception:
            pass
        if self.controller is not None:
            self.actions.safe_stop_rtde(self.controller)
        self._follow_was_active = False
        self._last_sent_pose_mm = None
        self._ref_target_xyz_mm = None
        try:
            self.shared_state.set_follow_thread_idle(True)
        except Exception:
            pass

    def _tick_follow(self) -> None:
        if self.controller is None:
            return
        if self.get_status().state != RobotWorkerState.FOLLOWING:
            return

        start_t = time.time()
        snap = self.shared_state.get_snapshot()
        control_target_xyz_mm = snap["control_target_xyz_mm"]
        target_source = snap["target_source"]

        active = True
        if snap["follow_pause_requested"]:
            active = False
        elif not snap["follow_enabled"]:
            active = False
        elif control_target_xyz_mm is None:
            active = False
        elif target_source != "predicted" and snap["valid_detection_streak"] < self.args.min_valid_count and not snap["prediction_armed"]:
            active = False
        elif target_source == "predicted" and not snap["prediction_armed"]:
            active = False
        elif not snap["motion_triggered"]:
            active = False
        elif target_source == "predicted" and (
            snap["prediction_age_s"] is None
            or float(snap["prediction_age_s"]) > float(self.args.prediction_max_horizon_s)
        ):
            active = False
        elif snap["fixed_z_mm"] is None or snap["fixed_orientation_base"] is None:
            active = False

        if not active:
            if self._follow_was_active:
                self.actions.safe_stop_rtde(self.controller)
                self._ref_target_xyz_mm = None
                self._last_sent_pose_mm = None
            self._follow_was_active = False
            self.shared_state.set_follow_thread_idle(True)
            return

        self.shared_state.set_follow_thread_idle(False)
        if self._ref_target_xyz_mm is None:
            state = self.controller.read_robot_state(now_timestamp=time.time())
            pose = state.actual_tcp_pose_base
            if pose is None:
                return
            self._ref_target_xyz_mm = self._meters_to_mm(pose[:3]).astype(np.float32)
            print(f"[INFO] ref_target initialized from current EEF xyz: {self._ref_target_xyz_mm}")

        interval = self._follow_interval()
        max_step_mm = 200.0 / max(float(self.args.control_hz), 1e-6)
        max_step_z_mm = 250.0 / max(float(self.args.control_hz), 1e-6)
        fixed_z_mm = snap["fixed_z_mm"]
        fixed_orientation_base = snap["fixed_orientation_base"]
        ref_err_xyz = control_target_xyz_mm - self._ref_target_xyz_mm
        max_step_xy, max_step_z, dist_xy = self._get_close_range_step_mm(ref_err_xyz, max_step_mm, max_step_z_mm)
        ref_step_xyz, dominant_axis = self._compute_close_range_ref_step_xyz(
            ref_err_xyz,
            max_step_xy,
            max_step_z,
            follow_z=bool(self.args.follow_z),
        )
        if not self.args.follow_z:
            self._ref_target_xyz_mm[2] = fixed_z_mm
        self._ref_target_xyz_mm = self._ref_target_xyz_mm + ref_step_xyz
        cmd_z = float(self._ref_target_xyz_mm[2]) if self.args.follow_z else float(fixed_z_mm)
        cmd_x, cmd_y, cmd_z = self._clamp_pose_mm(
            float(self._ref_target_xyz_mm[0]),
            float(self._ref_target_xyz_mm[1]),
            cmd_z,
        )
        pose_mm = np.array([cmd_x, cmd_y, cmd_z], dtype=np.float32)
        if self._last_sent_pose_mm is not None:
            pos_delta = np.linalg.norm(pose_mm - self._last_sent_pose_mm)
            if pos_delta < 0.2:
                return

        target_position_base = self._mm_to_m_tuple(pose_mm)
        try:
            state = self.actions.send_robot_command(
                self.controller,
                ROBOT_CMD_SERVO_TO_POSITION,
                target_position_base=target_position_base,
                fixed_orientation_base=fixed_orientation_base,
                gripper_action=GRIPPER_HOLD,
                source_mode="follow_servo",
            )
            self._follow_was_active = True
            self._last_sent_pose_mm = pose_mm
            self._publish_robot_state(state)
            if getattr(self.args, "verbose_robot", False):
                print(
                    f"[ROBOT] source={target_source}, raw_mm={snap['latest_target_xyz_mm']}, "
                    f"control_mm={control_target_xyz_mm}, ref_mm={self._ref_target_xyz_mm}, "
                    f"close_range_dist_xy={dist_xy:.1f}, "
                    f"stage_axis={'xy' if dominant_axis is None else ('x' if dominant_axis == 0 else 'y')}, "
                    f"cmd_m={target_position_base}"
                )
        except Exception as exc:
            self._set_status(last_error=f"servo command failed: {exc}")
            print(f"[WARN] servo command failed: {exc}")

        elapsed = time.time() - start_t
        if elapsed < interval:
            time.sleep(max(0.0, interval - elapsed))

    def _follow_interval(self) -> float:
        return 1.0 / max(float(getattr(self.args, "control_hz", 30.0)), 1e-6)

    def _refresh_robot_status(self) -> None:
        if self.controller is None:
            return
        try:
            state = self.controller.read_robot_state(now_timestamp=time.time())
            self._publish_robot_state(state)
        except Exception as exc:
            self._set_status(last_error=str(exc))

    def _publish_robot_state(self, state: Any) -> None:
        if state is None:
            return
        pose = getattr(state, "actual_tcp_pose_base", None)
        pose_tuple = None if pose is None else tuple(float(v) for v in pose[:6])
        debug = getattr(self.controller, "last_debug", None)
        last_command_type = getattr(debug, "last_command_type", None)
        self._set_status(
            is_connected=bool(getattr(state, "is_connected", False)),
            using_mock=bool(getattr(self.controller, "using_mock", False)),
            last_robot_pose=pose_tuple,
            last_robot_state=state,
            last_command_type=last_command_type,
        )

    def _safe_shutdown_controller(self) -> None:
        if self.controller is None:
            return
        try:
            self.actions.safe_stop_rtde(self.controller)
        finally:
            self.actions.disconnect_rtde(self.controller)
            self.controller = None
            self._set_status(is_connected=False, active_request=None)

    def _set_status(self, **changes: Any) -> None:
        with self._status_lock:
            self._status = replace(self._status, **changes)

    @staticmethod
    def _meters_to_mm(values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)[:3] * 1000.0

    @staticmethod
    def _mm_to_m_tuple(values: Any) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=np.float32).reshape(3) / 1000.0
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    def _clamp_pose_mm(self, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float, float]:
        return (
            float(np.clip(x_mm, self.args.workspace_x[0], self.args.workspace_x[1])),
            float(np.clip(y_mm, self.args.workspace_y[0], self.args.workspace_y[1])),
            float(np.clip(z_mm, self.args.workspace_z[0], self.args.workspace_z[1])),
        )

    @staticmethod
    def _get_close_range_step_mm(ref_err_xyz: Any, max_step_mm: float, max_step_z_mm: float) -> tuple[float, float, float]:
        ref_err_xyz = np.asarray(ref_err_xyz, dtype=np.float32).reshape(3)
        dist_xy = float(np.linalg.norm(ref_err_xyz[:2]))
        if dist_xy < 35.0:
            return 1.5, 1.0, dist_xy
        if dist_xy < 65.0:
            return 3.0, 1.5, dist_xy
        if dist_xy < 95.0:
            return 5.0, 2.2, dist_xy
        return float(max_step_mm), float(max_step_z_mm), dist_xy

    @staticmethod
    def _compute_close_range_ref_step_xyz(
        ref_err_xyz: Any,
        max_step_xy: float,
        max_step_z: float,
        *,
        follow_z: bool,
    ) -> tuple[np.ndarray, int | None]:
        ref_err_xyz = np.asarray(ref_err_xyz, dtype=np.float32).reshape(3)
        ref_step_xyz = np.zeros(3, dtype=np.float32)
        dist_xy = float(np.linalg.norm(ref_err_xyz[:2]))
        dominant_axis = None
        if dist_xy < 65.0 and abs(float(ref_err_xyz[0])) > 15.0 and abs(float(ref_err_xyz[1])) > 15.0:
            dominant_axis = 0 if abs(float(ref_err_xyz[0])) >= abs(float(ref_err_xyz[1])) else 1
            ref_step_xyz[dominant_axis] = np.clip(ref_err_xyz[dominant_axis], -max_step_xy, max_step_xy)
        else:
            ref_step_xyz[0:2] = np.clip(ref_err_xyz[0:2], -max_step_xy, max_step_xy)
        if follow_z:
            ref_step_xyz[2] = np.clip(ref_err_xyz[2], -max_step_z, max_step_z)
        return ref_step_xyz, dominant_axis


__all__ = [
    "RobotActionCallbacks",
    "RobotRequest",
    "RobotRequestType",
    "RobotWorker",
    "RobotWorkerState",
    "RobotWorkerStatus",
]
