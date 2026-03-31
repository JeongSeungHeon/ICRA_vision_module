"""Single-process runner that wires perception, task logic, safety, and RTDE together."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml
import cv2 as cv
import numpy as np

from calibration.extrinsics import load_transform_chain
from perception.fusion import PerceptionFusion
from perception.grasp_target import GraspTargetPlanner
from perception.hand_selector import HandSelector
from robot.live_follow_controller import LiveFollowController
from perception.hand_worker import HandWorkerCam0, HandWorkerCam1
from perception.object_merger import ObjectMerger
from perception.object_worker import ObjectWorkerCam0, ObjectWorkerCam1
from robot.rtde_controller import RtdeController
from robot.safety import SafetyValidator
from system.dual_sensor_hub import DualSensorHub
from system.shared_state import RobotState, SharedStateBundle
from system.task_manager import TaskManager

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


class HandoverSystemRunner:
    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        force_mock: bool = False,
        disable_robot: bool = False,
        log_every_sec: Optional[float] = None,
        show_cam0_target: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

        system_cfg = self.config.get("system", {})
        self.log_every_sec = float(
            log_every_sec if log_every_sec is not None else system_cfg.get("runner_heartbeat_sec", 1.0)
        )
        self.loop_sleep_sec = float(system_cfg.get("runner_loop_sleep_sec", 0.0))
        self.force_mock = bool(force_mock)
        self.disable_robot = bool(disable_robot)
        self.show_cam0_target = bool(show_cam0_target or system_cfg.get("runner_show_cam0_target", False))

        self.sensor_hub = DualSensorHub.from_config(self.config_path)
        self.object_worker_cam0 = ObjectWorkerCam0.from_config(self.config_path)
        self.object_worker_cam1 = ObjectWorkerCam1.from_config(self.config_path)
        self.object_merger = ObjectMerger.from_config(self.config_path)
        self.hand_worker_cam0 = HandWorkerCam0.from_config(self.config_path)
        self.hand_worker_cam1 = HandWorkerCam1.from_config(self.config_path)
        self.hand_selector = HandSelector.from_config(self.config_path)
        self.fusion = PerceptionFusion.from_config(self.config_path)
        self.grasp_planner = GraspTargetPlanner.from_config(self.config_path)
        self.task_manager = TaskManager.from_config(self.config_path)
        self.live_follow_controller = LiveFollowController.from_config(self.config_path)
        self.safety = SafetyValidator.from_config(self.config_path)
        self.rtde_controller = RtdeController.from_config(self.config_path)
        self.transform_chain = load_transform_chain(self.config_path)
        self._t_cam0_base = np.linalg.inv(self.transform_chain.t_base_cam0).astype(np.float32)
        self._cam0_window_name = "cam0 robot target debug"
        if self.force_mock:
            self.rtde_controller.force_mock = True

        self.shared_state = SharedStateBundle()
        self._previous_hand_approach = False
        self._last_heartbeat_at = 0.0
        self._running = False
        self._loop_index = 0
        self._last_step_time: float = 0.0  # wall clock of previous step start

    def start(self) -> "HandoverSystemRunner":
        self.sensor_hub.start()
        if not self.disable_robot:
            self.rtde_controller.connect()
            # Open gripper first so we start with a known gripper state.
            self.rtde_controller._apply_gripper_action("open")
            # Move to home pose (blocking — waits until robot arrives).
            self.rtde_controller.move_home()
            self.shared_state.robot = self.rtde_controller.read_robot_state(now_timestamp=time.time())
        else:
            self.shared_state.robot = RobotState(is_connected=False, robot_mode="disabled", valid=True, timestamp=time.time())
        if self.show_cam0_target:
            cv.namedWindow(self._cam0_window_name, cv.WINDOW_NORMAL)
        self._running = True
        self._last_heartbeat_at = 0.0
        return self

    def stop(self) -> None:
        self._running = False
        try:
            self.hand_worker_cam0.close()
        finally:
            try:
                self.hand_worker_cam1.close()
            finally:
                try:
                    self.sensor_hub.stop()
                finally:
                    if not self.disable_robot:
                        self.rtde_controller.close()
        if self.show_cam0_target:
            try:
                cv.destroyWindow(self._cam0_window_name)
            except Exception:
                pass

    def step(self) -> SharedStateBundle:
        # Use wall clock as the authoritative timestamp so all modules share
        # the same epoch.  Camera hardware timestamps use a different epoch
        # (often device-boot-relative) and directly comparing them with
        # Python's time.time() causes watchdog / stale-perception false-trips.
        step_start = time.time()
        loop_dt = max(step_start - self._last_step_time, 1.0 / 125.0) if self._last_step_time > 0.0 else (1.0 / 30.0)
        self._last_step_time = step_start
        current_time = step_start

        snapshot = self.sensor_hub.read_next_pair()

        self.shared_state.sensor_cam0, self.shared_state.sensor_cam1 = self.sensor_hub.get_latest_sensor_states()
        self.shared_state.object_cam0 = self.object_worker_cam0.process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
        self.shared_state.object_cam1 = self.object_worker_cam1.process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
        self.shared_state.hand_cam0 = self.hand_worker_cam0.process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
        self.shared_state.hand_cam1 = self.hand_worker_cam1.process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
        self.shared_state.selected_hand = self.hand_selector.process_states(
            self.shared_state.hand_cam0,
            self.shared_state.hand_cam1,
        )
        self.shared_state.merged_object = self.object_merger.process_states(
            self.shared_state.object_cam0,
            self.shared_state.object_cam1,
            hand_approach_detected=self._previous_hand_approach,
        )
        self.shared_state.fusion = self.fusion.process_states(
            self.shared_state.merged_object,
            self.shared_state.selected_hand,
            now_timestamp=current_time,
        )
        self.shared_state.grasp_target = self.grasp_planner.process_states(
            self.shared_state.merged_object,
            self.shared_state.selected_hand,
            self.shared_state.fusion,
        )
        self.shared_state.live_follow = self.live_follow_controller.process_states(
            self.shared_state.grasp_target,
            self.shared_state.fusion,
            self.shared_state.robot,
            source_mode=self.shared_state.task.mode,
            now_timestamp=current_time,
        )
        self._previous_hand_approach = bool(
            self.shared_state.fusion.hand_approach_detected or self.shared_state.fusion.hand_approach_latched
        )

        task_state, robot_command = self.task_manager.process_states(
            self.shared_state.merged_object,
            self.shared_state.selected_hand,
            self.shared_state.grasp_target,
            self.shared_state.robot,
            fusion_state=self.shared_state.fusion,
            live_follow_state=self.shared_state.live_follow,
            safety_ok=True,
            now_timestamp=current_time,
        )
        safety_result = self.safety.validate(
            task_state,
            robot_command,
            self.shared_state.merged_object,
            self.shared_state.selected_hand,
            self.shared_state.grasp_target,
            self.shared_state.robot,
            fusion_state=self.shared_state.fusion,
            now_timestamp=current_time,
        )
        self.shared_state.task = task_state.copy_with(
            safety_ok=bool(safety_result.safe),
            timestamp=current_time,
            valid=True,
        )
        self.shared_state.robot_command = safety_result.sanitized_command.copy_with(
            timestamp=current_time,
            valid=True,
        )

        if not self.disable_robot:
            self.shared_state.robot = self.rtde_controller.step(
                self.shared_state.robot_command,
                now_timestamp=current_time,
                loop_dt=loop_dt,
            )
        else:
            self.shared_state.robot = self.shared_state.robot.copy_with(timestamp=current_time, valid=True)

        if self.show_cam0_target:
            self._render_cam0_target_debug(snapshot.cam0)

        self.shared_state.timestamp = current_time
        self.shared_state.valid = True
        self._loop_index += 1
        self._maybe_log_heartbeat(safety_result=safety_result)
        if self.loop_sleep_sec > 0.0:
            time.sleep(self.loop_sleep_sec)
        return self.shared_state

    def _render_cam0_target_debug(self, frame_bundle) -> None:
        image_bgr = np.asarray(frame_bundle.color_image).copy()

        object_debug = getattr(self.object_worker_cam0, "last_debug", None)
        combined_mask = getattr(object_debug, "combined_mask", None)
        if combined_mask is not None:
            mask_bool = np.asarray(combined_mask, dtype=bool)
            if mask_bool.shape[:2] == image_bgr.shape[:2]:
                overlay = np.zeros_like(image_bgr, dtype=np.uint8)
                overlay[mask_bool] = np.array([0, 180, 255], dtype=np.uint8)
                image_bgr = cv.addWeighted(image_bgr, 1.0, overlay, 0.35, 0.0)

        lines = []
        lines.append(f"mode={self.shared_state.task.mode} live_follow={self.shared_state.live_follow.reason}")
        lines.append(f"cmd={self.shared_state.robot_command.command_type}")
        lines.append(f"grasp_target={self._format_vec3(self.shared_state.grasp_target.target_position_base)}")
        lines.append(f"follow_target={self._format_vec3(self.shared_state.live_follow.servo_target_position_base)}")
        lines.append(f"tcp={self._format_vec3(self._tcp_position())}")

        draw_specs = [
            (self.shared_state.grasp_target.target_position_base, (0, 255, 0), "grasp"),
            (self.shared_state.live_follow.servo_target_position_base, (0, 255, 255), "follow"),
            (self._tcp_position(), (255, 0, 0), "tcp"),
        ]

        for point_base, color_bgr, label in draw_specs:
            pixel = self._project_base_point_to_cam0(point_base, frame_bundle.intrinsics, image_bgr.shape[1], image_bgr.shape[0])
            if pixel is None:
                continue
            cv.circle(image_bgr, pixel, 6, color_bgr, -1, cv.LINE_AA)
            cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv.LINE_AA)

        y = 24
        for line in lines:
            cv.putText(image_bgr, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv.LINE_AA)
            cv.putText(image_bgr, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv.LINE_AA)
            y += 22

        cv.imshow(self._cam0_window_name, image_bgr)
        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            self._running = False

    def _project_base_point_to_cam0(self, point_base, intrinsics, width: int, height: int):
        if point_base is None or intrinsics is None:
            return None
        point = np.asarray(point_base, dtype=np.float32).reshape(3)
        point_h = np.concatenate([point, np.array([1.0], dtype=np.float32)], axis=0)
        point_cam = (self._t_cam0_base @ point_h.reshape(4, 1)).reshape(-1)[:3]
        z = float(point_cam[2])
        if not np.isfinite(z) or z <= 1e-6:
            return None
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        u = int(round((float(point_cam[0]) * fx / z) + cx))
        v = int(round((float(point_cam[1]) * fy / z) + cy))
        if u < 0 or u >= int(width) or v < 0 or v >= int(height):
            return None
        return (u, v)

    def _tcp_position(self):
        if self.shared_state.robot.actual_tcp_pose_base is None:
            return None
        return tuple(float(v) for v in self.shared_state.robot.actual_tcp_pose_base[:3])

    @staticmethod
    def _format_vec3(values) -> str:
        if values is None:
            return "None"
        vec = tuple(float(v) for v in values[:3])
        return "(%.3f, %.3f, %.3f)" % vec

    def run(self, *, max_loops: int = 0) -> SharedStateBundle:
        self.start()
        try:
            while self._running:
                if max_loops > 0 and self._loop_index >= max_loops:
                    break
                self.step()
            return self.shared_state
        finally:
            self.stop()

    def _maybe_log_heartbeat(self, *, safety_result) -> None:
        current_time = float(self.shared_state.timestamp)
        if self.log_every_sec <= 0.0:
            return
        if (current_time - self._last_heartbeat_at) < self.log_every_sec:
            return
        self._last_heartbeat_at = current_time

        task_mode = self.shared_state.task.mode
        selected = "none"
        hand_selection_reason = "none"
        if self.shared_state.selected_hand.valid:
            selected = "cam%s/%s" % (
                self.shared_state.selected_hand.selected_camera,
                self.shared_state.selected_hand.handedness,
            )
        if self.hand_selector.last_debug is not None:
            hand_selection_reason = str(self.hand_selector.last_debug.selection_reason)
        target = self.shared_state.robot_command.target_position_base
        target_str = "None" if target is None else "(%.3f, %.3f, %.3f)" % target
        raw_target = self.shared_state.task.target_position_base
        raw_target_str = "None" if raw_target is None else "(%.3f, %.3f, %.3f)" % raw_target
        hand_dist = self.shared_state.fusion.hand_object_distance_m
        hand_dist_str = "None" if hand_dist is None else "%.3f" % hand_dist
        lift_str = "%.3f" % float(self.shared_state.fusion.lift_height_delta_m)
        live_follow_reason = str(self.shared_state.live_follow.reason)
        follow_target = self.shared_state.live_follow.servo_target_position_base
        follow_target_str = "None" if follow_target is None else "(%.3f, %.3f, %.3f)" % follow_target
        print(
            "[runner %05d] mode=%s task_reason=%s merged_pts=%d selected=%s hand_sel_reason=%s grasp=%s hand_dist=%s approach=%s lift=%s lift_delta=%s activation=%s live_follow=%s follow_target=%s safe=%s reason=%s cmd=%s target=%s raw_target=%s robot=%s robot_err=%s" % (
                self._loop_index,
                task_mode,
                str(self.shared_state.task.active_reason),
                int(self.shared_state.merged_object.merged_point_count),
                selected,
                hand_selection_reason,
                str(bool(self.shared_state.grasp_target.valid)),
                hand_dist_str,
                str(bool(self.shared_state.fusion.hand_approach_detected or self.shared_state.fusion.hand_approach_latched)),
                str(bool(self.shared_state.fusion.object_lifted)),
                lift_str,
                str(bool(self.shared_state.fusion.robot_activation_ready)),
                live_follow_reason,
                follow_target_str,
                str(bool(safety_result.safe)),
                str(safety_result.reason),
                self.shared_state.robot_command.command_type,
                target_str,
                raw_target_str,
                self.shared_state.robot.robot_mode,
                str(self.shared_state.robot.last_error),
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full handover stack in one process.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/handover.yaml")
    parser.add_argument("--max-loops", type=int, default=0, help="Stop after N loops. 0 means run until interrupted.")
    parser.add_argument("--log-every-sec", type=float, default=None, help="Heartbeat logging interval in seconds.")
    parser.add_argument("--force-mock", action="store_true", help="Force RTDE controller into mock mode.")
    parser.add_argument("--disable-robot", action="store_true", help="Skip RTDE control and only run perception/task/safety.")
    parser.add_argument("--show-cam0-target", action="store_true", help="Show a live cam0 2D debug window with projected robot targets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = HandoverSystemRunner(
        config_path=args.config,
        force_mock=args.force_mock,
        disable_robot=args.disable_robot,
        log_every_sec=args.log_every_sec,
        show_cam0_target=args.show_cam0_target,
    )

    stop_requested = {"value": False}

    def _request_stop(signum, frame):  # pragma: no cover - signal handler
        stop_requested["value"] = True
        runner._running = False

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        runner.start()
        while runner._running:
            if stop_requested["value"]:
                break
            if args.max_loops > 0 and runner._loop_index >= args.max_loops:
                break
            runner.step()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
