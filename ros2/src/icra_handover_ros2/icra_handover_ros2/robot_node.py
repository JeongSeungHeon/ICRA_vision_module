"""ROS2 adapter for the existing RobotWorker and RTDE controller."""

from __future__ import annotations

import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header

from icra_handover_interfaces.action import ExecuteHandover
from icra_handover_interfaces.msg import RobotStatus as RobotStatusMsg
from icra_handover_interfaces.srv import EmergencyStop, ResetHome, SetFollow

from .conversions import robot_status_msg
from .live_args import build_runtime_args
from .repo import ensure_repo_on_path


class RobotNode(Node):
    def __init__(self) -> None:
        super().__init__("robot_node")
        self.declare_parameter("repo_root", "")
        self.declare_parameter("config_path", "configs/handover.yaml")
        self.declare_parameter("enable_follow", True)
        self.declare_parameter("auto_init", False)
        self.declare_parameter("auto_start_follow", False)
        self.declare_parameter("robot_ip", "")
        self.declare_parameter("status_rate_hz", 20.0)

        repo_root_param = self.get_parameter("repo_root").get_parameter_value().string_value
        self.repo_root = ensure_repo_on_path(repo_root_param or None)
        self.args = build_runtime_args(
            repo_root=str(self.repo_root),
            config_path=self.get_parameter("config_path").get_parameter_value().string_value,
            overrides={
                "enable_follow": bool(self.get_parameter("enable_follow").value),
                "robot_ip": self._optional_string_param("robot_ip"),
            },
        )

        from robot_control_rtde_fitting_final import FollowSharedState, RobotStatus, RobotWorker

        self._root = __import__("robot_control_rtde_fitting_final")
        self.shared_state = FollowSharedState(self.args)
        self.worker = RobotWorker(self.args, self.shared_state, metadata_recorder=None, tactile_manager=None)
        self.worker.start()
        self._fallback_status = RobotStatus()

        qos_depth = 10
        self.status_pub = self.create_publisher(RobotStatusMsg, "/robot/status", qos_depth)
        self.create_service(SetFollow, "/robot/set_follow", self._set_follow)
        self.create_service(ResetHome, "/robot/reset_home", self._reset_home)
        self.create_service(EmergencyStop, "/robot/emergency_stop", self._emergency_stop)
        self.action_server = ActionServer(
            self,
            ExecuteHandover,
            "/robot/execute_handover",
            execute_callback=self._execute_handover,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )

        rate_hz = max(float(self.get_parameter("status_rate_hz").value), 0.5)
        self.timer = self.create_timer(1.0 / rate_hz, self._publish_status)

        if bool(self.get_parameter("auto_init").value):
            self._submit(self._root.ROBOT_REQ_INIT_ROBOT)
            if bool(self.get_parameter("auto_start_follow").value):
                self._submit(self._root.ROBOT_REQ_START_FOLLOW)

        self.get_logger().info("robot_node ready. auto_init is disabled by default for hardware safety.")

    def _optional_string_param(self, name: str) -> str | None:
        value = self.get_parameter(name).get_parameter_value().string_value
        return value or None

    def _header(self) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "robot_base"
        return header

    def _status(self):
        if self.worker is None:
            return self._fallback_status
        return self.worker.get_status()

    def _publish_status(self) -> None:
        self.status_pub.publish(robot_status_msg(self._status(), self._header()))

    def _submit(self, request_type: str, payload: dict | None = None) -> int:
        request = self._root.RobotRequest(request_type, payload=dict(payload or {}))
        request_id = int(self.worker.submit(request))
        self.get_logger().info(f"submitted robot request {request_type} id={request_id}")
        return request_id

    def _set_follow(self, request, response):
        request_type = self._root.ROBOT_REQ_START_FOLLOW if request.enable else self._root.ROBOT_REQ_STOP_FOLLOW
        self._submit(request_type)
        status = self._status()
        response.accepted = True
        response.state = str(status.state)
        response.error = "" if status.last_error is None else str(status.last_error)
        return response

    def _reset_home(self, request, response):
        self.args.enable_follow = bool(request.restart_follow)
        self._submit(self._root.ROBOT_REQ_RESET_HOME)
        status = self._status()
        response.accepted = True
        response.state = str(status.state)
        response.error = "" if status.last_error is None else str(status.last_error)
        return response

    def _emergency_stop(self, request, response):
        self.get_logger().warning(f"emergency stop requested: {request.reason}")
        self._submit(self._root.ROBOT_REQ_EMERGENCY_STOP)
        status = self._status()
        response.accepted = True
        response.state = str(status.state)
        response.error = "" if status.last_error is None else str(status.last_error)
        return response

    def _goal_callback(self, goal_request):
        del goal_request
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        del goal_handle
        self._submit(self._root.ROBOT_REQ_EMERGENCY_STOP)
        return CancelResponse.ACCEPT

    def _execute_handover(self, goal_handle):
        goal = goal_handle.request
        if goal.reset_before_start:
            self._submit(self._root.ROBOT_REQ_RESET_HOME)
        if goal.init_robot:
            self._submit(self._root.ROBOT_REQ_INIT_ROBOT)
        if goal.start_follow:
            self._submit(self._root.ROBOT_REQ_START_FOLLOW)
        if goal.start_grasp_place:
            self._submit(self._root.ROBOT_REQ_START_GRASP_PLACE)

        timeout_s = float(goal.timeout_s)
        deadline = time.monotonic() + timeout_s if timeout_s > 0.0 else time.monotonic()
        result = ExecuteHandover.Result()

        while time.monotonic() <= deadline:
            if goal_handle.is_cancel_requested:
                self._submit(self._root.ROBOT_REQ_EMERGENCY_STOP)
                goal_handle.canceled()
                result.success = False
                result.final_state = str(self._status().state)
                result.error = "cancelled"
                return result

            status = self._status()
            feedback = ExecuteHandover.Feedback()
            feedback.state = str(status.state)
            feedback.active_request = "" if status.active_request is None else str(status.active_request)
            feedback.active_request_id = int(status.active_request_id)
            goal_handle.publish_feedback(feedback)

            if status.state in {self._root.ROBOT_STATE_DONE, self._root.ROBOT_STATE_ERROR}:
                break
            time.sleep(0.05)

        status = self._status()
        result.success = bool(status.state != self._root.ROBOT_STATE_ERROR)
        result.final_state = str(status.state)
        result.error = "" if status.last_error is None else str(status.last_error)
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def destroy_node(self) -> bool:
        try:
            if self.worker is not None:
                self._submit(self._root.ROBOT_REQ_SHUTDOWN)
                self.worker.join(timeout=2.0)
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
