"""Lightweight ROS2 coordinator for handover task state aggregation."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from icra_handover_interfaces.msg import GraspTarget, HandoverTaskState, RobotStatus
from icra_handover_interfaces.srv import SetFollow

from .conversions import handover_task_state_msg


class HandoverCoordinatorNode(Node):
    def __init__(self) -> None:
        super().__init__("handover_coordinator_node")
        self.declare_parameter("state_rate_hz", 10.0)
        self.declare_parameter("auto_request_follow", False)

        self.latest_grasp_target: GraspTarget | None = None
        self.latest_robot_status: RobotStatus | None = None
        self.task_epoch = 0
        self._follow_request_sent = False

        qos_depth = 10
        self.create_subscription(GraspTarget, "/perception/grasp_target", self._on_grasp_target, qos_depth)
        self.create_subscription(RobotStatus, "/robot/status", self._on_robot_status, qos_depth)
        self.state_pub = self.create_publisher(HandoverTaskState, "/handover/task_state", qos_depth)
        self.set_follow_client = self.create_client(SetFollow, "/robot/set_follow")

        rate_hz = max(float(self.get_parameter("state_rate_hz").value), 0.5)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info("handover_coordinator_node ready")

    def _on_grasp_target(self, msg: GraspTarget) -> None:
        self.latest_grasp_target = msg

    def _on_robot_status(self, msg: RobotStatus) -> None:
        previous = self.latest_robot_status
        self.latest_robot_status = msg
        if previous is not None and msg.task_ready_epoch != previous.task_ready_epoch:
            self.task_epoch += 1
            self._follow_request_sent = False

    def _header(self) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "robot_base"
        return header

    def _maybe_request_follow(self) -> None:
        if self._follow_request_sent:
            return
        if not bool(self.get_parameter("auto_request_follow").value):
            return
        if self.latest_robot_status is None:
            return
        if not self.latest_robot_status.is_connected:
            return
        if not self.set_follow_client.service_is_ready():
            return

        request = SetFollow.Request()
        request.enable = True
        self.set_follow_client.call_async(request)
        self._follow_request_sent = True
        self.get_logger().info("requested robot follow via /robot/set_follow")

    def _tick(self) -> None:
        self._maybe_request_follow()
        robot_ready = bool(self.latest_robot_status and self.latest_robot_status.is_connected)
        target_valid = bool(self.latest_grasp_target and self.latest_grasp_target.valid)

        if self.latest_robot_status and self.latest_robot_status.state:
            state = self.latest_robot_status.state
        elif target_valid:
            state = "TARGET_READY"
        else:
            state = "WAITING"

        msg = handover_task_state_msg(
            self._header(),
            state=state,
            reason="ros2_coordinator_status",
            task_epoch=self.task_epoch,
            follow_enabled=bool(self.latest_robot_status and self.latest_robot_status.state == "FOLLOWING"),
            motion_triggered=False,
            home_object_locked=False,
            grasp_request_pending=False,
            perception_target_valid=target_valid,
            robot_ready=robot_ready,
            latest_object_base=None if self.latest_grasp_target is None else self.latest_grasp_target.object_point_base,
            latest_grasp_base=None if self.latest_grasp_target is None else self.latest_grasp_target.grasp_point_base,
        )
        self.state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandoverCoordinatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
