"""RViz marker publisher for the ROS2 handover adapter."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from icra_handover_interfaces.msg import GraspTarget, RobotStatus


def _finite_point(point: Point) -> bool:
    return all(math.isfinite(float(v)) for v in (point.x, point.y, point.z))


class VisualizationNode(Node):
    def __init__(self) -> None:
        super().__init__("visualization_node")
        self.declare_parameter("marker_rate_hz", 10.0)
        self.declare_parameter("frame_id", "robot_base")

        self.latest_grasp_target: GraspTarget | None = None
        self.latest_robot_status: RobotStatus | None = None

        qos_depth = 10
        self.create_subscription(GraspTarget, "/perception/grasp_target", self._on_grasp_target, qos_depth)
        self.create_subscription(RobotStatus, "/robot/status", self._on_robot_status, qos_depth)
        self.marker_pub = self.create_publisher(MarkerArray, "/handover/markers", qos_depth)

        rate_hz = max(float(self.get_parameter("marker_rate_hz").value), 0.5)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info("visualization_node ready")

    def _on_grasp_target(self, msg: GraspTarget) -> None:
        self.latest_grasp_target = msg

    def _on_robot_status(self, msg: RobotStatus) -> None:
        self.latest_robot_status = msg

    def _marker(self, marker_id: int, name: str, point: Point, color: tuple[float, float, float, float], scale: float) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        marker.ns = name
        marker.id = int(marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def _delete_all(self) -> MarkerArray:
        marker = Marker()
        marker.action = Marker.DELETEALL
        return MarkerArray(markers=[marker])

    def _tick(self) -> None:
        markers = []
        if self.latest_grasp_target is not None:
            if _finite_point(self.latest_grasp_target.object_point_base):
                markers.append(
                    self._marker(
                        1,
                        "object_centroid",
                        self.latest_grasp_target.object_point_base,
                        (0.0, 0.6, 1.0, 0.9),
                        0.035,
                    )
                )
            if _finite_point(self.latest_grasp_target.grasp_point_base):
                markers.append(
                    self._marker(
                        2,
                        "grasp_target",
                        self.latest_grasp_target.grasp_point_base,
                        (0.0, 1.0, 0.0, 0.95),
                        0.045,
                    )
                )

        if self.latest_robot_status is not None and _finite_point(self.latest_robot_status.tcp_pose_base.position):
            markers.append(
                self._marker(
                    3,
                    "tcp_pose",
                    self.latest_robot_status.tcp_pose_base.position,
                    (1.0, 0.5, 0.0, 0.95),
                    0.03,
                )
            )

        self.marker_pub.publish(MarkerArray(markers=markers) if markers else self._delete_all())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
