"""ROS2 wrapper around the existing DualSensorHub."""

from __future__ import annotations

import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from .conversions import camera_info_from_intrinsics, image_msg_from_array, sensor_sync_status_msg
from .repo import ensure_repo_on_path, resolve_repo_path


class DualCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("dual_camera_node")
        self.declare_parameter("repo_root", "")
        self.declare_parameter("config_path", "configs/handover.yaml")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("start_on_init", True)
        self.declare_parameter("frame_prefix", "")

        repo_root_param = self.get_parameter("repo_root").get_parameter_value().string_value
        self.repo_root = ensure_repo_on_path(repo_root_param or None)
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self.config_path = str(resolve_repo_path(self.repo_root, config_path))

        from system.dual_sensor_hub import DualSensorHub

        self.sensor_hub = DualSensorHub.from_config(self.config_path)
        self.sensor_hub.width = int(self.get_parameter("width").value)
        self.sensor_hub.height = int(self.get_parameter("height").value)
        self.sensor_hub.fps = int(self.get_parameter("fps").value)
        self._started = False
        self._last_start_error = ""

        qos_depth = 5
        self.cam0_color_pub = self.create_publisher(Image, "/cam0/color/image_raw", qos_depth)
        self.cam0_depth_pub = self.create_publisher(Image, "/cam0/depth/image_raw", qos_depth)
        self.cam0_info_pub = self.create_publisher(CameraInfo, "/cam0/camera_info", qos_depth)
        self.cam1_color_pub = self.create_publisher(Image, "/cam1/color/image_raw", qos_depth)
        self.cam1_depth_pub = self.create_publisher(Image, "/cam1/depth/image_raw", qos_depth)
        self.cam1_info_pub = self.create_publisher(CameraInfo, "/cam1/camera_info", qos_depth)

        from icra_handover_interfaces.msg import SensorSyncStatus

        self.sync_pub = self.create_publisher(SensorSyncStatus, "/camera_pair/sync_status", qos_depth)

        if bool(self.get_parameter("start_on_init").value):
            self._ensure_started()

        rate_hz = max(float(self.get_parameter("publish_rate_hz").value), 0.1)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f"dual_camera_node ready, config={self.config_path}")

    def _ensure_started(self) -> bool:
        if self._started:
            return True
        try:
            self.sensor_hub.start()
            self._started = True
            self._last_start_error = ""
            self.get_logger().info(
                f"DualSensorHub started: cam0={self.sensor_hub.serial_cam0}, cam1={self.sensor_hub.serial_cam1}"
            )
            return True
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message != self._last_start_error:
                self.get_logger().warning(f"DualSensorHub is not available yet: {message}")
                self.get_logger().debug(traceback.format_exc())
                self._last_start_error = message
            return False

    def _header(self, frame_id: str) -> Header:
        prefix = self.get_parameter("frame_prefix").get_parameter_value().string_value
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f"{prefix}{frame_id}"
        return header

    def _publish_camera(self, *, camera_name: str, frame, color_pub, depth_pub, info_pub) -> None:
        header = self._header(camera_name)
        color_pub.publish(image_msg_from_array(np.asarray(frame.color_image), header, "bgr8"))
        depth_pub.publish(image_msg_from_array(np.asarray(frame.depth_image_m, dtype=np.float32), header, "32FC1"))
        info_pub.publish(
            camera_info_from_intrinsics(
                frame.intrinsics or {},
                header,
                width=int(frame.color_image.shape[1]),
                height=int(frame.color_image.shape[0]),
            )
        )

    def _tick(self) -> None:
        if not self._ensure_started():
            return
        try:
            snapshot = self.sensor_hub.read_next_pair()
        except Exception as exc:
            self.get_logger().warning(f"Failed to read paired frames: {exc}")
            return

        self._publish_camera(
            camera_name="cam0",
            frame=snapshot.cam0,
            color_pub=self.cam0_color_pub,
            depth_pub=self.cam0_depth_pub,
            info_pub=self.cam0_info_pub,
        )
        self._publish_camera(
            camera_name="cam1",
            frame=snapshot.cam1,
            color_pub=self.cam1_color_pub,
            depth_pub=self.cam1_depth_pub,
            info_pub=self.cam1_info_pub,
        )
        self.sync_pub.publish(sensor_sync_status_msg(snapshot, self._header("camera_pair")))

    def destroy_node(self) -> bool:
        try:
            self.sensor_hub.stop()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DualCameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
