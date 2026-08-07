"""ROS2 perception adapter around the existing Python perception pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from .conversions import (
    fusion_state_msg,
    grasp_target_msg,
    hand_state_msg,
    object_state_msg,
    pointcloud2_from_xyz,
    shape_fit_state_msg,
)
from .live_args import build_runtime_args
from .repo import ensure_repo_on_path


def _intrinsics_from_camera_info(msg: CameraInfo | None) -> dict[str, float]:
    if msg is None:
        return {}
    return {
        "fx": float(msg.k[0]),
        "fy": float(msg.k[4]),
        "cx": float(msg.k[2]),
        "cy": float(msg.k[5]),
    }


def _array_from_image(msg: Image, *, depth_scale: float = 0.001) -> np.ndarray:
    if msg.encoding in {"bgr8", "rgb8"}:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == "rgb8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width)).copy()
    if msg.encoding in {"16UC1", "mono16"}:
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32)
        return arr * float(depth_scale)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("repo_root", "")
        self.declare_parameter("config_path", "configs/handover.yaml")
        self.declare_parameter("use_local_sensor_hub", False)
        self.declare_parameter("process_rate_hz", 10.0)
        self.declare_parameter("base_frame", "robot_base")
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("model", "yoloe-26l-seg.pt")
        self.declare_parameter("select_mode", "highest_score")
        self.declare_parameter("device", "")
        self.declare_parameter("enable_follow_args", False)

        repo_root_param = self.get_parameter("repo_root").get_parameter_value().string_value
        self.repo_root = ensure_repo_on_path(repo_root_param or None)
        self.args = build_runtime_args(
            repo_root=str(self.repo_root),
            config_path=self.get_parameter("config_path").get_parameter_value().string_value,
            overrides={
                "model": self.get_parameter("model").get_parameter_value().string_value,
                "select_mode": self.get_parameter("select_mode").get_parameter_value().string_value,
                "device": self._optional_string_param("device"),
                "enable_follow": bool(self.get_parameter("enable_follow_args").value),
            },
        )

        from robot_control_rtde_fitting_final import build_dual_perception_pipeline

        self.pipeline = build_dual_perception_pipeline(self.args)
        self.use_local_sensor_hub = bool(self.get_parameter("use_local_sensor_hub").value)
        self.sensor_hub = self.pipeline["sensor_hub"] if self.use_local_sensor_hub else None
        if self.sensor_hub is not None:
            self.sensor_hub.start()

        self._frame_index = 0
        self._last_processed_stamp: tuple[int, int] | None = None
        self._latest: dict[str, Any] = {
            "cam0_color": None,
            "cam0_depth": None,
            "cam0_info": None,
            "cam1_color": None,
            "cam1_depth": None,
            "cam1_info": None,
        }

        qos_depth = 5
        if not self.use_local_sensor_hub:
            self.create_subscription(Image, "/cam0/color/image_raw", lambda msg: self._store("cam0_color", msg), qos_depth)
            self.create_subscription(Image, "/cam0/depth/image_raw", lambda msg: self._store("cam0_depth", msg), qos_depth)
            self.create_subscription(CameraInfo, "/cam0/camera_info", lambda msg: self._store("cam0_info", msg), qos_depth)
            self.create_subscription(Image, "/cam1/color/image_raw", lambda msg: self._store("cam1_color", msg), qos_depth)
            self.create_subscription(Image, "/cam1/depth/image_raw", lambda msg: self._store("cam1_depth", msg), qos_depth)
            self.create_subscription(CameraInfo, "/cam1/camera_info", lambda msg: self._store("cam1_info", msg), qos_depth)

        from icra_handover_interfaces.msg import FusionState, GraspTarget, HandState, ObjectState, ShapeFitState
        from sensor_msgs.msg import PointCloud2

        self.object_cam0_pub = self.create_publisher(ObjectState, "/perception/cam0/object_state", qos_depth)
        self.object_cam1_pub = self.create_publisher(ObjectState, "/perception/cam1/object_state", qos_depth)
        self.merged_object_pub = self.create_publisher(ObjectState, "/perception/merged_object_state", qos_depth)
        self.hand_cam0_pub = self.create_publisher(HandState, "/perception/cam0/hand_state", qos_depth)
        self.hand_cam1_pub = self.create_publisher(HandState, "/perception/cam1/hand_state", qos_depth)
        self.selected_hand_pub = self.create_publisher(HandState, "/perception/selected_hand_state", qos_depth)
        self.fusion_pub = self.create_publisher(FusionState, "/perception/fusion_state", qos_depth)
        self.shape_fit_pub = self.create_publisher(ShapeFitState, "/perception/shape_fit_state", qos_depth)
        self.grasp_target_pub = self.create_publisher(GraspTarget, "/perception/grasp_target", qos_depth)
        self.merged_cloud_pub = self.create_publisher(PointCloud2, "/perception/merged_object_cloud", qos_depth)
        self.fitted_cloud_pub = self.create_publisher(PointCloud2, "/perception/fitted_template_cloud", qos_depth)

        rate_hz = max(float(self.get_parameter("process_rate_hz").value), 0.1)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"perception_node ready, source={'local DualSensorHub' if self.use_local_sensor_hub else 'ROS topics'}"
        )

    def _optional_string_param(self, name: str) -> str | None:
        value = self.get_parameter(name).get_parameter_value().string_value
        return value or None

    def _store(self, key: str, msg: Any) -> None:
        self._latest[key] = msg

    def _header(self, frame_id: str | None = None) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id or self.get_parameter("base_frame").get_parameter_value().string_value
        return header

    def _snapshot_from_topics(self):
        required = ("cam0_color", "cam0_depth", "cam0_info", "cam1_color", "cam1_depth", "cam1_info")
        if any(self._latest[key] is None for key in required):
            return None

        stamp_key = (
            int(self._latest["cam0_color"].header.stamp.sec * 1_000_000_000 + self._latest["cam0_color"].header.stamp.nanosec),
            int(self._latest["cam1_color"].header.stamp.sec * 1_000_000_000 + self._latest["cam1_color"].header.stamp.nanosec),
        )
        if stamp_key == self._last_processed_stamp:
            return None
        self._last_processed_stamp = stamp_key

        from utils.realsense_stream import FrameBundle

        depth_scale = float(self.get_parameter("depth_scale").value)
        cam0_color = self._latest["cam0_color"]
        cam1_color = self._latest["cam1_color"]
        cam0 = FrameBundle(
            color_image=_array_from_image(cam0_color),
            depth_image_m=_array_from_image(self._latest["cam0_depth"], depth_scale=depth_scale),
            intrinsics=_intrinsics_from_camera_info(self._latest["cam0_info"]),
            timestamp_ms=stamp_key[0] / 1_000_000.0,
            serial="ros_cam0",
        )
        cam1 = FrameBundle(
            color_image=_array_from_image(cam1_color),
            depth_image_m=_array_from_image(self._latest["cam1_depth"], depth_scale=depth_scale),
            intrinsics=_intrinsics_from_camera_info(self._latest["cam1_info"]),
            timestamp_ms=stamp_key[1] / 1_000_000.0,
            serial="ros_cam1",
        )
        self._frame_index += 1
        return SimpleNamespace(
            pair_index=self._frame_index,
            cam0=cam0,
            cam1=cam1,
            timestamp_delta_ms=abs(cam0.timestamp_ms - cam1.timestamp_ms),
            within_sync_tolerance=True,
        )

    def _read_snapshot(self):
        if self.sensor_hub is not None:
            return self.sensor_hub.read_next_pair()
        return self._snapshot_from_topics()

    def _tick(self) -> None:
        snapshot = self._read_snapshot()
        if snapshot is None:
            return

        try:
            self._process_snapshot(snapshot)
        except Exception as exc:
            self.get_logger().error(f"Perception tick failed: {exc}")

    def _process_snapshot(self, snapshot) -> None:
        from robot_control_rtde_fitting_final import (
            GRASP_POINT_Y_OFFSET_MM,
            GRASP_POINT_Z_OFFSET_MM,
            build_fitted_merged_object,
            build_silhouette_observations,
            choose_point,
            offset_point_base_mm,
        )

        header = self._header()
        object_cam0 = self.pipeline["object_worker_cam0"].process_frame(
            snapshot.cam0,
            frame_id=snapshot.pair_index,
            class_name_filter=self.pipeline["object_class_lock"].locked_class,
        )
        object_cam1 = self.pipeline["object_worker_cam1"].process_frame(
            snapshot.cam1,
            frame_id=snapshot.pair_index,
            class_name_filter=self.pipeline["object_class_lock"].locked_class,
        )
        hand_cam0 = self.pipeline["hand_worker_cam0"].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
        hand_cam1 = self.pipeline["hand_worker_cam1"].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
        merged_object = self.pipeline["object_merger"].process_states(object_cam0, object_cam1)

        silhouette_observations = build_silhouette_observations(snapshot, self.pipeline)
        shape_fitting_state = self.pipeline["shape_fitting_tracker"].process(
            merged_object,
            silhouette_observations=silhouette_observations,
            freeze_silhouette_scale=False,
        )

        object_center_base = shape_fitting_state.centroid_base if bool(getattr(shape_fitting_state, "valid", False)) else None
        selected_hand = self.pipeline["hand_selector"].process_states(
            hand_cam0,
            hand_cam1,
            object_center_base=object_center_base,
        )
        self.pipeline["object_class_lock"].process_states(
            selected_hand=selected_hand,
            hand_cam0=hand_cam0,
            hand_cam1=hand_cam1,
            object_cam0=object_cam0,
            object_cam1=object_cam1,
            object_worker_cam0=self.pipeline["object_worker_cam0"],
            object_worker_cam1=self.pipeline["object_worker_cam1"],
        )

        fitted_merged_object = build_fitted_merged_object(merged_object, shape_fitting_state)
        fusion_state = self.pipeline["fusion"].process_states(
            fitted_merged_object,
            selected_hand,
            now_timestamp=self.get_clock().now().nanoseconds / 1e9,
        )
        grasp_target = self.pipeline["grasp_planner"].process_states(
            fitted_merged_object,
            selected_hand,
            fusion_state,
        )

        object_point_base = choose_point(fusion_state.filtered_object_centroid_base, fitted_merged_object.centroid_base)
        grasp_point_base = grasp_target.target_position_base if grasp_target.valid else None
        grasp_point_base = offset_point_base_mm(
            grasp_point_base,
            y_mm=GRASP_POINT_Y_OFFSET_MM,
            z_mm=GRASP_POINT_Z_OFFSET_MM,
        )
        measurement_source = "measured" if object_point_base is not None else "none"

        self.object_cam0_pub.publish(object_state_msg(object_cam0, header, camera_id=0))
        self.object_cam1_pub.publish(object_state_msg(object_cam1, header, camera_id=1))
        self.merged_object_pub.publish(object_state_msg(fitted_merged_object, header, camera_id=-1))
        self.hand_cam0_pub.publish(hand_state_msg(hand_cam0, header, camera_id=0))
        self.hand_cam1_pub.publish(hand_state_msg(hand_cam1, header, camera_id=1))
        self.selected_hand_pub.publish(hand_state_msg(selected_hand, header, camera_id=-1))
        self.fusion_pub.publish(fusion_state_msg(fusion_state, header))
        self.shape_fit_pub.publish(shape_fit_state_msg(shape_fitting_state, header))
        self.grasp_target_pub.publish(
            grasp_target_msg(
                header=header,
                grasp_target=grasp_target,
                object_point_base=object_point_base,
                grasp_point_base=grasp_point_base,
                measurement_source=measurement_source,
            )
        )
        self.merged_cloud_pub.publish(pointcloud2_from_xyz(getattr(fitted_merged_object, "merged_points_base", []), header))
        self.fitted_cloud_pub.publish(pointcloud2_from_xyz(getattr(shape_fitting_state, "fitted_points_base", []), header))

    def destroy_node(self) -> bool:
        try:
            for key in ("hand_worker_cam0", "hand_worker_cam1"):
                worker = self.pipeline.get(key)
                if worker is not None and hasattr(worker, "close"):
                    worker.close()
            if self.sensor_hub is not None:
                self.sensor_hub.stop()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
