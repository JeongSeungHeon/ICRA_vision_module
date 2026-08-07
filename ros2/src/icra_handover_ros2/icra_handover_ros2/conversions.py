"""Conversion helpers between existing runtime objects and ROS messages."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

from icra_handover_interfaces.msg import (
    FusionState as FusionStateMsg,
    GraspTarget as GraspTargetMsg,
    HandState as HandStateMsg,
    HandoverTaskState,
    ObjectState as ObjectStateMsg,
    RobotStatus as RobotStatusMsg,
    SensorSyncStatus,
    ShapeFitState,
)


def point_or_nan(value: Any) -> Point:
    point = Point()
    if value is None:
        point.x = point.y = point.z = math.nan
        return point
    if all(hasattr(value, attr) for attr in ("x", "y", "z")):
        point.x = float(value.x)
        point.y = float(value.y)
        point.z = float(value.z)
        return point
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    point.x = float(arr[0]) if len(arr) > 0 else math.nan
    point.y = float(arr[1]) if len(arr) > 1 else math.nan
    point.z = float(arr[2]) if len(arr) > 2 else math.nan
    return point


def vector_or_nan(value: Any) -> Vector3:
    vec = Vector3()
    if value is None:
        vec.x = vec.y = vec.z = math.nan
        return vec
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    vec.x = float(arr[0]) if len(arr) > 0 else math.nan
    vec.y = float(arr[1]) if len(arr) > 1 else math.nan
    vec.z = float(arr[2]) if len(arr) > 2 else math.nan
    return vec


def pose_from_xyz_rpy_or_rotvec(xyz: Any = None, orientation: Any = None) -> Pose:
    pose = Pose()
    pose.position = point_or_nan(xyz)
    # The existing runtime mostly stores fixed_orientation_base as rotvec or RPY
    # depending on context. Until a canonical ROS orientation policy is added,
    # keep pose orientation neutral and publish raw orientation through state
    # topics/logs rather than guessing.
    del orientation
    pose.orientation = Quaternion(w=1.0)
    return pose


def image_msg_from_array(array: np.ndarray, header: Header, encoding: str) -> Image:
    arr = np.ascontiguousarray(array)
    msg = Image()
    msg.header = header
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = int(arr.strides[0])
    msg.data = arr.tobytes()
    return msg


def camera_info_from_intrinsics(intrinsics: dict[str, float], header: Header, width: int, height: int) -> CameraInfo:
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    msg = CameraInfo()
    msg.header = header
    msg.width = int(width)
    msg.height = int(height)
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.distortion_model = "plumb_bob"
    return msg


def pointcloud2_from_xyz(points: Any, header: Header) -> PointCloud2:
    arr = np.asarray(points if points is not None else [], dtype=np.float32).reshape((-1, 3))
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(len(arr))
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = bool(len(arr) > 0 and np.isfinite(arr).all())
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg


def object_state_msg(state: Any, header: Header, camera_id: int | None = None) -> ObjectStateMsg:
    msg = ObjectStateMsg()
    msg.header = header
    msg.camera_id = int(getattr(state, "camera_id", -1) if camera_id is None else camera_id)
    msg.frame_id = int(getattr(state, "frame_id", getattr(state, "frame_id_cam0", -1)))
    msg.valid = bool(getattr(state, "valid", False))
    msg.object_detected = bool(getattr(state, "object_detected", False))
    msg.label = str(getattr(state, "label", "") or "")
    msg.confidence = float(getattr(state, "confidence", 0.0) or 0.0)
    msg.centroid_base = point_or_nan(getattr(state, "centroid_base", None))
    msg.point_count = int(getattr(state, "point_count", getattr(state, "merged_point_count", 0)) or 0)
    return msg


def hand_state_msg(state: Any, header: Header, camera_id: int | None = None) -> HandStateMsg:
    msg = HandStateMsg()
    msg.header = header
    msg.camera_id = int(getattr(state, "camera_id", -1) if camera_id is None else camera_id)
    msg.frame_id = int(getattr(state, "frame_id", -1))
    msg.valid = bool(getattr(state, "valid", False))
    msg.hand_detected = bool(getattr(state, "hand_detected", getattr(state, "valid", False)))
    msg.handedness = str(getattr(state, "handedness", "") or "")
    msg.confidence = float(getattr(state, "confidence", 0.0) or 0.0)
    msg.palm_center_base = point_or_nan(getattr(state, "palm_center_base", None))
    msg.palm_normal_base = vector_or_nan(getattr(state, "palm_normal_base", None))
    msg.wrist_base = point_or_nan(getattr(state, "wrist_base", None))
    msg.hand_velocity_base = vector_or_nan(getattr(state, "hand_velocity_base", None))
    msg.candidate_count = int(len(getattr(state, "hand_candidates", []) or []))
    return msg


def fusion_state_msg(state: Any, header: Header) -> FusionStateMsg:
    msg = FusionStateMsg()
    msg.header = header
    msg.valid = bool(getattr(state, "valid", False))
    msg.object_fresh = bool(getattr(state, "object_fresh", False))
    msg.hand_fresh = bool(getattr(state, "hand_fresh", False))
    msg.hand_approach_detected = bool(getattr(state, "hand_approach_detected", False))
    msg.hand_approach_latched = bool(getattr(state, "hand_approach_latched", False))
    msg.object_lifted = bool(getattr(state, "object_lifted", False))
    msg.robot_activation_ready = bool(getattr(state, "robot_activation_ready", False))
    msg.hand_object_distance_m = float(getattr(state, "hand_object_distance_m", math.nan) or math.nan)
    msg.lift_height_delta_m = float(getattr(state, "lift_height_delta_m", 0.0) or 0.0)
    msg.filtered_object_centroid_base = point_or_nan(getattr(state, "filtered_object_centroid_base", None))
    msg.filtered_hand_center_base = point_or_nan(getattr(state, "filtered_hand_center_base", None))
    msg.filtered_hand_normal_base = vector_or_nan(getattr(state, "filtered_hand_normal_base", None))
    return msg


def shape_fit_state_msg(state: Any, header: Header) -> ShapeFitState:
    msg = ShapeFitState()
    msg.header = header
    msg.valid = bool(getattr(state, "valid", False))
    msg.initialized = bool(getattr(state, "initialized", False))
    msg.label = str(getattr(state, "label", "") or "")
    msg.template_id = str(getattr(state, "template_id", "") or "")
    msg.reason = str(getattr(state, "reason", "") or "")
    msg.confidence = float(getattr(state, "tracking_confidence", getattr(state, "confidence", 0.0)) or 0.0)
    msg.centroid_base = point_or_nan(getattr(state, "centroid_base", None))
    msg.pose_base = pose_from_xyz_rpy_or_rotvec(getattr(state, "centroid_base", None), None)
    points = getattr(state, "fitted_points_base", None)
    msg.fitted_point_count = 0 if points is None else int(len(np.asarray(points).reshape((-1, 3))))
    return msg


def grasp_target_msg(
    *,
    header: Header,
    grasp_target: Any,
    object_point_base: Any,
    grasp_point_base: Any,
    measurement_source: str,
) -> GraspTargetMsg:
    msg = GraspTargetMsg()
    msg.header = header
    msg.valid = bool(getattr(grasp_target, "valid", False) and grasp_point_base is not None)
    msg.measurement_source = str(measurement_source or "none")
    msg.object_point_base = point_or_nan(object_point_base)
    msg.grasp_point_base = point_or_nan(grasp_point_base)
    msg.target_pose_base = pose_from_xyz_rpy_or_rotvec(grasp_point_base, getattr(grasp_target, "fixed_orientation_base", None))
    msg.hand_height_clearance_m = float(getattr(grasp_target, "hand_height_clearance_m", math.nan) or math.nan)
    msg.distance_to_centroid_m = float(getattr(grasp_target, "distance_to_centroid_m", math.nan) or math.nan)
    msg.source_point_count = int(getattr(grasp_target, "source_point_count", 0) or 0)
    msg.used_temporal_hold = bool(getattr(grasp_target, "used_temporal_hold", False))
    msg.dropout_hold_active = bool(getattr(grasp_target, "dropout_hold_active", False))
    return msg


def robot_status_msg(status: Any, header: Header) -> RobotStatusMsg:
    msg = RobotStatusMsg()
    msg.header = header
    msg.state = str(getattr(status, "state", "") or "")
    msg.last_error = "" if getattr(status, "last_error", None) is None else str(status.last_error)
    msg.is_connected = bool(getattr(status, "is_connected", False))
    msg.using_mock = bool(getattr(status, "using_mock", False))
    msg.active_request = "" if getattr(status, "active_request", None) is None else str(status.active_request)
    msg.active_request_id = int(getattr(status, "active_request_id", 0) or 0)
    pose = getattr(status, "last_robot_pose", None)
    msg.tcp_pose_base = pose_from_xyz_rpy_or_rotvec(None if pose is None else pose[:3], None if pose is None else pose[3:6])
    msg.last_command_type = "" if getattr(status, "last_command_type", None) is None else str(status.last_command_type)
    msg.grasp_ok = bool(getattr(status, "grasp_ok", False))
    msg.task_ready_epoch = int(getattr(status, "task_ready_epoch", 0) or 0)
    msg.reset_done_epoch = int(getattr(status, "reset_done_epoch", 0) or 0)
    msg.task_done_epoch = int(getattr(status, "task_done_epoch", 0) or 0)
    return msg


def sensor_sync_status_msg(snapshot: Any, header: Header) -> SensorSyncStatus:
    msg = SensorSyncStatus()
    msg.header = header
    msg.pair_index = int(getattr(snapshot, "pair_index", -1))
    msg.timestamp_delta_ms = float(getattr(snapshot, "timestamp_delta_ms", 0.0))
    msg.within_sync_tolerance = bool(getattr(snapshot, "within_sync_tolerance", False))
    msg.cam0_serial = str(getattr(getattr(snapshot, "cam0", None), "serial", "") or "")
    msg.cam1_serial = str(getattr(getattr(snapshot, "cam1", None), "serial", "") or "")
    return msg


def handover_task_state_msg(header: Header, **fields: Any) -> HandoverTaskState:
    msg = HandoverTaskState()
    msg.header = header
    msg.state = str(fields.get("state", "IDLE"))
    msg.reason = str(fields.get("reason", ""))
    msg.task_epoch = int(fields.get("task_epoch", 0) or 0)
    msg.follow_enabled = bool(fields.get("follow_enabled", False))
    msg.motion_triggered = bool(fields.get("motion_triggered", False))
    msg.home_object_locked = bool(fields.get("home_object_locked", False))
    msg.grasp_request_pending = bool(fields.get("grasp_request_pending", False))
    msg.perception_target_valid = bool(fields.get("perception_target_valid", False))
    msg.robot_ready = bool(fields.get("robot_ready", False))
    msg.latest_object_base = point_or_nan(fields.get("latest_object_base"))
    msg.latest_grasp_base = point_or_nan(fields.get("latest_grasp_base"))
    return msg
