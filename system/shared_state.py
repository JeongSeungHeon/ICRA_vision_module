"""Shared state contracts for the dual-camera receive-and-place system.

These dataclasses are intentionally lightweight and use only standard-library
containers so they can be stored in multiprocessing-friendly structures with
minimal conversion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from time import time
from typing import Any, Tuple

Vec3 = Tuple[float, float, float]
Vec6 = Tuple[float, float, float, float, float, float]
Shape2D = Tuple[int, int]
Shape3D = Tuple[int, int, int]
Matrix3x3 = Tuple[Vec3, Vec3, Vec3]

HEIGHT_AXIS_X = "x"
HEIGHT_AXIS_Y = "y"
HEIGHT_AXIS_Z = "z"

HANDEDNESS_LEFT = "left"
HANDEDNESS_RIGHT = "right"
HANDEDNESS_UNKNOWN = "unknown"

GRIPPER_OPEN = "open"
GRIPPER_CLOSE = "close"
GRIPPER_HOLD = "hold"

ROBOT_CMD_HOLD = "hold"
ROBOT_CMD_MOVE_TO_POSITION = "move_to_position"
ROBOT_CMD_SERVO_TO_POSITION = "servo_to_position"
ROBOT_CMD_STOP = "stop"

TASK_IDLE = "IDLE"
TASK_OBSERVE_INITIAL_OBJECT = "OBSERVE_INITIAL_OBJECT"
TASK_WAIT_FOR_HAND_APPROACH = "WAIT_FOR_HAND_APPROACH"
TASK_WAIT_FOR_OBJECT_LIFT = "WAIT_FOR_OBJECT_LIFT"
TASK_ACTIVATE_ROBOT = "ACTIVATE_ROBOT"
TASK_FOLLOW_SERVO = "FOLLOW_SERVO"
TASK_APPROACH_GRASP_POINT = "APPROACH_GRASP_POINT"
TASK_RECEIVE_OBJECT = "RECEIVE_OBJECT"
TASK_VERIFY_GRASP = "VERIFY_GRASP"
TASK_MOVE_TO_INITIAL_PLACE = "MOVE_TO_INITIAL_PLACE"
TASK_RELEASE = "RELEASE"
TASK_RETREAT = "RETREAT"
TASK_FAIL_SAFE = "FAIL_SAFE"


def now_timestamp() -> float:
    return time()


@dataclass
class BaseState:
    timestamp: float = field(default_factory=now_timestamp)
    valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy_with(self, **changes: Any) -> "BaseState":
        return replace(self, **changes)


@dataclass
class SensorState(BaseState):
    camera_id: int = -1
    frame_id: int = -1
    rgb_shape: Shape3D | None = None
    depth_shape: Shape2D | None = None
    intrinsics: Matrix3x3 | None = None
    serial_number: str | None = None


@dataclass
class ObjectState(BaseState):
    camera_id: int = -1
    frame_id: int = -1
    object_detected: bool = False
    label: str | None = None
    confidence: float = 0.0
    centroid_base: Vec3 | None = None
    point_count: int = 0
    points_base: list[Vec3] = field(default_factory=list)


@dataclass
class MergedObjectState(BaseState):
    frame_id_cam0: int = -1
    frame_id_cam1: int = -1
    object_detected: bool = False
    label: str | None = None
    confidence: float = 0.0
    centroid_base: Vec3 | None = None
    initial_centroid_base: Vec3 | None = None
    merged_point_count: int = 0
    merged_points_base: list[Vec3] = field(default_factory=list)
    object_lifted: bool = False
    lift_height_delta_m: float = 0.0
    height_axis_name: str = HEIGHT_AXIS_Z


@dataclass
class HandState(BaseState):
    camera_id: int = -1
    frame_id: int = -1
    hand_detected: bool = False
    handedness: str = HANDEDNESS_UNKNOWN
    confidence: float = 0.0
    palm_center_base: Vec3 | None = None
    palm_normal_base: Vec3 | None = None
    wrist_base: Vec3 | None = None
    hand_velocity_base: Vec3 | None = None


@dataclass
class SelectedHandState(BaseState):
    frame_id: int = -1
    selected_camera: int | None = None
    handedness: str | None = None
    confidence: float = 0.0
    palm_center_base: Vec3 | None = None
    palm_normal_base: Vec3 | None = None
    wrist_base: Vec3 | None = None
    hand_velocity_base: Vec3 | None = None


@dataclass
class FusionState(BaseState):
    filtered_object_centroid_base: Vec3 | None = None
    filtered_hand_center_base: Vec3 | None = None
    filtered_hand_normal_base: Vec3 | None = None
    hand_object_distance_m: float | None = None
    lift_height_delta_m: float = 0.0
    hand_approach_detected: bool = False
    hand_approach_latched: bool = False
    object_lifted: bool = False
    robot_activation_ready: bool = False
    object_fresh: bool = False
    hand_fresh: bool = False
    stable_event_frames: int = 0
    height_axis_name: str = HEIGHT_AXIS_Z


@dataclass
class GraspTargetState(BaseState):
    target_position_base: Vec3 | None = None
    fixed_orientation_base: Vec3 | Vec6 | None = None
    hand_height_clearance_m: float | None = None
    distance_to_centroid_m: float | None = None
    source_point_count: int = 0


@dataclass
class LiveFollowState(BaseState):
    follow_enabled: bool = False
    source_mode: str = TASK_IDLE
    reason: str | None = None
    raw_target_position_base: Vec3 | None = None
    approach_target_position_base: Vec3 | None = None
    servo_target_position_base: Vec3 | None = None
    fixed_orientation_base: Vec3 | Vec6 | None = None
    target_is_fresh: bool = False
    timed_out: bool = False
    workspace_clamped: bool = False
    follow_z_enabled: bool = True
    valid_streak_count: int = 0


@dataclass
class RobotState(BaseState):
    is_connected: bool = False
    robot_mode: str | None = None
    actual_tcp_pose_base: Vec6 | None = None
    actual_tcp_speed: Vec6 | None = None
    actual_tcp_force_base: Vec3 | None = None
    tcp_force_norm_n: float | None = None
    joint_positions: Tuple[float, float, float, float, float, float] | None = None
    joint_currents_a: Tuple[float, float, float, float, float, float] | None = None
    mean_joint_current_a: float | None = None
    gripper_state: str | None = None
    grasp_verified_force_current: bool = False
    last_error: str | None = None


@dataclass
class RobotCommandState(BaseState):
    command_type: str = ROBOT_CMD_HOLD
    target_position_base: Vec3 | None = None
    fixed_orientation_base: Vec3 | Vec6 | None = None
    gripper_action: str | None = None
    source_mode: str = TASK_IDLE
    stop_requested: bool = False


@dataclass
class TaskState(BaseState):
    mode: str = TASK_IDLE
    active_reason: str | None = None
    target_position_base: Vec3 | None = None
    fixed_orientation_base: Vec3 | Vec6 | None = None
    placement_position_base: Vec3 | None = None
    gripper_cmd: str = GRIPPER_HOLD
    selected_hand_camera: int | None = None
    object_lifted: bool = False
    hand_approach_detected: bool = False
    grasp_verified: bool = False
    safety_ok: bool = False


@dataclass
class SharedStateBundle(BaseState):
    sensor_cam0: SensorState = field(default_factory=lambda: SensorState(camera_id=0))
    sensor_cam1: SensorState = field(default_factory=lambda: SensorState(camera_id=1))
    object_cam0: ObjectState = field(default_factory=lambda: ObjectState(camera_id=0))
    object_cam1: ObjectState = field(default_factory=lambda: ObjectState(camera_id=1))
    merged_object: MergedObjectState = field(default_factory=MergedObjectState)
    hand_cam0: HandState = field(default_factory=lambda: HandState(camera_id=0))
    hand_cam1: HandState = field(default_factory=lambda: HandState(camera_id=1))
    selected_hand: SelectedHandState = field(default_factory=SelectedHandState)
    fusion: FusionState = field(default_factory=FusionState)
    grasp_target: GraspTargetState = field(default_factory=GraspTargetState)
    live_follow: LiveFollowState = field(default_factory=LiveFollowState)
    robot: RobotState = field(default_factory=RobotState)
    robot_command: RobotCommandState = field(default_factory=RobotCommandState)
    task: TaskState = field(default_factory=TaskState)


__all__ = [
    "BaseState",
    "SensorState",
    "ObjectState",
    "MergedObjectState",
    "HandState",
    "SelectedHandState",
    "FusionState",
    "GraspTargetState",
    "LiveFollowState",
    "RobotState",
    "RobotCommandState",
    "TaskState",
    "SharedStateBundle",
    "HEIGHT_AXIS_X",
    "HEIGHT_AXIS_Y",
    "HEIGHT_AXIS_Z",
    "HANDEDNESS_LEFT",
    "HANDEDNESS_RIGHT",
    "HANDEDNESS_UNKNOWN",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSE",
    "GRIPPER_HOLD",
    "ROBOT_CMD_HOLD",
    "ROBOT_CMD_MOVE_TO_POSITION",
    "ROBOT_CMD_SERVO_TO_POSITION",
    "ROBOT_CMD_STOP",
    "TASK_IDLE",
    "TASK_OBSERVE_INITIAL_OBJECT",
    "TASK_WAIT_FOR_HAND_APPROACH",
    "TASK_WAIT_FOR_OBJECT_LIFT",
    "TASK_ACTIVATE_ROBOT",
    "TASK_FOLLOW_SERVO",
    "TASK_APPROACH_GRASP_POINT",
    "TASK_RECEIVE_OBJECT",
    "TASK_VERIFY_GRASP",
    "TASK_MOVE_TO_INITIAL_PLACE",
    "TASK_RELEASE",
    "TASK_RETREAT",
    "TASK_FAIL_SAFE",
    "now_timestamp",
]
