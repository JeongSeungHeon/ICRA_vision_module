"""Core helpers and runner for the CORSMAL s8 end-effector benchmark."""

from __future__ import annotations

import csv
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from system.shared_state import GRIPPER_CLOSE, GRIPPER_HOLD, ROBOT_CMD_HOLD, ROBOT_CMD_MOVE_TO_POSITION, RobotCommandState

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_TEMPLATE_PATH = REPO_ROOT / "benchmarks" / "metadata_template.json"
S8_SUBMISSION_COLUMNS = [
    "run_id",
    "repetition",
    "target_id",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "start_time",
    "end_time",
    "motion_time_ms",
]
S8_POSITION_SCORE_THRESHOLD_MM = 30.0


class BenchmarkConfigError(ValueError):
    """Raised when the benchmark configuration is incomplete or inconsistent."""


class SystemClock:
    """Small wrapper that makes time deterministic in tests."""

    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(float(seconds), 0.0))

    def utc_now_iso_ms(self) -> str:
        return isoformat_utc_from_timestamp(self.time())


@dataclass(frozen=True)
class S8OrientationSettings:
    mode: str = "look_at_center"
    tool_forward_axis: str = "x+"
    tool_up_axis: str = "z+"
    world_up_axis_hint: str = "z+"
    roll_offset_deg: float = 0.0


@dataclass(frozen=True)
class HomeFrame:
    position_base_m: tuple[float, float, float]
    rotation_base_from_local: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    orientation_rotvec_base: tuple[float, float, float]

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.asarray(self.rotation_base_from_local, dtype=np.float64).reshape(3, 3)


@dataclass(frozen=True)
class S8Settings:
    team_name: str
    run_id: int
    repetitions: int
    horizontal_reach_mm: float
    vertical_reach_mm: float
    max_motion_time_ms: int
    target_hold_sec: float
    move_timeout_s: float
    position_tolerance_m: float
    init_position_tolerance_mm: float
    init_orientation_tolerance_deg: float
    output_dir: Path
    submission_csv_path: Path
    metadata_json_path: Path
    metadata_template_path: Path
    orientation: S8OrientationSettings
    home_frame: HomeFrame
    frame_mapping_signs: tuple[float, float, float]
    config_path: Path | None = None


@dataclass
class S8MotionLeg:
    leg_name: str
    source_mode: str
    target_local_mm: list[float]
    target_base_m: list[float]
    target_rotvec_base: list[float]
    start_time: str
    end_time: str
    motion_time_ms: int
    reached: bool
    within_benchmark_time: bool
    actual_pose_base: list[float]
    actual_pose_local_mm: list[float]
    actual_quaternion_local_xyzw: list[float]
    position_error_mm: float | None = None


@dataclass
class S8PoseRecord:
    repetition: int
    target_id: int
    desired_local_mm: list[float]
    outbound: S8MotionLeg
    return_leg: S8MotionLeg

    @property
    def submission_row(self) -> dict[str, str]:
        return {
            "run_id": "",
            "repetition": str(self.repetition),
            "target_id": str(self.target_id),
            "x": format_float(self.outbound.actual_pose_local_mm[0]),
            "y": format_float(self.outbound.actual_pose_local_mm[1]),
            "z": format_float(self.outbound.actual_pose_local_mm[2]),
            "qx": format_float(self.outbound.actual_quaternion_local_xyzw[0]),
            "qy": format_float(self.outbound.actual_quaternion_local_xyzw[1]),
            "qz": format_float(self.outbound.actual_quaternion_local_xyzw[2]),
            "qw": format_float(self.outbound.actual_quaternion_local_xyzw[3]),
            "start_time": self.outbound.start_time,
            "end_time": self.outbound.end_time,
            "motion_time_ms": str(int(self.outbound.motion_time_ms)),
        }


@dataclass
class S8RunResult:
    status: str
    manifest_path: Path
    validation_path: Path
    submission_csv_path: Path | None = None
    metadata_json_path: Path | None = None
    records: list[S8PoseRecord] = field(default_factory=list)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


def format_float(value: float) -> str:
    return f"{float(value):.6f}"


def isoformat_utc_from_timestamp(timestamp_s: float) -> str:
    return datetime.fromtimestamp(float(timestamp_s), tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def elapsed_ms_between_iso(start_time_iso: str, end_time_iso: str) -> int:
    start_dt = datetime.fromisoformat(str(start_time_iso).replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(str(end_time_iso).replace("Z", "+00:00"))
    return int(round((end_dt - start_dt).total_seconds() * 1000.0))


def validate_duration_fields(start_time_iso: str, end_time_iso: str, motion_time_ms: int, tolerance_ms: int = 1) -> dict[str, Any]:
    computed_ms = elapsed_ms_between_iso(start_time_iso, end_time_iso)
    mismatch_ms = abs(int(computed_ms) - int(motion_time_ms))
    return {
        "start_time_valid": True,
        "end_time_valid": True,
        "computed_motion_time_ms": int(computed_ms),
        "duration_matches": mismatch_ms <= int(tolerance_ms),
        "duration_mismatch_ms": int(mismatch_ms),
    }


def _normalize_vector(values: Sequence[float], *, allow_zero: bool = False) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        if allow_zero:
            return np.zeros(3, dtype=np.float64)
        raise ValueError("Zero-length vector is not allowed.")
    return vector / norm


def parse_axis_spec(axis_spec: str) -> np.ndarray:
    normalized = str(axis_spec or "").strip().lower()
    if normalized in {"x", "x+"}:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if normalized == "x-":
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    if normalized in {"y", "y+"}:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if normalized == "y-":
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if normalized in {"z", "z+"}:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if normalized == "z-":
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    raise BenchmarkConfigError(f"Unsupported axis spec: {axis_spec!r}")


def rotation_matrix_from_axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_unit = _normalize_vector(axis)
    x_value, y_value, z_value = axis_unit
    cos_theta = float(math.cos(float(angle_rad)))
    sin_theta = float(math.sin(float(angle_rad)))
    one_minus = 1.0 - cos_theta
    return np.array(
        [
            [cos_theta + x_value * x_value * one_minus, x_value * y_value * one_minus - z_value * sin_theta, x_value * z_value * one_minus + y_value * sin_theta],
            [y_value * x_value * one_minus + z_value * sin_theta, cos_theta + y_value * y_value * one_minus, y_value * z_value * one_minus - x_value * sin_theta],
            [z_value * x_value * one_minus - y_value * sin_theta, z_value * y_value * one_minus + x_value * sin_theta, cos_theta + z_value * z_value * one_minus],
        ],
        dtype=np.float64,
    )


def rpy_to_rotation_matrix(rpy_rad: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy_rad[:3]]
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def rotvec_to_rotation_matrix(rotvec: Sequence[float]) -> np.ndarray:
    rotvec_array = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec_array))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec_array / angle
    return rotation_matrix_from_axis_angle(axis, angle)


def rotation_matrix_to_rotvec(rotation: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    cos_theta = max(min((trace - 1.0) * 0.5, 1.0), -1.0)
    theta = float(math.acos(cos_theta))
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        principal_index = int(np.argmax(eigenvalues))
        axis = np.asarray(eigenvectors[:, principal_index], dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return (0.0, 0.0, 0.0)
    axis = axis / axis_norm
    rotvec = axis * theta
    return (float(rotvec[0]), float(rotvec[1]), float(rotvec[2]))


def rotation_matrix_to_quaternion_xyzw(rotation: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return tuple(float(value) for value in quaternion)


def _coerce_signs(position_signs: Sequence[float] | None) -> tuple[float, float, float]:
    if position_signs is None:
        return (1.0, 1.0, 1.0)
    values = tuple(float(value) for value in position_signs)
    if len(values) != 3:
        raise BenchmarkConfigError("frame_mapping.position_signs must contain exactly three values.")
    return tuple(-1.0 if value < 0.0 else 1.0 for value in values)


def map_position_with_signs(position_m: Sequence[float], signs: Sequence[float]) -> tuple[float, float, float]:
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    sign_array = np.asarray(tuple(float(value) for value in signs), dtype=np.float64).reshape(3)
    mapped = sign_array * position
    return tuple(float(value) for value in mapped)


def resolve_home_frame(config: Mapping[str, Any]) -> tuple[HomeFrame, tuple[float, float, float]]:
    home_cfg = dict(config.get("home_pose", {}) or {})
    if not bool(home_cfg.get("enabled", False)):
        raise BenchmarkConfigError("home_pose.enabled must be true for the s8 benchmark.")
    position_raw = home_cfg.get("position_m")
    if position_raw is None or len(position_raw) < 3:
        raise BenchmarkConfigError("home_pose.position_m must contain three values.")

    robot_cfg = dict(config.get("robot", {}) or {})
    frame_mapping_cfg = dict(robot_cfg.get("frame_mapping", {}) or {})
    frame_mapping_enabled = bool(frame_mapping_cfg.get("enabled", True))
    frame_mapping_signs = _coerce_signs(frame_mapping_cfg.get("position_signs", (-1.0, -1.0, 1.0)))
    # Benchmark home pose must match RtdeController.move_home(), which uses the
    # configured home pose directly in the RTDE robot-base frame without
    # applying frame_mapping.position_signs.
    position_base_m = tuple(float(value) for value in position_raw[:3])
    if not frame_mapping_enabled:
        frame_mapping_signs = (1.0, 1.0, 1.0)

    grasp_cfg = dict(config.get("grasp", {}) or {})
    orientation_format = str(grasp_cfg.get("fixed_orientation_format", "rotvec")).strip().lower()
    orientation_values = grasp_cfg.get("fixed_orientation_base", (0.0, 0.0, 0.0))
    if orientation_values is None or len(orientation_values) < 3:
        raise BenchmarkConfigError("grasp.fixed_orientation_base must contain three values.")
    if orientation_format == "rpy":
        home_rotation = rpy_to_rotation_matrix(orientation_values[:3])
        home_rotvec = rotation_matrix_to_rotvec(home_rotation)
    else:
        home_rotvec = tuple(float(value) for value in orientation_values[:3])
        home_rotation = rotvec_to_rotation_matrix(home_rotvec)

    home_frame = HomeFrame(
        position_base_m=tuple(float(value) for value in position_base_m),
        rotation_base_from_local=tuple(tuple(float(cell) for cell in row) for row in home_rotation.tolist()),
        orientation_rotvec_base=tuple(float(value) for value in home_rotvec),
    )
    return home_frame, frame_mapping_signs


def render_path_template(path_template: str, settings_like: Mapping[str, Any]) -> str:
    return str(path_template).format(
        team_name=settings_like["team_name"],
        TEAM_NAME=settings_like["team_name"],
        run_id=settings_like["run_id"],
    )


def load_s8_settings(
    config: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    config_path: Path | None = None,
    move_timeout_s: float = 10.0,
    position_tolerance_m: float = 0.01,
) -> S8Settings:
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    benchmark_cfg = dict(config.get("benchmark", {}) or {})
    s8_cfg = dict(benchmark_cfg.get("s8", {}) or {})
    if not s8_cfg:
        raise BenchmarkConfigError("benchmark.s8 block is missing from the configuration.")

    home_frame, frame_mapping_signs = resolve_home_frame(config)
    orientation_cfg = dict(s8_cfg.get("orientation", {}) or {})
    settings_dict = {
        "team_name": str(s8_cfg.get("team_name", "TEAM_NAME")).strip() or "TEAM_NAME",
        "run_id": int(s8_cfg.get("run_id", 1)),
    }
    output_dir = repo_root / render_path_template(str(s8_cfg.get("output_dir", "output/benchmark")), settings_dict)
    submission_csv_name = render_path_template(str(s8_cfg.get("submission_csv_name", "s8_submission_{team_name}.csv")), settings_dict)
    metadata_json_path = repo_root / render_path_template(
        str(s8_cfg.get("metadata_json_path", "output/benchmark/metadata_{team_name}.json")),
        settings_dict,
    )
    metadata_template_path = repo_root / str(s8_cfg.get("metadata_template_path", "benchmarks/metadata_template.json"))

    return S8Settings(
        team_name=settings_dict["team_name"],
        run_id=settings_dict["run_id"],
        repetitions=int(s8_cfg.get("repetitions", 3)),
        horizontal_reach_mm=float(s8_cfg.get("horizontal_reach_mm", 800.0)),
        vertical_reach_mm=float(s8_cfg.get("vertical_reach_mm", 600.0)),
        max_motion_time_ms=int(s8_cfg.get("max_motion_time_ms", 5000)),
        target_hold_sec=float(s8_cfg.get("target_hold_sec", 1.0)),
        move_timeout_s=float(move_timeout_s),
        position_tolerance_m=float(position_tolerance_m),
        init_position_tolerance_mm=float(s8_cfg.get("init_position_tolerance_mm", 15.0)),
        init_orientation_tolerance_deg=float(s8_cfg.get("init_orientation_tolerance_deg", 10.0)),
        output_dir=output_dir,
        submission_csv_path=output_dir / submission_csv_name,
        metadata_json_path=metadata_json_path,
        metadata_template_path=metadata_template_path,
        orientation=S8OrientationSettings(
            mode=str(orientation_cfg.get("mode", "look_at_center")).strip().lower(),
            tool_forward_axis=str(orientation_cfg.get("tool_forward_axis", "x+")).strip().lower(),
            tool_up_axis=str(orientation_cfg.get("tool_up_axis", "z+")).strip().lower(),
            world_up_axis_hint=str(orientation_cfg.get("world_up_axis_hint", "z+")).strip().lower(),
            roll_offset_deg=float(orientation_cfg.get("roll_offset_deg", 0.0)),
        ),
        home_frame=home_frame,
        frame_mapping_signs=frame_mapping_signs,
        config_path=None if config_path is None else Path(config_path),
    )


def build_target_positions_local_mm(horizontal_reach_mm: float, vertical_reach_mm: float) -> dict[int, tuple[float, float, float]]:
    del horizontal_reach_mm, vertical_reach_mm
    return {
        1: (-400.0, 0.0, 0.0),
        2: (-400.0, 400.0, 0.0),
        3: (-400.0, -400.0, 0.0),
        4: (-400.0, 0.0, 150.0),
        5: (-400.0, 400.0, 150.0),
        6: (-400.0, -400.0, 150.0),
    }


def _build_perpendicular_reference(forward_world: np.ndarray, up_hint_world: np.ndarray) -> np.ndarray:
    projected = up_hint_world - np.dot(up_hint_world, forward_world) * forward_world
    norm = float(np.linalg.norm(projected))
    if norm >= 1e-12:
        return projected / norm
    fallback_axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    for axis in fallback_axes:
        projected = axis - np.dot(axis, forward_world) * forward_world
        norm = float(np.linalg.norm(projected))
        if norm >= 1e-12:
            return projected / norm
    raise ValueError("Failed to build a perpendicular reference axis.")


def compute_look_at_rotation_local(
    target_local_mm: Sequence[float],
    center_local_mm: Sequence[float],
    orientation_settings: S8OrientationSettings,
) -> np.ndarray:
    if orientation_settings.mode != "look_at_center":
        raise BenchmarkConfigError(f"Unsupported benchmark.s8 orientation mode: {orientation_settings.mode!r}")

    target = np.asarray(target_local_mm, dtype=np.float64).reshape(3)
    center = np.asarray(center_local_mm, dtype=np.float64).reshape(3)
    forward_world = center - target
    if float(np.linalg.norm(forward_world)) < 1e-9:
        base_rotation = np.eye(3, dtype=np.float64)
    else:
        forward_world = _normalize_vector(forward_world)
        world_up = _normalize_vector(parse_axis_spec(orientation_settings.world_up_axis_hint))
        up_world = _build_perpendicular_reference(forward_world, world_up)

        tool_forward = _normalize_vector(parse_axis_spec(orientation_settings.tool_forward_axis))
        tool_up = _build_perpendicular_reference(tool_forward, _normalize_vector(parse_axis_spec(orientation_settings.tool_up_axis)))

        tool_basis = np.column_stack((tool_forward, tool_up, np.cross(tool_forward, tool_up)))
        world_basis = np.column_stack((forward_world, up_world, np.cross(forward_world, up_world)))
        base_rotation = world_basis @ tool_basis.T

    if abs(float(orientation_settings.roll_offset_deg)) < 1e-12:
        return base_rotation

    if float(np.linalg.norm(center - target)) < 1e-9:
        roll_axis = _normalize_vector(parse_axis_spec(orientation_settings.tool_forward_axis))
    else:
        roll_axis = _normalize_vector(center - target)
    roll_rotation = rotation_matrix_from_axis_angle(roll_axis, math.radians(float(orientation_settings.roll_offset_deg)))
    return roll_rotation @ base_rotation


def target_local_to_base_position_m(home_frame: HomeFrame, target_local_mm: Sequence[float]) -> tuple[float, float, float]:
    local_position_m = np.asarray(target_local_mm, dtype=np.float64).reshape(3) / 1000.0
    base_position_m = np.asarray(home_frame.position_base_m, dtype=np.float64).reshape(3) + local_position_m
    return tuple(float(value) for value in base_position_m)


def target_local_rotation_to_base_rotvec(home_frame: HomeFrame, local_rotation: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    base_rotation = home_frame.rotation_matrix @ np.asarray(local_rotation, dtype=np.float64).reshape(3, 3)
    return rotation_matrix_to_rotvec(base_rotation)


def actual_pose_base_to_local(home_frame: HomeFrame, pose_base: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_base, dtype=np.float64).reshape(6)
    position_base_m = pose[:3]
    rotation_base = rotvec_to_rotation_matrix(pose[3:6])
    position_local_m = position_base_m - np.asarray(home_frame.position_base_m, dtype=np.float64).reshape(3)
    rotation_local = rotation_base
    return position_local_m, rotation_local


def build_metadata_document(settings: S8Settings, *, submission_timestamp: str | None = None) -> dict[str, Any]:
    if settings.metadata_template_path.exists():
        with settings.metadata_template_path.open("r", encoding="utf-8") as handle:
            template = json.load(handle)
    else:
        with DEFAULT_METADATA_TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
            template = json.load(handle)

    document = deepcopy(template)
    document["submission_timestamp"] = str(submission_timestamp or isoformat_utc_from_timestamp(time.time()))
    document["team"] = settings.team_name
    execution_policy = dict(document.get("execution_policy", {}) or {})
    execution_policy["end_effector_reachability"] = [
        float(settings.horizontal_reach_mm),
        float(settings.vertical_reach_mm),
    ]
    document["execution_policy"] = execution_policy
    return document


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def write_submission_csv_atomic(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=S8_SUBMISSION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _serialize_pose_record(record: S8PoseRecord) -> dict[str, Any]:
    return {
        "repetition": int(record.repetition),
        "target_id": int(record.target_id),
        "desired_local_mm": [float(value) for value in record.desired_local_mm],
        "outbound": {
            "leg_name": record.outbound.leg_name,
            "source_mode": record.outbound.source_mode,
            "target_local_mm": [float(value) for value in record.outbound.target_local_mm],
            "target_base_m": [float(value) for value in record.outbound.target_base_m],
            "target_rotvec_base": [float(value) for value in record.outbound.target_rotvec_base],
            "start_time": record.outbound.start_time,
            "end_time": record.outbound.end_time,
            "motion_time_ms": int(record.outbound.motion_time_ms),
            "reached": bool(record.outbound.reached),
            "within_benchmark_time": bool(record.outbound.within_benchmark_time),
            "actual_pose_base": [float(value) for value in record.outbound.actual_pose_base],
            "actual_pose_local_mm": [float(value) for value in record.outbound.actual_pose_local_mm],
            "actual_quaternion_local_xyzw": [float(value) for value in record.outbound.actual_quaternion_local_xyzw],
            "position_error_mm": None if record.outbound.position_error_mm is None else float(record.outbound.position_error_mm),
        },
        "return_leg": {
            "leg_name": record.return_leg.leg_name,
            "source_mode": record.return_leg.source_mode,
            "target_local_mm": [float(value) for value in record.return_leg.target_local_mm],
            "target_base_m": [float(value) for value in record.return_leg.target_base_m],
            "target_rotvec_base": [float(value) for value in record.return_leg.target_rotvec_base],
            "start_time": record.return_leg.start_time,
            "end_time": record.return_leg.end_time,
            "motion_time_ms": int(record.return_leg.motion_time_ms),
            "reached": bool(record.return_leg.reached),
            "within_benchmark_time": bool(record.return_leg.within_benchmark_time),
            "actual_pose_base": [float(value) for value in record.return_leg.actual_pose_base],
            "actual_pose_local_mm": [float(value) for value in record.return_leg.actual_pose_local_mm],
            "actual_quaternion_local_xyzw": [float(value) for value in record.return_leg.actual_quaternion_local_xyzw],
            "position_error_mm": None if record.return_leg.position_error_mm is None else float(record.return_leg.position_error_mm),
        },
    }


def _make_robot_command(
    command_type: str,
    *,
    target_position_base: Sequence[float],
    fixed_orientation_base: Sequence[float],
    timestamp: float,
    source_mode: str,
) -> RobotCommandState:
    return RobotCommandState(
        command_type=command_type,
        target_position_base=tuple(float(value) for value in target_position_base[:3]),
        fixed_orientation_base=tuple(float(value) for value in fixed_orientation_base[:3]),
        gripper_action=GRIPPER_HOLD,
        source_mode=source_mode,
        stop_requested=False,
        timestamp=float(timestamp),
        valid=True,
    )


def wait_until_target_reached(
    controller: Any,
    target_position_base: Sequence[float],
    *,
    timeout_s: float,
    tolerance_m: float,
    poll_dt: float,
    clock: Any,
) -> tuple[bool, Sequence[float] | None]:
    deadline = float(clock.time()) + max(float(timeout_s), 0.0)
    target = np.asarray(target_position_base, dtype=np.float64).reshape(3)
    last_pose = None
    while float(clock.time()) <= deadline:
        state = controller.read_robot_state(now_timestamp=float(clock.time()))
        pose = getattr(state, "actual_tcp_pose_base", None)
        if pose is not None:
            pose_tuple = tuple(float(value) for value in pose[:6])
            last_pose = pose_tuple
            current = np.asarray(pose_tuple[:3], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(current - target)) <= float(tolerance_m):
                return True, pose_tuple
        clock.sleep(poll_dt)
    return False, last_pose


def read_current_orientation_rotvec(controller: Any, clock: Any) -> tuple[float, float, float]:
    state = controller.read_robot_state(now_timestamp=float(clock.time()))
    pose = getattr(state, "actual_tcp_pose_base", None)
    if pose is None:
        raise RuntimeError("Failed to read current TCP pose while resolving command orientation.")
    if len(pose) < 6:
        raise RuntimeError("TCP pose is incomplete; expected 6 values.")
    return tuple(float(value) for value in pose[3:6])


def build_runtime_home_frame(home_frame: HomeFrame, orientation_rotvec_base: Sequence[float]) -> HomeFrame:
    orientation = tuple(float(value) for value in orientation_rotvec_base[:3])
    rotation = rotvec_to_rotation_matrix(orientation)
    return replace(
        home_frame,
        rotation_base_from_local=tuple(tuple(float(cell) for cell in row) for row in rotation.tolist()),
        orientation_rotvec_base=orientation,
    )


def move_to_home_pose_with_current_orientation(
    controller: Any,
    settings: S8Settings,
    clock: Any,
) -> HomeFrame:
    current_orientation = read_current_orientation_rotvec(controller, clock)
    command_timestamp = float(clock.time())
    command = _make_robot_command(
        ROBOT_CMD_MOVE_TO_POSITION,
        target_position_base=settings.home_frame.position_base_m,
        fixed_orientation_base=current_orientation,
        timestamp=command_timestamp,
        source_mode="startup_home_pose",
    )
    controller.step(command, now_timestamp=command_timestamp)
    reached, _last_pose = wait_until_target_reached(
        controller,
        settings.home_frame.position_base_m,
        timeout_s=settings.move_timeout_s,
        tolerance_m=settings.position_tolerance_m,
        poll_dt=0.05,
        clock=clock,
    )
    if not reached:
        raise RuntimeError("Failed to reach startup home pose while keeping current orientation.")
    return build_runtime_home_frame(settings.home_frame, current_orientation)


def close_gripper_for_benchmark(controller: Any, clock: Any, settle_sec: float = 0.5) -> None:
    close_started = False
    if hasattr(controller, "start_gripper_close"):
        try:
            close_started = bool(controller.start_gripper_close())
        except Exception:
            close_started = False

    if not close_started:
        command_timestamp = float(clock.time())
        command = RobotCommandState(
            command_type=ROBOT_CMD_HOLD,
            target_position_base=None,
            fixed_orientation_base=None,
            gripper_action=GRIPPER_CLOSE,
            source_mode="startup_gripper_close",
            stop_requested=False,
            timestamp=command_timestamp,
            valid=True,
        )
        controller.step(command, now_timestamp=command_timestamp)

    clock.sleep(max(float(settle_sec), 0.0))


def execute_motion_leg(
    controller: Any,
    home_frame: HomeFrame,
    target_local_mm: Sequence[float],
    target_rotvec_base: Sequence[float],
    *,
    source_mode: str,
    leg_name: str,
    move_timeout_s: float,
    position_tolerance_m: float,
    max_motion_time_ms: int,
    clock: Any,
) -> S8MotionLeg:
    target_base_m = target_local_to_base_position_m(home_frame, target_local_mm)
    start_time = clock.utc_now_iso_ms()
    command_timestamp = float(clock.time())
    command = _make_robot_command(
        ROBOT_CMD_MOVE_TO_POSITION,
        target_position_base=target_base_m,
        fixed_orientation_base=target_rotvec_base,
        timestamp=command_timestamp,
        source_mode=source_mode,
    )
    controller.step(command, now_timestamp=command_timestamp)
    reached, last_pose = wait_until_target_reached(
        controller,
        target_base_m,
        timeout_s=move_timeout_s,
        tolerance_m=position_tolerance_m,
        poll_dt=0.05,
        clock=clock,
    )
    end_time = clock.utc_now_iso_ms()
    if last_pose is None:
        state = controller.read_robot_state(now_timestamp=float(clock.time()))
        pose = getattr(state, "actual_tcp_pose_base", None)
        last_pose = tuple(float(value) for value in (pose[:6] if pose is not None else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    motion_time_ms = elapsed_ms_between_iso(start_time, end_time)
    actual_local_m, actual_local_rotation = actual_pose_base_to_local(home_frame, last_pose)
    actual_local_mm = (actual_local_m * 1000.0).tolist()
    actual_quaternion = rotation_matrix_to_quaternion_xyzw(actual_local_rotation)
    return S8MotionLeg(
        leg_name=leg_name,
        source_mode=source_mode,
        target_local_mm=[float(value) for value in target_local_mm],
        target_base_m=[float(value) for value in target_base_m],
        target_rotvec_base=[float(value) for value in target_rotvec_base[:3]],
        start_time=start_time,
        end_time=end_time,
        motion_time_ms=int(motion_time_ms),
        reached=bool(reached),
        within_benchmark_time=int(motion_time_ms) <= int(max_motion_time_ms),
        actual_pose_base=[float(value) for value in last_pose[:6]],
        actual_pose_local_mm=[float(value) for value in actual_local_mm[:3]],
        actual_quaternion_local_xyzw=[float(value) for value in actual_quaternion],
    )


def validate_initial_pose(controller: Any, settings: S8Settings, clock: Any) -> dict[str, Any]:
    state = controller.read_robot_state(now_timestamp=float(clock.time()))
    pose = getattr(state, "actual_tcp_pose_base", None)
    if pose is None:
        raise RuntimeError("Failed to read the robot TCP pose after move_home().")

    local_position_m, _local_rotation = actual_pose_base_to_local(settings.home_frame, pose)
    position_error_mm = float(np.linalg.norm(local_position_m) * 1000.0)
    current_rotation = rotvec_to_rotation_matrix(pose[3:6])
    home_rotation = settings.home_frame.rotation_matrix
    relative_rotation = home_rotation.T @ current_rotation
    orientation_error_deg = float(math.degrees(np.linalg.norm(rotation_matrix_to_rotvec(relative_rotation))))
    return {
        "position_error_mm": position_error_mm,
        "orientation_error_deg": orientation_error_deg,
        "position_ok": position_error_mm <= float(settings.init_position_tolerance_mm),
        "orientation_ok": orientation_error_deg <= float(settings.init_orientation_tolerance_deg),
        "actual_pose_base": [float(value) for value in pose[:6]],
        "actual_pose_local_mm": [float(value) for value in (local_position_m * 1000.0).tolist()],
        "actual_quaternion_local_xyzw": [float(value) for value in rotation_matrix_to_quaternion_xyzw(current_rotation)],
    }


def compute_validation_summary(
    records: Sequence[S8PoseRecord],
    settings: S8Settings,
    *,
    assume_valid_returns_without_manifest: bool = False,
) -> dict[str, Any]:
    desired_targets = build_target_positions_local_mm(settings.horizontal_reach_mm, settings.vertical_reach_mm)
    per_pose_results = []
    final_scores = []

    for target_id in sorted(desired_targets):
        per_rep_entries = []
        valid_errors = []
        for repetition in range(1, int(settings.repetitions) + 1):
            record = next((item for item in records if item.repetition == repetition and item.target_id == target_id), None)
            if record is None:
                per_rep_entries.append(
                    {
                        "repetition": repetition,
                        "present": False,
                        "valid_for_scoring": False,
                    }
                )
                continue

            record.outbound.position_error_mm = float(
                np.linalg.norm(np.asarray(record.outbound.actual_pose_local_mm, dtype=np.float64) - np.asarray(record.desired_local_mm, dtype=np.float64))
            )
            outbound_time_check = validate_duration_fields(
                record.outbound.start_time,
                record.outbound.end_time,
                record.outbound.motion_time_ms,
            )
            return_time_check = validate_duration_fields(
                record.return_leg.start_time,
                record.return_leg.end_time,
                record.return_leg.motion_time_ms,
            )
            outbound_valid = bool(record.outbound.reached) and bool(record.outbound.within_benchmark_time) and bool(outbound_time_check["duration_matches"])
            return_valid = bool(record.return_leg.reached) and bool(record.return_leg.within_benchmark_time) and bool(return_time_check["duration_matches"])
            if assume_valid_returns_without_manifest:
                return_valid = True

            valid_for_scoring = bool(outbound_valid and return_valid)
            if valid_for_scoring:
                valid_errors.append(float(record.outbound.position_error_mm))

            per_rep_entries.append(
                {
                    "repetition": repetition,
                    "present": True,
                    "desired_local_mm": [float(value) for value in record.desired_local_mm],
                    "actual_local_mm": [float(value) for value in record.outbound.actual_pose_local_mm],
                    "position_error_mm": float(record.outbound.position_error_mm),
                    "outbound_motion_time_ms": int(record.outbound.motion_time_ms),
                    "return_motion_time_ms": int(record.return_leg.motion_time_ms),
                    "outbound_duration_matches": bool(outbound_time_check["duration_matches"]),
                    "return_duration_matches": bool(return_time_check["duration_matches"]),
                    "outbound_within_benchmark_time": bool(record.outbound.within_benchmark_time),
                    "return_within_benchmark_time": bool(record.return_leg.within_benchmark_time),
                    "valid_for_scoring": valid_for_scoring,
                }
            )

        if len(valid_errors) >= 2:
            median_error_mm = float(np.median(np.asarray(valid_errors, dtype=np.float64)))
            pose_score = max(0.0, 1.0 - (median_error_mm / float(S8_POSITION_SCORE_THRESHOLD_MM)))
        else:
            median_error_mm = None
            pose_score = 0.0
        final_scores.append(float(pose_score))
        per_pose_results.append(
            {
                "target_id": int(target_id),
                "desired_local_mm": [float(value) for value in desired_targets[target_id]],
                "valid_repetition_count": int(len(valid_errors)),
                "median_error_mm": None if median_error_mm is None else float(median_error_mm),
                "pose_score": float(pose_score),
                "repetitions": per_rep_entries,
            }
        )

    final_score = float(sum(final_scores) / max(len(final_scores), 1))
    return {
        "status": "validated",
        "tau_position_mm": float(S8_POSITION_SCORE_THRESHOLD_MM),
        "max_motion_time_ms": int(settings.max_motion_time_ms),
        "required_repetitions": int(settings.repetitions),
        "records_present": int(len(records)),
        "complete_record_count": int(len(records)) == int(settings.repetitions) * 6,
        "final_score": final_score,
        "per_pose": per_pose_results,
        "return_leg_assumption_used": bool(assume_valid_returns_without_manifest),
    }


def run_s8_benchmark(
    controller: Any,
    settings: S8Settings,
    *,
    clock: Any | None = None,
    dry_run_score_only: bool = False,
    console: Any = print,
) -> S8RunResult:
    clock = SystemClock() if clock is None else clock
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = clock.utc_now_iso_ms()
    timestamp_tag = started_at.replace(":", "").replace("-", "").replace(".", "").replace("Z", "Z")
    manifest_path = settings.output_dir / f"s8_run_manifest_{timestamp_tag}.json"
    validation_path = settings.output_dir / f"s8_validation_{timestamp_tag}.json"

    manifest: dict[str, Any] = {
        "benchmark": "s8_end_effector",
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "team_name": settings.team_name,
        "run_id": int(settings.run_id),
        "config_path": None if settings.config_path is None else str(settings.config_path),
        "submission_csv_path": str(settings.submission_csv_path),
        "metadata_json_path": str(settings.metadata_json_path),
        "home_frame": {
            "position_base_m": [float(value) for value in settings.home_frame.position_base_m],
            "orientation_rotvec_base": [float(value) for value in settings.home_frame.orientation_rotvec_base],
        },
        "settings": {
            "repetitions": int(settings.repetitions),
            "horizontal_reach_mm": float(settings.horizontal_reach_mm),
            "vertical_reach_mm": float(settings.vertical_reach_mm),
            "max_motion_time_ms": int(settings.max_motion_time_ms),
            "target_hold_sec": float(settings.target_hold_sec),
            "move_timeout_s": float(settings.move_timeout_s),
            "position_tolerance_m": float(settings.position_tolerance_m),
        },
        "records": [],
    }

    if dry_run_score_only:
        rows = load_submission_rows(settings.submission_csv_path)
        records = records_from_submission_rows(
            rows,
            build_target_positions_local_mm(settings.horizontal_reach_mm, settings.vertical_reach_mm),
            max_motion_time_ms=settings.max_motion_time_ms,
        )
        validation_summary = compute_validation_summary(records, settings, assume_valid_returns_without_manifest=True)
        validation_summary["status"] = "score_only"
        validation_summary["submission_csv_path"] = str(settings.submission_csv_path)
        write_json_atomic(validation_path, validation_summary)
        manifest["status"] = "score_only"
        manifest["completed_at"] = clock.utc_now_iso_ms()
        manifest["records"] = [_serialize_pose_record(record) for record in records]
        write_json_atomic(manifest_path, manifest)
        return S8RunResult(
            status="score_only",
            manifest_path=manifest_path,
            validation_path=validation_path,
            submission_csv_path=settings.submission_csv_path if settings.submission_csv_path.exists() else None,
            records=records,
            validation_summary=validation_summary,
            manifest=manifest,
        )

    target_positions_local = build_target_positions_local_mm(settings.horizontal_reach_mm, settings.vertical_reach_mm)
    center_local_mm = np.asarray(target_positions_local[1], dtype=np.float64)
    records: list[S8PoseRecord] = []

    try:
        controller.connect()
        close_gripper_for_benchmark(controller, clock)
        runtime_home_frame = move_to_home_pose_with_current_orientation(controller, settings, clock)
        manifest["home_frame"] = {
            "position_base_m": [float(value) for value in runtime_home_frame.position_base_m],
            "orientation_rotvec_base": [float(value) for value in runtime_home_frame.orientation_rotvec_base],
        }

        runtime_settings = replace(settings, home_frame=runtime_home_frame)
        initial_validation = validate_initial_pose(controller, runtime_settings, clock)
        manifest["initial_pose_validation"] = initial_validation
        if not (initial_validation["position_ok"] and initial_validation["orientation_ok"]):
            raise RuntimeError(
                "Initial pose validation failed: "
                f"position_error_mm={initial_validation['position_error_mm']:.3f}, "
                f"orientation_error_deg={initial_validation['orientation_error_deg']:.3f}"
            )

        for repetition in range(1, int(settings.repetitions) + 1):
            console(f"[s8] repetition {repetition}/{settings.repetitions} start")
            for target_id in range(1, 7):
                target_local_mm = target_positions_local[target_id]
                _ = compute_look_at_rotation_local(target_local_mm, center_local_mm, settings.orientation)
                target_rotvec_base = read_current_orientation_rotvec(controller, clock)
                outbound_source = f"s8_rep{repetition}_target{target_id}_outbound"
                return_source = f"s8_rep{repetition}_target{target_id}_return"
                outbound = execute_motion_leg(
                    controller,
                    runtime_settings.home_frame,
                    target_local_mm,
                    target_rotvec_base,
                    source_mode=outbound_source,
                    leg_name="outbound",
                    move_timeout_s=runtime_settings.move_timeout_s,
                    position_tolerance_m=runtime_settings.position_tolerance_m,
                    max_motion_time_ms=runtime_settings.max_motion_time_ms,
                    clock=clock,
                )
                if not outbound.reached:
                    raise RuntimeError(f"Failed to reach target {target_id} in repetition {repetition}.")

                outbound.position_error_mm = float(
                    np.linalg.norm(np.asarray(outbound.actual_pose_local_mm, dtype=np.float64) - np.asarray(target_local_mm, dtype=np.float64))
                )
                console(
                    "[s8] rep=%d target=%d outbound=%dms err=%.2fmm"
                    % (repetition, target_id, outbound.motion_time_ms, float(outbound.position_error_mm))
                )
                clock.sleep(settings.target_hold_sec)

                return_leg = execute_motion_leg(
                    controller,
                    runtime_settings.home_frame,
                    (0.0, 0.0, 0.0),
                    read_current_orientation_rotvec(controller, clock),
                    source_mode=return_source,
                    leg_name="return_to_init",
                    move_timeout_s=runtime_settings.move_timeout_s,
                    position_tolerance_m=runtime_settings.position_tolerance_m,
                    max_motion_time_ms=runtime_settings.max_motion_time_ms,
                    clock=clock,
                )
                if not return_leg.reached:
                    raise RuntimeError(f"Failed to return to init after target {target_id} in repetition {repetition}.")

                record = S8PoseRecord(
                    repetition=repetition,
                    target_id=target_id,
                    desired_local_mm=[float(value) for value in target_local_mm],
                    outbound=outbound,
                    return_leg=return_leg,
                )
                records.append(record)
                manifest["records"].append(_serialize_pose_record(record))

        submission_rows = []
        for record in records:
            row = dict(record.submission_row)
            row["run_id"] = str(int(settings.run_id))
            submission_rows.append(row)
        write_submission_csv_atomic(settings.submission_csv_path, submission_rows)

        metadata_document = build_metadata_document(runtime_settings, submission_timestamp=clock.utc_now_iso_ms())
        write_json_atomic(settings.metadata_json_path, metadata_document)

        validation_summary = compute_validation_summary(records, runtime_settings)
        validation_summary["submission_csv_path"] = str(settings.submission_csv_path)
        validation_summary["metadata_json_path"] = str(settings.metadata_json_path)
        write_json_atomic(validation_path, validation_summary)

        manifest["status"] = "completed"
        manifest["completed_at"] = clock.utc_now_iso_ms()
        manifest["submission_written"] = True
        manifest["validation_path"] = str(validation_path)
        manifest["metadata_written"] = str(settings.metadata_json_path)
        write_json_atomic(manifest_path, manifest)

        return S8RunResult(
            status="completed",
            manifest_path=manifest_path,
            validation_path=validation_path,
            submission_csv_path=settings.submission_csv_path,
            metadata_json_path=settings.metadata_json_path,
            records=records,
            validation_summary=validation_summary,
            manifest=manifest,
        )
    except Exception as exc:
        validation_summary = {
            "status": "aborted",
            "error": str(exc),
            "records_present": int(len(records)),
            "submission_csv_path": str(settings.submission_csv_path),
        }
        write_json_atomic(validation_path, validation_summary)
        manifest["status"] = "aborted"
        manifest["completed_at"] = clock.utc_now_iso_ms()
        manifest["error"] = str(exc)
        manifest["submission_written"] = False
        write_json_atomic(manifest_path, manifest)
        return S8RunResult(
            status="aborted",
            manifest_path=manifest_path,
            validation_path=validation_path,
            records=records,
            validation_summary=validation_summary,
            manifest=manifest,
        )
    finally:
        try:
            controller.close()
        except Exception:
            pass


def load_submission_rows(csv_path: Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def records_from_submission_rows(
    rows: Sequence[Mapping[str, Any]],
    desired_targets: Mapping[int, Sequence[float]],
    *,
    max_motion_time_ms: int,
) -> list[S8PoseRecord]:
    records: list[S8PoseRecord] = []
    for row in rows:
        target_id = int(row["target_id"])
        desired_local_mm = [float(value) for value in desired_targets[target_id]]
        outbound = S8MotionLeg(
            leg_name="outbound",
            source_mode="submission_row",
            target_local_mm=desired_local_mm,
            target_base_m=[0.0, 0.0, 0.0],
            target_rotvec_base=[0.0, 0.0, 0.0],
            start_time=str(row["start_time"]),
            end_time=str(row["end_time"]),
            motion_time_ms=int(float(row["motion_time_ms"])),
            reached=True,
            within_benchmark_time=int(float(row["motion_time_ms"])) <= int(max_motion_time_ms),
            actual_pose_base=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            actual_pose_local_mm=[float(row["x"]), float(row["y"]), float(row["z"])],
            actual_quaternion_local_xyzw=[
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
                float(row["qw"]),
            ],
        )
        return_leg = S8MotionLeg(
            leg_name="return_to_init",
            source_mode="unverified_return",
            target_local_mm=[0.0, 0.0, 0.0],
            target_base_m=[0.0, 0.0, 0.0],
            target_rotvec_base=[0.0, 0.0, 0.0],
            start_time=str(row["end_time"]),
            end_time=str(row["end_time"]),
            motion_time_ms=0,
            reached=True,
            within_benchmark_time=True,
            actual_pose_base=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            actual_pose_local_mm=[0.0, 0.0, 0.0],
            actual_quaternion_local_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        records.append(
            S8PoseRecord(
                repetition=int(row["repetition"]),
                target_id=target_id,
                desired_local_mm=desired_local_mm,
                outbound=outbound,
                return_leg=return_leg,
            )
        )
    return records
