"""Helpers for organizer-style handover metadata CSV logging."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_METADATA_CSV_PATH = Path("output/handover_metadata.csv")
DEFAULT_GEOMETRY_BAND_RATIO = 0.08
METADATA_COLUMNS = [
    "config_id",
    "robot_initial_pose_x",
    "robot_initial_pose_y",
    "robot_initial_pose_z",
    "robot_initial_pose_q1",
    "robot_initial_pose_q2",
    "robot_initial_pose_q3",
    "robot_initial_pose_q4",
    "initial_mass_measured_g",
    "width_top_est_mm_vision",
    "width_bottom_est_mm_vision",
    "height_est_mm_vision",
    "geometry_est_timepoint",
    "spill_observed_during_human_maneuvering",
    "robot_mass_est_available",
    "robot_mass_est_g",
    "robot_mass_est_timepoint",
    "delivery_location_est_x_mm",
    "delivery_location_est_y_mm",
    "delivery_location_est_z_mm",
    "final_mass_null_flag",
    "final_mass_measured_g",
    "t_human_first_contact_ms",
    "t_human_last_contact_ms",
    "t_robot_first_contact_ms",
    "t_robot_last_contact_ms",
]


def utc_now_iso_ms():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def elapsed_ms_between_iso(start_time_iso, end_time_iso):
    if not start_time_iso or not end_time_iso:
        return None
    start_dt = datetime.fromisoformat(str(start_time_iso).replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(str(end_time_iso).replace("Z", "+00:00"))
    return int((end_dt - start_dt).total_seconds() * 1000)


def rotvec_to_quaternion_xyzw(rotvec):
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)

    axis = vector / theta
    half_theta = 0.5 * theta
    sin_half = float(np.sin(half_theta))
    cos_half = float(np.cos(half_theta))
    quat = np.array(
        [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half,
        ],
        dtype=np.float64,
    )
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    quat /= quat_norm
    return tuple(float(v) for v in quat)


def _sample_points_for_diameter(points_xy, max_points=128):
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2))
    if len(points_xy) <= max_points:
        return points_xy
    sample_indices = np.linspace(0, len(points_xy) - 1, max_points, dtype=np.int32)
    return points_xy[sample_indices]


def _estimate_band_diameter_mm(points_xy):
    points_xy = _sample_points_for_diameter(points_xy)
    if len(points_xy) == 0:
        return None
    if len(points_xy) == 1:
        return 0.0
    deltas = points_xy[:, None, :] - points_xy[None, :, :]
    distances_mm = np.sqrt(np.sum(deltas * deltas, axis=2)) * 1000.0
    return float(np.max(distances_mm))


def estimate_geometry_from_fitted_points_mm(fitted_points_base, band_ratio=DEFAULT_GEOMETRY_BAND_RATIO):
    points = np.asarray(fitted_points_base, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return None

    z_values = points[:, 2]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    height_m = z_max - z_min
    if not np.isfinite(height_m) or height_m <= 1e-9:
        return None

    band_height_m = max(height_m * float(band_ratio), 1e-6)
    top_mask = z_values >= (z_max - band_height_m)
    bottom_mask = z_values <= (z_min + band_height_m)
    top_points_xy = points[top_mask, :2]
    bottom_points_xy = points[bottom_mask, :2]
    if len(top_points_xy) == 0:
        top_points_xy = points[:, :2]
    if len(bottom_points_xy) == 0:
        bottom_points_xy = points[:, :2]

    top_width_mm = _estimate_band_diameter_mm(top_points_xy)
    bottom_width_mm = _estimate_band_diameter_mm(bottom_points_xy)
    if top_width_mm is None or bottom_width_mm is None:
        return None

    return {
        "width_top_est_mm_vision": top_width_mm,
        "width_bottom_est_mm_vision": bottom_width_mm,
        "height_est_mm_vision": float(height_m * 1000.0),
    }


def _format_metadata_float(value):
    if value is None:
        return ""
    value = float(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.6f}"


def _format_metadata_int(value):
    if value is None:
        return ""
    return str(int(value))


class HandoverMetadataRecorder:
    def __init__(self, csv_path=DEFAULT_METADATA_CSV_PATH):
        self.csv_path = Path(csv_path)
        self._lock = threading.Lock()
        self._task_start_perf = None
        self._task_start_timestamp_iso = None
        self._config_id_by_task_timestamp = {}
        self._row_index_by_task_timestamp = {}
        self._initial_pose_base = None
        self._geometry_fields = {}
        self._geometry_timepoint_ms = None
        self._geometry_timestamp_iso = None
        self._delivery_location_mm = {}
        self._robot_first_contact_ms = None
        self._robot_first_contact_timestamp_iso = None
        self._robot_last_contact_ms = None
        self._robot_last_contact_timestamp_iso = None
        self._row_written = False

    def mark_task_ready(self, shared_state, *, now_perf=None, now_timestamp_iso=None):
        snapshot = shared_state.get_snapshot()
        with self._lock:
            self._task_start_perf = time.perf_counter() if now_perf is None else float(now_perf)
            self._task_start_timestamp_iso = utc_now_iso_ms() if now_timestamp_iso is None else str(now_timestamp_iso)
            self._initial_pose_base = snapshot.get("initial_pose_base")
            self._geometry_fields = {}
            self._geometry_timepoint_ms = None
            self._geometry_timestamp_iso = None
            self._delivery_location_mm = {}
            self._robot_first_contact_ms = None
            self._robot_first_contact_timestamp_iso = None
            self._robot_last_contact_ms = None
            self._robot_last_contact_timestamp_iso = None
            self._row_written = False
            return self._task_start_timestamp_iso

    def attach_config_id(self, config_id, *, task_start_timestamp_iso=None):
        if config_id is None or str(config_id).strip() == "":
            return False

        task_timestamp = self._task_start_timestamp_iso if task_start_timestamp_iso is None else str(task_start_timestamp_iso)
        if not task_timestamp:
            return False

        normalized_config_id = int(config_id)
        with self._lock:
            self._config_id_by_task_timestamp[task_timestamp] = normalized_config_id
            row_index = self._row_index_by_task_timestamp.get(task_timestamp)
            if row_index is None or not self.csv_path.exists():
                return False

            with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            if row_index < 0 or row_index >= len(rows):
                return False

            rows[row_index]["config_id"] = _format_metadata_int(normalized_config_id)
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            return True

    def _elapsed_ms_locked(self, now_perf=None, now_timestamp_iso=None):
        if self._task_start_timestamp_iso is not None:
            sample_timestamp_iso = utc_now_iso_ms() if now_timestamp_iso is None else str(now_timestamp_iso)
            elapsed_ms = elapsed_ms_between_iso(self._task_start_timestamp_iso, sample_timestamp_iso)
            if elapsed_ms is not None:
                return max(int(elapsed_ms), 0)
        if self._task_start_perf is None or now_perf is None:
            return None
        return max(int(round((float(now_perf) - float(self._task_start_perf)) * 1000.0)), 0)

    def update_geometry(self, shape_fitting_state, *, now_perf=None, now_timestamp_iso=None):
        if shape_fitting_state is None or not getattr(shape_fitting_state, "valid", False):
            return False
        geometry_fields = estimate_geometry_from_fitted_points_mm(
            getattr(shape_fitting_state, "fitted_points_base", None)
        )
        if not geometry_fields:
            return False
        sample_perf = time.perf_counter() if now_perf is None else float(now_perf)
        sample_timestamp_iso = utc_now_iso_ms() if now_timestamp_iso is None else str(now_timestamp_iso)
        with self._lock:
            elapsed_ms = self._elapsed_ms_locked(sample_perf, sample_timestamp_iso)
            if elapsed_ms is None:
                return False
            self._geometry_fields = dict(geometry_fields)
            self._geometry_timepoint_ms = elapsed_ms
            self._geometry_timestamp_iso = sample_timestamp_iso
        return True

    def note_robot_first_contact(self, *, now_perf=None, now_timestamp_iso=None):
        sample_perf = time.perf_counter() if now_perf is None else float(now_perf)
        sample_timestamp_iso = utc_now_iso_ms() if now_timestamp_iso is None else str(now_timestamp_iso)
        with self._lock:
            if self._robot_first_contact_ms is not None:
                return self._robot_first_contact_timestamp_iso
            elapsed_ms = self._elapsed_ms_locked(sample_perf, sample_timestamp_iso)
            if elapsed_ms is not None:
                self._robot_first_contact_ms = elapsed_ms
                self._robot_first_contact_timestamp_iso = sample_timestamp_iso
            return self._robot_first_contact_timestamp_iso

    def note_robot_last_contact(self, *, now_perf=None, now_timestamp_iso=None):
        sample_perf = time.perf_counter() if now_perf is None else float(now_perf)
        sample_timestamp_iso = utc_now_iso_ms() if now_timestamp_iso is None else str(now_timestamp_iso)
        with self._lock:
            elapsed_ms = self._elapsed_ms_locked(sample_perf, sample_timestamp_iso)
            if elapsed_ms is not None:
                self._robot_last_contact_ms = elapsed_ms
                self._robot_last_contact_timestamp_iso = sample_timestamp_iso
            return self._robot_last_contact_timestamp_iso

    def note_delivery_location(self, object_centroid_xyz_mm):
        if object_centroid_xyz_mm is None:
            return
        centroid = np.asarray(object_centroid_xyz_mm, dtype=np.float32).reshape(3)
        with self._lock:
            delivery_z = float(centroid[2])
            height_mm = self._geometry_fields.get("height_est_mm_vision")
            if height_mm is not None:
                delivery_z -= 0.5 * float(height_mm)
            self._delivery_location_mm = {
                "delivery_location_est_x_mm": float(centroid[0]),
                "delivery_location_est_y_mm": float(centroid[1]),
                "delivery_location_est_z_mm": float(delivery_z),
            }

    def record_completion(self):
        with self._lock:
            if self._task_start_perf is None or self._row_written:
                return None

            task_timestamp = self._task_start_timestamp_iso
            row = {column: "" for column in METADATA_COLUMNS}
            initial_pose = self._initial_pose_base
            if initial_pose is not None and len(initial_pose) >= 6:
                pose_values = tuple(float(v) for v in initial_pose[:6])
                quaternion = rotvec_to_quaternion_xyzw(pose_values[3:6])
                row.update(
                    {
                        "robot_initial_pose_x": _format_metadata_float(pose_values[0] * 1000.0),
                        "robot_initial_pose_y": _format_metadata_float(pose_values[1] * 1000.0),
                        "robot_initial_pose_z": _format_metadata_float(pose_values[2] * 1000.0),
                        "robot_initial_pose_q1": _format_metadata_float(quaternion[0]),
                        "robot_initial_pose_q2": _format_metadata_float(quaternion[1]),
                        "robot_initial_pose_q3": _format_metadata_float(quaternion[2]),
                        "robot_initial_pose_q4": _format_metadata_float(quaternion[3]),
                    }
                )

            for key, value in self._geometry_fields.items():
                row[key] = _format_metadata_float(value)
            if self._geometry_timepoint_ms is not None:
                row["geometry_est_timepoint"] = f"single_ms:{int(self._geometry_timepoint_ms)}"
            row["robot_mass_est_available"] = "0"
            row["robot_mass_est_g"] = "-1"

            config_id = self._config_id_by_task_timestamp.get(task_timestamp)
            if config_id is not None:
                row["config_id"] = _format_metadata_int(config_id)

            for key, value in self._delivery_location_mm.items():
                row[key] = _format_metadata_float(value)

            row["t_robot_first_contact_ms"] = _format_metadata_int(self._robot_first_contact_ms)
            row["t_robot_last_contact_ms"] = _format_metadata_int(self._robot_last_contact_ms)

            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            existing_row_count = 0
            if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                    existing_row_count = sum(1 for _ in csv.DictReader(handle))
            should_write_header = existing_row_count == 0
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
                if should_write_header:
                    writer.writeheader()
                writer.writerow(row)

            if task_timestamp:
                self._row_index_by_task_timestamp[task_timestamp] = existing_row_count
            self._row_written = True
            return self.csv_path
