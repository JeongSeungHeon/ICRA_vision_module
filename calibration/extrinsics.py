"""Utilities for loading and applying camera-to-robot extrinsic chains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class TransformChain:
    t_base_cam0: np.ndarray
    t_cam0_cam1: np.ndarray
    t_base_cam1: np.ndarray
    source_config: str

    def transform_points_cam0_to_base(self, points_xyz: np.ndarray) -> np.ndarray:
        return transform_points(points_xyz, self.t_base_cam0)

    def transform_points_cam1_to_base(self, points_xyz: np.ndarray) -> np.ndarray:
        return transform_points(points_xyz, self.t_base_cam1)

    def transform_points_camera_to_base(self, camera_id: int, points_xyz: np.ndarray) -> np.ndarray:
        if int(camera_id) == 0:
            return self.transform_points_cam0_to_base(points_xyz)
        if int(camera_id) == 1:
            return self.transform_points_cam1_to_base(points_xyz)
        raise ValueError(f"Unsupported camera_id: {camera_id}")

    def transform_point_camera_to_base(self, camera_id: int, point_xyz: np.ndarray) -> np.ndarray:
        transformed = self.transform_points_camera_to_base(camera_id, np.asarray(point_xyz, dtype=np.float32).reshape(1, 3))
        return transformed.reshape(3)


def _as_homogeneous_matrix(matrix_like: Any) -> np.ndarray:
    matrix = np.asarray(matrix_like, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float32)
        homogeneous[:3, :] = matrix
        return homogeneous
    if matrix.shape == (3, 3):
        homogeneous = np.eye(4, dtype=np.float32)
        homogeneous[:3, :3] = matrix
        return homogeneous
    if matrix.size == 16:
        return matrix.reshape(4, 4).astype(np.float32)
    raise ValueError(f"Unsupported transform shape: {matrix.shape}")


def _rotation_translation_to_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return matrix


def _read_rotation_translation_text(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rotation_rows: list[list[float]] = []
    translation_rows: list[list[float]] = []

    with open(file_path, "r", encoding="ascii") as handle:
        header = handle.readline().strip()
        if header != "R:":
            raise ValueError(f"Unexpected rotation header in {file_path}: {header}")
        for _ in range(3):
            rotation_rows.append([float(value) for value in handle.readline().split()])

        translation_header = handle.readline().strip()
        if translation_header != "T:":
            raise ValueError(f"Unexpected translation header in {file_path}: {translation_header}")
        for _ in range(3):
            translation_rows.append([float(value) for value in handle.readline().split()])

    rotation = np.asarray(rotation_rows, dtype=np.float32)
    translation = np.asarray(translation_rows, dtype=np.float32).reshape(3)
    return rotation, translation


def load_pickle_transform(file_path: str | Path) -> np.ndarray:
    with open(file_path, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        for key in ("transform", "matrix", "T", "pose", "extrinsic"):
            if key in payload:
                return _as_homogeneous_matrix(payload[key])
        if "R" in payload and "T" in payload:
            return _rotation_translation_to_matrix(payload["R"], payload["T"])
        raise ValueError(f"Unsupported pickle transform keys in {file_path}: {sorted(payload.keys())}")

    return _as_homogeneous_matrix(payload)


def load_dat_transform(file_path: str | Path, translation_unit: str = "m") -> np.ndarray:
    rotation, translation = _read_rotation_translation_text(file_path)
    translation_scale = 1.0 if str(translation_unit).lower() == "m" else 0.001
    return _rotation_translation_to_matrix(rotation, translation * float(translation_scale))


def load_transform_chain(config_path: str | Path = DEFAULT_CONFIG_PATH) -> TransformChain:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    calibration_cfg = config.get("calibration", {}).get("chain", {})
    cam0_cfg = calibration_cfg.get("cam0_to_robot_base", {})
    cam1_cfg = calibration_cfg.get("cam1_to_cam0", {})

    if not cam0_cfg:
        raise ValueError("Missing calibration.chain.cam0_to_robot_base in config")
    if not cam1_cfg:
        raise ValueError("Missing calibration.chain.cam1_to_cam0 in config")

    t_base_cam0 = load_pickle_transform(cam0_cfg["file"])
    t_cam0_cam1 = load_dat_transform(
        cam1_cfg["file"],
        translation_unit=cam1_cfg.get("translation_unit", "m"),
    )
    t_base_cam1 = np.asarray(t_base_cam0 @ t_cam0_cam1, dtype=np.float32)

    return TransformChain(
        t_base_cam0=np.asarray(t_base_cam0, dtype=np.float32),
        t_cam0_cam1=np.asarray(t_cam0_cam1, dtype=np.float32),
        t_base_cam1=t_base_cam1,
        source_config=str(config_path),
    )


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    transform = _as_homogeneous_matrix(transform)
    rotated = points @ transform[:3, :3].T
    translated = rotated + transform[:3, 3].reshape(1, 3)
    return translated.astype(np.float32)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "TransformChain",
    "load_pickle_transform",
    "load_dat_transform",
    "load_transform_chain",
    "transform_points",
]
