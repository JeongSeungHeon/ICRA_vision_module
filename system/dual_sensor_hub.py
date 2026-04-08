"""Dual-camera RealSense sensor hub.

This module centralizes ownership of two RealSense devices and exposes the
latest per-camera frames plus a near-synchronized frame pair. It is designed as
Step 3 of the receive-and-place architecture and intentionally focuses only on
camera acquisition and lightweight metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional during unit tests
    yaml = None

from object_pt_extraction.dual_realsense_manager import DualRealSenseManager
from system.shared_state import SensorState
from utils.realsense_stream import FrameBundle, list_realsense_serials

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class DualFrameSnapshot:
    pair_index: int
    cam0: FrameBundle
    cam1: FrameBundle
    timestamp_delta_ms: float
    within_sync_tolerance: bool


class DualSensorHub:
    """Own and read two RealSense cameras through one shared interface."""

    def __init__(
        self,
        serial_cam0: str | None = None,
        serial_cam1: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_pair_time_delta_sec: float = 0.03,
        depth_filters_cam0: dict[str, Any] | None = None,
        depth_filters_cam1: dict[str, Any] | None = None,
    ) -> None:
        self.serial_cam0 = serial_cam0
        self.serial_cam1 = serial_cam1
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.max_pair_time_delta_sec = float(max_pair_time_delta_sec)
        self.depth_filters_cam0 = dict(depth_filters_cam0 or {})
        self.depth_filters_cam1 = dict(depth_filters_cam1 or {})

        self._manager: DualRealSenseManager | None = None
        self._pair_index = -1
        self._latest_snapshot: DualFrameSnapshot | None = None
        self._latest_sensor_states: dict[int, SensorState] = {
            0: SensorState(camera_id=0),
            1: SensorState(camera_id=1),
        }

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "DualSensorHub":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load DualSensorHub configuration.")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        cam0_cfg = (config or {}).get("cameras", {}).get("cam0", {})
        cam1_cfg = (config or {}).get("cameras", {}).get("cam1", {})
        sync_cfg = (config or {}).get("perception", {}).get("synchronization", {})
        system_cfg = (config or {}).get("system", {})

        width0 = int(cam0_cfg.get("width", 640))
        width1 = int(cam1_cfg.get("width", width0))
        height0 = int(cam0_cfg.get("height", 480))
        height1 = int(cam1_cfg.get("height", height0))
        fps0 = int(cam0_cfg.get("fps", system_cfg.get("sensor_fps", 30)))
        fps1 = int(cam1_cfg.get("fps", fps0))

        if width0 != width1 or height0 != height1 or fps0 != fps1:
            raise ValueError(
                "DualSensorHub currently requires matching width, height, and fps for cam0 and cam1."
            )

        return cls(
            serial_cam0=cam0_cfg.get("serial"),
            serial_cam1=cam1_cfg.get("serial"),
            width=width0,
            height=height0,
            fps=fps0,
            max_pair_time_delta_sec=float(sync_cfg.get("max_pair_time_delta_sec", 0.03)),
            depth_filters_cam0=dict(cam0_cfg.get("depth_filters", {})),
            depth_filters_cam1=dict(cam1_cfg.get("depth_filters", {})),
        )

    @staticmethod
    def list_available_serials() -> list[str]:
        return list_realsense_serials()

    def start(self) -> "DualSensorHub":
        if self._manager is not None:
            return self

        resolved_serials = self._resolve_serials()
        self._manager = DualRealSenseManager(
            serials=resolved_serials,
            width=self.width,
            height=self.height,
            fps=self.fps,
            anchor_index=0,
            depth_filters_by_serial={
                resolved_serials[0]: self.depth_filters_cam0,
                resolved_serials[1]: self.depth_filters_cam1,
            },
        )
        self.serial_cam0, self.serial_cam1 = self._manager.serials
        return self

    def _resolve_serials(self) -> list[str]:
        requested = [self.serial_cam0, self.serial_cam1]
        available = list_realsense_serials()

        if all(requested):
            return [str(requested[0]), str(requested[1])]

        if len(available) < 2:
            raise RuntimeError("At least two RealSense devices are required for DualSensorHub.")

        resolved: list[str] = []
        for serial in requested:
            if serial:
                resolved.append(str(serial))
                continue
            for candidate in available:
                if candidate not in resolved:
                    resolved.append(candidate)
                    break

        if len(resolved) != 2:
            raise RuntimeError("Failed to resolve two RealSense serial numbers for DualSensorHub.")
        return resolved

    def read_next_pair(self) -> DualFrameSnapshot:
        if self._manager is None:
            self.start()
        assert self._manager is not None

        paired = self._manager.read_paired_frames()
        self._pair_index += 1
        within_tolerance = abs(paired.timestamp_delta_ms) <= self.max_pair_time_delta_sec * 1000.0

        snapshot = DualFrameSnapshot(
            pair_index=self._pair_index,
            cam0=paired.anchor_frame,
            cam1=paired.paired_frame,
            timestamp_delta_ms=float(paired.timestamp_delta_ms),
            within_sync_tolerance=within_tolerance,
        )
        self._latest_snapshot = snapshot
        self._latest_sensor_states[0] = self._build_sensor_state(0, snapshot.pair_index, snapshot.cam0)
        self._latest_sensor_states[1] = self._build_sensor_state(1, snapshot.pair_index, snapshot.cam1)
        return snapshot

    def get_latest_pair(self) -> DualFrameSnapshot:
        if self._latest_snapshot is None:
            return self.read_next_pair()
        return self._latest_snapshot

    def get_latest_cam0(self) -> FrameBundle:
        return self.get_latest_pair().cam0

    def get_latest_cam1(self) -> FrameBundle:
        return self.get_latest_pair().cam1

    def get_latest_sensor_state(self, camera_id: int) -> SensorState:
        if camera_id not in (0, 1):
            raise ValueError("camera_id must be 0 or 1")
        if self._latest_snapshot is None:
            self.read_next_pair()
        return self._latest_sensor_states[camera_id]

    def get_latest_sensor_states(self) -> tuple[SensorState, SensorState]:
        if self._latest_snapshot is None:
            self.read_next_pair()
        return self._latest_sensor_states[0], self._latest_sensor_states[1]

    def stop(self) -> None:
        if self._manager is None:
            return
        self._manager.stop()
        self._manager = None

    def __enter__(self) -> "DualSensorHub":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    @staticmethod
    def _build_sensor_state(camera_id: int, frame_id: int, frame_bundle: FrameBundle) -> SensorState:
        rgb_shape = tuple(int(value) for value in frame_bundle.color_image.shape)
        depth_shape = tuple(int(value) for value in frame_bundle.depth_image_m.shape)
        intrinsics_dict = frame_bundle.intrinsics or {}
        intrinsics = (
            (
                float(intrinsics_dict.get("fx", 0.0)),
                0.0,
                float(intrinsics_dict.get("cx", 0.0)),
            ),
            (
                0.0,
                float(intrinsics_dict.get("fy", 0.0)),
                float(intrinsics_dict.get("cy", 0.0)),
            ),
            (0.0, 0.0, 1.0),
        )
        return SensorState(
            camera_id=camera_id,
            frame_id=frame_id,
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            intrinsics=intrinsics,
            serial_number=frame_bundle.serial,
            timestamp=float(frame_bundle.timestamp_ms) / 1000.0,
            valid=True,
        )


__all__ = ["DualFrameSnapshot", "DualSensorHub", "DEFAULT_CONFIG_PATH"]
