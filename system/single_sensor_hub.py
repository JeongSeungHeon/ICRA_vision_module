"""Single-camera sensor hub for ZED-based verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from system.shared_state import SensorState
from utils.realsense_stream import FrameBundle
from utils.zed_stream import ZedCamera, list_zed_serials

DEFAULT_CONFIG_PATH = Path("configs/handover_zed_single.yaml")


@dataclass
class SingleFrameSnapshot:
    frame_id: int
    cam0: FrameBundle


class SingleSensorHub:
    """Minimal single-camera hub used by the ZED verification tools."""

    def __init__(
        self,
        *,
        backend: str = "zed",
        serial_cam0: str | None = None,
        resolution: str = "HD720",
        fps: int = 30,
        depth_mode: str = "NEURAL",
    ) -> None:
        self.backend = str(backend).strip().lower()
        self.serial_cam0 = serial_cam0
        self.resolution = str(resolution)
        self.fps = int(fps)
        self.depth_mode = str(depth_mode)

        self._camera: Any | None = None
        self._frame_id = -1
        self._latest_snapshot: SingleFrameSnapshot | None = None
        self._latest_sensor_state = SensorState(camera_id=0)

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "SingleSensorHub":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        system_cfg = config.get("system", {})
        camera_cfg = config.get("cameras", {}).get("cam0", {})
        return cls(
            backend=str(camera_cfg.get("backend") or system_cfg.get("camera_backend") or "zed"),
            serial_cam0=camera_cfg.get("serial"),
            resolution=str(camera_cfg.get("resolution", "HD720")),
            fps=int(camera_cfg.get("fps", system_cfg.get("sensor_fps", 30))),
            depth_mode=str(camera_cfg.get("depth_mode", "NEURAL")),
        )

    @staticmethod
    def list_available_serials(backend: str = "zed") -> list[str]:
        normalized = str(backend).strip().lower()
        if normalized == "zed":
            return list_zed_serials()
        raise ValueError(f"Unsupported backend: {backend}")

    def start(self) -> "SingleSensorHub":
        if self._camera is not None:
            return self
        if self.backend != "zed":
            raise ValueError(f"Unsupported backend for SingleSensorHub: {self.backend}")

        self._camera = ZedCamera(
            serial=self.serial_cam0,
            resolution=self.resolution,
            fps=self.fps,
            depth_mode=self.depth_mode,
        )
        self.serial_cam0 = self._camera.serial
        return self

    def read(self) -> SingleFrameSnapshot:
        if self._camera is None:
            self.start()
        assert self._camera is not None

        frame_bundle = self._camera.read()
        self._frame_id += 1
        snapshot = SingleFrameSnapshot(frame_id=self._frame_id, cam0=frame_bundle)
        self._latest_snapshot = snapshot
        self._latest_sensor_state = self._build_sensor_state(snapshot.frame_id, frame_bundle)
        return snapshot

    def get_latest_frame(self) -> FrameBundle:
        if self._latest_snapshot is None:
            return self.read().cam0
        return self._latest_snapshot.cam0

    def get_latest_sensor_state(self) -> SensorState:
        if self._latest_snapshot is None:
            self.read()
        return self._latest_sensor_state

    def stop(self) -> None:
        if self._camera is None:
            return
        self._camera.stop()
        self._camera = None

    def __enter__(self) -> "SingleSensorHub":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    @staticmethod
    def _build_sensor_state(frame_id: int, frame_bundle: FrameBundle) -> SensorState:
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
            camera_id=0,
            frame_id=int(frame_id),
            rgb_shape=tuple(int(value) for value in frame_bundle.color_image.shape),
            depth_shape=tuple(int(value) for value in frame_bundle.depth_image_m.shape),
            intrinsics=intrinsics,
            serial_number=str(frame_bundle.serial),
            timestamp=float(frame_bundle.timestamp_ms) / 1000.0,
            valid=True,
        )


__all__ = ["DEFAULT_CONFIG_PATH", "SingleFrameSnapshot", "SingleSensorHub"]
