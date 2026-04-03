"""ZED Mini RGB/depth stream adapter.

This module provides a small runtime wrapper that exposes ZED frames through
the same ``FrameBundle`` contract used by the existing RealSense-based
perception stack.
"""

from __future__ import annotations

from typing import Any

import cv2 as cv
import numpy as np

from utils.realsense_stream import FrameBundle

try:
    import pyzed.sl as sl
except ImportError:
    sl = None


def require_zed() -> None:
    if sl is None:
        raise RuntimeError(
            "pyzed is required for ZED camera support. "
            "Install the Stereolabs ZED SDK Python API in the active environment."
        )


def list_zed_serials() -> list[str]:
    require_zed()
    if not hasattr(sl.Camera, "get_device_list"):
        return []
    devices = sl.Camera.get_device_list()
    serials: list[str] = []
    for device in devices:
        serial_number = getattr(device, "serial_number", None)
        if serial_number is not None:
            serials.append(str(serial_number))
    return serials


def _enum_value(enum_group: Any, name: str | None, default_name: str):
    candidate = default_name if name is None else str(name).strip().upper().replace("-", "_").replace(" ", "_")
    if hasattr(enum_group, candidate):
        return getattr(enum_group, candidate)
    fallback = str(default_name).strip().upper().replace("-", "_").replace(" ", "_")
    if hasattr(enum_group, fallback):
        return getattr(enum_group, fallback)
    raise ValueError(f"Unsupported enum value `{name}` for {enum_group}.")


class ZedCamera:
    """Read left RGB and aligned depth from a ZED camera."""

    def __init__(
        self,
        serial: str | None = None,
        resolution: str = "HD720",
        fps: int = 30,
        depth_mode: str = "NEURAL",
    ) -> None:
        require_zed()

        self.serial = None if serial in {None, "", "null"} else str(serial)
        self.resolution = str(resolution)
        self.fps = int(fps)
        self.depth_mode = str(depth_mode)

        self.camera = sl.Camera()
        self.runtime_parameters = sl.RuntimeParameters()
        self._left_image = sl.Mat()
        self._depth_measure = sl.Mat()

        init_params = sl.InitParameters()
        init_params.camera_resolution = _enum_value(sl.RESOLUTION, self.resolution, "HD720")
        init_params.camera_fps = self.fps
        init_params.depth_mode = _enum_value(sl.DEPTH_MODE, self.depth_mode, "NEURAL")
        init_params.coordinate_units = sl.UNIT.METER
        if hasattr(sl.COORDINATE_SYSTEM, "IMAGE"):
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        if self.serial is not None:
            try:
                init_params.set_from_serial_number(int(self.serial))
            except Exception as exc:
                raise RuntimeError(f"Invalid ZED serial number `{self.serial}`.") from exc

        open_status = self.camera.open(init_params)
        if open_status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {open_status}")

        self.serial = self._read_serial_number()
        self.intrinsics = self._read_left_intrinsics()

    def _read_serial_number(self) -> str:
        camera_info = self.camera.get_camera_information()
        serial_number = getattr(camera_info, "serial_number", None)
        if serial_number is None and hasattr(camera_info, "camera_configuration"):
            serial_number = getattr(camera_info.camera_configuration, "serial_number", None)
        return "zed" if serial_number is None else str(serial_number)

    def _read_left_intrinsics(self) -> dict[str, float]:
        camera_info = self.camera.get_camera_information()
        calibration = None
        if hasattr(camera_info, "camera_configuration"):
            calibration = getattr(camera_info.camera_configuration, "calibration_parameters", None)
        if calibration is None:
            calibration = getattr(camera_info, "calibration_parameters", None)
        if calibration is None or not hasattr(calibration, "left_cam"):
            raise RuntimeError("Failed to read left camera intrinsics from ZED SDK.")

        left_cam = calibration.left_cam
        return {
            "fx": float(left_cam.fx),
            "fy": float(left_cam.fy),
            "cx": float(left_cam.cx),
            "cy": float(left_cam.cy),
        }

    @staticmethod
    def _normalize_color_image(image_data: np.ndarray) -> np.ndarray:
        color_image = np.asarray(image_data)
        if color_image.ndim == 3 and color_image.shape[2] == 4:
            return cv.cvtColor(color_image, cv.COLOR_BGRA2BGR)
        if color_image.ndim == 3 and color_image.shape[2] == 3:
            return color_image.copy()
        raise RuntimeError(f"Unexpected ZED color image shape: {color_image.shape}")

    @staticmethod
    def _normalize_depth_image(depth_data: np.ndarray) -> np.ndarray:
        depth_image = np.asarray(depth_data, dtype=np.float32)
        if depth_image.ndim == 3:
            depth_image = depth_image[..., 0]
        depth_image = depth_image.astype(np.float32, copy=False)
        depth_image[~np.isfinite(depth_image)] = np.nan
        depth_image[depth_image <= 0.0] = np.nan
        return depth_image

    def read(self) -> FrameBundle:
        grab_status = self.camera.grab(self.runtime_parameters)
        if grab_status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab a frame from ZED camera {self.serial}: {grab_status}")

        self.camera.retrieve_image(self._left_image, sl.VIEW.LEFT)
        self.camera.retrieve_measure(self._depth_measure, sl.MEASURE.DEPTH)

        color_image = self._normalize_color_image(self._left_image.get_data())
        depth_image_m = self._normalize_depth_image(self._depth_measure.get_data())
        timestamp_ms = float(self.camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())

        return FrameBundle(
            color_image=color_image,
            depth_image_m=depth_image_m,
            intrinsics=dict(self.intrinsics),
            timestamp_ms=timestamp_ms,
            serial=str(self.serial),
        )

    def stop(self) -> None:
        self.camera.close()


__all__ = ["ZedCamera", "list_zed_serials", "require_zed"]
