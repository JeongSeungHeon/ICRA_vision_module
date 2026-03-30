from dataclasses import dataclass

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


@dataclass
class FrameBundle:
    color_image: np.ndarray
    depth_image_m: np.ndarray
    intrinsics: dict
    timestamp_ms: float
    serial: str


def _require_realsense():
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for depth mode.")


def list_realsense_serials():
    _require_realsense()
    context = rs.context()
    return [device.get_info(rs.camera_info.serial_number) for device in context.query_devices()]


class RealSenseCamera:
    def __init__(self, serial=None, width=640, height=480, fps=30, depth_filters=None):
        _require_realsense()

        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth_filters_config = dict(depth_filters or {})

        if self.serial:
            self.config.enable_device(self.serial)

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self._spatial_filter = self._build_spatial_filter()
        self._temporal_filter = self._build_temporal_filter()
        self._hole_filling_filter = self._build_hole_filling_filter()

        device = self.profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def _get_intrinsics(self, aligned_frames):
        color_frame = aligned_frames.get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        return {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
        }

    def _build_spatial_filter(self):
        cfg = self.depth_filters_config.get("spatial", {})
        if not cfg.get("enabled", False):
            return None
        filter_obj = rs.spatial_filter()
        if "magnitude" in cfg:
            filter_obj.set_option(rs.option.filter_magnitude, float(cfg["magnitude"]))
        if "smooth_alpha" in cfg:
            filter_obj.set_option(rs.option.filter_smooth_alpha, float(cfg["smooth_alpha"]))
        if "smooth_delta" in cfg:
            filter_obj.set_option(rs.option.filter_smooth_delta, float(cfg["smooth_delta"]))
        return filter_obj

    def _build_temporal_filter(self):
        cfg = self.depth_filters_config.get("temporal", {})
        if not cfg.get("enabled", False):
            return None
        filter_obj = rs.temporal_filter()
        if "smooth_alpha" in cfg:
            filter_obj.set_option(rs.option.filter_smooth_alpha, float(cfg["smooth_alpha"]))
        if "smooth_delta" in cfg:
            filter_obj.set_option(rs.option.filter_smooth_delta, float(cfg["smooth_delta"]))
        return filter_obj

    def _build_hole_filling_filter(self):
        cfg = self.depth_filters_config.get("hole_filling", {})
        if not cfg.get("enabled", False):
            return None
        filter_obj = rs.hole_filling_filter()
        if "mode" in cfg:
            filter_obj.set_option(rs.option.holes_fill, float(cfg["mode"]))
        return filter_obj

    def _apply_depth_filters(self, depth_frame):
        filtered = depth_frame
        if self._spatial_filter is not None:
            filtered = self._spatial_filter.process(filtered)
        if self._temporal_filter is not None:
            filtered = self._temporal_filter.process(filtered)
        if self._hole_filling_filter is not None:
            filtered = self._hole_filling_filter.process(filtered)
        return filtered

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = self._apply_depth_filters(aligned_frames.get_depth_frame())

        if not color_frame or not depth_frame:
            raise RuntimeError(f"Failed to read aligned frames from RealSense {self.serial}.")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_image_m = depth_image * self.depth_scale

        return FrameBundle(
            color_image=color_image,
            depth_image_m=depth_image_m,
            intrinsics=self._get_intrinsics(aligned_frames),
            timestamp_ms=float(frames.get_timestamp()),
            serial=self.serial,
        )

    def stop(self):
        self.pipeline.stop()
