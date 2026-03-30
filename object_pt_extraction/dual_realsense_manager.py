from dataclasses import dataclass

from utils.realsense_stream import RealSenseCamera, list_realsense_serials


@dataclass
class PairedFrameBundle:
    anchor_frame: object
    paired_frame: object
    anchor_serial: str
    paired_serial: str
    timestamp_delta_ms: float


class DualRealSenseManager:
    def __init__(self, serials=None, width=640, height=480, fps=30, anchor_index=0, depth_filters_by_serial=None):
        available_serials = list(serials or [])
        if not available_serials:
            available_serials = list_realsense_serials()

        if len(available_serials) < 2:
            raise RuntimeError("At least two RealSense devices are required for dual-camera mode.")

        self.serials = available_serials[:2]
        if anchor_index not in (0, 1):
            raise ValueError("anchor_index must be 0 or 1")

        self.anchor_index = int(anchor_index)
        self.anchor_serial = self.serials[self.anchor_index]
        self.paired_serial = self.serials[1 - self.anchor_index]
        depth_filters_by_serial = dict(depth_filters_by_serial or {})
        self.cameras = {
            serial: RealSenseCamera(
                serial=serial,
                width=width,
                height=height,
                fps=fps,
                depth_filters=depth_filters_by_serial.get(serial),
            )
            for serial in self.serials
        }
        self._paired_latest_frame = None

    def _read_follower_frame_near(self, target_timestamp_ms):
        follower_camera = self.cameras[self.paired_serial]

        if self._paired_latest_frame is None:
            self._paired_latest_frame = follower_camera.read()

        before_frame = self._paired_latest_frame
        if before_frame.timestamp_ms >= target_timestamp_ms:
            return before_frame

        while self._paired_latest_frame.timestamp_ms < target_timestamp_ms:
            before_frame = self._paired_latest_frame
            self._paired_latest_frame = follower_camera.read()

        after_frame = self._paired_latest_frame
        if abs(before_frame.timestamp_ms - target_timestamp_ms) <= abs(after_frame.timestamp_ms - target_timestamp_ms):
            return before_frame
        return after_frame

    def read_paired_frames(self):
        anchor_frame = self.cameras[self.anchor_serial].read()
        paired_frame = self._read_follower_frame_near(anchor_frame.timestamp_ms)
        return PairedFrameBundle(
            anchor_frame=anchor_frame,
            paired_frame=paired_frame,
            anchor_serial=self.anchor_serial,
            paired_serial=self.paired_serial,
            timestamp_delta_ms=float(paired_frame.timestamp_ms - anchor_frame.timestamp_ms),
        )

    def stop(self):
        for camera in self.cameras.values():
            camera.stop()
