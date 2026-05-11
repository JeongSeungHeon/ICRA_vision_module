import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from perception.object_worker import ObjectWorker, TemporalClassLocker
from utils.realsense_stream import FrameBundle


class TemporalClassLockerTests(unittest.TestCase):
    def test_locks_after_stable_segmented_labels(self):
        locker = TemporalClassLocker(stable_frames=3)

        self.assertEqual(locker.update("cup", segmented=True), "cup")
        self.assertEqual(locker.update("cup", segmented=True), "cup")
        self.assertEqual(locker.update("cup", segmented=True), "cup")

        self.assertEqual(locker.locked_label, "cup")
        self.assertEqual(locker.update("wine glass", segmented=True), "cup")
        self.assertEqual(locker.update(None, segmented=False), "cup")

    def test_label_change_before_lock_delays_locking(self):
        locker = TemporalClassLocker(stable_frames=3)

        locker.update("cup", segmented=True)
        locker.update("wine glass", segmented=True)
        locker.update("wine glass", segmented=True)

        self.assertIsNone(locker.locked_label)
        self.assertEqual(locker.update("wine glass", segmented=True), "wine glass")
        self.assertEqual(locker.locked_label, "wine glass")

    def test_reset_clears_history_and_lock(self):
        locker = TemporalClassLocker(stable_frames=2)

        locker.update("box-shaped snack", segmented=True)
        locker.update("box-shaped snack", segmented=True)
        locker.reset()

        self.assertIsNone(locker.locked_label)
        self.assertEqual(tuple(locker.history), ())


class ObjectWorkerClassLockLogTests(unittest.TestCase):
    def test_prints_once_when_class_lock_is_acquired(self):
        segmentation_engine = SimpleNamespace()
        transform_chain = SimpleNamespace(
            transform_points_camera_to_base=lambda camera_id, points: np.asarray(points, dtype=np.float32)
        )
        worker = ObjectWorker(
            camera_id=1,
            segmentation_engine=segmentation_engine,
            transform_chain=transform_chain,
            config={
                "perception": {
                    "object": {
                        "confidence_threshold": 0.1,
                        "point_cloud": {"min_points_per_camera": 1},
                    }
                }
            },
        )
        worker.class_locker = TemporalClassLocker(stable_frames=2)
        instance = SimpleNamespace(class_name="wine glass", score=0.9)
        segmentation_result = SimpleNamespace(instances=[instance], infer_ms=1.0)
        segmentation_engine.predict = lambda image: segmentation_result
        frame = FrameBundle(
            color_image=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_image_m=np.ones((2, 2), dtype=np.float32),
            intrinsics={},
            timestamp_ms=1000.0,
            serial="test",
        )

        with patch(
            "perception.object_worker.build_point_cloud_from_instances",
            return_value=(
                np.ones((2, 2), dtype=bool),
                np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
                None,
                None,
                np.asarray([[255, 255, 255]], dtype=np.uint8),
            ),
        ), patch("builtins.print") as mock_print:
            worker.process_frame(frame, frame_id=10)
            worker.process_frame(frame, frame_id=11)
            worker.process_frame(frame, frame_id=12)

        self.assertEqual(mock_print.call_count, 1)
        printed_text = " ".join(str(part) for part in mock_print.call_args.args)
        self.assertIn("[CLASS_LOCK]", printed_text)
        self.assertIn("cam1", printed_text)
        self.assertIn("frame=11", printed_text)
        self.assertIn("locked_label=wine glass", printed_text)


if __name__ == "__main__":
    unittest.main()
