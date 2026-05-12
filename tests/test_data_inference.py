import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_inference import (
    build_offline_snapshot,
    find_input_files,
    intrinsics_vector_to_dict,
    select_frame_indices,
    validate_raw_recording,
)


def _raw_recording(frame_count=3):
    return {
        "frame_count": np.asarray(frame_count, dtype=np.int32),
        "cam0_color_image": np.zeros((frame_count, 2, 3, 3), dtype=np.uint8),
        "cam0_depth_image_m": np.ones((frame_count, 2, 3), dtype=np.float32),
        "cam0_intrinsics": np.tile(np.asarray([100.0, 101.0, 50.0, 51.0], dtype=np.float32), (frame_count, 1)),
        "cam0_timestamp_ms": np.asarray([10.0, 20.0, 30.0], dtype=np.float32)[:frame_count],
        "cam0_serial": np.asarray(["cam0_a", "cam0_b", "cam0_c"])[:frame_count],
        "cam1_color_image": np.full((frame_count, 2, 3, 3), 7, dtype=np.uint8),
        "cam1_depth_image_m": np.full((frame_count, 2, 3), 2.0, dtype=np.float32),
        "cam1_intrinsics": np.tile(np.asarray([200.0, 201.0, 60.0, 61.0], dtype=np.float32), (frame_count, 1)),
        "cam1_timestamp_ms": np.asarray([11.0, 21.0, 31.0], dtype=np.float32)[:frame_count],
        "cam1_serial": np.asarray(["cam1_a", "cam1_b", "cam1_c"])[:frame_count],
        "timestamp_delta_ms": np.asarray([1.0, 1.0, 1.0], dtype=np.float32)[:frame_count],
        "within_sync_tolerance": np.asarray([True, True, False])[:frame_count],
        "eef_pose_base": np.asarray(
            [
                [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.4, 0.5, 0.6, 0.0, 0.1, 0.2],
            ],
            dtype=np.float32,
        )[:frame_count],
    }


class DataInferenceTests(unittest.TestCase):
    def test_validate_raw_recording_requires_saved_images(self):
        data = _raw_recording()
        del data["cam1_depth_image_m"]

        with self.assertRaisesRegex(ValueError, "--3d-debug --save-image"):
            validate_raw_recording(data, "recording.npz")

    def test_validate_raw_recording_checks_frame_count(self):
        data = _raw_recording()
        data["cam0_depth_image_m"] = data["cam0_depth_image_m"][:2]

        with self.assertRaisesRegex(ValueError, "inconsistent frame count"):
            validate_raw_recording(data, "recording.npz")

    def test_select_frame_indices_applies_bounds_stride_and_max(self):
        self.assertEqual(
            select_frame_indices(10, start_frame=2, end_frame=9, stride=3, max_frames=2),
            [2, 5],
        )
        self.assertEqual(select_frame_indices(4, start_frame=5), [])

    def test_intrinsics_vector_to_dict(self):
        self.assertEqual(
            intrinsics_vector_to_dict(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
            {"fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0},
        )

    def test_build_offline_snapshot_reconstructs_frame_bundles(self):
        data = _raw_recording()

        snapshot = build_offline_snapshot(data, source_index=2, replay_index=0, fps=30)

        self.assertEqual(snapshot.pair_index, 2)
        self.assertEqual(snapshot.cam0.color_image.shape, (2, 3, 3))
        self.assertEqual(snapshot.cam0.intrinsics["fx"], 100.0)
        self.assertEqual(snapshot.cam1.intrinsics["cy"], 61.0)
        self.assertEqual(snapshot.cam0.timestamp_ms, 30.0)
        self.assertEqual(snapshot.cam1.serial, "cam1_c")
        self.assertFalse(snapshot.within_sync_tolerance)

    def test_find_input_files_accepts_file_and_sorts_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a_path = root / "a.npz"
            b_path = root / "b.npz"
            a_path.touch()
            b_path.touch()

            self.assertEqual(find_input_files(b_path), [b_path])
            self.assertEqual(find_input_files(root), [a_path, b_path])


if __name__ == "__main__":
    unittest.main()
