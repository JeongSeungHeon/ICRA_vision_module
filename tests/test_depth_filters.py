import unittest

import numpy as np

import utils.depth_filters as depth_filters
from utils.depth_filters import bilateral_filter_depth
from utils.realsense_stream import RealSenseCamera


class DepthFilterTests(unittest.TestCase):
    def test_realsense_bilateral_disabled_returns_original_depth_object(self):
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.depth_filters_config = {"bilateral": {"enabled": False}}
        camera._bilateral_filter_config = camera._build_bilateral_filter_config()
        depth = np.ones((3, 3), dtype=np.float32)

        filtered = camera._apply_bilateral_filter(depth)

        self.assertIs(filtered, depth)

    def test_bilateral_filter_keeps_invalid_depth_zeroed(self):
        if depth_filters.cv2 is None:
            self.skipTest("OpenCV is not installed in the test environment.")
        depth = np.asarray(
            [
                [np.nan, np.inf, 0.0],
                [0.0005, 1.0, 200.0],
            ],
            dtype=np.float32,
        )

        filtered = bilateral_filter_depth(depth, radius=1, zfar=100.0)

        self.assertEqual(filtered.dtype, np.float32)
        self.assertEqual(float(filtered[0, 0]), 0.0)
        self.assertEqual(float(filtered[0, 1]), 0.0)
        self.assertEqual(float(filtered[0, 2]), 0.0)
        self.assertEqual(float(filtered[1, 0]), 0.0)
        self.assertEqual(float(filtered[1, 2]), 0.0)
        self.assertGreater(float(filtered[1, 1]), 0.0)

    def test_bilateral_filter_smooths_valid_depth_values(self):
        if depth_filters.cv2 is None:
            self.skipTest("OpenCV is not installed in the test environment.")
        depth = np.ones((7, 7), dtype=np.float32)
        depth[3, 3] = 1.5

        filtered = bilateral_filter_depth(
            depth,
            radius=2,
            zfar=100.0,
            sigma_space=2.0,
            sigma_color=1.0,
        )

        self.assertEqual(filtered.shape, depth.shape)
        self.assertEqual(filtered.dtype, np.float32)
        self.assertLess(float(filtered[3, 3]), 1.5)
        self.assertGreater(float(filtered[3, 3]), 1.0)
        self.assertGreater(float(filtered[3, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
