import unittest
from unittest import mock

import numpy as np

from perception.fill_level_estimator import CUP_FILL_HEIGHT_BIAS_MM, FillLevelEstimator, cv, get_adaptive_container_mask


def make_camera_to_base_y_as_z() -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return transform


@unittest.skipIf(cv is None, "OpenCV is not installed in the test environment.")
class FillLevelEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = FillLevelEstimator()
        self.intrinsics = {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 40.0,
            "cy": 60.0,
        }
        self.camera_to_base = make_camera_to_base_y_as_z()

    def test_estimate_fill_level_from_cam0_returns_valid_for_cup(self) -> None:
        image = np.full((120, 80, 3), 190, dtype=np.uint8)
        mask = np.zeros((120, 80), dtype=bool)
        mask[10:110, 20:60] = True

        rng = np.random.default_rng(0)
        textured_fill = rng.integers(40, 140, size=(45, 40), dtype=np.uint8)
        for channel in range(3):
            image[65:110, 20:60, channel] = textured_fill

        depth = np.full((120, 80), 1.0, dtype=np.float32)

        result = self.estimator.estimate_fill_level_from_cam0(
            color_image_bgr=image,
            depth_image_m=depth,
            intrinsics=self.intrinsics,
            container_mask=mask,
            camera_to_base=self.camera_to_base,
            label="cup",
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.mask_mode, "raw")
        self.assertIsNotNone(result.fill_height_mm)
        self.assertGreater(float(result.fill_height_mm), 0.0)

    def test_get_adaptive_container_mask_returns_bowl_only_for_glass_shape(self) -> None:
        mask = np.zeros((120, 80), dtype=bool)
        mask[15:75, 18:62] = True
        mask[75:115, 34:46] = True

        usable_mask, mask_mode = get_adaptive_container_mask(mask)

        self.assertEqual(mask_mode, "bowl_only")
        self.assertTrue(np.any(usable_mask[15:75]))
        self.assertFalse(np.any(usable_mask[90:115]))

    def test_estimate_fill_level_from_cam0_invalid_without_mask(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        depth = np.ones((80, 80), dtype=np.float32)

        result = self.estimator.estimate_fill_level_from_cam0(
            color_image_bgr=image,
            depth_image_m=depth,
            intrinsics=self.intrinsics,
            container_mask=None,
            camera_to_base=self.camera_to_base,
            label="cup",
        )

        self.assertFalse(result.valid)
        self.assertIsNone(result.fill_height_mm)

    def test_estimate_fill_level_from_cam0_adds_bias_for_cup_only(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        depth = np.ones((20, 20), dtype=np.float32)

        with mock.patch("perception.fill_level_estimator.build_inner_roi_mask_from_cup_mask", return_value=mask):
            with mock.patch(
                "perception.fill_level_estimator.compute_row_features",
                return_value=(np.array([5, 6, 7], dtype=np.int32), np.ones((3, 6), dtype=np.float32)),
            ):
                with mock.patch("perception.fill_level_estimator.segment_vertical_profile", return_value={"mock": True}):
                    with mock.patch("perception.fill_level_estimator.estimate_rice_surface_y_center", return_value=6):
                        with mock.patch(
                            "perception.fill_level_estimator.estimate_fill_height_metrics",
                            return_value={"fill_h_mm_axis": 30.0},
                        ):
                            cup_result = self.estimator.estimate_fill_level_from_cam0(
                                color_image_bgr=image,
                                depth_image_m=depth,
                                intrinsics=self.intrinsics,
                                container_mask=mask,
                                camera_to_base=self.camera_to_base,
                                label="cup",
                            )
                            wine_result = self.estimator.estimate_fill_level_from_cam0(
                                color_image_bgr=image,
                                depth_image_m=depth,
                                intrinsics=self.intrinsics,
                                container_mask=mask,
                                camera_to_base=self.camera_to_base,
                                label="wine glass",
                            )

        self.assertTrue(cup_result.valid)
        self.assertTrue(wine_result.valid)
        self.assertAlmostEqual(float(cup_result.fill_height_mm), 30.0 + CUP_FILL_HEIGHT_BIAS_MM, places=5)
        self.assertAlmostEqual(float(wine_result.fill_height_mm), 30.0, places=5)

    def test_estimate_fill_level_from_cam0_forced_empty_overrides_bias(self) -> None:
        estimator = FillLevelEstimator(fillings_empty=True)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        depth = np.ones((20, 20), dtype=np.float32)

        result = estimator.estimate_fill_level_from_cam0(
            color_image_bgr=image,
            depth_image_m=depth,
            intrinsics=self.intrinsics,
            container_mask=mask,
            camera_to_base=self.camera_to_base,
            label="cup",
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.mask_mode, "forced_empty")
        self.assertEqual(float(result.fill_height_mm), 0.0)
        self.assertIsNone(result.rice_top_y_center)
        self.assertIsNone(result.bottom_center_y)


if __name__ == "__main__":
    unittest.main()
