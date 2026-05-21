import unittest

import numpy as np

from perception.silhouette_constraint import (
    SilhouetteObservation,
    compute_mask_iou,
    compute_outside_distance_loss,
    compute_silhouette_score,
    project_points_to_image,
    rasterize_projected_template,
)


class SilhouetteConstraintTests(unittest.TestCase):
    def test_project_points_to_image_uses_intrinsics(self):
        points_cam = np.asarray([[0.0, 0.0, 1.0], [0.1, -0.1, 1.0]], dtype=np.float32)
        projected = project_points_to_image(
            points_cam,
            {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 60.0},
            (120, 140),
        )

        np.testing.assert_allclose(projected[0], [50.0, 60.0])
        np.testing.assert_allclose(projected[1], [60.0, 50.0])

    def test_rasterized_mask_has_nonzero_pixels(self):
        rendered = rasterize_projected_template(
            np.asarray([[10.0, 10.0], [12.0, 12.0]], dtype=np.float32),
            (32, 32),
            point_radius_px=1,
            close_kernel_px=0,
            dilate_px=0,
        )

        self.assertGreater(int(np.count_nonzero(rendered)), 0)

    def test_iou_of_identical_masks_is_one(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:10, 5:11] = True

        self.assertEqual(compute_mask_iou(mask, mask.copy()), 1.0)

    def test_outside_distance_loss_is_zero_when_rendered_inside_segmentation(self):
        segmentation = np.zeros((32, 32), dtype=bool)
        segmentation[8:24, 8:24] = True
        rendered = np.zeros_like(segmentation)
        rendered[12:18, 12:18] = True

        self.assertEqual(compute_outside_distance_loss(rendered, segmentation), 0.0)

    def test_oversized_candidate_scores_worse_than_inside_candidate(self):
        segmentation = np.zeros((100, 100), dtype=bool)
        segmentation[40:61, 40:61] = True
        observation = SilhouetteObservation(
            camera_id=0,
            mask=segmentation,
            intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
            t_base_cam=np.eye(4, dtype=np.float32),
            image_shape=(100, 100),
        )
        inside_points = np.asarray(
            [
                [-0.06, -0.06, 1.0],
                [-0.06, 0.06, 1.0],
                [0.06, -0.06, 1.0],
                [0.06, 0.06, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        oversized_points = inside_points.copy()
        oversized_points[:, :2] *= 3.0

        inside_score = compute_silhouette_score(
            inside_points,
            [observation],
            point_radius_px=5,
            close_kernel_px=7,
            dilate_px=1,
            min_rendered_pixels=1,
            min_segmentation_pixels=1,
        )
        oversized_score = compute_silhouette_score(
            oversized_points,
            [observation],
            point_radius_px=5,
            close_kernel_px=7,
            dilate_px=1,
            min_rendered_pixels=1,
            min_segmentation_pixels=1,
        )

        inside_loss = inside_score.loss_iou + inside_score.outside_loss
        oversized_loss = oversized_score.loss_iou + oversized_score.outside_loss
        self.assertLess(inside_loss, oversized_loss)

    def test_empty_or_missing_masks_are_ignored_without_crashing(self):
        observation = SilhouetteObservation(
            camera_id=0,
            mask=np.zeros((20, 20), dtype=bool),
            intrinsics={"fx": 10.0, "fy": 10.0, "cx": 10.0, "cy": 10.0},
            t_base_cam=np.eye(4, dtype=np.float32),
            image_shape=(20, 20),
        )
        score = compute_silhouette_score(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            [observation],
            min_segmentation_pixels=1,
        )

        self.assertFalse(score.valid)
        self.assertEqual(score.valid_camera_count, 0)


if __name__ == "__main__":
    unittest.main()
