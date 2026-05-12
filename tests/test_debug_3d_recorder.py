import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.debug_3d_recorder import Debug3DRecorder, load_debug_3d_npz


def _shape_state(points=None, valid=True):
    return SimpleNamespace(
        label="cup",
        template_id="cup_template",
        fitted_points_base=np.empty((0, 3), dtype=np.float32) if points is None else points,
        centroid_base=(0.2, 0.3, 0.4),
        scale=1.25,
        bowl_height_fraction=None,
        initialized=True,
        reason="ok",
        valid=bool(valid),
    )


def _merged_object(points=None, valid=True):
    points = np.empty((0, 3), dtype=np.float32) if points is None else points
    return SimpleNamespace(
        label="cup",
        merged_points_base=points,
        merged_point_count=len(points),
        centroid_base=(0.1, 0.2, 0.3),
        valid=bool(valid),
    )


def _hand_debug(offset=0.0):
    points = np.full((21, 3), np.nan, dtype=np.float32)
    points[:5] = np.arange(15, dtype=np.float32).reshape(5, 3) * 0.01 + float(offset)
    mask = np.zeros((21,), dtype=bool)
    mask[:5] = True
    return SimpleNamespace(
        points_3d_base=points,
        valid_mask=mask,
        palm_pose_base={"rotation_matrix": np.eye(3, dtype=np.float32), "valid": True, "quality": 0.8},
    )


def _invalid_hand_debug():
    return SimpleNamespace(
        points_3d_base=np.full((21, 3), np.nan, dtype=np.float32),
        valid_mask=np.zeros((21,), dtype=bool),
        palm_pose_base={"valid": False, "quality": 0.0},
    )


def _make_snapshot():
    cam0 = SimpleNamespace(
        color_image=np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        depth_image_m=np.arange(2 * 3, dtype=np.float32).reshape(2, 3) * 0.01,
        intrinsics={"fx": 100.0, "fy": 101.0, "cx": 50.0, "cy": 51.0},
        timestamp_ms=1234.5,
        serial="cam0_serial",
    )
    cam1 = SimpleNamespace(
        color_image=(np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) + 20),
        depth_image_m=(np.arange(2 * 3, dtype=np.float32).reshape(2, 3) + 10.0) * 0.01,
        intrinsics={"fx": 200.0, "fy": 201.0, "cx": 60.0, "cy": 61.0},
        timestamp_ms=1235.0,
        serial="cam1_serial",
    )
    return SimpleNamespace(
        pair_index=7,
        cam0=cam0,
        cam1=cam1,
        timestamp_delta_ms=0.5,
        within_sync_tolerance=True,
    )


def _append(
    recorder,
    frame_index,
    object_points,
    template_points,
    *,
    missing=False,
    snapshot=None,
    hand_debug_cam0=None,
    hand_debug_cam1=None,
):
    selected_hand = SimpleNamespace(
        valid=not missing,
        selected_camera=None if missing else 0,
        handedness=None if missing else "right",
        palm_center_base=None if missing else (0.4, 0.5, 0.6),
        palm_normal_base=None if missing else (0.0, 0.0, 1.0),
        wrist_base=None if missing else (0.3, 0.4, 0.5),
        confidence=0.9,
    )
    recorder.append_frame(
        frame_index=frame_index,
        timestamp_unix_s=100.0 + frame_index,
        timestamp_perf_s=10.0 + frame_index,
        record_elapsed_s=1.2 + frame_index,
        task_epoch=2,
        selected_hand=selected_hand,
        hand_debug_cam0=None if missing else (hand_debug_cam0 if hand_debug_cam0 is not None else _hand_debug(0.0)),
        hand_debug_cam1=None if missing else (hand_debug_cam1 if hand_debug_cam1 is not None else _hand_debug(0.1)),
        raw_merged_object=_merged_object(object_points, valid=not missing),
        shape_fitting_state=_shape_state(template_points, valid=not missing),
        object_point_base=None if missing else (0.1, 0.2, 0.3),
        grasp_point_base=None if missing else (0.2, 0.3, 0.4),
        eef_pose_base=None if missing else (0.5, 0.6, 0.7, 0.0, 0.1, 0.2),
        measurement_source="measured",
        hand_selector_debug=SimpleNamespace(
            selection_reason="no_valid_candidate" if missing else "select_best_available",
            cam0_reject_reason="hand_not_detected" if missing else "",
            cam1_reject_reason="hand_not_detected" if missing else "",
            chosen_camera=None if missing else 0,
        ),
        snapshot=snapshot,
    )


class Debug3DRecorderTests(unittest.TestCase):
    def test_round_trip_variable_clouds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), max_object_points=10, max_template_points=8)
            object_points_0 = np.arange(45, dtype=np.float32).reshape(15, 3)
            template_points_0 = np.arange(36, dtype=np.float32).reshape(12, 3) * 0.1
            object_points_1 = np.arange(12, dtype=np.float32).reshape(4, 3)
            template_points_1 = np.arange(18, dtype=np.float32).reshape(6, 3) * 0.1

            _append(recorder, 0, object_points_0, template_points_0)
            _append(recorder, 1, object_points_1, template_points_1)

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(int(data["frame_count"]), 2)
            self.assertEqual(data["object_points_base"][0].shape, (10, 3))
            self.assertEqual(data["template_points_base"][0].shape, (8, 3))
            self.assertEqual(data["object_points_base"][1].shape, (4, 3))
            self.assertEqual(data["template_points_base"][1].shape, (6, 3))
            np.testing.assert_allclose(data["grasp_point_base"][0], np.asarray([0.2, 0.3, 0.4], dtype=np.float32))
            np.testing.assert_allclose(
                data["eef_pose_base"][0],
                np.asarray([0.5, 0.6, 0.7, 0.0, 0.1, 0.2], dtype=np.float32),
            )
            self.assertEqual(str(data["record_clock_text"][0]), "REC 00:01.1")
            self.assertAlmostEqual(float(data["record_elapsed_s"][0]), 1.2, places=6)
            self.assertNotIn("cam0_color_image", data)
            self.assertNotIn("cam0_depth_image_m", data)

    def test_missing_values_are_nan_or_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                missing=True,
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(data["object_points_base"][0].shape, (0, 3))
            self.assertEqual(data["template_points_base"][0].shape, (0, 3))
            self.assertTrue(np.isnan(data["object_point_base"][0]).all())
            self.assertTrue(np.isnan(data["grasp_point_base"][0]).all())
            self.assertTrue(np.isnan(data["eef_pose_base"][0]).all())
            self.assertEqual(data["selected_hand_camera"][0], -1)
            self.assertFalse(bool(data["selected_hand_valid"][0]))

    def test_raw_images_round_trip_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = _make_snapshot()
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), save_images=True)
            _append(
                recorder,
                0,
                np.arange(12, dtype=np.float32).reshape(4, 3),
                np.arange(9, dtype=np.float32).reshape(3, 3),
                snapshot=snapshot,
            )

            data = load_debug_3d_npz(recorder.save())

            np.testing.assert_array_equal(data["cam0_color_image"][0], snapshot.cam0.color_image)
            np.testing.assert_allclose(data["cam0_depth_image_m"][0], snapshot.cam0.depth_image_m)
            np.testing.assert_allclose(data["cam0_intrinsics"][0], np.asarray([100.0, 101.0, 50.0, 51.0]))
            self.assertAlmostEqual(float(data["cam0_timestamp_ms"][0]), 1234.5)
            self.assertEqual(str(data["cam0_serial"][0]), "cam0_serial")
            self.assertEqual(int(data["cam0_hand_valid_count"][0]), 5)
            self.assertTrue(bool(data["cam0_hand_detected"][0]))
            self.assertEqual(str(data["hand_selection_reason"][0]), "select_best_available")

            np.testing.assert_array_equal(data["cam1_color_image"][0], snapshot.cam1.color_image)
            np.testing.assert_allclose(data["cam1_depth_image_m"][0], snapshot.cam1.depth_image_m)
            np.testing.assert_allclose(data["cam1_intrinsics"][0], np.asarray([200.0, 201.0, 60.0, 61.0]))
            self.assertAlmostEqual(float(data["cam1_timestamp_ms"][0]), 1235.0)
            self.assertEqual(str(data["cam1_serial"][0]), "cam1_serial")
            self.assertAlmostEqual(float(data["timestamp_delta_ms"][0]), 0.5)
            self.assertTrue(bool(data["within_sync_tolerance"][0]))

    def test_saved_arrays_are_not_aliased_to_mutated_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = _make_snapshot()
            object_points = np.arange(12, dtype=np.float32).reshape(4, 3)
            template_points = np.arange(9, dtype=np.float32).reshape(3, 3)
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), save_images=True)

            original_color = snapshot.cam0.color_image.copy()
            original_depth = snapshot.cam0.depth_image_m.copy()
            original_object = object_points.copy()
            original_template = template_points.copy()
            _append(recorder, 0, object_points, template_points, snapshot=snapshot)

            snapshot.cam0.color_image[...] = 255
            snapshot.cam0.depth_image_m[...] = 9.0
            object_points[...] = -1.0
            template_points[...] = -2.0

            data = load_debug_3d_npz(recorder.save())

            np.testing.assert_array_equal(data["cam0_color_image"][0], original_color)
            np.testing.assert_allclose(data["cam0_depth_image_m"][0], original_depth)
            np.testing.assert_allclose(data["object_points_base"][0], original_object)
            np.testing.assert_allclose(data["template_points_base"][0], original_template)

    def test_multiple_image_frames_round_trip_independently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), save_images=True)
            snapshot0 = _make_snapshot()
            snapshot1 = _make_snapshot()
            snapshot1.cam0.color_image = snapshot1.cam0.color_image + 7
            snapshot1.cam1.color_image = snapshot1.cam1.color_image + 13
            snapshot1.cam0.depth_image_m = snapshot1.cam0.depth_image_m + 0.5
            snapshot1.cam1.depth_image_m = snapshot1.cam1.depth_image_m + 0.7

            _append(recorder, 0, np.arange(6, dtype=np.float32).reshape(2, 3), np.empty((0, 3)), snapshot=snapshot0)
            _append(recorder, 1, np.arange(6, dtype=np.float32).reshape(2, 3), np.empty((0, 3)), snapshot=snapshot1)

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(data["cam0_color_image"].shape[0], 2)
            self.assertNotEqual(int(data["cam0_color_image"][0].sum()), int(data["cam0_color_image"][1].sum()))
            self.assertNotEqual(float(data["cam1_depth_image_m"][0].mean()), float(data["cam1_depth_image_m"][1].mean()))

    def test_invalid_hand_debug_diagnostics_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                hand_debug_cam0=_invalid_hand_debug(),
                hand_debug_cam1=_invalid_hand_debug(),
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertFalse(bool(data["cam0_hand_detected"][0]))
            self.assertEqual(int(data["cam0_hand_valid_count"][0]), 0)
            self.assertEqual(str(data["cam0_hand_reason"][0]), "no_valid_depth_landmarks")


if __name__ == "__main__":
    unittest.main()
