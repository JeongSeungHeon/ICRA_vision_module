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
        palm_pose_base={"rotation_matrix": np.eye(3, dtype=np.float32)},
    )


def _append(recorder, frame_index, object_points, template_points, *, missing=False):
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
        hand_debug_cam0=None if missing else _hand_debug(0.0),
        hand_debug_cam1=None if missing else _hand_debug(0.1),
        raw_merged_object=_merged_object(object_points, valid=not missing),
        shape_fitting_state=_shape_state(template_points, valid=not missing),
        object_point_base=None if missing else (0.1, 0.2, 0.3),
        grasp_point_base=None if missing else (0.2, 0.3, 0.4),
        eef_pose_base=None if missing else (0.5, 0.6, 0.7, 0.0, 0.1, 0.2),
        measurement_source="measured",
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


if __name__ == "__main__":
    unittest.main()
