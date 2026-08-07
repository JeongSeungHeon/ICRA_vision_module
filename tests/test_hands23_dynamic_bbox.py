"""Unit tests for Hands23 output parsing, selection, and bbox lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from perception.dynamic_bbox import (
    BOOTSTRAP_FIXED,
    DYNAMIC_TRACKING,
    WAITING_DYNAMIC,
    DynamicFastSAMBBoxState,
)
from perception.hands23_runtime import (
    Hands23BBoxSelector,
    Hands23Candidate,
    Hands23PredictorAdapter,
    expected_hand_sides_from_config,
)


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeBoxes:
    def __init__(self, values):
        self.tensor = _FakeTensor(values)


class _FakeInstances:
    def __init__(self, fields):
        self.fields = fields

    def get(self, name):
        return self.fields[name]


def candidate(
    bbox,
    *,
    side="left",
    object_score=0.8,
    hand_score=0.9,
):
    return Hands23Candidate(
        hand_bbox=(0.0, 0.0, 5.0, 5.0),
        object_bbox=tuple(float(value) for value in bbox),
        hand_side=side,
        hand_score=hand_score,
        object_score=object_score,
        contact_state="object_contact",
    )


class Hands23RuntimeTest(unittest.TestCase):
    def test_parser_returns_only_associated_first_objects(self):
        boxes = [
            [0, 0, 10, 10],    # left hand
            [20, 20, 50, 50],  # first object
            [60, 0, 75, 15],   # unassociated right hand
        ]
        classes = [0, 1, 0]
        scores = [0.9, 0.8, 0.95]
        pred_dz = np.zeros((3, 9), dtype=np.float32)
        pred_dz[:, 4] = -1
        pred_dz[0, 4] = 1
        pred_dz[0, 5] = 0
        pred_dz[0, 8] = 3
        outputs = {
            "instances": _FakeInstances(
                {
                    "pred_boxes": _FakeBoxes(boxes),
                    "pred_classes": _FakeTensor(classes),
                    "scores": _FakeTensor(scores),
                    "pred_dz": _FakeTensor(pred_dz),
                }
            )
        }
        adapter = Hands23PredictorAdapter(
            repo_path="unused",
            config_path="unused",
            weights_path="unused",
            min_size_test=640,
            predictor=lambda _image: outputs,
        )
        self.assertEqual(adapter.min_size_test, 640)
        parsed = adapter.candidates_from_outputs(outputs)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].object_bbox, (20.0, 20.0, 50.0, 50.0))
        self.assertEqual(parsed[0].hand_side, "left")
        self.assertEqual(parsed[0].contact_state, "object_contact")

    def test_expected_side_then_continuity_then_confidence(self):
        config = {
            "perception": {
                "hand_selection": {
                    "allowed_active_candidate_ids": ["cam0:left", "cam1:right"]
                }
            }
        }
        self.assertEqual(expected_hand_sides_from_config(config), {0: "left", 1: "right"})
        selector = Hands23BBoxSelector(
            initial_bboxes={0: (10, 10, 30, 30), 1: (10, 10, 30, 30)},
            expected_hand_sides={0: "left", 1: "right"},
            padding_ratio=0.0,
            ema_alpha=0.6,
            min_bbox_size_px=2,
        )
        selected = selector.select(
            0,
            [
                candidate((11, 11, 31, 31), side="right", object_score=0.99),
                candidate((12, 12, 32, 32), side="left", object_score=0.7),
            ],
            width=100,
            height=100,
        )
        self.assertEqual(selected.hand_side, "left")
        self.assertEqual(selected.bbox_xyxy, (12.0, 12.0, 32.0, 32.0))

        # Both candidates now have the expected side; continuity wins before score.
        selected = selector.select(
            0,
            [
                candidate((13, 13, 33, 33), side="left", object_score=0.6),
                candidate((60, 60, 90, 90), side="left", object_score=0.99),
            ],
            width=100,
            height=100,
        )
        np.testing.assert_allclose(selected.bbox_xyxy, (12.6, 12.6, 32.6, 32.6))

    def test_padding_and_clamping(self):
        selector = Hands23BBoxSelector(
            initial_bboxes={0: (0, 0, 20, 20)},
            padding_ratio=0.10,
            ema_alpha=1.0,
            min_bbox_size_px=4,
        )
        selected = selector.select(
            0,
            [candidate((1, 2, 21, 22))],
            width=20,
            height=20,
        )
        self.assertEqual(selected.bbox_xyxy, (0.0, 0.0, 20.0, 20.0))


class DynamicBBoxStateTest(unittest.TestCase):
    def test_bootstrap_dynamic_dropout_recovery_and_reset(self):
        state = DynamicFastSAMBBoxState(
            hold_timeout_s=0.5,
            result_age_timeout_s=1.0,
        )
        self.assertEqual(state.mode(now_monotonic_s=10.0), BOOTSTRAP_FIXED)
        self.assertTrue(state.activate(7))
        self.assertEqual(state.mode(now_monotonic_s=10.0), WAITING_DYNAMIC)
        self.assertIsNone(state.bbox_for_camera(0, now_monotonic_s=10.0))

        self.assertTrue(
            state.accept(
                camera_id=0,
                task_id=7,
                frame_seq=10,
                valid=True,
                bbox_xyxy=(10, 20, 30, 40),
                capture_time_s=100.0,
                now_ros_s=100.2,
                received_monotonic_s=10.0,
                hand_side="left",
            )
        )
        self.assertEqual(state.mode(now_monotonic_s=10.4), DYNAMIC_TRACKING)
        self.assertEqual(
            state.bbox_for_camera(0, now_monotonic_s=10.4),
            (10.0, 20.0, 30.0, 40.0),
        )
        self.assertIsNone(state.bbox_for_camera(0, now_monotonic_s=10.6))
        self.assertEqual(state.mode(now_monotonic_s=10.6), WAITING_DYNAMIC)

        self.assertTrue(
            state.accept(
                camera_id=1,
                task_id=7,
                frame_seq=11,
                valid=True,
                bbox_xyxy=(50, 60, 80, 90),
                capture_time_s=101.0,
                now_ros_s=101.1,
                received_monotonic_s=11.0,
            )
        )
        self.assertEqual(state.mode(now_monotonic_s=11.1), DYNAMIC_TRACKING)
        state.reset(8)
        self.assertEqual(state.mode(now_monotonic_s=11.1), BOOTSTRAP_FIXED)

    def test_rejects_old_task_stale_and_out_of_order_results(self):
        state = DynamicFastSAMBBoxState(
            hold_timeout_s=0.5,
            result_age_timeout_s=1.0,
        )
        state.activate(2)
        common = dict(
            camera_id=0,
            valid=True,
            bbox_xyxy=(1, 1, 20, 20),
            received_monotonic_s=5.0,
        )
        self.assertFalse(
            state.accept(
                task_id=1,
                frame_seq=1,
                capture_time_s=10.0,
                now_ros_s=10.0,
                **common,
            )
        )
        self.assertFalse(
            state.accept(
                task_id=2,
                frame_seq=2,
                capture_time_s=10.0,
                now_ros_s=11.1,
                **common,
            )
        )
        self.assertFalse(
            state.accept(
                task_id=2,
                frame_seq=1,
                capture_time_s=11.0,
                now_ros_s=11.0,
                **common,
            ),
            "frame 1 is older than the already observed frame 2",
        )


if __name__ == "__main__":
    unittest.main()
