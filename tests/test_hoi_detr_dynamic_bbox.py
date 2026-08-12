"""Unit tests for HOI-DETR output parsing, selection, and bbox lifecycle."""

from __future__ import annotations

import unittest

import numpy as np

from perception.dynamic_bbox import (
    BOOTSTRAP_FIXED,
    DYNAMIC_TRACKING,
    WAITING_DYNAMIC,
    DynamicFastSAMBBoxState,
)
from perception.hoi_detr_runtime import (
    HOIDETRBBoxSelector,
    HOIDETRCandidate,
    HOIDETRPredictorAdapter,
)

def candidate(
    bbox,
    *,
    relation_score=0.7,
    object_score=0.8,
    hand_score=0.9,
):
    return HOIDETRCandidate(
        hand_bbox=(0.0, 0.0, 5.0, 5.0),
        object_bbox=tuple(float(value) for value in bbox),
        hand_score=hand_score,
        object_score=object_score,
        relation_score=relation_score,
    )


class HOIDETRRuntimeTest(unittest.TestCase):
    def test_parser_returns_only_thresholded_hand_first_relations(self):
        detections = [
            {"box": [0, 0, 10, 10], "score": 0.9, "class_id": 0},
            {"box": [60, 0, 75, 15], "score": 0.95, "class_id": 0},
            {"box": [20, 20, 50, 50], "score": 0.8, "class_id": 1},
            {"box": [70, 20, 95, 50], "score": 0.2, "class_id": 1},
        ]
        adapter = HOIDETRPredictorAdapter(
            repo_path="unused",
            config_path="unused",
            weights_path="unused",
            hand_score_threshold=0.3,
            first_object_score_threshold=0.3,
            hand_first_relation_threshold=0.6,
            inference_backend=lambda _image: (detections, [[0.9, 0.95], [0.4, 0.99]]),
        )
        parsed = adapter.predict(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].object_bbox, (20.0, 20.0, 50.0, 50.0))
        self.assertEqual(parsed[0].hand_side, "unknown")
        self.assertEqual(parsed[0].contact_state, "unknown")
        self.assertAlmostEqual(parsed[0].relation_score, 0.9)

    def test_continuity_then_relation_and_detection_confidence(self):
        selector = HOIDETRBBoxSelector(
            initial_bboxes={0: (10, 10, 30, 30), 1: (10, 10, 30, 30)},
            padding_ratio=0.0,
            ema_alpha=0.6,
            min_bbox_size_px=2,
        )
        selected = selector.select(
            0,
            [
                candidate((11, 11, 31, 31), relation_score=0.6),
                candidate((12, 12, 32, 32), relation_score=0.99),
            ],
            width=100,
            height=100,
        )
        self.assertEqual(selected.hand_side, "unknown")
        self.assertEqual(selected.bbox_xyxy, (11.0, 11.0, 31.0, 31.0))

        # Continuity wins before relation and detection confidence.
        selected = selector.select(
            0,
            [
                candidate((13, 13, 33, 33), relation_score=0.6, object_score=0.6),
                candidate((60, 60, 90, 90), relation_score=0.99, object_score=0.99),
            ],
            width=100,
            height=100,
        )
        np.testing.assert_allclose(selected.bbox_xyxy, (12.2, 12.2, 32.2, 32.2))

    def test_padding_and_clamping(self):
        selector = HOIDETRBBoxSelector(
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
