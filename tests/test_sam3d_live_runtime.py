"""State-transition tests for standalone SAM3D dynamic bbox gating."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from perception.hands23_ipc import Hands23BBoxResult
from perception.sam3d_live_runtime import (
    READINESS_RAW_CLOUD,
    READINESS_SHAPE_FIT,
    Sam3DLiveRuntime,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Engine:
    def __init__(self, bbox):
        self.initial_bbox = tuple(bbox)
        self.bbox = tuple(bbox)

    def reset_bbox(self):
        self.bbox = self.initial_bbox

    def set_bbox(self, bbox):
        self.bbox = None if bbox is None else tuple(bbox)


class _Client:
    def __init__(self):
        self.healthy = True
        self.last_error = None
        self.results = []
        self.submissions = []
        self.reset_calls = []

    def reset(self, task_id):
        self.reset_calls.append(int(task_id))

    def drain_results(self):
        results, self.results = self.results, []
        return results

    def submit(self, camera_id, _image, **metadata):
        self.submissions.append((int(camera_id), metadata))
        return True

    def start(self):
        return None

    def close(self):
        return None


def _result(camera_id, bbox, *, frame_seq=4, task_id=3, capture_time_s=100.0):
    return Hands23BBoxResult(
        task_id=task_id,
        frame_seq=frame_seq,
        camera_id=camera_id,
        capture_time_s=capture_time_s,
        valid=True,
        bbox_xyxy=tuple(bbox),
        hand_side="left" if camera_id == 0 else "right",
        hand_score=0.9,
        object_score=0.8,
        contact_state="object_contact",
        reason="ok",
        inference_ms=12.0,
        roundtrip_ms=18.0,
    )


def _snapshot(pair_index=5):
    frame = SimpleNamespace(
        color_image=np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp_ms=0.0,
    )
    return SimpleNamespace(cam0=frame, cam1=frame, pair_index=pair_index)


class Sam3DLiveRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.client = _Client()
        self.engines = [_Engine((1, 1, 6, 6)), _Engine((2, 2, 7, 7))]
        pipeline = {
            "object_worker_cam0": SimpleNamespace(segmentation_engine=self.engines[0]),
            "object_worker_cam1": SimpleNamespace(segmentation_engine=self.engines[1]),
        }
        self.runtime = Sam3DLiveRuntime(
            pipeline,
            REPO_ROOT / "configs" / "handover.yaml",
            repo_root=REPO_ROOT,
            client=self.client,
        )
        self.runtime.task_id = 3
        self.runtime.state.activate(3)
        self.runtime._initialization_phase = "dynamic"
        self.runtime._gate_blocked = False

    @staticmethod
    def _shape_state(*, valid=True, silhouette_valid=True):
        return SimpleNamespace(
            valid=bool(valid),
            initialized=bool(valid),
            silhouette_enabled=True,
            silhouette_candidate_count=5 if silhouette_valid else 0,
            silhouette_valid_camera_count=1 if silhouette_valid else 0,
            silhouette_reason="kept" if silhouette_valid else "no_valid_silhouette",
        )

    def test_fixed_scaling_requires_four_actual_silhouette_frames_before_hands23(self):
        self.runtime.reset(3)

        self.assertEqual(self.runtime.initialization_phase, "fixed_scaling")
        self.assertFalse(self.runtime.silhouette_scale_frozen)
        blocked = self.runtime.before_frame(_snapshot(), task_id=3, now_ros_s=100.0)
        self.assertTrue(blocked)
        self.assertEqual(self.client.submissions, [])

        for frame_index in range(4):
            switched = self.runtime.after_shape_fit(self._shape_state())
            self.assertEqual(switched, frame_index == 3)
            if frame_index < 3:
                self.assertFalse(self.runtime.silhouette_scale_frozen)

        self.assertEqual(self.runtime.initial_scaling_progress, (4, 4))
        self.assertEqual(self.runtime.initialization_phase, "waiting_hands23")
        self.assertTrue(self.runtime.silhouette_scale_frozen)

        blocked = self.runtime.before_frame(_snapshot(), task_id=3, now_ros_s=100.0)
        self.assertTrue(blocked)
        self.assertEqual(len(self.client.submissions), 2)

        self.client.results = [_result(0, (1, 2, 5, 7))]
        with mock.patch("perception.sam3d_live_runtime.time.monotonic", return_value=10.0), mock.patch(
            "perception.dynamic_bbox.time.monotonic", return_value=10.0
        ):
            blocked = self.runtime.before_frame(_snapshot(pair_index=6), task_id=3, now_ros_s=100.1)
        self.assertFalse(blocked)
        self.assertEqual(self.runtime.initialization_phase, "dynamic")

    def test_invalid_fitting_or_silhouette_resets_initial_scaling_progress(self):
        self.runtime.reset(3)
        self.runtime.after_shape_fit(self._shape_state())
        self.runtime.after_shape_fit(self._shape_state())
        self.assertEqual(self.runtime.initial_scaling_progress, (2, 4))

        self.runtime.after_shape_fit(self._shape_state(silhouette_valid=False))
        self.assertEqual(self.runtime.initial_scaling_progress, (0, 4))

    def test_shape_fit_readiness_does_not_require_silhouette_evidence(self):
        runtime = Sam3DLiveRuntime(
            self.runtime.pipeline,
            REPO_ROOT / "configs" / "handover.yaml",
            repo_root=REPO_ROOT,
            client=self.client,
            readiness_strategy=READINESS_SHAPE_FIT,
            ready_valid_frames=2,
        )
        runtime.reset(3)
        state = self._shape_state(silhouette_valid=False)

        self.assertFalse(runtime.after_shape_fit(state))
        self.assertTrue(runtime.after_shape_fit(state))
        self.assertEqual(runtime.initialization_phase, "waiting_hands23")
        self.assertFalse(runtime.silhouette_scale_frozen)

    def test_raw_cloud_readiness_and_regeneration_disable(self):
        runtime = Sam3DLiveRuntime(
            self.runtime.pipeline,
            REPO_ROOT / "configs" / "handover.yaml",
            repo_root=REPO_ROOT,
            client=self.client,
            readiness_strategy=READINESS_RAW_CLOUD,
            ready_valid_frames=2,
            regeneration_enabled=False,
        )
        runtime.reset(3)
        valid = SimpleNamespace(valid=True, merged_point_count=20)
        invalid = SimpleNamespace(valid=False, merged_point_count=0)

        self.assertEqual(runtime.initialization_phase, "fixed_raw_cloud")
        self.assertFalse(runtime.after_raw_cloud(valid))
        self.assertFalse(runtime.after_raw_cloud(invalid))
        self.assertEqual(runtime.initial_scaling_progress, (0, 2))
        self.assertFalse(runtime.after_raw_cloud(valid))
        self.assertTrue(runtime.after_raw_cloud(valid))
        self.assertFalse(runtime.begin_regeneration(3))
        self.runtime.after_shape_fit(self._shape_state())
        self.runtime.after_shape_fit(self._shape_state(valid=False))
        self.assertEqual(self.runtime.initial_scaling_progress, (0, 4))

    def test_reset_reenters_fixed_scaling_and_closes_target_gate(self):
        self.runtime.reset(7)

        self.assertEqual(self.runtime.initialization_phase, "fixed_scaling")
        self.assertFalse(self.runtime.state.active)
        self.assertTrue(self.runtime.before_frame(_snapshot(), task_id=7, now_ros_s=100.0))
        self.assertEqual(self.client.submissions, [])

    def test_one_camera_bbox_keeps_fitting_and_suspends_other_camera(self):
        self.client.results = [_result(0, (1, 2, 5, 7))]
        with mock.patch("perception.sam3d_live_runtime.time.monotonic", return_value=10.0), mock.patch(
            "perception.dynamic_bbox.time.monotonic", return_value=10.0
        ):
            blocked = self.runtime.before_frame(
                _snapshot(), task_id=3, now_ros_s=100.1
            )

        self.assertFalse(blocked)
        self.assertEqual(self.engines[0].bbox, (1.0, 2.0, 5.0, 7.0))
        self.assertIsNone(self.engines[1].bbox)

    def test_both_camera_holds_expiring_suspends_both_and_closes_gate(self):
        self.client.results = [
            _result(0, (1, 2, 5, 7)),
            _result(1, (2, 1, 7, 6)),
        ]
        with mock.patch("perception.sam3d_live_runtime.time.monotonic", return_value=10.0), mock.patch(
            "perception.dynamic_bbox.time.monotonic", return_value=10.0
        ):
            self.assertFalse(
                self.runtime.before_frame(_snapshot(), task_id=3, now_ros_s=100.1)
            )

        with mock.patch("perception.sam3d_live_runtime.time.monotonic", return_value=10.6):
            blocked = self.runtime.before_frame(
                _snapshot(pair_index=6), task_id=3, now_ros_s=100.2
            )

        self.assertTrue(blocked)
        self.assertIsNone(self.engines[0].bbox)
        self.assertIsNone(self.engines[1].bbox)

    def test_sidecar_failure_blocks_even_while_bbox_hold_is_fresh(self):
        self.client.results = [_result(0, (1, 2, 5, 7))]
        self.client.healthy = False
        self.client.last_error = "socket disconnected"
        with mock.patch("perception.sam3d_live_runtime.time.monotonic", return_value=10.0), mock.patch(
            "perception.dynamic_bbox.time.monotonic", return_value=10.0
        ):
            blocked = self.runtime.before_frame(
                _snapshot(), task_id=3, now_ros_s=100.1
            )

        self.assertTrue(blocked)
        self.assertIsNone(self.engines[0].bbox)
        self.assertIsNone(self.engines[1].bbox)

    def test_both_fastsam_observations_invalid_close_dynamic_gate(self):
        self.runtime._gate_blocked = False
        invalid = SimpleNamespace(valid=False)
        self.assertTrue(self.runtime.after_object_perception(invalid, invalid))

    def test_one_valid_fastsam_observation_keeps_dynamic_gate_open(self):
        self.runtime._gate_blocked = False
        self.assertFalse(
            self.runtime.after_object_perception(
                SimpleNamespace(valid=True),
                SimpleNamespace(valid=False),
            )
        )


if __name__ == "__main__":
    unittest.main()
