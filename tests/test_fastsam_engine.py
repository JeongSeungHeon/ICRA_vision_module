"""Unit tests for the classless FastSAM object backend."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from calibration.extrinsics import TransformChain
from object_pt_extraction.fastsam_engine import (
    FastSAMSegmentationEngine,
    SharedFastSAMModel,
    binary_mask_iou,
    normalize_bbox,
    resolve_fastsam_device,
    select_mask_by_bbox_iou,
)
from perception.object_worker import ObjectWorker
from utils.realsense_stream import FrameBundle


class _FakeFastSAM:
    def __init__(self, masks, confidences=None):
        self.masks = np.asarray(masks, dtype=np.float32)
        self.confidences = np.asarray(
            confidences if confidences is not None else np.ones((len(self.masks),)),
            dtype=np.float32,
        )
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes = np.tile(np.array([[1.0, 1.0, 7.0, 7.0]], dtype=np.float32), (len(self.masks), 1))
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=self.masks),
                boxes=SimpleNamespace(conf=self.confidences, xyxy=boxes),
            )
        ]


class _FailingFastSAM:
    def predict(self, **_kwargs):
        raise RuntimeError("CUDA inference failed")


class FastSAMHelpersTest(unittest.TestCase):
    def test_require_cuda_rejects_cpu_build_or_unavailable_driver(self):
        with (
            mock.patch("torch.version.cuda", None),
            self.assertRaisesRegex(RuntimeError, "CPU-only build"),
        ):
            resolve_fastsam_device(None, require_cuda=True)

        with (
            mock.patch("torch.version.cuda", "12.1"),
            mock.patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "is_available"),
        ):
            resolve_fastsam_device("cuda:0", require_cuda=True)

    def test_require_cuda_defaults_to_cuda_zero_and_rejects_cpu_device(self):
        with (
            mock.patch("torch.version.cuda", "12.1"),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            self.assertEqual(
                resolve_fastsam_device(None, require_cuda=True),
                "cuda:0",
            )
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            resolve_fastsam_device("cpu", require_cuda=True)

    def test_bbox_is_clamped_and_small_bbox_rejected(self):
        self.assertEqual(normalize_bbox((-2, -3, 20, 15), 10, 10, min_size_px=8), (0, 0, 10, 10))
        with self.assertRaisesRegex(ValueError, "too small"):
            normalize_bbox((9, 9, 12, 12), 10, 10, min_size_px=8)

    def test_selects_mask_with_best_bbox_iou(self):
        first = np.zeros((10, 10), dtype=bool)
        first[0:2, 0:2] = True
        second = np.zeros((10, 10), dtype=bool)
        second[2:8, 2:8] = True
        selected, index, score = select_mask_by_bbox_iou(
            np.stack([first, second]),
            (2, 2, 8, 8),
            width=10,
            height=10,
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(score, 1.0)
        self.assertTrue(np.array_equal(selected, second))
        self.assertAlmostEqual(binary_mask_iou(first, first), 1.0)

    def test_engine_returns_fixed_label_and_prompt_bbox(self):
        wrong = np.zeros((10, 10), dtype=np.float32)
        wrong[0:2, 0:2] = 1.0
        right = np.zeros((10, 10), dtype=np.float32)
        right[2:8, 2:8] = 1.0
        fake = _FakeFastSAM([wrong, right], confidences=[0.9, 0.7])
        shared = SharedFastSAMModel("unused.pt", model=fake)
        engine = FastSAMSegmentationEngine(
            shared,
            bbox=(2, 2, 8, 8),
            label="sam3d_object",
            min_bbox_size_px=2,
        )
        result = engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(len(result.instances), 1)
        self.assertEqual(result.instances[0].class_name, "sam3d_object")
        self.assertAlmostEqual(result.instances[0].score, 0.7)
        self.assertEqual(engine.last_selection.selected_index, 1)
        self.assertEqual(fake.calls[0]["bboxes"], [[2, 2, 8, 8]])

    def test_engine_can_suspend_and_restore_runtime_bbox(self):
        mask = np.ones((10, 10), dtype=np.float32)
        fake = _FakeFastSAM([mask], confidences=[0.8])
        shared = SharedFastSAMModel("unused.pt", model=fake)
        engine = FastSAMSegmentationEngine(
            shared,
            bbox=(1, 1, 9, 9),
            label="sam3d_object",
            min_bbox_size_px=2,
        )

        engine.set_bbox(None)
        suspended = engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(suspended.instances, [])
        self.assertEqual(suspended.infer_ms, 0.0)
        self.assertEqual(fake.calls, [], "suspended camera must not invoke FastSAM")

        engine.set_bbox((2, 2, 8, 8))
        engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(fake.calls[-1]["bboxes"], [[2, 2, 8, 8]])

        engine.reset_bbox()
        engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(fake.calls[-1]["bboxes"], [[1, 1, 9, 9]])

    def test_engine_converts_inference_exception_to_invalid_observation(self):
        shared = SharedFastSAMModel("unused.pt", model=_FailingFastSAM())
        engine = FastSAMSegmentationEngine(
            shared,
            bbox=(1, 1, 9, 9),
            label="sam3d_object",
            min_bbox_size_px=2,
        )
        result = engine.predict(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(result.instances, [])
        self.assertIn("CUDA inference failed", engine.last_error)


class FastSAMObjectWorkerTest(unittest.TestCase):
    def _worker(self, depth):
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[1:9, 1:9] = 1.0
        shared = SharedFastSAMModel("unused.pt", model=_FakeFastSAM([mask], [0.9]))
        engine = FastSAMSegmentationEngine(
            shared,
            bbox=(1, 1, 9, 9),
            label="sam3d_object",
            min_bbox_size_px=2,
        )
        identity = np.eye(4, dtype=np.float32)
        chain = TransformChain(
            t_base_cam0=identity,
            t_cam0_cam1=identity,
            t_base_cam1=identity,
            source_config="test",
        )
        config = {
            "perception": {
                "object": {
                    "confidence_threshold": 0.15,
                    "segmentation": {
                        "selection_mode": "highest_score",
                        "selection_class_names": ["sam3d_object"],
                    },
                    "point_cloud": {
                        "stride": 1,
                        "max_points": 1000,
                        "min_depth_m": 0.1,
                        "max_depth_m": 2.0,
                        "per_camera_voxel_size_m": 0.0,
                        "outlier_method": "none",
                        "min_points_per_camera": 10,
                    },
                }
            }
        }
        worker = ObjectWorker(0, engine, chain, config)
        frame = FrameBundle(
            color_image=np.zeros((10, 10, 3), dtype=np.uint8),
            depth_image_m=np.asarray(depth, dtype=np.float32),
            intrinsics={"fx": 10.0, "fy": 10.0, "cx": 5.0, "cy": 5.0},
            timestamp_ms=1000.0,
            serial="fake",
        )
        return worker, frame

    def test_mask_and_depth_produce_sam3d_object_state(self):
        worker, frame = self._worker(np.ones((10, 10), dtype=np.float32))
        state = worker.process_frame(frame, frame_id=5)
        self.assertTrue(state.valid)
        self.assertEqual(state.label, "sam3d_object")
        self.assertGreaterEqual(state.point_count, 10)
        self.assertTrue(np.any(worker.last_debug.combined_mask))

    def test_missing_depth_produces_invalid_state(self):
        worker, frame = self._worker(np.zeros((10, 10), dtype=np.float32))
        state = worker.process_frame(frame, frame_id=5)
        self.assertFalse(state.valid)
        self.assertEqual(state.point_count, 0)


if __name__ == "__main__":
    unittest.main()
