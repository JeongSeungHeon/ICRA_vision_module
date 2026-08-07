"""Runtime SAM3D stable-capture and template validation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

from perception.sam3d_runtime import (
    StableCaptureAccumulator,
    generate_runtime_template,
    validate_template_points,
    write_capture_artifacts,
)


def selection(*, bbox_iou=0.8, confidence=0.9, pixels=16):
    return SimpleNamespace(
        bbox=(1, 1, 5, 5),
        bbox_iou=bbox_iou,
        model_confidence=confidence,
        mask_pixels=pixels,
    )


class Sam3DReinitializerTest(unittest.TestCase):
    def test_stable_capture_requires_consecutive_temporal_iou(self):
        accumulator = StableCaptureAccumulator(
            stable_frames=2,
            min_mask_pixels=4,
            min_temporal_iou=0.8,
        )
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        first = np.zeros((8, 8), dtype=bool)
        first[1:5, 1:5] = True
        shifted = np.zeros((8, 8), dtype=bool)
        shifted[3:7, 3:7] = True

        self.assertIsNone(accumulator.add(image, first, selection()))
        self.assertIsNone(accumulator.add(image, shifted, selection()))
        result = accumulator.add(
            image,
            shifted,
            selection(bbox_iou=0.9),
        )
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.mask, shifted)

    def test_capture_artifacts_are_standalone_compatible(self):
        accumulator = StableCaptureAccumulator(
            stable_frames=1,
            min_mask_pixels=4,
            min_temporal_iou=0.8,
        )
        image = np.full((8, 8, 3), 20, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:5, 1:5] = True
        capture = accumulator.add(image, mask, selection())
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "run"
            write_capture_artifacts(
                artifact_dir,
                capture,
                label="sam3d_object",
                configured_bboxes={"cam0": (1, 1, 5, 5)},
            )
            for name in ("image.png", "mask.png", "preview.png", "metadata.yaml"):
                self.assertTrue((artifact_dir / name).is_file(), name)
            metadata = yaml.safe_load(
                (artifact_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["capture"]["source"],
                "standalone_rtde_loop",
            )

    def test_template_validation_rejects_bad_shape_and_nonfinite_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.npy"
            np.save(path, np.zeros((100, 3), dtype=np.float32))
            points = validate_template_points(path, min_template_points=100)
            self.assertEqual(points.shape, (100, 3))

            np.save(path, np.zeros((100, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape"):
                validate_template_points(path, min_template_points=100)

            invalid = np.zeros((100, 3), dtype=np.float32)
            invalid[0, 0] = np.nan
            np.save(path, invalid)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_template_points(path, min_template_points=100)

    def test_server_generation_uses_existing_artifacts_and_ipc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            sam3d_repo = root / "sam3d"
            sam3d_repo.mkdir()
            config_path = root / "config.yaml"
            config_path.write_text("perception: {}\n", encoding="utf-8")

            def generate_side_effect(*_args, **kwargs):
                np.save(
                    Path(kwargs["artifact_dir"]) / "template.npy",
                    np.zeros((100, 3), dtype=np.float32),
                )
                return {"ok": True}

            with (
                mock.patch(
                    "tools.sam3d_ipc.build_model_spec",
                    return_value={"signature": "sig"},
                ) as build_spec,
                mock.patch(
                    "tools.sam3d_ipc.wait_until_ready",
                    return_value={"state": "ready"},
                ) as wait_ready,
                mock.patch(
                    "tools.sam3d_ipc.generate",
                    side_effect=generate_side_effect,
                ) as generate,
            ):
                template = generate_runtime_template(
                    config_path=config_path,
                    artifact_dir=artifact_dir,
                    repo_root=root,
                    execution_mode="server",
                    sam3d_python=root / "unused-python",
                    sam3d_repo=sam3d_repo,
                    server_socket=str(root / "sam3d.sock"),
                    server_ready_timeout_s=5.0,
                    request_timeout_s=7.0,
                )
            self.assertEqual(template, artifact_dir / "template.npy")
            build_spec.assert_called_once()
            wait_ready.assert_called_once()
            generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
