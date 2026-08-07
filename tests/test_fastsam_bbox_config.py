"""Tests for interactive FastSAM bbox persistence and config application."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml


from perception.fastsam_bbox_config import (
    FastSAMBBoxSelectionError,
    apply_selection_to_config,
    atomic_write_selection,
    build_selection_payload,
    load_selection,
    resolve_effective_config,
    validate_selection,
    xywh_to_xyxy,
)
from tools.select_fastsam_bbox import mask_rejection_reason
from tools import select_fastsam_bbox as selector_module
from object_pt_extraction import fastsam_engine


def _config() -> dict:
    return {
        "system": {"sensor_fps": 30},
        "cameras": {
            "cam0": {"serial": "cam0-serial", "width": 640, "height": 480},
            "cam1": {"serial": "cam1-serial", "width": 640, "height": 480},
        },
        "perception": {
            "object": {
                "fastsam": {
                    "bboxes": {
                        "cam0": [10, 20, 110, 220],
                        "cam1": [30, 40, 130, 240],
                    },
                    "min_bbox_size_px": 8,
                    "min_mask_pixels": 300,
                }
            }
        },
    }


def _camera_results() -> dict:
    return {
        "cam0": {
            "serial": "cam0-serial",
            "width": 640,
            "height": 480,
            "bbox_xyxy": [100, 110, 300, 330],
            "mask_pixels": 1200,
            "bbox_iou": 0.72,
            "model_confidence": 0.91,
        },
        "cam1": {
            "serial": "cam1-serial",
            "width": 640,
            "height": 480,
            "bbox_xyxy": [55, 65, 205, 285],
            "mask_pixels": 980,
            "bbox_iou": 0.68,
            "model_confidence": 0.87,
        },
    }


def _selection(config: dict | None = None) -> dict:
    return build_selection_payload(
        config or _config(),
        fastsam_model="/models/FastSAM-s.pt",
        camera_results=_camera_results(),
        selected_at_utc="2026-07-28T10:00:00+00:00",
    )


class FastSAMBBoxOverrideTest(unittest.TestCase):
    def test_xywh_drag_converts_to_clamped_xyxy(self):
        self.assertEqual(
            xywh_to_xyxy((-3, 20, 103, 80), 640, 480),
            (0, 20, 100, 100),
        )
        with self.assertRaises(FastSAMBBoxSelectionError):
            xywh_to_xyxy((10, 10, 4, 20), 640, 480)
        with self.assertRaises(FastSAMBBoxSelectionError):
            xywh_to_xyxy((10, 10, float("nan"), 20), 640, 480)

    def test_two_camera_selection_is_validated_and_applied_independently(self):
        config = _config()
        original = deepcopy(config)
        selection = _selection(config)

        bboxes = validate_selection(selection, config)
        self.assertEqual(bboxes["cam0"], (100, 110, 300, 330))
        self.assertEqual(bboxes["cam1"], (55, 65, 205, 285))

        effective, applied = apply_selection_to_config(config, selection)
        self.assertEqual(
            effective["perception"]["object"]["fastsam"]["bboxes"],
            {
                "cam0": [100, 110, 300, 330],
                "cam1": [55, 65, 205, 285],
            },
        )
        self.assertEqual(applied, bboxes)
        self.assertEqual(config, original, "base config must never be mutated")

    def test_empty_and_too_small_masks_cannot_be_accepted(self):
        empty_result = SimpleNamespace(instances=[])
        self.assertIn(
            "no non-empty mask",
            mask_rejection_reason(
                empty_result,
                None,
                min_mask_pixels=300,
            ),
        )
        result = SimpleNamespace(instances=[object()])
        selection = SimpleNamespace(mask_pixels=299)
        self.assertIn(
            "299 < required 300",
            mask_rejection_reason(
                result,
                selection,
                min_mask_pixels=300,
            ),
        )
        selection.mask_pixels = 300
        self.assertIsNone(
            mask_rejection_reason(
                result,
                selection,
                min_mask_pixels=300,
            )
        )

    def test_invalid_mask_serial_resolution_and_schema_are_rejected(self):
        config = _config()
        for mutate in (
            lambda payload: payload.update(schema_version="bad"),
            lambda payload: payload["cameras"]["cam0"].update(mask_pixels=20),
            lambda payload: payload["cameras"]["cam0"].update(serial="other"),
            lambda payload: payload["cameras"]["cam1"].update(width=1280),
            lambda payload: payload["cameras"]["cam1"].update(
                bbox_xyxy=[10, 10, 12, 100]
            ),
        ):
            payload = _selection(config)
            mutate(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(FastSAMBBoxSelectionError):
                    validate_selection(payload, config)

    def test_atomic_write_preserves_previous_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "fastsam_bbox.yaml"
            target.write_text("previous: true\n", encoding="utf-8")
            with patch(
                "perception.fastsam_bbox_config.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaises(OSError):
                    atomic_write_selection(target, _selection())
            self.assertEqual(target.read_text(encoding="utf-8"), "previous: true\n")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_selector_writes_only_after_both_cameras_are_accepted(self):
        fake_model = SimpleNamespace(model_name="/models/FastSAM-s.pt")
        camera_results = _camera_results()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fastsam_bbox.yaml"
            with (
                patch.dict(os.environ, {"DISPLAY": ":test"}, clear=False),
                patch.object(
                    fastsam_engine,
                    "SharedFastSAMModel",
                    return_value=fake_model,
                ),
                patch.object(
                    selector_module,
                    "_select_camera",
                    side_effect=[
                        camera_results["cam0"],
                        selector_module.BBoxSelectionCancelled("cam1 cancelled"),
                    ],
                ),
                patch.object(selector_module, "atomic_write_selection") as writer,
                patch.object(selector_module, "_destroy_all_windows"),
            ):
                with self.assertRaises(selector_module.BBoxSelectionCancelled):
                    selector_module.run_selector(_config(), output)
                writer.assert_not_called()

            with (
                patch.dict(os.environ, {"DISPLAY": ":test"}, clear=False),
                patch.object(
                    fastsam_engine,
                    "SharedFastSAMModel",
                    return_value=fake_model,
                ),
                patch.object(
                    selector_module,
                    "_select_camera",
                    side_effect=[
                        camera_results["cam0"],
                        camera_results["cam1"],
                    ],
                ),
                patch.object(selector_module, "atomic_write_selection") as writer,
                patch.object(selector_module, "_destroy_all_windows"),
            ):
                payload = selector_module.run_selector(_config(), output)
                writer.assert_called_once()
                self.assertEqual(
                    writer.call_args.args[0],
                    output,
                )
                self.assertEqual(
                    payload["cameras"]["cam1"]["bbox_xyxy"],
                    camera_results["cam1"]["bbox_xyxy"],
                )

    def test_resolve_effective_config_writes_runtime_copy_only(self):
        config = _config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "base.yaml"
            selection_path = root / "fastsam_bbox.yaml"
            cache_dir = root / "cache"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            atomic_write_selection(selection_path, _selection(config))

            resolved = resolve_effective_config(
                config_path,
                selection_path,
                cache_dir=cache_dir,
            )
            self.assertTrue(resolved.selection_applied)
            self.assertNotEqual(Path(resolved.config_path), config_path)
            with open(resolved.config_path, "r", encoding="utf-8") as handle:
                effective = yaml.safe_load(handle)
            self.assertEqual(
                effective["perception"]["object"]["fastsam"]["bboxes"]["cam0"],
                [100, 110, 300, 330],
            )
            with open(config_path, "r", encoding="utf-8") as handle:
                self.assertEqual(yaml.safe_load(handle), config)

    def test_missing_or_invalid_selection_is_fail_closed_unless_fallback_is_explicit(self):
        config = _config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "base.yaml"
            missing = root / "missing.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaises(FastSAMBBoxSelectionError):
                resolve_effective_config(config_path, missing, required=True)

            fallback = resolve_effective_config(
                config_path,
                missing,
                required=False,
            )
            self.assertFalse(fallback.selection_applied)
            self.assertEqual(fallback.config_path, str(config_path))
            self.assertIn("does not exist", fallback.fallback_reason)
            self.assertEqual(fallback.bboxes_xyxy["cam1"], (30, 40, 130, 240))

            missing.write_text("not: [valid", encoding="utf-8")
            invalid_fallback = resolve_effective_config(
                config_path,
                missing,
                required=False,
            )
            self.assertFalse(invalid_fallback.selection_applied)
            self.assertIn("cannot read", invalid_fallback.fallback_reason)

    def test_written_selection_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "fastsam_bbox.yaml"
            payload = _selection()
            atomic_write_selection(path, payload)
            self.assertEqual(load_selection(path), payload)


if __name__ == "__main__":
    unittest.main()
