"""Runtime SAM3D template override tests for ShapeFittingTracker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from perception.shape_fitting_tracker_v2 import ShapeFittingTracker
from system.shared_state import MergedObjectState


class Sam3DRuntimeTemplateTest(unittest.TestCase):
    @staticmethod
    def _runtime_override(runtime_path):
        return {
            "label": "sam3d_object",
            "asset_path": str(runtime_path),
            "template_id": "sam3d_runtime",
            "unit_scale_m": 1.0,
            "scale_mode": "axis_xyz",
        }

    def test_override_replaces_static_library_and_survives_reset(self):
        points = np.array(
            [
                [-0.05, -0.03, -0.02],
                [-0.05, -0.03, 0.02],
                [-0.05, 0.03, -0.02],
                [-0.05, 0.03, 0.02],
                [0.05, -0.03, -0.02],
                [0.05, -0.03, 0.02],
                [0.05, 0.03, -0.02],
                [0.05, 0.03, 0.02],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / "runtime.npy"
            static_path = root / "static.npy"
            np.save(runtime_path, points)
            np.save(static_path, points * 2.0)
            config = {
                "perception": {
                    "shape_fitting": {
                        "enabled": True,
                        "template_library": {
                            "cup": {
                                "asset_path": str(static_path),
                                "template_id": "cup_static",
                                "unit_scale_m": 1.0,
                            }
                        },
                        "silhouette_constraint": {
                            "enabled": True,
                            "apply_to_labels": ["cup"],
                        },
                    }
                }
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            tracker = ShapeFittingTracker(
                config,
                config_path=config_path,
                runtime_template_override=self._runtime_override(runtime_path),
            )
            self.assertEqual(set(tracker._templates), {"sam3d_object"})
            self.assertEqual(tracker._templates["sam3d_object"].template_id, "sam3d_runtime")
            self.assertIn("sam3d_object", tracker.silhouette_apply_to_labels)
            tracker.reset()
            self.assertEqual(set(tracker._templates), {"sam3d_object"})
            np.testing.assert_array_equal(
                tracker._templates["sam3d_object"].canonical_points,
                points,
            )

    def test_synthetic_cloud_initializes_runtime_template_again_after_reset(self):
        rng = np.random.default_rng(7)
        points = rng.uniform(
            low=np.array([-0.05, -0.03, -0.02]),
            high=np.array([0.05, 0.03, 0.02]),
            size=(600, 3),
        ).astype(np.float32)
        observed = points + np.array([0.4, -0.1, 0.3], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / "runtime.npy"
            np.save(runtime_path, points)
            config = {
                "perception": {
                    "shape_fitting": {
                        "enabled": True,
                        "template_library": {
                            "unused": {
                                "asset_path": str(runtime_path),
                                "unit_scale_m": 1.0,
                            }
                        },
                        "cluster": {
                            "dbscan_eps_m": 0.03,
                            "dbscan_min_points": 3,
                            "max_cluster_jump_m": 1.0,
                        },
                        "scale_init": {
                            "stable_frames": 1,
                            "percentile_low": 0.0,
                            "percentile_high": 100.0,
                            "min_scale": 0.5,
                            "max_scale": 1.8,
                        },
                        "icp": {
                            "max_points": 1000,
                            "max_iterations": 5,
                            "distance_threshold_m": 0.2,
                            "min_fitness": 0.0,
                            "max_centroid_jump_m": 1.0,
                            "crop": {"enabled": False},
                        },
                        "downsample": {"voxel_size_m": 0.0, "max_points": 1000},
                        "silhouette_constraint": {"enabled": False},
                    }
                }
            }
            tracker = ShapeFittingTracker(
                config,
                config_path=root / "config.yaml",
                runtime_template_override=self._runtime_override(runtime_path),
            )
            merged = MergedObjectState(
                frame_id_cam0=1,
                frame_id_cam1=1,
                object_detected=True,
                label="sam3d_object",
                confidence=0.9,
                centroid_base=tuple(np.mean(observed, axis=0)),
                merged_point_count=len(observed),
                merged_points_base=[tuple(float(value) for value in point) for point in observed],
                valid=True,
            )
            first = tracker.process(merged)
            self.assertTrue(first.valid, first.reason)
            self.assertEqual(first.template_id, "sam3d_runtime")
            np.testing.assert_allclose(first.centroid_base, np.mean(observed, axis=0), atol=0.02)

            tracker.reset()
            second = tracker.process(merged)
            self.assertTrue(second.valid, second.reason)
            self.assertEqual(second.template_id, "sam3d_runtime")

    def test_silhouette_ablation_override_keeps_shape_fit_and_marks_reason(self):
        rng = np.random.default_rng(11)
        points = rng.uniform(-0.03, 0.03, size=(300, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / "runtime.npy"
            np.save(runtime_path, points)
            config = {
                "perception": {
                    "shape_fitting": {
                        "enabled": True,
                        "template_library": {"unused": {"asset_path": str(runtime_path)}},
                        "cluster": {"dbscan_eps_m": 0.05, "dbscan_min_points": 3},
                        "scale_init": {"stable_frames": 1},
                        "icp": {"crop": {"enabled": False}},
                        "silhouette_constraint": {"enabled": True},
                    }
                }
            }
            tracker = ShapeFittingTracker(
                config,
                config_path=root / "config.yaml",
                runtime_template_override=self._runtime_override(runtime_path),
                silhouette_enabled_override=False,
            )
            observed = points + np.asarray([0.4, 0.0, 0.4], dtype=np.float32)
            merged = MergedObjectState(
                object_detected=True,
                label="sam3d_object",
                centroid_base=tuple(np.mean(observed, axis=0)),
                merged_point_count=len(observed),
                merged_points_base=[tuple(point) for point in observed],
                valid=True,
            )

            state = tracker.process(merged, silhouette_observations=[])

            self.assertTrue(state.valid)
            self.assertTrue(state.initialized)
            self.assertFalse(state.silhouette_enabled)
            self.assertEqual(state.silhouette_reason, "disabled_by_ablation")


if __name__ == "__main__":
    unittest.main()
