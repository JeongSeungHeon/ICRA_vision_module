import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import perception.shape_fitting_tracker_v2 as shape_fitting_tracker_module
from perception.shape_fitting_tracker_v2 import (
    SCALE_MODE_AXIS_XYZ,
    SCALE_MODE_UNIFORM,
    ShapeFittingTracker,
    _estimate_axis_scale,
)
from system.shared_state import MergedObjectState


def make_template_points() -> np.ndarray:
    return np.asarray(
        [
            [-0.05, -0.02, -0.01],
            [-0.05, -0.02, 0.01],
            [-0.05, 0.02, -0.01],
            [-0.05, 0.02, 0.01],
            [0.05, -0.02, -0.01],
            [0.05, -0.02, 0.01],
            [0.05, 0.02, -0.01],
            [0.05, 0.02, 0.01],
        ],
        dtype=np.float32,
    )


def make_box_points(extent_xyz: tuple[float, float, float], *, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    extent = np.asarray(extent_xyz, dtype=np.float32).reshape(3)
    center_arr = np.asarray(center, dtype=np.float32).reshape(3)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return center_arr.reshape(1, 3) + signs * (extent.reshape(1, 3) * 0.5)


def rotate_z_points(points: np.ndarray, angle_deg: float, *, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotation = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center_arr = np.asarray(center, dtype=np.float32).reshape(1, 3)
    return (np.asarray(points, dtype=np.float32) @ rotation.T) + center_arr


class ShapeFittingZRotationTests(unittest.TestCase):
    def test_handover_config_registers_bottle_template(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "configs" / "handover.yaml"
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        segmentation_cfg = config["perception"]["object"]["segmentation"]
        self.assertIn("bottle", segmentation_cfg["prompt_classes"])
        self.assertIn("bottle", segmentation_cfg["selection_class_names"])

        template_cfg = config["perception"]["shape_fitting"]["template_library"]
        loader = SimpleNamespace(_config_path=config_path)
        templates = ShapeFittingTracker._load_template_library(loader, template_cfg)

        self.assertIn("bottle", templates)
        bottle_template = templates["bottle"]
        self.assertEqual(bottle_template.template_id, "bottle_shape_fit")
        self.assertEqual(bottle_template.asset_path, repo_root / "shape_fitting" / "beer_bottle.npy")
        self.assertEqual(bottle_template.unit_scale_m, 0.001)
        self.assertEqual(bottle_template.scale_mode, SCALE_MODE_AXIS_XYZ)
        self.assertFalse(bottle_template.z_rotation_enabled)

    def test_template_z_rotation_is_per_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "box.npy", make_template_points())
            np.save(root / "cup.npy", make_template_points())

            loader = SimpleNamespace(_config_path=root / "handover.yaml")
            templates = ShapeFittingTracker._load_template_library(
                loader,
                {
                    "box-shaped snack": {
                        "template_id": "box",
                        "asset_path": "box.npy",
                        "unit_scale_m": 1.0,
                        "z_rotation": {
                            "enabled": True,
                            "min_deg": -10.0,
                            "max_deg": 10.0,
                            "step_deg": 10.0,
                        },
                    },
                    "cup": {
                        "template_id": "cup",
                        "asset_path": "cup.npy",
                        "unit_scale_m": 1.0,
                    },
                },
            )

            box_template = templates["box-shaped snack"]
            cup_template = templates["cup"]

            self.assertTrue(box_template.z_rotation_enabled)
            self.assertEqual(
                ShapeFittingTracker._z_rotation_candidate_degrees(loader, box_template),
                [-10.0, 0.0, 10.0],
            )
            self.assertFalse(cup_template.z_rotation_enabled)
            self.assertEqual(ShapeFittingTracker._z_rotation_candidate_degrees(loader, cup_template), [0.0])

    def test_axis_scale_estimation_matches_ranked_extents(self) -> None:
        scale_xyz = _estimate_axis_scale(
            np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            np.asarray([2.0, 4.0, 3.0], dtype=np.float64),
            0.5,
            3.0,
            fallback_scale=1.0,
        )

        np.testing.assert_allclose(scale_xyz, np.asarray([2.0, 1.5, 4.0 / 3.0]), rtol=1e-6, atol=1e-6)
        self.assertGreater(float(np.max(scale_xyz) - np.min(scale_xyz)), 0.25)

    def test_axis_scale_estimation_clips_and_falls_back_per_axis(self) -> None:
        scale_xyz = _estimate_axis_scale(
            np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([10.0, 0.2, 20.0], dtype=np.float64),
            0.5,
            3.0,
            fallback_scale=1.25,
        )

        self.assertEqual(float(scale_xyz[0]), 1.25)
        self.assertGreaterEqual(float(np.min(scale_xyz)), 0.5)
        self.assertLessEqual(float(np.max(scale_xyz)), 3.0)

    @unittest.skipIf(shape_fitting_tracker_module.o3d is None, "open3d is not installed")
    def test_tracker_uses_axis_scale_only_for_configured_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_points = make_box_points((1.0, 2.0, 3.0))
            target_points = make_box_points((2.0, 4.0, 3.0), center=(0.2, 0.1, 0.3))
            np.save(root / "box.npy", template_points)
            np.save(root / "cup.npy", template_points)

            config = {
                "perception": {
                    "shape_fitting": {
                        "enabled": True,
                        "template_library": {
                            "box-shaped snack": {
                                "template_id": "box",
                                "asset_path": "box.npy",
                                "unit_scale_m": 1.0,
                                "scale_mode": "axis_xyz",
                            },
                            "cup": {
                                "template_id": "cup",
                                "asset_path": "cup.npy",
                                "unit_scale_m": 1.0,
                            },
                        },
                        "cluster": {"dbscan_eps_m": 10.0, "dbscan_min_points": 1, "max_cluster_jump_m": 10.0},
                        "scale_init": {
                            "mode": "uniform",
                            "stable_frames": 1,
                            "percentile_low": 0.0,
                            "percentile_high": 100.0,
                            "min_scale": 0.5,
                            "max_scale": 3.0,
                        },
                        "tracking": {"min_cluster_extent_m": 1.0e-8},
                        "icp": {
                            "max_points": 100,
                            "max_iterations": 2,
                            "distance_threshold_m": 10.0,
                            "crop": {"enabled": False},
                        },
                        "downsample": {"voxel_size_m": 0.0, "max_points": 100},
                    }
                }
            }

            box_tracker = ShapeFittingTracker(config=config, config_path=root / "handover.yaml")
            box_state = box_tracker.process(
                MergedObjectState(
                    valid=True,
                    object_detected=True,
                    label="box-shaped snack",
                    merged_point_count=len(target_points),
                    merged_points_base=[tuple(float(v) for v in point) for point in target_points],
                )
            )

            self.assertTrue(box_state.valid)
            self.assertEqual(box_state.scale_mode, SCALE_MODE_AXIS_XYZ)
            self.assertIsNotNone(box_state.scale_xyz)
            self.assertIsNotNone(box_state.template_axes_base)
            assert box_state.scale_xyz is not None
            assert box_state.template_axes_base is not None
            self.assertGreater(float(np.max(box_state.scale_xyz) - np.min(box_state.scale_xyz)), 0.25)
            axes = np.asarray(box_state.template_axes_base, dtype=np.float32)
            self.assertEqual(axes.shape, (3, 3))
            np.testing.assert_allclose(np.linalg.norm(axes, axis=1), np.ones((3,), dtype=np.float32), atol=1e-5)

            cup_tracker = ShapeFittingTracker(config=config, config_path=root / "handover.yaml")
            cup_state = cup_tracker.process(
                MergedObjectState(
                    valid=True,
                    object_detected=True,
                    label="cup",
                    merged_point_count=len(target_points),
                    merged_points_base=[tuple(float(v) for v in point) for point in target_points],
                )
            )

            self.assertTrue(cup_state.valid)
            self.assertEqual(cup_state.scale_mode, SCALE_MODE_UNIFORM)
            self.assertIsNotNone(cup_state.scale_xyz)
            assert cup_state.scale_xyz is not None
            np.testing.assert_allclose(cup_state.scale_xyz, np.full((3,), cup_state.scale), rtol=1e-6, atol=1e-6)

    @unittest.skipIf(shape_fitting_tracker_module.o3d is None, "open3d is not installed")
    def test_template_axes_base_tracks_configured_z_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_points = make_box_points((1.0, 2.0, 3.0))
            target_points = rotate_z_points(template_points, 90.0, center=(0.2, 0.1, 0.3))
            np.save(root / "box.npy", template_points)

            config = {
                "perception": {
                    "shape_fitting": {
                        "enabled": True,
                        "template_library": {
                            "box-shaped snack": {
                                "template_id": "box",
                                "asset_path": "box.npy",
                                "unit_scale_m": 1.0,
                                "scale_mode": "uniform",
                                "z_rotation": {
                                    "enabled": True,
                                    "min_deg": 90.0,
                                    "max_deg": 90.0,
                                    "step_deg": 5.0,
                                },
                            },
                        },
                        "cluster": {"dbscan_eps_m": 10.0, "dbscan_min_points": 1, "max_cluster_jump_m": 10.0},
                        "scale_init": {
                            "mode": "uniform",
                            "stable_frames": 1,
                            "percentile_low": 0.0,
                            "percentile_high": 100.0,
                            "min_scale": 0.5,
                            "max_scale": 3.0,
                        },
                        "tracking": {"min_cluster_extent_m": 1.0e-8},
                        "icp": {
                            "max_points": 100,
                            "max_iterations": 2,
                            "distance_threshold_m": 10.0,
                            "crop": {"enabled": False},
                        },
                        "downsample": {"voxel_size_m": 0.0, "max_points": 100},
                    }
                }
            }

            tracker = ShapeFittingTracker(config=config, config_path=root / "handover.yaml")
            state = tracker.process(
                MergedObjectState(
                    valid=True,
                    object_detected=True,
                    label="box-shaped snack",
                    merged_point_count=len(target_points),
                    merged_points_base=[tuple(float(v) for v in point) for point in target_points],
                )
            )

            self.assertTrue(state.valid)
            self.assertEqual(state.z_rotation_deg, 90.0)
            self.assertIsNotNone(state.template_axes_base)
            assert state.template_axes_base is not None
            axes = np.asarray(state.template_axes_base, dtype=np.float32)
            np.testing.assert_allclose(axes[0], np.asarray([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
