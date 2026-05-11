import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from perception.shape_fitting_tracker_v2 import ShapeFittingTracker


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


class ShapeFittingZRotationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
