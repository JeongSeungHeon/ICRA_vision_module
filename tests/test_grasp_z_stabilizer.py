import sys
import types
import unittest

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *args, **kwargs: {}))

from perception.grasp_z_stabilizer import GraspPointZStabilizer


def make_config():
    return {
        "grasp": {
            "z_stabilization": {
                "enabled": True,
                "activation_dist_xy_m": 0.065,
                "max_step_z_m": 0.005,
                "ema_alpha": 0.70,
            }
        }
    }


class GraspPointZStabilizerTests(unittest.TestCase):
    def test_uses_reference_target_to_eef_xy_distance_for_activation(self):
        stabilizer = GraspPointZStabilizer(make_config())

        result = stabilizer.process(
            measured_grasp_point_base=(1.03, 0.54, 0.10),
            reference_xyz_mm=(30.0, 40.0, 100.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(result[0], 1.03, places=6)
        self.assertAlmostEqual(result[1], 0.54, places=6)
        self.assertAlmostEqual(result[2], 0.10, places=6)
        self.assertTrue(stabilizer.last_debug.active)
        self.assertAlmostEqual(stabilizer.last_debug.dist_xy_m, 0.05, places=6)

    def test_large_z_jump_is_step_limited_then_ema_smoothed(self):
        stabilizer = GraspPointZStabilizer(make_config())
        stabilizer.process(
            (0.01, 0.01, 0.100),
            reference_xyz_mm=(30.0, 40.0, 100.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        result = stabilizer.process(
            (0.02, 0.01, 0.200),
            reference_xyz_mm=(30.0, 40.0, 200.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(result[2], 0.1015, places=6)
        self.assertAlmostEqual(stabilizer.last_debug.limited_z_m, 0.105, places=6)
        self.assertAlmostEqual(stabilizer.last_debug.filtered_z_m, 0.1015, places=6)

    def test_raw_xy_passes_through_unchanged(self):
        stabilizer = GraspPointZStabilizer(make_config())
        stabilizer.process(
            (0.01, 0.01, 0.100),
            reference_xyz_mm=(30.0, 40.0, 100.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        result = stabilizer.process(
            (0.031, 0.032, 0.200),
            reference_xyz_mm=(30.0, 40.0, 200.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(result[0], 0.031, places=6)
        self.assertAlmostEqual(result[1], 0.032, places=6)
        self.assertNotAlmostEqual(result[2], 0.200, places=6)

    def test_inactive_distance_returns_raw_z_and_resets_reentry(self):
        stabilizer = GraspPointZStabilizer(make_config())
        stabilizer.process(
            (0.01, 0.01, 0.100),
            reference_xyz_mm=(30.0, 40.0, 100.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        inactive = stabilizer.process(
            (0.07, 0.0, 0.200),
            reference_xyz_mm=(70.0, 0.0, 200.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )
        inactive_reason = stabilizer.last_debug.reset_reason
        reentered = stabilizer.process(
            (0.01, 0.0, 0.300),
            reference_xyz_mm=(30.0, 40.0, 300.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(inactive[2], 0.200, places=6)
        self.assertEqual(inactive_reason, "outside_activation_distance")
        self.assertAlmostEqual(reentered[2], 0.300, places=6)
        self.assertTrue(stabilizer.last_debug.active)
        self.assertEqual(stabilizer.last_debug.reset_reason, "initialized")

    def test_missing_inputs_clear_history_and_return_raw_or_none(self):
        stabilizer = GraspPointZStabilizer(make_config())
        stabilizer.process(
            (0.01, 0.01, 0.100),
            reference_xyz_mm=(30.0, 40.0, 100.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        raw_after_missing_reference = stabilizer.process(
            (0.01, 0.01, 0.200),
            reference_xyz_mm=None,
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )
        none_after_missing_grasp = stabilizer.process(
            None,
            reference_xyz_mm=(30.0, 40.0, 100.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )
        reinitialized = stabilizer.process(
            (0.01, 0.01, 0.300),
            reference_xyz_mm=(30.0, 40.0, 300.0),
            eef_xyz_mm=(0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(raw_after_missing_reference[2], 0.200, places=6)
        self.assertIsNone(none_after_missing_grasp)
        self.assertAlmostEqual(reinitialized[2], 0.300, places=6)
        self.assertEqual(stabilizer.last_debug.reset_reason, "initialized")


if __name__ == "__main__":
    unittest.main()
