import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from perception.grasp_target import GraspTargetPlanner
from system.shared_state import MergedObjectState, SelectedHandState


def make_config(**fallback_overrides):
    fallback_cfg = {
        "enabled": True,
        "z_crop_margin_m": 0.01,
    }
    fallback_cfg.update(fallback_overrides)
    return {
        "frames": {"height_axis": {"name": "z"}},
        "grasp": {
            "candidate_search_radius_from_centroid_m": 1.0,
            "min_hand_height_clearance_m": 0.07,
            "object_cloud_fallback": fallback_cfg,
            "target_selection": {
                "enabled": True,
                "hold_frames": 6,
                "dropout_hold_frames": 8,
                "switch_margin": 0.01,
                "score_previous_distance_weight": 0.90,
            },
        },
    }


def make_object(points, *, valid=True, centroid=(0.0, 0.0, 0.0)):
    return MergedObjectState(
        valid=valid,
        object_detected=valid,
        centroid_base=centroid,
        merged_point_count=len(points),
        merged_points_base=[tuple(float(v) for v in point) for point in points],
    )


def make_hand(z=0.0):
    return SelectedHandState(
        valid=True,
        palm_center_base=(0.0, 0.0, float(z)),
    )


class GraspTargetPlannerTests(unittest.TestCase):
    def test_raw_object_fallback_supplies_candidate_when_template_fails_clearance(self):
        planner = GraspTargetPlanner(make_config())
        template = make_object([(0.0, 0.0, 0.02), (0.01, 0.0, -0.02)])
        raw = make_object([
            (0.0, 0.0, -0.10),
            (0.0, 0.0, -0.08),
            (0.0, 0.0, 0.10),
        ])

        state = planner.process_states(template, make_hand(), fallback_object=raw)

        self.assertTrue(state.valid)
        self.assertAlmostEqual(state.target_position_base[2], -0.08, places=6)
        self.assertEqual(planner.last_debug.candidate_source, "raw_object")
        self.assertEqual(planner.last_debug.selection_reason, "raw_object_selected_within_search_radius")

    def test_raw_object_fallback_excludes_top_and_bottom_z_caps(self):
        planner = GraspTargetPlanner(make_config())
        template = make_object([(0.0, 0.0, 0.02)])
        raw = make_object([
            (0.0, 0.0, -0.10),
            (0.0, 0.0, 0.10),
        ])

        state = planner.process_states(template, make_hand(), fallback_object=raw)

        self.assertFalse(state.valid)
        self.assertEqual(planner.last_debug.selection_reason, "raw_object_empty_after_z_crop")
        self.assertEqual(planner.last_debug.candidate_source, "none")

    def test_template_candidate_wins_without_using_raw_fallback(self):
        planner = GraspTargetPlanner(make_config())
        template = make_object([(0.0, 0.0, -0.08)])
        raw = make_object([(0.0, 0.0, 0.08)])

        state = planner.process_states(template, make_hand(), fallback_object=raw)

        self.assertTrue(state.valid)
        self.assertAlmostEqual(state.target_position_base[2], -0.08, places=6)
        self.assertEqual(planner.last_debug.candidate_source, "template")

    def test_uses_dropout_hold_only_after_template_and_raw_fail(self):
        planner = GraspTargetPlanner(make_config())
        template_ok = make_object([(0.0, 0.0, -0.08)])
        raw_ok = make_object([(0.0, 0.0, 0.08)])
        first = planner.process_states(template_ok, make_hand(), fallback_object=raw_ok)
        self.assertTrue(first.valid)

        template_fail = make_object([(0.0, 0.0, 0.02)])
        raw_fail = make_object([(0.0, 0.0, -0.10), (0.0, 0.0, 0.10)])
        held = planner.process_states(template_fail, make_hand(), fallback_object=raw_fail)

        self.assertTrue(held.valid)
        self.assertTrue(held.dropout_hold_active)
        self.assertAlmostEqual(held.target_position_base[2], -0.08, places=6)
        self.assertEqual(planner.last_debug.selection_reason, "hold_raw_object_empty_after_z_crop")

    def test_raw_object_fallback_z_crop_margin_can_be_overridden(self):
        planner = GraspTargetPlanner(make_config(z_crop_margin_m=0.02))
        template = make_object([(0.0, 0.0, 0.02)])
        raw = make_object([
            (0.0, 0.0, -0.10),
            (0.0, 0.0, -0.085),
            (0.0, 0.0, 0.10),
        ])

        state = planner.process_states(template, make_hand(), fallback_object=raw)

        self.assertFalse(state.valid)
        self.assertEqual(planner.last_debug.selection_reason, "raw_object_empty_after_z_crop")
        self.assertAlmostEqual(planner.object_cloud_fallback_z_crop_margin_m, 0.02, places=6)

    def test_raw_object_fallback_default_z_crop_margin_is_one_centimeter(self):
        planner = GraspTargetPlanner({"grasp": {}})

        self.assertAlmostEqual(planner.object_cloud_fallback_z_crop_margin_m, 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
