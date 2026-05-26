import sys
import types
import unittest
import contextlib
import io

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *args, **kwargs: {}))

from perception.hand_selector import HandSelector, object_center_for_hand_selection
from system.shared_state import HandCandidateState, HandState, ObjectState


OBJECT_CENTER = (0.0, 0.0, 0.0)


def make_config():
    return {
        "perception": {
            "hand_selection": {
                "candidate_identity_mode": "handedness",
                "allowed_active_candidate_ids": ["cam0:left", "cam1:right"],
                "active_hand_distance_threshold_m": 0.15,
                "switch_margin_m": 0.05,
                "switch_confirm_frames": 5,
                "lost_timeout_s": 0.5,
                "lock_active_hand_during_task": True,
            }
        }
    }


def make_candidate(
    camera_id,
    candidate_index,
    *,
    center,
    frame_id=0,
    handedness="right",
    confidence=0.9,
    valid=True,
    timestamp=None,
    candidate_id=None,
):
    normalized_handedness = str(handedness).strip().lower()
    if candidate_id is None:
        if normalized_handedness in {"left", "right"}:
            candidate_id = f"cam{int(camera_id)}:{normalized_handedness}"
        else:
            candidate_id = f"cam{int(camera_id)}:unknown:hand{int(candidate_index)}"
    return HandCandidateState(
        camera_id=int(camera_id),
        frame_id=int(frame_id),
        candidate_index=int(candidate_index),
        candidate_id=candidate_id,
        hand_detected=bool(valid),
        handedness=str(handedness),
        confidence=float(confidence),
        palm_center_base=None if not valid else tuple(float(v) for v in center),
        palm_normal_base=None if not valid else (0.0, 0.0, 1.0),
        wrist_base=None if not valid else (float(center[0]) - 0.02, float(center[1]), float(center[2])),
        hand_velocity_base=None if not valid else (0.0, 0.0, 0.0),
        timestamp=float(frame_id if timestamp is None else timestamp),
        valid=bool(valid),
    )


def make_hand(camera_id, candidates=(), *, frame_id=0, timestamp=None):
    candidates = list(candidates)
    primary = candidates[0] if candidates else None
    return HandState(
        camera_id=int(camera_id),
        frame_id=int(frame_id),
        hand_detected=primary is not None,
        handedness="unknown" if primary is None else primary.handedness,
        confidence=0.0 if primary is None else primary.confidence,
        palm_center_base=None if primary is None else primary.palm_center_base,
        palm_normal_base=None if primary is None else primary.palm_normal_base,
        wrist_base=None if primary is None else primary.wrist_base,
        hand_velocity_base=None if primary is None else primary.hand_velocity_base,
        hand_candidates=candidates,
        timestamp=float(frame_id if timestamp is None else timestamp),
        valid=primary is not None,
    )


def empty_hand(camera_id, *, frame_id=0, timestamp=None):
    return make_hand(camera_id, (), frame_id=frame_id, timestamp=timestamp)


class HandSelectorDistanceTests(unittest.TestCase):
    def test_selects_only_hand_within_object_threshold(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.10, 0.0, 0.0), handedness="left")]),
            make_hand(1, [make_candidate(1, 0, center=(0.30, 0.0, 0.0), handedness="right")]),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 0)
        self.assertEqual(selected.selected_candidate_id, "cam0:left")
        self.assertEqual(selector.last_debug.selection_reason, "select_within_object_threshold")

    def test_chooses_closer_hand_when_both_are_within_threshold(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.12, 0.0, 0.0), handedness="left")]),
            make_hand(1, [make_candidate(1, 0, center=(0.08, 0.0, 0.0), handedness="right")]),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 1)
        self.assertEqual(selected.selected_candidate_id, "cam1:right")

    def test_returns_invalid_when_no_hand_has_entered_threshold(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.20, 0.0, 0.0), handedness="left")]),
            make_hand(1, [make_candidate(1, 0, center=(0.30, 0.0, 0.0), handedness="right")]),
            object_center_base=OBJECT_CENTER,
        )

        self.assertFalse(selected.valid)
        self.assertEqual(selector.last_debug.selection_reason, "no_allowed_candidate_within_threshold")

    def test_keeps_previous_selected_hand_when_none_are_currently_within_threshold(self):
        selector = HandSelector(make_config())
        selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.10, 0.0, 0.0), handedness="left", frame_id=0)]),
            empty_hand(1, frame_id=0),
            object_center_base=OBJECT_CENTER,
        )

        selected = selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.20, 0.0, 0.0), handedness="left", frame_id=1)]),
            make_hand(1, [make_candidate(1, 0, center=(0.25, 0.0, 0.0), frame_id=1, handedness="right")]),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 0)
        self.assertEqual(selected.palm_center_base, (0.20, 0.0, 0.0))
        self.assertEqual(selector.last_debug.selection_reason, "keep_locked_active_hand")

    def test_switches_only_after_challenger_is_clearly_closer_for_confirm_frames(self):
        selector = HandSelector(make_config())
        selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.14, 0.0, 0.0), handedness="left", frame_id=0)]),
            empty_hand(1, frame_id=0),
            object_center_base=OBJECT_CENTER,
        )

        selected = None
        for frame_id in range(1, 5):
            selected = selector.process_states(
                make_hand(0, [make_candidate(0, 0, center=(0.14, 0.0, 0.0), handedness="left", frame_id=frame_id)]),
                make_hand(1, [make_candidate(1, 0, center=(0.08, 0.0, 0.0), frame_id=frame_id, handedness="right")]),
                object_center_base=OBJECT_CENTER,
            )
            self.assertTrue(selected.valid)
            self.assertEqual(selected.selected_camera, 0)

        selected = selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.14, 0.0, 0.0), handedness="left", frame_id=5)]),
            make_hand(1, [make_candidate(1, 0, center=(0.08, 0.0, 0.0), frame_id=5, handedness="right")]),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 1)
        self.assertEqual(selector.last_debug.selection_reason, "switch_after_confirmed_closer_to_object")

    def test_releases_current_hand_after_lost_timeout(self):
        selector = HandSelector(make_config())
        selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.10, 0.0, 0.0), handedness="left", timestamp=0.0)], timestamp=0.0),
            empty_hand(1, timestamp=0.0),
            object_center_base=OBJECT_CENTER,
        )

        held = selector.process_states(
            empty_hand(0, timestamp=0.4),
            empty_hand(1, timestamp=0.4),
            object_center_base=OBJECT_CENTER,
        )
        released = selector.process_states(
            empty_hand(0, timestamp=0.6),
            empty_hand(1, timestamp=0.6),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(held.valid)
        self.assertFalse(released.valid)
        self.assertEqual(selector.last_debug.selection_reason, "no_allowed_candidate")

    def test_object_center_helper_uses_point_count_weighted_average(self):
        center = object_center_for_hand_selection(
            ObjectState(camera_id=0, centroid_base=(0.0, 0.0, 0.0), point_count=1, valid=True),
            ObjectState(camera_id=1, centroid_base=(1.0, 0.0, 0.0), point_count=3, valid=True),
        )

        self.assertEqual(center, (0.75, 0.0, 0.0))

    def test_initial_selection_chooses_closest_handedness_candidate(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(
                0,
                [
                    make_candidate(0, 0, center=(0.13, 0.0, 0.0), handedness="right"),
                    make_candidate(0, 1, center=(0.11, 0.0, 0.0), handedness="left"),
                ],
            ),
            make_hand(
                1,
                [
                    make_candidate(1, 0, center=(0.09, 0.0, 0.0), handedness="right"),
                    make_candidate(1, 1, center=(0.07, 0.0, 0.0), handedness="left"),
                ],
            ),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam1:right")
        self.assertNotIn("cam0:right", selector.last_debug.active_candidate_ids)
        self.assertNotIn("cam1:left", selector.last_debug.active_candidate_ids)
        self.assertIn("cam0:right", selector.last_debug.disallowed_candidate_ids)
        self.assertIn("cam1:left", selector.last_debug.disallowed_candidate_ids)

    def test_only_disallowed_candidates_are_invalid_before_distance_calculation(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.02, 0.0, 0.0), handedness="right")]),
            make_hand(1, [make_candidate(1, 0, center=(0.03, 0.0, 0.0), handedness="left")]),
            object_center_base=OBJECT_CENTER,
        )

        self.assertFalse(selected.valid)
        self.assertEqual(selector.last_debug.selection_reason, "no_allowed_candidate")
        self.assertEqual(selector.last_debug.active_candidate_ids, ())
        self.assertIn("cam0:right", selector.last_debug.disallowed_candidate_ids)
        self.assertIn("cam1:left", selector.last_debug.disallowed_candidate_ids)
        self.assertIsNone(selector.last_debug.chosen_distance_m)

    def test_disallowed_candidate_closer_to_object_does_not_override_allowed_candidate(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(
                0,
                [
                    make_candidate(0, 0, center=(0.01, 0.0, 0.0), handedness="right"),
                    make_candidate(0, 1, center=(0.12, 0.0, 0.0), handedness="left"),
                ],
            ),
            make_hand(1, [make_candidate(1, 0, center=(0.02, 0.0, 0.0), handedness="left")]),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam0:left")
        self.assertAlmostEqual(selector.last_debug.chosen_distance_m, 0.12, places=6)

    def test_current_active_hand_does_not_switch_to_closer_disallowed_candidate(self):
        selector = HandSelector(make_config())
        selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.12, 0.0, 0.0), handedness="left", frame_id=0)]),
            empty_hand(1, frame_id=0),
            object_center_base=OBJECT_CENTER,
        )

        for frame_id in range(1, 7):
            selected = selector.process_states(
                make_hand(
                    0,
                    [
                        make_candidate(0, 0, center=(0.12, 0.0, 0.0), handedness="left", frame_id=frame_id),
                        make_candidate(0, 1, center=(0.01, 0.0, 0.0), handedness="right", frame_id=frame_id),
                    ],
                ),
                make_hand(1, [make_candidate(1, 0, center=(0.02, 0.0, 0.0), handedness="left", frame_id=frame_id)]),
                object_center_base=OBJECT_CENTER,
            )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam0:left")
        self.assertEqual(selector.last_debug.selection_reason, "keep_locked_active_hand")
        self.assertIsNone(selector.last_debug.pending_candidate_id)

    def test_lost_timeout_replacement_selects_only_allowed_candidates(self):
        config = make_config()
        config["perception"]["hand_selection"]["lost_timeout_s"] = 0.1
        selector = HandSelector(config)
        selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.10, 0.0, 0.0), handedness="left", timestamp=0.0)], timestamp=0.0),
            empty_hand(1, timestamp=0.0),
            object_center_base=OBJECT_CENTER,
        )

        selected = selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.01, 0.0, 0.0), handedness="right", timestamp=0.2)], timestamp=0.2),
            make_hand(
                1,
                [
                    make_candidate(1, 0, center=(0.02, 0.0, 0.0), handedness="left", timestamp=0.2),
                    make_candidate(1, 1, center=(0.09, 0.0, 0.0), handedness="right", timestamp=0.2),
                ],
                timestamp=0.2,
            ),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam1:right")
        self.assertEqual(selector.last_debug.selection_reason, "select_after_lost_timeout")

    def test_keeps_same_handedness_candidate_when_mediapipe_index_swaps(self):
        selector = HandSelector(make_config())
        selector.process_states(
            make_hand(0, [make_candidate(0, 0, center=(0.10, 0.0, 0.0), handedness="left", frame_id=0)]),
            empty_hand(1, frame_id=0),
            object_center_base=OBJECT_CENTER,
        )

        selected = selector.process_states(
            make_hand(
                0,
                [
                    make_candidate(0, 1, center=(0.11, 0.0, 0.0), handedness="left", frame_id=1),
                    make_candidate(0, 0, center=(0.02, 0.0, 0.0), handedness="right", frame_id=1),
                ],
            ),
            empty_hand(1, frame_id=1),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam0:left")
        self.assertEqual(selected.selected_candidate_index, 1)
        self.assertEqual(selector.last_debug.selection_reason, "keep_locked_active_hand")

    def test_duplicate_same_handedness_keeps_higher_confidence_candidate(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(
                0,
                [
                    make_candidate(0, 0, center=(0.06, 0.0, 0.0), handedness="left", confidence=0.40),
                    make_candidate(0, 1, center=(0.12, 0.0, 0.0), handedness="left", confidence=0.90),
                ],
            ),
            empty_hand(1),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam0:left")
        self.assertEqual(selected.selected_candidate_index, 1)
        self.assertIn("duplicate_handedness_candidate", selector.last_debug.duplicate_drop_reasons[0])

    def test_duplicate_same_handedness_tie_keeps_closer_candidate(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(
                0,
                [
                    make_candidate(0, 0, center=(0.12, 0.0, 0.0), handedness="left", confidence=0.90),
                    make_candidate(0, 1, center=(0.06, 0.0, 0.0), handedness="left", confidence=0.90),
                ],
            ),
            empty_hand(1),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_index, 1)

    def test_unknown_handedness_does_not_override_known_candidate(self):
        selector = HandSelector(make_config())

        selected = selector.process_states(
            make_hand(
                0,
                [
                    make_candidate(0, 0, center=(0.12, 0.0, 0.0), handedness="left"),
                    make_candidate(0, 1, center=(0.03, 0.0, 0.0), handedness="unknown"),
                ],
            ),
            empty_hand(1),
            object_center_base=OBJECT_CENTER,
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam0:left")

    def test_logs_selected_active_hand_candidate_when_enabled(self):
        config = make_config()
        config["perception"]["hand_selection"]["log_active_hand_selection"] = True
        selector = HandSelector(config)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            selected = selector.process_states(
                make_hand(0, [make_candidate(0, 0, center=(0.12, 0.0, 0.0), handedness="right")]),
                make_hand(
                    1,
                    [
                        make_candidate(1, 0, center=(0.08, 0.0, 0.0), handedness="right"),
                        make_candidate(1, 1, center=(0.04, 0.0, 0.0), handedness="left"),
                    ],
                ),
                object_center_base=OBJECT_CENTER,
            )

        log_text = output.getvalue()
        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_candidate_id, "cam1:right")
        self.assertIn("[HandSelector] ACTIVE_HAND_SELECTED", log_text)
        self.assertIn("active=cam1:right", log_text)
        self.assertIn("cam0:right:blocked", log_text)
        self.assertIn("cam1:left:blocked", log_text)
        self.assertIn("cam0:left:missing", log_text)
        self.assertIn("cam1:right:", log_text)


if __name__ == "__main__":
    unittest.main()
