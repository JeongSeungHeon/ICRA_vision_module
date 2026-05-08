import unittest
import sys
import types

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *args, **kwargs: {}))

from perception.hand_selector import HandSelector
from system.shared_state import HandState


def make_config(lock_after_stable_frames=8):
    return {
        "perception": {
            "hand": {
                "handedness_selection": {
                    "right_hand_camera": 0,
                    "left_hand_camera": 1,
                    "hysteresis_frames": 5,
                    "dropout_hold_frames": 12,
                    "confidence_drop_margin": 0.15,
                    "lock_after_stable_frames": int(lock_after_stable_frames),
                    "lock_hold_last_on_dropout": True,
                }
            }
        }
    }


def make_hand(
    camera_id,
    *,
    frame_id=0,
    handedness="right",
    confidence=0.9,
    center=(0.30, 0.10, 0.40),
    valid=True,
):
    return HandState(
        camera_id=int(camera_id),
        frame_id=int(frame_id),
        hand_detected=bool(valid),
        handedness=str(handedness),
        confidence=float(confidence),
        palm_center_base=None if not valid else tuple(float(v) for v in center),
        palm_normal_base=None if not valid else (0.0, 0.0, 1.0),
        wrist_base=None if not valid else (center[0] - 0.02, center[1], center[2]),
        hand_velocity_base=None if not valid else (0.0, 0.0, 0.0),
        timestamp=float(frame_id),
        valid=bool(valid),
    )


def invalid_hand(camera_id, frame_id=0):
    return make_hand(camera_id, frame_id=frame_id, valid=False)


class HandSelectorLockTests(unittest.TestCase):
    def test_locks_after_eight_stable_frames(self):
        selector = HandSelector(make_config(lock_after_stable_frames=8))

        selected = None
        for frame_id in range(8):
            selected = selector.process_states(
                make_hand(0, frame_id=frame_id, handedness="right"),
                invalid_hand(1, frame_id=frame_id),
            )

        self.assertIsNotNone(selected)
        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 0)
        self.assertEqual(selector.last_debug.locked_camera, 0)
        self.assertEqual(selector.last_debug.lock_stable_frames, 8)
        self.assertEqual(selector.last_debug.selection_reason, "lock_acquired")

    def test_keeps_locked_camera_when_opposite_camera_has_higher_confidence(self):
        selector = HandSelector(make_config(lock_after_stable_frames=8))
        for frame_id in range(8):
            selector.process_states(
                make_hand(0, frame_id=frame_id, handedness="right", confidence=0.7),
                invalid_hand(1, frame_id=frame_id),
            )

        selected = selector.process_states(
            make_hand(0, frame_id=8, handedness="right", confidence=0.4, center=(0.31, 0.10, 0.40)),
            make_hand(1, frame_id=8, handedness="left", confidence=0.99, center=(0.80, 0.10, 0.40)),
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 0)
        self.assertEqual(selected.palm_center_base, (0.31, 0.10, 0.40))
        self.assertEqual(selector.last_debug.locked_camera, 0)
        self.assertEqual(selector.last_debug.selection_reason, "locked_camera_current")

    def test_uses_latest_locked_camera_state_even_if_handedness_flips(self):
        selector = HandSelector(make_config(lock_after_stable_frames=8))
        for frame_id in range(8):
            selector.process_states(
                make_hand(0, frame_id=frame_id, handedness="right"),
                invalid_hand(1, frame_id=frame_id),
            )

        selected = selector.process_states(
            make_hand(0, frame_id=8, handedness="left", confidence=0.8, center=(0.35, 0.10, 0.40)),
            invalid_hand(1, frame_id=8),
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 0)
        self.assertEqual(selected.handedness, "left")
        self.assertEqual(selected.palm_center_base, (0.35, 0.10, 0.40))
        self.assertEqual(selector.last_debug.selection_reason, "locked_camera_current")

    def test_holds_last_locked_state_when_locked_camera_drops_out(self):
        selector = HandSelector(make_config(lock_after_stable_frames=8))
        for frame_id in range(8):
            selector.process_states(
                make_hand(0, frame_id=frame_id, handedness="right", center=(0.30 + frame_id * 0.01, 0.10, 0.40)),
                invalid_hand(1, frame_id=frame_id),
            )

        selected = selector.process_states(
            invalid_hand(0, frame_id=8),
            make_hand(1, frame_id=8, handedness="left", confidence=0.99, center=(0.90, 0.10, 0.40)),
        )

        self.assertTrue(selected.valid)
        self.assertEqual(selected.selected_camera, 0)
        self.assertEqual(selected.palm_center_base, (0.37, 0.10, 0.40))
        self.assertEqual(selector.last_debug.locked_camera, 0)
        self.assertEqual(selector.last_debug.selection_reason, "locked_camera_hold_last_dropout")

    def test_reset_clears_lock_state(self):
        selector = HandSelector(make_config(lock_after_stable_frames=8))
        for frame_id in range(8):
            selector.process_states(
                make_hand(0, frame_id=frame_id, handedness="right"),
                invalid_hand(1, frame_id=frame_id),
            )

        selector.reset()

        self.assertIsNone(selector._locked_camera)
        self.assertIsNone(selector._lock_stable_camera)
        self.assertEqual(selector._lock_stable_frames, 0)
        self.assertIsNone(selector._locked_state)

    def test_records_reject_reason_for_invalid_candidate(self):
        selector = HandSelector(make_config(lock_after_stable_frames=8))

        selected = selector.process_states(
            make_hand(0, handedness="left"),
            invalid_hand(1),
        )

        self.assertFalse(selected.valid)
        self.assertEqual(selector.last_debug.cam0_reject_reason, "handedness_mismatch:left->right")
        self.assertEqual(selector.last_debug.cam1_reject_reason, "hand_not_detected")
        self.assertEqual(selector.last_debug.selection_reason, "no_valid_candidate")


if __name__ == "__main__":
    unittest.main()
