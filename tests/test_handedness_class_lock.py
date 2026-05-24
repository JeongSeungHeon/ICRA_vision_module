import unittest
from types import SimpleNamespace

from perception.object_worker import HandednessAwareObjectClassLock, correct_handedness
from system.shared_state import HandState, ObjectState, SelectedHandState


def make_object(camera_id, label, *, valid=True, confidence=0.8):
    return ObjectState(
        camera_id=int(camera_id),
        frame_id=1,
        object_detected=bool(valid),
        label=label,
        confidence=float(confidence),
        centroid_base=(0.1, 0.2, 0.3) if valid else None,
        point_count=10 if valid else 0,
        points_base=[(0.1, 0.2, 0.3)] if valid else [],
        timestamp=1.0,
        valid=bool(valid),
    )


def make_hand(camera_id, handedness):
    return HandState(
        camera_id=int(camera_id),
        frame_id=1,
        hand_detected=True,
        handedness=str(handedness),
        confidence=0.9,
        palm_center_base=(0.1, 0.2, 0.3),
        palm_normal_base=(0.0, 0.0, 1.0),
        timestamp=1.0,
        valid=True,
    )


class HandednessAwareObjectClassLockTests(unittest.TestCase):
    def test_correct_handedness_flips_configured_camera(self):
        self.assertEqual(correct_handedness("Right", 0), "left")
        self.assertEqual(correct_handedness("left", 0), "right")
        self.assertEqual(correct_handedness("Right", 1), "left")
        self.assertEqual(correct_handedness("left", 1), "right")

    def test_locks_from_corrected_handedness_reference_camera(self):
        locker = HandednessAwareObjectClassLock()
        selected_hand = SelectedHandState(
            selected_camera=0,
            handedness="right",
            palm_center_base=(0.1, 0.2, 0.3),
            palm_normal_base=(0.0, 0.0, 1.0),
            valid=True,
        )
        object_cam0 = make_object(0, "wine glass", confidence=0.65)
        object_cam1 = make_object(1, "cup", confidence=0.82)
        worker_cam0 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(0, label))
        worker_cam1 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(1, label))

        filtered_cam0, filtered_cam1 = locker.process_states(
            selected_hand=selected_hand,
            hand_cam0=make_hand(0, "right"),
            hand_cam1=make_hand(1, "unknown"),
            object_cam0=object_cam0,
            object_cam1=object_cam1,
            object_worker_cam0=worker_cam0,
            object_worker_cam1=worker_cam1,
        )

        self.assertEqual(locker.locked_class, "wine glass")
        self.assertEqual(locker.locked_from_camera, 0)
        self.assertEqual(filtered_cam0.label, "wine glass")
        self.assertEqual(filtered_cam1.label, "wine glass")

    def test_actual_right_hand_uses_cam1_reference_after_cam0_flip(self):
        locker = HandednessAwareObjectClassLock()
        object_cam0 = make_object(0, "wine glass", confidence=0.65)
        object_cam1 = make_object(1, "cup", confidence=0.82)
        worker_cam0 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(0, label))
        worker_cam1 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(1, label))

        filtered_cam0, filtered_cam1 = locker.process_states(
            selected_hand=SelectedHandState(selected_camera=0, handedness="left", valid=True),
            hand_cam0=make_hand(0, "left"),
            hand_cam1=HandState(camera_id=1, valid=False),
            object_cam0=object_cam0,
            object_cam1=object_cam1,
            object_worker_cam0=worker_cam0,
            object_worker_cam1=worker_cam1,
        )

        self.assertEqual(locker.locked_class, "cup")
        self.assertEqual(locker.locked_from_camera, 1)
        self.assertEqual(filtered_cam0.label, "cup")
        self.assertEqual(filtered_cam1.label, "cup")

    def test_actual_left_hand_uses_cam0_reference_after_cam1_flip(self):
        locker = HandednessAwareObjectClassLock()
        object_cam0 = make_object(0, "wine glass", confidence=0.65)
        object_cam1 = make_object(1, "cup", confidence=0.82)
        worker_cam0 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(0, label))
        worker_cam1 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(1, label))

        filtered_cam0, filtered_cam1 = locker.process_states(
            selected_hand=SelectedHandState(selected_camera=1, handedness="right", valid=True),
            hand_cam0=HandState(camera_id=0, valid=False),
            hand_cam1=make_hand(1, "right"),
            object_cam0=object_cam0,
            object_cam1=object_cam1,
            object_worker_cam0=worker_cam0,
            object_worker_cam1=worker_cam1,
        )

        self.assertEqual(locker.locked_class, "wine glass")
        self.assertEqual(locker.locked_from_camera, 0)
        self.assertEqual(filtered_cam0.label, "wine glass")
        self.assertEqual(filtered_cam1.label, "wine glass")

    def test_can_lock_from_corrected_camera_hand_when_selected_hand_is_invalid(self):
        locker = HandednessAwareObjectClassLock()
        object_cam0 = make_object(0, "wine glass", confidence=0.65)
        object_cam1 = make_object(1, "cup", confidence=0.82)
        worker_cam0 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(0, label))
        worker_cam1 = SimpleNamespace(reselect_last_frame_by_label=lambda label: make_object(1, label))

        locker.process_states(
            selected_hand=SelectedHandState(selected_camera=None, handedness=None, valid=False),
            hand_cam0=make_hand(0, "right"),
            hand_cam1=HandState(camera_id=1, valid=False),
            object_cam0=object_cam0,
            object_cam1=object_cam1,
            object_worker_cam0=worker_cam0,
            object_worker_cam1=worker_cam1,
        )

        self.assertEqual(locker.locked_class, "wine glass")
        self.assertEqual(locker.locked_from_camera, 0)

    def test_does_not_switch_locked_class_when_one_camera_drops(self):
        locker = HandednessAwareObjectClassLock()
        locker.locked_class = "wine glass"
        object_cam0 = make_object(0, "wine glass", valid=True)
        object_cam1 = make_object(1, "wine glass", valid=False)

        filtered_cam0, filtered_cam1 = locker.process_states(
            selected_hand=SelectedHandState(selected_camera=0, handedness="right", valid=True),
            hand_cam0=make_hand(0, "right"),
            hand_cam1=make_hand(1, "unknown"),
            object_cam0=object_cam0,
            object_cam1=object_cam1,
            object_worker_cam0=SimpleNamespace(reselect_last_frame_by_label=lambda label: object_cam0),
            object_worker_cam1=SimpleNamespace(reselect_last_frame_by_label=lambda label: object_cam1),
        )

        self.assertEqual(locker.locked_class, "wine glass")
        self.assertTrue(filtered_cam0.valid)
        self.assertFalse(filtered_cam1.valid)


if __name__ == "__main__":
    unittest.main()
