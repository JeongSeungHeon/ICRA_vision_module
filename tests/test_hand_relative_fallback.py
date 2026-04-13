import threading
import time
import unittest
import types
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *args, **kwargs: {}))


def _stub_module(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)


class _Dummy:
    @classmethod
    def from_config(cls, *args, **kwargs):
        del args, kwargs
        return cls()


_stub_module("cv2")
_stub_module("calibration.extrinsics", load_transform_chain=lambda *args, **kwargs: None)
_stub_module(
    "object_pt_extraction.segmentation_engine",
    SegmentationEngine=_Dummy,
    parse_prompt_classes=lambda *args, **kwargs: [],
)
_stub_module(
    "perception.fdct_depth_completion",
    bilateral_filter_depth=lambda depth, **kwargs: depth,
    FDCTDepthCompleter=_Dummy,
    FDCTDepthCompletionConfig=_Dummy,
    format_depth_completion_stats=lambda *args, **kwargs: "",
    resolve_checkpoint=lambda value: value,
)
_stub_module("perception.fusion", PerceptionFusion=_Dummy)
_stub_module("perception.grasp_target", GraspTargetPlanner=_Dummy)
_stub_module("perception.hand_selector", HandSelector=_Dummy)
_stub_module("perception.hand_worker", HandWorkerCam0=_Dummy, HandWorkerCam1=_Dummy)
_stub_module("perception.object_merger", ObjectMerger=_Dummy)
_stub_module("perception.object_worker", ObjectWorkerCam0=_Dummy, ObjectWorkerCam1=_Dummy)
_stub_module("perception.shape_fitting_tracker", ShapeFittingTracker=_Dummy)
_stub_module("robot.rtde_controller", RtdeController=_Dummy)
_stub_module("system.dual_sensor_hub", DualSensorHub=_Dummy)
_stub_module("utils.handover_metadata", HandoverMetadataRecorder=_Dummy)
_stub_module("utils.realsense_stream", list_realsense_serials=lambda: [])

from perception.hand_relative_fallback import HandRelativeFallbackTracker
from robot_control_rtde_fitting_v2 import FollowSharedState, robot_control_loop
from system.shared_state import FusionState, SelectedHandState


def make_fallback_config() -> dict:
    return {
        "grasp": {
            "hand_relative_fallback": {
                "enabled": True,
                "lock_frames": 5,
                "max_dropout_sec": 1.0,
                "require_hand_approach": True,
                "require_motion_triggered": True,
            }
        }
    }


def make_selected_hand(center=(0.30, 0.10, 0.40), timestamp=1.0) -> SelectedHandState:
    return SelectedHandState(
        selected_camera=0,
        handedness="right",
        confidence=0.9,
        palm_center_base=tuple(float(v) for v in center),
        palm_normal_base=(0.0, 0.0, 1.0),
        timestamp=float(timestamp),
        valid=True,
    )


def make_fusion_state(center=(0.30, 0.10, 0.40), timestamp=1.0, approached=True) -> FusionState:
    return FusionState(
        filtered_hand_center_base=tuple(float(v) for v in center),
        hand_approach_detected=bool(approached),
        hand_approach_latched=bool(approached),
        hand_fresh=True,
        timestamp=float(timestamp),
        valid=True,
    )


def make_args(min_valid_count=1) -> SimpleNamespace:
    return SimpleNamespace(
        prediction_process_noise_mm_s2=800.0,
        prediction_measurement_noise_mm=25.0,
        prediction_max_xy_speed_mm_s=200.0,
        prediction_reinit_jump_mm=120.0,
        enable_follow=True,
        min_valid_count=int(min_valid_count),
        enable_target_prediction=True,
        target_timeout_s=0.5,
        prediction_max_horizon_s=0.25,
        follow_z=True,
        control_hz=30.0,
        verbose_robot=False,
        workspace_x=[-1000.0, 1000.0],
        workspace_y=[-1000.0, 1000.0],
        workspace_z=[0.0, 1000.0],
    )


def prime_follow_state(shared_state: FollowSharedState) -> None:
    with shared_state.lock:
        shared_state.reference_locked = True
        shared_state.reference_object_xy_mm = np.array([0.0, 0.0], dtype=np.float32)
        shared_state.motion_triggered = True
        shared_state.fixed_z_mm = 300.0
        shared_state.fixed_orientation_base = (0.0, 0.0, 0.0)


class FakeRobotState:
    def __init__(self, pose_m=(0.20, 0.10, 0.30, 0.0, 0.0, 0.0)) -> None:
        self.actual_tcp_pose_base = np.asarray(pose_m, dtype=np.float32)


class FakeController:
    def __init__(self) -> None:
        self.state = FakeRobotState()

    def read_robot_state(self, now_timestamp=None):
        del now_timestamp
        return self.state


class HandRelativeFallbackTests(unittest.TestCase):
    def test_anchor_locks_after_stable_frames(self) -> None:
        tracker = HandRelativeFallbackTracker(make_fallback_config())
        hand = make_selected_hand()
        fusion = make_fusion_state()
        measured_object = (0.40, 0.20, 0.50)
        measured_grasp = (0.42, 0.22, 0.53)

        for frame_idx in range(5):
            state = tracker.process(
                measured_object_position_base=measured_object,
                measured_grasp_position_base=measured_grasp,
                selected_hand=hand,
                fusion_state=fusion,
                motion_triggered=False,
                now_timestamp=1.0 + frame_idx * 0.01,
            )
            self.assertFalse(state.valid)

        fallback_state = tracker.process(
            measured_object_position_base=None,
            measured_grasp_position_base=None,
            selected_hand=hand,
            fusion_state=fusion,
            motion_triggered=True,
            now_timestamp=1.10,
        )
        self.assertTrue(fallback_state.valid)
        np.testing.assert_allclose(fallback_state.object_position_base, measured_object, atol=1e-6)
        np.testing.assert_allclose(fallback_state.grasp_position_base, measured_grasp, atol=1e-6)

    def test_anchor_does_not_lock_without_hand_approach(self) -> None:
        tracker = HandRelativeFallbackTracker(make_fallback_config())
        hand = make_selected_hand()
        fusion = make_fusion_state(approached=False)

        for frame_idx in range(5):
            tracker.process(
                measured_object_position_base=(0.40, 0.20, 0.50),
                measured_grasp_position_base=(0.42, 0.22, 0.53),
                selected_hand=hand,
                fusion_state=fusion,
                motion_triggered=False,
                now_timestamp=2.0 + frame_idx * 0.01,
            )

        fallback_state = tracker.process(
            measured_object_position_base=None,
            measured_grasp_position_base=None,
            selected_hand=hand,
            fusion_state=fusion,
            motion_triggered=True,
            now_timestamp=2.10,
        )
        self.assertFalse(fallback_state.valid)
        self.assertEqual(fallback_state.reason, "anchor_not_locked")

    def test_fallback_reconstructs_from_hand_offset(self) -> None:
        tracker = HandRelativeFallbackTracker(make_fallback_config())
        hand = make_selected_hand(center=(0.30, 0.10, 0.40), timestamp=3.0)
        fusion = make_fusion_state(center=(0.30, 0.10, 0.40), timestamp=3.0)
        measured_object = (0.40, 0.20, 0.50)
        measured_grasp = (0.42, 0.22, 0.53)

        for frame_idx in range(5):
            tracker.process(
                measured_object_position_base=measured_object,
                measured_grasp_position_base=measured_grasp,
                selected_hand=hand,
                fusion_state=fusion,
                motion_triggered=False,
                now_timestamp=3.0 + frame_idx * 0.01,
            )

        moved_hand = make_selected_hand(center=(0.35, 0.15, 0.45), timestamp=3.2)
        moved_fusion = make_fusion_state(center=(0.35, 0.15, 0.45), timestamp=3.2)
        fallback_state = tracker.process(
            measured_object_position_base=None,
            measured_grasp_position_base=None,
            selected_hand=moved_hand,
            fusion_state=moved_fusion,
            motion_triggered=True,
            now_timestamp=3.2,
        )

        self.assertTrue(fallback_state.valid)
        np.testing.assert_allclose(fallback_state.object_position_base, (0.45, 0.25, 0.55), atol=1e-6)
        np.testing.assert_allclose(fallback_state.grasp_position_base, (0.47, 0.27, 0.58), atol=1e-6)

    def test_fallback_times_out_after_max_dropout(self) -> None:
        tracker = HandRelativeFallbackTracker(make_fallback_config())
        hand = make_selected_hand(timestamp=4.0)
        fusion = make_fusion_state(timestamp=4.0)

        for frame_idx in range(5):
            tracker.process(
                measured_object_position_base=(0.40, 0.20, 0.50),
                measured_grasp_position_base=(0.42, 0.22, 0.53),
                selected_hand=hand,
                fusion_state=fusion,
                motion_triggered=False,
                now_timestamp=4.0 + frame_idx * 0.01,
            )

        fallback_state = tracker.process(
            measured_object_position_base=None,
            measured_grasp_position_base=None,
            selected_hand=make_selected_hand(timestamp=5.2),
            fusion_state=make_fusion_state(timestamp=5.2),
            motion_triggered=True,
            now_timestamp=5.2,
        )
        self.assertFalse(fallback_state.valid)
        self.assertEqual(fallback_state.reason, "dropout_timeout")

    def test_recovery_returns_control_source_to_measured(self) -> None:
        args = make_args(min_valid_count=1)
        shared_state = FollowSharedState(args)
        prime_follow_state(shared_state)

        shared_state.update_target(
            grasp_xyz_m=np.array([0.42, 0.22, 0.53], dtype=np.float32),
            object_xyz_m=np.array([0.40, 0.20, 0.50], dtype=np.float32),
            measurement_source="measured",
        )
        shared_state.update_target(
            grasp_xyz_m=np.array([0.43, 0.23, 0.54], dtype=np.float32),
            object_xyz_m=np.array([0.41, 0.21, 0.51], dtype=np.float32),
            measurement_source="hand_fallback",
        )
        fallback_snapshot = shared_state.get_snapshot()
        self.assertEqual(fallback_snapshot["target_source"], "hand_fallback")

        shared_state.update_target(
            grasp_xyz_m=np.array([0.44, 0.24, 0.55], dtype=np.float32),
            object_xyz_m=np.array([0.42, 0.22, 0.52], dtype=np.float32),
            measurement_source="measured",
        )
        measured_snapshot = shared_state.get_snapshot()
        self.assertEqual(measured_snapshot["target_source"], "measured")


class FollowSharedStateTests(unittest.TestCase):
    def test_source_priority_measured_then_hand_fallback_then_predicted(self) -> None:
        args = make_args(min_valid_count=1)
        shared_state = FollowSharedState(args)
        prime_follow_state(shared_state)

        shared_state.update_target(
            grasp_xyz_m=np.array([0.42, 0.22, 0.53], dtype=np.float32),
            object_xyz_m=np.array([0.40, 0.20, 0.50], dtype=np.float32),
            measurement_source="measured",
        )
        measured_snapshot = shared_state.get_snapshot()
        self.assertEqual(measured_snapshot["target_source"], "measured")

        shared_state.update_target(
            grasp_xyz_m=np.array([0.43, 0.23, 0.54], dtype=np.float32),
            object_xyz_m=np.array([0.41, 0.21, 0.51], dtype=np.float32),
            measurement_source="hand_fallback",
        )
        fallback_snapshot = shared_state.get_snapshot()
        self.assertEqual(fallback_snapshot["target_source"], "hand_fallback")

        shared_state.clear_target(reset_prediction=False, reset_arm=False)
        predicted_snapshot = shared_state.get_snapshot(now_perf=shared_state.last_measured_target_t + 0.05)
        self.assertEqual(predicted_snapshot["target_source"], "predicted")

    def test_plain_dropout_keeps_predictor_state(self) -> None:
        args = make_args(min_valid_count=1)
        shared_state = FollowSharedState(args)
        prime_follow_state(shared_state)

        shared_state.update_target(
            grasp_xyz_m=np.array([0.42, 0.22, 0.53], dtype=np.float32),
            object_xyz_m=np.array([0.40, 0.20, 0.50], dtype=np.float32),
            measurement_source="measured",
        )
        self.assertTrue(shared_state.prediction_armed)
        self.assertTrue(shared_state.target_predictor.has_state())

        shared_state.clear_target(reset_prediction=False, reset_arm=False)

        self.assertTrue(shared_state.prediction_armed)
        self.assertTrue(shared_state.target_predictor.has_state())
        predicted_snapshot = shared_state.get_snapshot(now_perf=shared_state.last_measured_target_t + 0.05)
        self.assertEqual(predicted_snapshot["target_source"], "predicted")

    def test_hand_fallback_is_treated_as_active_control_source(self) -> None:
        args = make_args(min_valid_count=1)
        shared_state = FollowSharedState(args)
        prime_follow_state(shared_state)
        controller = FakeController()
        command_calls = []

        shared_state.update_target(
            grasp_xyz_m=np.array([0.43, 0.23, 0.54], dtype=np.float32),
            object_xyz_m=np.array([0.41, 0.21, 0.51], dtype=np.float32),
            measurement_source="hand_fallback",
        )

        def capture_send(*call_args, **call_kwargs):
            command_calls.append((call_args, call_kwargs))

        thread = threading.Thread(target=robot_control_loop, args=(controller, shared_state, args), daemon=True)
        with patch("robot_control_rtde_fitting_v2.send_robot_command", side_effect=capture_send):
            thread.start()
            time.sleep(0.08)
            shared_state.stop_event.set()
            thread.join(timeout=1.0)

        self.assertTrue(command_calls)


if __name__ == "__main__":
    unittest.main()
