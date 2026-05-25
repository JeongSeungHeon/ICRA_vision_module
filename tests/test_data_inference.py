import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_inference import (
    build_offline_snapshot,
    find_input_files,
    intrinsics_vector_to_dict,
    optional_bool_array_value,
    prepare_runtime_args,
    run_replay,
    select_frame_indices,
    validate_raw_recording,
)


def _raw_recording(frame_count=3):
    return {
        "frame_count": np.asarray(frame_count, dtype=np.int32),
        "cam0_color_image": np.zeros((frame_count, 2, 3, 3), dtype=np.uint8),
        "cam0_depth_image_m": np.ones((frame_count, 2, 3), dtype=np.float32),
        "cam0_intrinsics": np.tile(np.asarray([100.0, 101.0, 50.0, 51.0], dtype=np.float32), (frame_count, 1)),
        "cam0_timestamp_ms": np.asarray([10.0, 20.0, 30.0], dtype=np.float32)[:frame_count],
        "cam0_serial": np.asarray(["cam0_a", "cam0_b", "cam0_c"])[:frame_count],
        "cam1_color_image": np.full((frame_count, 2, 3, 3), 7, dtype=np.uint8),
        "cam1_depth_image_m": np.full((frame_count, 2, 3), 2.0, dtype=np.float32),
        "cam1_intrinsics": np.tile(np.asarray([200.0, 201.0, 60.0, 61.0], dtype=np.float32), (frame_count, 1)),
        "cam1_timestamp_ms": np.asarray([11.0, 21.0, 31.0], dtype=np.float32)[:frame_count],
        "cam1_serial": np.asarray(["cam1_a", "cam1_b", "cam1_c"])[:frame_count],
        "timestamp_delta_ms": np.asarray([1.0, 1.0, 1.0], dtype=np.float32)[:frame_count],
        "within_sync_tolerance": np.asarray([True, True, False])[:frame_count],
        "eef_pose_base": np.asarray(
            [
                [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.4, 0.5, 0.6, 0.0, 0.1, 0.2],
            ],
            dtype=np.float32,
        )[:frame_count],
    }


class DataInferenceTests(unittest.TestCase):
    def test_validate_raw_recording_requires_saved_images(self):
        data = _raw_recording()
        del data["cam1_depth_image_m"]

        with self.assertRaisesRegex(ValueError, "--3d-debug --save-image"):
            validate_raw_recording(data, "recording.npz")

    def test_validate_raw_recording_checks_frame_count(self):
        data = _raw_recording()
        data["cam0_depth_image_m"] = data["cam0_depth_image_m"][:2]

        with self.assertRaisesRegex(ValueError, "inconsistent frame count"):
            validate_raw_recording(data, "recording.npz")

    def test_select_frame_indices_applies_bounds_stride_and_max(self):
        self.assertEqual(
            select_frame_indices(10, start_frame=2, end_frame=9, stride=3, max_frames=2),
            [2, 5],
        )
        self.assertEqual(select_frame_indices(4, start_frame=5), [])

    def test_intrinsics_vector_to_dict(self):
        self.assertEqual(
            intrinsics_vector_to_dict(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
            {"fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0},
        )

    def test_build_offline_snapshot_reconstructs_frame_bundles(self):
        data = _raw_recording()

        snapshot = build_offline_snapshot(data, source_index=2, replay_index=0, fps=30)

        self.assertEqual(snapshot.pair_index, 2)
        self.assertEqual(snapshot.cam0.color_image.shape, (2, 3, 3))
        self.assertEqual(snapshot.cam0.intrinsics["fx"], 100.0)
        self.assertEqual(snapshot.cam1.intrinsics["cy"], 61.0)
        self.assertEqual(snapshot.cam0.timestamp_ms, 30.0)
        self.assertEqual(snapshot.cam1.serial, "cam1_c")
        self.assertFalse(snapshot.within_sync_tolerance)

    def test_find_input_files_accepts_file_and_sorts_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a_path = root / "a.npz"
            b_path = root / "b.npz"
            a_path.touch()
            b_path.touch()

            self.assertEqual(find_input_files(b_path), [b_path])
            self.assertEqual(find_input_files(root), [a_path, b_path])

    def test_optional_bool_array_value_uses_recording_or_default(self):
        data = {
            "flags": np.asarray([True, 0, 1, np.nan], dtype=object),
            "strings": np.asarray(["true", "off", "unknown"], dtype=object),
        }

        self.assertTrue(optional_bool_array_value(data, "flags", 0, default=False))
        self.assertFalse(optional_bool_array_value(data, "flags", 1, default=True))
        self.assertTrue(optional_bool_array_value(data, "flags", 2, default=False))
        self.assertTrue(optional_bool_array_value(data, "flags", 3, default=True))
        self.assertTrue(optional_bool_array_value(data, "strings", 0, default=False))
        self.assertFalse(optional_bool_array_value(data, "strings", 1, default=True))
        self.assertTrue(optional_bool_array_value(data, "strings", 2, default=True))
        self.assertFalse(optional_bool_array_value(data, "missing", 0, default=False))

    def test_prepare_runtime_args_sets_live_default_fields(self):
        args = SimpleNamespace(output_dir="output/data_inference")

        runtime_args = prepare_runtime_args(args, _raw_recording(frame_count=1))

        self.assertIsNone(runtime_args.serial)
        self.assertIsNone(runtime_args.pre_release_descend_before_open)
        self.assertIsNone(runtime_args.pre_release_descend_m)
        self.assertFalse(runtime_args.record_video)
        self.assertEqual(runtime_args.width, 3)
        self.assertEqual(runtime_args.height, 2)

    def test_run_replay_matches_live_perception_state_flow(self):
        data = _raw_recording(frame_count=2)
        data["task_epoch"] = np.asarray([1, 2], dtype=np.int32)
        data["motion_triggered"] = np.asarray([True, False], dtype=bool)
        data["home_object_locked"] = np.asarray([False, True], dtype=bool)
        calls = SimpleNamespace(
            merger_kwargs=[],
            freeze_values=[],
            motion_values=[],
            reset_count=0,
            debug_frames=0,
        )

        class Worker:
            def process_frame(self, frame, *, frame_id):
                del frame, frame_id
                return SimpleNamespace(label="cup")

            def close(self):
                pass

        class HandSelector:
            def process_states(self, hand_cam0, hand_cam1, *, object_center_base=None):
                del hand_cam0, hand_cam1, object_center_base
                return SimpleNamespace(valid=True)

        class ObjectMerger:
            def process_states(self, object_cam0, object_cam1, **kwargs):
                del object_cam0, object_cam1
                calls.merger_kwargs.append(dict(kwargs))
                return SimpleNamespace(valid=True, centroid_base=(0.1, 0.2, 0.3))

        class ShapeFittingTracker:
            def process(self, merged_object, *, silhouette_observations=None, freeze_silhouette_scale=False):
                del merged_object, silhouette_observations
                calls.freeze_values.append(bool(freeze_silhouette_scale))
                return SimpleNamespace(
                    valid=True,
                    centroid_base=(0.4, 0.5, 0.6),
                    fitted_points_base=None,
                )

        class Fusion:
            def process_states(self, fitted_merged_object, selected_hand, *, now_timestamp):
                del fitted_merged_object, selected_hand, now_timestamp
                return SimpleNamespace(
                    filtered_object_centroid_base=None,
                    hand_approach_detected=True,
                    hand_approach_latched=True,
                )

        class GraspPlanner:
            def process_states(self, fitted_merged_object, selected_hand, fusion_state):
                del fitted_merged_object, selected_hand, fusion_state
                return SimpleNamespace(valid=False, target_position_base=None)

        class Fallback:
            def reset(self):
                calls.reset_count += 1

            def process(self, **kwargs):
                calls.motion_values.append(bool(kwargs["motion_triggered"]))
                return SimpleNamespace(valid=False)

        class Recorder:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.save_images = False

            def save(self, *, prefix):
                del prefix
                return Path("output/data_inference/replayed.npz")

            def close(self):
                pass

        pipeline = {
            "object_worker_cam0": Worker(),
            "object_worker_cam1": Worker(),
            "hand_worker_cam0": Worker(),
            "hand_worker_cam1": Worker(),
            "hand_selector": HandSelector(),
            "object_merger": ObjectMerger(),
            "shape_fitting_tracker": ShapeFittingTracker(),
            "fusion": Fusion(),
            "grasp_planner": GraspPlanner(),
            "hand_relative_fallback": Fallback(),
            "sensor_hub": SimpleNamespace(stop=lambda: None),
        }
        live = SimpleNamespace(
            load_yaml_config=lambda config: {},
            apply_config_defaults=lambda runtime_args, config: runtime_args,
            build_dual_perception_pipeline=lambda runtime_args: pipeline,
            object_center_for_hand_selection=lambda object_cam0, object_cam1: None,
            build_silhouette_observations=lambda snapshot, live_pipeline: [],
            build_fitted_merged_object=lambda merged_object, shape_fitting_state: SimpleNamespace(
                centroid_base=shape_fitting_state.centroid_base
            ),
            choose_point=lambda preferred, fallback: preferred if preferred is not None else fallback,
            offset_point_base_mm=lambda point, y_mm: point,
            GRASP_POINT_Y_OFFSET_MM=0.0,
        )

        def append_debug_3d_frame(*args, **kwargs):
            del args, kwargs
            calls.debug_frames += 1

        live.append_debug_3d_frame = append_debug_3d_frame
        args = SimpleNamespace(
            output_dir="output/data_inference",
            prefix="offline_inference",
            config="configs/handover.yaml",
            start_frame=0,
            end_frame=None,
            stride=1,
            max_frames=2,
            fps=30,
            save_image=False,
            debug_3d_max_object_points=8000,
            debug_3d_max_template_points=8000,
            debug_3d_template_axes=True,
            offline_motion_triggered=False,
            offline_home_object_locked=False,
        )

        with (
            patch("data_inference.load_debug_3d_npz", return_value=data),
            patch("data_inference._load_live_module", return_value=live),
            patch("data_inference.Debug3DRecorder", Recorder),
        ):
            result = run_replay("recording.npz", args)

        self.assertEqual(result.frame_count, 2)
        self.assertEqual(calls.merger_kwargs, [{}, {}])
        self.assertEqual(calls.freeze_values, [False, True])
        self.assertEqual(calls.motion_values, [True, False])
        self.assertEqual(calls.reset_count, 2)
        self.assertEqual(calls.debug_frames, 2)


if __name__ == "__main__":
    unittest.main()
