import contextlib
import importlib
import io
import sys
import threading
import types
import unittest
import unittest.mock
from types import SimpleNamespace

import numpy as np


def _stub_module(name: str, **attrs):
    original = sys.modules.get(name)
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return name, original


class _Dummy:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    @classmethod
    def from_config(cls, *args, **kwargs):
        del args, kwargs
        return cls()


_STUBS = [
    _stub_module("cv2"),
    _stub_module("yaml", safe_load=lambda *args, **kwargs: {}),
    _stub_module("calibration.extrinsics", load_transform_chain=lambda *args, **kwargs: None),
    _stub_module(
        "object_pt_extraction.segmentation_engine",
        SegmentationEngine=_Dummy,
        parse_prompt_classes=lambda *args, **kwargs: [],
    ),
    _stub_module(
        "perception.fdct_depth_completion",
        bilateral_filter_depth=lambda depth, **kwargs: depth,
        FDCTDepthCompleter=_Dummy,
        FDCTDepthCompletionConfig=_Dummy,
        format_depth_completion_stats=lambda *args, **kwargs: "",
        resolve_checkpoint=lambda value: value,
    ),
    _stub_module("perception.fusion", PerceptionFusion=_Dummy),
    _stub_module("perception.fill_level_estimator", FillLevelEstimator=_Dummy),
    _stub_module("perception.grasp_target", GraspTargetPlanner=_Dummy),
    _stub_module("perception.hand_relative_fallback", HandRelativeFallbackTracker=_Dummy),
    _stub_module("perception.hand_selector", HandSelector=_Dummy),
    _stub_module("perception.hand_worker", HandWorkerCam0=_Dummy, HandWorkerCam1=_Dummy),
    _stub_module("perception.object_merger", ObjectMerger=_Dummy),
    _stub_module("perception.object_worker", ObjectWorkerCam0=_Dummy, ObjectWorkerCam1=_Dummy),
    _stub_module("perception.shape_fitting_tracker_v2", ShapeFittingTracker=_Dummy),
    _stub_module("perception.target_predictor", TargetPredictor=_Dummy),
    _stub_module("robot.rtde_controller", RtdeController=_Dummy),
    _stub_module("system.dual_sensor_hub", DualSensorHub=_Dummy),
    _stub_module(
        "system.shared_state",
        GRIPPER_CLOSE="close",
        GRIPPER_HOLD="hold",
        GRIPPER_OPEN="open",
        ROBOT_CMD_HOLD="hold",
        ROBOT_CMD_MOVE_TO_POSITION="move_to_position",
        ROBOT_CMD_SERVO_TO_POSITION="servo_to_position",
        ROBOT_CMD_STOP="stop",
        RobotCommandState=lambda **kwargs: SimpleNamespace(**kwargs),
    ),
    _stub_module("video_record", HandoverVideoRecorderService=_Dummy),
    _stub_module("utils.debug_3d_recorder", Debug3DRecorder=_Dummy),
    _stub_module("utils.handover_metadata", HandoverMetadataRecorder=_Dummy),
    _stub_module("utils.runtime_profiler", RuntimeProfiler=_Dummy),
]

_MODULE = importlib.import_module("robot_control_rtde_fitting_final")
configure_gripper_position_threshold_from_geometry = _MODULE.configure_gripper_position_threshold_from_geometry
compute_pre_release_descend_target_mm = _MODULE.compute_pre_release_descend_target_mm
execute_gripper_close = _MODULE.execute_gripper_close
execute_return_and_place = _MODULE.execute_return_and_place
execute_tactile_release_descent = _MODULE.execute_tactile_release_descent
estimate_gripper_template_width_cm = _MODULE.estimate_gripper_template_width_cm
AnySkinTactileManager = _MODULE.AnySkinTactileManager
HOME_PLACE_MIN_Z_MM = _MODULE.HOME_PLACE_MIN_Z_MM
PRE_RELEASE_MIN_Z_EPSILON_MM = _MODULE.PRE_RELEASE_MIN_Z_EPSILON_MM
RELEASE_PARAMETER_MM = _MODULE.RELEASE_PARAMETER_MM
apply_config_defaults = _MODULE.apply_config_defaults
wait_for_tactile_release_trigger = _MODULE.wait_for_tactile_release_trigger
resolve_gripper_position_stall_detection_config = _MODULE.resolve_gripper_position_stall_detection_config
resolve_gripper_position_threshold_from_geometry = _MODULE.resolve_gripper_position_threshold_from_geometry
reset_gripper_position_threshold_to_config_default = _MODULE.reset_gripper_position_threshold_to_config_default
reset_tactile_state_for_system_reset = _MODULE.reset_tactile_state_for_system_reset
sys.modules.pop("robot_control_rtde_fitting_final", None)

for _name, _original in _STUBS:
    if _original is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original


def circle_points(radius_m=0.025, z_m=0.5, count=64):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float64)
    return np.column_stack(
        (
            radius_m * np.cos(angles),
            radius_m * np.sin(angles),
            np.full((count,), z_m, dtype=np.float64),
        )
    )


def rectangle_points(width_m=0.03, length_m=0.05, z_m=0.5, repeats=8):
    corners = np.asarray(
        [
            [-0.5 * length_m, -0.5 * width_m, z_m],
            [-0.5 * length_m, 0.5 * width_m, z_m],
            [0.5 * length_m, -0.5 * width_m, z_m],
            [0.5 * length_m, 0.5 * width_m, z_m],
        ],
        dtype=np.float32,
    )
    return np.repeat(corners, repeats, axis=0)


def local_box_points(x_width_m=0.03, y_length_m=0.05, z_m=0.5, repeats=8):
    corners = np.asarray(
        [
            [-0.5 * x_width_m, -0.5 * y_length_m, z_m],
            [-0.5 * x_width_m, 0.5 * y_length_m, z_m],
            [0.5 * x_width_m, -0.5 * y_length_m, z_m],
            [0.5 * x_width_m, 0.5 * y_length_m, z_m],
        ],
        dtype=np.float64,
    )
    return np.repeat(corners, repeats, axis=0)


def rotate_z(points, angle_deg):
    angle_rad = np.deg2rad(float(angle_deg))
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotation = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(points, dtype=np.float64) @ rotation.T, np.eye(3, dtype=np.float64) @ rotation.T


def release_args(*, enabled=False, descend_mm=RELEASE_PARAMETER_MM, workspace_z=(0.0, 800.0)):
    return SimpleNamespace(
        pre_release_descend_before_open=enabled,
        pre_release_descend_mm=descend_mm,
        workspace_x=[-1000.0, 1000.0],
        workspace_y=[-1000.0, 1000.0],
        workspace_z=list(workspace_z),
    )


class DummyController:
    def __init__(self, gripper_cfg=None):
        self.config = {
            "robot": {
                "gripper": gripper_cfg or {
                    "position_complete_threshold": 40,
                    "position_complete_threshold_by_label": {
                        "cup": 75,
                    },
                    "position_threshold_geometry": {
                        "enabled": True,
                        "alpha_cm": 0.1,
                        "formula_opening_cm": 9.0,
                        "slice_band_m": 0.005,
                        "min_slice_points": 8,
                        "fallback_nearest_points": 64,
                        "local_width_axis_by_label": {
                            "box-shaped snack": "x",
                        },
                        "anisotropic_ratio": 1.25,
                        "clamp_min": 0,
                        "clamp_max": 255,
                    },
                }
            }
        }
        self.gripper_position_complete_threshold = None


class DummyCloseController:
    def __init__(self, positions, *, threshold=40, stall_cfg=None):
        gripper_cfg = {
            "position_complete_threshold": int(threshold),
            "position_stall_detection": {
                "enabled": True,
                "stable_reads_required": 3,
                "tolerance": 1,
                "min_elapsed_s": 0.0,
            },
        }
        if stall_cfg is not None:
            gripper_cfg["position_stall_detection"].update(stall_cfg)
        self.config = {"robot": {"gripper": gripper_cfg}}
        self.gripper_position_complete_threshold = int(threshold)
        self.min_tcp_force_norm_n = 999.0
        self._positions = list(positions)
        self.stop_count = 0
        self.close_started = False

    def start_gripper_close(self):
        self.close_started = True
        return True

    def read_robot_state(self, *, now_timestamp=None):
        del now_timestamp
        return SimpleNamespace(
            tcp_force_norm_n=0.0,
            mean_joint_current_a=0.0,
            grasp_verified_force_current=False,
        )

    def get_gripper_close_state(self):
        if not self._positions:
            return None
        return {"position": self._positions.pop(0)}

    def stop_gripper_motion(self):
        self.stop_count += 1


class DummyTactile:
    enabled = True

    def __init__(self, norms):
        self._norms = list(norms)
        self.num_mags = 5
        self.latest = np.ones((15,), dtype=np.float32)
        self.latest_norm = 0.0
        self.release_reference_norm = None
        self.release_delta_norm = None
        self.release_status = "off"
        self.last_error = None
        self.reset_count = 0

    def total_norm(self):
        if self._norms:
            self.latest_norm = float(self._norms.pop(0))
        return self.latest_norm

    def set_release_reference(self, reference_norm):
        self.release_reference_norm = None if reference_norm is None else float(reference_norm)
        self.release_status = "armed"

    def update_release_delta(self, current_norm):
        self.release_delta_norm = abs(float(current_norm) - float(self.release_reference_norm))
        return self.release_delta_norm

    def reset_baseline(self):
        self.reset_count += 1
        self.latest = np.zeros((15,), dtype=np.float32)
        self.latest_norm = 0.0
        return True


class DummySharedStateForPlace:
    def __init__(self):
        self.lock = threading.Lock()
        self.home_object_xyz_mm = np.asarray([100.0, 100.0, 100.0], dtype=np.float32)
        self.grasp_offset_xyz_mm = None
        self.frozen_place_z_mm = None
        self.frozen_place_z_raw_mm = None
        self.frozen_place_z_grasp_median_mm = None
        self.frozen_place_z_template_bottom_median_mm = None
        self.task_state = None

    def get_snapshot(self):
        return {
            "fixed_orientation_base": (0.0, 0.0, 0.0),
            "initial_pose_base": None,
        }

    def set_task_state(self, task_state, *, reset_prediction=False, reset_arm=False):
        del reset_prediction, reset_arm
        self.task_state = task_state


def config_args():
    return SimpleNamespace(
        robot_ip=None,
        control_hz=None,
        follow_z=None,
        pre_release_descend_before_open=None,
        pre_release_descend_m=None,
        workspace_x=None,
        workspace_y=None,
        workspace_z=None,
        fdct_depth_enabled=None,
        fdct_cameras=None,
        fdct_checkpoint=None,
        fdct_device=None,
        fdct_debug_stats=False,
    )


def place_args(*, tactile_enabled=False, pre_release_enabled=True, pre_release_descend_mm=20.0):
    return SimpleNamespace(
        tactile_enabled=tactile_enabled,
        tactile_release_ref_delay_s=0.0,
        tactile_release_delta_threshold=5.0,
        tactile_release_timeout_s=0.0,
        tactile_release_descent_min_z_mm=45.0,
        tactile_release_descent_step_mm=2.0,
        tactile_release_descent_poll_dt_s=0.0,
        tactile_auto_baseline_reset_after_open_s=0.0,
        pre_release_descend_before_open=pre_release_enabled,
        pre_release_descend_mm=pre_release_descend_mm,
        post_release_z_offset_mm=0.0,
        workspace_x=[-1000.0, 1000.0],
        workspace_y=[-1000.0, 1000.0],
        workspace_z=[0.0, 800.0],
        move_timeout_s=1.0,
        position_tolerance_m=0.0,
        gripper_release_dwell_s=0.0,
    )


class GripperThresholdConfigTests(unittest.TestCase):
    def test_pre_release_descend_disabled_keeps_place_z(self):
        target, debug = compute_pre_release_descend_target_mm(10.0, 20.0, 100.0, release_args())

        self.assertEqual(target, (10.0, 20.0, 100.0))
        self.assertFalse(debug["enabled"])

    def test_pre_release_descend_enabled_subtracts_configured_distance(self):
        descend_mm = 10.0
        target, debug = compute_pre_release_descend_target_mm(
            10.0,
            20.0,
            100.0,
            release_args(enabled=True, descend_mm=descend_mm),
        )

        self.assertEqual(target, (10.0, 20.0, 100.0 - descend_mm))
        self.assertTrue(debug["enabled"])
        self.assertFalse(debug["home_guard_applied"])

    def test_pre_release_descend_clamps_strictly_above_home_place_min_z(self):
        target, debug = compute_pre_release_descend_target_mm(
            10.0,
            20.0,
            HOME_PLACE_MIN_Z_MM + 5.0,
            release_args(enabled=True, descend_mm=RELEASE_PARAMETER_MM),
        )

        self.assertGreater(target[2], HOME_PLACE_MIN_Z_MM)
        self.assertAlmostEqual(target[2], HOME_PLACE_MIN_Z_MM + PRE_RELEASE_MIN_Z_EPSILON_MM)
        self.assertTrue(debug["home_guard_applied"])

    def test_pre_release_descend_workspace_lower_cannot_drop_below_home_guard(self):
        target, debug = compute_pre_release_descend_target_mm(
            10.0,
            20.0,
            HOME_PLACE_MIN_Z_MM + 2.0,
            release_args(enabled=True, descend_mm=RELEASE_PARAMETER_MM, workspace_z=(0.0, 800.0)),
        )

        self.assertGreater(target[2], HOME_PLACE_MIN_Z_MM)
        self.assertTrue(debug["home_guard_applied"])

    def test_formula_uses_template_diameter_and_rounds_half_up(self):
        controller = DummyController()
        points = circle_points(radius_m=0.025)

        threshold, debug = resolve_gripper_position_threshold_from_geometry(
            controller.config["robot"]["gripper"],
            points,
            (0.0, 0.0, 0.5),
        )

        self.assertAlmostEqual(debug["width_cm"], 5.0, places=5)
        self.assertAlmostEqual(debug["threshold_float"], 110.5, places=5)
        self.assertEqual(threshold, 111)

    def test_circular_slice_uses_max_xy_diameter(self):
        controller = DummyController()
        width_cm, debug = estimate_gripper_template_width_cm(
            circle_points(radius_m=0.025),
            (0.0, 0.0, 0.5),
            controller.config["robot"]["gripper"],
        )

        self.assertEqual(debug["width_mode"], "diameter")
        self.assertAlmostEqual(width_cm, 5.0, places=5)

    def test_box_uses_template_local_x_axis_even_when_rotated(self):
        controller = DummyController()
        rotated_points, axes_base = rotate_z(local_box_points(x_width_m=0.03, y_length_m=0.05), 35.0)

        width_cm, debug = estimate_gripper_template_width_cm(
            rotated_points,
            (0.0, 0.0, 0.5),
            controller.config["robot"]["gripper"],
            object_label="box-shaped snack",
            template_axes_base=axes_base,
        )

        self.assertEqual(debug["width_mode"], "local_axis_x")
        self.assertAlmostEqual(width_cm, 3.0, places=5)

    def test_box_without_template_axes_falls_back_to_default_threshold(self):
        controller = DummyController()

        threshold, debug = resolve_gripper_position_threshold_from_geometry(
            controller.config["robot"]["gripper"],
            local_box_points(x_width_m=0.03, y_length_m=0.05),
            (0.0, 0.0, 0.5),
            object_label="box-shaped snack",
            template_axes_base=None,
        )

        self.assertEqual(debug["reason"], "missing_template_axes")
        self.assertEqual(debug["width_mode"], "local_axis_x")
        self.assertEqual(threshold, 40)

    def test_rectangular_slice_uses_pca_short_axis_not_diagonal(self):
        controller = DummyController()
        width_cm, debug = estimate_gripper_template_width_cm(
            rectangle_points(width_m=0.03, length_m=0.05),
            (0.0, 0.0, 0.5),
            controller.config["robot"]["gripper"],
        )

        self.assertEqual(debug["width_mode"], "pca_short_axis")
        self.assertAlmostEqual(width_cm, 3.0, places=5)

    def test_falls_back_to_nearest_z_points_when_band_is_sparse(self):
        controller = DummyController()
        points = np.vstack(
            (
                rectangle_points(width_m=0.03, length_m=0.05, z_m=0.49, repeats=4),
                rectangle_points(width_m=0.03, length_m=0.05, z_m=0.51, repeats=4),
            )
        )

        width_cm, debug = estimate_gripper_template_width_cm(
            points,
            (0.0, 0.0, 0.5),
            controller.config["robot"]["gripper"],
        )

        self.assertEqual(debug["slice_source"], "nearest_z")
        self.assertAlmostEqual(width_cm, 3.0, places=5)

    def test_clamps_threshold_to_configured_range(self):
        controller = DummyController()
        gripper_cfg = controller.config["robot"]["gripper"]

        low_threshold, low_debug = resolve_gripper_position_threshold_from_geometry(
            gripper_cfg,
            circle_points(radius_m=0.10),
            (0.0, 0.0, 0.5),
        )
        self.assertLess(low_debug["threshold_float"], 0.0)
        self.assertEqual(low_threshold, 0)

        high_cfg = dict(gripper_cfg)
        high_cfg["position_threshold_geometry"] = dict(gripper_cfg["position_threshold_geometry"])
        high_cfg["position_threshold_geometry"]["alpha_cm"] = -1.0
        high_threshold, high_debug = resolve_gripper_position_threshold_from_geometry(
            high_cfg,
            np.zeros((8, 3), dtype=np.float32),
            (0.0, 0.0, 0.0),
        )
        self.assertGreater(high_debug["threshold_float"], 255.0)
        self.assertEqual(high_threshold, 255)

    def test_missing_geometry_falls_back_to_default_not_label_map(self):
        controller = DummyController()

        with contextlib.redirect_stdout(io.StringIO()):
            resolved = configure_gripper_position_threshold_from_geometry(
                controller,
                fitted_points_base=None,
                grasp_point_base=None,
            )

        self.assertEqual(resolved, 40)
        self.assertEqual(controller.gripper_position_complete_threshold, 40)

    def test_tactile_static_threshold_reset_ignores_template_geometry(self):
        controller = DummyController()
        controller.gripper_position_complete_threshold = 111

        with contextlib.redirect_stdout(io.StringIO()):
            resolved = reset_gripper_position_threshold_to_config_default(controller)

        self.assertEqual(resolved, 40)
        self.assertEqual(controller.gripper_position_complete_threshold, 40)


class GripperCloseStallFallbackTests(unittest.TestCase):
    def test_stall_fallback_accepts_three_stable_position_reads_below_threshold(self):
        controller = DummyCloseController([10, 20, 25, 25, 25], threshold=40)

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(controller, timeout_s=0.1, poll_dt=0.0, verbose=False)

        self.assertTrue(result)
        self.assertTrue(controller.close_started)
        self.assertEqual(controller.stop_count, 1)

    def test_stall_fallback_accepts_small_position_jitter_within_tolerance(self):
        controller = DummyCloseController([10, 20, 25, 26, 25], threshold=40)

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(controller, timeout_s=0.1, poll_dt=0.0, verbose=False)

        self.assertTrue(result)
        self.assertEqual(controller.stop_count, 1)

    def test_still_moving_position_sequence_does_not_trigger_stall(self):
        controller = DummyCloseController([10, 20, 25, 30, 35], threshold=40)

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(controller, timeout_s=0.001, poll_dt=0.0, verbose=False)

        self.assertFalse(result)
        self.assertEqual(controller.stop_count, 1)

    def test_stall_disabled_preserves_timeout_behavior(self):
        controller = DummyCloseController(
            [10, 20, 25, 25, 25],
            threshold=40,
            stall_cfg={"enabled": False},
        )

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(controller, timeout_s=0.001, poll_dt=0.0, verbose=False)

        self.assertFalse(result)
        self.assertEqual(controller.stop_count, 1)

    def test_threshold_reached_still_succeeds_without_waiting_for_stall(self):
        controller = DummyCloseController([10, 40], threshold=40)

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(controller, timeout_s=0.1, poll_dt=0.0, verbose=False)

        self.assertTrue(result)
        self.assertEqual(controller.stop_count, 1)
        self.assertEqual(controller._positions, [])

    def test_stall_config_defaults_are_resolved_from_controller_config(self):
        controller = DummyCloseController([], stall_cfg={"stable_reads_required": 0, "tolerance": -5})

        resolved = resolve_gripper_position_stall_detection_config(controller)

        self.assertTrue(resolved["enabled"])
        self.assertEqual(resolved["stable_reads_required"], 1)
        self.assertEqual(resolved["tolerance"], 0)
        self.assertEqual(resolved["min_elapsed_s"], 0.0)


class TactileConfigAndBehaviorTests(unittest.TestCase):
    def test_tactile_config_defaults_are_read_from_yaml(self):
        args = config_args()
        config = {
            "robot": {
                "tactile": {
                    "enabled": True,
                    "port": "/dev/ttyUSB9",
                    "num_mags": 7,
                    "baseline_samples": 3,
                    "startup_delay_s": 0.2,
                    "contact_norm_threshold": 12.5,
                    "extra_grasp_pos": 6,
                    "release_delta_threshold": 4.5,
                    "release_ref_delay_s": 0.1,
                    "release_timeout_s": 0.7,
                    "release_descent_min_z_mm": 45.0,
                    "release_descent_step_mm": 3.0,
                    "release_descent_poll_dt_s": 0.02,
                    "auto_baseline_reset_after_open_s": 0.0,
                    "debug": False,
                }
            }
        }

        resolved = apply_config_defaults(args, config)

        self.assertTrue(resolved.tactile_enabled)
        self.assertEqual(resolved.tactile_port, "/dev/ttyUSB9")
        self.assertEqual(resolved.tactile_num_mags, 7)
        self.assertEqual(resolved.tactile_baseline_samples, 3)
        self.assertAlmostEqual(resolved.tactile_contact_norm_threshold, 12.5)
        self.assertEqual(resolved.tactile_extra_grasp_pos, 6)
        self.assertAlmostEqual(resolved.tactile_release_delta_threshold, 4.5)
        self.assertAlmostEqual(resolved.tactile_release_descent_min_z_mm, 45.0)
        self.assertAlmostEqual(resolved.tactile_release_descent_step_mm, 3.0)
        self.assertAlmostEqual(resolved.tactile_release_descent_poll_dt_s, 0.02)
        self.assertFalse(resolved.tactile_debug)

    def test_system_reset_clears_tactile_reference_delta_and_latest(self):
        tactile = DummyTactile([10.0])
        tactile.latest = np.ones((15,), dtype=np.float32) * 3.0
        tactile.latest_norm = 123.0
        tactile.release_reference_norm = 99.0
        tactile.release_delta_norm = 55.0
        tactile.release_status = "release_triggered"
        tactile.last_error = "old error"

        with contextlib.redirect_stdout(io.StringIO()):
            baseline_reset = reset_tactile_state_for_system_reset(tactile)

        self.assertTrue(baseline_reset)
        self.assertEqual(tactile.reset_count, 1)
        self.assertIsNone(tactile.release_reference_norm)
        self.assertIsNone(tactile.release_delta_norm)
        self.assertEqual(tactile.release_status, "reset_ready")
        self.assertEqual(tactile.latest_norm, 0.0)
        self.assertIsNone(tactile.last_error)
        np.testing.assert_allclose(tactile.latest, np.zeros((15,), dtype=np.float32))

    def test_tactile_disabled_preserves_existing_close_stall_fallback(self):
        controller = DummyCloseController([10, 20, 25, 25, 25], threshold=40)

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(
                controller,
                timeout_s=0.1,
                poll_dt=0.0,
                verbose=False,
                tactile_manager=None,
            )

        self.assertTrue(result)
        self.assertEqual(controller.stop_count, 1)

    def test_tactile_contact_waits_for_extra_close_target(self):
        controller = DummyCloseController([10, 20, 25, 30], threshold=40)
        tactile = DummyTactile([0.0, 31.0, 31.0, 31.0])

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(
                controller,
                timeout_s=0.1,
                poll_dt=0.0,
                verbose=False,
                tactile_manager=tactile,
                tactile_contact_threshold=30.0,
                tactile_extra_grasp_pos=10,
            )

        self.assertTrue(result)
        self.assertEqual(controller.stop_count, 1)
        self.assertEqual(tactile.release_status, "close_done")

    def test_tactile_contact_without_position_stops_immediately(self):
        controller = DummyCloseController([], threshold=40)
        tactile = DummyTactile([31.0])

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(
                controller,
                timeout_s=0.1,
                poll_dt=0.0,
                verbose=False,
                tactile_manager=tactile,
                tactile_contact_threshold=30.0,
                tactile_extra_grasp_pos=10,
            )

        self.assertTrue(result)
        self.assertEqual(controller.stop_count, 1)
        self.assertEqual(tactile.release_status, "close_tactile_stop")

    def test_tactile_release_triggers_on_delta(self):
        tactile = DummyTactile([10.0, 25.0])
        tactile.set_release_reference(10.0)

        with contextlib.redirect_stdout(io.StringIO()):
            result = wait_for_tactile_release_trigger(
                tactile,
                delta_threshold=5.0,
                timeout_s=0.1,
                poll_dt=0.0,
            )

        self.assertTrue(result)
        self.assertEqual(tactile.release_status, "release_triggered")
        self.assertAlmostEqual(tactile.release_delta_norm, 15.0)

    def test_tactile_release_times_out(self):
        tactile = DummyTactile([10.0, 12.0, 12.0])
        tactile.set_release_reference(10.0)

        with contextlib.redirect_stdout(io.StringIO()):
            result = wait_for_tactile_release_trigger(
                tactile,
                delta_threshold=5.0,
                timeout_s=0.0,
                poll_dt=0.0,
            )

        self.assertFalse(result)
        self.assertEqual(tactile.release_status, "release_timeout")

    def test_tactile_release_descent_triggers_and_stops(self):
        tactile = DummyTactile([10.0, 20.0])
        tactile.set_release_reference(10.0)
        targets_mm = []

        def fake_move(controller, target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
            del controller, fixed_orientation_base, timeout_s, tolerance_m, source_mode
            targets_mm.append(tuple(np.asarray(target_position_base, dtype=np.float32) * 1000.0))
            return True

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "move_robot_and_wait", side_effect=fake_move):
                with unittest.mock.patch.object(_MODULE, "safe_stop_rtde") as safe_stop:
                    result = execute_tactile_release_descent(
                        SimpleNamespace(),
                        tactile,
                        (0.0, 0.0, 0.0),
                        (100.0, 100.0, 50.0),
                        place_args(tactile_enabled=True),
                    )

        self.assertTrue(result["triggered"])
        self.assertFalse(result["reached_min_z"])
        self.assertAlmostEqual(result["release_pose_mm"][2], 48.0)
        self.assertEqual(len(targets_mm), 1)
        safe_stop.assert_called()

    def test_tactile_release_descent_stops_at_min_z_without_trigger(self):
        tactile = DummyTactile([10.0, 11.0, 11.0, 11.0, 11.0])
        tactile.set_release_reference(10.0)
        targets_mm = []

        def fake_move(controller, target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
            del controller, fixed_orientation_base, timeout_s, tolerance_m, source_mode
            targets_mm.append(tuple(np.asarray(target_position_base, dtype=np.float32) * 1000.0))
            return True

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "move_robot_and_wait", side_effect=fake_move):
                with unittest.mock.patch.object(_MODULE, "safe_stop_rtde") as safe_stop:
                    result = execute_tactile_release_descent(
                        SimpleNamespace(),
                        tactile,
                        (0.0, 0.0, 0.0),
                        (100.0, 100.0, 50.0),
                        place_args(tactile_enabled=True),
                    )

        self.assertFalse(result["triggered"])
        self.assertTrue(result["reached_min_z"])
        self.assertAlmostEqual(result["release_pose_mm"][2], 45.0)
        self.assertEqual([round(target[2], 3) for target in targets_mm], [48.0, 46.0, 45.0])
        safe_stop.assert_called()

    def test_return_place_non_tactile_uses_fixed_pre_release_move(self):
        shared_state = DummySharedStateForPlace()
        sources = []

        def fake_move(controller, target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
            del controller, target_position_base, fixed_orientation_base, timeout_s, tolerance_m
            sources.append(source_mode)
            return True

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "move_robot_and_wait", side_effect=fake_move):
                with unittest.mock.patch.object(_MODULE, "execute_gripper_open", return_value=True):
                    with unittest.mock.patch.object(_MODULE, "get_base_pose_target", return_value=(None, None)):
                        ok = execute_return_and_place(
                            SimpleNamespace(),
                            shared_state,
                            place_args(tactile_enabled=False, pre_release_enabled=True),
                        )

        self.assertTrue(ok)
        self.assertIn("pre_release_descend_before_open", sources)

    def test_return_place_tactile_ignores_fixed_pre_release_move(self):
        shared_state = DummySharedStateForPlace()
        tactile = DummyTactile([77.0])
        sources = []
        refs_during_moves = []

        def fake_move(controller, target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
            del controller, target_position_base, fixed_orientation_base, timeout_s, tolerance_m
            sources.append(source_mode)
            if source_mode in ("return_hover", "return_place"):
                refs_during_moves.append(tactile.release_reference_norm)
            return True

        def fake_descent(controller, tactile_manager, fixed_orientation_base, start_pose_mm, args):
            del controller, fixed_orientation_base, start_pose_mm, args
            self.assertIs(tactile_manager, tactile)
            self.assertEqual(tactile_manager.release_reference_norm, 77.0)
            return {
                "triggered": False,
                "timed_out": False,
                "reached_min_z": True,
                "release_pose_mm": (100.0, 97.0, 45.0),
            }

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "move_robot_and_wait", side_effect=fake_move):
                with unittest.mock.patch.object(_MODULE, "execute_tactile_release_descent", side_effect=fake_descent) as descent:
                    with unittest.mock.patch.object(_MODULE, "execute_gripper_open", return_value=True):
                        with unittest.mock.patch.object(_MODULE, "get_base_pose_target", return_value=(None, None)):
                            ok = execute_return_and_place(
                                SimpleNamespace(),
                                shared_state,
                                place_args(
                                    tactile_enabled=True,
                                    pre_release_enabled=True,
                                    pre_release_descend_mm=500.0,
                                ),
                                tactile_manager=tactile,
                            )

        self.assertTrue(ok)
        self.assertNotIn("pre_release_descend_before_open", sources)
        self.assertEqual(refs_during_moves, [None, None])
        self.assertEqual(tactile.release_reference_norm, 77.0)
        descent.assert_called_once()

    def test_return_place_tactile_move_failure_does_not_capture_release_reference(self):
        shared_state = DummySharedStateForPlace()
        tactile = DummyTactile([77.0])
        sources = []

        def fake_move(controller, target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
            del controller, target_position_base, fixed_orientation_base, timeout_s, tolerance_m
            sources.append(source_mode)
            return source_mode != "return_place"

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "move_robot_and_wait", side_effect=fake_move):
                with unittest.mock.patch.object(_MODULE, "execute_tactile_release_descent") as descent:
                    with unittest.mock.patch.object(_MODULE, "execute_gripper_open") as gripper_open:
                        with unittest.mock.patch.object(_MODULE, "get_base_pose_target", return_value=(None, None)):
                            ok = execute_return_and_place(
                                SimpleNamespace(),
                                shared_state,
                                place_args(tactile_enabled=True),
                                tactile_manager=tactile,
                            )

        self.assertFalse(ok)
        self.assertEqual(sources, ["return_hover", "return_place"])
        self.assertIsNone(tactile.release_reference_norm)
        descent.assert_not_called()
        gripper_open.assert_not_called()

    def test_anyskin_import_failure_only_fatal_when_enabled(self):
        disabled = AnySkinTactileManager(SimpleNamespace(tactile_enabled=False))
        disabled.start()
        self.assertEqual(disabled.status, "disabled")

        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "anyskin":
                raise ImportError("missing anyskin")
            return original_import(name, *args, **kwargs)

        enabled_args = SimpleNamespace(
            tactile_enabled=True,
            tactile_port="/dev/null",
            tactile_num_mags=5,
            tactile_baseline_samples=5,
            tactile_startup_delay_s=0.0,
            tactile_debug=False,
        )
        with unittest.mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError):
                AnySkinTactileManager(enabled_args).start()


if __name__ == "__main__":
    unittest.main()
