import contextlib
import importlib
import io
import sys
import types
import unittest
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
estimate_gripper_template_width_cm = _MODULE.estimate_gripper_template_width_cm
HOME_PLACE_MIN_Z_MM = _MODULE.HOME_PLACE_MIN_Z_MM
PRE_RELEASE_MIN_Z_EPSILON_MM = _MODULE.PRE_RELEASE_MIN_Z_EPSILON_MM
RELEASE_PARAMETER_MM = _MODULE.RELEASE_PARAMETER_MM
resolve_gripper_position_stall_detection_config = _MODULE.resolve_gripper_position_stall_detection_config
resolve_gripper_position_threshold_from_geometry = _MODULE.resolve_gripper_position_threshold_from_geometry
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


class GripperThresholdConfigTests(unittest.TestCase):
    def test_pre_release_descend_disabled_keeps_place_z(self):
        target, debug = compute_pre_release_descend_target_mm(10.0, 20.0, 100.0, release_args())

        self.assertEqual(target, (10.0, 20.0, 100.0))
        self.assertFalse(debug["enabled"])

    def test_pre_release_descend_enabled_subtracts_configured_distance(self):
        target, debug = compute_pre_release_descend_target_mm(
            10.0,
            20.0,
            100.0,
            release_args(enabled=True, descend_mm=RELEASE_PARAMETER_MM),
        )

        self.assertEqual(target, (10.0, 20.0, 90.0))
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


if __name__ == "__main__":
    unittest.main()
