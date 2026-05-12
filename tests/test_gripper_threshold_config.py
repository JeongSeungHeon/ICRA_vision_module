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
estimate_gripper_template_width_cm = _MODULE.estimate_gripper_template_width_cm
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
                        "anisotropic_ratio": 1.25,
                        "clamp_min": 0,
                        "clamp_max": 255,
                    },
                }
            }
        }
        self.gripper_position_complete_threshold = None


class GripperThresholdConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
