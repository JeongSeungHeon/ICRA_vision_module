import contextlib
import importlib
import io
import sys
import types
import unittest
from types import SimpleNamespace


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
configure_gripper_position_threshold_for_label = _MODULE.configure_gripper_position_threshold_for_label
sys.modules.pop("robot_control_rtde_fitting_final", None)

for _name, _original in _STUBS:
    if _original is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original


class DummyController:
    def __init__(self):
        self.config = {
            "robot": {
                "gripper": {
                    "position_complete_threshold": 40,
                    "position_complete_threshold_by_label": {
                        "cup": 75,
                        "wine glass": 120,
                        "box-shaped snack": 40,
                    },
                }
            }
        }
        self.gripper_position_complete_threshold = None


class GripperThresholdConfigTests(unittest.TestCase):
    def test_configures_threshold_from_label_map(self):
        expected = {
            "cup": 75,
            "wine glass": 120,
            "box-shaped snack": 40,
            "unknown object": 40,
        }
        for label, threshold in expected.items():
            with self.subTest(label=label):
                controller = DummyController()
                with contextlib.redirect_stdout(io.StringIO()):
                    resolved = configure_gripper_position_threshold_for_label(controller, label)
                self.assertEqual(resolved, threshold)
                self.assertEqual(controller.gripper_position_complete_threshold, threshold)


if __name__ == "__main__":
    unittest.main()
