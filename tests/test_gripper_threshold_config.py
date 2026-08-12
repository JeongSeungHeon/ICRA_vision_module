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
    _stub_module(
        "perception.object_worker",
        HandednessAwareObjectClassLock=_Dummy,
        ObjectWorkerCam0=_Dummy,
        ObjectWorkerCam1=_Dummy,
    ),
    _stub_module("perception.silhouette_constraint", SilhouetteObservation=_Dummy),
    _stub_module("perception.shape_fitting_tracker_v2", ShapeFittingTracker=_Dummy),
    _stub_module(
        "perception.sam3d_backend",
        bootstrap_runtime_template=lambda *args, **kwargs: None,
        build_fastsam_object_workers=lambda *args, **kwargs: None,
        build_runtime_shape_fitting_tracker=lambda *args, **kwargs: None,
        validate_hoi_detr_runtime_assets=lambda *args, **kwargs: None,
        validate_main_runtime=lambda *args, **kwargs: None,
    ),
    _stub_module(
        "perception.sam3d_live_runtime",
        Sam3DLiveRuntime=_Dummy,
        enforce_sam3d_target_gate=lambda *args, **kwargs: False,
    ),
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
compute_place_target = _MODULE.compute_place_target
execute_gripper_close = _MODULE.execute_gripper_close
execute_return_and_place = _MODULE.execute_return_and_place
execute_tactile_release_descent = _MODULE.execute_tactile_release_descent
estimate_gripper_template_width_cm = _MODULE.estimate_gripper_template_width_cm
AnySkinTactileManager = _MODULE.AnySkinTactileManager
HOME_PLACE_MIN_Z_MM = _MODULE.HOME_PLACE_MIN_Z_MM
HOME_JOINTS_DEG = _MODULE.HOME_JOINTS_DEG
PRE_RELEASE_MIN_Z_EPSILON_MM = _MODULE.PRE_RELEASE_MIN_Z_EPSILON_MM
RELEASE_PARAMETER_MM = _MODULE.RELEASE_PARAMETER_MM
apply_config_defaults = _MODULE.apply_config_defaults
build_raw_point_cloud_geometry_state = _MODULE.build_raw_point_cloud_geometry_state
resolve_gripper_position_stall_detection_config = _MODULE.resolve_gripper_position_stall_detection_config
resolve_gripper_position_threshold_from_geometry = _MODULE.resolve_gripper_position_threshold_from_geometry
reset_gripper_position_threshold_to_config_default = _MODULE.reset_gripper_position_threshold_to_config_default
reset_tactile_state_for_system_reset = _MODULE.reset_tactile_state_for_system_reset


class AblationOptionTests(unittest.TestCase):
    @staticmethod
    def _minimal_config(backend="sam3d", tactile_enabled=True):
        return {
            "perception": {"object": {"backend": backend}},
            "safety": {
                "workspace_bounds_m": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [0.0, 1.0],
                }
            },
            "robot": {"tactile": {"enabled": tactile_enabled}},
        }

    def test_cli_defaults_to_baseline_and_accepts_single_mode(self):
        with unittest.mock.patch.object(sys, "argv", ["handover"]):
            baseline = _MODULE.parse_args()
        with unittest.mock.patch.object(
            sys,
            "argv",
            ["handover", "--ablation", "shape-fitting"],
        ):
            shape = _MODULE.parse_args()

        self.assertEqual(baseline.ablation, "baseline")
        self.assertEqual(shape.ablation, "shape-fitting")

    def test_cli_can_disable_and_reenable_place_grasp_offset_xy(self):
        with unittest.mock.patch.object(
            sys,
            "argv",
            ["handover", "--disable-place-grasp-offset-xy"],
        ):
            disabled = _MODULE.parse_args()
        with unittest.mock.patch.object(
            sys,
            "argv",
            ["handover", "--apply-place-grasp-offset-xy"],
        ):
            enabled = _MODULE.parse_args()

        self.assertFalse(disabled.apply_place_grasp_offset_xy)
        self.assertTrue(enabled.apply_place_grasp_offset_xy)

    def test_place_grasp_offset_xy_defaults_on_and_cli_overrides_yaml(self):
        default_args = config_args()
        default_resolved = apply_config_defaults(default_args, self._minimal_config())
        self.assertTrue(default_resolved.apply_place_grasp_offset_xy)

        override_args = config_args()
        override_args.apply_place_grasp_offset_xy = True
        config = self._minimal_config()
        config["robot"]["return_sequence"] = {"apply_grasp_offset_xy": False}
        override_resolved = apply_config_defaults(override_args, config)
        self.assertTrue(override_resolved.apply_place_grasp_offset_xy)

    def test_tactile_ablation_overrides_yaml_and_selects_obj_only(self):
        args = config_args()
        args.ablation = "tactile-sensing"

        resolved = apply_config_defaults(args, self._minimal_config())

        self.assertFalse(resolved.tactile_enabled)
        self.assertEqual(resolved.grasp_detection_mode, "robotiq_object_only")
        self.assertEqual(resolved.release_parameter_mm, 20.0)

        shared_state = _MODULE.FollowSharedState.__new__(_MODULE.FollowSharedState)
        shared_state.args = resolved
        shared_state.lock = threading.Lock()
        shared_state.recent_grasp_z_mm_buffer = [100.0, 100.0, 100.0]
        shared_state.recent_template_bottom_z_mm_buffer = [50.0, 50.0, 50.0]
        place_z = shared_state.finalize_place_z_from_recent_samples()

        self.assertEqual(place_z["release_parameter_mm"], 20.0)
        self.assertEqual(place_z["raw_place_z_mm"], 70.0)

    def test_baseline_keeps_eighty_mm_release_parameter(self):
        args = config_args()
        args.ablation = "baseline"

        resolved = apply_config_defaults(args, self._minimal_config())

        self.assertEqual(resolved.release_parameter_mm, RELEASE_PARAMETER_MM)

    def test_shape_and_silhouette_ablation_reject_legacy_backend(self):
        for mode in ("shape-fitting", "silhouette-scaling"):
            args = config_args()
            args.ablation = mode
            args.object_backend = "legacy"
            with self.assertRaisesRegex(ValueError, "requires --object-backend sam3d"):
                apply_config_defaults(args, self._minimal_config(backend="legacy"))

    def test_raw_geometry_preserves_cloud_centroid_and_source(self):
        merged = SimpleNamespace(
            valid=True,
            label="sam3d_object",
            centroid_base=(0.1, 0.2, 0.3),
            merged_points_base=[(0.0, 0.0, 0.1), (0.2, 0.4, 0.5)],
        )

        state = build_raw_point_cloud_geometry_state(merged)

        self.assertTrue(state.valid)
        self.assertEqual(state.centroid_base, merged.centroid_base)
        self.assertEqual(state.scale_mode, "raw_point_cloud")
        self.assertIsNone(state.template_id)
        np.testing.assert_allclose(state.fitted_points_base, merged.merged_points_base)
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


class DummyRobotiqObjectController(DummyCloseController):
    def __init__(self, statuses, *, position=255):
        super().__init__([position] * max(len(statuses), 1), threshold=1)
        self._statuses = list(statuses)
        self.min_tcp_force_norm_n = -1.0

    def get_gripper_close_state(self):
        if not self._statuses:
            return None
        return {
            "position": 255,
            "closed_position": 255,
            "fully_closed": True,
            "object_status": self._statuses.pop(0),
        }


class DummyPoseController:
    def __init__(self, poses_mm):
        self._poses_mm = list(poses_mm)

    def read_robot_state(self, *, now_timestamp=None):
        del now_timestamp
        if not self._poses_mm:
            pose_mm = None
        else:
            pose_mm = self._poses_mm.pop(0)
        pose_base = None if pose_mm is None else tuple(float(v) / 1000.0 for v in (*pose_mm, 0.0, 0.0, 0.0))
        return SimpleNamespace(actual_tcp_pose_base=pose_base)


class DummyStopController:
    def __init__(self, speeds, *, steady_values=None):
        self._speeds = list(speeds)
        self._rtde_control = SimpleNamespace(isSteady=self._read_steady)
        self._steady_values = list(steady_values or [])

    def _read_steady(self):
        if not self._steady_values:
            return False
        return bool(self._steady_values.pop(0))

    def read_robot_state(self, *, now_timestamp=None):
        del now_timestamp
        if self._speeds:
            speed = self._speeds.pop(0)
        else:
            speed = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return SimpleNamespace(
            actual_tcp_speed=tuple(float(v) for v in speed),
            actual_tcp_pose_base=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )


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
        self.grasp_offset_xyz_mm = np.asarray([10.0, -5.0, 20.0], dtype=np.float32)
        self.frozen_place_z_mm = 100.0
        self.frozen_place_z_raw_mm = 100.0
        self.frozen_place_z_grasp_median_mm = 80.0
        self.frozen_place_z_template_bottom_median_mm = 60.0
        self.task_state = None

    def get_snapshot(self):
        return {
            "fixed_orientation_base": (0.0, 0.0, 0.0),
            "initial_pose_base": None,
        }

    def set_task_state(self, task_state, *, reset_prediction=False, reset_arm=False):
        del reset_prediction, reset_arm
        self.task_state = task_state


class DummyJointHomeController:
    def __init__(self):
        self.calls = []
        self.joint_positions = None
        self.stop_called = False

    def move_to_joint_positions(self, joints_rad, *, speed_rad_s, acceleration_rad_s2, async_move):
        self.calls.append(
            {
                "joints_rad": tuple(float(v) for v in joints_rad),
                "speed_rad_s": float(speed_rad_s),
                "acceleration_rad_s2": float(acceleration_rad_s2),
                "async_move": bool(async_move),
            }
        )
        self.joint_positions = self.calls[-1]["joints_rad"]
        return True

    def read_robot_state(self, *, now_timestamp=None):
        del now_timestamp
        return SimpleNamespace(joint_positions=self.joint_positions)

    def stop_joint_motion(self):
        self.stop_called = True


class DummyWorkerController:
    def __init__(self):
        self.commands = []
        self.closed = False
        self.config = {"robot": {"gripper": {"position_complete_threshold": 40}}}
        self.gripper_position_complete_threshold = 40

    def read_robot_state(self, *, now_timestamp=None):
        del now_timestamp
        return SimpleNamespace(
            is_connected=True,
            using_mock=True,
            actual_tcp_pose_base=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            last_error=None,
            tcp_force_norm_n=0.0,
            mean_joint_current_a=0.0,
            grasp_verified_force_current=False,
        )

    def step(self, command, now_timestamp=None):
        del now_timestamp
        self.commands.append(command)
        return True

    def close(self):
        self.closed = True


class DummyWorkerSharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.follow_idle = True
        self.follow_enabled = True
        self.follow_pause_requested = False
        self.pregrasp_ok = True
        self.reset_count = 0
        self.task_state = "FOLLOW"
        self.frozen_place_z_mm = 100.0

    def get_snapshot(self, now_perf=None):
        del now_perf
        return {
            "follow_enabled": self.follow_enabled,
            "follow_pause_requested": self.follow_pause_requested,
            "control_target_xyz_mm": np.asarray([5.0, 5.0, 0.0], dtype=np.float32),
            "target_source": "hand_fallback",
            "valid_detection_streak": 1,
            "prediction_armed": True,
            "motion_triggered": True,
            "prediction_age_s": None,
            "fixed_z_mm": 0.0,
            "fixed_orientation_base": (0.0, 0.0, 0.0),
            "latest_target_xyz_mm": np.asarray([5.0, 5.0, 0.0], dtype=np.float32),
            "latest_grasp_xyz_mm": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "latest_object_xyz_mm": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "initial_pose_base": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "task_epoch": 0,
        }

    def set_follow_thread_idle(self, is_idle):
        self.follow_idle = bool(is_idle)

    def clear_follow_pause(self):
        self.follow_pause_requested = False

    def request_follow_pause(self):
        self.follow_pause_requested = True

    def set_follow_enabled(self, follow_enabled, *, reset_prediction=True, reset_arm=True):
        del reset_prediction, reset_arm
        self.follow_enabled = bool(follow_enabled)

    def stop_follow(self):
        self.follow_enabled = False

    def set_task_state(self, task_state, *, reset_prediction=False, reset_arm=False):
        del reset_prediction, reset_arm
        self.task_state = task_state

    def reset_for_restart(self, *, follow_enabled=None):
        self.reset_count += 1
        self.follow_enabled = bool(follow_enabled)

    def set_fixed_pose_from_robot(self, controller):
        del controller

    def is_pregrasp_pose_reached(self, eef_pose_base, x_tol_mm=210.0, y_tol_mm=30.0, z_tol_mm=30.0):
        del eef_pose_base, x_tol_mm, y_tol_mm, z_tol_mm
        return self.pregrasp_ok


def worker_args():
    return SimpleNamespace(
        control_hz=30.0,
        min_valid_count=1,
        prediction_max_horizon_s=0.25,
        follow_z=False,
        verbose_robot=False,
        gripper_close_timeout_s=0.1,
        tactile_contact_norm_threshold=30.0,
        tactile_extra_grasp_pos=10,
        gripper_release_dwell_s=0.0,
        move_timeout_s=0.1,
        position_tolerance_m=0.0,
        enable_follow=True,
        workspace_x=[-1000.0, 1000.0],
        workspace_y=[-1000.0, 1000.0],
        workspace_z=[-1000.0, 1000.0],
        post_release_z_offset_mm=0.0,
        pre_release_descend_before_open=False,
        pre_release_descend_mm=0.0,
    )


def config_args():
    return SimpleNamespace(
        object_backend=None,
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


class RobotWorkerThreadStructureTests(unittest.TestCase):
    def make_worker(self):
        shared_state = DummyWorkerSharedState()
        worker = _MODULE.RobotWorker(worker_args(), shared_state)
        worker.controller = DummyWorkerController()
        return worker, shared_state, worker.controller

    def test_submit_marks_urgent_requests_cancelled(self):
        worker, _shared_state, _controller = self.make_worker()

        request_id = worker.submit(_MODULE.RobotRequest(_MODULE.ROBOT_REQ_RESET_HOME))

        self.assertEqual(request_id, 1)
        self.assertTrue(worker._cancel_event.is_set())

    def test_worker_follow_sends_servo_command_for_hand_fallback_target(self):
        worker, _shared_state, controller = self.make_worker()

        worker._handle_start_follow()
        worker._follow_once()

        command_types = [command.command_type for command in controller.commands]
        self.assertIn(_MODULE.ROBOT_CMD_SERVO_TO_POSITION, command_types)
        self.assertEqual(worker.get_status().state, _MODULE.ROBOT_STATE_FOLLOWING)

    def test_worker_grasp_place_advances_to_done(self):
        worker, _shared_state, _controller = self.make_worker()
        worker._set_status(state=_MODULE.ROBOT_STATE_FOLLOWING)
        calls = []

        def fake_gripper_close(*args, **kwargs):
            del args, kwargs
            calls.append("grasp")
            return True

        def fake_save_offset(*args, **kwargs):
            del args, kwargs
            calls.append("offset")
            return True

        def fake_return_place(*args, **kwargs):
            del args, kwargs
            calls.append("place")
            return True

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "execute_gripper_close", side_effect=fake_gripper_close):
                with unittest.mock.patch.object(_MODULE, "save_grasp_offset", side_effect=fake_save_offset):
                    with unittest.mock.patch.object(_MODULE, "execute_return_and_place", side_effect=fake_return_place):
                        worker._handle_start_grasp_place(
                            {
                                "context": _MODULE.RobotActionContext(
                                    fitted_points_base=np.zeros((4, 3), dtype=np.float32),
                                    grasp_point_base=(0.0, 0.0, 0.0),
                                    object_label="cup",
                                )
                            }
                        )

        status = worker.get_status()
        self.assertEqual(calls, ["grasp", "offset", "place"])
        self.assertEqual(status.state, _MODULE.ROBOT_STATE_DONE)
        self.assertTrue(status.grasp_ok)
        self.assertEqual(status.task_done_epoch, 1)

    def test_worker_does_not_close_gripper_without_frozen_place_z(self):
        worker, shared_state, _controller = self.make_worker()
        worker._set_status(state=_MODULE.ROBOT_STATE_FOLLOWING)
        shared_state.frozen_place_z_mm = None

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "execute_gripper_close") as close:
                worker._handle_start_grasp_place({"context": _MODULE.RobotActionContext()})

        close.assert_not_called()
        self.assertEqual(worker.get_status().state, _MODULE.ROBOT_STATE_FOLLOWING)
        self.assertEqual(
            worker.get_status().last_error,
            "grasp request ignored because place Z is not ready",
        )

    def test_worker_grasp_place_publishes_pose_from_blocking_callbacks(self):
        worker, _shared_state, _controller = self.make_worker()
        worker._set_status(state=_MODULE.ROBOT_STATE_FOLLOWING)

        def make_state(x):
            return SimpleNamespace(
                is_connected=True,
                using_mock=True,
                actual_tcp_pose_base=(float(x), 0.0, 0.0, 0.0, 0.0, 0.0),
                last_error=None,
            )

        def fake_gripper_close(*args, **kwargs):
            del args
            kwargs["on_state_read"](make_state(0.1))
            return True

        def fake_save_offset(*args, **kwargs):
            del args
            kwargs["on_state_read"](make_state(0.2))
            return True

        def fake_return_place(*args, **kwargs):
            del args
            kwargs["on_state_read"](make_state(0.3))
            return True

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "execute_gripper_close", side_effect=fake_gripper_close):
                with unittest.mock.patch.object(_MODULE, "save_grasp_offset", side_effect=fake_save_offset):
                    with unittest.mock.patch.object(_MODULE, "execute_return_and_place", side_effect=fake_return_place):
                        worker._handle_start_grasp_place({"context": _MODULE.RobotActionContext()})

        status = worker.get_status()
        self.assertEqual(status.state, _MODULE.ROBOT_STATE_DONE)
        self.assertEqual(status.last_robot_pose, (0.3, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_wait_until_target_reached_reports_each_robot_state(self):
        controller = DummyPoseController([(100.0, 0.0, 0.0)])
        states = []

        ok = _MODULE.wait_until_target_reached(
            controller,
            (0.1, 0.0, 0.0),
            timeout_s=0.1,
            tolerance_m=0.0,
            poll_dt=0.0,
            on_state_read=states.append,
        )

        self.assertTrue(ok)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].actual_tcp_pose_base, (0.1, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_wait_until_robot_stopped_accepts_low_tcp_speed(self):
        controller = DummyStopController([(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)])

        ok = _MODULE.wait_until_robot_stopped(
            controller,
            timeout_s=0.1,
            speed_threshold_mps=0.002,
            poll_dt=0.0,
        )

        self.assertTrue(ok)

    def test_wait_until_robot_stopped_accepts_rtde_steady(self):
        controller = DummyStopController(
            [(0.05, 0.0, 0.0, 0.0, 0.0, 0.0)],
            steady_values=[True],
        )

        ok = _MODULE.wait_until_robot_stopped(
            controller,
            timeout_s=0.1,
            speed_threshold_mps=0.002,
            poll_dt=0.0,
        )

        self.assertTrue(ok)

    def test_wait_until_robot_stopped_times_out_when_motion_continues(self):
        controller = DummyStopController([(0.05, 0.0, 0.0, 0.0, 0.0, 0.0)])

        ok = _MODULE.wait_until_robot_stopped(
            controller,
            timeout_s=0.0,
            speed_threshold_mps=0.002,
            poll_dt=0.0,
        )

        self.assertFalse(ok)

    def test_wait_until_robot_stopped_accepts_mock_zero_speed(self):
        controller = DummyStopController([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])

        ok = _MODULE.wait_until_robot_stopped(
            controller,
            timeout_s=0.1,
            speed_threshold_mps=0.002,
            poll_dt=0.0,
        )

        self.assertTrue(ok)

    def test_cancelled_wait_sends_stop_command(self):
        controller = DummyWorkerController()
        cancel_event = threading.Event()
        cancel_event.set()

        ok = _MODULE.wait_until_target_reached(
            controller,
            (0.0, 0.0, 0.0),
            timeout_s=0.1,
            tolerance_m=0.0,
            cancel_event=cancel_event,
        )

        self.assertFalse(ok)
        self.assertEqual(controller.commands[-1].command_type, _MODULE.ROBOT_CMD_STOP)


class GripperThresholdConfigTests(unittest.TestCase):
    def test_fixed_place_target_applies_both_xy_grasp_offsets(self):
        shared_state = DummySharedStateForPlace()

        target, debug = compute_place_target(
            shared_state,
            place_object_xy_mm=(600.0, 0.0),
        )

        np.testing.assert_allclose(target, np.asarray([590.0, 5.0, 100.0]))
        self.assertEqual(debug["fallback_reason"], "none")
        self.assertEqual(debug["place_object_x_mm"], 600.0)
        self.assertEqual(debug["place_object_y_mm"], 0.0)
        self.assertTrue(debug["grasp_offset_xy_applied"])

    def test_fixed_place_target_can_ignore_grasp_xy_offset(self):
        shared_state = DummySharedStateForPlace()

        target, debug = compute_place_target(
            shared_state,
            place_object_xy_mm=(600.0, 0.0),
            apply_grasp_offset_xy=False,
        )

        np.testing.assert_allclose(target, np.asarray([600.0, 0.0, 100.0]))
        self.assertFalse(debug["grasp_offset_xy_applied"])
        self.assertEqual(debug["applied_grasp_offset_x_mm"], 0.0)
        self.assertEqual(debug["applied_grasp_offset_y_mm"], 0.0)

    def test_fixed_place_target_without_offset_does_not_require_grasp_offset(self):
        shared_state = DummySharedStateForPlace()
        shared_state.grasp_offset_xyz_mm = None

        target, debug = compute_place_target(
            shared_state,
            apply_grasp_offset_xy=False,
        )

        np.testing.assert_allclose(target, np.asarray([600.0, 0.0, 100.0]))
        self.assertEqual(debug["fallback_reason"], "none")

    def test_fixed_place_target_fails_closed_without_place_z(self):
        shared_state = DummySharedStateForPlace()
        shared_state.frozen_place_z_mm = None

        target, debug = compute_place_target(shared_state)

        self.assertIsNone(target)
        self.assertEqual(debug["fallback_reason"], "no_frozen_place_z")

    def test_home_joint_degrees_are_converted_to_radians(self):
        expected = tuple(float(np.deg2rad(value)) for value in HOME_JOINTS_DEG)
        self.assertEqual(_MODULE.get_home_joints_rad(), expected)

    def test_move_robot_to_home_pose_sends_joint_home_target(self):
        controller = DummyJointHomeController()
        args = SimpleNamespace(move_timeout_s=0.1)

        with contextlib.redirect_stdout(io.StringIO()):
            _MODULE.move_robot_to_home_pose(controller, args)

        self.assertEqual(len(controller.calls), 1)
        self.assertEqual(controller.calls[0]["joints_rad"], _MODULE.get_home_joints_rad())
        self.assertEqual(controller.calls[0]["speed_rad_s"], _MODULE.HOME_JOINT_SPEED_RAD_S)
        self.assertEqual(controller.calls[0]["acceleration_rad_s2"], _MODULE.HOME_JOINT_ACCELERATION_RAD_S2)
        self.assertTrue(controller.calls[0]["async_move"])
        self.assertFalse(controller.stop_called)

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


class RobotiqObjectOnlyTests(unittest.TestCase):
    def _run(self, statuses, timeout_s=0.1):
        controller = DummyRobotiqObjectController(statuses)
        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_gripper_close(
                controller,
                timeout_s=timeout_s,
                poll_dt=0.0,
                verbose=False,
                detection_mode="robotiq_object_only",
            )
        return result, controller

    def test_inner_object_is_the_only_success_status(self):
        result, controller = self._run(["MOVING", "STOPPED_INNER_OBJECT"])
        self.assertTrue(result)
        self.assertEqual(controller.stop_count, 1)

    def test_stale_at_dest_before_motion_does_not_cancel_close(self):
        result, controller = self._run(
            ["AT_DEST", "MOVING", "STOPPED_INNER_OBJECT"]
        )
        self.assertTrue(result)
        self.assertEqual(controller.stop_count, 1)

    def test_full_close_does_not_succeed_from_force_position_or_stall(self):
        result, controller = self._run(["MOVING", "AT_DEST"])
        self.assertFalse(result)
        self.assertEqual(controller.stop_count, 1)

    def test_outer_object_and_missing_status_fail_closed(self):
        outer_result, _ = self._run(["STOPPED_OUTER_OBJECT"])
        missing_result, missing_controller = self._run([], timeout_s=0.001)
        self.assertFalse(outer_result)
        self.assertFalse(missing_result)
        self.assertEqual(missing_controller.stop_count, 1)


class TactileConfigAndBehaviorTests(unittest.TestCase):
    def test_tactile_config_defaults_are_read_from_yaml(self):
        args = config_args()
        config = {
            "perception": {"object": {"backend": "sam3d"}},
            "safety": {
                "workspace_bounds_m": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [0.0, 1.0],
                }
            },
            "robot": {
                "return_sequence": {
                    "place_object_xy_mm": [600.0, 0.0],
                    "apply_grasp_offset_xy": False,
                    "post_backoff_stop_check_enabled": True,
                    "post_backoff_stop_speed_threshold_mps": 0.003,
                    "post_backoff_stop_timeout_s": 0.4,
                    "post_backoff_stop_poll_dt_s": 0.02,
                    "post_backoff_stop_require_confirmed": True,
                },
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
        self.assertEqual(resolved.place_object_xy_mm, (600.0, 0.0))
        self.assertFalse(resolved.apply_place_grasp_offset_xy)
        self.assertTrue(resolved.post_backoff_stop_check_enabled)
        self.assertAlmostEqual(resolved.post_backoff_stop_speed_threshold_mps, 0.003)
        self.assertAlmostEqual(resolved.post_backoff_stop_timeout_s, 0.4)
        self.assertAlmostEqual(resolved.post_backoff_stop_poll_dt_s, 0.02)
        self.assertTrue(resolved.post_backoff_stop_require_confirmed)

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

    def test_anyskin_snapshot_includes_sample_timestamps(self):
        class FakeAnySkinStream:
            def __init__(self):
                self.calls = 0

            def get_data(self, num_samples):
                del num_samples
                self.calls += 1
                if self.calls == 1:
                    return np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
                return np.asarray([[0.0, 1.0, 2.0, 2.0]], dtype=np.float32)

        tactile = AnySkinTactileManager(
            SimpleNamespace(
                tactile_enabled=True,
                tactile_port="/dev/null",
                tactile_num_mags=1,
                tactile_baseline_samples=1,
                tactile_startup_delay_s=0.0,
                tactile_contact_norm_threshold=30.0,
                tactile_debug=False,
            )
        )
        tactile.stream = FakeAnySkinStream()
        self.assertTrue(tactile.reset_baseline())

        snapshot = tactile.snapshot(refresh=True)

        np.testing.assert_allclose(snapshot["values"], np.asarray([1.0, 2.0, 2.0], dtype=np.float32))
        self.assertAlmostEqual(float(snapshot["total_norm"]), 3.0)
        self.assertTrue(np.isfinite(float(snapshot["timestamp_perf_s"])))
        self.assertTrue(np.isfinite(float(snapshot["timestamp_unix_s"])))

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

    def test_tactile_release_descent_triggers_and_stops(self):
        tactile = DummyTactile([10.0, 20.0])
        tactile.set_release_reference(10.0)
        controller = DummyPoseController([(100.0, 100.0, 50.0), (100.0, 100.0, 49.0)])
        sent_commands = []

        def fake_send(controller, command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode):
            del controller, fixed_orientation_base, gripper_action
            sent_commands.append((command_type, tuple(np.asarray(target_position_base, dtype=np.float32) * 1000.0), source_mode))
            return None

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "move_robot_and_wait") as move_wait:
                with unittest.mock.patch.object(_MODULE, "send_robot_command", side_effect=fake_send) as send_command:
                    with unittest.mock.patch.object(_MODULE, "safe_stop_rtde") as safe_stop:
                        result = execute_tactile_release_descent(
                            controller,
                            tactile,
                            (0.0, 0.0, 0.0),
                            (100.0, 100.0, 50.0),
                            place_args(tactile_enabled=True),
                        )

        self.assertTrue(result["triggered"])
        self.assertFalse(result["reached_min_z"])
        self.assertAlmostEqual(result["release_pose_mm"][2], 49.0)
        self.assertEqual(len(sent_commands), 1)
        self.assertEqual(sent_commands[0][0], "move_to_position")
        self.assertAlmostEqual(sent_commands[0][1][2], 45.0)
        self.assertEqual(sent_commands[0][2], "tactile_release_descent")
        send_command.assert_called_once()
        move_wait.assert_not_called()
        safe_stop.assert_called()

    def test_tactile_release_descent_stops_at_min_z_without_trigger(self):
        tactile = DummyTactile([10.0, 11.0, 11.0, 11.0, 11.0])
        tactile.set_release_reference(10.0)
        controller = DummyPoseController([
            (100.0, 100.0, 50.0),
            (100.0, 100.0, 48.0),
            (100.0, 100.0, 45.0),
        ])
        sent_commands = []

        def fake_send(controller, command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode):
            del controller, fixed_orientation_base, gripper_action
            sent_commands.append((command_type, tuple(np.asarray(target_position_base, dtype=np.float32) * 1000.0), source_mode))
            return None

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "move_robot_and_wait") as move_wait:
                with unittest.mock.patch.object(_MODULE, "send_robot_command", side_effect=fake_send):
                    with unittest.mock.patch.object(_MODULE, "safe_stop_rtde") as safe_stop:
                        result = execute_tactile_release_descent(
                            controller,
                            tactile,
                            (0.0, 0.0, 0.0),
                            (100.0, 100.0, 50.0),
                            place_args(tactile_enabled=True),
                        )

        self.assertFalse(result["triggered"])
        self.assertTrue(result["reached_min_z"])
        self.assertAlmostEqual(result["release_pose_mm"][2], 45.0)
        self.assertEqual(len(sent_commands), 1)
        self.assertAlmostEqual(sent_commands[0][1][2], 45.0)
        move_wait.assert_not_called()
        safe_stop.assert_called()

    def test_tactile_release_descent_times_out_when_pose_missing(self):
        tactile = DummyTactile([10.0])
        tactile.set_release_reference(10.0)
        controller = DummyPoseController([None])

        with contextlib.redirect_stdout(io.StringIO()):
            with unittest.mock.patch.object(_MODULE, "send_robot_command") as send_command:
                with unittest.mock.patch.object(_MODULE, "safe_stop_rtde") as safe_stop:
                    result = execute_tactile_release_descent(
                        controller,
                        tactile,
                        (0.0, 0.0, 0.0),
                        (100.0, 100.0, 50.0),
                        place_args(tactile_enabled=True),
                    )

        self.assertFalse(result["triggered"])
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["reached_min_z"])
        self.assertEqual(result["release_pose_mm"], (100.0, 100.0, 50.0))
        send_command.assert_called_once()
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
                    with unittest.mock.patch.object(_MODULE, "move_robot_to_home_pose", return_value=None):
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
                        with unittest.mock.patch.object(_MODULE, "move_robot_to_home_pose", return_value=None):
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
                        with unittest.mock.patch.object(_MODULE, "move_robot_to_home_pose", return_value=None):
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
