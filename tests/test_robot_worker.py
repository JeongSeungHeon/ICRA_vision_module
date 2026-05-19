import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from robot.robot_worker import (
    RobotActionCallbacks,
    RobotRequestType,
    RobotWorker,
    RobotWorkerState,
)


class FakeRobotState:
    def __init__(self, pose=(0.20, 0.10, 0.30, 0.0, 0.0, 0.0), connected=True):
        self.is_connected = connected
        self.actual_tcp_pose_base = tuple(float(v) for v in pose)
        self.last_error = None


class FakeController:
    def __init__(self):
        self.state = FakeRobotState()
        self.commands = []
        self.closed = False
        self.using_mock = True
        self.last_debug = SimpleNamespace(last_command_type=None)

    def read_robot_state(self, now_timestamp=None):
        del now_timestamp
        return self.state

    def close(self):
        self.closed = True


class FakeSharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.follow_enabled = False
        self.follow_pause_requested = False
        self.fixed_pose_calls = 0
        self.reset_calls = 0
        self.follow_idle = True
        self.snapshot = {
            "control_target_xyz_mm": np.array([260.0, 160.0, 320.0], dtype=np.float32),
            "target_source": "measured",
            "follow_pause_requested": False,
            "follow_enabled": True,
            "valid_detection_streak": 3,
            "prediction_armed": False,
            "motion_triggered": True,
            "prediction_age_s": None,
            "fixed_z_mm": 300.0,
            "fixed_orientation_base": (0.0, 0.0, 0.0),
            "latest_target_xyz_mm": np.array([260.0, 160.0, 320.0], dtype=np.float32),
        }

    def clear_follow_pause(self):
        self.follow_pause_requested = False
        self.snapshot["follow_pause_requested"] = False

    def request_follow_pause(self):
        self.follow_pause_requested = True
        self.snapshot["follow_pause_requested"] = True

    def stop_follow(self):
        self.follow_enabled = False
        self.snapshot["follow_enabled"] = False

    def set_follow_thread_idle(self, value):
        self.follow_idle = bool(value)

    def set_fixed_pose_from_robot(self, controller):
        del controller
        self.fixed_pose_calls += 1

    def reset_for_restart(self, follow_enabled=None):
        self.reset_calls += 1
        self.follow_enabled = bool(follow_enabled)
        self.snapshot["follow_enabled"] = bool(follow_enabled)
        self.snapshot["follow_pause_requested"] = False

    def get_snapshot(self):
        snap = dict(self.snapshot)
        for key, value in list(snap.items()):
            if isinstance(value, np.ndarray):
                snap[key] = value.copy()
        return snap


def make_args():
    return SimpleNamespace(
        control_hz=200.0,
        min_valid_count=1,
        prediction_max_horizon_s=0.25,
        follow_z=True,
        verbose_robot=False,
        workspace_x=[-1000.0, 1000.0],
        workspace_y=[-1000.0, 1000.0],
        workspace_z=[0.0, 1000.0],
        move_timeout_s=0.2,
        gripper_close_timeout_s=0.2,
        gripper_release_dwell_s=0.0,
        tactile_contact_norm_threshold=30.0,
        tactile_extra_grasp_pos=10,
        enable_follow=True,
    )


class ActionRecorder:
    def __init__(self):
        self.controller = FakeController()
        self.calls = []
        self.close_blocks_until_cancel = False

    def callbacks(self):
        return RobotActionCallbacks(
            init_rtde=self.init_rtde,
            move_robot_to_home_pose=self.move_robot_to_home_pose,
            send_robot_command=self.send_robot_command,
            safe_stop_rtde=self.safe_stop_rtde,
            disconnect_rtde=self.disconnect_rtde,
            execute_gripper_close=self.execute_gripper_close,
            execute_gripper_open=self.execute_gripper_open,
            execute_return_and_place=self.execute_return_and_place,
            save_grasp_offset=self.save_grasp_offset,
            configure_gripper_position_threshold_from_geometry=self.configure_threshold,
            reset_gripper_position_threshold_to_config_default=self.reset_threshold,
        )

    def init_rtde(self, args):
        del args
        self.calls.append("init")
        return self.controller

    def move_robot_to_home_pose(self, controller, args, cancel_event=None):
        del controller, args
        self.calls.append("home")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")

    def send_robot_command(self, controller, command_type, **kwargs):
        controller.commands.append((command_type, kwargs))
        controller.last_debug = SimpleNamespace(last_command_type=command_type)
        self.calls.append(command_type)
        return controller.state

    def safe_stop_rtde(self, controller):
        del controller
        self.calls.append("safe_stop")

    def disconnect_rtde(self, controller):
        self.calls.append("disconnect")
        controller.close()

    def execute_gripper_close(self, controller, **kwargs):
        del controller
        self.calls.append("close")
        cancel_event = kwargs.get("cancel_event")
        if self.close_blocks_until_cancel:
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    return False
                time.sleep(0.005)
            return False
        return True

    def execute_gripper_open(self, controller, **kwargs):
        del controller, kwargs
        self.calls.append("open")
        return True

    def execute_return_and_place(self, controller, shared_state, args, **kwargs):
        del controller, shared_state, args, kwargs
        self.calls.append("place")
        return True

    def save_grasp_offset(self, controller, shared_state):
        del controller, shared_state
        self.calls.append("offset")
        return True

    def configure_threshold(self, *args, **kwargs):
        del args, kwargs
        self.calls.append("threshold")

    def reset_threshold(self, *args, **kwargs):
        del args, kwargs
        self.calls.append("reset_threshold")


class RobotWorkerTests(unittest.TestCase):
    def make_worker(self):
        actions = ActionRecorder()
        shared_state = FakeSharedState()
        worker = RobotWorker(args=make_args(), shared_state=shared_state, actions=actions.callbacks())
        return worker, actions, shared_state

    def test_duplicate_grasp_request_is_rejected_while_queued(self):
        worker, actions, _ = self.make_worker()
        worker.controller = actions.controller

        self.assertTrue(worker.submit(RobotRequestType.START_GRASP_PLACE))
        self.assertFalse(worker.submit(RobotRequestType.START_GRASP_PLACE))

    def test_init_robot_runs_controller_setup_and_home_in_worker(self):
        worker, actions, shared_state = self.make_worker()
        worker.start()
        self.assertTrue(worker.submit(RobotRequestType.INIT_ROBOT))

        status = worker.wait_for_state({RobotWorkerState.IDLE}, timeout_s=1.0)
        worker.submit(RobotRequestType.SHUTDOWN)
        worker.join(timeout=1.0)

        self.assertEqual(status.state, RobotWorkerState.IDLE)
        self.assertEqual(actions.calls[:2], ["init", "home"])
        self.assertEqual(shared_state.fixed_pose_calls, 1)

    def test_following_sends_servo_command_from_shared_snapshot(self):
        worker, actions, _ = self.make_worker()
        worker.controller = actions.controller
        worker._set_status(state=RobotWorkerState.FOLLOWING)
        worker._tick_follow()

        self.assertTrue(actions.controller.commands)
        self.assertEqual(actions.controller.commands[0][0], "servo_to_position")

    def test_grasp_place_runs_close_offset_and_place_sequence(self):
        worker, actions, _ = self.make_worker()
        worker.controller = actions.controller
        worker.start()

        self.assertTrue(worker.submit(RobotRequestType.START_GRASP_PLACE))
        status = worker.wait_for_state({RobotWorkerState.DONE, RobotWorkerState.ERROR}, timeout_s=1.0)
        worker.submit(RobotRequestType.SHUTDOWN)
        worker.join(timeout=1.0)

        self.assertEqual(status.state, RobotWorkerState.DONE)
        self.assertTrue(status.grasp_ok)
        self.assertLess(actions.calls.index("close"), actions.calls.index("offset"))
        self.assertLess(actions.calls.index("offset"), actions.calls.index("place"))

    def test_reset_during_active_grasp_cancels_then_homes(self):
        worker, actions, shared_state = self.make_worker()
        actions.close_blocks_until_cancel = True
        worker.controller = actions.controller
        worker.start()

        self.assertTrue(worker.submit(RobotRequestType.START_GRASP_PLACE))
        worker.wait_for_state({RobotWorkerState.GRASPING}, timeout_s=1.0)
        self.assertTrue(worker.submit(RobotRequestType.RESET_HOME))
        status = worker.wait_for_state({RobotWorkerState.FOLLOWING, RobotWorkerState.IDLE}, timeout_s=1.0)
        worker.submit(RobotRequestType.SHUTDOWN)
        worker.join(timeout=1.0)

        self.assertTrue(status.reset_done)
        self.assertIn("home", actions.calls)
        self.assertEqual(shared_state.reset_calls, 1)

    def test_shutdown_stops_and_disconnects_controller(self):
        worker, actions, _ = self.make_worker()
        worker.controller = actions.controller
        worker.start()
        self.assertTrue(worker.submit(RobotRequestType.SHUTDOWN))
        worker.join(timeout=1.0)

        self.assertIn("safe_stop", actions.calls)
        self.assertIn("disconnect", actions.calls)
        self.assertTrue(actions.controller.closed)


if __name__ == "__main__":
    unittest.main()
