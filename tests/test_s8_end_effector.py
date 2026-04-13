import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.s8_end_effector_utils import (
    S8MotionLeg,
    S8PoseRecord,
    build_metadata_document,
    build_target_positions_local_mm,
    compute_look_at_rotation_local,
    compute_validation_summary,
    elapsed_ms_between_iso,
    isoformat_utc_from_timestamp,
    load_s8_settings,
    parse_axis_spec,
    run_s8_benchmark,
)


def make_base_config() -> dict:
    return {
        "grasp": {
            "fixed_orientation_format": "rotvec",
            "fixed_orientation_base": [0.0, 0.0, 0.0],
        },
        "home_pose": {
            "enabled": True,
            "position_m": [0.4, 0.1, 0.3],
        },
        "robot": {
            "frame_mapping": {
                "enabled": False,
                "position_signs": [1.0, 1.0, 1.0],
            }
        },
        "benchmark": {
            "s8": {
                "team_name": "TestTeam",
                "run_id": 7,
                "horizontal_reach_mm": 800.0,
                "vertical_reach_mm": 600.0,
                "repetitions": 3,
                "max_motion_time_ms": 5000,
                "target_hold_sec": 0.0,
                "output_dir": "benchmark_output",
                "submission_csv_name": "s8_submission_{team_name}.csv",
                "metadata_json_path": "benchmark_output/metadata_{team_name}.json",
                "orientation": {
                    "mode": "look_at_center",
                    "tool_forward_axis": "x+",
                    "tool_up_axis": "z+",
                    "world_up_axis_hint": "z+",
                    "roll_offset_deg": 0.0,
                },
            }
        },
    }


class FakeClock:
    def __init__(self, start_time_s: float = 1_700_000_000.0) -> None:
        self._now = float(start_time_s)

    def time(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += max(float(seconds), 0.0)

    def utc_now_iso_ms(self) -> str:
        return isoformat_utc_from_timestamp(self._now)


class FakeController:
    def __init__(self, settings, clock: FakeClock, durations_by_source: dict[str, float] | None = None) -> None:
        self.settings = settings
        self.clock = clock
        self.durations_by_source = dict(durations_by_source or {})
        self.connected = False
        self.closed = False
        self._pending_pose = None
        self._pending_complete_time = None
        self._current_pose = tuple(settings.home_frame.position_base_m) + tuple(settings.home_frame.orientation_rotvec_base)

    def connect(self):
        self.connected = True
        return self

    def move_home(self) -> bool:
        self._current_pose = tuple(self.settings.home_frame.position_base_m) + tuple(self.settings.home_frame.orientation_rotvec_base)
        self._pending_pose = None
        self._pending_complete_time = None
        return True

    def close(self) -> None:
        self.closed = True

    def step(self, robot_command, *, now_timestamp=None):
        if robot_command.command_type == "move_to_position":
            duration_s = float(self.durations_by_source.get(robot_command.source_mode, 0.2))
            self._pending_pose = tuple(robot_command.target_position_base) + tuple(robot_command.fixed_orientation_base)
            self._pending_complete_time = self.clock.time() + duration_s
        return self.read_robot_state(now_timestamp=now_timestamp)

    def read_robot_state(self, *, now_timestamp=None):
        if self._pending_pose is not None and self._pending_complete_time is not None and self.clock.time() >= self._pending_complete_time:
            self._current_pose = tuple(self._pending_pose)
            self._pending_pose = None
            self._pending_complete_time = None
        return SimpleNamespace(
            is_connected=self.connected,
            actual_tcp_pose_base=self._current_pose,
            last_error=None,
        )


def make_record(
    repetition: int,
    target_id: int,
    desired_local_mm,
    actual_local_mm,
    outbound_motion_time_ms: int,
    return_motion_time_ms: int,
):
    outbound_start_s = 1_775_800_000.0
    outbound_end_s = outbound_start_s + (outbound_motion_time_ms / 1000.0)
    return_start_s = outbound_end_s + 0.25
    return_end_s = return_start_s + (return_motion_time_ms / 1000.0)
    outbound = S8MotionLeg(
        leg_name="outbound",
        source_mode=f"rep{repetition}_target{target_id}_outbound",
        target_local_mm=list(desired_local_mm),
        target_base_m=[0.0, 0.0, 0.0],
        target_rotvec_base=[0.0, 0.0, 0.0],
        start_time=isoformat_utc_from_timestamp(outbound_start_s),
        end_time=isoformat_utc_from_timestamp(outbound_end_s),
        motion_time_ms=outbound_motion_time_ms,
        reached=True,
        within_benchmark_time=outbound_motion_time_ms <= 5000,
        actual_pose_base=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        actual_pose_local_mm=list(actual_local_mm),
        actual_quaternion_local_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    return_leg = S8MotionLeg(
        leg_name="return_to_init",
        source_mode=f"rep{repetition}_target{target_id}_return",
        target_local_mm=[0.0, 0.0, 0.0],
        target_base_m=[0.0, 0.0, 0.0],
        target_rotvec_base=[0.0, 0.0, 0.0],
        start_time=isoformat_utc_from_timestamp(return_start_s),
        end_time=isoformat_utc_from_timestamp(return_end_s),
        motion_time_ms=return_motion_time_ms,
        reached=True,
        within_benchmark_time=return_motion_time_ms <= 5000,
        actual_pose_base=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        actual_pose_local_mm=[0.0, 0.0, 0.0],
        actual_quaternion_local_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    return S8PoseRecord(
        repetition=repetition,
        target_id=target_id,
        desired_local_mm=list(desired_local_mm),
        outbound=outbound,
        return_leg=return_leg,
    )


class S8EndEffectorTests(unittest.TestCase):
    def test_build_target_positions_local_mm(self) -> None:
        targets = build_target_positions_local_mm(800.0, 600.0)
        self.assertEqual(targets[1], (-400.0, 0.0, 0.0))
        self.assertEqual(targets[2], (-400.0, 400.0, 0.0))
        self.assertEqual(targets[3], (-400.0, -400.0, 0.0))
        self.assertEqual(targets[4], (-400.0, 0.0, 150.0))
        self.assertEqual(targets[5], (-400.0, 400.0, 150.0))
        self.assertEqual(targets[6], (-400.0, -400.0, 150.0))

    def test_compute_look_at_rotation_local_aligns_forward_axis(self) -> None:
        rotation = compute_look_at_rotation_local(
            target_local_mm=(400.0, -400.0, 150.0),
            center_local_mm=(400.0, 0.0, 0.0),
            orientation_settings=load_s8_settings(make_base_config()).orientation,
        )
        forward_world = rotation @ parse_axis_spec("x+")
        desired_forward = np.array([0.0, 400.0, -150.0], dtype=np.float64)
        desired_forward /= np.linalg.norm(desired_forward)
        self.assertTrue(np.allclose(forward_world, desired_forward, atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.norm(forward_world)), 1.0, places=6)

    def test_elapsed_ms_between_iso(self) -> None:
        self.assertEqual(
            elapsed_ms_between_iso("2026-04-11T10:00:00.100Z", "2026-04-11T10:00:02.450Z"),
            2350,
        )

    def test_validation_penalizes_time_and_uses_median(self) -> None:
        settings = load_s8_settings(make_base_config())
        desired_targets = build_target_positions_local_mm(settings.horizontal_reach_mm, settings.vertical_reach_mm)
        records = []
        records.append(make_record(1, 1, desired_targets[1], (-390.0, 0.0, 0.0), 2000, 2000))
        records.append(make_record(2, 1, desired_targets[1], (-380.0, 0.0, 0.0), 2100, 2100))
        records.append(make_record(3, 1, desired_targets[1], (-360.0, 0.0, 0.0), 2200, 6000))
        for target_id in range(2, 7):
            for repetition in range(1, 4):
                records.append(make_record(repetition, target_id, desired_targets[target_id], desired_targets[target_id], 2000, 2000))
        summary = compute_validation_summary(records, settings)
        pose1 = next(item for item in summary["per_pose"] if item["target_id"] == 1)
        self.assertEqual(pose1["valid_repetition_count"], 2)
        self.assertAlmostEqual(pose1["median_error_mm"], 15.0, places=6)
        self.assertAlmostEqual(pose1["pose_score"], 0.5, places=6)
        self.assertTrue(summary["final_score"] > 0.9)

    def test_build_metadata_document_patches_team_and_reach(self) -> None:
        settings = load_s8_settings(make_base_config())
        metadata = build_metadata_document(settings, submission_timestamp="2026-04-11T10:00:00.000Z")
        self.assertEqual(metadata["team"], "TestTeam")
        self.assertEqual(metadata["submission_timestamp"], "2026-04-11T10:00:00.000Z")
        self.assertEqual(metadata["execution_policy"]["end_effector_reachability"], [800.0, 600.0])

    def test_run_s8_benchmark_completes_with_fake_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            settings = load_s8_settings(
                make_base_config(),
                repo_root=repo_root,
                move_timeout_s=2.0,
                position_tolerance_m=1.0e-6,
            )
            clock = FakeClock()
            controller = FakeController(settings, clock)
            result = run_s8_benchmark(controller, settings, clock=clock, console=lambda *_args, **_kwargs: None)

            self.assertEqual(result.status, "completed")
            self.assertTrue(result.submission_csv_path and result.submission_csv_path.exists())
            self.assertTrue(result.validation_path.exists())
            self.assertTrue(result.metadata_json_path and result.metadata_json_path.exists())
            self.assertTrue(controller.closed)

            with result.submission_csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 18)

            with result.validation_path.open("r", encoding="utf-8") as handle:
                validation = json.load(handle)
            self.assertAlmostEqual(validation["final_score"], 1.0, places=6)

            with result.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["records"]), 18)
            for entry in manifest["records"]:
                self.assertTrue(entry["outbound"]["source_mode"].startswith("s8_"))
                self.assertTrue(entry["return_leg"]["source_mode"].startswith("s8_"))
                self.assertNotIn("BASE_POSE", entry["outbound"]["source_mode"])

    def test_run_s8_benchmark_aborts_without_partial_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            settings = load_s8_settings(
                make_base_config(),
                repo_root=repo_root,
                move_timeout_s=1.0,
                position_tolerance_m=1.0e-6,
            )
            clock = FakeClock()
            controller = FakeController(
                settings,
                clock,
                durations_by_source={"s8_rep1_target3_outbound": 5.0},
            )
            result = run_s8_benchmark(controller, settings, clock=clock, console=lambda *_args, **_kwargs: None)

            self.assertEqual(result.status, "aborted")
            self.assertFalse(settings.submission_csv_path.exists())
            self.assertTrue(result.validation_path.exists())

            with result.validation_path.open("r", encoding="utf-8") as handle:
                validation = json.load(handle)
            self.assertEqual(validation["status"], "aborted")
            self.assertEqual(validation["records_present"], 2)


if __name__ == "__main__":
    unittest.main()
