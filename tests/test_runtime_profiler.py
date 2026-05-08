import csv
import json
import tempfile
import unittest
from pathlib import Path

from utils.runtime_profiler import RuntimeProfiler


def _record_fake_frame(profiler, frame_index, loop_ms=10.0, fps=30.0):
    profiler.start_frame(timestamp_unix_s=100.0 + frame_index, task_epoch=1)
    profiler.add_stage_ms("loop_total", loop_ms)
    profiler.add_stage_ms("object_cam0", 3.0 + frame_index)
    profiler.record_frame(
        frame_index=frame_index,
        timestamp_unix_s=100.0 + frame_index,
        task_epoch=1,
        fps=fps,
        metrics={
            "cam0_yolo_infer_ms": 4.0 + frame_index,
            "cam1_yolo_infer_ms": 5.0 + frame_index,
            "shape_fit_icp_time_ms": 1.5 + frame_index,
            "merged_object_points": 100 + frame_index,
            "object_detected": True,
        },
    )


class RuntimeProfilerTests(unittest.TestCase):
    def test_save_writes_frames_resources_and_summary_then_resets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = RuntimeProfiler(
                enabled=True,
                output_dir=Path(temp_dir),
                sample_interval_s=0.05,
                print_every_s=0.0,
                run_context={"model": "fake.pt"},
                nvidia_smi_path="/definitely/missing/nvidia-smi",
            )
            _record_fake_frame(profiler, 0, loop_ms=10.0, fps=31.0)
            _record_fake_frame(profiler, 1, loop_ms=20.0, fps=29.0)

            paths = profiler.save_current_session(reason="unit_test")

            self.assertEqual(profiler.session_index, 1)
            self.assertEqual(profiler.frames, [])
            self.assertEqual(profiler.resources, [])
            self.assertTrue(Path(paths["frames"]).is_file())
            self.assertTrue(Path(paths["resources"]).is_file())
            self.assertTrue(Path(paths["summary"]).is_file())

            with Path(paths["frames"]).open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["frame_index"], "0")
            self.assertIn("stage_loop_total_ms", rows[0])
            self.assertIn("cam0_yolo_infer_ms", rows[0])

            with Path(paths["summary"]).open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["reason"], "unit_test")
            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(summary["run_context"]["model"], "fake.pt")
            self.assertAlmostEqual(summary["stage_latency_ms"]["loop_total"]["p50"], 15.0)
            self.assertAlmostEqual(summary["frame_metrics"]["fps"]["min"], 29.0)

    def test_discard_resets_without_writing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            profiler = RuntimeProfiler(
                enabled=True,
                output_dir=output_dir,
                print_every_s=0.0,
                nvidia_smi_path="/definitely/missing/nvidia-smi",
            )
            _record_fake_frame(profiler, 0)

            profiler.discard_current_session(reason="reset_key")

            self.assertEqual(profiler.session_index, 1)
            self.assertEqual(profiler.frames, [])
            self.assertEqual(profiler.resources, [])
            self.assertEqual(list(output_dir.glob("*")), [])

    def test_missing_gpu_tools_do_not_raise(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = RuntimeProfiler(
                enabled=True,
                output_dir=Path(temp_dir),
                sample_interval_s=0.05,
                print_every_s=0.0,
                nvidia_smi_path="/definitely/missing/nvidia-smi",
            )
            _record_fake_frame(profiler, 0)
            self.assertGreaterEqual(len(profiler.resources), 1)
            resource = profiler.resources[0]
            self.assertIn("gpu_memory_used_mb", resource)
            self.assertIn("torch_cuda_available", resource)


if __name__ == "__main__":
    unittest.main()
