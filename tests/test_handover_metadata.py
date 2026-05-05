import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.handover_metadata import (
    METADATA_COLUMNS,
    DEFAULT_EMPTY_WINE_GLASS_MASS_G,
    HandoverMetadataRecorder,
    elapsed_ms_between_iso,
    estimate_container_and_filled_volume_ml,
    estimate_empty_container_mass_g,
    estimate_container_volume_ml_from_geometry,
    estimate_fill_level_percent,
    estimate_filled_volume_ml_from_geometry_and_fill_height,
    estimate_geometry_from_fitted_points_mm,
    estimate_mass_full_g_vision,
    estimate_wine_glass_bowl_filled_volume_ml,
    estimate_wine_glass_bowl_volume_ml,
    rotvec_to_quaternion_xyzw,
)


def make_frustum_points(top_radius_m: float, bottom_radius_m: float, height_m: float, count: int = 128) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float32)
    top_ring = np.stack(
        [
            top_radius_m * np.cos(angles),
            top_radius_m * np.sin(angles),
            np.full_like(angles, height_m),
        ],
        axis=1,
    )
    bottom_ring = np.stack(
        [
            bottom_radius_m * np.cos(angles),
            bottom_radius_m * np.sin(angles),
            np.zeros_like(angles),
        ],
        axis=1,
    )
    mid_ring = np.stack(
        [
            ((top_radius_m + bottom_radius_m) * 0.5) * np.cos(angles),
            ((top_radius_m + bottom_radius_m) * 0.5) * np.sin(angles),
            np.full_like(angles, height_m * 0.5),
        ],
        axis=1,
    )
    return np.concatenate([top_ring, mid_ring, bottom_ring], axis=0).astype(np.float32)


class HandoverMetadataTests(unittest.TestCase):
    def test_rotvec_to_quaternion_xyzw(self) -> None:
        quaternion = rotvec_to_quaternion_xyzw((0.0, 0.0, np.pi))
        self.assertAlmostEqual(quaternion[0], 0.0, places=6)
        self.assertAlmostEqual(quaternion[1], 0.0, places=6)
        self.assertAlmostEqual(abs(quaternion[2]), 1.0, places=6)
        self.assertAlmostEqual(quaternion[3], 0.0, places=6)
        self.assertAlmostEqual(np.linalg.norm(quaternion), 1.0, places=6)

    def test_estimate_geometry_from_fitted_points_mm(self) -> None:
        points = make_frustum_points(top_radius_m=0.04, bottom_radius_m=0.03, height_m=0.10)
        geometry = estimate_geometry_from_fitted_points_mm(points)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertAlmostEqual(geometry["height_est_mm_vision"], 100.0, delta=1.0)
        self.assertAlmostEqual(geometry["width_top_est_mm_vision"], 80.0, delta=2.0)
        self.assertAlmostEqual(geometry["width_bottom_est_mm_vision"], 60.0, delta=2.0)

    def test_elapsed_ms_between_iso(self) -> None:
        start_time = "2026-04-09T10:00:00.100Z"
        end_time = "2026-04-09T10:00:02.450Z"
        self.assertEqual(elapsed_ms_between_iso(start_time, end_time), 2350)

    def test_volume_and_mass_helpers(self) -> None:
        total_volume_ml = estimate_container_volume_ml_from_geometry(100.0, 100.0, 100.0)
        filled_volume_ml = estimate_filled_volume_ml_from_geometry_and_fill_height(100.0, 100.0, 100.0, 50.0)
        fill_level_percent = estimate_fill_level_percent(filled_volume_ml, total_volume_ml)
        mass_full_g = estimate_mass_full_g_vision(filled_volume_ml)

        self.assertAlmostEqual(total_volume_ml, np.pi * 250.0, places=5)
        self.assertAlmostEqual(filled_volume_ml, np.pi * 125.0, places=5)
        self.assertAlmostEqual(fill_level_percent, 50.0, places=5)
        self.assertAlmostEqual(mass_full_g, 15.0 + 0.87 * filled_volume_ml, places=5)

    def test_empty_container_mass_uses_label_and_height_rules(self) -> None:
        self.assertEqual(estimate_empty_container_mass_g("cup", 100.0), 15.0)
        self.assertEqual(estimate_empty_container_mass_g("cup", 110.0), 15.0)
        self.assertEqual(estimate_empty_container_mass_g("cup", 120.0), 9.0)
        self.assertEqual(estimate_empty_container_mass_g("cup", 135.0), 10.0)
        self.assertEqual(estimate_empty_container_mass_g("cup", None), 15.0)
        self.assertEqual(estimate_empty_container_mass_g("wine glass", 90.0), DEFAULT_EMPTY_WINE_GLASS_MASS_G)
        self.assertEqual(estimate_empty_container_mass_g("other", 120.0), 15.0)

    def test_wine_glass_bowl_volume_helpers(self) -> None:
        total_volume_ml = estimate_wine_glass_bowl_volume_ml(80.0, 130.0, 0.42)
        filled_volume_ml = estimate_wine_glass_bowl_filled_volume_ml(80.0, 130.0 * 0.42, 80.0)

        expected_total_ml = np.pi * (40.0 ** 2) * (130.0 * 0.42) / 1000.0
        expected_filled_ml = np.pi * (40.0 ** 2) * (130.0 * 0.42) / 1000.0
        self.assertAlmostEqual(total_volume_ml, expected_total_ml, places=5)
        self.assertAlmostEqual(filled_volume_ml, expected_filled_ml, places=5)

    def test_container_volume_switches_for_wine_glass(self) -> None:
        geometry = {
            "width_top_est_mm_vision": 80.0,
            "width_bottom_est_mm_vision": 20.0,
            "height_est_mm_vision": 130.0,
        }

        total_volume_ml, filled_volume_ml = estimate_container_and_filled_volume_ml(
            geometry,
            80.0,
            label="wine glass",
            bowl_height_fraction=0.42,
        )
        expected_total_ml = np.pi * (40.0 ** 2) * (130.0 * 0.42) / 1000.0
        expected_filled_ml = np.pi * (40.0 ** 2) * min(80.0, 130.0 * 0.42) / 1000.0
        self.assertAlmostEqual(total_volume_ml, expected_total_ml, places=5)
        self.assertAlmostEqual(filled_volume_ml, expected_filled_ml, places=5)

    def test_container_volume_falls_back_to_frustum_without_bowl_fraction(self) -> None:
        geometry = {
            "width_top_est_mm_vision": 80.0,
            "width_bottom_est_mm_vision": 20.0,
            "height_est_mm_vision": 130.0,
        }

        total_volume_ml, filled_volume_ml = estimate_container_and_filled_volume_ml(
            geometry,
            50.0,
            label="wine glass",
            bowl_height_fraction=None,
        )
        expected_total_ml = estimate_container_volume_ml_from_geometry(80.0, 20.0, 130.0)
        expected_filled_ml = estimate_filled_volume_ml_from_geometry_and_fill_height(80.0, 20.0, 130.0, 50.0)
        self.assertAlmostEqual(total_volume_ml, expected_total_ml, places=5)
        self.assertAlmostEqual(filled_volume_ml, expected_filled_ml, places=5)

    def test_metadata_recorder_writes_header_and_row(self) -> None:
        shared_state = SimpleNamespace(
            get_snapshot=lambda: {
                "initial_pose_base": (0.1, 0.2, 0.3, 0.0, 0.0, np.pi),
            }
        )
        shape_state = SimpleNamespace(
            valid=True,
            fitted_points_base=make_frustum_points(top_radius_m=0.04, bottom_radius_m=0.03, height_m=0.10),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "handover_metadata.csv"
            recorder = HandoverMetadataRecorder(csv_path=csv_path)
            recorder.mark_task_ready(
                shared_state,
                now_perf=100.0,
                now_timestamp_iso="2026-04-09T10:00:00.000Z",
            )
            recorder.update_geometry(
                shape_state,
                now_perf=100.2,
                now_timestamp_iso="2026-04-09T10:00:00.200Z",
            )
            recorder.update_fill_and_mass(
                SimpleNamespace(valid=True, fill_height_mm=50.0),
                now_perf=100.25,
                now_timestamp_iso="2026-04-09T10:00:00.250Z",
            )
            recorder.update_fill_and_mass(
                SimpleNamespace(valid=True, fill_height_mm=80.0),
                now_perf=100.30,
                now_timestamp_iso="2026-04-09T10:00:00.300Z",
            )
            recorder.note_robot_first_contact(
                now_perf=100.4,
                now_timestamp_iso="2026-04-09T10:00:00.400Z",
            )
            recorder.note_robot_last_contact(
                now_perf=100.8,
                now_timestamp_iso="2026-04-09T10:00:00.800Z",
            )
            recorder.note_delivery_location((120.0, 230.0, 340.0))

            written_path = recorder.record_completion()
            self.assertEqual(written_path, csv_path)
            self.assertIsNone(recorder.record_completion())

            recorder_second = HandoverMetadataRecorder(csv_path=csv_path)
            recorder_second.mark_task_ready(
                shared_state,
                now_perf=200.0,
                now_timestamp_iso="2026-04-09T10:01:00.000Z",
            )
            recorder_second.note_delivery_location((150.0, 250.0, 350.0))
            recorder_second.record_completion()

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, METADATA_COLUMNS)
            self.assertEqual(len(rows), 2)

            first_row = rows[0]
            self.assertEqual(first_row["robot_initial_pose_x"], "100.000000")
            self.assertEqual(first_row["robot_initial_pose_y"], "200.000000")
            self.assertEqual(first_row["robot_initial_pose_z"], "300.000000")
            self.assertEqual(first_row["robot_mass_est_available"], "0")
            self.assertEqual(first_row["robot_mass_est_g"], "-1")
            self.assertEqual(first_row["geometry_est_timepoint"], "single_ms:200")
            self.assertEqual(first_row["fill_level_vision_timepoint"], "single_ms:250")
            self.assertEqual(first_row["mass_full_est_vision_timepoint"], "single_ms:250")
            self.assertEqual(first_row["t_robot_first_contact_ms"], "400")
            self.assertEqual(first_row["t_robot_last_contact_ms"], "800")
            self.assertEqual(first_row["delivery_location_est_x_mm"], "120.000000")
            self.assertEqual(first_row["delivery_location_est_y_mm"], "230.000000")
            self.assertAlmostEqual(float(first_row["delivery_location_est_z_mm"]), 290.0, places=5)
            expected_total_volume_ml = estimate_container_volume_ml_from_geometry(80.0, 60.0, 100.0)
            expected_filled_volume_ml = estimate_filled_volume_ml_from_geometry_and_fill_height(80.0, 60.0, 100.0, 50.0)
            expected_fill_level_percent = estimate_fill_level_percent(expected_filled_volume_ml, expected_total_volume_ml)
            expected_mass_full_g = estimate_mass_full_g_vision(expected_filled_volume_ml)
            self.assertAlmostEqual(float(first_row["fill_level_est_percent_vision"]), expected_fill_level_percent, places=5)
            self.assertAlmostEqual(float(first_row["mass_full_est_g_vision"]), expected_mass_full_g, places=4)
            self.assertEqual(first_row["config_id"], "")
            self.assertEqual(first_row["t_human_first_contact_ms"], "")
            self.assertEqual(first_row["final_mass_measured_g"], "")

    def test_metadata_recorder_uses_wine_glass_bowl_model(self) -> None:
        shared_state = SimpleNamespace(get_snapshot=lambda: {"initial_pose_base": None})

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HandoverMetadataRecorder(csv_path=Path(tmpdir) / "handover_metadata.csv")
            recorder.mark_task_ready(
                shared_state,
                now_perf=10.0,
                now_timestamp_iso="2026-04-12T10:00:00.000Z",
            )
            recorder._geometry_fields = {
                "width_top_est_mm_vision": 80.0,
                "width_bottom_est_mm_vision": 20.0,
                "height_est_mm_vision": 130.0,
            }
            updated = recorder.update_fill_and_mass(
                SimpleNamespace(valid=True, fill_height_mm=80.0),
                shape_fitting_state=SimpleNamespace(label="wine glass", bowl_height_fraction=0.42),
                now_perf=10.1,
                now_timestamp_iso="2026-04-12T10:00:00.100Z",
            )

            self.assertTrue(updated)
            expected_filled_ml = np.pi * (40.0 ** 2) * (130.0 * 0.42) / 1000.0
            expected_total_ml = expected_filled_ml
            expected_fill_level = estimate_fill_level_percent(expected_filled_ml, expected_total_ml)
            expected_mass = estimate_mass_full_g_vision(
                expected_filled_ml,
                empty_cup_mass_g=DEFAULT_EMPTY_WINE_GLASS_MASS_G,
            )
            self.assertAlmostEqual(recorder._fill_mass_fields["fill_level_est_percent_vision"], expected_fill_level, places=5)
            self.assertAlmostEqual(recorder._fill_mass_fields["mass_full_est_g_vision"], expected_mass, places=5)

    def test_metadata_recorder_zero_fill_uses_height_based_cup_mass(self) -> None:
        shared_state = SimpleNamespace(get_snapshot=lambda: {"initial_pose_base": None})

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = HandoverMetadataRecorder(csv_path=Path(tmpdir) / "handover_metadata.csv")
            recorder.mark_task_ready(
                shared_state,
                now_perf=10.0,
                now_timestamp_iso="2026-04-14T10:00:00.000Z",
            )
            recorder._geometry_fields = {
                "width_top_est_mm_vision": 80.0,
                "width_bottom_est_mm_vision": 60.0,
                "height_est_mm_vision": 120.0,
            }
            updated = recorder.update_fill_and_mass(
                SimpleNamespace(valid=True, fill_height_mm=0.0),
                shape_fitting_state=SimpleNamespace(label="cup", bowl_height_fraction=None),
                now_perf=10.1,
                now_timestamp_iso="2026-04-14T10:00:00.100Z",
            )

            self.assertTrue(updated)
            self.assertAlmostEqual(recorder._fill_mass_fields["fill_level_est_percent_vision"], 0.0, places=5)
            self.assertAlmostEqual(recorder._fill_mass_fields["mass_full_est_g_vision"], 9.0, places=5)


if __name__ == "__main__":
    unittest.main()
