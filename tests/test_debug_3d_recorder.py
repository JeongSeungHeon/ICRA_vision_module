import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.debug_3d_recorder import Debug3DRecorder, load_debug_3d_npz
from tools.visualize_handover_3d_debug_rerun import (
    SELECTED_HAND_COLOR,
    UNSELECTED_HAND_COLOR,
    _build_blueprint,
    build_hand_payload,
    count_frame_entities,
    log_frame,
    log_points,
    log_tactile,
    point_coordinate_labels,
    tactile_status_text,
)


def _shape_state(points=None, valid=True, template_axes_base=None):
    if template_axes_base is None:
        template_axes_base = np.eye(3, dtype=np.float32)
    return SimpleNamespace(
        label="cup",
        template_id="cup_template",
        fitted_points_base=np.empty((0, 3), dtype=np.float32) if points is None else points,
        centroid_base=(0.2, 0.3, 0.4),
        scale=1.25,
        scale_xyz=(1.25, 1.25, 1.25),
        scale_mode="uniform",
        template_axes_base=template_axes_base,
        bowl_height_fraction=None,
        initialized=True,
        reason="ok",
        valid=bool(valid),
    )


def _shape_state_without_axes(points=None, valid=True):
    return SimpleNamespace(
        label="cup",
        template_id="cup_template",
        fitted_points_base=np.empty((0, 3), dtype=np.float32) if points is None else points,
        centroid_base=(0.2, 0.3, 0.4),
        scale=1.25,
        scale_xyz=(1.25, 1.25, 1.25),
        scale_mode="uniform",
        bowl_height_fraction=None,
        initialized=True,
        reason="ok",
        valid=bool(valid),
    )


def _merged_object(points=None, valid=True):
    points = np.empty((0, 3), dtype=np.float32) if points is None else points
    return SimpleNamespace(
        label="cup",
        merged_points_base=points,
        merged_point_count=len(points),
        centroid_base=(0.1, 0.2, 0.3),
        valid=bool(valid),
    )


def _hand_debug(offset=0.0):
    points = np.full((21, 3), np.nan, dtype=np.float32)
    points[:5] = np.arange(15, dtype=np.float32).reshape(5, 3) * 0.01 + float(offset)
    mask = np.zeros((21,), dtype=bool)
    mask[:5] = True
    return SimpleNamespace(
        points_3d_base=points,
        valid_mask=mask,
        palm_pose_base={"rotation_matrix": np.eye(3, dtype=np.float32), "valid": True, "quality": 0.8},
    )


def _invalid_hand_debug():
    return SimpleNamespace(
        points_3d_base=np.full((21, 3), np.nan, dtype=np.float32),
        valid_mask=np.zeros((21,), dtype=bool),
        palm_pose_base={"valid": False, "quality": 0.0},
    )


def _make_snapshot():
    cam0 = SimpleNamespace(
        color_image=np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        depth_image_m=np.arange(2 * 3, dtype=np.float32).reshape(2, 3) * 0.01,
        intrinsics={"fx": 100.0, "fy": 101.0, "cx": 50.0, "cy": 51.0},
        timestamp_ms=1234.5,
        serial="cam0_serial",
    )
    cam1 = SimpleNamespace(
        color_image=(np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) + 20),
        depth_image_m=(np.arange(2 * 3, dtype=np.float32).reshape(2, 3) + 10.0) * 0.01,
        intrinsics={"fx": 200.0, "fy": 201.0, "cx": 60.0, "cy": 61.0},
        timestamp_ms=1235.0,
        serial="cam1_serial",
    )
    return SimpleNamespace(
        pair_index=7,
        cam0=cam0,
        cam1=cam1,
        timestamp_delta_ms=0.5,
        within_sync_tolerance=True,
    )


def _append(
    recorder,
    frame_index,
    object_points,
    template_points,
    *,
    missing=False,
    snapshot=None,
    hand_debug_cam0=None,
    hand_debug_cam1=None,
    shape_fitting_state=None,
    tactile_snapshot=None,
    geometry_source="fitted_template",
):
    selected_hand = SimpleNamespace(
        valid=not missing,
        selected_camera=None if missing else 0,
        handedness=None if missing else "right",
        palm_center_base=None if missing else (0.4, 0.5, 0.6),
        palm_normal_base=None if missing else (0.0, 0.0, 1.0),
        wrist_base=None if missing else (0.3, 0.4, 0.5),
        confidence=0.9,
    )
    recorder.append_frame(
        frame_index=frame_index,
        timestamp_unix_s=100.0 + frame_index,
        timestamp_perf_s=10.0 + frame_index,
        record_elapsed_s=1.2 + frame_index,
        task_epoch=2,
        selected_hand=selected_hand,
        hand_debug_cam0=None if missing else (hand_debug_cam0 if hand_debug_cam0 is not None else _hand_debug(0.0)),
        hand_debug_cam1=None if missing else (hand_debug_cam1 if hand_debug_cam1 is not None else _hand_debug(0.1)),
        raw_merged_object=_merged_object(object_points, valid=not missing),
        shape_fitting_state=shape_fitting_state if shape_fitting_state is not None else _shape_state(template_points, valid=not missing),
        object_point_base=None if missing else (0.1, 0.2, 0.3),
        grasp_point_base=None if missing else (0.2, 0.3, 0.4),
        eef_pose_base=None if missing else (0.5, 0.6, 0.7, 0.0, 0.1, 0.2),
        measurement_source="measured",
        geometry_source=geometry_source,
        hand_selector_debug=SimpleNamespace(
            selection_reason="no_valid_candidate" if missing else "select_best_available",
            cam0_reject_reason="hand_not_detected" if missing else "",
            cam1_reject_reason="hand_not_detected" if missing else "",
            chosen_camera=None if missing else 0,
        ),
        tactile_snapshot=tactile_snapshot,
        snapshot=snapshot,
    )


class _FakeRerun:
    class Points3D:
        def __init__(self, points, **kwargs):
            self.points = np.asarray(points)
            self.kwargs = kwargs

    class Clear:
        def __init__(self, recursive=False):
            self.recursive = recursive

    class LineStrips3D:
        def __init__(self, strips, **kwargs):
            self.strips = [np.asarray(strip) for strip in strips]
            self.kwargs = kwargs

    class Scalars:
        def __init__(self, value):
            self.value = float(value)

    class TextDocument:
        def __init__(self, text):
            self.text = str(text)

    def __init__(self):
        self.logged = []
        self.times = []

    def log(self, entity, payload):
        self.logged.append((entity, payload))

    def set_time(self, timeline, **kwargs):
        self.times.append((timeline, kwargs))


class _FakeBlueprintNode:
    def __init__(self, kind, *children, origin=None, name=None, contents=None, **kwargs):
        self.kind = kind
        if children == () and contents is not None and kind in {"Grid", "Horizontal", "Vertical"}:
            self.children = list(contents)
        else:
            self.children = list(children)
        self.origin = origin
        self.name = name
        self.contents = [] if contents is None or kind in {"Grid", "Horizontal", "Vertical"} else list(contents)
        self.kwargs = kwargs


class _FakeBlueprintModule:
    class Blueprint(_FakeBlueprintNode):
        def __init__(self, *children, **kwargs):
            super().__init__("Blueprint", *children, **kwargs)

    class Spatial3DView(_FakeBlueprintNode):
        def __init__(self, **kwargs):
            super().__init__("Spatial3DView", **kwargs)

    class Spatial2DView(_FakeBlueprintNode):
        def __init__(self, **kwargs):
            super().__init__("Spatial2DView", **kwargs)

    class TextDocumentView(_FakeBlueprintNode):
        def __init__(self, **kwargs):
            super().__init__("TextDocumentView", **kwargs)

    class TimeSeriesView(_FakeBlueprintNode):
        def __init__(self, **kwargs):
            super().__init__("TimeSeriesView", **kwargs)

    class Grid(_FakeBlueprintNode):
        def __init__(self, *children, **kwargs):
            super().__init__("Grid", *children, **kwargs)

    class Horizontal(_FakeBlueprintNode):
        def __init__(self, *children, **kwargs):
            super().__init__("Horizontal", *children, **kwargs)

    class Vertical(_FakeBlueprintNode):
        def __init__(self, *children, **kwargs):
            super().__init__("Vertical", *children, **kwargs)


def _collect_blueprint_nodes(node, kind):
    matches = []
    if getattr(node, "kind", None) == kind:
        matches.append(node)
    for child in getattr(node, "children", []):
        matches.extend(_collect_blueprint_nodes(child, kind))
    return matches


class Debug3DRecorderTests(unittest.TestCase):
    def test_point_coordinate_labels_format_and_precision(self):
        points = np.asarray([[0.123456, -0.045678, 0.789012], [1.0, 2.0, 3.0]], dtype=np.float32)

        labels = point_coordinate_labels(points, "object", precision=3)

        self.assertEqual(labels[0], "object[0000] base=(0.123, -0.046, 0.789)m")
        self.assertEqual(labels[1], "object[0001] base=(1.000, 2.000, 3.000)m")

    def test_point_coordinate_labels_filters_invalid_and_empty_points(self):
        points = np.asarray([[np.nan, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=np.float32)

        labels = point_coordinate_labels(points, "template", precision=1)

        self.assertEqual(labels, ["template[0000] base=(0.1, 0.2, 0.3)m"])
        self.assertEqual(point_coordinate_labels(np.empty((0, 3), dtype=np.float32), "object"), [])

    def test_log_points_passes_coordinate_labels_to_rerun(self):
        fake_rr = _FakeRerun()
        points = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
        labels = point_coordinate_labels(points, "object", precision=2)

        log_points(fake_rr, "/world/object/cloud", points, (255, 255, 255), radius=0.01, labels=labels, show_labels=False)

        self.assertEqual(fake_rr.logged[0][0], "/world/object/cloud")
        payload = fake_rr.logged[0][1]
        np.testing.assert_allclose(payload.points, points)
        self.assertEqual(payload.kwargs["labels"], ["object[0000] base=(0.10, 0.20, 0.30)m"])
        self.assertFalse(payload.kwargs["show_labels"])
        np.testing.assert_allclose(payload.kwargs["radii"], np.asarray([0.01], dtype=np.float32))

    def test_log_points_rejects_mismatched_label_count(self):
        fake_rr = _FakeRerun()
        points = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

        with self.assertRaises(ValueError):
            log_points(fake_rr, "/world/object/cloud", points, (255, 255, 255), labels=[])

    def test_blueprint_includes_tactile_graphs_and_leaf_status_views(self):
        data = {
            "frame_count": np.asarray(1, dtype=np.int32),
            "tactile_enabled": np.asarray([True]),
            "tactile_values": np.zeros((1, 2, 3), dtype=np.float32),
            "max_tactile_mags": np.asarray(2, dtype=np.int32),
        }

        blueprint = _build_blueprint(_FakeBlueprintModule, data, include_images=True)

        text_origins = [node.origin for node in _collect_blueprint_nodes(blueprint, "TextDocumentView")]
        self.assertIn("/status/frame", text_origins)
        self.assertIn("/status/sync", text_origins)
        self.assertIn("/status/tactile", text_origins)
        self.assertNotIn("/status", text_origins)

        time_series = _collect_blueprint_nodes(blueprint, "TimeSeriesView")
        self.assertEqual([node.name for node in time_series], ["Tactile Norms", "Tactile XYZ"])
        self.assertTrue(all(node.origin == "/tactile" for node in time_series))
        self.assertIn("/tactile/total_norm", time_series[0].contents)
        self.assertIn("/tactile/mag_1/norm", time_series[0].contents)
        self.assertIn("/tactile/mag_0/x", time_series[1].contents)
        self.assertIn("/tactile/mag_1/z", time_series[1].contents)

        world_views = _collect_blueprint_nodes(blueprint, "Spatial3DView")
        self.assertEqual(len(world_views), 1)
        self.assertEqual(world_views[0].kwargs["background"], [255, 255, 255])

    def test_blueprint_omits_tactile_graph_without_tactile_stream(self):
        data = {"frame_count": np.asarray(1, dtype=np.int32)}

        blueprint = _build_blueprint(_FakeBlueprintModule, data, include_images=False)

        self.assertEqual(_collect_blueprint_nodes(blueprint, "TimeSeriesView"), [])
        text_origins = [node.origin for node in _collect_blueprint_nodes(blueprint, "TextDocumentView")]
        self.assertEqual(text_origins, ["/status/frame"])

    def test_round_trip_variable_clouds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), max_object_points=10, max_template_points=8)
            object_points_0 = np.arange(45, dtype=np.float32).reshape(15, 3)
            template_points_0 = np.arange(36, dtype=np.float32).reshape(12, 3) * 0.1
            object_points_1 = np.arange(12, dtype=np.float32).reshape(4, 3)
            template_points_1 = np.arange(18, dtype=np.float32).reshape(6, 3) * 0.1

            _append(recorder, 0, object_points_0, template_points_0)
            _append(recorder, 1, object_points_1, template_points_1)

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(int(data["frame_count"]), 2)
            self.assertEqual(data["object_points_base"][0].shape, (10, 3))
            self.assertEqual(data["template_points_base"][0].shape, (8, 3))
            self.assertEqual(data["object_points_base"][1].shape, (4, 3))
            self.assertEqual(data["template_points_base"][1].shape, (6, 3))
            np.testing.assert_allclose(data["grasp_point_base"][0], np.asarray([0.2, 0.3, 0.4], dtype=np.float32))
            np.testing.assert_allclose(
                data["eef_pose_base"][0],
                np.asarray([0.5, 0.6, 0.7, 0.0, 0.1, 0.2], dtype=np.float32),
            )
            self.assertEqual(str(data["record_clock_text"][0]), "REC 00:01.1")
            self.assertEqual(str(data["geometry_source"][0]), "fitted_template")
            self.assertAlmostEqual(float(data["record_elapsed_s"][0]), 1.2, places=6)
            self.assertNotIn("cam0_color_image", data)
            self.assertNotIn("cam0_depth_image_m", data)
            self.assertEqual(data["template_axes_base"].shape, (2, 3, 3))
            np.testing.assert_allclose(data["template_axes_base"][0], np.eye(3, dtype=np.float32))

    def test_raw_geometry_source_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
                geometry_source="raw_point_cloud",
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(int(data["schema_version"]), 8)
            self.assertEqual(str(data["geometry_source"][0]), "raw_point_cloud")

    def test_template_axes_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), record_template_axes=False)
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertNotIn("template_axes_base", data)

    def test_missing_template_axes_are_saved_as_nan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                shape_fitting_state=_shape_state_without_axes(np.empty((0, 3), dtype=np.float32)),
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(data["template_axes_base"].shape, (1, 3, 3))
            self.assertTrue(np.isnan(data["template_axes_base"][0]).all())

    def test_hand_payload_uses_high_contrast_colors_for_white_background(self):
        points = np.arange(9, dtype=np.float32).reshape(3, 3)
        valid_mask = np.ones((3,), dtype=bool)

        _points, _strips, selected_color = build_hand_payload(points, valid_mask, selected=True)
        _points, _strips, unselected_color = build_hand_payload(points, valid_mask, selected=False)

        self.assertEqual(selected_color, SELECTED_HAND_COLOR)
        self.assertEqual(unselected_color, UNSELECTED_HAND_COLOR)

    def test_rerun_logging_omits_template_axes_but_keeps_template_and_eef(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
                np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32),
            )
            data = load_debug_3d_npz(recorder.save())

            fake_rr = _FakeRerun()
            log_frame(
                fake_rr,
                data,
                0,
                normal_length_m=0.08,
                depth_max_m=1.7,
                include_coordinate_labels=False,
                point_coordinate_precision=4,
                selected_hand_only=False,
                point_radius_scale=0.5,
                hand_line_radius_scale=0.5,
            )
            logged_entities = [entity for entity, _payload in fake_rr.logged]
            logged_payloads = {entity: payload for entity, payload in fake_rr.logged}

            self.assertNotIn("/world/template/local_axes", logged_entities)
            self.assertIn("/world/template/cloud", logged_entities)
            self.assertIn("/world/template/centroid", logged_entities)
            self.assertIn("/world/eef/axes", logged_entities)
            np.testing.assert_array_equal(
                logged_payloads["/world/hands/cam0/keypoints"].kwargs["colors"][0],
                np.asarray(SELECTED_HAND_COLOR, dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                logged_payloads["/world/hands/cam1/keypoints"].kwargs["colors"][0],
                np.asarray(UNSELECTED_HAND_COLOR, dtype=np.uint8),
            )

    def test_missing_values_are_nan_or_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                missing=True,
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(data["object_points_base"][0].shape, (0, 3))
            self.assertEqual(data["template_points_base"][0].shape, (0, 3))
            self.assertTrue(np.isnan(data["object_point_base"][0]).all())
            self.assertTrue(np.isnan(data["grasp_point_base"][0]).all())
            self.assertTrue(np.isnan(data["eef_pose_base"][0]).all())
            self.assertEqual(data["selected_hand_camera"][0], -1)
            self.assertFalse(bool(data["selected_hand_valid"][0]))

    def test_raw_images_round_trip_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = _make_snapshot()
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), save_images=True)
            _append(
                recorder,
                0,
                np.arange(12, dtype=np.float32).reshape(4, 3),
                np.arange(9, dtype=np.float32).reshape(3, 3),
                snapshot=snapshot,
            )

            data = load_debug_3d_npz(recorder.save())

            np.testing.assert_array_equal(data["cam0_color_image"][0], snapshot.cam0.color_image)
            np.testing.assert_allclose(data["cam0_depth_image_m"][0], snapshot.cam0.depth_image_m)
            np.testing.assert_allclose(data["cam0_intrinsics"][0], np.asarray([100.0, 101.0, 50.0, 51.0]))
            self.assertAlmostEqual(float(data["cam0_timestamp_ms"][0]), 1234.5)
            self.assertEqual(str(data["cam0_serial"][0]), "cam0_serial")
            self.assertEqual(int(data["cam0_hand_valid_count"][0]), 5)
            self.assertTrue(bool(data["cam0_hand_detected"][0]))
            self.assertEqual(str(data["hand_selection_reason"][0]), "select_best_available")

            np.testing.assert_array_equal(data["cam1_color_image"][0], snapshot.cam1.color_image)
            np.testing.assert_allclose(data["cam1_depth_image_m"][0], snapshot.cam1.depth_image_m)
            np.testing.assert_allclose(data["cam1_intrinsics"][0], np.asarray([200.0, 201.0, 60.0, 61.0]))
            self.assertAlmostEqual(float(data["cam1_timestamp_ms"][0]), 1235.0)
            self.assertEqual(str(data["cam1_serial"][0]), "cam1_serial")
            self.assertAlmostEqual(float(data["timestamp_delta_ms"][0]), 0.5)
            self.assertTrue(bool(data["within_sync_tolerance"][0]))

    def test_saved_arrays_are_not_aliased_to_mutated_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = _make_snapshot()
            object_points = np.arange(12, dtype=np.float32).reshape(4, 3)
            template_points = np.arange(9, dtype=np.float32).reshape(3, 3)
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), save_images=True)

            original_color = snapshot.cam0.color_image.copy()
            original_depth = snapshot.cam0.depth_image_m.copy()
            original_object = object_points.copy()
            original_template = template_points.copy()
            _append(recorder, 0, object_points, template_points, snapshot=snapshot)

            snapshot.cam0.color_image[...] = 255
            snapshot.cam0.depth_image_m[...] = 9.0
            object_points[...] = -1.0
            template_points[...] = -2.0

            data = load_debug_3d_npz(recorder.save())

            np.testing.assert_array_equal(data["cam0_color_image"][0], original_color)
            np.testing.assert_allclose(data["cam0_depth_image_m"][0], original_depth)
            np.testing.assert_allclose(data["object_points_base"][0], original_object)
            np.testing.assert_allclose(data["template_points_base"][0], original_template)

    def test_multiple_image_frames_round_trip_independently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), save_images=True)
            snapshot0 = _make_snapshot()
            snapshot1 = _make_snapshot()
            snapshot1.cam0.color_image = snapshot1.cam0.color_image + 7
            snapshot1.cam1.color_image = snapshot1.cam1.color_image + 13
            snapshot1.cam0.depth_image_m = snapshot1.cam0.depth_image_m + 0.5
            snapshot1.cam1.depth_image_m = snapshot1.cam1.depth_image_m + 0.7

            _append(recorder, 0, np.arange(6, dtype=np.float32).reshape(2, 3), np.empty((0, 3)), snapshot=snapshot0)
            _append(recorder, 1, np.arange(6, dtype=np.float32).reshape(2, 3), np.empty((0, 3)), snapshot=snapshot1)

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(data["cam0_color_image"].shape[0], 2)
            self.assertNotEqual(int(data["cam0_color_image"][0].sum()), int(data["cam0_color_image"][1].sum()))
            self.assertNotEqual(float(data["cam1_depth_image_m"][0].mean()), float(data["cam1_depth_image_m"][1].mean()))

    def test_raw_images_and_tactile_round_trip_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = _make_snapshot()
            recorder = Debug3DRecorder(output_dir=Path(temp_dir), save_images=True)
            tactile_values = np.asarray([0.5, 1.5, 2.5], dtype=np.float32)
            _append(
                recorder,
                0,
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.empty((0, 3), dtype=np.float32),
                snapshot=snapshot,
                tactile_snapshot={
                    "values": tactile_values,
                    "num_mags": 1,
                    "total_norm": float(np.linalg.norm(tactile_values)),
                    "status": "synced_with_images",
                    "timestamp_perf_s": 10.02,
                    "timestamp_unix_s": 100.02,
                },
            )

            data = load_debug_3d_npz(recorder.save())

            np.testing.assert_array_equal(data["cam0_color_image"][0], snapshot.cam0.color_image)
            np.testing.assert_allclose(data["cam1_depth_image_m"][0], snapshot.cam1.depth_image_m)
            self.assertTrue(bool(data["within_sync_tolerance"][0]))
            self.assertTrue(bool(data["tactile_enabled"][0]))
            self.assertEqual(data["tactile_values"].shape, (1, 1, 3))
            np.testing.assert_allclose(data["tactile_values"][0, 0], tactile_values)
            self.assertEqual(str(data["tactile_status"][0]), "synced_with_images")
            self.assertAlmostEqual(float(data["tactile_frame_delta_ms"][0]), 20.0, places=3)

    def test_tactile_snapshot_round_trip_and_sync_delta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            tactile_values = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0, 0.0], dtype=np.float32)
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                tactile_snapshot={
                    "values": tactile_values,
                    "num_mags": 2,
                    "total_norm": 5.477,
                    "release_ref_norm": 10.0,
                    "release_delta_norm": 1.5,
                    "status": "close_monitoring",
                    "error": None,
                    "timestamp_perf_s": 10.05,
                    "timestamp_unix_s": 100.05,
                },
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertTrue(bool(data["tactile_enabled"][0]))
            self.assertEqual(int(data["tactile_num_mags"][0]), 2)
            self.assertEqual(data["tactile_values"].shape, (1, 2, 3))
            np.testing.assert_allclose(data["tactile_values"][0], tactile_values.reshape(2, 3))
            np.testing.assert_allclose(data["tactile_mag_norms"][0], np.asarray([3.0, 5.0], dtype=np.float32))
            self.assertAlmostEqual(float(data["tactile_total_norm"][0]), 5.477, places=3)
            self.assertAlmostEqual(float(data["tactile_release_ref_norm"][0]), 10.0)
            self.assertAlmostEqual(float(data["tactile_release_delta_norm"][0]), 1.5)
            self.assertEqual(str(data["tactile_status"][0]), "close_monitoring")
            self.assertEqual(str(data["tactile_error"][0]), "")
            self.assertAlmostEqual(float(data["tactile_frame_delta_ms"][0]), 50.0, places=3)

    def test_tactile_disabled_fields_are_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertIn("tactile_values", data)
            self.assertFalse(bool(data["tactile_enabled"][0]))
            self.assertEqual(int(data["tactile_num_mags"][0]), 0)
            self.assertEqual(data["tactile_values"].shape, (1, 0, 3))
            self.assertEqual(str(data["tactile_status"][0]), "disabled")
            self.assertTrue(np.isnan(float(data["tactile_total_norm"][0])))

    def test_tactile_values_are_padded_and_copied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            values0 = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
            values1 = np.asarray([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                tactile_snapshot={"values": values0, "num_mags": 1, "status": "one"},
            )
            _append(
                recorder,
                1,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                tactile_snapshot={"values": values1, "num_mags": 2, "status": "two"},
            )
            values0[...] = -1.0
            values1[...] = -2.0

            data = load_debug_3d_npz(recorder.save())

            self.assertEqual(data["tactile_values"].shape, (2, 2, 3))
            np.testing.assert_allclose(data["tactile_values"][0, 0], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            self.assertTrue(np.isnan(data["tactile_values"][0, 1]).all())
            np.testing.assert_allclose(data["tactile_values"][1, 0], np.asarray([4.0, 5.0, 6.0], dtype=np.float32))
            self.assertAlmostEqual(float(data["tactile_values"][1, 1, 0]), 7.0)
            self.assertTrue(np.isnan(float(data["tactile_values"][1, 1, 1])))

    def test_rerun_entity_count_and_logging_include_tactile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )
            disabled_data = load_debug_3d_npz(recorder.save())

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                tactile_snapshot={
                    "values": np.asarray([1.0, 2.0, 2.0], dtype=np.float32),
                    "num_mags": 1,
                    "total_norm": 3.0,
                    "status": "armed",
                },
            )
            tactile_data = load_debug_3d_npz(recorder.save())

        self.assertGreater(count_frame_entities(tactile_data, 0), count_frame_entities(disabled_data, 0))
        self.assertIn("status=armed", tactile_status_text(tactile_data, 0))

        fake_rr = _FakeRerun()
        log_tactile(fake_rr, tactile_data, 0)
        logged_entities = [entity for entity, _payload in fake_rr.logged]
        self.assertIn("/status/tactile", logged_entities)
        self.assertIn("/tactile/total_norm", logged_entities)
        self.assertIn("/tactile/mag_0/norm", logged_entities)

    def test_invalid_hand_debug_diagnostics_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Debug3DRecorder(output_dir=Path(temp_dir))
            _append(
                recorder,
                0,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                hand_debug_cam0=_invalid_hand_debug(),
                hand_debug_cam1=_invalid_hand_debug(),
            )

            data = load_debug_3d_npz(recorder.save())

            self.assertFalse(bool(data["cam0_hand_detected"][0]))
            self.assertEqual(int(data["cam0_hand_valid_count"][0]), 0)
            self.assertEqual(str(data["cam0_hand_reason"][0]), "no_valid_depth_landmarks")


if __name__ == "__main__":
    unittest.main()
