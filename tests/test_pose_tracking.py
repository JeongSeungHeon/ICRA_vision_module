import unittest
from pathlib import Path

import numpy as np

from calibration.extrinsics import TransformChain
from perception.pose_tracking import (
    FoundationPoseTracker,
    ScaleEstimator,
    TemplateLibrary,
    TemplateModel,
    TemplateSpec,
)
from system.dual_sensor_hub import DualFrameSnapshot
from system.shared_state import MergedObjectState, ObjectState, POSE_TRACKING_REINIT_PENDING, POSE_TRACKING_TRACKING
from utils.realsense_stream import FrameBundle


def _robust_span(values: np.ndarray, low_q: float = 5.0, high_q: float = 95.0) -> float:
    return float(np.percentile(values, high_q) - np.percentile(values, low_q))


class SimpleVisual:
    def __init__(self, vertex_colors: np.ndarray) -> None:
        self.vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)


class SimpleMesh:
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.visual = SimpleVisual(np.tile(np.array([180, 180, 180], dtype=np.uint8), (len(self.vertices), 1)))

    def copy(self) -> "SimpleMesh":
        copied = SimpleMesh(self.vertices.copy())
        copied.visual.vertex_colors = self.visual.vertex_colors.copy()
        return copied

    def sample(self, count: int) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.empty((0, 3), dtype=np.float32)
        tiled = np.tile(self.vertices, (max(1, int(np.ceil(count / len(self.vertices)))), 1))
        return tiled[:count].astype(np.float32)


def make_box_vertices(extents: tuple[float, float, float]) -> np.ndarray:
    ex, ey, ez = [float(v) / 2.0 for v in extents]
    return np.asarray(
        [
            [-ex, -ey, -ez],
            [-ex, -ey, ez],
            [-ex, ey, -ez],
            [-ex, ey, ez],
            [ex, -ey, -ez],
            [ex, -ey, ez],
            [ex, ey, -ez],
            [ex, ey, ez],
        ],
        dtype=np.float32,
    )


def make_template_model(label: str, template_id: str, extents=(0.06, 0.06, 0.10)) -> TemplateModel:
    mesh = SimpleMesh(make_box_vertices(tuple(float(v) for v in extents)))
    canonical_points = mesh.sample(512).astype(np.float32)
    centered_points = canonical_points - np.mean(canonical_points, axis=0, keepdims=True)
    radial_extent = _robust_span(np.linalg.norm(centered_points[:, :2], axis=1)) * 2.0
    height_extent = _robust_span(centered_points[:, 2])
    return TemplateModel(
        spec=TemplateSpec(
            label=label,
            template_id=template_id,
            asset_path=Path(f"{template_id}.obj"),
            unit_scale_m=1.0,
            mesh_sample_count=512,
        ),
        mesh=mesh,
        canonical_points=canonical_points,
        centered_points=centered_points,
        radial_extent_m=max(radial_extent, 1e-3),
        height_extent_m=max(height_extent, 1e-3),
    )


def make_frame_bundle(depth_value: float = 1.0, width: int = 64, height: int = 64) -> FrameBundle:
    color = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), depth_value, dtype=np.float32)
    intrinsics = {"fx": 90.0, "fy": 90.0, "cx": width / 2.0, "cy": height / 2.0}
    return FrameBundle(
        color_image=color,
        depth_image_m=depth,
        intrinsics=intrinsics,
        timestamp_ms=0.0,
        serial="test",
    )


def make_snapshot() -> DualFrameSnapshot:
    frame = make_frame_bundle()
    return DualFrameSnapshot(pair_index=0, cam0=frame, cam1=frame, timestamp_delta_ms=0.0, within_sync_tolerance=True)


def make_mask(width: int = 64, height: int = 64, radius: int = 14) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    distances = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return distances <= float(radius)


def make_object_state(label: str) -> ObjectState:
    return ObjectState(camera_id=0, frame_id=0, object_detected=True, label=label, confidence=0.9, valid=True)


def make_merged_object(points: np.ndarray) -> MergedObjectState:
    centroid = np.mean(points, axis=0).astype(np.float32)
    return MergedObjectState(
        frame_id_cam0=0,
        frame_id_cam1=0,
        object_detected=True,
        label="cup",
        confidence=0.9,
        centroid_base=tuple(float(v) for v in centroid),
        merged_point_count=int(len(points)),
        merged_points_base=[tuple(float(v) for v in point) for point in points],
        valid=True,
    )


def make_cylinder_points(radius: float, height: float, count: int = 2048) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    z_values = np.linspace(-height / 2.0, height / 2.0, count, endpoint=True)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = z_values
    return np.stack([x, y, z], axis=1).astype(np.float32)


class FakeBackend:
    def __init__(self, register_pose: np.ndarray, track_pose: np.ndarray) -> None:
        self.available = True
        self.register_pose = np.asarray(register_pose, dtype=np.float32)
        self.track_pose = np.asarray(track_pose, dtype=np.float32)
        self.register_calls = 0
        self.track_calls = 0
        self.mesh = None

    def reset_object(self, mesh) -> None:
        self.mesh = mesh.copy()

    def register(self, frame, object_mask, iteration) -> np.ndarray:
        del frame, object_mask, iteration
        self.register_calls += 1
        return self.register_pose.copy()

    def track(self, frame, iteration) -> np.ndarray:
        del frame, iteration
        self.track_calls += 1
        return self.track_pose.copy()


class PoseTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template_cup = make_template_model("cup", "cup_template")
        self.template_wine = make_template_model("wine_glass", "wine_template")
        self.library = TemplateLibrary({"cup": self.template_cup, "wine_glass": self.template_wine})
        self.config = {
            "frames": {"height_axis": {"name": "z"}},
            "perception": {
                "pose_tracking": {
                    "enabled": True,
                    "anchor_camera": "cam0",
                    "surface_sample_count": 512,
                    "scale_estimation": {
                        "stable_frames": 1,
                        "min_points": 50,
                        "percentile_low": 5.0,
                        "percentile_high": 95.0,
                        "min_scale": 0.25,
                        "max_scale": 4.0,
                    },
                    "registration": {"refine_iter": 1},
                    "tracking": {"refine_iter": 1},
                    "health": {
                        "mask_iou_threshold": 0.01,
                        "depth_inlier_threshold": 0.10,
                        "depth_tolerance_m": 0.05,
                    },
                    "reinit": {
                        "low_conf_frames": 2,
                        "min_merged_points": 50,
                    },
                }
            },
        }
        self.transform_chain = TransformChain(
            t_base_cam0=np.eye(4, dtype=np.float32),
            t_cam0_cam1=np.eye(4, dtype=np.float32),
            t_base_cam1=np.eye(4, dtype=np.float32),
            source_config="test",
        )

    def build_tracker(self, backend: FakeBackend) -> FoundationPoseTracker:
        return FoundationPoseTracker(
            config=self.config,
            template_library=self.library,
            scale_estimator=ScaleEstimator(self.config),
            transform_chain=self.transform_chain,
            backend=backend,
        )

    def test_template_library_resolves_labels(self) -> None:
        self.assertIs(self.library.resolve("cup"), self.template_cup)
        self.assertIs(self.library.resolve("wine_glass"), self.template_wine)
        self.assertIsNone(self.library.resolve("missing"))

    def test_scale_estimator_stabilizes_radial_height_scale(self) -> None:
        estimator_config = {
            "frames": {"height_axis": {"name": "z"}},
            "perception": {"pose_tracking": {"scale_estimation": {"stable_frames": 3, "min_points": 50}}},
        }
        estimator = ScaleEstimator(estimator_config)
        template = self.template_cup
        observed = make_cylinder_points(radius=0.05, height=0.15)
        scale = None
        for _ in range(3):
            scale = estimator.estimate(template, observed)
        self.assertIsNotNone(scale)
        assert scale is not None
        self.assertGreater(scale.scale_xyz[0], 0.5)
        self.assertGreater(scale.scale_xyz[2], 1.0)

    def test_tracker_reinitializes_after_low_confidence_frames(self) -> None:
        register_pose = np.eye(4, dtype=np.float32)
        register_pose[2, 3] = 1.0
        track_pose = register_pose.copy()
        track_pose[0, 3] = 0.5
        backend = FakeBackend(register_pose=register_pose, track_pose=track_pose)
        tracker = self.build_tracker(backend)
        snapshot = make_snapshot()
        merged_points = make_cylinder_points(radius=0.03, height=0.08, count=512)
        merged_object = make_merged_object(merged_points)
        object_state = make_object_state("cup")
        mask = make_mask()

        tracker.set_anchor_observation(snapshot.cam0, mask)
        first_state = tracker.update(snapshot, object_state, object_state, merged_object)
        self.assertEqual(first_state.mode, POSE_TRACKING_TRACKING)
        self.assertTrue(first_state.valid)
        self.assertEqual(backend.register_calls, 1)

        tracker.set_anchor_observation(snapshot.cam0, mask)
        second_state = tracker.update(snapshot, object_state, object_state, merged_object)
        self.assertEqual(second_state.mode, POSE_TRACKING_TRACKING)
        self.assertEqual(backend.track_calls, 1)

        tracker.set_anchor_observation(snapshot.cam0, mask)
        third_state = tracker.update(snapshot, object_state, object_state, merged_object)
        self.assertEqual(third_state.mode, POSE_TRACKING_REINIT_PENDING)
        self.assertEqual(third_state.reinit_reason, "low_confidence")
        self.assertFalse(third_state.valid)

    def test_tracker_reinitializes_when_mask_disappears(self) -> None:
        register_pose = np.eye(4, dtype=np.float32)
        register_pose[2, 3] = 1.0
        backend = FakeBackend(register_pose=register_pose, track_pose=register_pose)
        tracker = self.build_tracker(backend)
        snapshot = make_snapshot()
        merged_points = make_cylinder_points(radius=0.03, height=0.08, count=512)
        merged_object = make_merged_object(merged_points)
        object_state = make_object_state("cup")

        tracker.set_anchor_observation(snapshot.cam0, make_mask())
        good_state = tracker.update(snapshot, object_state, object_state, merged_object)
        self.assertTrue(good_state.valid)

        tracker.set_anchor_observation(snapshot.cam0, np.zeros((64, 64), dtype=bool))
        reset_state = tracker.update(snapshot, object_state, object_state, merged_object)
        self.assertEqual(reset_state.mode, POSE_TRACKING_REINIT_PENDING)
        self.assertEqual(reset_state.reinit_reason, "mask_missing")

    def test_tracker_switches_template_on_label_change(self) -> None:
        register_pose = np.eye(4, dtype=np.float32)
        register_pose[2, 3] = 1.0
        backend = FakeBackend(register_pose=register_pose, track_pose=register_pose)
        tracker = self.build_tracker(backend)
        snapshot = make_snapshot()
        merged_points = make_cylinder_points(radius=0.03, height=0.08, count=512)
        merged_object = make_merged_object(merged_points)

        tracker.set_anchor_observation(snapshot.cam0, make_mask())
        cup_state = tracker.update(snapshot, make_object_state("cup"), make_object_state("cup"), merged_object)
        self.assertEqual(cup_state.template_id, "cup_template")

        tracker.set_anchor_observation(snapshot.cam0, make_mask())
        wine_state = tracker.update(
            snapshot,
            make_object_state("wine_glass"),
            make_object_state("wine_glass"),
            merged_object,
        )
        self.assertEqual(wine_state.template_id, "wine_template")
        self.assertGreaterEqual(backend.register_calls, 2)


if __name__ == "__main__":
    unittest.main()
