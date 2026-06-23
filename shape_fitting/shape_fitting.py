from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.extrinsics import load_transform_chain  # noqa: E402
from object_pt_extraction.pointcloud_utils import (  # noqa: E402
    colorize_selected_mask,
    merge_point_clouds,
    render_projected_point_cloud_overlay,
    voxel_downsample_point_cloud,
)
from object_pt_extraction.segmentation_engine import (  # noqa: E402
    SegmentationEngine,
    format_instance_summary,
    select_instances,
)
from system.dual_sensor_hub import DualSensorHub  # noqa: E402
from utils.depth_filters import bilateral_filter_depth  # noqa: E402


WIDTH = 640
HEIGHT = 480
FPS = 30
YOLO_MODEL_PATH = REPO_ROOT / "yoloe-26l-seg.pt"
CONFIG_PATH = REPO_ROOT / "configs" / "handover.yaml"
CONFIG_PATH = REPO_ROOT / "configs" / "handover.yaml"
TEMPLATE_PATH = SCRIPT_DIR / "glass.npy"
TARGET_CLASSES = ["wine glass", "cup"]
DEPTH_MIN_M = 0.30
DEPTH_MAX_M = 1.50
BILATERAL_ENABLED = True
BILATERAL_RADIUS = 2
BILATERAL_SIGMA_SPACE = 2.0
BILATERAL_ZFAR = 100.0
POINT_STRIDE = 2
POINT_MAX_POINTS = 12000
PER_CAMERA_VOXEL_SIZE_M = 0.004
MERGE_VOXEL_SIZE_M = 0.010
ICP_MAX_POINTS = 6000
DBSCAN_EPS_M = 0.02
DBSCAN_MIN_POINTS = 10
TRACKING_CLUSTER_MAX_JUMP_M = 0.08
SCALE_INIT_VALID_FRAMES = 8
ROBUST_EXTENT_LOW_PERCENTILE = 5.0
ROBUST_EXTENT_HIGH_PERCENTILE = 95.0
MIN_TEMPLATE_SCALE = 0.5
MAX_TEMPLATE_SCALE = 1.8
RESET_REALSENSE_ON_EXIT = False

DISPLAY_TRANSFORM = np.asarray(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


@dataclass
class ProcessedCameraFrame:
    camera_id: int
    valid: bool
    points_base: np.ndarray
    colors_rgb: np.ndarray
    preview_bgr: np.ndarray
    status: str


def reset_realsense() -> None:
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not available; skipping RealSense hardware reset.")
        return

    print("Finding RealSense devices...")
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    for dev in devices:
        print(f"Sending hardware reset to: {dev.get_info(rs.camera_info.name)}")
        dev.hardware_reset()
    print("Reset command sent. Wait 3-5 seconds before running the camera again.")


def overlay_status(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    preview = frame_bgr.copy()
    for line_index, text in enumerate(lines):
        origin = (12, 28 + line_index * 24)
        cv2.putText(preview, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def stack_previews(previews: list[np.ndarray]) -> np.ndarray:
    max_height = max(preview.shape[0] for preview in previews)
    resized = []
    for preview in previews:
        if preview.shape[0] == max_height:
            resized.append(preview)
            continue
        scale = max_height / preview.shape[0]
        width = int(round(preview.shape[1] * scale))
        resized.append(cv2.resize(preview, (width, max_height)))
    return cv2.hconcat(resized)


def combine_instance_masks(instances: list, image_shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=bool)
    for instance in instances:
        mask |= np.asarray(instance.mask, dtype=bool)
    return mask


def build_masked_point_cloud(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    intrinsics: dict[str, float],
    depth_min: float,
    depth_max: float,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(int(stride), 1)
    mask_s = object_mask[::stride, ::stride]
    depth_s = depth_m[::stride, ::stride]
    valid = np.logical_and(mask_s, np.isfinite(depth_s))
    valid = np.logical_and(valid, depth_s >= float(depth_min))
    valid = np.logical_and(valid, depth_s <= float(depth_max))
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    u = (cols * stride).astype(np.float32)
    v = (rows * stride).astype(np.float32)
    z = depth_s[rows, cols].astype(np.float32)
    x = (u - float(intrinsics["cx"])) / float(intrinsics["fx"]) * z
    y = (v - float(intrinsics["cy"])) / float(intrinsics["fy"]) * z
    points = np.stack((x, y, z), axis=1).astype(np.float32)

    colors_rgb = color_bgr[(rows * stride).astype(np.int64), (cols * stride).astype(np.int64), ::-1].copy()
    if max_points > 0 and points.shape[0] > max_points:
        selection = np.random.default_rng().choice(points.shape[0], max_points, replace=False)
        points = points[selection]
        colors_rgb = colors_rgb[selection]
    return points, colors_rgb.astype(np.uint8)


def limit_point_count(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if max_points <= 0 or len(points) <= max_points:
        return points
    sample_indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[sample_indices]


def compute_robust_extent(
    points: np.ndarray,
    low_percentile: float = ROBUST_EXTENT_LOW_PERCENTILE,
    high_percentile: float = ROBUST_EXTENT_HIGH_PERCENTILE,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return np.zeros((3,), dtype=np.float32)

    lower = np.percentile(points, low_percentile, axis=0)
    upper = np.percentile(points, high_percentile, axis=0)
    return (upper - lower).astype(np.float32)


def translate_points_to_centroid(points: np.ndarray, target_centroid: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return points

    current_centroid = np.mean(points, axis=0)
    translation = np.asarray(target_centroid, dtype=np.float64).reshape(3) - current_centroid
    return points + translation.reshape(1, 3)


def scale_template_points(template_points: np.ndarray, uniform_scale: float) -> np.ndarray:
    template_points = np.asarray(template_points, dtype=np.float64).reshape((-1, 3))
    if len(template_points) == 0:
        return template_points

    centroid = np.mean(template_points, axis=0, keepdims=True)
    return (template_points - centroid) * float(uniform_scale) + centroid


def filter_pcd(points: np.ndarray, prev_centroid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return points, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("Running DBSCAN clustering...")
    labels = np.array(
        pcd.cluster_dbscan(
            eps=DBSCAN_EPS_M,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=False,
        )
    )
    valid_cluster_ids = np.unique(labels[labels >= 0])
    num_clusters = len(valid_cluster_ids)
    print(f"Found {num_clusters} valid clusters.")

    if num_clusters == 0:
        print("No valid DBSCAN clusters. Skipping this frame.")
        return np.empty((0, 3), dtype=np.float64), None

    cluster_points = {cluster_id: points[labels == cluster_id] for cluster_id in valid_cluster_ids}
    cluster_counts = {cluster_id: len(cluster_pts) for cluster_id, cluster_pts in cluster_points.items()}
    cluster_centroids = {
        cluster_id: np.mean(cluster_pts, axis=0)
        for cluster_id, cluster_pts in cluster_points.items()
    }

    selected_cluster_id = max(cluster_counts, key=cluster_counts.get)
    selection_reason = "largest cluster"
    if prev_centroid is not None:
        prev_centroid = np.asarray(prev_centroid, dtype=np.float64).reshape(3)
        closest_cluster_id = min(
            cluster_centroids,
            key=lambda cluster_id: np.linalg.norm(cluster_centroids[cluster_id] - prev_centroid),
        )
        closest_distance = float(np.linalg.norm(cluster_centroids[closest_cluster_id] - prev_centroid))
        if closest_distance <= TRACKING_CLUSTER_MAX_JUMP_M:
            selected_cluster_id = closest_cluster_id
            selection_reason = f"closest cluster ({closest_distance:.3f} m)"
        else:
            selection_reason = f"largest cluster (closest was {closest_distance:.3f} m away)"

    print(f"Selected cluster {selected_cluster_id} using {selection_reason}.")
    selected_points = cluster_points[selected_cluster_id].astype(np.float64)
    selected_centroid = cluster_centroids[selected_cluster_id].astype(np.float64)
    return selected_points, selected_centroid


def scale_template(template: o3d.geometry.PointCloud, pcd: o3d.geometry.PointCloud) -> np.ndarray:
    extent_source = template.get_axis_aligned_bounding_box().get_extent()
    extent_target = pcd.get_axis_aligned_bounding_box().get_extent()
    if np.any(extent_source <= 1e-9) or np.any(extent_target <= 1e-9):
        print("Skipping template scale: invalid source/target extent.")
        return np.asarray(template.points)

    scale = extent_target / extent_source
    template_points = np.asarray(template.points)
    centroid = np.mean(template_points, axis=0, keepdims=True)
    return (template_points - centroid) * scale.reshape(1, 3) + centroid


def run_open3d_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray | None = None,
) -> np.ndarray:
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    if len(source_points) == 0 or len(target_points) == 0:
        return np.eye(4, dtype=np.float64)

    threshold = 0.005
    transform = np.eye(4, dtype=np.float64) if init_transform is None else np.array(init_transform, dtype=np.float64, copy=True)
    transform[:3, 3] = np.mean(target_points, axis=0) - np.mean(source_points, axis=0)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return reg_p2p.transformation


def transform_points_base_to_cam0(points_base: np.ndarray, transform_chain) -> np.ndarray:
    points_base = np.asarray(points_base, dtype=np.float32).reshape((-1, 3))
    if len(points_base) == 0:
        return np.empty((0, 3), dtype=np.float32)

    t_cam0_base = np.linalg.inv(np.asarray(transform_chain.t_base_cam0, dtype=np.float32))
    rotation = t_cam0_base[:3, :3]
    translation = t_cam0_base[:3, 3].reshape(1, 3)
    return (points_base @ rotation.T + translation).astype(np.float32)


def transform_points_display_to_base(points_display: np.ndarray) -> np.ndarray:
    points_display = np.asarray(points_display, dtype=np.float32).reshape((-1, 3))
    if len(points_display) == 0:
        return np.empty((0, 3), dtype=np.float32)

    display_to_base = np.linalg.inv(DISPLAY_TRANSFORM).astype(np.float32)
    rotation = display_to_base[:3, :3]
    translation = display_to_base[:3, 3].reshape(1, 3)
    return (points_display @ rotation.T + translation).astype(np.float32)


def create_segmentation_engine() -> SegmentationEngine:
    return SegmentationEngine(
        model_name=str(YOLO_MODEL_PATH),
        prompt_classes=TARGET_CLASSES,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        max_det=100,
        device=None,
        classes=None,
        half=False,
        retina_masks=True,
    )


def process_camera_frame(
    frame_bundle,
    camera_id: int,
    segmentation_engine: SegmentationEngine,
    transform_chain,
) -> ProcessedCameraFrame:
    raw_depth_m = np.asarray(frame_bundle.depth_image_m, dtype=np.float32)
    filtered_depth_m = raw_depth_m
    bilateral_status = "off"
    if BILATERAL_ENABLED:
        try:
            filtered_depth_m = bilateral_filter_depth(
                filtered_depth_m,
                radius=BILATERAL_RADIUS,
                zfar=BILATERAL_ZFAR,
                sigma_space=BILATERAL_SIGMA_SPACE,
            )
            bilateral_status = "on"
        except Exception as exc:
            print(f"[WARN] cam{camera_id} bilateral filter failed; using raw depth: {exc}")
            filtered_depth_m = raw_depth_m
            bilateral_status = "fallback"

    segmentation_result = segmentation_engine.predict(frame_bundle.color_image)
    selected_instances = select_instances(
        segmentation_result.instances,
        mode="highest_score",
        class_names=TARGET_CLASSES,
    )
    combined_mask = combine_instance_masks(selected_instances, frame_bundle.color_image.shape)
    points_camera, colors_rgb = build_masked_point_cloud(
        color_bgr=frame_bundle.color_image,
        depth_m=filtered_depth_m,
        object_mask=combined_mask,
        intrinsics=frame_bundle.intrinsics,
        depth_min=DEPTH_MIN_M,
        depth_max=DEPTH_MAX_M,
        stride=POINT_STRIDE,
        max_points=POINT_MAX_POINTS,
    )
    points_base = transform_chain.transform_points_camera_to_base(camera_id, points_camera)
    points_base, colors_rgb = voxel_downsample_point_cloud(
        points_base,
        colors_rgb,
        voxel_size_m=PER_CAMERA_VOXEL_SIZE_M,
    )
    valid = len(selected_instances) > 0 and len(points_base) > 0

    annotated = segmentation_engine.render(segmentation_result)
    annotated = colorize_selected_mask(annotated, combined_mask)
    preview = overlay_status(
        annotated,
        [
            f"cam{camera_id} serial: {frame_bundle.serial}",
            format_instance_summary(selected_instances),
            f"seg: {segmentation_result.infer_ms:.1f} ms | depth: raw | bilateral: {bilateral_status}",
            f"points_base: {len(points_base)} | {'ok' if valid else 'no_object_points'}",
        ],
    )
    return ProcessedCameraFrame(
        camera_id=camera_id,
        valid=valid,
        points_base=np.asarray(points_base, dtype=np.float32),
        colors_rgb=np.asarray(colors_rgb, dtype=np.uint8),
        preview_bgr=preview,
        status="ok" if valid else "no_object_points",
    )


def load_template_cloud() -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    template = np.load(TEMPLATE_PATH).astype(np.float64) * 0.001
    template_pcd = o3d.geometry.PointCloud()
    template_pcd.points = o3d.utility.Vector3dVector(template)
    template_pcd.paint_uniform_color([0.0, 0.0, 0.0])
    template_pcd.transform(DISPLAY_TRANSFORM)
    return template_pcd, np.asarray(template_pcd.points).copy()


def initialize_template_from_buffer(
    original_template_points: np.ndarray,
    extent_buffer: list[np.ndarray],
    target_centroid_display: np.ndarray,
) -> tuple[np.ndarray, float]:
    median_extent = np.median(np.asarray(extent_buffer, dtype=np.float64), axis=0)
    source_extent = compute_robust_extent(
        np.asarray(original_template_points, dtype=np.float32),
        low_percentile=0.0,
        high_percentile=100.0,
    ).astype(np.float64)

    valid_axes = source_extent > 1e-6
    if not np.any(valid_axes):
        return np.asarray(original_template_points, dtype=np.float64).copy(), 1.0

    raw_axis_scales = median_extent[valid_axes] / source_extent[valid_axes]
    uniform_scale = float(np.median(raw_axis_scales))
    uniform_scale = float(np.clip(uniform_scale, MIN_TEMPLATE_SCALE, MAX_TEMPLATE_SCALE))

    scaled_points = scale_template_points(original_template_points, uniform_scale)
    scaled_points = translate_points_to_centroid(scaled_points, target_centroid_display)
    return scaled_points.astype(np.float64), uniform_scale


def main() -> None:
    sensor_hub = None
    vis = None
    try:
        sensor_hub = DualSensorHub.from_config(CONFIG_PATH)
        sensor_hub.width = WIDTH
        sensor_hub.height = HEIGHT
        sensor_hub.fps = FPS
        sensor_hub.start()

        segmentation_engine = create_segmentation_engine()
        transform_chain = load_transform_chain(CONFIG_PATH)

        vis = o3d.visualization.Visualizer()
        vis.create_window("Real-Time 3D Cup", width=800, height=600)

        live_pcd = o3d.geometry.PointCloud()
        live_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        live_geometry_added = False

        template_pcd, original_template_points = load_template_cloud()
        vis.add_geometry(template_pcd)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
        mesh_frame.transform(DISPLAY_TRANSFORM)
        vis.add_geometry(mesh_frame)

        template_scaled = False
        tracked_cluster_centroid = None
        scale_extent_buffer: list[np.ndarray] = []
        frozen_template_scale = None
        print("Dual-camera shape fitting is running. Press q or ESC to quit.")

        while True:
            start = time.time()
            pair = sensor_hub.read_next_pair()
            processed_frames = [
                process_camera_frame(pair.cam0, 0, segmentation_engine, transform_chain),
                process_camera_frame(pair.cam1, 1, segmentation_engine, transform_chain),
            ]

            point_clouds = [frame.points_base for frame in processed_frames if frame.valid and len(frame.points_base) > 0]
            colors_rgb = [frame.colors_rgb for frame in processed_frames if frame.valid and len(frame.points_base) > 0]
            merge_status = "no valid camera clouds"

            if point_clouds:
                merged_points, _, merge_stats = merge_point_clouds(
                    point_clouds,
                    colors_rgb,
                    voxel_size_m=MERGE_VOXEL_SIZE_M,
                    outlier_method="statistical",
                    nb_neighbors=20,
                    std_ratio=2.0,
                    radius_m=0.01,
                    min_neighbors=8,
                )
                merged_points = limit_point_count(merged_points, ICP_MAX_POINTS)
                filtered_points, tracked_cluster_centroid = filter_pcd(merged_points, tracked_cluster_centroid)
                merge_status = (
                    f"merged: {merge_stats['raw_summary']['point_count']} -> "
                    f"{merge_stats['voxel_summary']['point_count']} -> "
                    f"{merge_stats['filtered_summary']['point_count']} | "
                    f"cluster: {len(filtered_points)}"
                )

                if len(filtered_points) > 0:
                    live_pcd.points = o3d.utility.Vector3dVector(filtered_points.astype(np.float64))
                    live_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                    live_pcd.transform(DISPLAY_TRANSFORM)

                    if not live_geometry_added:
                        vis.add_geometry(live_pcd)
                        live_geometry_added = True
                    else:
                        vis.update_geometry(live_pcd)

                    filtered_points_display = np.asarray(live_pcd.points)
                    filtered_centroid_display = np.mean(filtered_points_display, axis=0)

                    if not template_scaled:
                        robust_extent = compute_robust_extent(filtered_points.astype(np.float32))
                        if np.all(robust_extent > 1e-5):
                            scale_extent_buffer.append(robust_extent)
                        merge_status += f" | init_scale: {len(scale_extent_buffer)}/{SCALE_INIT_VALID_FRAMES}"

                        if len(scale_extent_buffer) >= SCALE_INIT_VALID_FRAMES:
                            initialized_template, frozen_template_scale = initialize_template_from_buffer(
                                original_template_points=original_template_points,
                                extent_buffer=scale_extent_buffer,
                                target_centroid_display=filtered_centroid_display,
                            )
                            template_pcd.points = o3d.utility.Vector3dVector(initialized_template)
                            vis.update_geometry(template_pcd)
                            template_scaled = True
                            merge_status += f" | scale={frozen_template_scale:.3f}"
                            vis.get_view_control().set_zoom(0.8)
                    else:
                        translated_template = translate_points_to_centroid(
                            np.asarray(template_pcd.points),
                            filtered_centroid_display,
                        )
                        template_pcd.points = o3d.utility.Vector3dVector(translated_template.astype(np.float64))
                        vis.update_geometry(template_pcd)
                        if frozen_template_scale is not None:
                            merge_status += f" | scale={frozen_template_scale:.3f} | translation-only"
                else:
                    tracked_cluster_centroid = None
            else:
                tracked_cluster_centroid = None

            latency = time.time() - start
            fps = 1.0 / latency if latency > 0 else 0.0

            cam0_overlay = processed_frames[0].preview_bgr
            if template_scaled:
                fitted_template_base = transform_points_display_to_base(np.asarray(template_pcd.points))
                fitted_template_cam0 = transform_points_base_to_cam0(fitted_template_base, transform_chain)
                cam0_overlay = render_projected_point_cloud_overlay(
                    base_frame_bgr=cam0_overlay,
                    point_clouds=[fitted_template_cam0],
                    intrinsics=pair.cam0.intrinsics,
                    point_colors_bgr=[(0, 0, 0)],
                    labels=["fitted template -> cam0"],
                    point_radius=1,
                    overlay_alpha=0.95,
                )

            preview = stack_previews([cam0_overlay, processed_frames[1].preview_bgr])
            preview = overlay_status(
                preview,
                [
                    f"pair dt: {pair.timestamp_delta_ms:+.1f} ms | fps: {fps:.1f}",
                    merge_status,
                    "q / ESC: quit",
                ],
            )
            cv2.imshow("Dual YOLOE Shape Fitting", preview)

            if not vis.poll_events():
                break
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Quit")
                break
    finally:
        if sensor_hub is not None:
            sensor_hub.stop()
        if vis is not None:
            vis.destroy_window()
        cv2.destroyAllWindows()
        if RESET_REALSENSE_ON_EXIT:
            reset_realsense()


if __name__ == "__main__":
    main()
