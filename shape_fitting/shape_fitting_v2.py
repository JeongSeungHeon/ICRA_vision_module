from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


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
TEMPLATE_PATH = SCRIPT_DIR / "template.npy"
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
ICP_MAX_POINTS = 2000
DBSCAN_EPS_M = 0.02
DBSCAN_MIN_POINTS = 10
TRACKING_CLUSTER_MAX_JUMP_M = 0.08
SCALE_INIT_VALID_FRAMES = 8
ROBUST_EXTENT_LOW_PERCENTILE = 5.0
ROBUST_EXTENT_HIGH_PERCENTILE = 95.0
MIN_TEMPLATE_SCALE = 0.5 
MAX_TEMPLATE_SCALE = 1.8
ICP_DISTANCE_THRESHOLD_M = 0.09 # This is the max distance for a point to be considered an inlier in ICP. We can be a bit generous here since we have a good initial alignment and we want to allow for some noise and partial views, but it should still be tight enough to reject outliers.
ICP_MAX_ITERATIONS = 12 # Default is 20, but we can reduce it since we have a good initial guess from the centroid alignment and we want to save time.
ICP_MIN_FITNESS = 0.02
ICP_MAX_CENTROID_JUMP_M = TRACKING_CLUSTER_MAX_JUMP_M
ICP_CROP_ENABLED = True
ICP_CROP_HEIGHT_AXIS_INDEX = 2
ICP_CROP_TARGET_TOP_FRACTION = 0.10
ICP_CROP_SOURCE_TOP_FRACTION = None
ICP_CROP_MIN_POINTS_AFTER_CROP = 80
ICP_CROP_MIN_HEIGHT_EXTENT_M = 0.01
ICP_TRANSLATION_TOLERANCE_M = 1e-5
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


def points_to_point_cloud(points: np.ndarray, color: tuple[float, float, float] | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape((-1, 3)))
    if color is not None:
        pcd.paint_uniform_color(list(color))
    return pcd


def build_oriented_bbox(points: np.ndarray) -> o3d.geometry.OrientedBoundingBox | None:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return None

    if len(points) < 4:
        extent = np.maximum(compute_robust_extent(points, 0.0, 100.0).astype(np.float64), 1e-6)
        return o3d.geometry.OrientedBoundingBox(np.mean(points, axis=0), np.eye(3), extent)

    pcd = points_to_point_cloud(points)
    try:
        return pcd.get_oriented_bounding_box()
    except RuntimeError:
        extent = np.maximum(compute_robust_extent(points, 0.0, 100.0).astype(np.float64), 1e-6)
        return o3d.geometry.OrientedBoundingBox(np.mean(points, axis=0), np.eye(3), extent)


def compute_oriented_robust_extent(
    points: np.ndarray,
    obb_rotation: np.ndarray,
    obb_center: np.ndarray | None = None,
    low_percentile: float = ROBUST_EXTENT_LOW_PERCENTILE,
    high_percentile: float = ROBUST_EXTENT_HIGH_PERCENTILE,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return np.zeros((3,), dtype=np.float64)

    center = np.mean(points, axis=0) if obb_center is None else np.asarray(obb_center, dtype=np.float64).reshape(3)
    rotation = np.asarray(obb_rotation, dtype=np.float64).reshape((3, 3))
    points_local = (points - center.reshape(1, 3)) @ rotation
    lower = np.percentile(points_local, low_percentile, axis=0)
    upper = np.percentile(points_local, high_percentile, axis=0)
    return (upper - lower).astype(np.float64)


def estimate_uniform_scale_from_extents(source_extent: np.ndarray, target_extent: np.ndarray) -> float:
    source_extent = np.sort(np.asarray(source_extent, dtype=np.float64).reshape(3))
    target_extent = np.sort(np.asarray(target_extent, dtype=np.float64).reshape(3))
    valid_axes = np.logical_and(source_extent > 1e-6, target_extent > 1e-6)
    if not np.any(valid_axes):
        return 1.0

    raw_axis_scales = target_extent[valid_axes] / source_extent[valid_axes]
    uniform_scale = float(np.median(raw_axis_scales))
    return float(np.clip(uniform_scale, MIN_TEMPLATE_SCALE, MAX_TEMPLATE_SCALE))


def apply_similarity_pose(
    points: np.ndarray,
    uniform_scale: float,
    rotation: np.ndarray,
    target_center: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return points

    centroid = np.mean(points, axis=0, keepdims=True)
    centered = points - centroid
    scaled = centered * float(uniform_scale)
    rotated = scaled @ np.asarray(rotation, dtype=np.float64).reshape((3, 3)).T
    return rotated + np.asarray(target_center, dtype=np.float64).reshape((1, 3))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return points

    transform = np.asarray(transform, dtype=np.float64).reshape((4, 4))
    homogeneous = np.ones((len(points), 4), dtype=np.float64)
    homogeneous[:, :3] = points
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def filter_points_by_local_height(
    points: np.ndarray,
    *,
    height_axis_index: int = ICP_CROP_HEIGHT_AXIS_INDEX,
    remove_top_fraction: float = ICP_CROP_TARGET_TOP_FRACTION,
    min_points_after_crop: int = ICP_CROP_MIN_POINTS_AFTER_CROP,
    min_height_extent_m: float = ICP_CROP_MIN_HEIGHT_EXTENT_M,
) -> tuple[np.ndarray, dict[str, float | int | bool | None]]:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    debug = {
        "crop_enabled": False,
        "cropped_point_count": int(len(points)),
        "crop_threshold_z": None,
        "height_extent_m": 0.0,
    }
    if len(points) == 0:
        return points, debug

    axis_index = int(height_axis_index)
    if axis_index < 0 or axis_index >= points.shape[1]:
        return points, debug

    remove_fraction = float(remove_top_fraction)
    if remove_fraction <= 0.0 or remove_fraction >= 1.0:
        return points, debug

    axis_values = points[:, axis_index]
    axis_min = float(np.min(axis_values))
    axis_max = float(np.max(axis_values))
    height_extent = axis_max - axis_min
    debug["height_extent_m"] = height_extent
    if height_extent < float(min_height_extent_m):
        return points, debug

    crop_threshold = axis_min + (1.0 - remove_fraction) * height_extent
    kept_mask = axis_values <= crop_threshold
    filtered_points = points[kept_mask]
    if len(filtered_points) < int(min_points_after_crop):
        return points, debug

    debug["crop_enabled"] = True
    debug["cropped_point_count"] = int(len(filtered_points))
    debug["crop_threshold_z"] = crop_threshold
    return filtered_points, debug


def subsample_points_for_icp(points: np.ndarray, max_points: int = ICP_MAX_POINTS) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if max_points <= 0 or len(points) <= max_points:
        return points

    sample_indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[sample_indices]


def extract_points_array(points_or_pcd: np.ndarray | o3d.geometry.PointCloud) -> np.ndarray:
    if isinstance(points_or_pcd, o3d.geometry.PointCloud):
        return np.asarray(points_or_pcd.points, dtype=np.float64).reshape((-1, 3))
    return np.asarray(points_or_pcd, dtype=np.float64).reshape((-1, 3))


def query_nearest_neighbors_batched(
    query_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_points = np.asarray(query_points, dtype=np.float64).reshape((-1, 3))
    target_points = np.asarray(target_points, dtype=np.float64).reshape((-1, 3))
    if len(query_points) == 0 or len(target_points) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)

    if cKDTree is not None:
        tree = cKDTree(target_points)
        distances, indices = tree.query(query_points, k=1)
        return (
            np.asarray(distances, dtype=np.float64).reshape(-1),
            np.asarray(indices, dtype=np.int64).reshape(-1),
        )

    target_tree = o3d.geometry.KDTreeFlann(points_to_point_cloud(target_points))
    indices = np.empty((len(query_points),), dtype=np.int64)
    distances = np.empty((len(query_points),), dtype=np.float64)
    for point_index, point in enumerate(query_points):
        _, nearest_indices, distance2 = target_tree.search_knn_vector_3d(point, 1)
        if nearest_indices:
            indices[point_index] = int(nearest_indices[0])
            distances[point_index] = float(np.sqrt(distance2[0]))
        else:
            indices[point_index] = 0
            distances[point_index] = float("inf")
    return distances, indices


def filter_pcd(points: np.ndarray, prev_centroid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return points, None

    pcd = points_to_point_cloud(points)

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


def run_open3d_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray | None = None,
    *,
    target_crop_top_fraction: float | None = ICP_CROP_TARGET_TOP_FRACTION if ICP_CROP_ENABLED else None,
    source_crop_top_fraction: float | None = ICP_CROP_SOURCE_TOP_FRACTION if ICP_CROP_ENABLED else None,
    height_axis_index: int = ICP_CROP_HEIGHT_AXIS_INDEX,
) -> tuple[np.ndarray, float, float]:
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    if len(source_points) == 0 or len(target_points) == 0:
        return np.eye(4, dtype=np.float64), 0.0, float("inf")

    source_fit_points = np.asarray(source_points, dtype=np.float64)
    target_fit_points = np.asarray(target_points, dtype=np.float64)
    if source_crop_top_fraction is not None:
        source_fit_points, _ = filter_points_by_local_height(
            source_fit_points,
            height_axis_index=height_axis_index,
            remove_top_fraction=float(source_crop_top_fraction),
        )
    if target_crop_top_fraction is not None:
        target_fit_points, _ = filter_points_by_local_height(
            target_fit_points,
            height_axis_index=height_axis_index,
            remove_top_fraction=float(target_crop_top_fraction),
        )
    if len(source_fit_points) == 0 or len(target_fit_points) == 0:
        return np.eye(4, dtype=np.float64), 0.0, float("inf")

    source_fit_pcd = points_to_point_cloud(source_fit_points)
    target_fit_pcd = points_to_point_cloud(target_fit_points)

    if init_transform is None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = np.mean(target_fit_points, axis=0) - np.mean(source_fit_points, axis=0)
    else:
        transform = np.array(init_transform, dtype=np.float64, copy=True)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_fit_pcd,
        target_fit_pcd,
        ICP_DISTANCE_THRESHOLD_M,
        transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS),
    )
    return reg_p2p.transformation, float(reg_p2p.fitness), float(reg_p2p.inlier_rmse)


def run_translation_only_icp(
    source_pcd: np.ndarray | o3d.geometry.PointCloud,
    target_pcd: np.ndarray | o3d.geometry.PointCloud,
    init_transform: np.ndarray | None = None,
    *,
    target_crop_top_fraction: float | None = ICP_CROP_TARGET_TOP_FRACTION if ICP_CROP_ENABLED else None,
    source_crop_top_fraction: float | None = ICP_CROP_SOURCE_TOP_FRACTION if ICP_CROP_ENABLED else None,
    height_axis_index: int = ICP_CROP_HEIGHT_AXIS_INDEX,
) -> tuple[np.ndarray, float, float, float, int, int, int]:
    source_points = extract_points_array(source_pcd)
    target_points = extract_points_array(target_pcd)
    if len(source_points) == 0 or len(target_points) == 0:
        return np.eye(4, dtype=np.float64), 0.0, float("inf"), 0.0, 0, 0, 0

    source_fit_points = source_points.copy()
    target_fit_points = target_points.copy()
    if source_crop_top_fraction is not None:
        source_fit_points, _ = filter_points_by_local_height(
            source_fit_points,
            height_axis_index=height_axis_index,
            remove_top_fraction=float(source_crop_top_fraction),
        )
    if target_crop_top_fraction is not None:
        target_fit_points, _ = filter_points_by_local_height(
            target_fit_points,
            height_axis_index=height_axis_index,
            remove_top_fraction=float(target_crop_top_fraction),
        )
    source_fit_points = subsample_points_for_icp(source_fit_points, ICP_MAX_POINTS)
    target_fit_points = subsample_points_for_icp(target_fit_points, ICP_MAX_POINTS)
    if len(source_fit_points) == 0 or len(target_fit_points) == 0:
        return np.eye(4, dtype=np.float64), 0.0, float("inf"), 0.0, len(source_fit_points), len(target_fit_points), 0

    total_transform = np.eye(4, dtype=np.float64)
    if init_transform is None:
        total_transform[:3, 3] = np.mean(target_fit_points, axis=0) - np.mean(source_fit_points, axis=0)
    else:
        total_transform[:3, 3] = np.asarray(init_transform, dtype=np.float64).reshape((4, 4))[:3, 3]

    transformed_source = source_fit_points + total_transform[:3, 3].reshape(1, 3)
    last_fitness = 0.0
    last_rmse = float("inf")
    iterations_used = 0

    for iteration_index in range(ICP_MAX_ITERATIONS):
        nearest_distances, nearest_indices = query_nearest_neighbors_batched(transformed_source, target_fit_points)
        if len(nearest_indices) == 0:
            return (
                total_transform,
                0.0,
                float("inf"),
                float(np.linalg.norm(total_transform[:3, 3])),
                len(source_fit_points),
                len(target_fit_points),
                iterations_used,
            )

        nearest_targets = target_fit_points[nearest_indices]
        translation_delta = np.mean(nearest_targets - transformed_source, axis=0)

        delta_transform = np.eye(4, dtype=np.float64)
        delta_transform[:3, 3] = translation_delta
        total_transform = delta_transform @ total_transform
        transformed_source = transformed_source + translation_delta.reshape(1, 3)

        inlier_mask = nearest_distances <= ICP_DISTANCE_THRESHOLD_M
        last_fitness = float(np.count_nonzero(inlier_mask) / max(1, len(nearest_distances)))
        if np.any(inlier_mask):
            last_rmse = float(np.sqrt(np.mean(np.square(nearest_distances[inlier_mask]))))
        else:
            last_rmse = float(np.sqrt(np.mean(np.square(nearest_distances))))
        iterations_used = iteration_index + 1
        if np.linalg.norm(translation_delta) <= ICP_TRANSLATION_TOLERANCE_M:
            break

    translation_distance_m = float(np.linalg.norm(total_transform[:3, 3]))
    return (
        total_transform,
        last_fitness,
        last_rmse,
        translation_distance_m,
        len(source_fit_points),
        len(target_fit_points),
        iterations_used,
    )


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


def initialize_template_from_obb_buffer(
    original_template_points: np.ndarray,
    obb_extent_buffer: list[np.ndarray],
    target_centroid_display: np.ndarray,
    target_points_display: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, float, float, float, int, int, int]:
    original_template_points = np.asarray(original_template_points, dtype=np.float64).reshape((-1, 3))
    target_centroid_display = np.asarray(target_centroid_display, dtype=np.float64).reshape(3)
    if len(original_template_points) == 0:
        return original_template_points.copy(), 1.0, np.eye(3), 0.0, float("inf"), float("nan"), 0, 0, 0

    template_obb = build_oriented_bbox(original_template_points)
    if template_obb is None or not obb_extent_buffer:
        return original_template_points.copy(), 1.0, np.eye(3), 0.0, float("inf"), float("nan"), 0, 0, 0

    median_target_extent = np.median(np.asarray(obb_extent_buffer, dtype=np.float64), axis=0)
    template_extent = compute_oriented_robust_extent(
        original_template_points,
        template_obb.R,
        template_obb.center,
        low_percentile=0.0,
        high_percentile=100.0,
    )

    uniform_scale = estimate_uniform_scale_from_extents(template_extent, median_target_extent)
    initialized_points = apply_similarity_pose(
        original_template_points,
        uniform_scale=uniform_scale,
        rotation=np.eye(3, dtype=np.float64),
        target_center=target_centroid_display,
    )
    icp_fitness = float("nan")
    icp_rmse = float("nan")
    icp_time_ms = float("nan")
    icp_source_points = 0
    icp_target_points = 0
    icp_iterations_used = 0
    if target_points_display is not None:
        target_points_display = np.asarray(target_points_display, dtype=np.float64).reshape((-1, 3))
        if len(target_points_display) > 0:
            icp_start = time.time()
            (
                icp_transform,
                icp_fitness,
                icp_rmse,
                _,
                icp_source_points,
                icp_target_points,
                icp_iterations_used,
            ) = run_translation_only_icp(
                initialized_points,
                target_points_display,
            )
            icp_time_ms = (time.time() - icp_start) * 1000.0
            initialized_points = transform_points(initialized_points, icp_transform)
    return (
        initialized_points.astype(np.float64),
        uniform_scale,
        np.eye(3, dtype=np.float64),
        icp_fitness,
        icp_rmse,
        icp_time_ms,
        icp_source_points,
        icp_target_points,
        icp_iterations_used,
    )


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
        template_obb = build_oriented_bbox(original_template_points)
        template_scale_extent = (
            compute_oriented_robust_extent(
                original_template_points,
                template_obb.R,
                template_obb.center,
                low_percentile=0.0,
                high_percentile=100.0,
            )
            if template_obb is not None
            else compute_robust_extent(original_template_points, 0.0, 100.0).astype(np.float64)
        )
        template_geometry_added = False

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
        mesh_frame.transform(DISPLAY_TRANSFORM)
        vis.add_geometry(mesh_frame)

        template_initialized = False
        tracked_cluster_centroid = None
        scale_obb_extent_buffer: list[np.ndarray] = []
        frozen_template_scale = None
        frozen_template_rotation = np.eye(3, dtype=np.float64)
        last_icp_fitness = 0.0
        last_icp_rmse = float("inf")
        last_icp_translation_m = 0.0
        last_icp_time_ms = float("nan")
        last_icp_source_points = 0
        last_icp_target_points = 0
        last_icp_iterations_used = 0
        tracking_mode = "init_pending"

        print("Dual-camera shape fitting v2 is running. Press q or ESC to quit.")

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
                    target_obb = build_oriented_bbox(filtered_points_display)

                    if not template_initialized:
                        if target_obb is not None:
                            robust_extent = compute_oriented_robust_extent(
                                filtered_points_display,
                                target_obb.R,
                                target_obb.center,
                            )
                            if np.all(robust_extent > 1e-5):
                                scale_obb_extent_buffer.append(robust_extent)
                        merge_status += f" | init_scale: {len(scale_obb_extent_buffer)}/{SCALE_INIT_VALID_FRAMES}"
                        tracking_mode = "init_pending"

                        if len(scale_obb_extent_buffer) >= SCALE_INIT_VALID_FRAMES:
                            (
                                initialized_template,
                                frozen_template_scale,
                                frozen_template_rotation,
                                last_icp_fitness,
                                last_icp_rmse,
                                last_icp_time_ms,
                                last_icp_source_points,
                                last_icp_target_points,
                                last_icp_iterations_used,
                            ) = initialize_template_from_obb_buffer(
                                original_template_points=original_template_points,
                                obb_extent_buffer=scale_obb_extent_buffer,
                                target_centroid_display=filtered_centroid_display,
                                target_points_display=filtered_points_display,
                            )
                            template_pcd.points = o3d.utility.Vector3dVector(initialized_template)
                            if not template_geometry_added:
                                vis.add_geometry(template_pcd)
                                template_geometry_added = True
                            else:
                                vis.update_geometry(template_pcd)
                            template_initialized = True
                            tracking_mode = "init"
                            if frozen_template_scale is not None:
                                merge_status += f" | scale={frozen_template_scale:.3f}"
                            vis.get_view_control().set_zoom(0.8)
                    else:
                        current_template_points = np.asarray(template_pcd.points)
                        tracked_scale = frozen_template_scale if frozen_template_scale is not None else 1.0

                        current_template_centroid = np.mean(current_template_points, axis=0)
                        scaled_points = apply_similarity_pose(
                            original_template_points,
                            uniform_scale=tracked_scale,
                            rotation=frozen_template_rotation,
                            target_center=current_template_centroid,
                        )
                        icp_start = time.time()
                        (
                            icp_transform,
                            last_icp_fitness,
                            last_icp_rmse,
                            last_icp_translation_m,
                            last_icp_source_points,
                            last_icp_target_points,
                            last_icp_iterations_used,
                        ) = run_translation_only_icp(
                            scaled_points,
                            filtered_points_display,
                        )
                        last_icp_time_ms = (time.time() - icp_start) * 1000.0
                        proposed_points = transform_points(scaled_points, icp_transform)
                        template_pcd.points = o3d.utility.Vector3dVector(proposed_points.astype(np.float64))
                        vis.update_geometry(template_pcd)
                        tracking_mode = "translation_icp"
                else:
                    tracked_cluster_centroid = None
                    last_icp_translation_m = 0.0
                    last_icp_time_ms = float("nan")
                    last_icp_source_points = 0
                    last_icp_target_points = 0
                    last_icp_iterations_used = 0
                    tracking_mode = "hold"
            else:
                tracked_cluster_centroid = None
                last_icp_translation_m = 0.0
                last_icp_time_ms = float("nan")
                last_icp_source_points = 0
                last_icp_target_points = 0
                last_icp_iterations_used = 0
                tracking_mode = "hold"

            latency = time.time() - start
            fps = 1.0 / latency if latency > 0 else 0.0
            icp_translation_speed_mps = last_icp_translation_m / latency if latency > 0 else 0.0
            icp_fps = 1000.0 / last_icp_time_ms if np.isfinite(last_icp_time_ms) and last_icp_time_ms > 0.0 else 0.0

            if template_initialized:
                print(
                    "[ICP] "
                    f"mode={tracking_mode} "
                    f"icp_time_ms={last_icp_time_ms:.2f} "
                    f"icp_fps={icp_fps:.2f} "
                    f"src_pts={last_icp_source_points} "
                    f"tgt_pts={last_icp_target_points} "
                    f"iters={last_icp_iterations_used} "
                    f"translation_mm={last_icp_translation_m * 1000.0:.2f} "
                    f"speed_mm_s={icp_translation_speed_mps * 1000.0:.2f} "
                    f"fitness={last_icp_fitness:.3f} "
                    f"rmse={last_icp_rmse:.4f}"
                )

            cam0_overlay = processed_frames[0].preview_bgr
            if template_initialized:
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

            scale_text = "pending" if frozen_template_scale is None else f"{frozen_template_scale:.3f}"
            fitness_text = "n/a" if not np.isfinite(last_icp_fitness) else f"{last_icp_fitness:.3f}"
            rmse_text = "n/a" if not np.isfinite(last_icp_rmse) else f"{last_icp_rmse:.4f}"
            translation_text = f"{last_icp_translation_m * 1000.0:.1f} mm"
            speed_text = f"{icp_translation_speed_mps * 1000.0:.1f} mm/s"
            icp_time_text = "n/a" if not np.isfinite(last_icp_time_ms) else f"{last_icp_time_ms:.1f} ms"
            icp_fps_text = "n/a" if not np.isfinite(last_icp_time_ms) or last_icp_time_ms <= 0.0 else f"{icp_fps:.1f}"
            icp_points_text = f"{last_icp_source_points}/{last_icp_target_points}"
            icp_iters_text = f"{last_icp_iterations_used}"

            preview = stack_previews([cam0_overlay, processed_frames[1].preview_bgr])
            preview = overlay_status(
                preview,
                [
                    f"pair dt: {pair.timestamp_delta_ms:+.1f} ms | fps: {fps:.1f}",
                    merge_status,
                    f"tracking: {tracking_mode} | scale: {scale_text} | fitness: {fitness_text} | rmse: {rmse_text}",
                    f"icp: {icp_time_text} | icp fps: {icp_fps_text} | pts s/t: {icp_points_text} | iters: {icp_iters_text}",
                    f"translation: {translation_text} | speed: {speed_text}",
                    "q / ESC: quit",
                ],
            )
            cv2.imshow("Dual YOLOE Shape Fitting v2", preview)

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
