from pathlib import Path

import cv2 as cv
import numpy as np

from utils.depth_lifter import lift_mask_to_point_cloud

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def render_depth(depth_image_m, max_depth_m):
    max_depth_m = max(max_depth_m, 1e-6)
    clipped = np.clip(depth_image_m, 0.0, max_depth_m)
    scaled = (255.0 * clipped / max_depth_m).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)


def combine_instance_masks(instances, image_shape):
    combined_mask = np.zeros(image_shape[:2], dtype=bool)
    for instance in instances:
        combined_mask |= np.asarray(instance.mask, dtype=bool)
    return combined_mask


def colorize_selected_mask(frame_bgr, mask, overlay_color=(0, 255, 255), alpha=0.35):
    if mask is None or not np.any(mask):
        return frame_bgr

    tinted_frame = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_layer[:] = np.array(overlay_color, dtype=np.uint8)
    tinted_frame[mask] = cv.addWeighted(frame_bgr[mask], 1.0 - alpha, color_layer[mask], alpha, 0.0)
    return tinted_frame


def build_point_cloud_from_instances(
    color_image_bgr,
    depth_image_m,
    intrinsics,
    instances,
    stride,
    max_points,
    min_depth_m,
    max_depth_m,
):
    combined_mask = combine_instance_masks(instances, color_image_bgr.shape)
    points_xyz, valid_pixel_indices, depth_values_m = lift_mask_to_point_cloud(
        combined_mask,
        depth_image_m,
        intrinsics,
        stride=stride,
        max_points=max_points,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )

    if len(valid_pixel_indices) == 0:
        colors_rgb = np.empty((0, 3), dtype=np.uint8)
    else:
        sampled_colors_bgr = color_image_bgr[valid_pixel_indices[:, 1], valid_pixel_indices[:, 0]]
        colors_rgb = sampled_colors_bgr[:, ::-1].copy()

    return combined_mask, points_xyz, valid_pixel_indices, depth_values_m, colors_rgb


def render_mask_preview(mask):
    mask_uint8 = np.zeros(mask.shape, dtype=np.uint8)
    mask_uint8[mask] = 255
    return mask_uint8


def _filter_finite_point_cloud(points_xyz, colors_rgb):
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8).reshape((-1, 3))
    if len(points_xyz) != len(colors_rgb):
        raise ValueError('points_xyz and colors_rgb must have the same length')
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    finite_mask = np.isfinite(points_xyz).all(axis=1)
    return points_xyz[finite_mask].astype(np.float32), colors_rgb[finite_mask].astype(np.uint8)


def concatenate_point_clouds(point_clouds, colors_rgb_list):
    filtered_points = []
    filtered_colors = []
    for points_xyz, colors_rgb in zip(point_clouds, colors_rgb_list):
        current_points, current_colors = _filter_finite_point_cloud(points_xyz, colors_rgb)
        if len(current_points) == 0:
            continue
        filtered_points.append(current_points)
        filtered_colors.append(current_colors)

    if not filtered_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return np.concatenate(filtered_points, axis=0), np.concatenate(filtered_colors, axis=0)


def voxel_downsample_point_cloud(points_xyz, colors_rgb, voxel_size_m):
    points_xyz, colors_rgb = _filter_finite_point_cloud(points_xyz, colors_rgb)
    voxel_size_m = float(voxel_size_m)
    if len(points_xyz) == 0 or voxel_size_m <= 0.0:
        return points_xyz, colors_rgb

    voxel_indices = np.floor(points_xyz / voxel_size_m).astype(np.int32)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    voxel_count = int(np.max(inverse_indices)) + 1

    point_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    color_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    counts = np.zeros((voxel_count,), dtype=np.int32)

    np.add.at(point_sums, inverse_indices, points_xyz)
    np.add.at(color_sums, inverse_indices, colors_rgb.astype(np.float64))
    np.add.at(counts, inverse_indices, 1)

    downsampled_points = point_sums / np.maximum(counts[:, None], 1)
    downsampled_colors = np.clip(np.rint(color_sums / np.maximum(counts[:, None], 1)), 0, 255).astype(np.uint8)
    return downsampled_points.astype(np.float32), downsampled_colors


def remove_radius_outliers(points_xyz, colors_rgb, radius_m, min_neighbors):
    points_xyz, colors_rgb = _filter_finite_point_cloud(points_xyz, colors_rgb)
    if len(points_xyz) == 0 or radius_m <= 0.0 or min_neighbors <= 1 or cKDTree is None:
        return points_xyz, colors_rgb

    tree = cKDTree(points_xyz)
    neighborhoods = tree.query_ball_point(points_xyz, r=float(radius_m))
    keep_mask = np.fromiter((len(neighbors) >= int(min_neighbors) for neighbors in neighborhoods), dtype=bool, count=len(points_xyz))
    return points_xyz[keep_mask], colors_rgb[keep_mask]


def remove_statistical_outliers(points_xyz, colors_rgb, nb_neighbors, std_ratio):
    points_xyz, colors_rgb = _filter_finite_point_cloud(points_xyz, colors_rgb)
    if len(points_xyz) <= 2 or nb_neighbors < 1 or cKDTree is None:
        return points_xyz, colors_rgb

    neighbor_count = min(int(nb_neighbors) + 1, len(points_xyz))
    if neighbor_count <= 2:
        return points_xyz, colors_rgb

    tree = cKDTree(points_xyz)
    distances, _ = tree.query(points_xyz, k=neighbor_count)
    mean_neighbor_distance = np.mean(distances[:, 1:], axis=1)
    distance_mean = float(np.mean(mean_neighbor_distance))
    distance_std = float(np.std(mean_neighbor_distance))
    threshold = distance_mean + float(std_ratio) * max(distance_std, 1e-6)
    keep_mask = mean_neighbor_distance <= threshold
    return points_xyz[keep_mask], colors_rgb[keep_mask]


def summarize_point_cloud(points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
    if len(points_xyz) == 0:
        empty = np.zeros((3,), dtype=np.float32)
        return {
            'point_count': 0,
            'centroid_xyz': empty.copy(),
            'min_xyz': empty.copy(),
            'max_xyz': empty.copy(),
            'extent_xyz': empty.copy(),
        }

    min_xyz = np.min(points_xyz, axis=0).astype(np.float32)
    max_xyz = np.max(points_xyz, axis=0).astype(np.float32)
    return {
        'point_count': int(len(points_xyz)),
        'centroid_xyz': np.mean(points_xyz, axis=0).astype(np.float32),
        'min_xyz': min_xyz,
        'max_xyz': max_xyz,
        'extent_xyz': (max_xyz - min_xyz).astype(np.float32),
    }


def merge_point_clouds(
    point_clouds,
    colors_rgb_list,
    voxel_size_m=0.003,
    outlier_method='statistical',
    nb_neighbors=20,
    std_ratio=1.5,
    radius_m=0.01,
    min_neighbors=8,
):
    merged_points, merged_colors = concatenate_point_clouds(point_clouds, colors_rgb_list)
    raw_summary = summarize_point_cloud(merged_points)

    if voxel_size_m > 0.0:
        merged_points, merged_colors = voxel_downsample_point_cloud(
            merged_points,
            merged_colors,
            voxel_size_m=voxel_size_m,
        )
    voxel_summary = summarize_point_cloud(merged_points)

    normalized_method = str(outlier_method).strip().lower()
    if normalized_method == 'statistical':
        merged_points, merged_colors = remove_statistical_outliers(
            merged_points,
            merged_colors,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
    elif normalized_method == 'radius':
        merged_points, merged_colors = remove_radius_outliers(
            merged_points,
            merged_colors,
            radius_m=radius_m,
            min_neighbors=min_neighbors,
        )
    elif normalized_method not in {'none', 'off'}:
        raise ValueError(f'Unsupported outlier_method: {outlier_method}')

    filtered_summary = summarize_point_cloud(merged_points)
    merge_stats = {
        'voxel_size_m': float(voxel_size_m),
        'outlier_method': normalized_method,
        'raw_summary': raw_summary,
        'voxel_summary': voxel_summary,
        'filtered_summary': filtered_summary,
    }
    return merged_points, merged_colors, merge_stats


def _project_points_to_canvas(points_2d, canvas, point_colors):
    if len(points_2d) == 0:
        return

    for point_2d, point_color in zip(points_2d.astype(np.int32), point_colors):
        cv.circle(canvas, tuple(point_2d), 1, tuple(int(c) for c in point_color), -1, cv.LINE_AA)


def render_point_cloud_preview(points_xyz, colors_rgb, canvas_size=480):
    canvas = np.full((canvas_size, canvas_size * 2, 3), 18, dtype=np.uint8)
    if len(points_xyz) == 0:
        cv.putText(canvas, 'No point cloud', (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv.LINE_AA)
        return canvas

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)
    finite_mask = np.isfinite(points_xyz).all(axis=1)
    points_xyz = points_xyz[finite_mask]
    colors_rgb = colors_rgb[finite_mask]
    if len(points_xyz) == 0:
        cv.putText(canvas, 'No finite points', (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv.LINE_AA)
        return canvas

    colors_bgr = colors_rgb[:, ::-1]
    margin = 28
    left_width = canvas_size
    right_offset = canvas_size

    x_values = points_xyz[:, 0]
    y_values = points_xyz[:, 1]
    z_values = points_xyz[:, 2]

    def normalize_pair(horizontal_values, vertical_values, width):
        h_min = float(np.min(horizontal_values))
        h_max = float(np.max(horizontal_values))
        v_min = float(np.min(vertical_values))
        v_max = float(np.max(vertical_values))
        h_span = max(h_max - h_min, 1e-6)
        v_span = max(v_max - v_min, 1e-6)

        px = margin + (horizontal_values - h_min) / h_span * (width - 2 * margin)
        py = canvas_size - margin - (vertical_values - v_min) / v_span * (canvas_size - 2 * margin)
        return np.stack([px, py], axis=1)

    xz_points = normalize_pair(x_values, z_values, left_width)
    xy_points = normalize_pair(x_values, -y_values, left_width)
    xy_points[:, 0] += right_offset

    _project_points_to_canvas(xz_points, canvas, colors_bgr)
    _project_points_to_canvas(xy_points, canvas, colors_bgr)

    cv.putText(canvas, 'XZ top', (18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
    cv.putText(canvas, 'XY front', (right_offset + 18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
    cv.line(canvas, (right_offset, 0), (right_offset, canvas_size), (70, 70, 70), 1, cv.LINE_AA)
    return canvas


def render_multi_point_cloud_preview(point_clouds, colors_rgb_list, canvas_size=480, labels=None):
    canvas = np.full((canvas_size, canvas_size * 2, 3), 18, dtype=np.uint8)

    stacked_points = []
    stacked_colors = []
    for points_xyz, colors_rgb in zip(point_clouds, colors_rgb_list):
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)
        if len(points_xyz) == 0:
            continue
        finite_mask = np.isfinite(points_xyz).all(axis=1)
        points_xyz = points_xyz[finite_mask]
        colors_rgb = colors_rgb[finite_mask]
        if len(points_xyz) == 0:
            continue
        stacked_points.append(points_xyz)
        stacked_colors.append(colors_rgb)

    if not stacked_points:
        cv.putText(canvas, 'No aligned point clouds', (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv.LINE_AA)
        return canvas

    all_points = np.concatenate(stacked_points, axis=0)
    all_colors_bgr = np.concatenate([colors_rgb[:, ::-1] for colors_rgb in stacked_colors], axis=0)
    margin = 28
    left_width = canvas_size
    right_offset = canvas_size

    x_values = all_points[:, 0]
    y_values = all_points[:, 1]
    z_values = all_points[:, 2]

    def normalize_pair(horizontal_values, vertical_values, width):
        h_min = float(np.min(horizontal_values))
        h_max = float(np.max(horizontal_values))
        v_min = float(np.min(vertical_values))
        v_max = float(np.max(vertical_values))
        h_span = max(h_max - h_min, 1e-6)
        v_span = max(v_max - v_min, 1e-6)

        px = margin + (horizontal_values - h_min) / h_span * (width - 2 * margin)
        py = canvas_size - margin - (vertical_values - v_min) / v_span * (canvas_size - 2 * margin)
        return np.stack([px, py], axis=1)

    xz_points = normalize_pair(x_values, z_values, left_width)
    xy_points = normalize_pair(x_values, -y_values, left_width)
    xy_points[:, 0] += right_offset

    _project_points_to_canvas(xz_points, canvas, all_colors_bgr)
    _project_points_to_canvas(xy_points, canvas, all_colors_bgr)

    cv.putText(canvas, 'Aligned XZ top', (18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
    cv.putText(canvas, 'Aligned XY front', (right_offset + 18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
    cv.line(canvas, (right_offset, 0), (right_offset, canvas_size), (70, 70, 70), 1, cv.LINE_AA)

    for label_index, label in enumerate(labels or []):
        color_rgb = colors_rgb_list[label_index][0] if len(colors_rgb_list[label_index]) else np.array([255, 255, 255], dtype=np.uint8)
        color_bgr = tuple(int(channel) for channel in color_rgb[::-1])
        origin = (18, canvas_size - 18 - label_index * 24)
        cv.putText(canvas, label, origin, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(canvas, label, origin, cv.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 1, cv.LINE_AA)

    return canvas


def project_points_to_image_pixels(points_xyz, intrinsics, image_shape):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    finite_mask = np.isfinite(points_xyz).all(axis=1)
    positive_depth_mask = points_xyz[:, 2] > 1e-6
    valid_mask = finite_mask & positive_depth_mask
    if not np.any(valid_mask):
        return np.empty((0, 2), dtype=np.int32)

    valid_points = points_xyz[valid_mask]
    pixel_x = intrinsics['fx'] * valid_points[:, 0] / valid_points[:, 2] + intrinsics['cx']
    pixel_y = intrinsics['fy'] * valid_points[:, 1] / valid_points[:, 2] + intrinsics['cy']
    projected_pixels = np.stack([pixel_x, pixel_y], axis=1)

    image_height, image_width = image_shape[:2]
    inside_mask = (
        (projected_pixels[:, 0] >= 0.0)
        & (projected_pixels[:, 0] < float(image_width))
        & (projected_pixels[:, 1] >= 0.0)
        & (projected_pixels[:, 1] < float(image_height))
    )
    return projected_pixels[inside_mask].astype(np.int32)


def render_projected_point_cloud_overlay(
    base_frame_bgr,
    point_clouds,
    intrinsics,
    point_colors_bgr,
    labels=None,
    point_radius=1,
    overlay_alpha=0.78,
):
    overlay_frame = base_frame_bgr.copy()
    rendered_frame = base_frame_bgr.copy()
    labels = labels or []

    for cloud_index, (points_xyz, point_color_bgr) in enumerate(zip(point_clouds, point_colors_bgr)):
        projected_pixels = project_points_to_image_pixels(points_xyz, intrinsics, base_frame_bgr.shape)
        for pixel_x, pixel_y in projected_pixels:
            cv.circle(
                overlay_frame,
                (int(pixel_x), int(pixel_y)),
                point_radius,
                tuple(int(channel) for channel in point_color_bgr),
                -1,
                cv.LINE_AA,
            )

        if cloud_index < len(labels):
            legend_origin = (12, base_frame_bgr.shape[0] - 18 - cloud_index * 24)
            cv.putText(overlay_frame, labels[cloud_index], legend_origin, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(
                overlay_frame,
                labels[cloud_index],
                legend_origin,
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                tuple(int(channel) for channel in point_color_bgr),
                1,
                cv.LINE_AA,
            )

    cv.addWeighted(overlay_frame, overlay_alpha, rendered_frame, 1.0 - overlay_alpha, 0.0, dst=rendered_frame)
    return rendered_frame


def write_ascii_ply(output_path, points_xyz, colors_rgb):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)
    if len(points_xyz) != len(colors_rgb):
        raise ValueError('points_xyz and colors_rgb must have the same length')

    with open(output_path, 'w', encoding='ascii') as output_file:
        output_file.write('ply\n')
        output_file.write('format ascii 1.0\n')
        output_file.write(f'element vertex {len(points_xyz)}\n')
        output_file.write('property float x\n')
        output_file.write('property float y\n')
        output_file.write('property float z\n')
        output_file.write('property uchar red\n')
        output_file.write('property uchar green\n')
        output_file.write('property uchar blue\n')
        output_file.write('end_header\n')

        for point_xyz, color_rgb in zip(points_xyz, colors_rgb):
            output_file.write(
                f"{float(point_xyz[0])} {float(point_xyz[1])} {float(point_xyz[2])} "
                f"{int(color_rgb[0])} {int(color_rgb[1])} {int(color_rgb[2])}\n"
            )


def save_pointcloud_snapshot(save_dir, base_name, color_overlay, combined_mask, points_xyz, colors_rgb):
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f'{base_name}_points.npy'
    ply_path = output_dir / f'{base_name}.ply'
    overlay_path = output_dir / f'{base_name}_overlay.png'
    mask_path = output_dir / f'{base_name}_mask.png'

    np.save(npy_path, points_xyz.astype(np.float32))
    write_ascii_ply(ply_path, points_xyz, colors_rgb)
    cv.imwrite(str(overlay_path), color_overlay)
    cv.imwrite(str(mask_path), render_mask_preview(combined_mask))
    return npy_path, ply_path, overlay_path, mask_path
