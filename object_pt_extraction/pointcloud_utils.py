from pathlib import Path

import cv2 as cv
import numpy as np

from utils.depth_lifter import lift_mask_to_point_cloud


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


def _project_points_to_canvas(points_2d, canvas, point_colors):
    if len(points_2d) == 0:
        return

    for point_2d, point_color in zip(points_2d.astype(np.int32), point_colors):
        cv.circle(canvas, tuple(point_2d), 1, tuple(int(c) for c in point_color), -1, cv.LINE_AA)


def render_point_cloud_preview(points_xyz, colors_rgb, canvas_size=480):
    canvas = np.full((canvas_size, canvas_size * 2, 3), 18, dtype=np.uint8)
    if len(points_xyz) == 0:
        cv.putText(canvas, "No point cloud", (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv.LINE_AA)
        return canvas

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)
    finite_mask = np.isfinite(points_xyz).all(axis=1)
    points_xyz = points_xyz[finite_mask]
    colors_rgb = colors_rgb[finite_mask]
    if len(points_xyz) == 0:
        cv.putText(canvas, "No finite points", (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv.LINE_AA)
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

    cv.putText(canvas, "XZ top", (18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
    cv.putText(canvas, "XY front", (right_offset + 18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
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
        cv.putText(canvas, "No aligned point clouds", (18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv.LINE_AA)
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

    cv.putText(canvas, "Aligned XZ top", (18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
    cv.putText(canvas, "Aligned XY front", (right_offset + 18, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv.LINE_AA)
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
    pixel_x = intrinsics["fx"] * valid_points[:, 0] / valid_points[:, 2] + intrinsics["cx"]
    pixel_y = intrinsics["fy"] * valid_points[:, 1] / valid_points[:, 2] + intrinsics["cy"]
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
        raise ValueError("points_xyz and colors_rgb must have the same length")

    with open(output_path, "w", encoding="ascii") as output_file:
        output_file.write("ply\n")
        output_file.write("format ascii 1.0\n")
        output_file.write(f"element vertex {len(points_xyz)}\n")
        output_file.write("property float x\n")
        output_file.write("property float y\n")
        output_file.write("property float z\n")
        output_file.write("property uchar red\n")
        output_file.write("property uchar green\n")
        output_file.write("property uchar blue\n")
        output_file.write("end_header\n")

        for point_xyz, color_rgb in zip(points_xyz, colors_rgb):
            output_file.write(
                f"{float(point_xyz[0])} {float(point_xyz[1])} {float(point_xyz[2])} "
                f"{int(color_rgb[0])} {int(color_rgb[1])} {int(color_rgb[2])}\n"
            )


def save_pointcloud_snapshot(save_dir, base_name, color_overlay, combined_mask, points_xyz, colors_rgb):
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f"{base_name}_points.npy"
    ply_path = output_dir / f"{base_name}.ply"
    overlay_path = output_dir / f"{base_name}_overlay.png"
    mask_path = output_dir / f"{base_name}_mask.png"

    np.save(npy_path, points_xyz.astype(np.float32))
    write_ascii_ply(ply_path, points_xyz, colors_rgb)
    cv.imwrite(str(overlay_path), color_overlay)
    cv.imwrite(str(mask_path), render_mask_preview(combined_mask))
    return npy_path, ply_path, overlay_path, mask_path
