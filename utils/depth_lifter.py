import numpy as np


def _build_valid_mask(keypoints_2d):
    return np.logical_and(keypoints_2d[:, 0] >= 0, keypoints_2d[:, 1] >= 0)


def _normalize_pixel_indices(pixel_indices):
    pixel_indices = np.asarray(pixel_indices, dtype=np.int32)
    if pixel_indices.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return pixel_indices.reshape((-1, 2))


def sample_depth_at_keypoint(depth_image_m, u, v, patch_radius=2, min_depth_m=0.1, max_depth_m=1.2):
    height, width = depth_image_m.shape[:2]
    u = int(round(u))
    v = int(round(v))

    if u < 0 or u >= width or v < 0 or v >= height:
        return np.nan, 0.0

    x0 = max(0, u - patch_radius)
    x1 = min(width, u + patch_radius + 1)
    y0 = max(0, v - patch_radius)
    y1 = min(height, v + patch_radius + 1)

    patch = depth_image_m[y0:y1, x0:x1].reshape(-1)
    valid_patch = patch[np.isfinite(patch)]
    valid_patch = valid_patch[(valid_patch >= min_depth_m) & (valid_patch <= max_depth_m)]

    if valid_patch.size == 0:
        return np.nan, 0.0

    depth_m = float(np.median(valid_patch))
    patch_std = float(np.std(valid_patch))
    patch_fill_ratio = float(valid_patch.size) / float(patch.size)
    stability_score = 1.0 / (1.0 + 25.0 * patch_std)
    quality = float(np.clip(patch_fill_ratio * stability_score, 0.0, 1.0))
    return depth_m, quality


def deproject_pixel_to_point(u, v, depth_m, intrinsics):
    if not np.isfinite(depth_m):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    x = (float(u) - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (float(v) - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return np.array([x, y, depth_m], dtype=np.float32)


def mask_to_pixel_indices(mask, stride=1, max_points=None):
    # segmentation mask 내부 픽셀을 (u, v) 좌표 목록으로 변환한다.
    if stride < 1:
        raise ValueError("stride must be >= 1")

    mask = np.asarray(mask, dtype=bool)
    pixel_locations_yx = np.argwhere(mask)
    if pixel_locations_yx.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    if stride > 1:
        pixel_locations_yx = pixel_locations_yx[::stride]

    if max_points is not None and max_points > 0 and len(pixel_locations_yx) > max_points:
        sample_indices = np.linspace(0, len(pixel_locations_yx) - 1, max_points, dtype=np.int32)
        pixel_locations_yx = pixel_locations_yx[sample_indices]

    # np.argwhere는 (v, u) 순서를 반환하므로 (u, v) 순서로 맞춘다.
    return pixel_locations_yx[:, [1, 0]].astype(np.int32)


def filter_depth_pixels(depth_image_m, pixel_indices, min_depth_m=0.1, max_depth_m=1.2):
    # mask 픽셀 중에서 유효한 depth를 가진 점만 남긴다.
    pixel_indices = _normalize_pixel_indices(pixel_indices)
    if pixel_indices.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32), np.zeros((0,), dtype=bool)

    depth_image_m = np.asarray(depth_image_m, dtype=np.float32)
    image_height, image_width = depth_image_m.shape[:2]

    u_coords = pixel_indices[:, 0]
    v_coords = pixel_indices[:, 1]
    inside_mask = (
        (u_coords >= 0)
        & (u_coords < image_width)
        & (v_coords >= 0)
        & (v_coords < image_height)
    )
    if not np.any(inside_mask):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32), inside_mask

    inside_pixels = pixel_indices[inside_mask]
    depth_values_m = depth_image_m[inside_pixels[:, 1], inside_pixels[:, 0]]
    valid_depth_mask = np.isfinite(depth_values_m)
    valid_depth_mask &= depth_values_m >= float(min_depth_m)
    valid_depth_mask &= depth_values_m <= float(max_depth_m)

    return (
        inside_pixels[valid_depth_mask].astype(np.int32),
        depth_values_m[valid_depth_mask].astype(np.float32),
        inside_mask,
    )


def deproject_pixels_to_points(pixel_indices, depth_values_m, intrinsics):
    # 여러 픽셀을 한 번에 카메라 좌표계 3D 점들로 변환한다.
    pixel_indices = _normalize_pixel_indices(pixel_indices)
    depth_values_m = np.asarray(depth_values_m, dtype=np.float32).reshape((-1,))

    if len(pixel_indices) != len(depth_values_m):
        raise ValueError("pixel_indices and depth_values_m must have the same length")
    if len(pixel_indices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    u_coords = pixel_indices[:, 0].astype(np.float32)
    v_coords = pixel_indices[:, 1].astype(np.float32)
    x_coords = (u_coords - float(intrinsics["cx"])) * depth_values_m / float(intrinsics["fx"])
    y_coords = (v_coords - float(intrinsics["cy"])) * depth_values_m / float(intrinsics["fy"])
    return np.stack([x_coords, y_coords, depth_values_m], axis=1).astype(np.float32)


def lift_mask_to_point_cloud(mask, depth_image_m, intrinsics, stride=1, max_points=None, min_depth_m=0.1, max_depth_m=1.2):
    # segmentation mask에 대응하는 depth만 골라 카메라 local point cloud를 생성한다.
    pixel_indices = mask_to_pixel_indices(mask, stride=stride, max_points=max_points)
    valid_pixel_indices, depth_values_m, _ = filter_depth_pixels(
        depth_image_m,
        pixel_indices,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    points_xyz = deproject_pixels_to_points(valid_pixel_indices, depth_values_m, intrinsics)
    return points_xyz, valid_pixel_indices, depth_values_m


def lift_hand_pose_3d(keypoints_2d, depth_image_m, intrinsics, patch_radius=2, min_depth_m=0.1, max_depth_m=1.2):
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32).reshape((21, 2))
    points_3d = np.full((21, 3), np.nan, dtype=np.float32)
    depth_values_m = np.full(21, np.nan, dtype=np.float32)
    quality_scores = np.zeros(21, dtype=np.float32)
    valid_mask_2d = _build_valid_mask(keypoints_2d)
    valid_mask_3d = np.zeros(21, dtype=bool)

    for point_index, is_valid in enumerate(valid_mask_2d):
        if not is_valid:
            continue

        u, v = keypoints_2d[point_index]
        depth_m, quality = sample_depth_at_keypoint(
            depth_image_m,
            u,
            v,
            patch_radius=patch_radius,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        if not np.isfinite(depth_m):
            continue

        points_3d[point_index] = deproject_pixel_to_point(u, v, depth_m, intrinsics)
        depth_values_m[point_index] = depth_m
        quality_scores[point_index] = quality
        valid_mask_3d[point_index] = True

    return points_3d, valid_mask_3d, depth_values_m, quality_scores
