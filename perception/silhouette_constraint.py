"""Lightweight multi-view silhouette scoring for fitted object templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class SilhouetteObservation:
    camera_id: int
    mask: np.ndarray
    intrinsics: dict
    t_base_cam: np.ndarray
    image_shape: tuple[int, int]
    weight: float = 1.0


@dataclass(frozen=True)
class SilhouetteScore:
    valid_camera_count: int
    loss_iou: float
    outside_loss: float
    mean_iou: float
    per_camera_iou: dict[int, float] = field(default_factory=dict)
    per_camera_outside_loss: dict[int, float] = field(default_factory=dict)
    rendered_pixel_count: int = 0

    @property
    def valid(self) -> bool:
        return self.valid_camera_count > 0


def invert_transform(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(T, dtype=np.float64).reshape((4, 4)))


def transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    transform = np.asarray(T, dtype=np.float64).reshape((4, 4))
    homogeneous = np.ones((len(points), 4), dtype=np.float64)
    homogeneous[:, :3] = points
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def _intrinsic_value(intrinsics: dict[str, Any], key: str) -> float:
    value = float(intrinsics[key])
    if not np.isfinite(value):
        raise ValueError(f"Camera intrinsic {key} is not finite: {value}")
    return value


def project_points_to_image(
    points_cam: np.ndarray,
    intrinsics: dict[str, Any],
    image_shape: tuple[int, int],
) -> np.ndarray:
    points = np.asarray(points_cam, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        return np.empty((0, 2), dtype=np.float64)

    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    points = points[valid]
    z = points[:, 2]
    fx = _intrinsic_value(intrinsics, "fx")
    fy = _intrinsic_value(intrinsics, "fy")
    cx = _intrinsic_value(intrinsics, "cx")
    cy = _intrinsic_value(intrinsics, "cy")

    u = (points[:, 0] * fx / z) + cx
    v = (points[:, 1] * fy / z) + cy
    finite = np.isfinite(u) & np.isfinite(v)
    if not np.any(finite):
        return np.empty((0, 2), dtype=np.float64)

    u = u[finite]
    v = v[finite]
    in_bounds = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    if not np.any(in_bounds):
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([u[in_bounds], v[in_bounds]], axis=1)


def rasterize_projected_template(
    projected_uv: np.ndarray,
    image_shape: tuple[int, int],
    point_radius_px: int = 2,
    close_kernel_px: int = 5,
    dilate_px: int = 1,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((max(height, 0), max(width, 0)), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return mask.astype(bool)

    uv = np.asarray(projected_uv, dtype=np.float64).reshape((-1, 2))
    if len(uv) == 0:
        return mask.astype(bool)

    pixels = np.rint(uv).astype(np.int32)
    in_bounds = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[in_bounds]
    if len(pixels) == 0:
        return mask.astype(bool)

    radius = max(int(point_radius_px), 0)
    if radius <= 0:
        mask[pixels[:, 1], pixels[:, 0]] = 255
    else:
        for u, v in pixels:
            cv.circle(mask, (int(u), int(v)), radius, 255, thickness=-1)

    close_size = max(int(close_kernel_px), 0)
    if close_size > 1:
        if close_size % 2 == 0:
            close_size += 1
        kernel = np.ones((close_size, close_size), dtype=np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    dilate_size = max(int(dilate_px), 0)
    if dilate_size > 0:
        kernel = np.ones((2 * dilate_size + 1, 2 * dilate_size + 1), dtype=np.uint8)
        mask = cv.dilate(mask, kernel)

    return mask.astype(bool)


def compute_mask_iou(rendered_mask: np.ndarray, segmentation_mask: np.ndarray) -> float:
    rendered = np.asarray(rendered_mask, dtype=bool)
    segmentation = np.asarray(segmentation_mask, dtype=bool)
    if rendered.shape != segmentation.shape:
        raise ValueError(f"Mask shapes differ: rendered={rendered.shape}, segmentation={segmentation.shape}")

    union = np.logical_or(rendered, segmentation)
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return 1.0
    intersection_count = int(np.count_nonzero(np.logical_and(rendered, segmentation)))
    return float(intersection_count / union_count)


def compute_outside_distance_loss(
    rendered_mask: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    distance_trunc_px: float = 25.0,
) -> float:
    rendered = np.asarray(rendered_mask, dtype=bool)
    segmentation = np.asarray(segmentation_mask, dtype=bool)
    if rendered.shape != segmentation.shape:
        raise ValueError(f"Mask shapes differ: rendered={rendered.shape}, segmentation={segmentation.shape}")

    rendered_count = int(np.count_nonzero(rendered))
    if rendered_count == 0:
        return 0.0

    outside = np.logical_and(rendered, np.logical_not(segmentation))
    if not np.any(outside):
        return 0.0
    if not np.any(segmentation):
        return 1.0

    outside_source = np.logical_not(segmentation).astype(np.uint8)
    distances = cv.distanceTransform(outside_source, cv.DIST_L2, 3)
    trunc = max(float(distance_trunc_px), 1e-6)
    outside_distances = np.minimum(distances[outside], trunc) / trunc
    outside_fraction = float(np.count_nonzero(outside) / max(rendered_count, 1))
    return float(np.mean(outside_distances) * outside_fraction)


def _normalize_mask(mask: np.ndarray, image_shape: tuple[int, int], *, erode_px: int = 0) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    normalized = np.asarray(mask, dtype=bool)
    if normalized.ndim != 2:
        normalized = np.squeeze(normalized)
    if normalized.shape != (height, width):
        normalized = cv.resize(
            normalized.astype(np.uint8),
            (width, height),
            interpolation=cv.INTER_NEAREST,
        ).astype(bool)

    erode_size = max(int(erode_px), 0)
    if erode_size > 0 and np.any(normalized):
        kernel = np.ones((2 * erode_size + 1, 2 * erode_size + 1), dtype=np.uint8)
        normalized = cv.erode(normalized.astype(np.uint8), kernel).astype(bool)
    return normalized


def compute_silhouette_score(
    template_points_base: np.ndarray,
    observations: list[SilhouetteObservation] | tuple[SilhouetteObservation, ...] | None,
    *,
    point_radius_px: int = 2,
    close_kernel_px: int = 5,
    dilate_px: int = 1,
    mask_erode_px: int = 0,
    min_rendered_pixels: int = 30,
    min_segmentation_pixels: int = 50,
    distance_trunc_px: float = 25.0,
    max_projection_points: int = 1500,
) -> SilhouetteScore:
    if not observations:
        return SilhouetteScore(
            valid_camera_count=0,
            loss_iou=0.0,
            outside_loss=0.0,
            mean_iou=0.0,
        )

    points_base = np.asarray(template_points_base, dtype=np.float64).reshape((-1, 3))
    if len(points_base) == 0:
        return SilhouetteScore(
            valid_camera_count=0,
            loss_iou=0.0,
            outside_loss=0.0,
            mean_iou=0.0,
        )
    if max_projection_points > 0 and len(points_base) > int(max_projection_points):
        sample_indices = np.linspace(0, len(points_base) - 1, int(max_projection_points), dtype=np.int32)
        points_base = points_base[sample_indices]

    weighted_iou_sum = 0.0
    weighted_outside_sum = 0.0
    total_weight = 0.0
    valid_count = 0
    rendered_pixel_count = 0
    per_camera_iou: dict[int, float] = {}
    per_camera_outside_loss: dict[int, float] = {}

    for observation in observations:
        try:
            image_shape = (int(observation.image_shape[0]), int(observation.image_shape[1]))
            segmentation_mask = _normalize_mask(observation.mask, image_shape, erode_px=mask_erode_px)
            if int(np.count_nonzero(segmentation_mask)) < int(min_segmentation_pixels):
                continue

            t_cam_base = invert_transform(observation.t_base_cam)
            points_cam = transform_points(points_base, t_cam_base)
            projected_uv = project_points_to_image(points_cam, observation.intrinsics, image_shape)
            rendered_mask = rasterize_projected_template(
                projected_uv,
                image_shape,
                point_radius_px=point_radius_px,
                close_kernel_px=close_kernel_px,
                dilate_px=dilate_px,
            )
            rendered_pixels = int(np.count_nonzero(rendered_mask))
            if rendered_pixels < int(min_rendered_pixels):
                iou = 0.0
                outside_loss = 1.0
            else:
                iou = compute_mask_iou(rendered_mask, segmentation_mask)
                outside_loss = compute_outside_distance_loss(
                    rendered_mask,
                    segmentation_mask,
                    distance_trunc_px=distance_trunc_px,
                )

            weight = max(float(observation.weight), 0.0)
            if weight <= 0.0:
                continue
            valid_count += 1
            rendered_pixel_count += rendered_pixels
            weighted_iou_sum += weight * float(iou)
            weighted_outside_sum += weight * float(outside_loss)
            total_weight += weight
            per_camera_iou[int(observation.camera_id)] = float(iou)
            per_camera_outside_loss[int(observation.camera_id)] = float(outside_loss)
        except Exception:
            continue

    if valid_count == 0 or total_weight <= 0.0:
        return SilhouetteScore(
            valid_camera_count=0,
            loss_iou=0.0,
            outside_loss=0.0,
            mean_iou=0.0,
        )

    mean_iou = weighted_iou_sum / total_weight
    mean_outside = weighted_outside_sum / total_weight
    return SilhouetteScore(
        valid_camera_count=valid_count,
        loss_iou=float(1.0 - mean_iou),
        outside_loss=float(mean_outside),
        mean_iou=float(mean_iou),
        per_camera_iou=per_camera_iou,
        per_camera_outside_loss=per_camera_outside_loss,
        rendered_pixel_count=int(rendered_pixel_count),
    )


__all__ = [
    "SilhouetteObservation",
    "SilhouetteScore",
    "compute_mask_iou",
    "compute_outside_distance_loss",
    "compute_silhouette_score",
    "invert_transform",
    "project_points_to_image",
    "rasterize_projected_template",
    "transform_points",
]
