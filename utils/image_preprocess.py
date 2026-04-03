"""Image preprocessing helpers for segmentation inputs."""

from __future__ import annotations

import cv2 as cv
import numpy as np


def adjust_contrast_brightness(image_bgr: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    alpha = float(alpha)
    beta = float(beta)
    if abs(alpha - 1.0) < 1e-6 and abs(beta) < 1e-6:
        return image.copy()
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)


def adjust_gamma(image_bgr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    gamma = max(float(gamma), 1e-6)
    if abs(gamma - 1.0) < 1e-6:
        return image.copy()

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((index / 255.0) ** inv_gamma) * 255.0 for index in range(256)],
        dtype=np.float32,
    )
    return cv.LUT(image, np.clip(table, 0.0, 255.0).astype(np.uint8))


def apply_clahe_bgr(
    image_bgr: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    grid_size = max(int(tile_grid_size), 1)
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv.split(lab_image)
    clahe = cv.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid_size, grid_size))
    l_channel = clahe.apply(l_channel)
    merged_lab = cv.merge([l_channel, a_channel, b_channel])
    return cv.cvtColor(merged_lab, cv.COLOR_LAB2BGR)


def apply_segmentation_preprocess(image_bgr: np.ndarray, preprocess_cfg: dict | None) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    cfg = dict(preprocess_cfg or {})
    if not cfg.get("enabled", False):
        return image.copy()

    processed = adjust_contrast_brightness(
        image,
        alpha=cfg.get("contrast_alpha", 1.0),
        beta=cfg.get("brightness_beta", 0.0),
    )
    processed = adjust_gamma(processed, gamma=cfg.get("gamma", 1.0))

    clahe_cfg = dict(cfg.get("clahe", {}) or {})
    if clahe_cfg.get("enabled", False):
        processed = apply_clahe_bgr(
            processed,
            clip_limit=clahe_cfg.get("clip_limit", 2.0),
            tile_grid_size=clahe_cfg.get("tile_grid_size", 8),
        )

    return processed


__all__ = [
    "adjust_contrast_brightness",
    "adjust_gamma",
    "apply_clahe_bgr",
    "apply_segmentation_preprocess",
]
