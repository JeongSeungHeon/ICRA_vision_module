"""Shared depth-image filtering helpers."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on deployment image
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV is required for bilateral depth filtering.")


def bilateral_filter_depth(
    depth_m: np.ndarray,
    radius: int = 2,
    zfar: float = 100.0,
    sigma_space: float = 2.0,
    sigma_color: float = 0.02,
) -> np.ndarray:
    _require_cv2()
    depth_np = np.asarray(depth_m, dtype=np.float32)
    if depth_np.size == 0:
        return depth_np.copy()

    valid = np.isfinite(depth_np)
    valid &= depth_np >= 0.001
    valid &= depth_np < float(zfar)
    if not np.any(valid):
        return np.zeros_like(depth_np, dtype=np.float32)

    filtered = depth_np.copy()
    filtered[~valid] = 0.0
    ksize = max(1, int(radius) * 2 + 1)
    filtered = cv2.bilateralFilter(
        filtered,
        d=ksize,
        sigmaColor=max(float(sigma_color), 1e-6),
        sigmaSpace=max(float(sigma_space), 1.0),
    )
    filtered = np.asarray(filtered, dtype=np.float32)
    filtered[~valid] = 0.0
    return filtered
