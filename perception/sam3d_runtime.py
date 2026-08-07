"""Runtime SAM3D capture and generation helpers.

Unlike the startup bootstrap, runtime capture consumes frames already owned
by the standalone RTDE loop. This avoids opening cam0 a second time while the
dual-camera sensor hub is active.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from object_pt_extraction.fastsam_engine import binary_mask_iou


@dataclass(frozen=True)
class StableCapture:
    image_bgr: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    bbox_iou: float
    model_confidence: float
    mask_pixels: int


class StableCaptureAccumulator:
    """Collect the same stable FastSAM input used by startup bootstrap."""

    def __init__(
        self,
        *,
        stable_frames: int,
        min_mask_pixels: int,
        min_temporal_iou: float,
    ) -> None:
        self.stable_frames = max(1, int(stable_frames))
        self.min_mask_pixels = max(1, int(min_mask_pixels))
        self.min_temporal_iou = float(min_temporal_iou)
        self._accepted: deque[StableCapture] = deque(maxlen=self.stable_frames)

    def clear(self) -> None:
        self._accepted.clear()

    def add(self, image_bgr, mask, selection) -> StableCapture | None:
        if selection is None or mask is None:
            self.clear()
            return None
        image = np.asarray(image_bgr, dtype=np.uint8)
        binary = np.asarray(mask, dtype=bool)
        if binary.shape != image.shape[:2]:
            self.clear()
            return None
        mask_pixels = int(binary.sum())
        if mask_pixels < self.min_mask_pixels:
            self.clear()
            return None
        if (
            self._accepted
            and binary_mask_iou(self._accepted[-1].mask, binary)
            < self.min_temporal_iou
        ):
            self.clear()
        capture = StableCapture(
            image_bgr=image.copy(),
            mask=binary.copy(),
            bbox=tuple(int(value) for value in selection.bbox),
            bbox_iou=float(selection.bbox_iou),
            model_confidence=float(selection.model_confidence),
            mask_pixels=mask_pixels,
        )
        self._accepted.append(capture)
        if len(self._accepted) < self.stable_frames:
            return None
        return max(
            self._accepted,
            key=lambda item: (item.bbox_iou, item.model_confidence),
        )


def write_capture_artifacts(
    artifact_dir: str | Path,
    capture: StableCapture,
    *,
    label: str,
    configured_bboxes: dict[str, Any],
) -> Path:
    """Write standalone image/mask/preview/metadata artifacts."""
    directory = Path(artifact_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    image_path = directory / "image.png"
    mask_path = directory / "mask.png"
    preview_path = directory / "preview.png"

    if not cv.imwrite(str(image_path), capture.image_bgr):
        raise RuntimeError(f"Failed to write captured image: {image_path}")
    if not cv.imwrite(str(mask_path), capture.mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write captured mask: {mask_path}")

    preview = capture.image_bgr.copy()
    tint = np.zeros_like(preview)
    tint[..., 1] = 255
    active = capture.mask.astype(bool)
    preview[active] = (
        0.55 * preview[active].astype(np.float32)
        + 0.45 * tint[active].astype(np.float32)
    ).astype(np.uint8)
    x1, y1, x2, y2 = capture.bbox
    cv.rectangle(
        preview,
        (x1, y1),
        (max(x1, x2 - 1), max(y1, y2 - 1)),
        (0, 255, 255),
        2,
    )
    if not cv.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to write capture preview: {preview_path}")

    metadata = {
        "backend": "sam3d",
        "capture": {
            "camera": "cam0",
            "source": "standalone_rtde_loop",
            "bbox_xyxy": list(capture.bbox),
            "mask_pixels": int(capture.mask_pixels),
            "bbox_iou": float(capture.bbox_iou),
            "model_confidence": float(capture.model_confidence),
        },
        "configuration": {
            "label": str(label),
            "bboxes_xyxy": {
                str(camera): [int(round(float(value))) for value in bbox]
                for camera, bbox in configured_bboxes.items()
            },
        },
        "artifacts": {
            "image": str(image_path),
            "mask": str(mask_path),
            "preview": str(preview_path),
        },
    }
    with open(directory / "metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
    return directory


def _environment_for(interpreter: Path) -> dict[str, str]:
    environment = dict(os.environ)
    env_root = interpreter.resolve().parents[1]
    environment["CONDA_PREFIX"] = str(env_root)
    environment["CUDA_HOME"] = str(env_root)
    environment["LIDRA_SKIP_INIT"] = "true"
    env_lib = env_root / "lib"
    if env_lib.is_dir():
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            f"{env_lib}:{existing}" if existing else str(env_lib)
        )
    return environment


def generate_runtime_template(
    *,
    config_path: str | Path,
    artifact_dir: str | Path,
    repo_root: str | Path,
    execution_mode: str,
    sam3d_python: str | Path,
    sam3d_repo: str | Path,
    server_socket: str,
    server_ready_timeout_s: float,
    request_timeout_s: float,
) -> Path:
    """Generate ``template.npy`` from already-written runtime artifacts."""
    config_path = Path(config_path).expanduser().resolve()
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    sam3d_python = Path(sam3d_python).expanduser().resolve()
    sam3d_repo = Path(sam3d_repo).expanduser()
    if not sam3d_repo.is_absolute():
        sam3d_repo = repo_root / sam3d_repo
    sam3d_repo = sam3d_repo.resolve()
    mode = str(execution_mode).strip().lower()
    if mode not in {"server", "oneshot"}:
        raise ValueError(f"Unsupported SAM3D execution mode: {execution_mode!r}")

    if mode == "server":
        from tools.sam3d_ipc import build_model_spec, generate, wait_until_ready

        model_spec = build_model_spec(config_path, sam3d_repo)
        wait_until_ready(
            server_socket,
            expected_signature=model_spec["signature"],
            ready_timeout_s=float(server_ready_timeout_s),
        )
        generate(
            server_socket,
            config_path=config_path,
            artifact_dir=artifact_dir,
            expected_signature=model_spec["signature"],
            timeout_s=float(request_timeout_s),
        )
    else:
        command = [
            str(sam3d_python),
            str(repo_root / "tools" / "sam3d_generate.py"),
            "--config",
            str(config_path),
            "--artifact-dir",
            str(artifact_dir),
            "--repo-root",
            str(repo_root),
            "--sam3d-repo",
            str(sam3d_repo),
        ]
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            env=_environment_for(sam3d_python),
            capture_output=True,
            text=True,
            timeout=float(request_timeout_s),
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(
                f"SAM3D oneshot generation failed with exit code "
                f"{result.returncode}: {detail}"
            )

    template_path = artifact_dir / "template.npy"
    if not template_path.is_file():
        raise RuntimeError(
            f"SAM3D generation completed without template: {template_path}"
        )
    return template_path


def validate_template_points(
    template_path: str | Path,
    *,
    min_template_points: int,
) -> np.ndarray:
    """Load and validate a generated point cloud before tracker replacement."""
    path = Path(template_path).expanduser().resolve()
    points = np.load(path, allow_pickle=False)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"SAM3D template must have shape (N,3), got {points.shape}"
        )
    if len(points) < max(1, int(min_template_points)):
        raise ValueError(
            f"SAM3D template has {len(points)} points; "
            f"minimum is {int(min_template_points)}"
        )
    if not np.isfinite(points).all():
        raise ValueError("SAM3D template contains non-finite coordinates")
    return points


__all__ = [
    "StableCapture",
    "StableCaptureAccumulator",
    "generate_runtime_template",
    "validate_template_points",
    "write_capture_artifacts",
]
