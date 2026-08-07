#!/usr/bin/env python3
"""Capture a stable cam0 FastSAM mask for one standalone SAM3D startup."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import cv2 as cv
import numpy as np
import yaml

# Executed by absolute script path, so add the repository root rather than
# relying on the caller's PYTHONPATH.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_pt_extraction.fastsam_engine import (
    DEFAULT_LABEL,
    FastSAMSegmentationEngine,
    SharedFastSAMModel,
    binary_mask_iou,
    normalize_bbox,
)
from utils.realsense_stream import RealSenseCamera


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_metadata(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _write_preview(image_bgr: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], path: Path) -> None:
    preview = np.asarray(image_bgr, dtype=np.uint8).copy()
    active = np.asarray(mask, dtype=bool)
    tint = np.zeros_like(preview)
    tint[..., 1] = 255
    preview[active] = (
        0.55 * preview[active].astype(np.float32)
        + 0.45 * tint[active].astype(np.float32)
    ).astype(np.uint8)
    x1, y1, x2, y2 = bbox
    cv.rectangle(preview, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (0, 255, 255), 2)
    if not cv.imwrite(str(path), preview):
        raise RuntimeError(f"Failed to write preview: {path}")


def capture_stable_input(config: dict, artifact_dir: Path) -> dict:
    camera_configs = config.get("cameras", {}) or {}
    camera_cfg = camera_configs.get("cam0", {})
    fastsam_cfg = config.get("perception", {}).get("object", {}).get("fastsam", {}) or {}
    bboxes = fastsam_cfg.get("bboxes", {}) or {}
    missing_bboxes = [camera for camera in ("cam0", "cam1") if camera not in bboxes]
    if missing_bboxes:
        raise ValueError(f"Missing FastSAM bboxes: {missing_bboxes}")
    frame_width = int(camera_cfg.get("width", 640))
    frame_height = int(camera_cfg.get("height", 480))
    min_bbox_size_px = int(fastsam_cfg.get("min_bbox_size_px", 8))
    configured_bboxes = {
        camera: normalize_bbox(
            bbox,
            int(camera_configs.get(camera, {}).get("width", frame_width)),
            int(camera_configs.get(camera, {}).get("height", frame_height)),
            min_size_px=min_bbox_size_px,
        )
        for camera, bbox in bboxes.items()
        if camera in {"cam0", "cam1"}
    }

    label = str(fastsam_cfg.get("label", DEFAULT_LABEL))
    shared_model = SharedFastSAMModel(
        fastsam_cfg.get("model_name", "sam-3d-objects/FastSAM-s.pt"),
        imgsz=int(fastsam_cfg.get("imgsz", 1024)),
        conf=float(fastsam_cfg.get("conf", 0.4)),
        iou=float(fastsam_cfg.get("iou", 0.9)),
        device=fastsam_cfg.get("device"),
        half=bool(fastsam_cfg.get("half", False)),
        require_cuda=bool(fastsam_cfg.get("require_cuda", False)),
    )
    engine = FastSAMSegmentationEngine(
        shared_model,
        bbox=configured_bboxes["cam0"],
        label=label,
        min_bbox_size_px=min_bbox_size_px,
    )

    stable_frames = max(1, int(fastsam_cfg.get("stable_frames", 3)))
    min_mask_pixels = max(1, int(fastsam_cfg.get("min_mask_pixels", 300)))
    min_temporal_iou = float(fastsam_cfg.get("min_temporal_iou", 0.8))
    timeout_s = max(0.1, float(fastsam_cfg.get("capture_timeout_s", 30.0)))
    warmup_frames = max(0, int(fastsam_cfg.get("warmup_frames", 5)))

    camera = RealSenseCamera(
        serial=camera_cfg.get("serial"),
        width=frame_width,
        height=frame_height,
        fps=int(camera_cfg.get("fps", config.get("system", {}).get("sensor_fps", 30))),
        depth_filters=dict(camera_cfg.get("depth_filters", {}) or {}),
    )
    accepted: deque[tuple[np.ndarray, np.ndarray, object]] = deque(maxlen=stable_frames)
    started = time.perf_counter()
    try:
        for _ in range(warmup_frames):
            camera.read()
        while time.perf_counter() - started < timeout_s:
            frame = camera.read()
            result = engine.predict(frame.color_image)
            selection = engine.last_selection
            if not result.instances or selection is None or selection.mask_pixels < min_mask_pixels:
                accepted.clear()
                continue
            mask = np.asarray(result.instances[0].mask, dtype=bool)
            if accepted and binary_mask_iou(accepted[-1][1], mask) < min_temporal_iou:
                accepted.clear()
            accepted.append((frame.color_image.copy(), mask.copy(), selection))
            if len(accepted) < stable_frames:
                continue

            best_image, best_mask, best_selection = max(
                accepted,
                key=lambda item: (float(item[2].bbox_iou), float(item[2].model_confidence)),
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            image_path = artifact_dir / "image.png"
            mask_path = artifact_dir / "mask.png"
            preview_path = artifact_dir / "preview.png"
            if not cv.imwrite(str(image_path), best_image):
                raise RuntimeError(f"Failed to write captured image: {image_path}")
            if not cv.imwrite(str(mask_path), best_mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Failed to write captured mask: {mask_path}")
            _write_preview(best_image, best_mask, best_selection.bbox, preview_path)

            payload = {
                "backend": "sam3d",
                "capture": {
                    "camera": "cam0",
                    "serial": str(camera.serial),
                    "bbox_xyxy": list(best_selection.bbox),
                    "stable_frames": stable_frames,
                    "min_temporal_iou": min_temporal_iou,
                    "mask_pixels": int(best_selection.mask_pixels),
                    "bbox_iou": float(best_selection.bbox_iou),
                    "model_confidence": float(best_selection.model_confidence),
                    "model_name": str(shared_model.model_name),
                    "elapsed_s": float(time.perf_counter() - started),
                },
                "configuration": {
                    "label": label,
                    "bboxes_xyxy": {
                        camera: list(bbox)
                        for camera, bbox in configured_bboxes.items()
                    },
                    "bbox_selection": dict(
                        fastsam_cfg.get("bbox_selection", {}) or {}
                    ),
                },
                "artifacts": {
                    "image": str(image_path.resolve()),
                    "mask": str(mask_path.resolve()),
                    "preview": str(preview_path.resolve()),
                },
            }
            _write_metadata(artifact_dir / "metadata.yaml", payload)
            return payload
    finally:
        camera.stop()
    raise TimeoutError(
        f"No stable FastSAM mask within {timeout_s:.1f}s "
        f"(required frames={stable_frames}, pixels>={min_mask_pixels}, IoU>={min_temporal_iou:.3f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"repo root does not exist: {repo_root}")
    config = _load_config(args.config)
    payload = capture_stable_input(config, Path(args.artifact_dir).expanduser())
    print(
        "[sam3d_capture] stable input saved: "
        f"pixels={payload['capture']['mask_pixels']} "
        f"bbox_iou={payload['capture']['bbox_iou']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
