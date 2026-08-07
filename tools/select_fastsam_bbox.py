"""Interactive cam0/cam1 FastSAM bbox selector.

This process opens each configured RealSense sequentially, lets the operator
freeze a frame and drag an ROI, validates the ROI with FastSAM, and only
replaces the runtime selection YAML after both cameras are accepted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2 as cv
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.fastsam_bbox_config import (
    CAMERA_NAMES,
    FastSAMBBoxSelectionError,
    atomic_write_selection,
    build_selection_payload,
    xywh_to_xyxy,
)


KEY_ESCAPE = 27
KEY_ENTER = {10, 13}
KEY_FREEZE = KEY_ENTER | {32}
KEY_RETRY = {ord("r"), ord("R")}


class BBoxSelectionCancelled(RuntimeError):
    """Raised when the operator cancels without replacing saved state."""


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return config


def _window_is_open(name: str) -> bool:
    try:
        return cv.getWindowProperty(name, cv.WND_PROP_VISIBLE) >= 1.0
    except cv.error:
        return False


def _destroy_window(name: str) -> None:
    try:
        cv.destroyWindow(name)
        cv.waitKey(1)
    except cv.error:
        pass


def _destroy_all_windows() -> None:
    try:
        cv.destroyAllWindows()
        cv.waitKey(1)
    except cv.error:
        pass


def _key_code(delay_ms: int) -> int:
    key = int(cv.waitKey(delay_ms))
    return key & 0xFF if key >= 0 else -1


def _draw_lines(
    image: np.ndarray,
    lines: list[str],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    rendered = np.asarray(image, dtype=np.uint8).copy()
    line_height = 25
    overlay_height = min(rendered.shape[0], 12 + line_height * len(lines))
    overlay = rendered[:overlay_height].copy()
    overlay[:] = (0, 0, 0)
    rendered[:overlay_height] = cv.addWeighted(
        rendered[:overlay_height],
        0.35,
        overlay,
        0.65,
        0.0,
    )
    for index, text in enumerate(lines):
        cv.putText(
            rendered,
            str(text),
            (10, 24 + index * line_height),
            cv.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv.LINE_AA,
        )
    return rendered


def _capture_frozen_frame(camera, window_name: str, camera_name: str) -> np.ndarray:
    while True:
        frame = camera.read()
        preview = _draw_lines(
            frame.color_image,
            [
                f"{camera_name}: live preview",
                "SPACE / ENTER: freeze frame    ESC: cancel without saving",
            ],
        )
        cv.imshow(window_name, preview)
        key = _key_code(1)
        if key in KEY_FREEZE:
            return np.asarray(frame.color_image, dtype=np.uint8).copy()
        if key == KEY_ESCAPE or not _window_is_open(window_name):
            raise BBoxSelectionCancelled(f"{camera_name} selection cancelled")


def _drag_roi_xywh(
    frozen_frame: np.ndarray,
    window_name: str,
    camera_name: str,
) -> tuple[int, int, int, int] | None:
    state: dict[str, Any] = {
        "dragging": False,
        "start": None,
        "current": None,
        "roi": None,
    }
    height, width = frozen_frame.shape[:2]

    def _mouse(event, x, y, _flags, _param):
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        if event == cv.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
            state["roi"] = None
        elif event == cv.EVENT_MOUSEMOVE and state["dragging"]:
            state["current"] = (x, y)
        elif event == cv.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            state["current"] = (x, y)
            start_x, start_y = state["start"]
            x1, x2 = sorted((start_x, x))
            y1, y2 = sorted((start_y, y))
            # Mouse coordinates identify pixels while XYXY uses an exclusive
            # max corner, so include the release pixel in the ROI.
            state["roi"] = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    cv.setMouseCallback(window_name, _mouse)
    try:
        while True:
            preview = _draw_lines(
                frozen_frame,
                [
                    f"{camera_name}: drag bbox with left mouse button",
                    "R: return to live preview    ESC: cancel without saving",
                ],
            )
            if state["start"] is not None and state["current"] is not None:
                x1, y1 = state["start"]
                x2, y2 = state["current"]
                cv.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv.imshow(window_name, preview)
            key = _key_code(10)
            if key == KEY_ESCAPE or not _window_is_open(window_name):
                raise BBoxSelectionCancelled(f"{camera_name} selection cancelled")
            if key in KEY_RETRY:
                return None
            if state["roi"] is not None:
                return tuple(int(value) for value in state["roi"])
    finally:
        cv.setMouseCallback(window_name, lambda *_args: None)


def _mask_preview(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    camera_name: str,
    mask_pixels: int,
    bbox_iou: float,
    confidence: float,
) -> np.ndarray:
    preview = np.asarray(frame_bgr, dtype=np.uint8).copy()
    active = np.asarray(mask, dtype=bool)
    tint = np.zeros_like(preview)
    tint[..., 1] = 255
    preview[active] = (
        0.55 * preview[active].astype(np.float32)
        + 0.45 * tint[active].astype(np.float32)
    ).astype(np.uint8)
    x1, y1, x2, y2 = bbox
    cv.rectangle(
        preview,
        (x1, y1),
        (max(x1, x2 - 1), max(y1, y2 - 1)),
        (0, 255, 255),
        2,
    )
    return _draw_lines(
        preview,
        [
            f"{camera_name}: FastSAM mask preview",
            (
                f"bbox={bbox}  pixels={mask_pixels}  "
                f"IoU={bbox_iou:.3f}  confidence={confidence:.3f}"
            ),
            "ENTER: accept    R: recapture/reselect    ESC: cancel without saving",
        ],
    )


def _invalid_preview(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    camera_name: str,
    reason: str,
) -> np.ndarray:
    preview = np.asarray(frame_bgr, dtype=np.uint8).copy()
    x1, y1, x2, y2 = bbox
    cv.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
    return _draw_lines(
        preview,
        [
            f"{camera_name}: FastSAM validation failed",
            reason,
            "R / ENTER: recapture/reselect    ESC: cancel without saving",
        ],
        color=(80, 80, 255),
    )


def mask_rejection_reason(result, selection, *, min_mask_pixels: int) -> str | None:
    """Return why a FastSAM preview cannot be accepted, or ``None``."""
    if not getattr(result, "instances", None) or selection is None:
        return "FastSAM returned no non-empty mask"
    mask_pixels = int(getattr(selection, "mask_pixels", 0))
    if mask_pixels < int(min_mask_pixels):
        return f"mask pixels {mask_pixels} < required {int(min_mask_pixels)}"
    return None


def _wait_for_decision(
    preview: np.ndarray,
    window_name: str,
    *,
    can_accept: bool,
) -> bool:
    while True:
        cv.imshow(window_name, preview)
        key = _key_code(10)
        if key == KEY_ESCAPE or not _window_is_open(window_name):
            raise BBoxSelectionCancelled("selection cancelled")
        if key in KEY_RETRY or (not can_accept and key in KEY_ENTER):
            return False
        if can_accept and key in KEY_ENTER:
            return True


def _select_camera(
    camera_name: str,
    *,
    config: Mapping[str, Any],
    shared_model,
) -> dict[str, Any]:
    from object_pt_extraction.fastsam_engine import FastSAMSegmentationEngine
    from utils.realsense_stream import RealSenseCamera

    camera_cfg = (config.get("cameras", {}) or {}).get(camera_name, {}) or {}
    fastsam_cfg = (
        (config.get("perception", {}) or {}).get("object", {}) or {}
    ).get("fastsam", {}) or {}
    width = int(camera_cfg.get("width", 640))
    height = int(camera_cfg.get("height", 480))
    min_bbox_size_px = max(1, int(fastsam_cfg.get("min_bbox_size_px", 8)))
    min_mask_pixels = max(1, int(fastsam_cfg.get("min_mask_pixels", 300)))
    warmup_frames = max(0, int(fastsam_cfg.get("warmup_frames", 5)))
    label = str(fastsam_cfg.get("label", "sam3d_object"))
    window_name = f"FastSAM bbox selector - {camera_name}"

    camera = RealSenseCamera(
        serial=camera_cfg.get("serial"),
        width=width,
        height=height,
        fps=int(camera_cfg.get("fps", config.get("system", {}).get("sensor_fps", 30))),
        depth_filters=dict(camera_cfg.get("depth_filters", {}) or {}),
    )
    try:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, width, height)
        for _ in range(warmup_frames):
            camera.read()
        while True:
            frozen = _capture_frozen_frame(camera, window_name, camera_name)
            roi_xywh = _drag_roi_xywh(frozen, window_name, camera_name)
            if roi_xywh is None:
                continue
            try:
                bbox = xywh_to_xyxy(
                    roi_xywh,
                    width,
                    height,
                    min_size_px=min_bbox_size_px,
                )
            except FastSAMBBoxSelectionError as exc:
                invalid = _invalid_preview(
                    frozen,
                    (0, 0, width, height),
                    camera_name=camera_name,
                    reason=str(exc),
                )
                _wait_for_decision(invalid, window_name, can_accept=False)
                continue

            engine = FastSAMSegmentationEngine(
                shared_model,
                bbox=bbox,
                label=label,
                min_bbox_size_px=min_bbox_size_px,
            )
            result = engine.predict(frozen)
            selection = engine.last_selection
            rejection_reason = mask_rejection_reason(
                result,
                selection,
                min_mask_pixels=min_mask_pixels,
            )
            if rejection_reason is not None:
                invalid = _invalid_preview(
                    frozen,
                    bbox,
                    camera_name=camera_name,
                    reason=rejection_reason,
                )
                _wait_for_decision(invalid, window_name, can_accept=False)
                continue

            preview = _mask_preview(
                frozen,
                result.instances[0].mask,
                selection.bbox,
                camera_name=camera_name,
                mask_pixels=selection.mask_pixels,
                bbox_iou=selection.bbox_iou,
                confidence=selection.model_confidence,
            )
            if not _wait_for_decision(preview, window_name, can_accept=True):
                continue
            return {
                "serial": str(camera.serial),
                "width": width,
                "height": height,
                "bbox_xyxy": list(selection.bbox),
                "mask_pixels": int(selection.mask_pixels),
                "bbox_iou": float(selection.bbox_iou),
                "model_confidence": float(selection.model_confidence),
            }
    finally:
        try:
            camera.stop()
        finally:
            _destroy_window(window_name)


def run_selector(config: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "FastSAM bbox selector requires a graphical session "
            "(DISPLAY or WAYLAND_DISPLAY is not set)"
        )

    from object_pt_extraction.fastsam_engine import SharedFastSAMModel

    fastsam_cfg = (
        (config.get("perception", {}) or {}).get("object", {}) or {}
    ).get("fastsam", {}) or {}
    shared_model = SharedFastSAMModel(
        fastsam_cfg.get("model_name", "sam-3d-objects/FastSAM-s.pt"),
        imgsz=int(fastsam_cfg.get("imgsz", 1024)),
        conf=float(fastsam_cfg.get("conf", 0.4)),
        iou=float(fastsam_cfg.get("iou", 0.9)),
        device=fastsam_cfg.get("device"),
        half=bool(fastsam_cfg.get("half", False)),
        require_cuda=bool(fastsam_cfg.get("require_cuda", False)),
    )
    accepted: dict[str, dict[str, Any]] = {}
    try:
        for camera_name in CAMERA_NAMES:
            accepted[camera_name] = _select_camera(
                camera_name,
                config=config,
                shared_model=shared_model,
            )
        payload = build_selection_payload(
            config,
            fastsam_model=str(shared_model.model_name),
            camera_results=accepted,
        )
        atomic_write_selection(output_path, payload)
        return payload
    finally:
        _destroy_all_windows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"repo root does not exist: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    config = _load_config(Path(args.config).expanduser().resolve())
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    try:
        payload = run_selector(config, output_path.resolve())
    except BBoxSelectionCancelled as exc:
        print(f"[fastsam_bbox_selector] cancelled: {exc}", flush=True)
        raise SystemExit(130) from exc

    print(
        "[fastsam_bbox_selector] saved "
        f"{output_path.resolve()} "
        f"cam0={payload['cameras']['cam0']['bbox_xyxy']} "
        f"cam1={payload['cameras']['cam1']['bbox_xyxy']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
