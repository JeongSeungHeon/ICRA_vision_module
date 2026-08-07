"""FastSAM bbox-prompt backend compatible with the object segmentation worker.

The live handover pipeline consumes :class:`SegmentationResult` objects.  This
module keeps that contract while replacing YOLOE class segmentation with one
classless FastSAM mask selected by overlap with a configured bbox prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import cv2 as cv
import numpy as np

from object_pt_extraction.segmentation_engine import SegmentationInstance, SegmentationResult

try:
    from ultralytics import FastSAM
except ImportError:  # pragma: no cover - depends on the runtime environment
    FastSAM = None


DEFAULT_LABEL = "sam3d_object"
MIN_BBOX_SIZE_PX = 8
REPO_ROOT = Path(__file__).resolve().parents[1]


class FastSAMError(RuntimeError):
    """Raised when FastSAM cannot produce a usable bbox-prompt mask."""


@dataclass(frozen=True)
class FastSAMSelection:
    bbox: tuple[int, int, int, int]
    selected_index: int
    bbox_iou: float
    mask_pixels: int
    model_confidence: float


def resolve_fastsam_device(device: Any, *, require_cuda: bool = False) -> Any:
    """Resolve the Ultralytics device without allowing a silent CPU fallback."""
    if not require_cuda:
        return device

    resolved = "cuda:0" if device is None else device
    normalized = str(resolved).strip().lower()
    is_cuda_device = (
        isinstance(resolved, int)
        and not isinstance(resolved, bool)
        and resolved >= 0
    ) or normalized.isdigit() or normalized == "cuda" or normalized.startswith(
        "cuda:"
    )
    if not is_cuda_device:
        raise RuntimeError(
            "FastSAM require_cuda=true requires a CUDA device such as 'cuda:0' "
            f"or 0, got {device!r}"
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "FastSAM require_cuda=true, but PyTorch is not installed in the "
            "perception Python environment"
        ) from exc
    if torch.version.cuda is None:
        raise RuntimeError(
            "FastSAM require_cuda=true, but the installed PyTorch is a CPU-only build"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FastSAM require_cuda=true, but torch.cuda.is_available() is false"
        )
    return resolved


def resolve_asset_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    for root in (Path.cwd(), REPO_ROOT):
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def normalize_bbox(
    bbox_values: Sequence[float],
    width: int,
    height: int,
    *,
    min_size_px: int = MIN_BBOX_SIZE_PX,
) -> tuple[int, int, int, int]:
    """Round, clamp, and validate one XYXY bbox."""
    if len(bbox_values) != 4:
        raise ValueError(f"FastSAM bbox must contain four XYXY values, got {bbox_values!r}")
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox_values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"FastSAM bbox must satisfy x1 < x2 and y1 < y2, got {(x1, y1, x2, y2)}")

    width = max(1, int(width))
    height = max(1, int(height))
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    min_size_px = max(1, int(min_size_px))
    if x2 - x1 < min_size_px or y2 - y1 < min_size_px:
        raise ValueError(
            "FastSAM bbox is outside the frame or too small after clamping: "
            f"{(x1, y1, x2, y2)}, minimum={min_size_px}px"
        )
    return x1, y1, x2, y2


def binary_mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    if first_mask.shape != second_mask.shape:
        raise ValueError(f"mask shapes differ: {first_mask.shape} vs {second_mask.shape}")
    union = np.logical_or(first_mask, second_mask).sum(dtype=np.float64)
    if union <= 0:
        return 0.0
    intersection = np.logical_and(first_mask, second_mask).sum(dtype=np.float64)
    return float(intersection / union)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = True
    return mask


def masks_from_result(result: Any, width: int, height: int) -> np.ndarray:
    masks_obj = getattr(result, "masks", None)
    mask_data = None if masks_obj is None else getattr(masks_obj, "data", None)
    if mask_data is None:
        return np.empty((0, height, width), dtype=bool)
    if hasattr(mask_data, "detach"):
        mask_data = mask_data.detach().cpu().numpy()
    mask_data = np.asarray(mask_data)
    if mask_data.ndim == 2:
        mask_data = mask_data[None, ...]
    if mask_data.ndim != 3:
        raise FastSAMError(f"Unexpected FastSAM mask shape: {mask_data.shape}")

    resized: list[np.ndarray] = []
    for mask in mask_data:
        binary = np.asarray(mask) > 0.5
        if binary.shape != (height, width):
            binary = cv.resize(
                binary.astype(np.uint8),
                (width, height),
                interpolation=cv.INTER_NEAREST,
            ).astype(bool)
        resized.append(binary)
    if not resized:
        return np.empty((0, height, width), dtype=bool)
    return np.stack(resized, axis=0)


def select_mask_by_bbox_iou(
    masks: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, float]:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise FastSAMError(f"Expected FastSAM masks shaped (N,H,W), got {masks.shape}")
    prompt_mask = _bbox_mask((height, width), bbox)
    best_mask: np.ndarray | None = None
    best_index = -1
    best_iou = -1.0
    for index, candidate in enumerate(masks):
        if candidate.shape != prompt_mask.shape or not np.any(candidate):
            continue
        score = binary_mask_iou(candidate, prompt_mask)
        if score > best_iou:
            best_mask = candidate
            best_index = int(index)
            best_iou = float(score)
    if best_mask is None:
        raise FastSAMError("FastSAM returned no non-empty mask for the configured bbox")
    return best_mask, best_index, best_iou


def _result_confidences(result: Any, count: int) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    values = None if boxes is None else getattr(boxes, "conf", None)
    if values is None:
        return np.ones((count,), dtype=np.float32)
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < count:
        values = np.pad(values, (0, count - len(values)), constant_values=1.0)
    return values[:count]


def _result_bboxes(
    result: Any,
    count: int,
    fallback: tuple[int, int, int, int],
) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    values = None if boxes is None else getattr(boxes, "xyxy", None)
    if values is None:
        return np.tile(np.asarray(fallback, dtype=np.float32), (count, 1))
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32).reshape((-1, 4))
    if len(values) < count:
        padding = np.tile(np.asarray(fallback, dtype=np.float32), (count - len(values), 1))
        values = np.concatenate([values, padding], axis=0)
    return values[:count]


class SharedFastSAMModel:
    """One FastSAM model shared by the sequential cam0/cam1 workers."""

    def __init__(
        self,
        model_name: str | Path,
        *,
        imgsz: int = 1024,
        conf: float = 0.4,
        iou: float = 0.9,
        device: Any = None,
        half: bool = False,
        require_cuda: bool = False,
        model: Any = None,
    ) -> None:
        resolved = resolve_asset_path(model_name)
        self.model_name = str(resolved if resolved.exists() else model_name)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.require_cuda = bool(require_cuda)
        self.device = resolve_fastsam_device(
            device,
            require_cuda=self.require_cuda,
        )
        self.half = bool(half)
        if model is not None:
            self.model = model
        else:
            if FastSAM is None:
                raise RuntimeError("ultralytics FastSAM is required for object_backend=sam3d")
            try:
                self.model = FastSAM(self.model_name)
            except Exception as exc:
                raise RuntimeError(f"Failed to load FastSAM model: {self.model_name}") from exc

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        bbox: Sequence[float],
        label: str = DEFAULT_LABEL,
        min_bbox_size_px: int = MIN_BBOX_SIZE_PX,
    ) -> tuple[SegmentationResult, FastSAMSelection | None]:
        frame = np.asarray(frame_bgr, dtype=np.uint8)
        height, width = frame.shape[:2]
        normalized_bbox = normalize_bbox(
            bbox,
            width,
            height,
            min_size_px=min_bbox_size_px,
        )
        started = time.perf_counter()
        results = self.model.predict(
            source=frame,
            bboxes=[list(normalized_bbox)],
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            retina_masks=True,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - started) * 1000.0
        raw_result = results[0] if results else None
        if raw_result is None:
            return (
                SegmentationResult(
                    instances=[],
                    raw_result=None,
                    infer_ms=infer_ms,
                    prompt_classes=[str(label)],
                    preprocessed_image=frame.copy(),
                ),
                None,
            )

        masks = masks_from_result(raw_result, width, height)
        if len(masks) == 0:
            return (
                SegmentationResult(
                    instances=[],
                    raw_result=raw_result,
                    infer_ms=infer_ms,
                    prompt_classes=[str(label)],
                    preprocessed_image=frame.copy(),
                ),
                None,
            )

        try:
            mask, selected_index, bbox_iou = select_mask_by_bbox_iou(
                masks,
                normalized_bbox,
                width=width,
                height=height,
            )
        except FastSAMError:
            return (
                SegmentationResult(
                    instances=[],
                    raw_result=raw_result,
                    infer_ms=infer_ms,
                    prompt_classes=[str(label)],
                    preprocessed_image=frame.copy(),
                ),
                None,
            )
        confidences = _result_confidences(raw_result, len(masks))
        bboxes = _result_bboxes(raw_result, len(masks), normalized_bbox)
        model_confidence = float(confidences[selected_index])
        instance = SegmentationInstance(
            mask=mask,
            score=model_confidence,
            class_id=0,
            class_name=str(label),
            bbox=bboxes[selected_index].astype(np.float32),
        )
        selection = FastSAMSelection(
            bbox=normalized_bbox,
            selected_index=selected_index,
            bbox_iou=bbox_iou,
            mask_pixels=int(mask.sum()),
            model_confidence=model_confidence,
        )
        return (
            SegmentationResult(
                instances=[instance],
                raw_result=raw_result,
                infer_ms=infer_ms,
                prompt_classes=[str(label)],
                preprocessed_image=frame.copy(),
            ),
            selection,
        )


class FastSAMSegmentationEngine:
    """Per-camera runtime bbox wrapper over a shared FastSAM model."""

    def __init__(
        self,
        shared_model: SharedFastSAMModel,
        *,
        bbox: Sequence[float] | None,
        label: str = DEFAULT_LABEL,
        min_bbox_size_px: int = MIN_BBOX_SIZE_PX,
    ) -> None:
        self.shared_model = shared_model
        self.model_name = shared_model.model_name
        self.prompt_classes = [str(label)]
        self.initial_bbox = (
            None if bbox is None else tuple(float(value) for value in bbox)
        )
        self.bbox = self.initial_bbox
        self.label = str(label)
        self.min_bbox_size_px = int(min_bbox_size_px)
        self.last_selection: FastSAMSelection | None = None
        self.last_preprocessed_frame: np.ndarray | None = None
        self.last_error: str | None = None

    def set_bbox(self, bbox: Sequence[float] | None) -> None:
        """Update the next frame's prompt; ``None`` suspends this camera."""
        if bbox is None:
            self.bbox = None
            return
        values = tuple(float(value) for value in bbox)
        if len(values) != 4:
            raise ValueError(f"FastSAM bbox must contain four XYXY values, got {bbox!r}")
        self.bbox = values

    def reset_bbox(self) -> None:
        """Restore the bootstrap bbox supplied at construction time."""
        self.bbox = self.initial_bbox

    def predict(self, frame_bgr: np.ndarray, **_kwargs: Any) -> SegmentationResult:
        frame = np.asarray(frame_bgr, dtype=np.uint8)
        if self.bbox is None:
            self.last_selection = None
            self.last_error = "suspended"
            self.last_preprocessed_frame = frame.copy()
            return SegmentationResult(
                instances=[],
                raw_result=None,
                infer_ms=0.0,
                prompt_classes=[self.label],
                preprocessed_image=self.last_preprocessed_frame,
            )
        try:
            result, selection = self.shared_model.predict(
                frame,
                bbox=self.bbox,
                label=self.label,
                min_bbox_size_px=self.min_bbox_size_px,
            )
        except Exception as exc:
            self.last_selection = None
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_preprocessed_frame = frame.copy()
            return SegmentationResult(
                instances=[],
                raw_result=None,
                infer_ms=0.0,
                prompt_classes=[self.label],
                preprocessed_image=self.last_preprocessed_frame,
            )
        self.last_selection = selection
        self.last_error = None if selection is not None else "no_mask"
        self.last_preprocessed_frame = result.preprocessed_image
        return result

    def render(self, segmentation_result: SegmentationResult) -> np.ndarray:
        raw = segmentation_result.raw_result
        if raw is not None and hasattr(raw, "plot"):
            return raw.plot()
        return np.asarray(segmentation_result.preprocessed_image, dtype=np.uint8).copy()


__all__ = [
    "DEFAULT_LABEL",
    "MIN_BBOX_SIZE_PX",
    "FastSAMError",
    "FastSAMSelection",
    "FastSAMSegmentationEngine",
    "SharedFastSAMModel",
    "binary_mask_iou",
    "masks_from_result",
    "normalize_bbox",
    "resolve_asset_path",
    "resolve_fastsam_device",
    "select_mask_by_bbox_iou",
]
