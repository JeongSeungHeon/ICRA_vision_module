"""Hands23 inference adapter and deterministic dynamic-bbox selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np


CONTACT_STATES = {
    0: "no_contact",
    1: "other_person_contact",
    2: "self_contact",
    3: "object_contact",
    4: "obj_to_obj_contact",
}


@dataclass(frozen=True)
class Hands23Candidate:
    hand_bbox: tuple[float, float, float, float]
    object_bbox: tuple[float, float, float, float]
    hand_side: str
    hand_score: float
    object_score: float
    contact_state: str


@dataclass(frozen=True)
class SelectedHands23BBox:
    bbox_xyxy: tuple[float, float, float, float]
    hand_side: str
    hand_score: float
    object_score: float
    contact_state: str


def normalize_hand_side(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"left", "left_hand"}:
        return "left"
    if normalized in {"right", "right_hand"}:
        return "right"
    return "unknown"


def expected_hand_sides_from_config(config: dict[str, Any]) -> dict[int, str]:
    """Extract cam0:left/cam1:right preferences from active-hand configuration."""
    values = (
        config.get("perception", {})
        .get("hand_selection", {})
        .get("allowed_active_candidate_ids", [])
        or []
    )
    result: dict[int, str] = {}
    for value in values:
        match = re.fullmatch(r"cam(\d+):(left|right)", str(value).strip().lower())
        if match is not None:
            result[int(match.group(1))] = match.group(2)
    return result


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def bbox_center_distance(first: Sequence[float], second: Sequence[float]) -> float:
    first_values = np.asarray(first, dtype=np.float64).reshape(4)
    second_values = np.asarray(second, dtype=np.float64).reshape(4)
    first_center = (first_values[:2] + first_values[2:]) * 0.5
    second_center = (second_values[:2] + second_values[2:]) * 0.5
    return float(np.linalg.norm(first_center - second_center))


def clamp_and_pad_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    padding_ratio: float,
    min_size_px: int,
) -> tuple[float, float, float, float] | None:
    values = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if len(values) != 4 or not np.all(np.isfinite(values)):
        return None
    x1, y1, x2, y2 = (float(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = (x2 - x1) * max(float(padding_ratio), 0.0)
    pad_y = (y2 - y1) * max(float(padding_ratio), 0.0)
    x1 = max(0.0, min(float(width), x1 - pad_x))
    y1 = max(0.0, min(float(height), y1 - pad_y))
    x2 = max(0.0, min(float(width), x2 + pad_x))
    y2 = max(0.0, min(float(height), y2 + pad_y))
    if x2 - x1 < int(min_size_px) or y2 - y1 < int(min_size_px):
        return None
    return x1, y1, x2, y2


class Hands23BBoxSelector:
    """Choose one associated first-object bbox and stabilize it per camera."""

    def __init__(
        self,
        *,
        initial_bboxes: dict[int, Sequence[float]],
        expected_hand_sides: dict[int, str] | None = None,
        padding_ratio: float = 0.10,
        ema_alpha: float = 0.60,
        min_bbox_size_px: int = 8,
    ) -> None:
        self.initial_bboxes = {
            int(camera_id): tuple(float(value) for value in bbox)
            for camera_id, bbox in initial_bboxes.items()
        }
        self.expected_hand_sides = {
            int(camera_id): normalize_hand_side(side)
            for camera_id, side in (expected_hand_sides or {}).items()
        }
        self.padding_ratio = max(float(padding_ratio), 0.0)
        self.ema_alpha = min(max(float(ema_alpha), 0.0), 1.0)
        self.min_bbox_size_px = max(int(min_bbox_size_px), 1)
        self._previous: dict[int, tuple[float, float, float, float]] = {}

    def reset(self) -> None:
        self._previous.clear()

    def select(
        self,
        camera_id: int,
        candidates: Iterable[Hands23Candidate],
        *,
        width: int,
        height: int,
    ) -> SelectedHands23BBox | None:
        camera_id = int(camera_id)
        candidates = list(candidates)
        if not candidates:
            return None
        reference = self._previous.get(camera_id, self.initial_bboxes.get(camera_id))
        expected_side = self.expected_hand_sides.get(camera_id, "unknown")

        def rank(candidate: Hands23Candidate):
            side_match = int(
                expected_side == "unknown"
                or normalize_hand_side(candidate.hand_side) == expected_side
            )
            overlap = 0.0 if reference is None else bbox_iou(candidate.object_bbox, reference)
            distance = (
                0.0
                if reference is None
                else bbox_center_distance(candidate.object_bbox, reference)
            )
            return (
                side_match,
                int(overlap > 0.0),
                overlap,
                -distance,
                float(candidate.object_score),
                float(candidate.hand_score),
            )

        chosen = max(candidates, key=rank)
        padded = clamp_and_pad_bbox(
            chosen.object_bbox,
            width=width,
            height=height,
            padding_ratio=self.padding_ratio,
            min_size_px=self.min_bbox_size_px,
        )
        if padded is None:
            return None

        previous = self._previous.get(camera_id)
        if previous is not None:
            alpha = self.ema_alpha
            padded_array = alpha * np.asarray(padded) + (1.0 - alpha) * np.asarray(previous)
            padded = tuple(float(value) for value in padded_array)
        self._previous[camera_id] = padded
        return SelectedHands23BBox(
            bbox_xyxy=padded,
            hand_side=normalize_hand_side(chosen.hand_side),
            hand_score=float(chosen.hand_score),
            object_score=float(chosen.object_score),
            contact_state=str(chosen.contact_state),
        )


class Hands23PredictorAdapter:
    """Thin adapter around the unmodified external Hands23 Detectron2 checkout."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        config_path: str | Path,
        weights_path: str | Path,
        hand_threshold: float = 0.7,
        first_object_threshold: float = 0.5,
        second_object_threshold: float = 0.3,
        hand_relation_threshold: float = 0.3,
        object_relation_threshold: float = 0.7,
        min_size_test: int = 640,
        require_cuda: bool = True,
        predictor: Any = None,
    ) -> None:
        self.hand_threshold = float(hand_threshold)
        self.first_object_threshold = float(first_object_threshold)
        self.min_size_test = max(1, int(min_size_test))
        if predictor is not None:
            self.predictor = predictor
            return

        repo = Path(repo_path).expanduser().resolve()
        config = Path(config_path).expanduser().resolve()
        weights = Path(weights_path).expanduser().resolve()
        for label, path in (
            ("Hands23 repo", repo),
            ("Hands23 config", config),
            ("Hands23 weights", weights),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        import torch

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "Hands23 requires CUDA, but torch.cuda.is_available() is false"
            )
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        # Import registers hoRCNNROIHeads with Detectron2.
        import hodetector.modeling.roi_heads  # noqa: F401

        cfg = get_cfg()
        cfg.merge_from_file(str(config))
        cfg.INPUT.MIN_SIZE_TEST = self.min_size_test
        cfg.MODEL.WEIGHTS = str(weights)
        cfg.HAND = float(hand_threshold)
        cfg.FIRSTOBJ = float(first_object_threshold)
        cfg.SECONDOBJ = float(second_object_threshold)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(
            float(hand_threshold),
            float(first_object_threshold),
            float(second_object_threshold),
        )
        cfg.HAND_RELA = float(hand_relation_threshold)
        cfg.OBJ_RELA = float(object_relation_threshold)
        cfg.MODEL.DEVICE = "cuda" if require_cuda else str(cfg.MODEL.DEVICE)
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)

    def predict(self, image_bgr: np.ndarray) -> list[Hands23Candidate]:
        outputs = self.predictor(np.asarray(image_bgr, dtype=np.uint8))
        return self.candidates_from_outputs(outputs)

    def candidates_from_outputs(self, outputs: Any) -> list[Hands23Candidate]:
        instances = outputs["instances"]
        boxes = self._to_numpy(instances.get("pred_boxes").tensor).reshape((-1, 4))
        classes = self._to_numpy(instances.get("pred_classes")).reshape(-1)
        scores = self._to_numpy(instances.get("scores")).reshape(-1)
        pred_dz = self._to_numpy(instances.get("pred_dz"))
        if pred_dz.ndim != 2 or pred_dz.shape[1] < 9:
            raise ValueError(f"Unexpected Hands23 pred_dz shape: {pred_dz.shape}")

        candidates: list[Hands23Candidate] = []
        for index, class_id in enumerate(classes):
            if int(class_id) != 0 or float(scores[index]) < self.hand_threshold:
                continue
            object_index = int(round(float(pred_dz[index, 4])))
            if object_index < 0 or object_index >= len(boxes):
                continue
            if int(classes[object_index]) != 1:
                continue
            if float(scores[object_index]) < self.first_object_threshold:
                continue
            side = "right" if int(round(float(pred_dz[index, 5]))) == 1 else "left"
            contact_id = int(round(float(pred_dz[index, 8])))
            candidates.append(
                Hands23Candidate(
                    hand_bbox=tuple(float(value) for value in boxes[index]),
                    object_bbox=tuple(float(value) for value in boxes[object_index]),
                    hand_side=side,
                    hand_score=float(scores[index]),
                    object_score=float(scores[object_index]),
                    contact_state=CONTACT_STATES.get(contact_id, f"unknown_{contact_id}"),
                )
            )
        return candidates

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)


__all__ = [
    "Hands23BBoxSelector",
    "Hands23Candidate",
    "Hands23PredictorAdapter",
    "SelectedHands23BBox",
    "bbox_center_distance",
    "bbox_iou",
    "clamp_and_pad_bbox",
    "expected_hand_sides_from_config",
    "normalize_hand_side",
]
