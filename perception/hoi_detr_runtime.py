"""HOI-DETR inference adapter and deterministic dynamic-bbox selection helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


CLASS_NAMES = ("hand", "firstobject", "secondobject")


@dataclass(frozen=True)
class HOIDETRCandidate:
    hand_bbox: tuple[float, float, float, float]
    object_bbox: tuple[float, float, float, float]
    hand_score: float
    object_score: float
    relation_score: float
    hand_side: str = "unknown"
    contact_state: str = "unknown"


@dataclass(frozen=True)
class SelectedHOIDETRBBox:
    bbox_xyxy: tuple[float, float, float, float]
    hand_score: float
    object_score: float
    relation_score: float
    hand_side: str = "unknown"
    contact_state: str = "unknown"


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


class HOIDETRBBoxSelector:
    """Choose one associated first-object bbox and stabilize it per camera."""

    def __init__(
        self,
        *,
        initial_bboxes: dict[int, Sequence[float]],
        padding_ratio: float = 0.10,
        ema_alpha: float = 0.60,
        min_bbox_size_px: int = 8,
    ) -> None:
        self.initial_bboxes = {
            int(camera_id): tuple(float(value) for value in bbox)
            for camera_id, bbox in initial_bboxes.items()
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
        candidates: Iterable[HOIDETRCandidate],
        *,
        width: int,
        height: int,
    ) -> SelectedHOIDETRBBox | None:
        camera_id = int(camera_id)
        candidates = list(candidates)
        if not candidates:
            return None
        reference = self._previous.get(camera_id, self.initial_bboxes.get(camera_id))

        def rank(candidate: HOIDETRCandidate):
            overlap = 0.0 if reference is None else bbox_iou(candidate.object_bbox, reference)
            distance = (
                0.0
                if reference is None
                else bbox_center_distance(candidate.object_bbox, reference)
            )
            return (
                int(overlap > 0.0),
                overlap,
                -distance,
                float(candidate.relation_score),
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
        return SelectedHOIDETRBBox(
            bbox_xyxy=padded,
            hand_score=float(chosen.hand_score),
            object_score=float(chosen.object_score),
            relation_score=float(chosen.relation_score),
        )


class HOIDETRPredictorAdapter:
    """Run the unmodified external HOI-DETR model on in-memory BGR frames."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        config_path: str | Path,
        weights_path: str | Path,
        hand_score_threshold: float = 0.3,
        first_object_score_threshold: float = 0.3,
        hand_first_relation_threshold: float = 0.6,
        nms_iou_threshold: float = 0.5,
        require_cuda: bool = True,
        device: str = "cuda:0",
        inference_backend: Any = None,
    ) -> None:
        self.hand_score_threshold = float(hand_score_threshold)
        self.first_object_score_threshold = float(first_object_score_threshold)
        self.hand_first_relation_threshold = float(hand_first_relation_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.inference_backend = inference_backend
        self.model = None
        self.test_pipeline = None
        self.interaction_branch = None
        self.device = str(device)
        if inference_backend is not None:
            return

        repo = Path(repo_path).expanduser().resolve()
        config = Path(config_path).expanduser().resolve()
        weights = Path(weights_path).expanduser().resolve()
        for label, path in (
            ("HOI-DETR repo", repo),
            ("HOI-DETR config", config),
            ("HOI-DETR weights", weights),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        import torch

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("HOI-DETR requires CUDA, but torch.cuda.is_available() is false")
        if not require_cuda and self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        from mmdet.apis import init_detector
        from mmdet.datasets import replace_ImageToTensor
        from mmdet.datasets.pipelines import Compose
        from projects import models as _projects_models  # noqa: F401

        self.model = init_detector(str(config), str(weights), device=self.device)
        self.model.eval()
        self.model_device = next(self.model.parameters()).device
        pipeline_cfg = deepcopy(self.model.cfg.data.test.pipeline)
        pipeline_cfg[0]["type"] = "LoadImageFromWebcam"
        self.test_pipeline = Compose(replace_ImageToTensor(pipeline_cfg))
        self.interaction_branch = self._find_interaction_branch(self.model.query_head)

    def predict(self, image_bgr: np.ndarray) -> list[HOIDETRCandidate]:
        image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got shape={image.shape}")
        if self.inference_backend is not None:
            detections, relation_scores = self.inference_backend(image)
            return self.candidates_from_detections(detections, relation_scores)
        detections, embeddings = self._run_detector(image)
        return self._build_candidates(detections, embeddings)

    def candidates_from_detections(
        self,
        detections: Sequence[dict[str, Any]],
        relation_scores: Sequence[Sequence[float]] | np.ndarray,
    ) -> list[HOIDETRCandidate]:
        hands = [d for d in detections if int(d["class_id"]) == 0]
        first_objects = [d for d in detections if int(d["class_id"]) == 1]
        scores = np.asarray(relation_scores, dtype=np.float64)
        expected_shape = (len(hands), len(first_objects))
        if scores.shape != expected_shape:
            raise ValueError(
                f"unexpected hand-first relation score shape: {scores.shape}, "
                f"expected {expected_shape}"
            )
        candidates: list[HOIDETRCandidate] = []
        for hand_index, hand in enumerate(hands):
            hand_score = float(hand["score"])
            if hand_score < self.hand_score_threshold:
                continue
            for object_index, first_object in enumerate(first_objects):
                object_score = float(first_object["score"])
                relation_score = float(scores[hand_index, object_index])
                if object_score < self.first_object_score_threshold:
                    continue
                if relation_score < self.hand_first_relation_threshold:
                    continue
                candidates.append(
                    HOIDETRCandidate(
                        hand_bbox=tuple(float(value) for value in hand["box"]),
                        object_bbox=tuple(float(value) for value in first_object["box"]),
                        hand_score=hand_score,
                        object_score=object_score,
                        relation_score=relation_score,
                    )
                )
        return candidates

    def _run_detector(self, image_bgr: np.ndarray):
        import torch
        from mmcv.parallel import collate, scatter
        from mmcv.ops import batched_nms
        from mmdet.core import bbox_cxcywh_to_xyxy

        data = self.test_pipeline(dict(img=image_bgr))
        data = collate([data], samples_per_gpu=1)
        target_device = self.model_device.index
        if target_device is None:
            raise RuntimeError("HOI-DETR inference requires a CUDA device")
        data = scatter(data, [target_device])[0]
        img_tensor = data["img"]
        if isinstance(img_tensor, list):
            img_tensor = img_tensor[0]
        img_metas = data["img_metas"]
        if isinstance(img_metas, list) and isinstance(img_metas[0], list):
            img_metas = img_metas[0]
        for meta in img_metas:
            meta["batch_input_shape"] = tuple(img_tensor.shape[-2:])

        with torch.no_grad():
            features = self.model.extract_feat(img_tensor)
            outputs, hidden_states = self.model.query_head(
                features, img_metas, return_hs=True
            )
        class_logits = outputs[0][-1][0]
        decoder_boxes = outputs[1][-1][0]
        embeddings = hidden_states[-1][0]
        scores_all = class_logits.sigmoid()
        max_per_img = int(self.model.query_head.test_cfg.get("max_per_img", 300))
        flat_scores, flat_indices = scores_all.reshape(-1).topk(max_per_img)
        flat_labels = flat_indices % self.model.query_head.num_classes
        flat_queries = flat_indices // self.model.query_head.num_classes

        image_height, image_width = img_metas[0]["img_shape"][:2]
        factor = decoder_boxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        topk_boxes = bbox_cxcywh_to_xyxy(decoder_boxes[flat_queries]) * factor
        nms_cfg = {
            "type": "soft_nms",
            "iou_threshold": self.nms_iou_threshold,
            "min_score": min(
                self.hand_score_threshold, self.first_object_score_threshold
            ),
        }
        _, keep = batched_nms(topk_boxes, flat_scores, flat_labels, nms_cfg)
        scale_factor = np.asarray(
            img_metas[0].get("scale_factor", 1.0), dtype=np.float32
        ).reshape(-1)
        if scale_factor.size == 1:
            scale_factor = np.tile(scale_factor, 4)
        elif scale_factor.size == 2:
            scale_factor = np.asarray(
                [scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]]
            )

        detections: list[dict[str, Any]] = []
        for kept_index in keep.tolist():
            class_id = int(flat_labels[kept_index].item())
            if class_id not in (0, 1):
                continue
            score = float(flat_scores[kept_index].item())
            threshold = (
                self.hand_score_threshold
                if class_id == 0
                else self.first_object_score_threshold
            )
            if score < threshold:
                continue
            box = topk_boxes[kept_index].detach().cpu().numpy() / scale_factor
            detections.append(
                {
                    "box": box,
                    "score": score,
                    "class_id": class_id,
                    "query_idx": int(flat_queries[kept_index].item()),
                }
            )
        return detections, embeddings

    def _build_candidates(self, detections, embeddings) -> list[HOIDETRCandidate]:
        import torch

        hands = [d for d in detections if int(d["class_id"]) == 0]
        first_objects = [d for d in detections if int(d["class_id"]) == 1]
        if not hands or not first_objects:
            return []
        pairs = [
            torch.cat(
                [embeddings[hand["query_idx"]], embeddings[obj["query_idx"]]], dim=0
            )
            for hand in hands
            for obj in first_objects
        ]
        with torch.no_grad():
            logits = self.interaction_branch.mlp(torch.stack(pairs, dim=0))
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            positive = logits.argmax(dim=-1) == 1
        scores = probabilities.reshape(len(hands), len(first_objects)).detach().cpu().numpy()
        positive_mask = positive.reshape(len(hands), len(first_objects)).detach().cpu().numpy()
        scores = np.where(positive_mask, scores, 0.0)
        return self.candidates_from_detections(detections, scores)

    @staticmethod
    def _find_interaction_branch(query_head: Any) -> Any:
        candidates = {
            name: module
            for name, module in query_head.named_modules()
            if "interaction_head" in name.lower() and name
        }
        if not candidates:
            raise AttributeError("interaction_head not found in HOI-DETR query head")
        return candidates[sorted(candidates, key=len)[0]]


__all__ = [
    "CLASS_NAMES",
    "HOIDETRBBoxSelector",
    "HOIDETRCandidate",
    "HOIDETRPredictorAdapter",
    "SelectedHOIDETRBBox",
    "bbox_center_distance",
    "bbox_iou",
    "clamp_and_pad_bbox",
]
