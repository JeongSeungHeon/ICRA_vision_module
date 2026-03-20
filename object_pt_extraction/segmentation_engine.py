from dataclasses import dataclass
import time

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class SegmentationInstance:
    mask: np.ndarray
    score: float
    class_id: int
    class_name: str
    bbox: np.ndarray


@dataclass
class SegmentationResult:
    instances: list
    raw_result: object
    infer_ms: float
    prompt_classes: list


def require_ultralytics():
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is required. Install it with `pip install ultralytics` and try again."
        )


def parse_prompt_classes(prompt_args):
    # `--prompt person bus` 와 `--prompt person,bus` 두 형태를 모두 지원한다.
    if not prompt_args:
        return []

    prompt_classes = []
    for raw_item in prompt_args:
        for class_name in raw_item.split(","):
            normalized = class_name.strip()
            if normalized:
                prompt_classes.append(normalized)
    return prompt_classes


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def format_instance_summary(instances, max_items=4):
    if not instances:
        return "detections: 0"

    counts = {}
    for instance in instances:
        counts[instance.class_name] = counts.get(instance.class_name, 0) + 1

    summary_parts = [f"{name}:{count}" for name, count in sorted(counts.items())]
    return "detections: " + ", ".join(summary_parts[:max_items])


def select_instances(instances, mode="all_instances", class_names=None):
    if class_names:
        class_name_set = set(class_names)
        instances = [instance for instance in instances if instance.class_name in class_name_set]

    if mode == "all_instances":
        return list(instances)
    if mode == "highest_score":
        if not instances:
            return []
        return [max(instances, key=lambda instance: instance.score)]
    if mode == "class_filter":
        return list(instances)

    raise ValueError(f"Unsupported selection mode: {mode}")


def _extract_instances(result):
    boxes = result.boxes
    masks = result.masks
    if boxes is None or len(boxes) == 0 or masks is None or masks.data is None:
        return []

    box_xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    mask_data = masks.data.detach().cpu().numpy()

    instance_count = min(len(box_xyxy), len(scores), len(class_ids), len(mask_data))
    instances = []
    for instance_index in range(instance_count):
        instances.append(
            SegmentationInstance(
                mask=mask_data[instance_index] > 0.5,
                score=float(scores[instance_index]),
                class_id=int(class_ids[instance_index]),
                class_name=get_class_name(result.names, int(class_ids[instance_index])),
                bbox=box_xyxy[instance_index].astype(np.float32),
            )
        )
    return instances


class SegmentationEngine:
    def __init__(
        self,
        model_name,
        prompt_classes=None,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        max_det=100,
        device=None,
        classes=None,
        half=False,
        retina_masks=True,
    ):
        require_ultralytics()

        self.model_name = model_name
        self.prompt_classes = list(prompt_classes or [])
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self.half = half
        self.retina_masks = retina_masks

        try:
            self.model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model `{model_name}`. Verify the weights path or model name."
            ) from exc

        if self.prompt_classes:
            # YOLOE는 텍스트 프롬프트를 한 번 설정해두고 이후 프레임들에 재사용한다.
            self.model.set_classes(self.prompt_classes)

    def predict(self, frame_bgr):
        infer_start = time.perf_counter()
        raw_result = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            classes=self.classes,
            retina_masks=self.retina_masks,
            verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        return SegmentationResult(
            instances=_extract_instances(raw_result),
            raw_result=raw_result,
            infer_ms=infer_ms,
            prompt_classes=list(self.prompt_classes),
        )

    def render(self, segmentation_result):
        # Ultralytics가 제공하는 기본 overlay를 그대로 재사용한다.
        return segmentation_result.raw_result.plot()

