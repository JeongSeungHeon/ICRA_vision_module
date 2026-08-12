from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_IMAGE_NAMES = ("left2.png", "right2.png")
DEFAULT_PROMPT = "cup"

DEFAULT_MODEL_CANDIDATES = [
    Path.cwd() / "yoloe-26l-seg.pt",
    PROJECT_DIR / "yoloe-26l-seg.pt",
    Path.home() / "ICRA_vision_module/yoloe-26l-seg.pt",
    Path.home() / "handover_2026_ICRA/ICRA_vision_module/yoloe-26l-seg.pt",
    Path.home() / "Downloads/yolo-seg/yoloe-26l-seg.pt",
    Path("/home/ur5/ICRA_vision_module/yoloe-26l-seg.pt"),
]

DEFAULT_TEXT_ENCODER_CANDIDATES = [
    Path.cwd() / "mobileclip2_b.ts",
    PROJECT_DIR / "mobileclip2_b.ts",
    Path.home() / "ICRA_vision_module/mobileclip2_b.ts",
    Path.home() / "handover_2026_ICRA/ICRA_vision_module/mobileclip2_b.ts",
    Path.home() / "Downloads/yolo-seg/mobileclip2_b.ts",
    Path("/home/ur5/ICRA_vision_module/mobileclip2_b.ts"),
]

OVERLAY_COLOR_BGR = (0, 255, 0)
BOX_COLOR_BGR = (0, 180, 255)
CONTOUR_COLOR_BGR = (0, 255, 255)


def require_ultralytics() -> None:
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is required. Activate the correct environment or install `ultralytics`."
        )


def parse_prompt_classes(prompt_args: Optional[Iterable[str]]) -> list[str]:
    if not prompt_args:
        return []

    prompt_classes: list[str] = []
    for raw_item in prompt_args:
        for class_name in raw_item.split(","):
            normalized = class_name.strip()
            if normalized:
                prompt_classes.append(normalized)
    return prompt_classes


def resolve_model_path(model_arg: Optional[str]) -> Union[Path, str]:
    if model_arg:
        candidate = Path(model_arg).expanduser()
        return candidate.resolve() if candidate.exists() else model_arg

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    return "yoloe-26l-seg.pt"


def ensure_local_text_encoder_asset() -> Optional[Path]:
    filename = "mobileclip2_b.ts"
    cwd_asset = Path.cwd() / filename
    if cwd_asset.exists():
        return cwd_asset

    if cwd_asset.is_symlink():
        cwd_asset.unlink()

    for candidate in DEFAULT_TEXT_ENCODER_CANDIDATES:
        if not candidate.exists():
            continue
        if candidate.resolve() == cwd_asset:
            return cwd_asset

        try:
            cwd_asset.symlink_to(candidate)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(candidate, cwd_asset)
        return cwd_asset if cwd_asset.exists() else candidate

    return None


def build_yoloe_model(model_name: Union[str, Path]):
    require_ultralytics()

    try:
        from ultralytics import YOLOE

        return YOLOE(str(model_name))
    except ImportError:
        return YOLO(str(model_name))
    except Exception:
        return YOLO(str(model_name))


def configure_yoloe_prompt(model, prompt_classes: list[str]) -> None:
    if not prompt_classes:
        return

    if hasattr(model, "get_text_pe") and hasattr(model, "set_classes"):
        try:
            text_pe = model.get_text_pe(prompt_classes)
            model.set_classes(prompt_classes, text_pe)
            return
        except Exception:
            pass

    if hasattr(model, "set_classes"):
        model.set_classes(prompt_classes)
        return

    raise RuntimeError(
        "The loaded Ultralytics model does not support text-prompt class configuration."
    )


def get_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def extract_instances(result, image_shape: tuple[int, int]) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None or result.masks.data is None:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = result.masks.data.detach().cpu().numpy() > 0.5

    image_height, image_width = image_shape
    instances: list[dict] = []
    count = min(len(boxes_xyxy), len(scores), len(class_ids), len(masks))
    for index in range(count):
        mask = masks[index]
        if mask.shape != (image_height, image_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        class_id = int(class_ids[index])
        instances.append(
            {
                "mask": mask,
                "score": float(scores[index]),
                "class_id": class_id,
                "class_name": get_class_name(result.names, class_id),
                "bbox": boxes_xyxy[index].astype(np.float32),
            }
        )

    return instances


def combine_masks(instances: list[dict], image_shape: tuple[int, int]) -> np.ndarray:
    combined_mask = np.zeros(image_shape, dtype=bool)
    for instance in instances:
        mask = instance["mask"]
        if mask.shape == combined_mask.shape:
            combined_mask |= mask
    return combined_mask


def get_mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def select_largest_mask_instance(instances: list[dict]) -> list[dict]:
    if not instances:
        return []
    return [max(instances, key=lambda instance: get_mask_area(instance["mask"]))]


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return mask

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas) + 1)
    return labels == largest_label


def draw_label(image_bgr: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    x, y = origin
    cv2.putText(
        image_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def render_overlay(
    image_bgr: np.ndarray,
    instances: list[dict],
    combined_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    overlay = image_bgr.copy()
    color_layer = np.full_like(image_bgr, OVERLAY_COLOR_BGR)
    overlay[combined_mask] = cv2.addWeighted(
        image_bgr[combined_mask],
        1.0 - alpha,
        color_layer[combined_mask],
        alpha,
        0.0,
    )

    mask_u8 = (combined_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, CONTOUR_COLOR_BGR, 2, lineType=cv2.LINE_AA)

    for instance in instances:
        x1, y1, x2, y2 = instance["bbox"].astype(int)
        #cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOR_BGR, 1, lineType=cv2.LINE_AA)
        #label = f"{instance['class_name']} {instance['score']:.2f}"
        #draw_label(overlay, label, (x1, max(18, y1 - 6)))
        pass

    return overlay


def render_transparent_mask(combined_mask: np.ndarray, alpha: float) -> np.ndarray:
    mask_layer = np.zeros((*combined_mask.shape, 4), dtype=np.uint8)
    mask_layer[combined_mask, :3] = OVERLAY_COLOR_BGR
    mask_layer[combined_mask, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)
    return mask_layer


def render_transparent_contour(combined_mask: np.ndarray) -> np.ndarray:
    contour_layer = np.zeros((*combined_mask.shape, 4), dtype=np.uint8)
    mask_u8 = (combined_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(
        contour_layer,
        contours,
        -1,
        (*CONTOUR_COLOR_BGR, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return contour_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOE text-prompt segmentation on static images and save masks."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where binary masks and overlays will be saved.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=DEFAULT_IMAGE_NAMES,
        help="Image filenames inside --input-dir.",
    )
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=[DEFAULT_PROMPT],
        help="YOLOE open-vocabulary prompt, e.g. --prompt cup or --prompt cup,bottle.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLOE weights path. Defaults to the first discovered local yoloe-26l-seg.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Max detections per image.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference when supported.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity.")
    return parser.parse_args()


def process_image(
    image_path: Path,
    output_dir: Path,
    model,
    args: argparse.Namespace,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        retina_masks=True,
        verbose=False,
    )[0]

    image_height, image_width = image.shape[:2]
    instances = extract_instances(result, (image_height, image_width))
    detected_instances = len(instances)
    instances = select_largest_mask_instance(instances)
    combined_mask = combine_masks(instances, (image_height, image_width))
    combined_mask = keep_largest_connected_component(combined_mask)
    binary_mask = (combined_mask.astype(np.uint8) * 255)
    overlay = render_overlay(image, instances, combined_mask, args.alpha)
    transparent_mask = render_transparent_mask(combined_mask, args.alpha)
    transparent_contour = render_transparent_contour(combined_mask)

    prompt_name = "_".join(parse_prompt_classes(args.prompt)) or "all"
    prompt_name = prompt_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    mask_path = output_dir / f"{image_path.stem}_{prompt_name}_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_{prompt_name}_overlay.png"
    transparent_mask_path = output_dir / f"{image_path.stem}_{prompt_name}_green_mask.png"
    transparent_contour_path = output_dir / f"{image_path.stem}_{prompt_name}_yellow_contour.png"

    if not cv2.imwrite(str(mask_path), binary_mask):
        raise RuntimeError(f"Failed to save binary mask: {mask_path}")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to save overlay image: {overlay_path}")
    if not cv2.imwrite(str(transparent_mask_path), transparent_mask):
        raise RuntimeError(f"Failed to save transparent mask: {transparent_mask_path}")
    if not cv2.imwrite(str(transparent_contour_path), transparent_contour):
        raise RuntimeError(f"Failed to save transparent contour: {transparent_contour_path}")

    print(
        f"[SAVE] {mask_path} size={image_width}x{image_height}, "
        f"detections={detected_instances}, kept={len(instances)}, "
        f"kept_area={get_mask_area(combined_mask)}"
    )
    print(f"[SAVE] {overlay_path} size={image_width}x{image_height}")
    print(f"[SAVE] {transparent_mask_path} size={image_width}x{image_height}")
    print(f"[SAVE] {transparent_contour_path} size={image_width}x{image_height}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_classes = parse_prompt_classes(args.prompt)
    if prompt_classes:
        local_text_encoder = ensure_local_text_encoder_asset()
        if local_text_encoder is not None:
            os.environ.setdefault("ULTRALYTICS_TEXT_ENCODER", str(local_text_encoder))

    model = build_yoloe_model(resolve_model_path(args.model))
    configure_yoloe_prompt(model, prompt_classes)

    for image_name in args.images:
        process_image(input_dir / image_name, output_dir, model, args)


if __name__ == "__main__":
    main()
