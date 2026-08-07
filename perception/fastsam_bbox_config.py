"""Persistent FastSAM bbox selection and runtime config override helpers.

The interactive selector writes a small runtime YAML file outside the tracked
configuration. The standalone startup validates that file against the active
camera configuration, then materializes one content-addressed effective config
for both bootstrap capture and live perception.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import yaml

SELECTION_SCHEMA_VERSION = 1
DEFAULT_SELECTION_PATH = "output/sam3d/fastsam_bbox.yaml"
CAMERA_NAMES = ("cam0", "cam1")


class FastSAMBBoxSelectionError(ValueError):
    """Raised when a saved selection cannot be safely applied."""


def resolve_config_file(path_or_dir: str | Path) -> str:
    """Resolve the standalone YAML configuration path."""
    candidate = Path(path_or_dir).expanduser()
    if candidate.is_dir():
        candidate = candidate / "handover.yaml"
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FastSAMBBoxSelectionError(f"config file does not exist: {candidate}")
    return str(candidate)


def load_config(path_or_dir: str | Path) -> dict[str, Any]:
    resolved = resolve_config_file(path_or_dir)
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise FastSAMBBoxSelectionError(f"cannot read config {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FastSAMBBoxSelectionError(f"config root must be a mapping: {resolved}")
    return payload


@dataclass(frozen=True)
class EffectiveBBoxConfig:
    config_path: str
    selection_applied: bool
    bboxes_xyxy: dict[str, tuple[int, int, int, int]]
    fallback_reason: str = ""


def _finite_float(value: Any, *, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise FastSAMBBoxSelectionError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(converted):
        raise FastSAMBBoxSelectionError(f"{field} must be finite, got {value!r}")
    return converted


def normalize_bbox(
    values: Sequence[Any],
    width: int,
    height: int,
    *,
    min_size_px: int = 8,
) -> tuple[int, int, int, int]:
    """Convert, clamp, and validate an XYXY bbox without importing OpenCV."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 4:
        raise FastSAMBBoxSelectionError(
            f"bbox_xyxy must contain exactly four values, got {values!r}"
        )
    converted = [
        int(round(_finite_float(value, field=f"bbox_xyxy[{index}]")))
        for index, value in enumerate(values)
    ]
    x1, y1, x2, y2 = converted
    if x2 <= x1 or y2 <= y1:
        raise FastSAMBBoxSelectionError(
            f"bbox_xyxy must satisfy x1 < x2 and y1 < y2, got {converted!r}"
        )

    width = max(1, int(width))
    height = max(1, int(height))
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    min_size_px = max(1, int(min_size_px))
    if x2 - x1 < min_size_px or y2 - y1 < min_size_px:
        raise FastSAMBBoxSelectionError(
            "bbox_xyxy is outside the frame or too small after clamping: "
            f"{(x1, y1, x2, y2)}, minimum={min_size_px}px"
        )
    return x1, y1, x2, y2


def xywh_to_xyxy(
    values: Sequence[Any],
    width: int,
    height: int,
    *,
    min_size_px: int = 8,
) -> tuple[int, int, int, int]:
    """Convert an OpenCV-style XYWH drag result into a validated XYXY bbox."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 4:
        raise FastSAMBBoxSelectionError(
            f"ROI must contain exactly four XYWH values, got {values!r}"
        )
    x, y, roi_width, roi_height = (
        _finite_float(value, field=f"roi_xywh[{index}]")
        for index, value in enumerate(values)
    )
    if roi_width <= 0 or roi_height <= 0:
        raise FastSAMBBoxSelectionError(f"ROI width and height must be positive, got {values!r}")
    return normalize_bbox(
        (x, y, x + roi_width, y + roi_height),
        width,
        height,
        min_size_px=min_size_px,
    )


def _camera_dimensions(config: Mapping[str, Any], camera_name: str) -> tuple[int, int]:
    camera_cfg = (config.get("cameras", {}) or {}).get(camera_name, {}) or {}
    try:
        width = int(camera_cfg.get("width", 640))
        height = int(camera_cfg.get("height", 480))
    except (TypeError, ValueError) as exc:
        raise FastSAMBBoxSelectionError(
            f"active config is missing a valid {camera_name} width/height"
        ) from exc
    if width <= 0 or height <= 0:
        raise FastSAMBBoxSelectionError(
            f"active config has invalid {camera_name} resolution: {width}x{height}"
        )
    return width, height


def validate_selection(
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, tuple[int, int, int, int]]:
    """Validate the saved selection against the active camera configuration."""
    if not isinstance(selection, Mapping):
        raise FastSAMBBoxSelectionError("selection YAML root must be a mapping")
    try:
        schema_version = int(selection.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise FastSAMBBoxSelectionError(
            "selection schema_version must be an integer"
        ) from exc
    if schema_version != SELECTION_SCHEMA_VERSION:
        raise FastSAMBBoxSelectionError(
            "unsupported FastSAM bbox selection schema_version: "
            f"{selection.get('schema_version')!r}"
        )
    if str(selection.get("backend", "")).strip().lower() != "sam3d":
        raise FastSAMBBoxSelectionError("selection backend must be 'sam3d'")
    if not str(selection.get("selected_at_utc", "")).strip():
        raise FastSAMBBoxSelectionError("selection is missing selected_at_utc")
    if not str(selection.get("fastsam_model", "")).strip():
        raise FastSAMBBoxSelectionError("selection is missing fastsam_model")

    camera_entries = selection.get("cameras")
    if not isinstance(camera_entries, Mapping):
        raise FastSAMBBoxSelectionError("selection is missing cameras mapping")

    fastsam_cfg = (
        (config.get("perception", {}) or {}).get("object", {}) or {}
    ).get("fastsam", {}) or {}
    min_size_px = max(1, int(fastsam_cfg.get("min_bbox_size_px", 8)))
    min_mask_pixels = max(1, int(fastsam_cfg.get("min_mask_pixels", 300)))
    configured_cameras = config.get("cameras", {}) or {}
    normalized: dict[str, tuple[int, int, int, int]] = {}

    for camera_name in CAMERA_NAMES:
        entry = camera_entries.get(camera_name)
        if not isinstance(entry, Mapping):
            raise FastSAMBBoxSelectionError(
                f"selection is missing camera entry: {camera_name}"
            )
        width, height = _camera_dimensions(config, camera_name)
        try:
            selected_width = int(entry["width"])
            selected_height = int(entry["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FastSAMBBoxSelectionError(
                f"{camera_name} selection is missing a valid width/height"
            ) from exc
        if (selected_width, selected_height) != (width, height):
            raise FastSAMBBoxSelectionError(
                f"{camera_name} resolution mismatch: selection="
                f"{selected_width}x{selected_height}, config={width}x{height}"
            )

        configured_serial = str(
            (configured_cameras.get(camera_name, {}) or {}).get("serial", "")
        ).strip()
        selected_serial = str(entry.get("serial", "")).strip()
        if not selected_serial:
            raise FastSAMBBoxSelectionError(f"{camera_name} selection is missing serial")
        if configured_serial and selected_serial != configured_serial:
            raise FastSAMBBoxSelectionError(
                f"{camera_name} serial mismatch: selection={selected_serial}, "
                f"config={configured_serial}"
            )

        try:
            mask_pixels = int(entry["mask_pixels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FastSAMBBoxSelectionError(
                f"{camera_name} selection is missing mask_pixels"
            ) from exc
        if mask_pixels < min_mask_pixels:
            raise FastSAMBBoxSelectionError(
                f"{camera_name} mask is too small: {mask_pixels} < {min_mask_pixels}"
            )

        bbox_iou = _finite_float(entry.get("bbox_iou"), field=f"{camera_name}.bbox_iou")
        confidence = _finite_float(
            entry.get("model_confidence"),
            field=f"{camera_name}.model_confidence",
        )
        if not 0.0 <= bbox_iou <= 1.0:
            raise FastSAMBBoxSelectionError(
                f"{camera_name}.bbox_iou must be within [0, 1], got {bbox_iou}"
            )
        if not 0.0 <= confidence <= 1.0:
            raise FastSAMBBoxSelectionError(
                f"{camera_name}.model_confidence must be within [0, 1], got {confidence}"
            )

        normalized[camera_name] = normalize_bbox(
            entry.get("bbox_xyxy"),
            width,
            height,
            min_size_px=min_size_px,
        )
    return normalized


def build_selection_payload(
    config: Mapping[str, Any],
    *,
    fastsam_model: str,
    camera_results: Mapping[str, Mapping[str, Any]],
    selected_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate the persisted selection document."""
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "backend": "sam3d",
        "selected_at_utc": selected_at_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fastsam_model": str(fastsam_model),
        "cameras": {
            camera_name: {
                "serial": str(camera_results[camera_name]["serial"]),
                "width": int(camera_results[camera_name]["width"]),
                "height": int(camera_results[camera_name]["height"]),
                "bbox_xyxy": [
                    int(value) for value in camera_results[camera_name]["bbox_xyxy"]
                ],
                "mask_pixels": int(camera_results[camera_name]["mask_pixels"]),
                "bbox_iou": float(camera_results[camera_name]["bbox_iou"]),
                "model_confidence": float(
                    camera_results[camera_name]["model_confidence"]
                ),
            }
            for camera_name in CAMERA_NAMES
        },
    }
    validate_selection(payload, config)
    return payload


def load_selection(path: str | Path) -> dict[str, Any]:
    selection_path = Path(path).expanduser()
    try:
        with open(selection_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise FastSAMBBoxSelectionError(
            f"cannot read FastSAM bbox selection: {selection_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FastSAMBBoxSelectionError(
            f"FastSAM bbox selection root must be a mapping: {selection_path}"
        )
    return payload


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def atomic_write_selection(path: str | Path, selection: Mapping[str, Any]) -> Path:
    """Atomically replace the runtime selection file."""
    selection_path = Path(path).expanduser()
    payload = yaml.safe_dump(
        dict(selection),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return _atomic_write_text(selection_path, payload)


def apply_selection_to_config(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[int, int, int, int]]]:
    bboxes = validate_selection(selection, config)
    effective = deepcopy(dict(config))
    fastsam_cfg = (
        effective.setdefault("perception", {})
        .setdefault("object", {})
        .setdefault("fastsam", {})
    )
    fastsam_cfg["bboxes"] = {
        camera_name: list(bboxes[camera_name]) for camera_name in CAMERA_NAMES
    }
    fastsam_cfg["bbox_selection"] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selected_at_utc": str(selection["selected_at_utc"]),
    }
    return effective, bboxes


def _configured_bboxes(
    config: Mapping[str, Any],
) -> dict[str, tuple[int, int, int, int]]:
    fastsam_cfg = (
        (config.get("perception", {}) or {}).get("object", {}) or {}
    ).get("fastsam", {}) or {}
    raw_bboxes = fastsam_cfg.get("bboxes", {}) or {}
    min_size_px = max(1, int(fastsam_cfg.get("min_bbox_size_px", 8)))
    normalized = {}
    for camera_name in CAMERA_NAMES:
        if camera_name not in raw_bboxes:
            raise FastSAMBBoxSelectionError(
                f"active config is missing FastSAM bbox: {camera_name}"
            )
        width, height = _camera_dimensions(config, camera_name)
        normalized[camera_name] = normalize_bbox(
            raw_bboxes[camera_name],
            width,
            height,
            min_size_px=min_size_px,
        )
    return normalized


def _write_content_addressed_config(
    config: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
) -> str:
    canonical = yaml.safe_dump(
        dict(config),
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    target_dir = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path(tempfile.gettempdir())
    )
    target = target_dir / f"handover_sam3d_bbox_{digest}.yaml"
    if not target.is_file():
        body = (
            "# Runtime config with interactive FastSAM bbox override.\n"
            + yaml.safe_dump(
                dict(config),
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
        )
        _atomic_write_text(target, body)
    return str(target)


def resolve_effective_config(
    base_config_path_or_dir: str | Path,
    selection_path: str | Path,
    *,
    required: bool = True,
    cache_dir: str | Path | None = None,
) -> EffectiveBBoxConfig:
    """Resolve one config containing the selected bboxes.

    With ``required=False``, a missing or invalid runtime selection explicitly
    falls back to the configured cam0/cam1 bboxes.
    """
    base_config = load_config(base_config_path_or_dir)
    configured_bboxes = _configured_bboxes(base_config)
    path = Path(selection_path).expanduser()
    try:
        if not path.is_file():
            raise FastSAMBBoxSelectionError(
                f"FastSAM bbox selection file does not exist: {path}"
            )
        selection = load_selection(path)
        effective, bboxes = apply_selection_to_config(base_config, selection)
    except FastSAMBBoxSelectionError as exc:
        if required:
            raise
        return EffectiveBBoxConfig(
            config_path=resolve_config_file(base_config_path_or_dir),
            selection_applied=False,
            bboxes_xyxy=configured_bboxes,
            fallback_reason=str(exc),
        )

    return EffectiveBBoxConfig(
        config_path=_write_content_addressed_config(
            effective,
            cache_dir=cache_dir,
        ),
        selection_applied=True,
        bboxes_xyxy=bboxes,
    )


__all__ = [
    "CAMERA_NAMES",
    "DEFAULT_SELECTION_PATH",
    "EffectiveBBoxConfig",
    "FastSAMBBoxSelectionError",
    "SELECTION_SCHEMA_VERSION",
    "apply_selection_to_config",
    "atomic_write_selection",
    "build_selection_payload",
    "load_selection",
    "normalize_bbox",
    "resolve_effective_config",
    "validate_selection",
    "xywh_to_xyxy",
]
