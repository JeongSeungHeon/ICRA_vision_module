"""Standalone SAM3D backend construction and startup bootstrap helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from calibration.extrinsics import load_transform_chain
from object_pt_extraction.fastsam_engine import (
    DEFAULT_LABEL,
    FastSAMSegmentationEngine,
    SharedFastSAMModel,
)
from perception.fastsam_bbox_config import (
    DEFAULT_SELECTION_PATH,
    EffectiveBBoxConfig,
    resolve_effective_config,
)
from perception.hands23_ipc import validate_hands23_assets
from perception.object_worker import ObjectWorkerCam0, ObjectWorkerCam1
from perception.sam3d_runtime import validate_template_points
from perception.shape_fitting_tracker_v2 import ShapeFittingTracker


REPO_ROOT = Path(__file__).resolve().parents[1]


def validate_main_runtime(*, require_rtde: bool) -> None:
    """Fail before camera/robot startup when the main CUDA runtime is incomplete."""
    required_modules = ("cv2", "mediapipe", "open3d", "pyrealsense2", "torch", "ultralytics")
    failures = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if require_rtde:
        for module_name in ("rtde_control", "rtde_receive"):
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("Main perception runtime is incomplete: " + "; ".join(failures))
    torch = importlib.import_module("torch")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(
            "SAM3D/FastSAM requires a CUDA-enabled main PyTorch runtime; "
            "torch.cuda.is_available() is false"
        )


class NoOpObjectClassLock:
    """Class-lock compatibility shim for the classless SAM3D object label."""

    locked_class = None

    def reset(self) -> None:
        self.locked_class = None

    def process_states(self, *, object_cam0, object_cam1, **_kwargs):
        return object_cam0, object_cam1


def resolve_repo_path(path_like: str | Path, *, repo_root: str | Path = REPO_ROOT) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path(repo_root).expanduser().resolve() / path
    return path.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return payload


def validate_hands23_runtime_assets(
    config_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Path]:
    """Fail before SAM3D capture when the isolated Hands23 checkout is incomplete."""
    repo_root = Path(repo_root).expanduser().resolve()
    config = load_config(config_path)
    runtime_cfg = config.get("runtime", {}).get("hands23_sidecar", {}) or {}
    return validate_hands23_assets(
        python_interpreter=resolve_repo_path(
            runtime_cfg.get(
                "python_interpreter",
                "/home/ur5/miniforge3/envs/hands23_ros2/bin/python",
            ),
            repo_root=repo_root,
        ),
        sidecar_script=repo_root / "tools" / "hands23_sidecar.py",
        config_path=Path(config_path).expanduser().resolve(),
        repo_path=resolve_repo_path(
            runtime_cfg.get("repo_path", "external/hands23_detector"),
            repo_root=repo_root,
        ),
        detector_config_path=resolve_repo_path(
            runtime_cfg.get(
                "config_path",
                "external/hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml",
            ),
            repo_root=repo_root,
        ),
        weights_path=resolve_repo_path(
            runtime_cfg.get(
                "weights_path",
                "external/hands23_detector/model_weights/model_hands23.pth",
            ),
            repo_root=repo_root,
        ),
    )


def resolve_sam3d_effective_config(
    config_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> EffectiveBBoxConfig:
    config = load_config(config_path)
    sam_runtime = config.get("runtime", {}).get("sam3d_server", {}) or {}
    selection_path = resolve_repo_path(
        sam_runtime.get("fastsam_bbox_file", DEFAULT_SELECTION_PATH),
        repo_root=repo_root,
    )
    required = bool(sam_runtime.get("require_fastsam_bbox_selection", True))
    return resolve_effective_config(
        config_path,
        selection_path,
        required=required,
    )


@dataclass(frozen=True)
class Sam3DBootstrapResult:
    effective_config_path: str
    template_path: str
    artifact_dir: str
    bboxes_xyxy: dict[str, tuple[int, int, int, int]]


def bootstrap_runtime_template(
    config_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    capture_python: str | Path | None = None,
) -> Sam3DBootstrapResult:
    """Validate the resident server, capture cam0, and generate one template."""
    repo_root = Path(repo_root).expanduser().resolve()
    effective = resolve_sam3d_effective_config(config_path, repo_root=repo_root)
    config = load_config(effective.config_path)
    runtime_cfg = config.get("runtime", {}).get("sam3d_server", {}) or {}
    sam_cfg = config.get("perception", {}).get("shape_fitting", {}).get("sam3d", {}) or {}

    sam3d_python = resolve_repo_path(
        runtime_cfg.get(
            "python_interpreter",
            "/home/ur5/miniforge3/envs/sam3d-objects/bin/python",
        ),
        repo_root=repo_root,
    )
    sam3d_repo = resolve_repo_path(
        runtime_cfg.get("repo_path", "external/sam-3d-objects"),
        repo_root=repo_root,
    )
    artifact_root = resolve_repo_path(
        runtime_cfg.get("artifact_root", "output/sam3d"),
        repo_root=repo_root,
    )
    artifact_dir = artifact_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capture_python = Path(capture_python or sys.executable).expanduser().resolve()
    command = [
        str(capture_python),
        str(repo_root / "tools" / "sam3d_bootstrap.py"),
        "--config", str(Path(effective.config_path).resolve()),
        "--artifact-dir", str(artifact_dir),
        "--repo-root", str(repo_root),
        "--capture-python", str(capture_python),
        "--sam3d-python", str(sam3d_python),
        "--sam3d-repo", str(sam3d_repo),
        "--execution-mode", str(runtime_cfg.get("execution_mode", "server")),
        "--server-socket", str(runtime_cfg.get("server_socket", "/tmp/handover_sam3d_stage1.sock")),
        "--server-ready-timeout-s", str(float(runtime_cfg.get("server_ready_timeout_s", 120.0))),
        "--request-timeout-s", str(float(runtime_cfg.get("request_timeout_s", 120.0))),
    ]
    result = subprocess.run(command, cwd=str(repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"SAM3D bootstrap failed with exit code {result.returncode}")
    template_path = artifact_dir / "template.npy"
    validate_template_points(
        template_path,
        min_template_points=int(sam_cfg.get("min_template_points", 100)),
    )
    return Sam3DBootstrapResult(
        effective_config_path=str(Path(effective.config_path).resolve()),
        template_path=str(template_path.resolve()),
        artifact_dir=str(artifact_dir.resolve()),
        bboxes_xyxy=effective.bboxes_xyxy,
    )


def runtime_template_override(
    config_path: str | Path,
    runtime_template_path: str | Path,
) -> dict[str, Any]:
    template_path = Path(runtime_template_path).expanduser().resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"SAM3D runtime template does not exist: {template_path}")
    config = load_config(config_path)
    sam_cfg = config.get("perception", {}).get("shape_fitting", {}).get("sam3d", {}) or {}
    label = str(
        config.get("perception", {}).get("object", {}).get("fastsam", {}).get("label", DEFAULT_LABEL)
    ).strip() or DEFAULT_LABEL
    return {
        "label": label,
        "template_id": str(sam_cfg.get("template_id", "sam3d_runtime")),
        "asset_path": str(template_path),
        "unit_scale_m": 1.0,
        "scale_mode": str(sam_cfg.get("scale_mode", "axis_xyz")),
        "z_rotation": dict(sam_cfg.get("z_rotation", {}) or {}),
        "axis_rotation_candidates": dict(sam_cfg.get("axis_rotation_candidates", {}) or {}),
    }


def build_runtime_shape_fitting_tracker(
    config_path: str | Path,
    runtime_template_path: str | Path,
) -> ShapeFittingTracker:
    return ShapeFittingTracker.from_config(
        config_path,
        runtime_template_override=runtime_template_override(config_path, runtime_template_path),
    )


def build_fastsam_object_workers(config_path: str | Path):
    config = load_config(config_path)
    fastsam_cfg = config.get("perception", {}).get("object", {}).get("fastsam", {}) or {}
    bboxes = fastsam_cfg.get("bboxes", {}) or {}
    missing = [camera for camera in ("cam0", "cam1") if camera not in bboxes]
    if missing:
        raise ValueError(f"SAM3D requires separate FastSAM bboxes; missing={missing}")

    label = str(fastsam_cfg.get("label", DEFAULT_LABEL)).strip() or DEFAULT_LABEL
    shared_model = SharedFastSAMModel(
        fastsam_cfg.get("model_name", "external/sam-3d-objects/FastSAM-s.pt"),
        imgsz=int(fastsam_cfg.get("imgsz", 640)),
        conf=float(fastsam_cfg.get("conf", 0.4)),
        iou=float(fastsam_cfg.get("iou", 0.9)),
        device=fastsam_cfg.get("device", "cuda:0"),
        half=bool(fastsam_cfg.get("half", False)),
        require_cuda=bool(fastsam_cfg.get("require_cuda", True)),
    )
    worker_config = deepcopy(config)
    object_cfg = worker_config.setdefault("perception", {}).setdefault("object", {})
    object_cfg["target_label"] = label
    segmentation_cfg = object_cfg.setdefault("segmentation", {})
    segmentation_cfg["selection_mode"] = "highest_score"
    segmentation_cfg["selection_class_names"] = [label]
    segmentation_cfg["prefer_wine_glass_over_cup"] = False
    transform_chain = load_transform_chain(config_path)
    min_bbox_size_px = int(fastsam_cfg.get("min_bbox_size_px", 8))
    engine0 = FastSAMSegmentationEngine(
        shared_model, bbox=bboxes["cam0"], label=label, min_bbox_size_px=min_bbox_size_px
    )
    engine1 = FastSAMSegmentationEngine(
        shared_model, bbox=bboxes["cam1"], label=label, min_bbox_size_px=min_bbox_size_px
    )
    return (
        ObjectWorkerCam0(0, engine0, transform_chain, worker_config),
        ObjectWorkerCam1(1, engine1, transform_chain, worker_config),
        NoOpObjectClassLock(),
        label,
    )


__all__ = [
    "NoOpObjectClassLock",
    "Sam3DBootstrapResult",
    "bootstrap_runtime_template",
    "build_fastsam_object_workers",
    "build_runtime_shape_fitting_tracker",
    "resolve_repo_path",
    "resolve_sam3d_effective_config",
    "runtime_template_override",
    "validate_hands23_runtime_assets",
    "validate_main_runtime",
]
