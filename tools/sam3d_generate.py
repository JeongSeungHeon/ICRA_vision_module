#!/usr/bin/env python3
"""Run SAM3D stage1 and atomically export a metric Nx3 template."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import yaml
from PIL import Image


def normalize_stage1_voxels(
    voxels: Any,
    *,
    canonical_extent_m: float = 0.1,
    min_points: int = 100,
    min_axis_extent_m: float = 1e-4,
    robust_percentile: float = 1.0,
) -> np.ndarray:
    """Validate, center, and uniformly scale native SAM3D voxel coordinates."""
    if hasattr(voxels, "detach"):
        voxels = voxels.detach().cpu().numpy()
    points = np.asarray(voxels)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"SAM3D stage1 voxel output must be shaped (N,3), got {points.shape}")
    if len(points) < int(min_points):
        raise ValueError(f"SAM3D template has too few points: {len(points)} < {int(min_points)}")
    points = points.astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError("SAM3D template contains NaN or infinite coordinates")

    q = float(robust_percentile)
    if not 0.0 <= q < 50.0:
        raise ValueError(f"robust_percentile must be in [0,50), got {q}")
    lower = np.percentile(points, q, axis=0)
    upper = np.percentile(points, 100.0 - q, axis=0)
    source_extent = upper - lower
    largest_extent = float(np.max(source_extent))
    canonical_extent_m = float(canonical_extent_m)
    if not np.isfinite(largest_extent) or largest_extent <= 1e-9:
        raise ValueError(f"SAM3D template has degenerate robust extent: {source_extent}")
    if not np.isfinite(canonical_extent_m) or canonical_extent_m <= 0.0:
        raise ValueError(f"canonical_extent_m must be positive, got {canonical_extent_m}")

    center = (lower + upper) * 0.5
    normalized = (points - center.reshape(1, 3)) * (canonical_extent_m / largest_extent)
    output = normalized.astype(np.float32)
    output_extent = np.percentile(output, 100.0 - q, axis=0) - np.percentile(output, q, axis=0)
    if np.any(output_extent < float(min_axis_extent_m)):
        raise ValueError(
            "SAM3D template has a degenerate axis after normalization: "
            f"extent={output_extent}, minimum={float(min_axis_extent_m)}"
        )
    return output


def atomic_save_npy(path: str | Path, points: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.save(handle, np.asarray(points, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    return destination


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def preflight(config: dict, *, sam3d_repo: Path) -> dict:
    sam_cfg = config.get("perception", {}).get("shape_fitting", {}).get("sam3d", {}) or {}
    checkpoint_config = Path(sam_cfg.get("checkpoint_config", "checkpoints/hf/pipeline.yaml")).expanduser()
    if not checkpoint_config.is_absolute():
        checkpoint_config = sam3d_repo / checkpoint_config
    if not sam3d_repo.is_dir():
        raise FileNotFoundError(f"SAM3D repo does not exist: {sam3d_repo}")
    if not checkpoint_config.is_file():
        raise FileNotFoundError(f"SAM3D checkpoint config does not exist: {checkpoint_config}")

    environment_root = Path(sys.executable).resolve().parents[1]
    os.environ["CONDA_PREFIX"] = str(environment_root)
    os.environ["CUDA_HOME"] = str(environment_root)
    os.environ["LIDRA_SKIP_INIT"] = "true"
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if bool(sam_cfg.get("require_cuda", True)) and not cuda_available:
        raise RuntimeError("SAM3D requires CUDA, but torch.cuda.is_available() is false")
    return {
        "checkpoint_config": checkpoint_config,
        "environment_root": environment_root,
        "cuda_available": cuda_available,
    }


class Sam3DStage1Engine:
    """Resident stage1-only SAM3D model with repeatable template export."""

    def __init__(self, config: dict, *, sam3d_repo: Path):
        self.sam3d_repo = Path(sam3d_repo).expanduser().resolve()
        sam_cfg = (
            config.get("perception", {})
            .get("shape_fitting", {})
            .get("sam3d", {})
            or {}
        )
        started = time.perf_counter()
        preflight_result = preflight(config, sam3d_repo=self.sam3d_repo)
        self.checkpoint_config = Path(preflight_result["checkpoint_config"]).resolve()

        # notebook/inference.py requires the active environment root during import.
        for import_path in (self.sam3d_repo, self.sam3d_repo / "notebook"):
            value = str(import_path)
            if value not in sys.path:
                sys.path.insert(0, value)

        from inference import Inference

        self.inference = Inference(
            str(self.checkpoint_config),
            compile=bool(sam_cfg.get("compile", False)),
            stage1_only_init=True,
        )
        self.model_load_elapsed_s = time.perf_counter() - started

    def generate_template(
        self,
        config: dict,
        *,
        artifact_dir: Path,
        server_metadata: dict | None = None,
    ) -> dict:
        sam_cfg = (
            config.get("perception", {})
            .get("shape_fitting", {})
            .get("sam3d", {})
            or {}
        )
        artifact_dir = Path(artifact_dir).expanduser().resolve()
        request_started = time.perf_counter()

        image_path = artifact_dir / "image.png"
        mask_path = artifact_dir / "mask.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"SAM3D input artifacts are missing in {artifact_dir}"
            )

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"SAM3D image/mask shapes differ: {image.shape[:2]} vs {mask.shape}"
            )
        if not np.any(mask):
            raise ValueError("SAM3D input mask is empty")

        seed = int(sam_cfg.get("seed", 42))
        inference_started = time.perf_counter()
        output = self.inference._pipeline.run(
            image,
            mask.astype(np.uint8) * 255,
            seed=seed,
            stage1_only=True,
        )
        inference_elapsed_s = time.perf_counter() - inference_started
        if "voxel" not in output:
            raise KeyError("SAM3D stage1 output does not contain `voxel`")

        export_started = time.perf_counter()
        template = normalize_stage1_voxels(
            output["voxel"],
            canonical_extent_m=float(sam_cfg.get("canonical_extent_m", 0.1)),
            min_points=int(sam_cfg.get("min_template_points", 100)),
            min_axis_extent_m=float(sam_cfg.get("min_axis_extent_m", 1e-4)),
            robust_percentile=float(sam_cfg.get("robust_percentile", 1.0)),
        )
        template_path = atomic_save_npy(artifact_dir / "template.npy", template)
        export_elapsed_s = time.perf_counter() - export_started
        request_elapsed_s = time.perf_counter() - request_started
        extent = np.max(template, axis=0) - np.min(template, axis=0)

        runtime = dict(server_metadata or {})
        model_reused = bool(runtime.pop("model_reused", False))
        elapsed_s = (
            request_elapsed_s
            if model_reused
            else self.model_load_elapsed_s + request_elapsed_s
        )
        metadata_path = artifact_dir / "metadata.yaml"
        metadata = _load_yaml(metadata_path)
        metadata["sam3d"] = {
            "repo_path": str(self.sam3d_repo),
            "checkpoint_config": str(self.checkpoint_config),
            "seed": seed,
            "stage1_only": True,
            "stage1_only_init": True,
            "template_id": str(sam_cfg.get("template_id", "sam3d_runtime")),
            "template_path": str(template_path.resolve()),
            "template_point_count": int(len(template)),
            "template_extent_m": [float(value) for value in extent],
            "canonical_extent_m": float(sam_cfg.get("canonical_extent_m", 0.1)),
            "elapsed_s": float(elapsed_s),
            "model_reused": model_reused,
            "model_load_elapsed_s": float(self.model_load_elapsed_s),
            "inference_elapsed_s": float(inference_elapsed_s),
            "export_elapsed_s": float(export_elapsed_s),
            "request_elapsed_s": float(request_elapsed_s),
            **runtime,
        }
        metadata.setdefault("artifacts", {})["template"] = str(
            template_path.resolve()
        )
        _write_yaml(metadata_path, metadata)
        return metadata["sam3d"]


def generate_template(
    config: dict,
    *,
    artifact_dir: Path,
    sam3d_repo: Path,
) -> dict:
    """One-shot compatibility entry point using the lightweight stage1 engine."""
    engine = Sam3DStage1Engine(config, sam3d_repo=sam3d_repo)
    return engine.generate_template(
        config,
        artifact_dir=artifact_dir,
        server_metadata={"model_reused": False},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--sam3d-repo", required=True)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    sam3d_repo = Path(args.sam3d_repo).expanduser().resolve()
    if args.preflight:
        result = preflight(config, sam3d_repo=sam3d_repo)
        print(
            "[sam3d_generate] preflight passed: "
            f"cuda={result['cuda_available']} "
            f"checkpoint={result['checkpoint_config']}",
            flush=True,
        )
        return
    result = generate_template(
        config,
        artifact_dir=Path(args.artifact_dir).expanduser().resolve(),
        sam3d_repo=sam3d_repo,
    )
    print(
        "[sam3d_generate] template saved: "
        f"points={result['template_point_count']} "
        f"elapsed={result['elapsed_s']:.2f}s "
        f"path={result['template_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
