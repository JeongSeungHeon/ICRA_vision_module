"""Shared FDCT depth-completion utilities for live perception pipelines."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils.depth_filters import bilateral_filter_depth


REPO_ROOT = Path(__file__).resolve().parents[1]
FDCT_ROOT = REPO_ROOT / "FDCT"


@dataclass(frozen=True)
class FDCTDepthCompletionConfig:
    checkpoint_path: Path
    width: int | None = None
    height: int | None = None
    net_width: int = 320
    net_height: int = 240
    depth_min: float = 0.3
    depth_max: float = 1.5
    depth_norm: float = 1.0
    depth_coeff: float = 10.0
    inpaint: bool = True


@dataclass(frozen=True)
class FDCTDepthCompletionResult:
    preprocessed_depth_m: np.ndarray
    completed_depth_m: np.ndarray
    elapsed_ms: float


def _load_fdct_base_class():
    if str(FDCT_ROOT) not in sys.path:
        sys.path.insert(0, str(FDCT_ROOT))
    from Model import FDCT

    return FDCT


def _make_fdct_transcg_variant_class():
    fdct_base_class = _load_fdct_base_class()

    class FDCTTransCGVariant(fdct_base_class):
        """FDCT variant matching the Sequential indices used by TransCG.tar."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            h = self.hidden_channels
            self.skip_down1 = self._make_checkpoint_skip_down(h, h)
            self.skip_down2 = self._make_checkpoint_skip_down(h * 2, h)
            self.skip_down3 = self._make_checkpoint_skip_down(h * 2, h)

        @staticmethod
        def _make_checkpoint_skip_down(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.ReLU6(),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(),
            )

    return FDCTTransCGVariant


def resolve_checkpoint(path: str | Path, base_dir: str | Path | None = None) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        root = Path(base_dir) if base_dir is not None else REPO_ROOT
        checkpoint = root / checkpoint
    return checkpoint


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return device


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_cls = _make_fdct_transcg_variant_class()
    model = model_cls().to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def nearest_inpaint(depth: np.ndarray) -> np.ndarray:
    from scipy.interpolate import NearestNDInterpolator

    mask = np.where(depth > 0)
    if mask[0].shape[0] == 0:
        return depth
    interp = NearestNDInterpolator(np.transpose(mask), depth[mask])
    return interp(*np.indices(depth.shape)).astype(np.float32)


def prepare_model_inputs(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    net_size: tuple[int, int],
    depth_min: float,
    depth_max: float,
    depth_norm: float,
    depth_coeff: float,
    inpaint: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, float, np.ndarray] | None:
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, net_size, interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth_m, net_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)

    depth[np.isnan(depth)] = 0.0
    depth = np.where(depth < depth_min, 0.0, depth)
    depth = np.where(depth > depth_max, 0.0, depth)

    depth_available = depth[depth > 0]
    if depth_available.shape[0] == 0:
        return None

    depth_mu = float(depth_available.mean())
    depth_std = float(depth_available.std()) if depth_available.shape[0] > 1 else 1.0
    if depth_std <= 1e-6:
        depth_std = 1.0
    depth = np.where(depth < depth_mu - depth_coeff * depth_std, 0.0, depth)
    depth = np.where(depth > depth_mu + depth_coeff * depth_std, 0.0, depth)

    if inpaint:
        depth = nearest_inpaint(depth)

    depth = depth / depth_norm
    depth_norm_min = float(depth.min() - 0.5 * depth.std() - 1e-6)
    depth_norm_max = float(depth.max() + 0.5 * depth.std() + 1e-6)
    if depth_norm_max - depth_norm_min <= 1e-6:
        return None

    depth_normalized = (depth - depth_norm_min) / (depth_norm_max - depth_norm_min)
    rgb_tensor = torch.from_numpy((rgb / 255.0).transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(depth_normalized).float().unsqueeze(0).to(device)
    return rgb_tensor, depth_tensor, depth_norm_min, depth_norm_max, depth


def infer_depth(
    model: nn.Module,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    args: object,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    prepared = prepare_model_inputs(
        color_bgr=color_bgr,
        depth_m=depth_m,
        net_size=(int(args.net_width), int(args.net_height)),
        depth_min=float(args.depth_min),
        depth_max=float(args.depth_max),
        depth_norm=float(args.depth_norm),
        depth_coeff=float(args.depth_coeff),
        inpaint=bool(args.inpaint),
        device=device,
    )
    if prepared is None:
        return None

    rgb_tensor, depth_tensor, depth_norm_min, depth_norm_max, preprocessed_depth = prepared
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    with torch.no_grad():
        completed = model(rgb_tensor, depth_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (perf_counter() - start) * 1000.0

    output_width = int(getattr(args, "width", 0) or color_bgr.shape[1])
    output_height = int(getattr(args, "height", 0) or color_bgr.shape[0])
    completed_np = completed.squeeze(0).detach().cpu().numpy()
    completed_np = completed_np * (depth_norm_max - depth_norm_min) + depth_norm_min
    completed_np = completed_np * float(args.depth_norm)
    completed_np = cv2.resize(
        completed_np.astype(np.float32),
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )
    preprocessed_depth = preprocessed_depth * float(args.depth_norm)
    preprocessed_depth = cv2.resize(
        preprocessed_depth.astype(np.float32),
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return preprocessed_depth, completed_np, elapsed_ms


class FDCTDepthCompleter:
    """Owns one FDCT model instance and completes BGR + depth frames."""

    def __init__(self, config: FDCTDepthCompletionConfig, device_arg: str = "auto") -> None:
        self.config = config
        self.device = resolve_device(device_arg)
        self.model = load_model(config.checkpoint_path, self.device)

    def complete(self, color_bgr: np.ndarray, depth_m: np.ndarray) -> FDCTDepthCompletionResult | None:
        result = infer_depth(self.model, color_bgr, depth_m, self.config, self.device)
        if result is None:
            return None
        preprocessed_depth_m, completed_depth_m, elapsed_ms = result
        return FDCTDepthCompletionResult(
            preprocessed_depth_m=preprocessed_depth_m.astype(np.float32),
            completed_depth_m=completed_depth_m.astype(np.float32),
            elapsed_ms=float(elapsed_ms),
        )

    def run_self_test(self) -> tuple[int, ...]:
        rgb = torch.rand(1, 3, self.config.net_height, self.config.net_width, device=self.device)
        depth = torch.rand(1, self.config.net_height, self.config.net_width, device=self.device)
        with torch.no_grad():
            output = self.model(rgb, depth)
        return tuple(int(value) for value in output.shape)


def format_depth_completion_stats(
    raw_depth_m: np.ndarray,
    completed_depth_m: np.ndarray,
    depth_min: float,
    depth_max: float,
    filtered_depth_m: np.ndarray | None = None,
) -> str:
    raw_valid = np.logical_and(raw_depth_m >= depth_min, raw_depth_m <= depth_max)
    fdct_valid = np.isfinite(completed_depth_m)
    paired_valid = np.logical_and(raw_valid, fdct_valid)

    def _stats(values: np.ndarray) -> str:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return "n=0"
        return (
            f"n={values.size} min={float(values.min()):.3f} "
            f"med={float(np.median(values)):.3f} max={float(values.max()):.3f}"
        )

    raw_values = raw_depth_m[raw_valid]
    fdct_values = completed_depth_m[fdct_valid]
    diff_values = np.abs(completed_depth_m[paired_valid] - raw_depth_m[paired_valid])
    summary = f"raw({_stats(raw_values)}) fdct({_stats(fdct_values)}) absdiff({_stats(diff_values)})"

    if filtered_depth_m is not None:
        filtered_valid = np.isfinite(filtered_depth_m)
        filtered_values = filtered_depth_m[filtered_valid]
        filtered_paired_valid = np.logical_and(fdct_valid, filtered_valid)
        filtered_diff_values = np.abs(filtered_depth_m[filtered_paired_valid] - completed_depth_m[filtered_paired_valid])
        summary += f" bilateral({_stats(filtered_values)}) bilateral_absdiff({_stats(filtered_diff_values)})"
    return summary
