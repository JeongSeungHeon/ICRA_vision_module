#!/usr/bin/env python3
"""Benchmark HOI-DETR and print safe dual-camera runtime recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.hoi_detr_runtime import HOIDETRPredictorAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/handover.yaml"))
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    detector_cfg = config["perception"]["object"]["hoi_detr_bbox"]
    sidecar_cfg = config["runtime"]["hoi_detr_sidecar"]
    images = []
    for value in args.image:
        image = cv2.imread(str(Path(value).expanduser().resolve()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read benchmark image: {value}")
        images.append(image)

    load_started = time.perf_counter()
    adapter = HOIDETRPredictorAdapter(
        repo_path=resolve(REPO_ROOT, sidecar_cfg["repo_path"]),
        config_path=resolve(REPO_ROOT, sidecar_cfg["config_path"]),
        weights_path=resolve(REPO_ROOT, sidecar_cfg["weights_path"]),
        hand_score_threshold=detector_cfg["hand_score_threshold"],
        first_object_score_threshold=detector_cfg["first_object_score_threshold"],
        hand_first_relation_threshold=detector_cfg["hand_first_relation_threshold"],
        nms_iou_threshold=detector_cfg["nms_iou_threshold"],
        require_cuda=detector_cfg.get("require_cuda", True),
        device=detector_cfg.get("device", "cuda:0"),
    )
    cold_load_s = time.perf_counter() - load_started

    for index in range(max(int(args.warmup), 0)):
        adapter.predict(images[index % len(images)])

    timings: list[float] = []
    candidate_counts: list[int] = []
    for index in range(max(int(args.iterations), 1)):
        started = time.perf_counter()
        candidates = adapter.predict(images[index % len(images)])
        timings.append(time.perf_counter() - started)
        candidate_counts.append(len(candidates))

    p50 = percentile(timings, 50)
    p95 = percentile(timings, 95)
    p99 = percentile(timings, 99)
    per_camera_hz = min(10.0, 0.4 / max(p95, 1e-6))
    recommendations = {
        "max_input_hz": per_camera_hz,
        "hold_timeout_s": max(0.5, 3.75 * p95),
        "result_age_timeout_s": max(1.0, 2.0 * p99),
        "request_timeout_s": max(10.0, 3.0 * p99),
        "startup_timeout_s": max(120.0, 2.0 * cold_load_s),
    }
    print(
        yaml.safe_dump(
            {
                "cold_load_s": cold_load_s,
                "inference_s": {"p50": p50, "p95": p95, "p99": p99},
                "candidate_count": {
                    "min": min(candidate_counts),
                    "max": max(candidate_counts),
                    "mean": float(np.mean(candidate_counts)),
                },
                "recommended_config": recommendations,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
