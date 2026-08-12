#!/usr/bin/env python3
"""Isolated HOI-DETR predictor served over a local Unix socket."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.hoi_detr_ipc import receive_packet, send_packet
from perception.hoi_detr_runtime import (
    HOIDETRBBoxSelector,
    HOIDETRPredictorAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--socket", required=True)
    return parser.parse_args()


def _bbox_response(header, selected, *, reason: str, inference_ms: float) -> dict:
    response = {
        "type": "bbox",
        "task_id": int(header["task_id"]),
        "frame_seq": int(header["frame_seq"]),
        "camera_id": int(header["camera_id"]),
        "capture_time_s": float(header["capture_time_s"]),
        "valid": selected is not None,
        "reason": str(reason),
        "inference_ms": float(inference_ms),
    }
    if selected is not None:
        response.update(
            bbox_xyxy=[float(value) for value in selected.bbox_xyxy],
            hand_side=str(selected.hand_side),
            hand_score=float(selected.hand_score),
            object_score=float(selected.object_score),
            relation_score=float(selected.relation_score),
            contact_state=str(selected.contact_state),
        )
    return response


def serve(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    dynamic_cfg = (
        config.get("perception", {}).get("object", {}).get("hoi_detr_bbox", {}) or {}
    )
    fastsam_cfg = config.get("perception", {}).get("object", {}).get("fastsam", {}) or {}
    predictor = HOIDETRPredictorAdapter(
        repo_path=args.repo,
        config_path=args.detector_config,
        weights_path=args.weights,
        hand_score_threshold=float(dynamic_cfg.get("hand_score_threshold", 0.3)),
        first_object_score_threshold=float(
            dynamic_cfg.get("first_object_score_threshold", 0.3)
        ),
        hand_first_relation_threshold=float(
            dynamic_cfg.get("hand_first_relation_threshold", 0.6)
        ),
        nms_iou_threshold=float(dynamic_cfg.get("nms_iou_threshold", 0.5)),
        require_cuda=bool(dynamic_cfg.get("require_cuda", True)),
        device=str(dynamic_cfg.get("device", "cuda:0")),
    )
    raw_bboxes = fastsam_cfg.get("bboxes", {}) or {}
    selector = HOIDETRBBoxSelector(
        initial_bboxes={camera_id: raw_bboxes[f"cam{camera_id}"] for camera_id in (0, 1)},
        padding_ratio=float(dynamic_cfg.get("padding_ratio", 0.10)),
        ema_alpha=float(dynamic_cfg.get("ema_alpha", 0.60)),
        min_bbox_size_px=int(fastsam_cfg.get("min_bbox_size_px", 8)),
    )

    socket_path = Path(args.socket).expanduser()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        raise RuntimeError(
            f"refusing to replace existing HOI-DETR socket path: {socket_path}"
        )
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    socket_inode = socket_path.stat().st_ino
    server.listen(1)
    active_task_id: int | None = None
    try:
        connection, _ = server.accept()
        with connection:
            send_packet(connection, {"type": "ready"})
            while True:
                header, payload = receive_packet(connection)
                if header.get("type") != "frame":
                    raise ValueError(f"unsupported request type: {header.get('type')!r}")
                task_id = int(header["task_id"])
                if active_task_id != task_id:
                    selector.reset()
                    active_task_id = task_id
                height = int(header["height"])
                width = int(header["width"])
                channels = int(header.get("channels", 3))
                expected_size = height * width * channels
                if header.get("dtype") != "uint8" or len(payload) != expected_size:
                    raise ValueError(
                        f"invalid image payload: dtype={header.get('dtype')} "
                        f"size={len(payload)} expected={expected_size}"
                    )
                image = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, channels))
                started = time.perf_counter()
                try:
                    candidates = predictor.predict(image)
                    selected = selector.select(
                        int(header["camera_id"]), candidates, width=width, height=height
                    )
                    reason = "ok" if selected is not None else "no_associated_first_object"
                except Exception as exc:
                    selected = None
                    reason = f"inference_error:{type(exc).__name__}:{exc}"
                send_packet(
                    connection,
                    _bbox_response(
                        header,
                        selected,
                        reason=reason,
                        inference_ms=(time.perf_counter() - started) * 1000.0,
                    ),
                )
    finally:
        server.close()
        try:
            if socket_path.exists() and socket_path.stat().st_ino == socket_inode:
                socket_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
