#!/usr/bin/env python3
"""Start the configured HOI-DETR sidecar and submit one image through IPC."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.hoi_detr_ipc import HOIDETRSidecarClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/handover.yaml"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera-id", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    detector_cfg = config["perception"]["object"]["hoi_detr_bbox"]
    sidecar_cfg = config["runtime"]["hoi_detr_sidecar"]
    image = cv2.imread(str(Path(args.image).expanduser().resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read smoke image: {args.image}")

    client = HOIDETRSidecarClient(
        python_interpreter=resolve(sidecar_cfg["python_interpreter"]),
        sidecar_script=REPO_ROOT / "tools/hoi_detr_sidecar.py",
        config_path=config_path,
        repo_path=resolve(sidecar_cfg["repo_path"]),
        detector_config_path=resolve(sidecar_cfg["config_path"]),
        weights_path=resolve(sidecar_cfg["weights_path"]),
        socket_path=sidecar_cfg["socket"],
        startup_timeout_s=sidecar_cfg["startup_timeout_s"],
        request_timeout_s=sidecar_cfg["request_timeout_s"],
        max_input_hz=detector_cfg["max_input_hz"],
    )
    try:
        client.start()
        client.reset(1)
        capture_time = time.time()
        if not client.submit(
            args.camera_id,
            image,
            task_id=1,
            frame_seq=1,
            capture_time_s=capture_time,
        ):
            raise RuntimeError("HOI-DETR client rejected the smoke frame")
        deadline = time.monotonic() + max(float(args.timeout), 0.1)
        while time.monotonic() < deadline:
            results = client.drain_results()
            if results:
                result = results[-1]
                print(result)
                if not result.reason.startswith(("ok", "no_associated_first_object")):
                    raise RuntimeError(f"HOI-DETR inference failed: {result.reason}")
                return
            if not client.healthy:
                raise RuntimeError(f"HOI-DETR sidecar failed: {client.last_error}")
            time.sleep(0.05)
        raise TimeoutError("timed out waiting for the HOI-DETR smoke result")
    finally:
        client.close()


if __name__ == "__main__":
    main()
